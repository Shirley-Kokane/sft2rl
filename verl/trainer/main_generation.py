# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import csv
import ray
import numpy as np
import hydra
import os
from tabulate import tabulate
import torch

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.torch_functional import entropy_from_logits, logprobs_from_logits
from transformers import AutoConfig, AutoModelForCausalLM
from verl.trainer.base_utils import rigidity_index_from_logits, similarity_score_cal

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # Check if output file already exists
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        try:
            dataset = pd.read_parquet(config.data.output_path)
        except Exception as e:
            # Read json
            try:
                import json
                config.data.output_path = config.data.output_path.replace('.parquet', '.json')
                with open(config.data.output_path, 'r') as f:
                    dataset = pd.read_json(f)
            except Exception as e:
                # user polars
                import polars as pl
                config.data.output_path = config.data.output_path.replace('.json', '.parquet')
                dataset = pl.read_parquet(config.data.output_path)
    else:
        local_path = copy_local_path_from_hdfs(config.model.path)
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)

        if config.rollout.temperature == 0.:
            assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
        try:
            dataset = pd.read_parquet(config.data.path)
            chat_lst = dataset[config.data.prompt_key].tolist()
            chat_lst = [chat.tolist() for chat in chat_lst]
        except Exception as e:
            # Read json
            import json
            config.data.path = config.data.path.replace('.parquet', '.json')
            with open(config.data.path, 'r') as f:
                dataset = pd.read_json(f)
            chat_lst = dataset[config.data.prompt_key].tolist()
            chat_lst = [chat for chat in chat_lst]

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        
        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()
        
        if config.rollout.calculate_log_probs:
            total_samples = len(dataset)
            # real_batch_size = data.batch['input_ids'].shape[0]
            config_batch_size = config.data.batch_size
            num_batch = (total_samples // config_batch_size) + 1
            output_lst = []  # We'll reshape at the end
        
            rollout_outputs = []
            for batch_idx in range(num_batch):
                print(f'[{batch_idx+1}/{num_batch}] Start to process.')
                batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
                
                # Repeat the batch n_samples times
                repeated_chat_lst = []
                for chat in batch_chat_lst:
                    repeated_chat_lst.extend([chat] * config.data.n_samples)
                
                inputs = tokenizer.apply_chat_template(repeated_chat_lst,
                                                    add_generation_prompt=True,
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=config.rollout.prompt_length,
                                                    return_tensors='pt',
                                                    return_dict=True,
                                                    tokenize=True)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                position_ids = compute_position_id_with_mask(attention_mask)

                batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

                data = DataProto.from_dict(batch_dict)
                real_batch_size = data.batch['input_ids'].shape[0]
                dp_size = wg.world_size
                
                if real_batch_size % dp_size != 0:
                    dummy_data_size = dp_size - real_batch_size % dp_size
                    dummy_data = data[:dummy_data_size]
                    data = DataProto.concat([data, dummy_data])
                    print(
                        f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
                    )

                batch_size = data.batch['input_ids'].shape[0]
                assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

                print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
                
                # Generate all samples at once
                rollout_output = wg.generate_sequences(data)
                rollout_outputs.append(rollout_output)
            
                rollout_output = rollout_output[:real_batch_size]
                output_text = tokenizer.batch_decode(rollout_output.batch['input_ids'][:, -config.rollout.response_length:],
                                               skip_special_tokens=False)
                
                rollout_outputs.append(rollout_output)
                # Remove padding
                pad_token = tokenizer.pad_token
                output_text_unpad = []
                for text in output_text:
                    output_text_unpad.append(text.replace(pad_token, ''))
                    
                output_lst.extend(output_text_unpad)
                
        # Reshape output_lst from (total_samples,) to (n_data, n_samples)
        total_samples = len(output_lst)
        n_data = total_samples // config.data.n_samples
        output_lst = np.array(output_lst).reshape(n_data, config.data.n_samples).tolist()
        
        # Add to the data frame
        dataset['responses'] = output_lst
        # Write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
    
    output_dir = os.path.dirname(config.data.output_path)
    # Compute evaluation metrics
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']  # Using the generated responses
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total = len(dataset)
    total_scores = []
    similarity_scores = []
    for i in dataset.index:
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i][-1]['content']
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        
        score_lst = []
        min_similarity_score, avg_similarity_score, ttr_score, avg_bleu_score, min_bleu_score = similarity_score_cal(prompt, response_lst)
        similarity_scores.append((min_similarity_score, avg_similarity_score, ttr_score, avg_bleu_score, min_bleu_score))
        for r in response_lst:
            try:
                score = reward_fn(data_source, r, ground_truth, {"use_format_reward": config.data.use_format_reward})
                score_lst.append(score)
            except Exception as e:
                score = reward_fn(r, ground_truth, {"use_format_reward": config.data.use_format_reward})
                score_lst.append(score)
                
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1

    n_samples = config.data.n_samples
    pass_at_n = passes / total
    pass_at_1 = np.mean(total_scores)
    dataset["scores"] = total_scores
    dataset["similarity_scores"] = similarity_scores
    
    if not os.path.exists(config.data.output_path):
        # Fix PyArrow nested data issue by using json serialization for nested columns
        try:
            dataset.to_parquet(config.data.output_path)
        except Exception as e:
            if "Nested data conversions not implemented" in str(e):
                # Convert nested data to JSON strings before saving
                import json
                dataset_copy = dataset.copy()
                dataset_copy['responses'] = dataset_copy['responses'].apply(json.dumps)
                dataset_copy.to_parquet(config.data.output_path)
                print("Saved with JSON serialization due to nested data structures")
            else:
                raise e

    data_name = config.data.path.split('/')[-1].split('.')[0]
    # Save metrics to CSV
    model_ckpt, ckpt_name = config.data.output_path.split("_")[-3] , config.data.output_path.split("_")[-2]
    try:
        csv_path = os.path.join(output_dir, f'pass_{data_source.split("-")[1]}_format_{config.data.use_format_reward}_{model_ckpt}_{ckpt_name}.csv')
    except Exception as e:
        csv_path = os.path.join(output_dir, f'pass_{data_source}_format_{config.data.use_format_reward}_{model_ckpt}_{ckpt_name}.csv')
    
    # Prepare the row data
    # Extract the dataset name from the path
    dataset_name = os.path.basename(config.data.path).split('.')[0]
    row_data = {
        'model_path': config.model.path.split('/')[-1],
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@_{n_samples}': pass_at_n,
        'response_length' : config.rollout.response_length,
        'temperature' : config.rollout.temperature,
        'avg_similarity_score' : np.mean([similarity_score[1] for similarity_score in similarity_scores]),
        'min_similarity_score' : np.min([similarity_score[0] for similarity_score in similarity_scores]),
        'ttr_score' : np.mean([similarity_score[2] for similarity_score in similarity_scores]),
        'avg_bleu_score' : np.mean([similarity_score[3] for similarity_score in similarity_scores]),
        'min_bleu_score' : np.min([similarity_score[4] for similarity_score in similarity_scores]),
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

    
    

# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from rllm.rewards.rl_reward import rllm_reward_fn
        return rllm_reward_fn

if __name__ == '__main__':
    main()