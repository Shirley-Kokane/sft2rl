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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

import datasets
import torch
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
from transformers import AutoTokenizer

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def filter_code(row):
    prompt = row['prompt']
    
    # Check for multiple occurrences of "input()" and functions ("def")
    if len(prompt) < 1024 and len(row['solution']) < 1024:
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/export/home/research/verl_mt/data/sft_math/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")

    args = parser.parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            
            question = example.pop("problem")

            question = question + " " + instruction_following
            
            prompt_text = tokenizer.apply_chat_template(
                [
                {
                    "role": "system",
                    "content": instruction_following,
                },
                {
                    "role": "user",
                    "content": question
                },                
            ],
                tokenize=False,
                add_generation_prompt=True,
            )

            answer = example.pop("solution")
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": prompt_text,
                "solution":answer,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx, "prompt": prompt_text, "solution": answer},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    train_dataset = train_dataset.filter(filter_code)
    test_dataset = test_dataset.filter(filter_code)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, f"train_{model_name.split('/')[-1]}.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, f"test_{model_name.split('/')[-1]}.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
