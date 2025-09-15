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
from socket import EAI_SYSTEM

import datasets
import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
from transformers import AutoTokenizer

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def filter_length(row):
    prompt = row['prompt']
    
    # Check for multiple occurrences of "input()" and functions ("def")
    if len(prompt) < 1024 and len(row["solution"]) < 12500 and 'Yes' not in row["reward_model"]["ground_truth"] and 'No' not in row["reward_model"]["ground_truth"]:
        return True
    return False

def filter_difficulty(row, threshold=6):
    if row["difficulty"] < threshold:
        return True
    return False

import json
jsonl = "./summary_log.jsonl"
data = []
indices = set()
with open(jsonl, "r") as f:
    for line in f:
        data.append(json.loads(line))
        if json.loads(line)["results"]["qwen3-32b"]["final_correct_rate"] >= 0.5:
            indices.add(json.loads(line)["sample_id"].split("_")[1])

print(len(indices))

def filter_data(row):
    if str(row["extra_info"]["index"]) in indices:
        return True
    return False

# add a row to each data item that represents a unique id
def make_map_fn(split):
    def process_fn(example, idx):
        question = example.pop("question")
        
        if use_think_prompt:
            instruction = "You are a helpful assistant. The user will ask you a question and you as the assistant will solve it. The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. The reasoning process should be concise and enclosed within <think>...</think> followed by the final answer within \\boxed{}." 
        else:
            instruction = "Let's think step by step and output the final answer within \\boxed{}."

        prompt_text = [{
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": question
            }]

        answer = str(example.pop("final_answer"))
        min_idx = 0
        min_len = float("inf")
        for sol_idx, i in enumerate([len(example["r1_solution_1"]), len(example["r1_solution_2"]), len(example["r1_solution_3"])]):
            if i < min_len:
                min_len = i
                min_idx = sol_idx
        solution = example["r1_solution_" + str(min_idx + 1)]
        data = {
            "data_source": "math-" + data_source.split("/")[-1],
            "prompt": prompt_text,
            "solution": solution,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": [answer]},
            "extra_info": {"split": split, "index": idx, "prompt": prompt_text, "solution": solution},
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/export/home/research/verl_mt/data/deepmath/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument('--use_think_prompt', default=False,
                       help='Use think prompt in system instruction')

    args = parser.parse_args()
    
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    use_think_prompt = args.use_think_prompt
    

    data_source = "zwhe99/DeepMath-103K"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"]
    
    process_fn = make_map_fn("train")
    train_data = []
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)

    #train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    
    train_data = [row for row in train_data if filter_length(row)] #train_dataset.filter(filter_length)
    
    print("before filter ", len(train_dataset))
    train_dataset = [row for row in train_data if filter_data(row)] #train_dataset.filter(filter_data)
    print("after filter ", len(train_dataset))
    
    train_dataset = pd.DataFrame(train_dataset)
    #easy_dataset = train_dataset.filter(filter_difficulty)
    #print(len(easy_dataset))
    
    train_data_1 = train_dataset[:int(len(train_dataset)*0.45)]
    
    train_data_2 = train_dataset[int(len(train_dataset)*0.45):int(len(train_dataset)*0.9)]
    
    test_data_1 = train_dataset[int(len(train_dataset)*0.9):]
    
    print(len(train_data_1), len(train_data_2), len(test_data_1))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_data_1.to_parquet(os.path.join(local_dir, f"sft/train_deepmath_{model_name.split('/')[-1]}.parquet"))
    train_data_2.to_parquet(os.path.join(local_dir, f"rl/train_deepmath_{model_name.split('/')[-1]}.parquet"))
    test_data_1.to_parquet(os.path.join(local_dir, f"sft/test_deepmath_{model_name.split('/')[-1]}.parquet"))
    test_data_1.to_parquet(os.path.join(local_dir, f"rl/test_deepmath_{model_name.split('/')[-1]}.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
