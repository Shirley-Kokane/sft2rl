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
Preprocess the OpenScience dataset to parquet format
"""

import argparse
import os
from socket import EAI_SYSTEM

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
from transformers import AutoTokenizer
from openai import OpenAI
from tqdm import tqdm
import pandas as pd

# Initialize the OpenAI client
# Make sure to set your API key as an environment variable: export OPENAI_API_KEY="your-api-key"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def filter_code(row):
    # prompt = row['prompt']
    # gpt35_answer = chat_completion_example(prompt, "gpt-3.5-turbo")
    
    # if row['reward_model']['ground_truth'] not in gpt4_answer and row['reward_model']['ground_truth'] not in gpt35_answer:
    #     print(gpt35_answer, gpt4_answer, row['reward_model']['ground_truth'])
    #     return True
    # Check for multiple occurrences of "input()" and functions ("def")
    solution = extract_solution(row["solution"])
    
    if len(row["prompt"]) < 1024 and len(row["solution"]) <= 7168 and len(row['reward_model']['ground_truth']) == 1 and solution == row['reward_model']['ground_truth'][0]: #
        #gpt4o_answer = chat_completion_example(prompt, "gpt-4o-mini")
        #if gpt4o_answer:
        return True
    return False

def filter_difficulty(row, threshold=6):
    if row["difficulty"] < threshold:
        return True
    return False

def chat_completion_example(question, model_name, expected_answer):
    """
    Basic chat completion example with OpenAI
    """
    try:
        response_1 = client.chat.completions.create(
            model=model_name,  # or "gpt-4", "gpt-4-turbo", etc.
            messages=question,
            max_tokens=7168,
            temperature=0.7
        )
        
        response_2 = client.chat.completions.create(
            model=model_name,  # or "gpt-4", "gpt-4-turbo", etc.
            messages=question,
            max_tokens=7168,
            temperature=1.0
        )
        
        # Extract and return the response
        answer_1 = extract_solution(response_1.choices[0].message.content)
        answer_2 = extract_solution(response_2.choices[0].message.content)
        if (answer_1 == expected_answer and answer_2 == expected_answer):
            return False
        elif (answer_1 == expected_answer or answer_2 == expected_answer) and len(answer_1) == len(answer_2):
            print(answer_1, answer_2, expected_answer)
            return True
        return False
        
    except Exception as e:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="../../data/openscience/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")

    args = parser.parse_args()
    
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

   
    data_source = "nvidia/OpenScienceReasoning-2"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True, split="train")

    #train_dataset = dataset['train']
    
    print(len(dataset))

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = str(example.pop("input"))

            solution = str(example.pop("output"))
            answer = extract_solution(solution)
            prompt_text =[
                {"role": "user", "content": question + " Lets' think step by step and output which of the following options is the correct answer within \\boxed{}."},
            ]
            data = {
            "data_source": "math-openscience",
            "prompt": prompt_text,
            "solution": solution,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": [example.pop("expected_answer")]},
            "extra_info": {"split": split, "index": idx, "prompt": prompt_text, "solution": solution},
            }
            return data

        return process_fn

    train_dataset = dataset.map(function=make_map_fn("train"), with_indices=True)
    
    print("before filtering ", len(train_dataset))
    
    train_dataset = train_dataset.filter(filter_code)
    
    print("filtered dataset length ", len(train_dataset))
    
    filtered_dataset = []
    for row in train_dataset:
        if chat_completion_example(row['prompt'], "gpt-4o-mini", row['reward_model']['ground_truth'][0]):
            filtered_dataset.append(row)
            
        if len(filtered_dataset) == 15000:
            break
            
    train_dataset = pd.DataFrame(filtered_dataset)
    
    
    n = len(train_dataset)
    train_dataset_1 = train_dataset[:int(0.4*n)]
    train_dataset_2 = train_dataset[int(0.4*n):int(0.8*n)]
    test_dataset_1 = train_dataset[int(0.8*n):]
    
    print(len(train_dataset_1), len(train_dataset_2), len(test_dataset_1))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, f"train_openscience_all.parquet"))
    
    train_dataset_1.to_parquet(os.path.join(local_dir, f"train_openscience_1.parquet"))
    train_dataset_2.to_parquet(os.path.join(local_dir, f"train_openscience_2.parquet"))
    test_dataset_1.to_parquet(os.path.join(local_dir, f"test_openscience_1.parquet"))
    # test_dataset_2.to_parquet(os.path.join(local_dir, f"test_openscience_2_{model_name.split('/')[-1]}.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
