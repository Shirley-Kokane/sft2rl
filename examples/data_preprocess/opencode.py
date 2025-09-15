"""Script to prepare code datasets for training and testing.

This script processes code problem datasets into a standardized format for training
and testing models. It loads problems from various code datasets (APPS, CodeForces,
LiveCodeBench etc.), adds appropriate instruction prompts, and saves the processed
data as parquet files.
"""
import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import json 

from verl.utils.hdfs_io import makedirs

from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.data.utils import load_dataset, fetch_live_code_bench_system_prompt
from datasets import concatenate_datasets

def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, dataset_name=None,use_think_prompt: bool = False) -> Optional[Dict[str, Any]]:
        
        if "question" in example:
            question = example.pop('question')
        else:
            question = example.pop('problem')
            
        tests = example.pop('tests')
        
        if example.get('metadata', {}):
            assert 'func_name' in example['metadata'], f"Function name is not found, check if your LCB data is preprocessed correctly: {example['metadata']}"
            if isinstance(tests, dict):
                tests['metadata'] = example['metadata']
            else:
                for test in tests:
                    assert isinstance(test, dict), "Test is not a dict"
                    test['metadata'] = example['metadata']
        
        tests = json.dumps(tests)

        if dataset_name == "livecodebench":
            starter_code = example.get("starter_code", None)
            question = fetch_live_code_bench_system_prompt(question, starter_code)
        if isinstance(question, dict):
            question = json.dumps(question)
        
        if use_think_prompt:
            instruction = "You are a helpful programming assistant. The user will ask you a question and you as the assistant solve it. The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. The reasoning process should be concise and enclosed within <think>...</think> followed by the final answer within a markdown code block: ```python ```." 
        else:
            instruction = "Let's think step by step and output the final answer within within a markdown code block: ```python ```."
            
        data = {
            "data_source": dataset_name,
            "prompt": [{
                        "role": "system",
                        "content": instruction
                    },
                {
                "role": "user",
                "content": question
            }],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": tests
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'reference': example.get('completion', ''), # For leetcode
                "solution": example.get('r1_generation', ''),
                "prompt":[{
                        "role": "system",
                        "content": instruction
                    },
                {
                "role": "user",
                "content": question
            }]
            }
        }
        return data
    return process_fn

def filter_data(row: Dict[str, Any]) -> bool:
    prompt = str(row['prompt'])
    solution = row['extra_info']['solution']
    if len(prompt) < 2048 and len(solution) < 10000:
        return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('../../data/deepcoder/'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    parser.add_argument('--use_think_prompt', default=False,
                       help='Use think prompt in system instruction')
    args = parser.parse_args()

    local_dir = args.local_dir
    print(f"Local_dir:{local_dir}")
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    if not os.path.exists(local_dir):
        makedirs(local_dir)


    #Initialize datasets
    train_datasets = [TrainDataset.Code.APPS, TrainDataset.Code.TACO, TrainDataset.Code.CODEFORCES, TrainDataset.Code.CODE_CONTESTS]
    test_datasets = [TestDataset.Code.LIVECODEBENCH, TestDataset.Code.CODEFORCES, TestDataset.Code.CODE_CONTESTS, TestDataset.Code.HUMANEVALPLUS, TestDataset.Code.LEETCODE]
    
    test_datasets_data = [load_dataset(d) for d in test_datasets]
    train_dataset_data = [load_dataset(d) for d in train_datasets]
    
    # Print dataset sizes
    for test_dataset, data in zip(test_datasets, test_datasets_data):
        print(f"Test dataset {test_dataset.value}: {len(data)} examples")
    for train_dataset, data in zip(train_datasets, train_dataset_data):
        print(f"Train dataset {train_dataset.value}: {len(data)} examples")

    # Process training data
    all_train_data = [] 
    process_fn = make_map_fn('train')

    for train_dataset, train_dataset_data in zip(train_datasets, train_dataset_data):
        train_data: List[Dict[str, Any]] = []
        dataset_name = train_dataset.value.lower()  # Extract name from enum
        for idx, example in enumerate(train_dataset_data):
            processed_example = process_fn(example, idx, dataset_name)
            if not processed_example:
                continue# Break here to inspect the problematic example
            if processed_example is not None:
                train_data.append(processed_example)
                all_train_data.append(processed_example)
        train_df = pd.DataFrame(train_data)
        #train_df.to_parquet(os.path.join(local_dir, f'train_{dataset_name}.parquet'))
    
    #all_train_data = [row for row in all_train_data if filter_data(row)]
    
    #shuffle all_train_data
    import random
    random.shuffle(all_train_data)
    
    train_data_1 = all_train_data[:int(len(all_train_data) * 0.4)]
    train_data_2 = all_train_data[int(len(all_train_data) * 0.4):int(len(all_train_data) * 0.8)]
    test_data = all_train_data[int(len(all_train_data) * 0.8):]
    print("train data size:", len(train_data_1), len(train_data_2), len(test_data))
    
    train_df_1 = pd.DataFrame(train_data_1)
    train_df_2 = pd.DataFrame(train_data_2)
    test_df = pd.DataFrame(test_data)
    train_df_1.to_parquet(os.path.join(local_dir, 'deepcoder_train_50_1.parquet'))
    train_df_2.to_parquet(os.path.join(local_dir, 'deepcoder_train_50_2.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test_deepcoder.parquet'))
    # save all code dataset
    # all_train_df = pd.DataFrame(all_train_data)
    # all_train_df.to_parquet(os.path.join(local_dir, 'deepcoder_train.parquet'))
    # # Save a json version of deepscaler_code.parquet
    # all_train_df.to_json(os.path.join(local_dir, 'deepcoder_train.json'), orient='records')

    #Process and save each test dataset separately
    all_test_data = []
    for test_dataset, test_data_list in zip(test_datasets, test_datasets_data):
        test_data: List[Dict[str, Any]] = []
        process_fn = make_map_fn('test')
        dataset_name = test_dataset.value.lower()  # Extract name from enum
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx, dataset_name)
            if processed_example is not None:
                test_data.append(processed_example)
                all_test_data.append(processed_example)
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f'test_{dataset_name}.parquet'))
        #test_df.to_json(os.path.join(local_dir, f'test_{dataset_name}.json'), orient='records')


# Test dataset LIVECODEBENCH: 279 examples
# Test dataset CODEFORCES: 408 examples
# Test dataset CODE_CONTESTS: 165 examples
# Test dataset HUMANEVALPLUS: 163 examples
# Test dataset LEETCODE: 200 examples
# Train dataset PRIMEINTELLECT: 23891 examples
# Train dataset TACO: 7892 examples
# Train dataset CODEFORCES: 6128 examples
# train data size: 8825 8826 4413