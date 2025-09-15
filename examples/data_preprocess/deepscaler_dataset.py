"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

from rllm.data.utils import load_dataset
from rllm.data.dataset_types import TrainDataset, TestDataset


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))

def filter_data(row: Dict[str, Any]) -> bool:
    prompt = str(row['prompt'])
    solution = row['extra_info']['solution']
    if len(prompt) < 1024 and len(solution) < 1024:
        return True
    return False


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, data_source: str, use_think_prompt: bool = False) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        
        if use_think_prompt:
            instruction = "You are a helpful assistant. The user will ask you a question and you as the assistant will solve it. The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. The reasoning process should be concise and enclosed within <think>...</think> followed by the final answer within \\boxed{}." 
        else:
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
            
        question = f"{question}"
        answer = example.pop('answer')
        solution = example.pop('solution', '')
        
        if isinstance(solution, list):
            solution = solution[0]

        if not isinstance(answer, list):
            answer = [answer]
        
        data = {
            "data_source": "math-" + data_source,
            "prompt": [{
                        "role": "system",
                        "content": instruction
                    },
                {
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                "solution": solution,
                "prompt": [{
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('../../data/deepscaler/'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    parser.add_argument('--use_think_prompt', default=False,
                       help='Use think prompt in system instruction')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    use_think_prompt = args.use_think_prompt
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)

    # Initialize datasets
    train_datasets = [TrainDataset.Math.DEEPSCALER]
    train_dataset = load_dataset(train_datasets[0])
    test_datasets = [TestDataset.Math.AIME, TestDataset.Math.GSM8k, TestDataset.Math.AMC, TestDataset.Math.MATH, TestDataset.Math.MINERVA, TestDataset.Math.OLYMPIAD_BENCH]
    test_datasets_name = ["aime", "gsm8k", "amc", "math", "minerva", "olympiad_bench"]
    test_datasets_data = [load_dataset(d) for d in test_datasets]

    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx, "deepscaler",  use_think_prompt)
        if processed_example is not None:
            train_data.append(processed_example)

    # Process and save each test dataset separately
    for test_dataset,test_dataset_name, test_data_list in zip(test_datasets, test_datasets_name, test_datasets_data):
        test_data: List[Dict[str, Any]] = []
        process_fn = make_map_fn('test')
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx, test_dataset_name, use_think_prompt)
            if processed_example is not None:
                test_data.append(processed_example)
           
        test_data = [row for row in test_data if filter_data(row)]
        
        dataset_name = test_dataset.value.lower()
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f'{dataset_name}.parquet'))
        print(f"{dataset_name} test data size:", len(test_data))

    # # Save training dataset
    train_data = [row for row in train_data if filter_data(row)]
    
    train_data_1 = train_data[:int(len(train_data) * 0.7)]
    train_data_2 = train_data[int(len(train_data) * 0.7):]
    
    print("train data size:", len(train_data_1), len(train_data_2))
    
    train_df_1 = pd.DataFrame(train_data_1)
    train_df_2 = pd.DataFrame(train_data_2)
    train_df_1.to_parquet(os.path.join(local_dir, 'deepscaler_train_70.parquet'))
    train_df_2.to_parquet(os.path.join(local_dir, 'deepscaler_train_30.parquet'))

    # # Optionally copy to HDFS
    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)