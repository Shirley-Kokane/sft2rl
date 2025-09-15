"""
Preprocess the GPQA dataset to parquet format
"""

import os
import re
import argparse
import random
from datasets import load_dataset, Dataset

from verl.utils.hdfs_io import copy, makedirs


def get_datasets():
    """
    Loads the GPQA dataset.
    """
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")["train"]
        print(f"GPQA dataset: {len(dataset)} examples")
        return None, dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


# adopted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/zeroshot/utils.py
def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def make_map_fn(split: str, data_source: str) -> callable:
    def process_fn(example, idx):
        # Create a default "skip" response with all required fields
        question = example["Question"].strip()
        correct = preprocess(example["Correct Answer"])
        incorrect1 = preprocess(example["Incorrect Answer 1"])
        incorrect2 = preprocess(example["Incorrect Answer 2"])
        incorrect3 = preprocess(example["Incorrect Answer 3"])

        all_choices = [incorrect1, incorrect2, incorrect3, correct]
        random.shuffle(all_choices)

        correct_index = all_choices.index(correct)
        correct_letter = chr(65 + correct_index)

        formatted_choices = ""
        for i, choice in enumerate(all_choices):
            letter = chr(65 + i)
            formatted_choices += f"{letter}) {choice}\n"
        
        # deepseek uses OpenAI's simple-eval for GPQA-Diamond, so we adopt prompts from here: https://github.com/openai/simple-evals/blob/main/gpqa_eval.py
        prompt = [{
            "role": "user", "content": "Answer the following multiple choice question. Lets' think step by step and output which of the following options is the correct answer within \\boxed{}." + "\n" + question + "\n" + formatted_choices
        }]
        
        data = {
            "data_source": "math-" + data_source,
            "prompt": prompt,
            "ability": "stem",
            "solution": correct_letter,
            "reward_model": {
                "style": "rule",
                "ground_truth": [correct_letter],
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "prompt": prompt,
                "solution": correct_letter,
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            print(f'\none prompt example is \n{prompt}')
            
        return data

    return process_fn

if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save GPQA dataset.")
    parser.add_argument('--data-dir', default='/export/home/research/verl_mt/data/openscience_filter/',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="stem",
                        help='Domain of the dataset.')
    parser.add_argument('--name', default="gpqa",
                        help='Name of the dataset.')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of samples to use from dataset. If None, use all samples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    data_source = f"{args.domain}__{args.name}"
    
    # Load the dataset
    _, dataset = get_datasets()

    # Process the dataset
    process_fn = make_map_fn('test', data_source)
    
    dataset = dataset.map(function=process_fn, with_indices=True)
    print(len(dataset))
    
    # Save the dataset to test directory

    dataset.to_parquet(os.path.join(args.data_dir, f"test_gpqa.parquet"))