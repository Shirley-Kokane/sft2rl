# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src/x_r1/benchmark.py \
# 	--model_name='xiaodongguaAIGC/X-R1-3B \
#   --dataset_name='HuggingFaceH4/MATH-500' \
# 	--output_name='./output/result_benchmark_math500'  \
# 	--max_output_tokens=1024 \
# 	--num_gpus=4

# CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
# 	--model_name='xiaodongguaAIGC/X-R1-3B' \
#     --dataset_name='HuggingFaceH4/MATH-500' \
# 	--output_name='./output/result_benchmark_math500'  \
# 	--max_output_tokens=1024 \
# 	--num_gpus=1


from datasets import load_dataset, Dataset, DatasetDict
from vllm import LLM, SamplingParams
import argparse
import json
# import torch
from verl.utils.reward_score.math_batch import compute_score_batched
from verl.utils.reward_score.ttrl.qwen.qwen_eval import qwen_reward_fn
from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer
import re
from transformers import AutoTokenizer 
import openai
from openai import OpenAI
import os
import pdb
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from utils import calculate_bleu, calculate_diversity
import numpy as np

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively. Output the final answer within the <answer>...</answer> tags after \"####\"."


def format_reward(completion):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    matches = re.match(pattern, completion)
    rewards = 1.0 if matches else 0.0 
    return rewards, matches


def create_dataset(dataset_name, tokenizer):
    
    dataset = load_dataset("json", data_files=dataset_name)

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["prompt"]},
            ],
        }

    dataset = dataset.map(make_conversation)

    def format_function(example):
        example['prompt'] = tokenizer.apply_chat_template(example['prompt'], tokenize = False, add_generation_prompt = True )
        return example
    
    dataset = dataset.map(format_function, batched = False)
        
    return dataset


def vllm_generate(args):
    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_path
    num_gpus = args.num_gpus
    max_output_tokens = args.max_output_tokens
    temperature = args.temperature
    n = args.n
    if "gpt" not in model_name:

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # evaluation dataset
        dataset = create_dataset(dataset_name, tokenizer)

        answers = []
        prompts = []
        for data in dataset["train"]:
            for i in range(n):
                answers.append(data['solution'])
                prompts.append(data['prompt'])

        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=temperature,
                                        max_tokens=max_output_tokens,
                                        logprobs=20,
                                        n=1,
                                        )
        # Create LLM object
        
        llm = LLM(model=model_path,  # replace your own model
                dtype='bfloat16',
                tensor_parallel_size=num_gpus,  # number of gpu
                gpu_memory_utilization=0.9,  # prevent OOM
                trust_remote_code=True,
                # use_cache=False,
            )
        # # vllm generation
        outputs = llm.generate(prompts,
                            sampling_params,)

        return outputs, prompts, answers 
    else:
        dataset = load_dataset(dataset_name, "default")
        # Initialize OpenAI client
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )
        
        
        answers = []
        prompts = []
        outputs = []
        for data in tqdm(dataset):
            answers.append(data['solution'])
            prompts.append(data['prompt'])
            
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": data['prompt']},
                ],
                temperature=0
            )
            outputs.append(response.choices[0].message.content)

        
        
        return outputs, prompts, answers

def calculate_scores(args,outputs, prompts, answers ):
    acc_scores = []
    format_scores = []
    result_all = []
    total_acc = 0
    total_format = 0
    bleu_scores = []
    diversity_scores = []
    diversity_score = 0
    for i in range(0, len(outputs),args.n):
        per_sample_acc =[]
        success_samples = []
        for j in range(args.n):
            completion = outputs[i+j].outputs[0].text
            prompt = prompts[i+j]
            gold_answer = answers[i+j]
            score = qwen_reward_fn(completion, gold_answer)
            if args.only_sucess and score != 0:
                success_samples.append(completion)
            else:
                success_samples.append(completion)
            per_sample_acc.append(score)
            
        if len(success_samples) > 0:
            bleu_score = calculate_bleu(success_samples)
            diversity_score = calculate_diversity(success_samples, diversity_score)
            bleu_scores.append(bleu_score)
            diversity_scores.append(diversity_score)
            print('bleu score', bleu_score)
            print('diversity score', diversity_score)
        
        acc_score = sum(per_sample_acc) / len(per_sample_acc)
        acc_scores.append(acc_score)

        print('accuracy score', acc_score)
        print('-'*100)
        
        result_all.append({
            'prompt': prompt, 
            'completions': success_samples,
            'gold_answer': gold_answer, 
            'source': args.data_name,
        })

    print('='*100)
    print('eval acc: ', np.mean(acc_scores).item())
    print('success samples: ', len(bleu_scores),'/', len(acc_scores))
    print('eval bleu: ', np.mean(bleu_scores).item())
    print('eval diversity: ', np.mean(diversity_scores).item())
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path',  type=str, default='/export/home/checkpoints/Qwen2.5-1.5B-Instruct/', required=False,
                        help='model name path')
    parser.add_argument('--model_name',  type=str, default='Qwen/Qwen2.5-Math-1.5B-Instruct', required=False,
                        help='model name path')
    parser.add_argument('--output_path', type=str, default=None, required=False,
                        help='output path')
    parser.add_argument('--dataset_path', type=str, default='/export/home/research/TTRL/verl/clue_data/AIME-ANS/test.json', required=False,
                        help='dataset path')
    parser.add_argument('--data_name', type=str, default='aime', required=False,
                        help='data name')
    parser.add_argument('--max_output_tokens', type=int, default=1024,
                        help='generation tokens')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='generation tokens')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='generation tokens')
    parser.add_argument('--n', type=int, default=5,
                        help='no. of samples')
    parser.add_argument("--only_sucess", type=bool, default=False,
                        help='check diversity for only sucessful samples')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='generation tokens')

    args = parser.parse_args()
    print(args)

    #convert_fsdp_hf("/export/home/research/TTRL/checkpoints/BASE-verl/AIME-TTT-Qwen2.5-Math-1.5B-Instruct/0625/BASE-Len@3k-grpo-142455/global_step_240/actor","Qwen/Qwen2.5-Math-1.5B-Instruct", "/export/home/research/TTRL/checkpoints/BASE-verl/AIME-TTT-Qwen2.5-Math-1.5B-Instruct/0625/BASE-Len@3k-grpo-142455/hf/")

    outputs, prompts, answers = vllm_generate(args)

    calculate_scores(args, outputs, prompts, answers)
    #print(f'toxicity score mean: {mean}, toxicity score std: {std}')