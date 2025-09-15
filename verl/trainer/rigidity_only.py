from operator import truediv
from types import NoneType
from verl.utils.torch_functional import entropy_from_logits, logprobs_from_logits
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import pandas as pd
import torch
import argparse
from verl.utils.model import compute_position_id_with_mask
import torch.nn.functional as F
from verl.trainer.ppo.core_algos import agg_loss
from base_utils import avg_delta_logprob_on_alternatives, rigidity_index_from_logits 
from tqdm import tqdm
import numpy as np


    

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
    actor_model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype="bfloat16")
    actor_model_config.torch_dtype = "bfloat16"
    actor_model_config.bos_token_id = tokenizer.bos_token_id
    actor_model_config.eos_token_id = tokenizer.eos_token_id
    actor_model_config.pad_token_id = tokenizer.pad_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=args.model_path,
                    torch_dtype="bfloat16",
                    trust_remote_code=True,
                    config=actor_model_config,
                    device_map="auto",
                    #max_memory={2: "135GB", 3 : "135GB"},
                    )#.to("cuda")
    model.gradient_checkpointing_enable()
    print("main model loaded")
    
    dataset = pd.read_parquet(args.data_path)
    responses = dataset["responses"].tolist()
    scores = dataset["scores"].tolist()
    prompts = dataset["prompt"].tolist()
    
    rollout_outputs = []
    prompt_length = []
    for i in range(len(responses)):
        for sample,score in zip(responses[i].tolist(), scores[i].tolist()):
            if args.positive_only and score <= 0:
                continue
            prompt_chat = tokenizer.apply_chat_template(prompts[i],
                                                    add_generation_prompt=True,
                                                    padding=True,
                                                    truncation=True,
                                                    tokenize=False)
            rollout_outputs.append(prompt_chat + " " + str(sample))
            prompt_length.append(len(prompt_chat)+1)
    
    
    rigidity_index_5 = 0
    rigidity_index_3 = 0
    count = 0
    batch_size = 8  # or 16, 32 depending on GPU memory
    for i in tqdm(range(0, len(rollout_outputs), batch_size)):
        batch_outputs = rollout_outputs[i:i+batch_size]
        batch_lengths = prompt_length[i:i+batch_size]
    
        # Tokenize batch together with padding
        batch = tokenizer(batch_outputs, return_tensors="pt", padding=True).to("cuda")
        
        position_ids = compute_position_id_with_mask(batch['attention_mask'])
        with torch.no_grad():
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], position_ids=position_ids, use_cache=False, temperature=args.temperature, return_dict=True)
        
        rigidity_value_5 = rigidity_index_from_logits(output.logits, loss_mask=batch["attention_mask"], k=5)
        rigidity_value_3 = rigidity_index_from_logits(output.logits, loss_mask=batch["attention_mask"], k=3)
        rigidity_index_5 += rigidity_value_5
        rigidity_index_3 += rigidity_value_3
        count += len(batch_outputs)
            
            #rigidity_value_5, rigidity_value_3 = rigidity_calculate(output, batch["attention_mask"], batch_lengths, args.temperature)
    #         
    print(rigidity_index_5 / len(rollout_outputs), rigidity_index_3 / len(rollout_outputs))
    df = pd.read_csv(args.csv_path)    
    # print(rigidity_index_5 / count , rigidity_index_3 / count )
    df.loc[(df['model_path'] == args.model_path.split('/')[-1]) & (df["temperature"] == args.temperature), f'rigidity_index_5_{args.positive_only}'] = rigidity_index_5 / count
    df.loc[(df['model_path'] == args.model_path.split('/')[-1]) & (df["temperature"] == args.temperature), f'rigidity_index_3_{args.positive_only}'] = rigidity_index_3 / count
    df.loc[(df['model_path'] == args.model_path.split('/')[-1]) & (df["temperature"] == args.temperature), f'count_{args.positive_only}'] = count
    df.to_csv(args.csv_path, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/fsx/home/skokane/research/verl_mt/checkpoints/global_step_100")
    parser.add_argument("--data_path", type=str, default="/fsx/home/skokane/research/eval_results/format_deepmath_100.parquet")
    parser.add_argument("--positive_only", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--csv_path", type=str, default="/fsx/home/skokane/research/eval_results/deepmath/rl_results/rl_deepmath.csv")
    parser.add_argument("--rollout_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
