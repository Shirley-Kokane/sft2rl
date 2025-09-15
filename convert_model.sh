#! /bin/bash
# Convert a sweep of GRPO RL checkpoints (FSDP shards) to Hugging Face format.
#
# Usage: ./convert_model.sh <domain: DeepMath|OpenScience> <model_size: 1.5B|7B>

set -x

domain=$1
model_size=$2

model_name=DeepSeek-R1-Distill-Qwen-$model_size
ckpt_root=./checkpoints/$domain-$model_size-GRPO

for global_step in 100 200 300 400 500; do
    ckpt_path=$ckpt_root/global_step_${global_step}/actor/
    hf_path=$ckpt_root-hf/global_step_${global_step}/

    mkdir -p ${hf_path}

    python convert_fsdp_hf.py ${ckpt_path} ./checkpoints/model/${model_name} ${hf_path}
done
