#! /bin/bash
# Supervised fine-tuning of a DeepSeek-R1-Distill-Qwen model on the DeepMath or
# OpenScience SFT split, checkpointing every epoch so that accuracy and the
# Rigidity Index can be tracked across SFT extent.
#
# Usage: ./scripts/train_sft.sh <data_name: deepmath|openscience> <model_name>
#   e.g. ./scripts/train_sft.sh deepmath deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

set -x

data_name=${1:-deepmath}
model_name=${2:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
model_short_name=$(basename "$model_name")

export PYTHONPATH=./:$PYTHONPATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn

response_length=12048
current_save_path=./checkpoints/model/$data_name-sft-$model_short_name

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
        data.train_files=./data/$data_name/sft/train_$data_name.parquet \
        data.val_files=./data/$data_name/sft/test_$data_name.parquet \
        data.train_batch_size=32 \
        data.max_length=$response_length \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        data.prompt_dict_keys=['prompt'] \
        data.response_dict_keys=['solution'] \
        data.micro_batch_size_per_gpu=1 \
        model.partial_pretrain=$model_name \
        trainer.default_local_dir=$current_save_path \
        trainer.project_name=$data_name-base-sft \
        trainer.experiment_name=$model_short_name-$response_length \
        trainer.total_epochs=5 \
        trainer.save_freq=50 \
        trainer.test_freq=5 \
        trainer.logger=['console','wandb'] \
        trainer.default_hdfs_dir=null

echo "SFT checkpoints written to $current_save_path"
