#! /bin/bash
# RL fine-tuning with GRPO, initialized from a given SFT checkpoint, on the
# DeepMath or OpenScience RL split. Run once per SFT checkpoint of interest
# (e.g. epochs 1-5) to reproduce the SFT-extent-vs-RL-performance curves.
#
# Usage: ./scripts/train_grpo.sh <data_name: deepmath|openscience> <sft_ckpt_dir>
#   e.g. ./scripts/train_grpo.sh deepmath ./checkpoints/model/deepmath-sft-DeepSeek-R1-Distill-Qwen-1.5B/global_step_150

set -e

data_name=${1:-deepmath}
backbone_path=${2:?"pass the path to the SFT checkpoint to initialize RL from"}
backbone_name=$(basename "$backbone_path")

export PYTHONPATH=./:$PYTHONPATH
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

ADVANTAGE=grpo
response_length=10000
prompt_length=2500

train_path=./data/$data_name/rl/train_$data_name.parquet
test_path=./data/$data_name/rl/test_$data_name.parquet

if [ "$data_name" = "deepmath" ]; then
    test_files="['$test_path', './data/deepmath/math.parquet', './data/deepmath/minerva.parquet', './data/deepmath/olympiad_bench.parquet']"
else
    test_files="['$test_path', './data/openscience/gpqa_diamond.parquet', './data/openscience/gpqa.parquet']"
fi

WANDB_PROJECT="$data_name-$backbone_name-GRPO"
LOG_NAME="$backbone_name-$ADVANTAGE-$response_length"
OUTPUT_DIR="./checkpoints/$WANDB_PROJECT-$LOG_NAME"

python -m verl.trainer.rllm_ppo \
    algorithm.adv_estimator=$ADVANTAGE \
    data.train_files="['$train_path']" \
    data.val_files="$test_files" \
    data.train_batch_size=16 \
    data.max_prompt_length=$prompt_length \
    data.max_response_length=$response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$backbone_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((prompt_length + response_length)) \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$LOG_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=1

echo "RL checkpoints written to $OUTPUT_DIR"
