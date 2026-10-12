#!/usr/bin/env bash
set -euo pipefail

# Countdown GRPO training with SEC (Sample-Efficient Curriculum)
# Usage: bash examples/countdown_sec.sh <experiment_name> [strategy]
#   experiment_name: required, used as wandb run name and log file name
#   strategy: optional, one of random|difficulty|reverse_difficulty|bandit|pgs|hybrid (default: bandit)

if [ -z "$1" ]; then
    echo "Usage: bash examples/countdown_sec.sh <experiment_name> [strategy]"
    echo "  strategy: random|difficulty|reverse_difficulty|bandit|pgs|hybrid (default: bandit)"
    exit 1
fi

EXP_NAME=$1
STRATEGY=${2:-bandit}

# === Config ===
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
export VLLM_ATTENTION_BACKEND=XFORMERS

LOG_DIR=logs/countdown_sec
mkdir -p "$LOG_DIR"

echo "Experiment: $EXP_NAME"
echo "SEC strategy: $STRATEGY"
echo "GPUs: $N_GPUS"
echo "Log: ${LOG_DIR}/${EXP_NAME}.log"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/countdown/train.parquet \
    data.val_files=data/countdown/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name=countdown_sec \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.total_training_steps=500 \
    trainer.sec.enable=True \
    trainer.sec.strategy="$STRATEGY" \
    trainer.rejection_sample=True \
    2>&1 | tee "${LOG_DIR}/${EXP_NAME}.log"
