#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7  # Set this variable as needed

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

accelerate launch --config_file "$REPO_ROOT/configs/accelerate_configs/multi_gpu.yaml" --num_processes 2 \
    "$REPO_ROOT/main_ppo.py" \
    --rag_method react_agent \
    --enable_stateful \
    --enable_sampling \
    --temperature 0.02 \
    --dataset_name hotpotqa \
    --dataset_train_split train \
    --output_dir "$REPO_ROOT/result/Qwen3-Embedding-0.6B-PPO" \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 10000 \
    --learning_rate 1e-4 \
    --kl_coef 5e-3 \
    --vf_coef 1e-3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 32 \
    --local_rollout_forward_batch_size 1 \
    --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
    --sft_model_path Qwen/Qwen3-Embedding-0.6B \
    --reward_model_path Qwen/Qwen3-Embedding-0.6B \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules all-linear \
    --lora_task_type FEATURE_EXTRACTION \
    --push_to_hub