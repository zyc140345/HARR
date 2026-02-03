#!/bin/bash

export CUDA_VISIBLE_DEVICES=5  # Set this variable as needed

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

accelerate launch --config_file "$REPO_ROOT/configs/accelerate_configs/deepspeed_zero3.yaml" --num_processes 1 --main_process_port 0 \
    "$REPO_ROOT/main_grpo.py" \
    --model_name_or_path Qwen/Qwen3-Embedding-4B \
    --rag_method react_agent \
    --dataset_name hotpotqa_filtered \
    --dataset_train_split train \
    --collection_name wiki18-qwen3-embedding-4b \
    --embedding_base_url http://localhost:8001 \
    --generator_api_base http://127.0.0.1:8003/v1 \
    --generator_model Qwen/Qwen3-8B \
    --output_dir "$REPO_ROOT/result/qwen3-embedding-4b_react-agent_hotpotqa_ablation" \
    --num_generations 8 \
    --num_iterations 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --temperature 0.05 \
    --beta 0.0 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --per_device_eval_batch_size 128 \
    --max_steps 100 \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --report_to wandb \
    --ddp_find_unused_parameters false \
    --reward_funcs f1 \
    --reward_weights 1 \
    --fp16 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target_modules all-linear \
    --lora_task_type FEATURE_EXTRACTION \
    --push_to_hub \
    --timeout 600