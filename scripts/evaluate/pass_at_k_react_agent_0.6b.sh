#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# ================== Frozen Retriever ==================
python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-0.6B \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --collection_name wiki18-qwen3-embedding-0.6b \
    --batch_size 1 \
    --num_samples 1000 \
    --num_generations 128 \
    --pass_at_ks 1 2 4 8 16 32 64 128 \
    --dataset_name hotpotqa

python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-0.6B \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --collection_name wiki18-qwen3-embedding-0.6b \
    --batch_size 1 \
    --num_samples 1000 \
    --num_generations 128 \
    --pass_at_ks 1 2 4 8 16 32 64 128 \
    --dataset_name nq
# ================== Frozen Retriever ==================

# ================== w/o RL (ablation) ==================
python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-0.6B \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-0.6b \
    --batch_size 1 \
    --num_samples 1000 \
    --num_generations 128 \
    --pass_at_ks 1 2 4 8 16 32 64 128 \
    --dataset_name hotpotqa

python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-0.6B \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-0.6b \
    --batch_size 1 \
    --num_samples 1000 \
    --num_generations 128 \
    --pass_at_ks 1 2 4 8 16 32 64 128 \
    --dataset_name nq
# ================== w/o RL (ablation) ==================