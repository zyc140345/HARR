#!/bin/bash

export CUDA_VISIBLE_DEVICES=7  # Set this variable as needed

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# ================== Frozen Retriever ==================
python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name hotpotqa

python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name nq
# ================== Frozen Retriever ==================

# ================== REPLUG ==================
python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --peft_model_id qwen3-embedding-4b_qwen3-8b_hotpotqa_lsr \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name hotpotqa

python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --peft_model_id qwen3-embedding-4b_qwen3-8b_nq_lsr \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name nq
# ================== REPLUG ==================

# ================== HARR ==================
python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --peft_model_id qwen3-embedding-4b_react-agent_hotpotqa \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name hotpotqa

python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --peft_model_id qwen3-embedding-4b_react-agent_nq \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name nq
# ================== HARR ==================

# ================== w/o History (ablation) ==================
python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --peft_model_id qwen3-embedding-4b_react-agent_hotpotqa_ablation \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name hotpotqa

python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --peft_model_id qwen3-embedding-4b_react-agent_nq_ablation \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name nq
# ================== w/o History (ablation) ==================

# ================== w/o RL (ablation) ==================
python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name hotpotqa

python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name nq
# ================== w/o RL (ablation) ==================