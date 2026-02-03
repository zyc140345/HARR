#!/bin/bash

export CUDA_VISIBLE_DEVICES=7  # Set this variable as needed

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# ================== Frozen Retriever ==================
python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --generator_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \
    --rag_method search_r1 \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name hotpotqa

python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --generator_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \
    --rag_method search_r1 \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name nq
# ================== Frozen Retriever ==================

# ================== REPLUG ==================
python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --peft_model_id qwen3-embedding-4b_qwen3-8b_hotpotqa_lsr \
    --generator_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \
    --rag_method search_r1 \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name hotpotqa

python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --peft_model_id qwen3-embedding-4b_qwen3-8b_nq_lsr \
    --generator_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \
    --rag_method search_r1 \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name nq
# ================== REPLUG ==================

# ================== HARR ==================
python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --peft_model_id qwen3-embedding-4b_search-r1_hotpotqa \
    --generator_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \
    --rag_method search_r1 \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name hotpotqa

python "$REPO_ROOT/scripts/evaluate_rag.py" \
    --model_id Qwen/Qwen3-Embedding-4B \
    --peft_model_id qwen3-embedding-4b_search-r1_nq \
    --generator_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \
    --rag_method search_r1 \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-4b \
    --batch_size 512 \
    --dataset_name nq
# ================== HARR ==================