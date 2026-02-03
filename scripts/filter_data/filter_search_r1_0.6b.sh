#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# HARR
python "$REPO_ROOT/scripts/filter_data.py" \
    --rag_method search_r1 \
    --enable_stateful \
    --generator_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \
    --collection_name wiki18-qwen3-embedding-0.6b \
    --dataset_names hotpotqa nq \
    --num_samples 2400 \
    --batch_size 64 \
    --num_generations 8 \
    --push_to_hub

# w/o History (ablation)
python "$REPO_ROOT/scripts/filter_data.py" \
    --rag_method search_r1 \
    --generator_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \
    --collection_name wiki18-qwen3-embedding-0.6b \
    --dataset_names hotpotqa nq \
    --num_samples 2400 \
    --batch_size 64 \
    --num_generations 8 \
    --push_to_hub