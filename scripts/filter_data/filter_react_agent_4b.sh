#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# HARR
python "$REPO_ROOT/scripts/filter_data.py" \
    --rag_method react_agent \
    --enable_stateful \
    --generator_model Qwen/Qwen3-8B \
    --collection_name wiki18-qwen3-embedding-4b \
    --dataset_names hotpotqa nq \
    --num_samples 2400 \
    --batch_size 64 \
    --num_generations 8 \
    --push_to_hub

# w/o History (ablation)
python "$REPO_ROOT/scripts/filter_data.py" \
    --rag_method react_agent \
    --generator_model Qwen/Qwen3-8B \
    --collection_name wiki18-qwen3-embedding-4b \
    --dataset_names hotpotqa nq \
    --num_samples 2400 \
    --batch_size 64 \
    --num_generations 8 \
    --push_to_hub