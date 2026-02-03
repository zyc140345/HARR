#!/bin/bash

# Default parameters
conda_home="$(conda info --base 2>/dev/null)"
session_name="vllm_serve_infer"
gpu_id="0"
port="8003"
max_len="4096"
gpu_util="0.95"
model="Qwen/Qwen2.5-7B-Instruct"
hf_overrides_arg=""

# Error handling
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Arg validation
validate_arg() {
    [[ -n "$2" && ! "$2" =~ ^- ]] || error_exit "Option $1 requires an argument"
    echo "$2"
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--session-name) session_name=$(validate_arg "$1" "$2"); shift ;;
        -g|--gpu) gpu_id=$(validate_arg "$1" "$2"); shift ;;
        -p|--port) port=$(validate_arg "$1" "$2"); shift ;;
        --max-len) max_len=$(validate_arg "$1" "$2"); shift ;;
        --gpu-util) gpu_util=$(validate_arg "$1" "$2"); shift ;;
        --model) model=$(validate_arg "$1" "$2"); shift ;;
        *) error_exit "Unknown option: $1" ;;
    esac
    shift
done

# Compute data_parallel_size
data_parallel_size=$(echo "$gpu_id" | awk -F, '{print NF}')
model_lc=$(echo "$model" | tr '[:upper:]' '[:lower:]')
if [[ "$model_lc" == *"qwen3-reranker"* ]]; then
    hf_overrides_arg="--hf_overrides '{\"architectures\": [\"Qwen3ForSequenceClassification\"],\"classifier_from_token\": [\"no\", \"yes\"],\"is_original_qwen3_reranker\": true}'"
fi

# Create and attach to the session
tmux new-session -d -s "$session_name" "bash <<'EOF'
source $conda_home/etc/profile.d/conda.sh
while true; do
    conda activate harr
    CUDA_VISIBLE_DEVICES=$gpu_id vllm serve \\
        $model \\
        --tensor-parallel-size 1 \\
        --data-parallel-size $data_parallel_size \\
        --max-model-len $max_len $hf_overrides_arg \\
        --gpu-memory-utilization $gpu_util \\
        --dtype bfloat16 \\
        --host 0.0.0.0 \\
        --port $port
    echo 'Service exited. Restarting in 3 seconds...'
    sleep 3
done
EOF"

# Wait for the server port to be ready, then return (no auto-attach).
echo "Waiting for vLLM to listen on port $port..."
while ! ss -ltn | awk '{print $4}' | grep -q ":$port\$"; do
    sleep 1
done
echo "vLLM server is up. tmux session: $session_name (attach with: tmux attach -t $session_name)"
