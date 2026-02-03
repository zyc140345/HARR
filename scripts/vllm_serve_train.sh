#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Default parameters
declare -A config=(
    [conda_home]="$(conda info --base 2>/dev/null)"
    [serve_script]="$REPO_ROOT/trl_hacked/vllm_serve.py"
    [session_name]="vllm_serve_train"
    [gpu_id]="0"
    [data_parallel_size]="1"
)

declare -A services=(
    [embed]="false:8001:0.05:Qwen/Qwen3-Embedding-0.6B"
    [rerank]="false:8002:0.2:Qwen/Qwen3-Reranker-0.6B"
)

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
        -s|--session-name) config[session_name]=$(validate_arg "$1" "$2"); shift ;;
        -g|--gpu)
            config[gpu_id]=$(validate_arg "$1" "$2")
            config[data_parallel_size]=$(echo "${config[gpu_id]}" | awk -F, '{print NF}')
            shift ;;
        --serve-embed) services[embed]="true:${services[embed]#*:}" ;;
        --serve-rerank) services[rerank]="true:${services[rerank]#*:}" ;;
        --embed-util|--rerank-util)
            svc=${1#--}; svc=${svc%-util}
            val=$(validate_arg "$1" "$2")
            IFS=: read -r enabled port _ model <<< "${services[$svc]}"
            services[$svc]="$enabled:$port:$val:$model"
            shift ;;
        --embed-port|--rerank-port)
            svc=${1#--}; svc=${svc%-port}
            port=$(validate_arg "$1" "$2")
            IFS=: read -r enabled _ gpu_util model <<< "${services[$svc]}"
            services[$svc]="$enabled:$port:$gpu_util:$model"
            shift ;;
        --embed-model|--rerank-model)
            svc=${1#--}; svc=${svc%-model}
            model=$(validate_arg "$1" "$2")
            IFS=: read -r enabled port gpu_util _ <<< "${services[$svc]}"
            services[$svc]="$enabled:$port:$gpu_util:$model"
            shift ;;
        *) error_exit "Unknown option: $1" ;;
    esac
    shift
done

# Helper: create a service window
create_service_window() {
    local svc=$1 window_id=$2 prev_svc=$3
    IFS=: read -r enabled port gpu_util model <<< "${services[$svc]}"

    [[ "$enabled" != "true" ]] && return

    local extra_args=""
    [[ "$svc" == "rerank" ]] && extra_args="--enable_prefix_caching true"

    # Decide whether to wait for a dependency service
    local startup_cmd=""
    if [[ -n "$prev_svc" ]] && [[ "${services[$prev_svc]%%:*}" == "true" ]]; then
        IFS=: read -r _ prev_port _ _ <<< "${services[$prev_svc]}"
        startup_cmd="echo 'Waiting for ${prev_svc} to start...'
while ! ss -ltn | awk '{print \$4}' | grep -q ':$prev_port\$'; do sleep 2; done
echo '${prev_svc} is ready. Starting ${svc}...'"
    else
        startup_cmd="echo 'Starting ${svc}...'"
    fi

    tmux new-window -t "${config[session_name]}":"$window_id" -n "$svc"
    tmux send-keys -t "${config[session_name]}":"$svc" "bash <<'EOF'
source ${config[conda_home]}/etc/profile.d/conda.sh
clear
$startup_cmd
while true; do
    conda activate harr
    CUDA_VISIBLE_DEVICES=${config[gpu_id]} python ${config[serve_script]} \\
        --model $model \\
        --enforce_eager false \\
        --tensor_parallel_size 1 \\
        --data_parallel_size ${config[data_parallel_size]} \\
        --max_model_len 1024 \\
        --gpu_memory_utilization $gpu_util \\
        --dtype bfloat16 \\
        --host 0.0.0.0 \\
        --port $port \\
        $extra_args
    echo '${svc} exited. Restarting in 3 seconds...'
    sleep 3
done
EOF" Enter
}

# Create the main session
tmux new-session -d -s "${config[session_name]}"

# Create service windows
create_service_window "embed" 2 ""
create_service_window "rerank" 3 "embed"

# Control panel
tmux send-keys -t "${config[session_name]}":1 "bash <<'EOF'
clear
echo 'Service control panel - press Ctrl+C to stop all services'
echo 'Windows:'
echo '  1: main (current)'
i=2
# Service info
declare -A services_info=(
    [embed]=\"${services[embed]}\"
    [rerank]=\"${services[rerank]}\"
)
for svc in embed rerank; do
    IFS=: read -r enabled port _ _ <<< \"\${services_info[\$svc]}\"
    echo \"  \$i: \$svc \$([ \"\$enabled\" = \"true\" ] && echo \"(port: \$port)\" || echo \"(disabled)\")\"
    ((i++))
done
echo -e '\\nSwitch window: Ctrl+b then <number>\\nLogs: switch to the service window\\n'
trap 'tmux kill-session -t ${config[session_name]}; exit' INT
while true; do echo -n '.'; sleep 5; done
EOF" Enter

# Wait for enabled services to listen, then return (no auto-attach).
wait_for_port() {
    local port="$1"
    echo "Waiting for port $port..."
    while ! ss -ltn | awk '{print $4}' | grep -q ":${port}\$"; do
        sleep 1
    done
}

IFS=: read -r embed_enabled embed_port _ _ <<< "${services[embed]}"
IFS=: read -r rerank_enabled rerank_port _ _ <<< "${services[rerank]}"

[[ "$embed_enabled" == "true" ]] && wait_for_port "$embed_port"
[[ "$rerank_enabled" == "true" ]] && wait_for_port "$rerank_port"

echo "Services are up. tmux session: ${config[session_name]} (attach with: tmux attach -t ${config[session_name]})"
