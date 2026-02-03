# Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG
This repository contains the official implementation for the work 
"Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG".
> Retrieval-augmented generation (RAG) enables large language models (LLMs) to produce evidence-based responses, 
> and its performance hinges on the matching between the retriever and LLMs. Retriever optimization has emerged as 
> an efficient alternative to fine-tuning LLMs. However, existing solutions suffer from objective mismatch between 
> retriever optimization and the goal of RAG pipeline. Reinforcement learning (RL) provides a promising solution to 
> address this limitation, yet applying RL to retriever optimization introduces two fundamental challenges: 1) the 
> deterministic retrieval is incompatible with RL formulations, and 2) state aliasing arises from query-only retrieval 
> in multi-hop reasoning. To address these challenges, we replace deterministic retrieval with stochastic sampling and 
> formulate RAG as a Markov decision process, making retriever optimizable by RL. Further, we incorporate retrieval 
> history into the state at each retrieval step to mitigate state aliasing. Extensive experiments across diverse RAG 
> pipelines, datasets, and retriever scales demonstrate consistent improvements of our approach in RAG performance.

## Project Structure
```
.
├── main_grpo.py                 # GRPO training entry point
├── main_ppo.py                  # PPO training entry point
├── generator.py                 # Multi-hop RAG rollout implementations (ReAct / Search-R1)
├── prompt.py                    # Prompt templates for retrieval, decomposition, and QA formatting
├── reward.py                    # Reward function definitions
├── llama_index_hacked/          # Minimal adaptations of LlamaIndex for HARR training
│   ├── model.py                 # LlamaIndex wrappers for the TRL vLLM client
│   ├── query_engine.py          # RetrieverQueryEngine with stochastic retrieval for RL training
│   └── tool.py                  # StatefulQueryEngineTool for ReAct agents enabling history-aware state representation
├── trl_hacked/                  # Minimal adaptations of TRL for HARR training
│   ├── trainer/                 # Retriever trainers and training utilities
│   │   ├── grpo_trainer.py      # GRPO trainer for retriever fine-tuning
│   │   └── ppo_trainer.py       # PPO trainer for retriever fine-tuning
│   ├── vllm_client.py           # vLLM client for embedding rollout and NCCL parameter synchronization
│   └── vllm_serve.py            # vLLM server for embedding rollout and NCCL parameter synchronization
├── scripts/                     # Training, evaluation, and utility scripts
│   ├── vllm_serve_train.sh      # Launch embedding vLLM service (tmux-based)
│   ├── vllm_serve_infer.sh      # Launch generator vLLM service (tmux-based)
│   ├── build_index.py           # Qdrant index construction entry
│   ├── filter_data.py           # Data filtering entry
│   ├── evaluate_rag.py          # RAG evaluation entry
│   ├── grpo/                    # GRPO training scripts
│   ├── ppo/                     # PPO training scripts
│   ├── evaluate/                # Evaluation scripts
│   ├── filter_data/             # Data filtering scripts
│   ├── build_index/             # Qdrant index construction scripts
│   └── qdrant/                  # Qdrant Docker Compose setup
└── configs/                     # Accelerate / DeepSpeed configuration files
```

## Environment Setup
```shell
wget https://anonymous.4open.science/api/repo/HARR-CF32/zip -O HARR.zip
unzip -d HARR HARR.zip && cd HARR

# step1: create conda env
conda create -n harr python=3.11 && conda activate harr
pip install -r requirements.txt

# step2: download retrieval corpus
mkdir -p data && cd data
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/retrieval-corpus/wiki18_100w.zip
unzip wiki18_100w.zip

# step3: login to huggingface and wandb
huggingface-cli login  # use a token with write permission
wandb login

# step4: build vector index
cd ../scripts/qdrant
docker compose up -d
cd ..
./vllm_serve_infer.sh \
    -s serve_embed \
    -g 0,1,2,3,4,5,6,7 \
    --gpu-util 0.95 \
    --model Qwen/Qwen3-Embedding-0.6B \
    --max-len 2048 \
    --port 8001
python build_index.py \
    --embedding_size 1024 \
    --collection_name wiki18-qwen3-embedding-0.6b
  
# step5: stop the vLLM server
tmux kill-session -t serve_embed
```

## Example Usage
Assume you are in the project root (same for the following sections).

### Filter Training Data
```shell
# step1: start vLLM servers for inference
cd scripts
./vllm_serve_train.sh \
    -s serve_embed -g 0 \
    --serve-embed \
    --embed-model Qwen/Qwen3-Embedding-0.6B \
    --embed-util 0.95
./vllm_serve_infer.sh \
    -s serve_gen --gpu 1,2,3,4,5,6 \
    --model Qwen/Qwen3-8B \
    --gpu-util 0.95

# step2: run data filtering script
python filter_data.py \
    --rag_method react_agent \
    --enable_stateful \
    --generator_model Qwen/Qwen3-8B \
    --collection_name wiki18-qwen3-embedding-0.6b \
    --dataset_names hotpotqa \
    --num_samples 2400 \
    --batch_size 64 \
    --num_generations 8 \
    --push_to_hub
    
# step3: stop the vLLM servers when filtering is done
tmux kill-session -t serve_embed
tmux kill-session -t serve_gen
```

### Train GRPO
```shell
# step1: start vLLM servers for rollout
cd scripts
./vllm_serve_train.sh \
    -s serve_embed \
    -g 0 \
    --serve-embed \
    --embed-model Qwen/Qwen3-Embedding-0.6B \
    --embed-util 0.95
./vllm_serve_infer.sh \
    -s serve_gen \
    --gpu 1,2 \
    --model Qwen/Qwen3-8B \
    --gpu-util 0.95

# step2: run training script
cd grpo
./train_react_agent_hotpotqa_0.6b.sh  # modify the CUDA_VISIBLE_DEVICES inside the script as needed

# step3: stop the vLLM servers when training is done
tmux kill-session -t serve_embed
tmux kill-session -t serve_gen
```

### Evaluate GRPO
```shell
# step1: start vLLM servers for inference
cd scripts
./vllm_serve_train.sh \
    -s serve_embed -g 0 \
    --serve-embed \
    --embed-model Qwen/Qwen3-Embedding-0.6B \
    --embed-util 0.95
./vllm_serve_infer.sh \
    -s serve_gen --gpu 1,2,3,4,5,6 \
    --model Qwen/Qwen3-8B \
    --gpu-util 0.95

# step2: run evaluation script
python evaluate_rag.py \
    --model_id Qwen/Qwen3-Embedding-0.6B \
    --peft_model_id qwen3-embedding-0.6b_react-agent_hotpotqa \
    --generator_model Qwen/Qwen3-8B \
    --rag_method react_agent \
    --enable_stateful \
    --collection_name wiki18-qwen3-embedding-0.6b \
    --batch_size 512 \
    --dataset_name hotpotqa
    
# step3: stop the vLLM servers when evaluation is done
tmux kill-session -t serve_embed
tmux kill-session -t serve_gen
```
