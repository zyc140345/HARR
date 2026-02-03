import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys
sys.path.append(repo_root)
from dotenv import load_dotenv
load_dotenv()

# Setup Tracer (for debugging)
# from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
# from phoenix.otel import register
# tracer_provider = register()
# LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

import argparse
import asyncio
import json
import torch
import datasets as ds
import evaluate

from math import comb
from typing import Any
from transformers import AutoModel, AutoTokenizer, set_seed
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from llama_index.core import Settings
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient
from llama_index_hacked.model import Qwen3EmbeddingServer, get_remote_llm
from llama_index_hacked.query_engine import RetrieverQueryEngine
from prompt import RETRIEVAL_INSTRUCTION, STATEFUL_RETRIEVAL_INSTRUCTION
from generator import agent_generate, workflow_generate, search_r1_generate


squad_metric = evaluate.load('squad')


def move_model_to_vllm(vllm_client, args):
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    if args.peft_model_id:
        model = PeftModel.from_pretrained(model, _resolve_peft_model_id(args.peft_model_id), force_download=True)
    model = model.to(f"cuda:{args.device}")

    if args.peft_model_id:
        model.merge_adapter()
        for name, param in model.named_parameters():
            # When using PEFT, we need to recover the original parameter name and discard some parameters
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            if model.prefix in name:
                continue
            # When module to save, remove its prefix and discard the original module
            if "original_module" in name:
                continue
            name = name.replace("modules_to_save.default.", "")
            vllm_client.update_named_param(name, param.data)
    else:
        # For non-PEFT models, simply update each parameter individually.
        for name, param in model.named_parameters():
            vllm_client.update_named_param(name, param.data)

    vllm_client.reset_prefix_cache()
    del model
    torch.cuda.empty_cache()


def compute_metric(response, ground_truth):
    out = squad_metric.compute(
        predictions=[{"id": "0", "prediction_text": response}],
        references=[
            {
                "id": "0",
                "answers": {
                    "text": ground_truth,
                    "answer_start": [0] * len(ground_truth),
                },
            }
        ],
    )
    em = float(out["exact_match"])
    f1 = float(out["f1"])
    return em, f1


def get_save_dir(args):
    if args.peft_model_id:
        dir_name = args.peft_model_id
    else:
        model_id = args.model_id.split("/")[-1].lower()
        dataset_name = args.dataset_name
        rag_method = args.rag_method.replace("_", "-")
        ablation = "_ablation" if not args.enable_stateful else ""
        dir_name = f"{model_id}_{dataset_name}_{rag_method}{ablation}"
    return f"{repo_root}/result/{dir_name}"


def _resolve_peft_model_id(peft_model_id: str) -> str:
    if peft_model_id and "/" in peft_model_id:
        return peft_model_id
    import huggingface_hub
    hf_username = huggingface_hub.whoami()["name"]
    if not hf_username:
        raise ValueError(
            "Could not infer HuggingFace username. Pass peft_model_id as 'username/model' "
            "or login using `huggingface-cli login`."
        )
    return f"{hf_username}/{peft_model_id}"


async def main(args):
    set_seed(args.seed)

    if args.num_generations <= 0:
        raise ValueError("num_generations must be a positive integer.")
    pass_at_ks = sorted(set(args.pass_at_ks)) if args.pass_at_ks else []
    if any(k <= 0 for k in pass_at_ks):
        raise ValueError("Values in pass_at_ks must be positive.")
    if pass_at_ks and max(pass_at_ks) > args.num_generations:
        raise ValueError("All pass@k values must be <= num_generations.")
    if args.trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if args.similarity_top_k <= 0:
        raise ValueError("similarity_top_k must be positive.")
    if args.sampling_pool_size <= 0:
        raise ValueError("sampling_pool_size must be positive.")
    if args.similarity_top_k > args.sampling_pool_size:
        raise ValueError("similarity_top_k must be <= sampling_pool_size.")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive.")

    # Setup Models
    if args.model_id:
        Settings.embed_model = Qwen3EmbeddingServer(
            base_url=args.embedding_base_url,
            instruction=STATEFUL_RETRIEVAL_INSTRUCTION if args.enable_stateful else RETRIEVAL_INSTRUCTION,
            init_communicator=True
        )
        vllm_client = Settings.embed_model.client
        move_model_to_vllm(vllm_client, args)
    else:
        Settings.embed_model = Qwen3EmbeddingServer(
            base_url=args.embedding_base_url,
            instruction=STATEFUL_RETRIEVAL_INSTRUCTION if args.enable_stateful else RETRIEVAL_INSTRUCTION,
        )
    Settings.llm = get_remote_llm(
        api_base=args.generator_api_base,
        api_key=args.generator_api_key,
        model=args.generator_model,
        context_window=args.generator_context_window,
        enable_thinking=args.generator_enable_thinking,
        seed=args.seed
    )
    if args.rag_method == "search_r1":
        processing_class = AutoTokenizer.from_pretrained("PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")

    # Setup Vector DB
    qdrant_client = AsyncQdrantClient(url=args.qdrant_url, prefer_grpc=True, timeout=args.timeout)
    vector_store = QdrantVectorStore(
        aclient=qdrant_client,
        collection_name=args.collection_name,
        timeout=args.timeout
    )
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Setup Query Engine
    retriever = index.as_retriever(similarity_top_k=args.sampling_pool_size)
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        similarity_top_k=args.similarity_top_k,
        temperature=args.temperature,
    )
    mode = "train" if args.num_generations > 1 else "eval"
    if mode == "train":
        query_engine.train()
    else:
        query_engine.eval()

    # Load Test Dataset
    dataset = ds.load_dataset("RUC-NLPIR/FlashRAG_datasets", args.dataset_name)
    dataset = dataset['dev'] if 'dev' in dataset else dataset['test']
    if args.num_samples is not None:
        dataset = dataset.shuffle()
        dataset = dataset.select(range(args.num_samples))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x
    )

    # Perform Evaluation
    def format_result(response, sub_queries, cur_responses, source_nodes):
        return (
            response,
            [sub_query.query_str for sub_query in sub_queries],
            cur_responses,
            [[node.text for node in source_node[:args.similarity_top_k]] for source_node in source_nodes],
        )

    def compute_pass_at_k_estimator(num_correct, num_samples, k):
        """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
        if k > num_samples or num_samples <= 0:
            return 0.0
        total = comb(num_samples, k)
        if total == 0:
            return 0.0
        fail_ways = comb(num_samples - num_correct, k) if (num_samples - num_correct) >= k else 0
        return 1.0 - fail_ways / total

    async def get_responses_for_example(example):
        tasks = []
        for _ in range(args.num_generations):
            if args.rag_method == "react_agent":
                task = agent_generate(
                    query=example["question"],
                    query_engine=query_engine,
                    retriever=retriever,
                    args=args,
                )
            elif args.rag_method == "multistep_workflow":
                task = workflow_generate(
                    query=example["question"],
                    query_engine=query_engine,
                    retriever=retriever,
                    args=args,
                )
            else:
                task = search_r1_generate(
                    query=example["question"],
                    retriever=retriever,
                    processing_class=processing_class,
                    mode=mode,
                    args=args,
                )
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        results = [format_result(*result) for result in results]
        return results

    sum_em = 0.0
    sum_f1 = 0.0
    num_total = 0
    pass_at_k_score = {k: 0.0 for k in pass_at_ks}
    traces = []
    example_idx = 0
    for batch in tqdm(dataloader, desc="Testing RAG"):
        tasks = [get_responses_for_example(example) for example in batch]
        batch_results = await asyncio.gather(*tasks)
        for attempt_results, example in zip(batch_results, batch):
            ground_truth = example['golden_answers']
            attempt_ems = []
            attempt_f1s = []
            attempt_corrects = []
            attempts = []
            for response, sub_queries, cur_responses, source_nodes in attempt_results:
                em, f1 = compute_metric(response, ground_truth)
                attempt_ems.append(em)
                attempt_f1s.append(f1)
                attempt_corrects.append(em > 0.0)
                attempts.append({
                    "response": response,
                    "sub_queries": sub_queries,
                    "cur_responses": cur_responses,
                    "source_nodes": source_nodes,
                    "em": em,
                    "f1": f1,
                    "is_correct": em > 0.0,
                })

            sum_em += attempt_ems[0]
            sum_f1 += attempt_f1s[0]
            num_total += 1

            pass_estimates = {
                k: compute_pass_at_k_estimator(sum(attempt_corrects), args.num_generations, k)
                for k in pass_at_ks
            }
            for k, val in pass_estimates.items():
                pass_at_k_score[k] += val

            trace_entry = {
                "question": example["question"],
                "ground_truth": ground_truth,
                "attempts": attempts,
            }
            if pass_estimates:
                trace_entry["pass_at_k"] = {str(k): val for k, val in pass_estimates.items()}
            if args.trace_every > 0 and example_idx % args.trace_every == 0:
                traces.append(trace_entry)
            example_idx += 1

    # Save Result
    result: dict[str, Any] = {
        "em": sum_em / num_total,
        "f1": sum_f1 / num_total,
    }
    if pass_at_ks:
        result.update({"pass_at_k": {str(k): pass_at_k_score[k] / num_total for k in pass_at_ks}})
    if traces:
        result.update({"traces": traces})

    save_dir = get_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RAG Evaluation")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--peft_model_id", type=str, default=None)
    parser.add_argument("--enable_stateful", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8001")
    parser.add_argument("--generator_api_base", type=str, default="http://localhost:8003/v1")
    parser.add_argument("--generator_api_key", type=str, default="anything")
    parser.add_argument("--generator_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--generator_context_window", type=int, default=4096)
    parser.add_argument("--generator_enable_thinking", action="store_true")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333")
    parser.add_argument("--collection_name", type=str, default="wiki18-qwen3-embedding-0.6b")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--rag_method", type=str, default="react_agent",
                        choices=["react_agent", "multistep_workflow", "search_r1"])
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_generations", type=int, default=1,
                        help="Number of responses to sample per question for pass@k.")
    parser.add_argument("--pass_at_ks", type=int, nargs="*", default=None,
                        help="List of k values for pass@k calculation (each must be <= num_generations).",)
    parser.add_argument("--trace_every", type=int, default=0,
                        help="Save one trace every N examples (0 to disable trace saving).",)
    parser.add_argument("--similarity_top_k", type=int, default=3,
                        help="Number of documents to use when composing the answer.",)
    parser.add_argument("--sampling_pool_size", type=int, default=30,
                        help="Number of documents to sample from when retrieval sampling is enabled.",)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for retrieval sampling.",)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(main(args))
