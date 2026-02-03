import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys
sys.path.append(repo_root)
from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import os
import datasets as ds
import evaluate
import numpy as np

from transformers import AutoTokenizer, set_seed
from datasets import Dataset
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
    f1 = float(out["f1"])
    return f1


def get_save_dirname(args):
    model_name = "-".join(args.collection_name.split("-")[1:])
    rag_method = args.rag_method.replace("_", "-")
    ablation = "ablation_" if not args.enable_stateful else ""
    return f"{model_name}_{rag_method}_{ablation}filtered"


async def main(args):
    set_seed(args.seed)

    # Setup Models
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
    query_engine.train()  # enable stochastic sampling only on retrieval

    # Perform Evaluation
    def format_result(response, sub_queries, cur_responses, source_nodes):
        return (
            response,
            [sub_query.query_str for sub_query in sub_queries],
            cur_responses,
            [[node.text for node in source_node[:args.similarity_top_k]] for source_node in source_nodes],
        )

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
                    mode="train",
                    args=args,
                )
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        results = [format_result(*result) for result in results]
        return results

    for dataset_name in args.dataset_names:
        # Load Train Dataset
        dataset = ds.load_dataset("RUC-NLPIR/FlashRAG_datasets", dataset_name, split="train")
        dataset = dataset.shuffle(seed=args.seed)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x
        )

        kept_examples = []
        target = args.num_samples
        if target is not None:
            with tqdm(total=target, desc=f"Filtering {dataset_name}", unit="ex") as pbar:
                for batch in dataloader:
                    # process the whole batch concurrently (batch_size * num_generations coroutines)
                    batch_tasks = [get_responses_for_example(example) for example in batch]
                    batch_results = await asyncio.gather(*batch_tasks)

                    for attempt_results, example in zip(batch_results, batch):
                        f1s = [
                            compute_metric(response, example["golden_answers"])
                            for response, sub_queries, cur_responses, source_nodes in attempt_results
                        ]
                        f1_mean = np.mean(f1s)
                        f1_std = np.std(f1s)
                        if f1_std > 0 and len(kept_examples) < target:
                            example["f1_mean"] = f1_mean
                            example["f1_std"] = f1_std
                            kept_examples.append(example)
                            pbar.update(1)

                        if len(kept_examples) >= target:
                            break

                    if len(kept_examples) >= target:
                        break
        else:
            for batch in tqdm(dataloader, desc=f"Filtering {dataset_name}"):
                # process the whole batch concurrently (batch_size * num_generations coroutines)
                batch_tasks = [get_responses_for_example(example) for example in batch]
                batch_results = await asyncio.gather(*batch_tasks)

                for attempt_results, example in zip(batch_results, batch):
                    f1s = [
                        compute_metric(response, example["golden_answers"])
                        for response, sub_queries, cur_responses, source_nodes in attempt_results
                    ]
                    f1_mean = np.mean(f1s)
                    f1_std = np.std(f1s)
                    if f1_std > 0:
                        example["f1_mean"] = f1_mean
                        example["f1_std"] = f1_std
                        kept_examples.append(example)

        if not kept_examples:
            raise ValueError(
                f"No samples remained after filtering for dataset={dataset_name!r}. "
                "Consider relaxing the criteria or increasing the pool."
            )

        dataset_filtered = Dataset.from_list(kept_examples)

        # Save Result
        output_dir = os.path.join(repo_root, "result", get_save_dirname(args), dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        dataset_filtered.to_parquet(os.path.join(output_dir, "train_filtered.parquet"))

        if args.push_to_hub:
            import huggingface_hub
            hf_username = huggingface_hub.whoami()["name"]
            hf_repo_id = f"{hf_username}/{get_save_dirname(args)}"
            ds_dict = ds.DatasetDict({"train": dataset_filtered})
            ds_dict.push_to_hub(hf_repo_id, config_name=dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filter Training Data")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8001")
    parser.add_argument("--generator_api_base", type=str, default="http://localhost:8003/v1")
    parser.add_argument("--generator_api_key", type=str, default="anything")
    parser.add_argument("--generator_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--generator_context_window", type=int, default=4096)
    parser.add_argument("--generator_enable_thinking", action="store_true")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333")
    parser.add_argument("--collection_name", type=str, default="wiki18-qwen3-embedding-0.6b")
    parser.add_argument("--dataset_names", type=str, nargs="+", default=["hotpotqa"])
    parser.add_argument("--enable_stateful", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--rag_method", type=str, default="react_agent",
                        choices=["react_agent", "multistep_workflow", "search_r1"])
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num_samples", type=int, default=None, help="Target number of filtered samples to keep")
    parser.add_argument("--num_generations", type=int, default=8, help="Number of retrieval runs per sample")
    parser.add_argument("--sampling_pool_size", type=int, default=30, help="Number of docs to sample from")
    parser.add_argument("--similarity_top_k", type=int, default=3, help="Top-k docs to feed to the generator")
    parser.add_argument("--temperature", type=float, default=0.05, help="Sampling temperature for retriever")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the filtered dataset to the Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(main(args))
