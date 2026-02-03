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
import os
import datasets as ds

from typing import Any
from transformers import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from llama_index_hacked.model import get_remote_llm
from prompt import CHAT_RAW_EM_PROMPT
from metric.qa_em import compute_score_em


def get_save_dir(args):
    model = args.model.split("/")[-1].lower()
    dataset_name = args.dataset_name
    dir_name = f"{model}_{dataset_name}_raw"
    return f"{repo_root}/result/{dir_name}"


async def main(args):
    set_seed(args.seed)

    llm = get_remote_llm(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        context_window=args.context_window,
        enable_thinking=args.enable_thinking
    )

    # Load Test Dataset
    dataset = ds.load_dataset("RUC-NLPIR/FlashRAG_datasets", args.dataset_name, split="dev")
    if args.num_samples is not None:
        dataset = dataset.shuffle()
        dataset = dataset.select(range(args.num_samples))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x
    )

    # Perform Evaluation
    async def get_response(query):
        response = await llm.apredict(
            CHAT_RAW_EM_PROMPT, query_str=query
        )
        return response

    num_correct = 0
    num_total = 0
    traces = []
    for batch in tqdm(dataloader, desc="Testing RAG"):
        tasks = [get_response(example["question"]) for example in batch]
        responses = await asyncio.gather(*tasks)
        for response, example in zip(responses, batch):
            response_str = str(response)
            ground_truth = example['golden_answers']
            num_correct += compute_score_em(
                solution_str=response_str, ground_truth=ground_truth
            )
            num_total += 1
            traces.append({
                "question": example["question"],
                "response": response_str,
                "ground_truth": ground_truth,
            })

    # Save Result
    result: dict[str, Any] = {"em": num_correct / num_total * 100}
    if traces:
        result.update({"traces": traces})

    save_dir = get_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RAW LLM Evaluation")
    parser.add_argument("--api_base", type=str, default="http://localhost:8003/v1")
    parser.add_argument("--api_key", type=str, default="anything")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--context_window", type=int, default=4096)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(main(args))
