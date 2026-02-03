import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys
sys.path.append(repo_root)
from dotenv import load_dotenv
load_dotenv()

import datasets
import argparse
import time

from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
from tqdm import tqdm
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionStatus,
    Distance,
    VectorParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    HnswConfigDiff,
    OptimizersConfigDiff,
)
from openai import OpenAI


def main(args):
    corpus = datasets.load_dataset('json', data_files=args.corpus_path, split="train", num_proc=args.num_workers)
    corpus = corpus.map(lambda x: {"contents": [f"passage: {content}" for content in x["contents"]]}, batched=True)
    dataloader = DataLoader(corpus, batch_size=args.batch_size, shuffle=False)

    openai_client = OpenAI(
        base_url=args.embedding_base_url,
        api_key="anything",
    )
    model_id = openai_client.models.list().data[0].id

    qdrant_client = QdrantClient(url=args.qdrant_url, prefer_grpc=True, timeout=60)
    if qdrant_client.collection_exists(args.collection_name):
        qdrant_client.delete_collection(args.collection_name)
    qdrant_client.create_collection(
        collection_name=args.collection_name,
        shard_number=args.shard_number,
        vectors_config=VectorParams(size=args.embedding_size, distance=Distance.DOT, on_disk=True),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=True,
            ),
        ),
        hnsw_config=HnswConfigDiff(m=0),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
    )
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=args.collection_name,
        batch_size=args.batch_size
    )

    with ThreadPoolExecutor(max_workers=args.num_workers) as upload_executor:
        futures = []

        for batch in tqdm(dataloader, desc="Computing embeddings"):
            texts = batch["contents"]
            response = openai_client.embeddings.create(
                input=texts,
                model=model_id,
            )
            embeddings = [data.embedding for data in response.data]

            nodes = [
                TextNode(text=text, embedding=embedding)
                for text, embedding in zip(texts, embeddings)
            ]
            future = upload_executor.submit(vector_store.add, nodes)
            futures.append(future)

            if len(futures) > args.num_workers:
                completed_future = futures.pop(0)
                completed_future.result()

        for f in futures:
            f.result()

    print("Building index...")
    qdrant_client.update_collection(
        collection_name=args.collection_name,
        hnsw_config=HnswConfigDiff(m=16),
        optimizer_config=OptimizersConfigDiff(indexing_threshold=20000),
    )

    while True:
        info = qdrant_client.get_collection(args.collection_name)
        if info.status == CollectionStatus.GREEN:
            break
        time.sleep(1)

    print("Index built successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--embedding_size", type=int, default=1024)
    parser.add_argument("--corpus_path", type=str, default=f"{repo_root}/data/wiki18_100w.jsonl")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333")
    parser.add_argument("--collection_name", type=str, default="wiki18-qwen3-embedding-0.6b")
    parser.add_argument("--shard_number", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    main(args)
