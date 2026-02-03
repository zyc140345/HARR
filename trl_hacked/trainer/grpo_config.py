from dataclasses import dataclass, field
from typing import Optional
from trl.trainer.grpo_config import GRPOConfig


@dataclass
class RetrieverGRPOConfig(GRPOConfig):
    """
    Configuration class for RetrieverGRPOTrainer.

    Extends GRPOConfig with retriever-specific parameters for RAG training.
    """
    embedding_base_url: Optional[str] = field(
        default="http://localhost:8001",
        metadata={"help": "Base URL of the embedding model API."},
    )
    embedding_max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum input length for the embedding model."},
    )
    enable_stateful: bool = field(
        default=False,
        metadata={"help": "Enable stateful retrieval across iterations."},
    )
    generator_api_base: Optional[str] = field(
        default="http://localhost:8003/v1",
        metadata={"help": "Base URL of the generator model API."},
    )
    generator_api_key: Optional[str] = field(
        default="anything",
        metadata={"help": "API key for the generator model."},
    )
    generator_model: Optional[str] = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "Name of the generator model to use."},
    )
    generator_context_window: Optional[int] = field(
        default=4096,
        metadata={"help": "Context window size for the generator model."},
    )
    relevance_base_url: Optional[str] = field(
        default="http://localhost:8002",
        metadata={"help": "Base URL of the relevance reward model API."},
    )
    relevance_model: Optional[str] = field(
        default="Qwen/Qwen3-Reranker-4B",
        metadata={"help": "Model name for relevance reward scoring."},
    )
    relevance_instruction: Optional[str] = field(
        default="Given a web search query, retrieve relevant passages that answer the query",
        metadata={"help": "Instruction used by the relevance reward model."},
    )
    qdrant_url: Optional[str] = field(
        default="http://localhost:6333",
        metadata={"help": "URL of the Qdrant vector database."},
    )
    collection_name: Optional[str] = field(
        default="wiki18-qwen3-embedding-0.6b",
        metadata={"help": "Name of the Qdrant collection to use."},
    )
    similarity_top_k: Optional[int] = field(
        default=3,
        metadata={"help": "Number of most similar documents to retrieve."},
    )
    sampling_pool_size: Optional[int] = field(
        default=30,
        metadata={"help": "Number of documents to sample from."},
    )
    max_iterations: Optional[int] = field(
        default=5,
        metadata={"help": "Maximum number of iterations to retrieve documents."},
    )
    rag_method: str = field(
        default="react_agent",
        metadata={
            "help": (
                "Multi-hop RAG method to use for rollouts. "
                "Supported: react_agent, multistep_workflow, search_r1."
            )
        },
    )
    timeout: Optional[int] = field(
        default=60,
        metadata={"help": "Maximum seconds for workflow execution."},
    )
    reward_train_ranges: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "Per-reward activation ranges in terms of `Trainer.state.global_step` (curriculum learning). "
                "Must match the length/order of `reward_funcs`. Each element is `start:end` (0-based, end-exclusive). "
                "Use `start:` for [start, +inf), `:end` for [0, end), and `:` or `*` for always-on."
            )
        },
    )
    curriculum_stages: int = field(
        default=1,
        metadata={
            "help": (
                "Number of curriculum stages per dataloader pass (1 disables). "
                "If >1, sampling progresses from easier to harder samples within each dataloader pass."
            )
        },
    )
    curriculum_column: str = field(
        default="accuracy",
        metadata={
            "help": (
                "Dataset column used to rank difficulty for curriculum learning. "
                "Used together with curriculum_sort_order to decide the sampling order."
            )
        },
    )
    curriculum_sort_order: str = field(
        default="desc",
        metadata={
            "help": (
                "Sort order for `curriculum_column` to decide which samples are seen first. "
                "Use `desc`/`descending` if higher values are easier (sample larger values first). "
                "Use `asc`/`ascending` if lower values are easier (sample smaller values first)."
            )
        },
    )
