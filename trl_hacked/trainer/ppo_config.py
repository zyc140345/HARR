from dataclasses import field, dataclass
from typing import Optional
from trl.trainer.ppo_config import PPOConfig


@dataclass
class RetrieverPPOConfig(PPOConfig):
    embedding_base_url: Optional[str] = field(
        default="http://localhost:8001",
        metadata={"help": "Base URL of the embedding model API."},
    )
    embedding_max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum input length for the embedding model."},
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
    qdrant_url: Optional[str] = field(
        default="http://localhost:6333",
        metadata={"help": "URL of the Qdrant vector database."},
    )
    collection_name: Optional[str] = field(
        default="wiki18-qwen3-embedding-0.6b",
        metadata={"help": "Name of the Qdrant collection to use."},
    )
    ent_coef: Optional[float] = field(
        default=0,
        metadata={"help": "Coefficient for the entropy loss."},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["rouge"],
        metadata={
            "help": "List of reward functions. Possible values: 'rouge', 'em'"
        },
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
                    "rewards are weighted equally with weight `1.0`."
        },
    )
    similarity_top_k: Optional[int] = field(
        default=3,
        metadata={"help": "Number of most similar documents to retrieve."},
    )
    sampling_pool_size: Optional[int] = field(
        default=30,
        metadata={"help": "Number of documents to sample from."},
    )
    enable_sampling: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to enable sampling for document retrieval."},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "Sampling temperature."},
    )
    max_iterations: Optional[int] = field(
        default=5,
        metadata={"help": "Maximum number of iterations to retrieve documents."},
    )
    enable_stateful: bool = field(
        default=False,
        metadata={"help": "Enable stateful retrieval across iterations."},
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
        metadata={"help": "Maximum number of seconds to retrieve documents."},
    )
