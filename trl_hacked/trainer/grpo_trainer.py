"""
RetrieverGRPOTrainer: GRPO-based trainer for RAG retriever fine-tuning.

This module adapts TRL's GRPOTrainer to fine-tune query encoders in a RAG pipeline
using Group Relative Policy Optimization. The key differences from the original GRPO:

1. **Policy**: Query encoder (embedding model) instead of LLM
2. **Action**: Retrieving k documents from vector database
3. **State**: Original question + reasoning history (sub-questions + retrieved summaries)
4. **Reward**: ROUGE-L/EM between generated answer and ground truth

The trainer samples G trajectories per query by leveraging retrieval stochasticity,
computes Plackett-Luce log-probabilities, and optimizes the encoder to align with
generator preferences.
"""

import asyncio
import os
import textwrap
import warnings
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union, Sequence

import datasets
import torch
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available, is_rich_available

from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import create_reference_model, prepare_deepspeed, prepare_fsdp
from trl.trainer.grpo_trainer import (
    RepeatSampler,
    nanmax,
    nanmin,
)
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    print_prompt_completions_sample,
)
from vllm import AsyncLLMEngine, AsyncEngineArgs

from llama_index.core import Settings
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient
from llama_index_hacked.model import Qwen3EmbeddingServer, Qwen3EmbeddingColocate, get_remote_llm
from llama_index_hacked.query_engine import RetrieverQueryEngine
from trl_hacked.trainer.grpo_config import RetrieverGRPOConfig
from trl_hacked.trainer.utils import (
    is_async_func,
    batch_generation,
    retrieve_embeddings,
    plackett_luce_logprob,
    forward,
)
from prompt import RETRIEVAL_INSTRUCTION, STATEFUL_RETRIEVAL_INSTRUCTION


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb


# Constants
INVALID_LOGPROB = 0.0
INVALID_LOGIT = -1e7
RewardFunc = Callable[..., Union[list[float], list[list[float]]]]


class CurriculumRepeatSampler(Sampler):
    """
    RepeatSampler variant that samples from an expanding prefix of a pre-sorted index list.

    Intended for curriculum learning: start from easiest samples and gradually include harder ones.
    """

    def __init__(
        self,
        data_source: Any,
        sorted_indices: list[int],
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int,
        shuffle: bool,
        seed: Optional[int],
        curriculum_stages: int,
    ):
        self.data_source = data_source
        self.sorted_indices = sorted_indices
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(sorted_indices)
        self.shuffle = shuffle
        self.seed = seed
        self.curriculum_stages = int(curriculum_stages)

        if self.shuffle:
            self.generator = torch.Generator()
            if self.seed is not None:
                self.generator.manual_seed(self.seed)

    def __iter__(self):
        total_chunks = self.num_samples // self.batch_size
        if total_chunks <= 0:
            return iter(())

        stages = max(1, self.curriculum_stages)

        chunks_per_stage = [total_chunks // stages] * stages
        for i in range(total_chunks % stages):
            chunks_per_stage[i] += 1

        indexes_flat: list[int] = []
        for stage_idx, n_chunks in enumerate(chunks_per_stage):
            if n_chunks <= 0:
                continue

            frac = min(1.0, float(stage_idx + 1) / float(stages))
            pool_size = max(self.batch_size, int(round(frac * self.num_samples)))
            pool = self.sorted_indices[:pool_size]
            if not pool:
                continue

            target_len = n_chunks * self.batch_size
            stage_indexes: list[int] = []
            while len(stage_indexes) < target_len:
                if self.shuffle:
                    perm = torch.randperm(len(pool), generator=self.generator).tolist()
                    stage_indexes.extend(pool[i] for i in perm)
                else:
                    stage_indexes.extend(pool)
            indexes_flat.extend(stage_indexes[:target_len])

        indexes_flat = indexes_flat[: total_chunks * self.batch_size]
        indexes = [
            indexes_flat[i : i + self.batch_size]
            for i in range(0, len(indexes_flat), self.batch_size)
        ]

        def generator():
            for chunk in indexes:
                for _ in range(self.repeat_count):
                    for index in chunk:
                        for _ in range(self.mini_repeat_count):
                            yield index

        return generator()

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


def split_tensor_dict(
    tensor_dict: dict[str, Optional[Union[torch.Tensor, dict[str, torch.Tensor]]]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.
    """
    split_points = (tensor_dict["sequence_lengths"] + 1).tolist()
    policy_inputs = {
        key: tensor.split(split_points)
        for key, tensor in tensor_dict['policy_inputs'].items()
    }

    batch_size = len(split_points)
    chunk_size = batch_size // num_chunks

    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items() if key != 'policy_inputs'
        } | {
            'policy_inputs': {
                key: torch.cat(tensor[i * chunk_size : (i + 1) * chunk_size])
                for key, tensor in policy_inputs.items()
            }
        }
        for i in range(num_chunks)
    ]


def shuffle_tensor_dict(
    tensor_dict: dict[str, Optional[Union[torch.Tensor, dict[str, torch.Tensor]]]]
) -> dict[str, Optional[torch.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.
    """
    split_points = (tensor_dict["sequence_lengths"] + 1).tolist()
    policy_inputs = {
        key: tensor.split(split_points)
        for key, tensor in tensor_dict["policy_inputs"].items()
    }

    batch_size = len(split_points)
    permutation = torch.randperm(batch_size)

    return {
        key: tensor[permutation] if tensor is not None else None
        for key, tensor in tensor_dict.items() if key != 'policy_inputs'
    } | {
        'policy_inputs': {
            key: torch.cat([tensor[i] for i in permutation])
            for key, tensor in policy_inputs.items()
        }
    }


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor, dim: Optional[Union[int, Sequence[int]]] = None) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs.

    Args:
        tensor (`torch.Tensor`):
            Input tensor.
        dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
            If ``None``, all dimensions are reduced.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    if dim is None:
        dim = tuple(range(tensor.ndim))
    variance = torch.nanmean((tensor - torch.nanmean(tensor, dim=dim, keepdim=True)) ** 2, dim=dim)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor), dim=dim)  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


class RetrieverGRPOTrainer(Trainer):
    """
    Trainer for Group Relative Policy Optimization (GRPO) adapted for RAG retriever fine-tuning.

    This trainer fine-tunes a query encoder to produce embeddings that lead to better
    retrieval results, as measured by the quality of downstream generated answers.

    Key components:
    - Policy model: Query encoder (e.g., Qwen3-Embedding)
    - Reference model: Frozen copy or PEFT-disabled version of the policy
    - Rollout: Multi-step RAG workflow with stochastic document sampling
    - Reward: ROUGE-L or Exact Match between generated and ground truth answers

    Example:
        ```python
        trainer = RetrieverGRPOTrainer(
            model="Qwen/Qwen3-Embedding-0.6B",
            reward_funcs=rouge_reward,
            train_dataset=dataset,
            args=RetrieverGRPOConfig(output_dir="output"),
        )
        trainer.train()
        ```
    """

    _tag_names = ["trl", "grpo", "retriever"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[RetrieverGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = RetrieverGRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `RetrieverGRPOConfig`. Expected either 'auto' or a string "
                    f"representing a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModel.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `RetrieverGRPOConfig`, but your model is already "
                    "instantiated. This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward scheduling (by training step)
        if args.reward_train_ranges is not None:
            if len(args.reward_train_ranges) != len(self.reward_funcs):
                raise ValueError(
                    f"Number of reward train ranges ({len(args.reward_train_ranges)}) must match number of reward "
                    f"functions ({len(self.reward_funcs)})"
                )
            self.reward_train_ranges = [self._parse_reward_train_range(spec) for spec in args.reward_train_ranges]
        else:
            self.reward_train_ranges = [(None, None) for _ in range(len(self.reward_funcs))]

        # Reward kwargs
        self.reward_extra_kwargs = {
            "similarity_top_k": args.similarity_top_k,
            "relevance_base_url": args.relevance_base_url,
            "relevance_model": args.relevance_model,
            "relevance_instruction": args.relevance_instruction,
        }

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.similarity_top_k = args.similarity_top_k
        self.sampling_pool_size = args.sampling_pool_size
        self.max_iterations = args.max_iterations
        self.timeout = args.timeout
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset
        self.curriculum_stages = args.curriculum_stages
        self.curriculum_column = getattr(args, "curriculum_column", "accuracy")
        self.curriculum_sort_order = str(getattr(args, "curriculum_sort_order", "desc")).strip().lower()

        if self.curriculum_sort_order in {"desc", "descending"}:
            curriculum_reverse = True
        elif self.curriculum_sort_order in {"asc", "ascending"}:
            curriculum_reverse = False
        else:
            raise ValueError(
                f"Invalid curriculum_sort_order={self.curriculum_sort_order!r}. "
                "Expected one of: 'asc', 'desc', 'ascending', 'descending'."
            )
        self._curriculum_sorted_indices: Optional[list[int]] = None

        if self.curriculum_stages > 1 and train_dataset is not None and isinstance(train_dataset, datasets.Dataset):
            if self.curriculum_column not in train_dataset.column_names:
                warnings.warn(
                    f"curriculum_stages>1 but train_dataset has no `{self.curriculum_column}` column; disabling curriculum."
                )
            else:
                values = train_dataset[self.curriculum_column]

                def key_fn(i: int) -> float:
                    v = values[i]
                    try:
                        return float(v)
                    except Exception:
                        return float("-inf")

                # Sort so that earlier indices are easier and sampled first.
                self._curriculum_sorted_indices = sorted(
                    range(len(train_dataset)),
                    key=key_fn,
                    reverse=curriculum_reverse,
                )

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "query". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled() or self.is_fsdp_enabled:
            self.ref_model = AutoModel.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "query": deque(maxlen=maxlen),
            "response": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "rewards_raw": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Setup RAG components (async event loop and vector store)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        qdrant_client = AsyncQdrantClient(url=args.qdrant_url, prefer_grpc=True, timeout=self.timeout)
        self.vector_store = QdrantVectorStore(
            aclient=qdrant_client,
            collection_name=args.collection_name,
            timeout=self.timeout
        )

        # Setup query engine
        Settings.llm = get_remote_llm(
            api_base=args.generator_api_base,
            api_key=args.generator_api_key,
            model=args.generator_model,
            context_window=args.generator_context_window,
            seed=args.seed,
        )
        if args.rag_method == "search_r1":
            self.search_r1_processing_class = AutoTokenizer.from_pretrained(
                "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
            )
        else:
            self.search_r1_processing_class = None

        if self.vllm_mode == "server":
            if self.accelerator.is_main_process:
                Settings.embed_model = Qwen3EmbeddingServer(
                    base_url=args.embedding_base_url,
                    instruction=STATEFUL_RETRIEVAL_INSTRUCTION if args.enable_stateful else RETRIEVAL_INSTRUCTION,
                    init_communicator=True
                )
                self.vllm_client = Settings.embed_model.client
                self.query_prompt = [Settings.embed_model.query_prompt]
                self.instruction = [Settings.embed_model.instruction]

                index = VectorStoreIndex.from_vector_store(self.vector_store)
                self.retriever = index.as_retriever(similarity_top_k=self.sampling_pool_size)
                self.query_engine = RetrieverQueryEngine.from_args(
                    self.retriever,
                    similarity_top_k=self.similarity_top_k,
                    temperature=self.temperature,
                )
            else:
                self.query_prompt = [None]
                self.instruction = [None]

            # Broadcast config from main process
            self.query_prompt = broadcast_object_list(self.query_prompt)[0]
            self.instruction = broadcast_object_list(self.instruction)[0]

        elif self.vllm_mode == "colocate":
            if self.vllm_tensor_parallel_size > 1:
                raise ValueError("vLLM colocation mode does not support tensor parallelism.")

            engine_args = AsyncEngineArgs(
                model=model.name_or_path,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                max_num_seqs=self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps,
                max_model_len=self.args.embedding_max_length,
                distributed_executor_backend="external_launcher",
                seed=self.accelerator.process_index
            )
            self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            Settings.embed_model = Qwen3EmbeddingColocate(
                llm_engine=self.llm_engine,
                loop=self.loop,
                instruction=STATEFUL_RETRIEVAL_INSTRUCTION if args.enable_stateful else RETRIEVAL_INSTRUCTION,
            )
            self.query_prompt = Settings.embed_model.query_prompt
            self.instruction = Settings.embed_model.instruction

            index = VectorStoreIndex.from_vector_store(self.vector_store)
            self.retriever = index.as_retriever(similarity_top_k=self.sampling_pool_size)
            self.query_engine = RetrieverQueryEngine.from_args(
                self.retriever,
                similarity_top_k=self.similarity_top_k,
                temperature=self.temperature,
            )

        self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    @staticmethod
    def _parse_reward_train_range(spec: str) -> tuple[Optional[int], Optional[int]]:
        spec = str(spec).strip()
        if spec in {"", "*", ":"}:
            return None, None
        if ":" not in spec:
            raise ValueError(
                f"Invalid reward_train_ranges entry {spec!r}. Expected `start:end`, `start:`, `:end`, `:`."
            )
        start_str, end_str = (part.strip() for part in spec.split(":", 1))
        start = int(start_str) if start_str else None
        end = int(end_str) if end_str else None
        if start is not None and start < 0:
            raise ValueError(f"Invalid reward_train_ranges entry {spec!r}. `start` must be >= 0.")
        if end is not None and end < 0:
            raise ValueError(f"Invalid reward_train_ranges entry {spec!r}. `end` must be >= 0.")
        if start is not None and end is not None and start >= end:
            raise ValueError(f"Invalid reward_train_ranges entry {spec!r}. Expected `start < end`.")
        return start, end

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In RetrieverGRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["query"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset

        if (
            self._curriculum_sorted_indices is not None
            and self.curriculum_stages > 1
        ):
            return CurriculumRepeatSampler(
                data_source=dataset,
                sorted_indices=self._curriculum_sorted_indices,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
                curriculum_stages=self.curriculum_stages,
            )
        else:
            return RepeatSampler(
                data_source=dataset,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: RetrieverGRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
            else:
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm_engine.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_tensor_dict(generation_batch)
                self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Generate RAG completions and compute rewards/advantages.

        This is the core rollout function that:
        1. Gathers all queries across processes
        2. Runs multi-step RAG workflow with stochastic retrieval (G times per query)
        3. Computes rewards based on answer quality
        4. Computes Plackett-Luce log-probabilities for document selections
        5. Normalizes rewards within groups to compute advantages
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Extract queries from inputs
        queries = [x["query"] for x in inputs]

        # Sync model weights to vLLM before generation
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Generate responses using vLLM: gather all queries and use them in a single call in the main process
        if self.vllm_mode == "server":
            all_queries = gather_object(queries)
            if self.accelerator.is_main_process:
                if mode == "train":
                    self.query_engine.train()  # Enable stochastic retrieval during training
                else:
                    self.query_engine.eval()
                with profiling_context(self, "batch_generation"):
                    responses, sub_queries, cur_responses, source_nodes = batch_generation(
                        self.query_engine,
                        self.retriever,
                        all_queries,
                        mode,
                        self.search_r1_processing_class,
                        self.args,
                        self.loop
                    )
            else:
                responses = [None] * len(all_queries)
                sub_queries = [None] * len(all_queries)
                cur_responses = [None] * len(all_queries)
                source_nodes = [None] * len(all_queries)

            # Broadcast results to all processes
            responses = broadcast_object_list(responses, from_process=0)
            sub_queries = broadcast_object_list(sub_queries, from_process=0)
            cur_responses = broadcast_object_list(cur_responses, from_process=0)
            source_nodes = broadcast_object_list(source_nodes, from_process=0)

            # Slice to keep only the local part of the data
            process_slice = slice(
                self.accelerator.process_index * len(queries),
                (self.accelerator.process_index + 1) * len(queries),
            )
            responses = responses[process_slice]
            sub_queries = sub_queries[process_slice]
            cur_responses = cur_responses[process_slice]
            source_nodes = source_nodes[process_slice]

        # Generate responses using colocated vLLM instances: each device holds vLLM copy and work on their own batch of queries
        elif self.vllm_mode == "colocate":
            if mode == "train":
                self.query_engine.train()  # Enable stochastic retrieval during training
            else:
                self.query_engine.eval()
            with profiling_context(self, "batch_generation"):
                responses, sub_queries, cur_responses, source_nodes = batch_generation(
                    self.query_engine,
                    self.retriever,
                    queries,
                    mode,
                    self.search_r1_processing_class,
                    self.args,
                    self.loop
                )

        # Compute sequence lengths and prepare policy inputs
        sequence_lengths = torch.tensor([len(seq) for seq in sub_queries], device=device) - 1
        split_points = (sequence_lengths + 1).tolist()
        all_sequence_lengths = gather(sequence_lengths)
        max_steps = int(all_sequence_lengths.max().item()) + 1

        # Tokenize sub-queries for policy forward pass
        policy_inputs = self.processing_class(
            [self.query_prompt.format(self.instruction, step.custom_embedding_strs[0])
             for seq in sub_queries for step in seq],
            padding=True,
            truncation=True,
            max_length=self.args.embedding_max_length,
            return_tensors="pt",
        )
        policy_inputs = {k: v.to(device) for k, v in policy_inputs.items()}

        # Forward pass through policy model to get query embeddings
        with torch.no_grad():
            # Get document embeddings from vector store
            doc_embeds = pad_sequence([
                torch.tensor([
                    retrieve_embeddings(self.vector_store, step, self.loop, self.args)
                    for step in seq
                ], device=device)
                for seq in source_nodes
            ], batch_first=True)  # (batch_size, max_steps, num_nodes, embedding_size)

            # Compute Plackett-Luce log-probabilities
            logits = pad_sequence([
                torch.tensor([
                    [node.score for node in step] + [INVALID_LOGIT] * (self.args.sampling_pool_size - len(step))
                    for step in seq
                ], device=device)
                for seq in source_nodes
            ], batch_first=True, padding_value=INVALID_LOGIT)
            logits = logits / (self.temperature + 1e-7)
            doc_index = torch.arange(self.similarity_top_k, device=device).view(1, 1, -1).expand(*logits.shape[:-1], -1)
            logprobs = plackett_luce_logprob(logits, doc_index)  # (batch_size, max_steps)

            # Compute reference log-probabilities
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_query_embed = forward(self.ref_model, policy_inputs, with_wrapper=False)
                else:
                    # Use PEFT adapter disable
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_query_embed = forward(self.model, policy_inputs, with_wrapper=False)
                ref_query_embed = pad_sequence(
                    ref_query_embed.split(split_points),
                    batch_first=True
                ).to(doc_embeds)
                ref_logits = torch.matmul(doc_embeds, ref_query_embed.unsqueeze(-1)).squeeze(-1)
                ref_logits = ref_logits.masked_fill((doc_embeds.abs().sum(dim=-1) == 0), INVALID_LOGIT)
                ref_logits = ref_logits / (self.temperature + 1e-7)
                ref_logprobs = plackett_luce_logprob(ref_logits, doc_index)
            else:
                ref_logprobs = None

        # Create padding mask for variable-length sequences
        sub_query_idxs = torch.arange(logprobs.shape[1], device=device).repeat(logprobs.shape[0], 1)
        padding_mask = sub_query_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)

        # Compute rewards
        rewards_per_func = torch.full(
            (len(queries), max_steps, len(self.reward_funcs)),
            torch.nan,
            device=device
        )
        outcome_reward_only = True

        # Repeat all input columns (but "query") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["query"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs_with_extra = {**reward_kwargs, **self.reward_extra_kwargs}

        for i, (reward_func, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_func_names)
        ):
            start_step, end_step = self.reward_train_ranges[i]
            cur_step = int(self.state.global_step)
            if (start_step is not None and cur_step < start_step) or (end_step is not None and cur_step >= end_step):
                continue

            all_reward_args = dict(
                queries=queries,
                responses=responses,
                sub_queries=sub_queries,
                cur_responses=cur_responses,
                source_nodes=source_nodes,
                **reward_kwargs_with_extra,
            )
            if is_async_func(reward_func):
                output_reward_func = self.loop.run_until_complete(reward_func(**all_reward_args))
            else:
                output_reward_func = reward_func(**all_reward_args)

            if (
                len(output_reward_func) > 0
                and isinstance(output_reward_func[0], (list, tuple))
            ):  # process reward
                # Convert None values to NaN
                output_reward_func = [
                    [reward if reward is not None else torch.nan for reward in rewards]
                    for rewards in output_reward_func
                ]

                for b, rewards in enumerate(output_reward_func):
                    rewards_per_func[b, :len(rewards), i] = torch.tensor(rewards, dtype=torch.float32, device=device)

                outcome_reward_only = False
            else:  # outcome reward
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                actual_start = torch.arange(rewards_per_func.size(0), device=rewards_per_func.device)
                actual_end = sequence_lengths
                rewards_per_func[actual_start, actual_end, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        # If all reward functions return None for a given example, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=(1, 2)).any():
            nan_example_idx = torch.isnan(rewards_per_func).all(dim=(1, 2)).nonzero(as_tuple=True)[0][0]
            example_reward_kwargs = {key: value[nan_example_idx] for key, value in reward_kwargs.items()}
            example_reward_kwargs["query"] = queries[nan_example_idx]
            example_reward_kwargs["response"] = responses[nan_example_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {example_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        if outcome_reward_only:
            rewards_per_func = rewards_per_func.nansum(dim=1, keepdim=True)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        rewards_per_func_raw = rewards_per_func.clone()

        reward_steps = rewards_per_func.shape[1]
        all_sub_query_idxs = torch.arange(reward_steps, device=device).repeat(all_sequence_lengths.shape[0], 1)
        all_padding_mask = all_sub_query_idxs > all_sequence_lengths.unsqueeze(1)

        if rewards_per_func.shape[-1] > 1:
            # Compute grouped-wise rewards per function
            mean_grouped_rewards_per_func = rewards_per_func.view(
                -1, self.num_generations, reward_steps, rewards_per_func.shape[-1]
            ).nanmean(dim=(1, 2))
            std_grouped_rewards_per_func = nanstd(rewards_per_func.view(
                -1, self.num_generations, reward_steps, rewards_per_func.shape[-1]
            ), dim=(1, 2))

            # Normalize the rewards per function
            mean_grouped_rewards_per_func = mean_grouped_rewards_per_func.repeat_interleave(
                self.num_generations, dim=0
            ).view(-1, 1, rewards_per_func.shape[-1])
            std_grouped_rewards_per_func = std_grouped_rewards_per_func.repeat_interleave(
                self.num_generations, dim=0
            ).view(-1, 1, rewards_per_func.shape[-1])
            rewards_per_func = rewards_per_func - mean_grouped_rewards_per_func
            if self.scale_rewards:
                rewards_per_func = rewards_per_func / (std_grouped_rewards_per_func + 1e-4)

        # Apply weights to each reward function's output and sum
        rewards_raw = (rewards_per_func_raw * self.reward_weights.to(device).view(1, 1, -1)).nansum(dim=-1)
        rewards_raw = torch.where(all_padding_mask, torch.nan, rewards_raw)
        rewards = (rewards_per_func * self.reward_weights.to(device).view(1, 1, -1)).nansum(dim=-1)
        rewards = torch.where(all_padding_mask, torch.nan, rewards)

        # Compute grouped-wise rewards
        mean_grouped_rewards_raw = rewards_raw.view(-1, self.num_generations, reward_steps).nanmean(dim=(1, 2))
        std_grouped_rewards_raw = nanstd(rewards_raw.view(-1, self.num_generations, reward_steps), dim=(1, 2))
        mean_grouped_rewards = rewards.view(-1, self.num_generations, reward_steps).nanmean(dim=(1, 2))
        std_grouped_rewards = nanstd(rewards.view(-1, self.num_generations, reward_steps), dim=(1, 2))
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards_raw = mean_grouped_rewards_raw.repeat_interleave(self.num_generations, dim=0).unsqueeze(1)
        std_grouped_rewards_raw = std_grouped_rewards_raw.repeat_interleave(self.num_generations, dim=0).unsqueeze(1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0).unsqueeze(1)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0).unsqueeze(1)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        advantages = torch.where(all_padding_mask, torch.zeros_like(advantages), advantages)
        advantages = advantages.flip([1]).cumsum(dim=1).flip([1])

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(queries),
            (self.accelerator.process_index + 1) * len(queries),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(sequence_lengths.float().mean().item() + 1)
        self._metrics[mode]["completions/min_length"].append(sequence_lengths.float().min().item() + 1)
        self._metrics[mode]["completions/max_length"].append(sequence_lengths.float().max().item() + 1)

        # Log mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            if torch.isnan(rewards_per_func[:, :, i]).all():
                continue
            mean_rewards = torch.nanmean(rewards_per_func[:, :, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, :, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
            mean_rewards_raw = torch.nanmean(rewards_per_func_raw[:, :, i]).item()
            self._metrics[mode][f"rewards_raw/{reward_func_name}/mean"].append(mean_rewards_raw)
            std_rewards_raw = nanstd(rewards_per_func_raw[:, :, i]).item()
            self._metrics[mode][f"rewards_raw/{reward_func_name}/std"].append(std_rewards_raw)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["reward_raw"].append(mean_grouped_rewards_raw.mean().item())
        self._metrics[mode]["reward_raw_std"].append(std_grouped_rewards_raw.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log query and response texts
        queries_text = [x["query"] for x in inputs]
        self._textual_logs["query"].extend(gather_object(queries_text))
        self._textual_logs["response"].extend(gather_object(responses))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, :, i].nansum(1).tolist())
            self._textual_logs["rewards_raw"][name].extend(rewards_per_func_raw[:, :, i].nansum(1).tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.nansum(1).tolist())

        return {
            "policy_inputs": policy_inputs,
            "advantages": advantages,
            "sequence_lengths": sequence_lengths,
            "doc_embeds": doc_embeds,
            "padding_mask": padding_mask,
            "old_logprobs": logprobs,
            "ref_logprobs": ref_logprobs,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("RetrieverGRPOTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        """Compute GRPO loss for retriever training."""
        device = self.accelerator.device

        policy_inputs = inputs["policy_inputs"]
        split_points = (inputs["sequence_lengths"] + 1).tolist()
        truc_point = inputs["sequence_lengths"].max().item() + 1  # to match the shape of `query_embed`
        advantages = inputs["advantages"][:, :truc_point]
        doc_embeds = inputs["doc_embeds"][:, :truc_point]
        padding_mask = inputs["padding_mask"][:, :truc_point]
        old_logprobs = inputs["old_logprobs"][:, :truc_point]
        ref_logprobs = inputs["ref_logprobs"][:, :truc_point] if inputs["ref_logprobs"] is not None else None

        # Forward pass through current policy
        query_embed = forward(model, policy_inputs, with_wrapper=False)
        query_embed = pad_sequence(
            query_embed.split(split_points),
            batch_first=True
        ).to(doc_embeds)

        # Compute logits and log-probabilities
        logits = torch.matmul(doc_embeds, query_embed.unsqueeze(-1)).squeeze(-1)
        logits = logits.masked_fill((doc_embeds.abs().sum(dim=-1) == 0), INVALID_LOGIT)
        logits = logits / (self.temperature + 1e-7)

        doc_index = torch.arange(self.similarity_top_k, device=device).view(1, 1, -1).expand(*logits.shape[:-1], -1)
        per_step_logps = plackett_luce_logprob(logits, doc_index)
        per_step_logps = torch.masked_fill(per_step_logps, padding_mask, INVALID_LOGPROB)

        # Compute KL divergence if beta > 0
        if self.beta != 0.0:
            per_step_kl = (
                torch.exp(ref_logprobs - per_step_logps) - (ref_logprobs - per_step_logps) - 1
            )
            per_step_kl = torch.masked_fill(per_step_kl, padding_mask, 0)

        # Compute policy ratio
        logprobs_diff = per_step_logps - old_logprobs
        coef_1 = torch.exp(logprobs_diff)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        # Compute per-step loss (sum over steps, then average over batch)
        per_step_loss1 = coef_1 * advantages
        per_step_loss2 = coef_2 * advantages
        per_step_loss = -torch.min(per_step_loss1, per_step_loss2)

        if self.beta != 0.0:
            per_step_loss = per_step_loss + self.beta * per_step_kl

        # Aggregate loss based on loss_type
        valid_mask = ~padding_mask
        if self.loss_type == "grpo":
            # Normalize by sequence length
            loss = ((per_step_loss * valid_mask).sum(-1) / valid_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            # Normalize by total valid tokens in batch
            loss = (per_step_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            # Normalize by constant (max possible length)
            max_len = per_step_loss.size(1)
            loss = (per_step_loss * valid_mask).sum() / (per_step_loss.size(0) * max_len)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_step_kl * valid_mask).sum() / valid_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * valid_mask).sum() / valid_mask.sum()
        high_clip = (is_high_clipped * valid_mask).sum() / valid_mask.sum()
        clip_ratio = (is_region_clipped * valid_mask).sum() / valid_mask.sum()

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        # Log ratio statistics
        ratio_mean = (coef_1 * valid_mask).sum() / valid_mask.sum()
        gathered_ratio_mean = self.accelerator.gather(ratio_mean)
        self._metrics[mode]["ratio/mean"].append(gathered_ratio_mean.nanmean().item())

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["query"],
                    self._textual_logs["response"],
                    self._textual_logs["rewards"],
                    self._textual_logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["query"]),
                    "query": list(self._textual_logs["query"]),
                    "response": list(self._textual_logs["response"]),
                    **{k: list(v) for k, v in self._textual_logs["rewards"].items()},
                    "advantage": list(self._textual_logs["advantages"]),
                }
                df = pd.DataFrame(table)
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="RetrieverGRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
