# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Optional, Union, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object, broadcast_object_list, GradientAccumulationPlugin
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available, is_rich_available

from trl.core import masked_mean, masked_whiten
from trl.models import create_reference_model
from trl.trainer.utils import (
    OnlineTrainerState,
    disable_dropout_in_model,
    empty_cache,
    exact_div,
    generate_model_card,
    get_comet_experiment_url,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from llama_index.core import Settings
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient
from llama_index_hacked.model import Qwen3EmbeddingServer, get_remote_llm
from llama_index_hacked.query_engine import RetrieverQueryEngine
from trl_hacked.trainer.ppo_config import RetrieverPPOConfig
from trl_hacked.trainer.utils import (
    batch_generation,
    retrieve_embeddings,
    plackett_luce_logprob,
    forward,
    get_value,
    get_critic_input,
    compute_critic_metrics,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb
INVALID_LOGPROB = 0.0
INVALID_LOGIT = -1e7
RewardFunc = Callable[[list, list], list[float]]


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, policy_input, critic_input=None):
        if critic_input:
            output = self.critic_backbone(**critic_input)
            values = self.value_model.score(output.last_hidden_state)
            return self.policy(**policy_input), values
        else:
            return self.policy(**policy_input)


class RetrieverPPOTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        args: RetrieverPPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        model: nn.Module,
        ref_model: Optional[nn.Module],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        train_dataset: Dataset,
        value_model: nn.Module,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif args.stop_token:
            if args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        elif self.policy_model.generation_config is not None:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int

        # Check that the kl estimator is valid
        if self.args.kl_estimator not in {"k1", "k3"}:
            raise ValueError(
                "kl_estimator must be either 'k1' (straightforward, unbiased) or 'k3' (lower variance, unbiased, "
                "appears to be a strictly better estimator). See "
                "[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) for details."
            )

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.policy_model)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
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

        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        grad_plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_with_dataloader=False,
        )
        accelerator = Accelerator(gradient_accumulation_plugin=grad_plugin)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert args.local_mini_batch_size >= 8, (
                f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
            )
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.ref_model, self.value_model]:
            if module is not None:
                disable_dropout_in_model(module)
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        # setup query engine
        #########
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        qdrant_client = AsyncQdrantClient(url=args.qdrant_url, prefer_grpc=True, timeout=args.timeout)
        self.vector_store = QdrantVectorStore(
            aclient=qdrant_client,
            collection_name=args.collection_name,
            timeout=args.timeout
        )

        self._last_loaded_step = -1

        if self.accelerator.is_main_process:
            Settings.embed_model = Qwen3EmbeddingServer(
                base_url=args.embedding_base_url,
                init_communicator=True
            )
            Settings.llm = get_remote_llm(
                api_base=args.generator_api_base,
                api_key=args.generator_api_key,
                model=args.generator_model,
                context_window=args.generator_context_window
            )
            if args.rag_method == "search_r1":
                self.search_r1_processing_class = AutoTokenizer.from_pretrained(
                    "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
                )
            else:
                self.search_r1_processing_class = None
            self.vllm_client = Settings.embed_model.client
            self.query_prompt = [Settings.embed_model.query_prompt]
            self.instruction = [Settings.embed_model.instruction]

            index = VectorStoreIndex.from_vector_store(self.vector_store)

            self.retriever = index.as_retriever(similarity_top_k=args.sampling_pool_size)
            self.query_engine = RetrieverQueryEngine.from_args(
                self.retriever,
                similarity_top_k=args.similarity_top_k,
                temperature=args.temperature,
            )
        else:
            self.query_prompt = [None]
            self.instruction = [None]
        self.query_prompt = broadcast_object_list(self.query_prompt)
        self.instruction = broadcast_object_list(self.instruction)
        self.query_prompt = self.query_prompt[0]
        self.instruction = self.instruction[0]

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        model = self.accelerator.unwrap_model(self.model)
        with (
            model.policy.disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                model.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                model.policy.set_adapter(self.model_adapter_name or "default")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if not hasattr(unwrapped_model, "policy"):  # avoid re-entry
            return
        self.model = unwrapped_model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

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

                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)

    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        model = self.accelerator.unwrap_model(self.model)
        if self.is_peft_model:
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(model.policy.parameters())):
                model.policy.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(model.policy)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in model.policy.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if model.policy.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                # Unmerge adapters while parameters are still gathered
                model.policy.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_vllm(model.policy)  # use memory-efficient post-order traversal for FSDP
            else:
                for name, param in model.policy.named_parameters():
                    with gather_if_zero3([param]):
                        if self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)

        # Reset cache on vLLM
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_funcs = self.reward_funcs
        reward_weights = self.reward_weights
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        seq_len_stats = torch.zeros(stats_shape, device=device)
        critic_baseline_mse_stats = torch.zeros(stats_shape, device=device)
        critic_value_mse_stats = torch.zeros(stats_shape, device=device)
        critic_corr_stats = torch.zeros(stats_shape, device=device)
        critic_r2_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["query"]
                doc_embeds = []
                logprobs = []
                ref_logprobs = []
                scores = []
                values = []

                all_queries = gather_object(queries)
                if self.accelerator.is_main_process:
                    self.query_engine.train() if args.enable_sampling else self.query_engine.eval()
                    responses, sub_queries, cur_responses, source_nodes = batch_generation(
                        self.query_engine,
                        self.retriever,
                        all_queries,
                        "train" if args.enable_sampling else "eval",
                        self.search_r1_processing_class,
                        args,
                        self.loop,
                    )
                else:
                    responses = [None] * len(all_queries)
                    sub_queries = [None] * len(all_queries)
                    cur_responses = [None] * len(all_queries)
                    source_nodes = [None] * len(all_queries)
                responses = broadcast_object_list(responses, from_process=0)
                sub_queries = broadcast_object_list(sub_queries, from_process=0)
                cur_responses = broadcast_object_list(cur_responses, from_process=0)
                source_nodes = broadcast_object_list(source_nodes, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(queries),
                    (self.accelerator.process_index + 1) * len(queries),
                )
                responses = responses[process_slice]
                sub_queries = sub_queries[process_slice]
                cur_responses = cur_responses[process_slice]
                source_nodes = source_nodes[process_slice]

                sequence_lengths = torch.tensor([len(seq) for seq in sub_queries], device=device) - 1
                split_points = (sequence_lengths + 1).tolist()
                policy_inputs = processing_class(
                    [self.query_prompt.format(self.instruction, step.custom_embedding_strs[0])
                     for seq in sub_queries for step in seq],
                    padding=True,
                    truncation=True,
                    max_length=args.embedding_max_length,
                    return_tensors="pt",
                )
                policy_inputs = {k: v.split(split_points) for k, v in policy_inputs.items()}
                critic_inputs, value_indices = get_critic_input(
                    queries, sub_queries, cur_responses, processing_class, args.embedding_max_length
                )

                for i in range(0, len(queries), args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    response = responses[i : i + args.local_rollout_forward_batch_size]
                    sub_query = sub_queries[i : i + args.local_rollout_forward_batch_size]
                    cur_response = cur_responses[i : i + args.local_rollout_forward_batch_size]
                    source_node = source_nodes[i : i + args.local_rollout_forward_batch_size]
                    sequence_length = sequence_lengths[i : i + args.local_rollout_forward_batch_size]
                    split_point = split_points[i : i + args.local_rollout_forward_batch_size]
                    policy_input = {
                        k: torch.cat(v[i : i + args.local_rollout_forward_batch_size]).to(device)
                        for k, v in policy_inputs.items()
                    }
                    critic_input = {
                        k: v[i : i + args.local_rollout_forward_batch_size].to(device)
                        for k, v in critic_inputs.items()
                    }
                    value_index = [
                        idx.to(device) for idx in value_indices[i : i + args.local_rollout_forward_batch_size]
                    ]

                    query_embed = forward(model, policy_input)
                    query_embed = pad_sequence(
                        query_embed.split(split_point),
                        batch_first=True
                    )  # (batch_size, max_steps, embedding_size)
                    doc_embed = pad_sequence([
                        torch.tensor([retrieve_embeddings(self.vector_store, step, self.loop, args) for step in seq], device=device)
                        for seq in source_node
                    ], batch_first=True)  # (batch_size, max_steps, num_nodes, embedding_size)
                    logits = torch.matmul(doc_embed, query_embed.unsqueeze(-1)).squeeze(-1)  # (batch_size, max_steps, num_nodes)
                    logits = logits.masked_fill((doc_embed.abs().sum(dim=-1) == 0), INVALID_LOGIT)
                    logits /= args.temperature + 1e-7
                    doc_index = torch.arange(args.similarity_top_k, device=device).view(1, 1, -1).expand(*logits.shape[:-1], -1)  # (batch_size, max_steps, similarity_top_k)
                    logprob = plackett_luce_logprob(logits, doc_index)  # (batch_size, max_steps)
                    del query_embed, logits
                    empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_query_embed = forward(model, policy_input)
                    else:
                        ref_query_embed = forward(ref_policy, policy_input, with_wrapper=False)
                    ref_query_embed = pad_sequence(
                        ref_query_embed.split(split_point),
                        batch_first=True
                    )  # (batch_size, max_steps, embedding_size)
                    ref_logits = torch.matmul(doc_embed, ref_query_embed.unsqueeze(-1)).squeeze(-1)  # (batch_size, max_steps, num_nodes)
                    ref_logits = ref_logits.masked_fill((doc_embed.abs().sum(dim=-1) == 0), INVALID_LOGIT)
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = plackett_luce_logprob(ref_logits, doc_index)  # (batch_size, max_steps)
                    del doc_index, ref_query_embed, ref_logits
                    empty_cache()

                    # Response Processing 1. run reward functions on the responses
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    value = get_value(unwrapped_value_model, critic_input, value_index)
                    value = pad_sequence(value, batch_first=True)  # (batch_size, max_steps)

                    # Repeat all input columns (but "query") to match the num of generations
                    keys = [key for key in data.keys() if key != "query"]
                    reward_kwargs = {key: data[key][i : i + args.local_rollout_forward_batch_size] for key in keys}

                    rewards_per_func = torch.zeros(len(query), len(self.reward_funcs), device=device)
                    for j, reward_func in enumerate(reward_funcs):
                        output_reward_func = reward_func(
                            queries=query,
                            responses=response,
                            sub_queries=sub_query,
                            cur_responses=cur_response,
                            source_nodes=source_node,
                            **reward_kwargs
                        )
                        rewards_per_func[:, j] = torch.tensor(output_reward_func, device=device)
                    score = (rewards_per_func * reward_weights.to(device).unsqueeze(0)).sum(dim=1)  # (batch_size,)

                    doc_embeds.append(doc_embed)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    scores.append(score)
                    values.append(value)
                doc_embeds = pad_sequence([
                    seq for doc_embed in doc_embeds for seq in doc_embed
                ], batch_first=True)  # (batch_size, max_steps, num_nodes, embedding_size)
                logprobs = pad_sequence([
                    seq for logprob in logprobs for seq in logprob
                ], batch_first=True)  # (batch_size, max_steps)
                ref_logprobs = pad_sequence([
                    seq for ref_logprob in ref_logprobs for seq in ref_logprob
                ], batch_first=True)  # (batch_size, max_steps)
                scores = torch.cat(scores, 0)
                values = pad_sequence([
                    seq for value in values for seq in value
                ], batch_first=True)  # (batch_size, max_steps)
                del (
                    policy_input, critic_input, value_index, split_point, doc_embed,
                    logprob, ref_logprob, sequence_length, score, value, rewards_per_func
                )
                empty_cache()
                gc.collect()

                # 2. Be very careful with `padding_mask`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                sub_query_idxs = torch.arange(logprobs.shape[1], device=logprobs.device).repeat(logprobs.shape[0], 1)
                padding_mask = sub_query_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                values = torch.masked_fill(values, padding_mask, 0)

                # 3. compute rewards
                # Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
                logr = ref_logprobs - logprobs
                kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = sequence_lengths
                rewards[actual_start, actual_end] += scores

                # 4. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask, 0)

                # 5. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = logprobs.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_sequence_lengths = sequence_lengths[micro_batch_inds]
                            mb_split_points = [split_points[ind] for ind in micro_batch_inds]
                            mb_truc_point = mb_sequence_lengths.max().item() + 1  # to match the shape of `new_query_embed`
                            mb_padding_mask = padding_mask[micro_batch_inds, :mb_truc_point]
                            mb_advantage = advantages[micro_batch_inds, :mb_truc_point]
                            mb_doc_embeds = doc_embeds[micro_batch_inds, :mb_truc_point]
                            mb_logprobs = logprobs[micro_batch_inds, :mb_truc_point]
                            mb_return = returns[micro_batch_inds, :mb_truc_point]
                            mb_values = values[micro_batch_inds, :mb_truc_point]
                            mb_policy_inputs = {
                                k: torch.cat([v[ind] for ind in micro_batch_inds]).to(device)
                                for k, v in policy_inputs.items()
                            }
                            mb_critic_inputs = {k: v[micro_batch_inds].to(device) for k, v in critic_inputs.items()}
                            mb_value_indices = [value_indices[ind] for ind in micro_batch_inds]

                            new_query_embed, vpred_temp = forward(
                                model, mb_policy_inputs, mb_critic_inputs, mb_value_indices
                            )
                            new_query_embed = pad_sequence(
                                new_query_embed.split(mb_split_points),
                                batch_first=True,
                            )
                            logits = torch.matmul(mb_doc_embeds, new_query_embed.unsqueeze(-1)).squeeze(-1)
                            logits = logits.masked_fill((mb_doc_embeds.abs().sum(dim=-1) == 0), INVALID_LOGIT)
                            logits /= args.temperature + 1e-7
                            doc_index = torch.arange(args.similarity_top_k, device=device).view(1, 1, -1).expand(*logits.shape[:-1], -1)
                            new_logprobs = plackett_luce_logprob(logits, doc_index)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, mb_padding_mask, INVALID_LOGPROB
                            )
                            vpred = pad_sequence(vpred_temp, batch_first=True)
                            vpred = torch.masked_fill(vpred, mb_padding_mask, 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~mb_padding_mask)
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~mb_padding_mask
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            # todo: When `per_device_train_batch_size > 1`,
                            #  batch effects may cause the `ratio` to deviate from 1 in on-policy settings.
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~mb_padding_mask)
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = -(prob_dist * torch.log(prob_dist.clamp_min(1e-7))).sum(dim=-1)
                            entropy_loss = masked_mean(entropy, ~mb_padding_mask)
                            loss = pg_loss + args.vf_coef * vf_loss
                            if args.ent_coef != 0:
                                loss -= args.ent_coef * entropy_loss
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                optimizer.step()
                                optimizer.zero_grad()
                            with torch.no_grad():
                                critic_metrics = compute_critic_metrics(mb_return, vpred, mb_padding_mask)
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~mb_padding_mask
                                )
                                approxkl = masked_mean(0.5 * (logprobs_diff**2), ~mb_padding_mask)
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy_loss
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = masked_mean(ratio, ~mb_padding_mask)
                                seq_len_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (mb_sequence_lengths + 1).float().mean()
                                critic_baseline_mse_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    critic_metrics["baseline_mse"]
                                )
                                critic_value_mse_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    critic_metrics["value_mse"]
                                )
                                critic_corr_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    critic_metrics["corr"]
                                )
                                critic_r2_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    critic_metrics["r2"]
                                )
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        new_query_embed, vpred_temp, logits, doc_index, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio,
                        pg_losses, pg_losses2, pg_loss_max, pg_loss, loss, pg_clipfrac,
                        prob_dist, entropy, approxkl, mb_sequence_lengths, mb_split_points, mb_padding_mask,
                        mb_advantage, mb_doc_embeds, mb_logprobs, mb_return, mb_values,
                        mb_policy_inputs, mb_critic_inputs, mb_value_indices,
                    )
                    # fmt: on
                    empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                ppo_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/ppo_reward"] = self.accelerator.gather_for_metrics(ppo_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["sequence_length"] = self.accelerator.gather_for_metrics(seq_len_stats).mean().item()
                metrics["critic/baseline_mse"] = (
                    self.accelerator.gather_for_metrics(critic_baseline_mse_stats).mean().item()
                )
                metrics["critic/value_mse"] = (
                    self.accelerator.gather_for_metrics(critic_value_mse_stats).mean().item()
                )
                metrics["critic/corr"] = (
                    self.accelerator.gather_for_metrics(critic_corr_stats).mean().item()
                )
                metrics["critic/r2"] = (
                    self.accelerator.gather_for_metrics(critic_r2_stats).mean().item()
                )
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                empty_cache()
            del (
                queries,
                doc_embeds,
                logprobs,
                ref_logprobs,
                values,
                responses,
                sub_queries,
                source_nodes,
                sequence_lengths,
                split_points,
                policy_inputs,
                critic_inputs,
                value_indices,
                sub_query_idxs,
                padding_mask,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args

        table = defaultdict(list)
        for batch in self.eval_dataloader:
            query = batch["query"]
            all_query = gather_object(query)
            with torch.no_grad():
                if self.accelerator.is_main_process:
                    self.query_engine.eval()
                    response, sub_query, cur_response, source_node = batch_generation(
                        self.query_engine,
                        self.retriever,
                        all_query,
                        "eval",
                        self.search_r1_processing_class,
                        args,
                        self.loop,
                    )
                else:
                    response = [None] * len(all_query)
                    sub_query = [None] * len(all_query)
                    cur_response = [None] * len(all_query)
                    source_node = [None] * len(all_query)
                response = broadcast_object_list(response, from_process=0)
                sub_query = broadcast_object_list(sub_query, from_process=0)
                cur_response = broadcast_object_list(cur_response, from_process=0)
                source_node = broadcast_object_list(source_node, from_process=0)
                table["query"].extend(all_query)
                table["model response"].extend(response)
                process_slice = slice(
                    self.accelerator.process_index * len(query),
                    (self.accelerator.process_index + 1) * len(query),
                )
                response = response[process_slice]
                sub_query = sub_query[process_slice]
                cur_response = cur_response[process_slice]
                source_node = source_node[process_slice]

                # Repeat all input columns (but "query") to match the num of generations
                keys = [key for key in batch.keys() if key != "query"]
                reward_kwargs = {key: batch[key] for key in keys}

                device = self.accelerator.device
                rewards_per_func = torch.zeros(len(query), len(self.reward_funcs), device=device)
                for i, reward_func in enumerate(self.reward_funcs):
                    output_reward_func = reward_func(
                        queries=query,
                        responses=response,
                        sub_queries=sub_query,
                        cur_responses=cur_response,
                        source_nodes=source_node,
                        **reward_kwargs
                    )
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, device=device)
                score = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
                table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

            if sampling:
                break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            if is_rich_available():
                print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

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

        model = self.accelerator.unwrap_model(self.model)
        if hasattr(model.config, "_name_or_path") and not os.path.isdir(model.config._name_or_path):
            base_model = model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="PPO",
            trainer_citation=citation,
            paper_title="Fine-Tuning Language Models from Human Preferences",
            paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
