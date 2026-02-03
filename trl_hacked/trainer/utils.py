import asyncio
import inspect
import torch
import torch.nn.functional as F

from typing import Union, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.utils.import_utils import _is_package_available
from llama_index.core.schema import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index_hacked.query_engine import RetrieverQueryEngine
from generator import agent_generate, workflow_generate, search_r1_generate


def is_httpx_available() -> bool:
    return _is_package_available("httpx")


def is_async_func(func):
    return inspect.iscoroutinefunction(inspect.unwrap(func))


def batch_generation(
    query_engine: RetrieverQueryEngine,
    retriever: BaseRetriever,
    queries: list[str],
    mode: str,
    search_r1_processing_class: Optional[PreTrainedTokenizerBase],
    args,
    loop: asyncio.AbstractEventLoop
):
    async def generate(query):
        if args.rag_method == "react_agent":
            return await agent_generate(
                query=query,
                query_engine=query_engine,
                retriever=retriever,
                args=args,
            )
        elif args.rag_method == "multistep_workflow":
            return await workflow_generate(
                query=query,
                query_engine=query_engine,
                retriever=retriever,
                args=args,
            )
        elif args.rag_method == "search_r1":
            return await search_r1_generate(
                query=query,
                retriever=retriever,
                processing_class=search_r1_processing_class,
                mode=mode,
                args=args,
            )
        else:
            raise ValueError(f"Unknown rag_method: {args.rag_method}")

    async def generate_parallel(prompts):
        return await asyncio.gather(*[generate(prompt) for prompt in prompts])

    results = loop.run_until_complete(generate_parallel(queries))
    responses = [result[0] for result in results]
    sub_queries = [result[1] for result in results]
    cur_responses = [result[2] for result in results]
    source_nodes = [result[3] for result in results]

    return responses, sub_queries, cur_responses, source_nodes


def retrieve_embeddings(vector_store, nodes, loop, args):
    nodes_with_embed = loop.run_until_complete(vector_store.aget_nodes([node.node_id for node in nodes]))
    node_dict = {node.node_id: node for node in nodes_with_embed}
    embeddings = [node_dict[node.node_id].get_embedding() for node in nodes]

    # Qdrant may return fewer nodes than requested
    nodes_requested = args.sampling_pool_size
    embeddings += [[0] * len(embeddings[-1]) for _ in range(nodes_requested - len(nodes))]

    return embeddings


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits]).unsqueeze(-1)
        per_step_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_step_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_step_logps = row_logps.gather(dim=-1, index=row_labels)
            per_step_logps.append(row_per_step_logps)
        per_step_logps = torch.stack(per_step_logps)
    return per_step_logps


def plackett_luce_logprob(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Compute joint log-prob under Plackett–Luce (sampling without replacement).

    Args:
        logits (`torch.Tensor`): `(..., N)` unnormalized scores.
        index  (`torch.Tensor`): `(..., K)` chosen indices in selection order.

    Returns:
        `torch.Tensor`: `(...,)` joint log-probability of the ordered selection.
    """
    # Promote dtype for numerical stability when needed (e.g., bfloat16)
    orig_dtype = logits.dtype
    if logits.dtype not in (torch.float32, torch.float64):
        logits = logits.float()

    N = logits.size(-1)

    # Build per-step mask of previously selected items using one-hot and exclusive cumsum
    # step_one_hot: (..., K, N)
    step_one_hot = F.one_hot(index.long(), num_classes=N)
    prev_mask = step_one_hot.cumsum(dim=-2)
    prev_mask = torch.roll(prev_mask, shifts=1, dims=-2)
    prev_mask[..., 0, :] = 0
    prev_mask = prev_mask.to(torch.bool)

    # Expand logits with a singleton step axis and mask previously taken items at each step
    # logits_expanded: (..., 1, N) -> broadcast to (..., K, N)
    logits_expanded = logits.unsqueeze(-2)
    masked_logits = logits_expanded.masked_fill(prev_mask, float("-inf"))
    step_log_probs = torch.log_softmax(masked_logits, dim=-1)  # (..., K, N)

    # Gather per-step log-prob and sum over K to obtain joint log-prob: (...,)
    per_step = torch.gather(step_log_probs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    joint = per_step.sum(dim=-1)
    return joint.to(orig_dtype)


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def forward(
    model: torch.nn.Module,
    policy_input: dict[str, torch.Tensor],
    critic_input: dict[str, torch.Tensor] = None,
    value_index: list[torch.Tensor] = None,
    with_wrapper: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, list[torch.Tensor]]]:
    """
    Performs a forward pass through the model with the given input.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        policy_input (`dict[str, torch.Tensor]`):
            The policy model input, including input_ids, attention_mask, and position_ids.
        critic_input (`dict[str, torch.Tensor]`):
            The critic model input, including input_ids, attention_mask, and position_ids.
        value_index (`list[torch.Tensor]`):
            The token indices whose critic outputs are used as values.
        with_wrapper (`bool`, *optional*, defaults to `True`):
            Whether the model is wrapped in a `PolicyAndValueWrapper`.

    Returns:
        `Union[torch.Tensor, Tuple[torch.Tensor, list[torch.Tensor]]]`:
            The output of the model.
    """
    if critic_input:
        outputs = model(policy_input, critic_input)
        embeddings = last_token_pool(outputs[0].last_hidden_state, policy_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize embeddings
        values = [value[idx].squeeze(-1) for value, idx in zip(outputs[1], value_index)]
        return embeddings, values
    else:
        outputs = model(policy_input) if with_wrapper else model(**policy_input)
        embeddings = last_token_pool(outputs.last_hidden_state, policy_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize embeddings
        return embeddings


def compute_critic_metrics(
    returns: torch.Tensor,
    values: torch.Tensor,
    padding_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute simple diagnostics for critic performance:
    - baseline_mse: MSE of a constant baseline predicting the mean return.
    - value_mse: MSE between critic predictions and returns.
    - corr: Pearson correlation between critic predictions and returns.
    - r2: Coefficient of determination between critic predictions and returns.

    Args:
        returns (`torch.Tensor`):
            Target returns, shape `(batch, time)`.
        values (`torch.Tensor`):
            Critic predictions, shape `(batch, time)`.
        padding_mask (`torch.Tensor`):
            Boolean mask with `True` for padding positions, same shape as `returns`.

    Returns:
        `dict[str, torch.Tensor]`:
            Dictionary of scalar tensors on the same device as `returns`.
    """
    assert returns.shape == values.shape == padding_mask.shape

    valid_mask = ~padding_mask
    num_valid = valid_mask.sum()
    device = returns.device

    if num_valid <= 1:
        zero = torch.zeros((), device=device)
        return {
            "baseline_mse": zero,
            "value_mse": zero,
            "corr": zero,
            "r2": zero,
        }

    r = returns[valid_mask].float()
    v = values[valid_mask].float()

    r_mean = r.mean()
    v_mean = v.mean()
    r_centered = r - r_mean
    v_centered = v - v_mean

    # baseline that predicts mean return
    baseline_mse = ((r - r_mean) ** 2).mean()
    value_mse = ((v - r) ** 2).mean()

    r_var = (r_centered ** 2).mean()
    cov = (r_centered * v_centered).mean()
    r_std = r_centered.std(unbiased=False)
    v_std = v_centered.std(unbiased=False)
    eps = 1e-8

    corr = cov / ((r_std + eps) * (v_std + eps))
    r2 = 1.0 - value_mse / (r_var + eps)

    return {
        "baseline_mse": baseline_mse,
        "value_mse": value_mse,
        "corr": corr,
        "r2": r2,
    }


def get_value(
    model: torch.nn.Module,
    model_input: dict[str, torch.Tensor],
    value_index: list[torch.Tensor]
) -> list[torch.Tensor]:
    """
    Computes the values for a given model and input.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the value logits.
        model_input (`dict[str, torch.Tensor]`):
            The model input, including input_ids, attention_mask, and position_ids.
        value_index (`torch.Tensor`):
            The token indices whose critic outputs are used as values.

    Returns:
        `values` (`list[torch.Tensor]`):
            The values for each input.
    """
    lm_backbone = getattr(model, model.base_model_prefix)
    output = lm_backbone(**model_input)
    values = model.score(output.last_hidden_state)
    return [value[idx].squeeze(-1) for value, idx in zip(values, value_index)]


def get_critic_input(
    queries: list[str],
    sub_queries: list[list[QueryBundle]],
    cur_responses: list[list[str]],
    processing_class: PreTrainedTokenizerBase,
    max_length: int,
) -> Tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    all_special_ids = set(processing_class.all_special_ids)
    add_bos, add_eos = False, False

    def remove_bos_eos(seq_id, seq):
        if seq_id != 0 and add_bos:
            seq = seq[1:]
        if seq_id != len(prev_reasoning) - 1 and add_eos:
            seq = seq[:-1]
        return seq

    critic_inputs = []
    value_indices = []
    for i, (query, sub_query, cur_response) in enumerate(zip(queries, sub_queries, cur_responses)):
        prev_reasoning = [f"Original Question: {query}\nReasoning History:\n- {sub_query[0].query_str}"] + [
            f"\n- {r}\n" f"- {q.query_str}" for r, q in zip(cur_response[:-1], sub_query[1:])
        ]
        critic_input = processing_class(
            prev_reasoning,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        if i == 0:
            add_bos = critic_input["input_ids"][0][0] in all_special_ids
            add_eos = critic_input["input_ids"][0][-1] in all_special_ids

        critic_input = {
            k: [remove_bos_eos(seq_id, seq) for seq_id, seq in enumerate(v)]
            for k, v in critic_input.items()
        }
        value_index = [len(seq) for seq in critic_input["input_ids"]]
        value_index = torch.cumsum(torch.tensor(value_index), dim=0) - 1
        critic_input = {k: [elem for seq in v for elem in seq] for k, v in critic_input.items()}

        critic_inputs.append(critic_input)
        value_indices.append(value_index)

    critic_inputs = {
        k: pad_sequence(
            [torch.tensor(critic_inputs[i][k][::-1]) for i in range(len(critic_inputs))],
            batch_first=True,
            padding_value=processing_class.pad_token_id if k == "input_ids" else 0
        ).flip(dims=[1])
        for k in critic_inputs[0].keys()
    }
    value_indices = [
        value_index + (attention_mask == 0).sum()
        for value_index, attention_mask in zip(value_indices, critic_inputs["attention_mask"])
    ]
    return critic_inputs, value_indices
