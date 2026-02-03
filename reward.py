import asyncio
import torch
import evaluate

from metric.qa_em import compute_score_subem
from rouge_score import rouge_scorer
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.evaluation import CorrectnessEvaluator


rouge_metric = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
squad_metric = evaluate.load('squad')

STATUS_REWARD = {
    "NOT_FOUND": 0.0,
    "AMBIGUOUS": 0.5,
    "FOUND": 1.0,
}


def em_reward(
    responses: list[str],
    ground_truths: list[list[str]],
    **kwargs
) -> list[float]:
    """Reward function that checks if the response is exactly matching the ground truth."""
    rewards = []
    for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
        out = squad_metric.compute(
            predictions=[{"id": str(i), "prediction_text": response}],
            references=[
                {
                    "id": str(i),
                    "answers": {
                        "text": ground_truth,
                        "answer_start": [0] * len(ground_truth),
                    },
                }
            ],
        )
        em = float(out["exact_match"]) / 100.0
        rewards.append(em)
    return rewards


def f1_reward(
    responses: list[str],
    ground_truths: list[list[str]],
    **kwargs
) -> list[float]:
    """Reward function that computes the F1 score between the response and the ground truth."""
    rewards = []
    for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
        out = squad_metric.compute(
            predictions=[{"id": str(i), "prediction_text": response}],
            references=[
                {
                    "id": str(i),
                    "answers": {
                        "text": ground_truth,
                        "answer_start": [0] * len(ground_truth),
                    },
                }
            ],
        )
        f1 = float(out["f1"]) / 100.0
        rewards.append(f1)
    return rewards


def sub_em_reward(
    responses: list[str],
    ground_truths: list[list[str]],
    **kwargs
) -> list[float]:
    """Reward function that checks if the response is exactly matching the ground truth."""
    rewards = [
        compute_score_subem(solution_str=response, ground_truth=ground_truth)
        for response, ground_truth in zip(responses, ground_truths)
    ]
    return rewards


def rouge_reward(
    responses: list[str],
    ground_truths: list[list[str]],
    max_chars=128,
    **kwargs
) -> list[float]:
    """Reward function that computes the ROUGE score between the response and the ground truth."""
    responses = [response[:max_chars] for response in responses]
    ground_truths = [[gt[:max_chars] for gt in ground_truth] for ground_truth in ground_truths]
    rewards = [
        max(rouge_metric.score(gt, response)['rougeL'].fmeasure for gt in ground_truth)
        for response, ground_truth in zip(responses, ground_truths)
    ]
    return rewards


async def llm_correctness_reward(
    queries: list[str],
    responses: list[str],
    ground_truths: list[list[str]],
    **kwargs
) -> list[float]:
    """Reward function that uses an LLM to evaluate the correctness of the response."""
    evaluator = CorrectnessEvaluator()

    tasks = [
        evaluator.aevaluate(
            query=query,
            response=response,
            reference=ground_truth[0],
        )
        for query, response, ground_truth in zip(queries, responses, ground_truths)
    ]
    results = await asyncio.gather(*tasks)

    rewards = [result.score for result in results]
    return rewards


async def relevance_reward(
    sub_queries: list[list[QueryBundle]],
    source_nodes: list[list[list[NodeWithScore]]],
    similarity_top_k: int,
    relevance_base_url: str,
    relevance_model: str,
    relevance_instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    **kwargs,
) -> list[list[float]]:
    prefix = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    document_template = "<Document>: {doc}{suffix}"

    import cohere
    client = cohere.AsyncClientV2(base_url=relevance_base_url)

    async def score_step(query, nodes):
        query = query_template.format(prefix=prefix, instruction=relevance_instruction, query=query.query_str)
        documents = [document_template.format(doc=node.text, suffix=suffix) for node in nodes[:similarity_top_k]]
        response = await client.rerank(
            model=relevance_model,
            query=query,
            documents=documents,
            top_n=len(documents),
        )
        scores = [result.relevance_score for result in response.results]
        return torch.tensor(scores).mean().item()

    split_points = [len(example) for example in sub_queries]
    tasks = [
        score_step(step_query, step_nodes)
        for example_queries, example_nodes in zip(sub_queries, source_nodes)
        for step_query, step_nodes in zip(example_queries, example_nodes)
    ]
    results = await asyncio.gather(*tasks)

    rewards = [example_reward.tolist() for example_reward in torch.tensor(results).split(split_points)]
    return rewards


def novelty_reward(
    source_nodes: list[list[list[NodeWithScore]]],
    similarity_top_k: int,
    **kwargs
) -> list[list[float]]:
    """Reward novelty of retrieved nodes at each step.

    For each example and step, compute the fraction of the current step's top-k retrieved nodes
    that have already appeared in any previous step. Lower overlap => higher reward.

    Reward per step is `1 - overlap_ratio`, in `[0, 1]`.
    """

    all_rewards: list[list[float]] = []
    for example_steps in source_nodes:
        seen_node_ids: set[str] = set()
        example_rewards: list[float] = []

        for step_nodes in example_steps:
            node_ids: list[str] = [node.node_id for node in step_nodes[:similarity_top_k]]
            overlap_count = sum(1 for node_id in node_ids if node_id in seen_node_ids)
            overlap_ratio = overlap_count / len(node_ids)
            example_rewards.append((1.0 - overlap_ratio))
            seen_node_ids.update(node_ids)

        all_rewards.append(example_rewards)

    return all_rewards
