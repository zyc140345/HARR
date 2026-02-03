import argparse
import re
import torch
import torch.nn.functional as F

from transformers import PreTrainedTokenizerBase
from llama_index.core import Settings
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.tools import ToolMetadata
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.workflow import Context, WorkflowRuntimeError
from llama_index.core.agent.workflow import ReActAgent, ToolCallResult
from llama_index_hacked.query_engine import RetrieverQueryEngine
from llama_index_hacked.tool import StatefulQueryEngineTool, build_stateful_embedding_query
from llama_index_hacked.workflow import MultiStepQueryEngineWorkflow, SubQaEvent
from prompt import react_system_prompt, SEARCH_R1_PROMPT_TEMPLATE, SEARCH_R1_SEARCH_TEMPLATE


async def agent_generate(
    query: str,
    query_engine: RetrieverQueryEngine,
    retriever: BaseRetriever,
    args,
    **kwargs,
) -> tuple[
    str,
    list[QueryBundle],
    list[str],
    list[list[NodeWithScore]]
]:
    sub_query = []
    cur_response = []
    source_node = []

    try:
        query_engine_tools = [
            StatefulQueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name=args.collection_name,
                    description=(
                        "Retrieves information from Wikipedia-2018 articles. "
                        "Use a detailed plain text question as input."
                    ),
                ),
                original_question=query,
                enable_stateful=args.enable_stateful,
            ),
        ]
        agent = ReActAgent(tools=query_engine_tools)
        agent.update_prompts({"react_header": react_system_prompt})

        ctx = Context(agent)
        handler = agent.run(query, ctx=ctx, max_iterations=args.max_iterations)
        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                if ev.tool_output.is_error:
                    raise WorkflowRuntimeError(ev.tool_output.raw_output)

                sub_query.append(ev.tool_output.raw_input["input"])
                cur_response.append(ev.tool_output.raw_output.response)
                source_node.append(ev.tool_output.raw_output.source_nodes)

        response = await handler
        response = response.response.content
    except WorkflowRuntimeError as e:
        response = str(e)

    if not source_node:
        if args.enable_stateful:
            embedding_query = build_stateful_embedding_query(
                original_question=query,
                history=[],
                step_query=query,
            )
        else:
            embedding_query = f"Query: {query}"
        query_bundle = QueryBundle(
            query_str=query, custom_embedding_strs=[embedding_query]
        )
        source_node = [await retriever.aretrieve(query_bundle)]
        sub_query = [query_bundle]
        cur_response = [response]

    return response, sub_query, cur_response, source_node


async def workflow_generate(
    query: str,
    query_engine: RetrieverQueryEngine,
    retriever: BaseRetriever,
    args,
    **kwargs,
) -> tuple[
    str,
    list[QueryBundle],
    list[str],
    list[list[NodeWithScore]]
]:
    sub_query = []
    cur_response = []
    source_node = []

    try:
        workflow = MultiStepQueryEngineWorkflow(timeout=args.timeout)
        handler = workflow.run(
            query=query,
            query_engine=query_engine,
            index_summary="Used to retrieve factual information from Wikipedia-2018 articles with natural language questions",
            enable_stateful=args.enable_stateful,
            num_steps=args.max_iterations,
        )
        async for ev in handler.stream_events():
            if isinstance(ev, SubQaEvent):
                sub_query.append(ev.sub_qa)
                cur_response.append(ev.cur_response)
                source_node.append(ev.source_nodes)
        response = await handler
        response = response.response
    except WorkflowRuntimeError as e:
        response = str(e)

    if not source_node:
        if args.enable_stateful:
            embedding_query = build_stateful_embedding_query(
                original_question=query,
                history=[],
                step_query=query,
            )
        else:
            embedding_query = f"Query: {query}"
        query_bundle = QueryBundle(
            query_str=query, custom_embedding_strs=[embedding_query]
        )
        source_node = [await retriever.aretrieve(query_bundle)]
        sub_query = [query_bundle]
        cur_response = [response]

    return response, sub_query, cur_response, source_node


_target_sequences = ["</search>", "</answer>"]
_sub_query_pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_cur_response_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def _extract_result(pattern, text):
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def _nodes2string(nodes):
    format_reference = ""
    for idx, node in enumerate(nodes):
        content = node.get_content()
        title = content.split("\n")[0].lstrip("passage: ")
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return format_reference

def _sample(nodes, args):
    with torch.no_grad():
        logits = torch.tensor([node.score for node in nodes])
        logits /= args.temperature + 1e-7
        prob_dist = F.softmax(logits, dim=-1)
        index = torch.multinomial(
            prob_dist, num_samples=args.similarity_top_k, replacement=False
        )
        index_set = set(index.tolist())
    additional_nodes = [node for i, node in enumerate(nodes) if i not in index_set]
    nodes = [nodes[i] for i in index.tolist()]
    return nodes, additional_nodes

async def search_r1_generate(
    query: str,
    retriever: BaseRetriever,
    processing_class: PreTrainedTokenizerBase,
    mode: str,
    args,
    **kwargs,
) -> tuple[
    str,
    list[QueryBundle],
    list[str],
    list[list[NodeWithScore]]
]:
    response = ""
    sub_query = []
    cur_response = []
    source_node = []
    history = []

    prompt = SEARCH_R1_PROMPT_TEMPLATE.format(question=query)
    if processing_class.chat_template:
        prompt = processing_class.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    for _ in range(args.max_iterations):
        output = await Settings.llm.acomplete(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            extra_body={"include_stop_str_in_output": True},
            stop=_target_sequences,
        )
        output_text = output.text

        response = _extract_result(_answer_pattern, output_text)
        if response:
            prompt += output_text
            break

        sub_qa = _extract_result(_sub_query_pattern, output_text)
        cur_resp = _extract_result(_cur_response_pattern, output_text)
        if sub_qa:
            if args.enable_stateful:
                embedding_query = build_stateful_embedding_query(
                    original_question=query,
                    history=history,
                    step_query=sub_qa,
                )
            else:
                embedding_query = f"Query: {sub_qa}"
            query_bundle = QueryBundle(
                query_str=sub_qa, custom_embedding_strs=[embedding_query]
            )
            nodes = await retriever.aretrieve(query_bundle)

            if mode == "train":
                nodes, additional_nodes = _sample(nodes, args)
            else:
                nodes, additional_nodes = (
                    nodes[: args.similarity_top_k],
                    nodes[args.similarity_top_k :],
                )

            search_results = _nodes2string(nodes)
            sub_query.append(query_bundle)
            cur_response.append(str(cur_resp))
            source_node.append(nodes + additional_nodes)
        else:
            search_results = ""

        search_text = SEARCH_R1_SEARCH_TEMPLATE.format(
            output_text=output_text, search_results=search_results
        )
        prompt += search_text
        history.append({
            "sub_query": sub_qa,
            "answer": cur_resp,
        })

    if not source_node:
        if args.enable_stateful:
            embedding_query = build_stateful_embedding_query(
                original_question=query,
                history=[],
                step_query=query,
            )
        else:
            embedding_query = f"Query: {query}"
        query_bundle = QueryBundle(
            query_str=query, custom_embedding_strs=[embedding_query]
        )
        source_node = [await retriever.aretrieve(query_bundle)]
        sub_query = [query_bundle]
        cur_response = [response]

    return str(response), sub_query, cur_response, source_node
