import json
import re

from typing import List, Dict, Optional, Any
from json import JSONDecodeError
from llama_index.core import BasePromptTemplate, Settings
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle, QueryType, TextNode, NodeWithScore
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.workflow import Event, Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.utils import print_text
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.indices.query.query_transform.prompts import (
    DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT,
    StepDecomposeQueryTransformPrompt
)
from llama_index_hacked.query_engine import parse_answer, ANSWER_PATTERN
from llama_index_hacked.tool import build_stateful_embedding_query
from prompt import QA_SM_PROMPT_SEL, QUERY_DECOMPOSE_SM_PROMPT


class QueryMultiStepEvent(Event):
    """
    Event containing results of a multistep query process.

    Attributes:
        nodes (List[NodeWithScore]): List of nodes with their associated scores.
        source_nodes (List[NodeWithScore]): List of source nodes with their scores.
        final_response_metadata (Dict[str, Any]): Metadata associated with the final response.
    """

    nodes: List[NodeWithScore]
    source_nodes: List[NodeWithScore]
    final_response_metadata: Dict[str, Any]


class SubQaEvent(Event):
    sub_qa: QueryBundle
    cur_response: str
    cur_status: str
    source_nodes: List[NodeWithScore]


class StepDecomposeQueryTransform(BaseQueryTransform):
    """
    Step decompose query transform.

    Decomposes query into a subquery given the current index struct
    and previous reasoning.

    NOTE: doesn't work yet.

    Args:
        llm (Optional[LLM]): LLM for generating
            hypothetical documents

    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        step_decompose_query_prompt: Optional[StepDecomposeQueryTransformPrompt] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        super().__init__()
        self._llm = llm or Settings.llm
        self._step_decompose_query_prompt: BasePromptTemplate = (
            step_decompose_query_prompt or DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT
        )
        self.verbose = verbose

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"step_decompose_query_prompt": self._step_decompose_query_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "step_decompose_query_prompt" in prompts:
            self._step_decompose_query_prompt = prompts["step_decompose_query_prompt"]

    def _parse_llm_output(self, llm_output: str) -> str:
        """
        Parses the LLM output to extract the sub-question.
        Expected format: optional thought text + a JSON object.
        """
        # JSON state-machine schema:
        # {"action":"ASK|STOP","sub_query": "...", "strategy":"NEXT|REPHRASE|BROADEN|DISAMBIGUATE"}
        llm_output = (llm_output or "").strip()
        if llm_output:
            llm_output = re.sub(r"```(?:json)?", "", llm_output, flags=re.I).replace("```", "").strip()
            m = ANSWER_PATTERN.search(llm_output)
            if m:
                try:
                    obj = json.loads(m.group("json"))
                    action = str(obj.get("action", "")).strip().upper()
                    if action == "STOP":
                        return "None"
                    sub_query = str(obj.get("sub_query", "")).strip()
                    if not sub_query:
                        return "None"
                    return sub_query
                except JSONDecodeError:
                    pass

        if self.verbose:
            print_text(f"> Fallback: Could not parse LLM output: {llm_output}", color="red")
        return "None"

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform."""
        index_summary = metadata.get("index_summary", "None")
        prev_reasoning = metadata.get("prev_reasoning")
        fmt_prev_reasoning = str(prev_reasoning) if prev_reasoning else "[]"

        # given the text from the index, we can use the query bundle to generate
        # a new query bundle
        query_str = query_bundle.query_str
        raw_llm_output = self._llm.predict(
            self._step_decompose_query_prompt,
            prev_reasoning=fmt_prev_reasoning,
            query_str=query_str,
            context_str=index_summary,
        )
        new_query_str = self._parse_llm_output(raw_llm_output)
        if self.verbose:
            print_text(f"> Current query: {query_str}\n", color="yellow")
            print_text(f"> New query: {new_query_str}\n", color="pink")
        return QueryBundle(
            query_str=new_query_str,
            custom_embedding_strs=query_bundle.custom_embedding_strs,
        )

    async def _arun(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform asynchronously."""
        index_summary = metadata.get("index_summary", "None")
        prev_reasoning = metadata.get("prev_reasoning")
        fmt_prev_reasoning = str(prev_reasoning) if prev_reasoning else "[]"

        # given the text from the index, we can use the query bundle to generate
        # a new query bundle
        query_str = query_bundle.query_str
        raw_llm_output = await self._llm.apredict(
            self._step_decompose_query_prompt,
            prev_reasoning=fmt_prev_reasoning,
            query_str=query_str,
            context_str=index_summary,
        )
        new_query_str = self._parse_llm_output(raw_llm_output)
        if self.verbose:
            print_text(f"> Current query: {query_str}\n", color="yellow")
            print_text(f"> New query: {new_query_str}\n", color="pink")
        return QueryBundle(
            query_str=new_query_str,
            custom_embedding_strs=query_bundle.custom_embedding_strs,
        )

    async def arun(
        self,
        query_bundle_or_str: QueryType,
        metadata: Optional[Dict] = None,
    ) -> QueryBundle:
        """Run query transform asynchronously."""
        metadata = metadata or {}
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(
                query_str=query_bundle_or_str,
                custom_embedding_strs=[query_bundle_or_str],
            )
        else:
            query_bundle = query_bundle_or_str

        return await self._arun(query_bundle, metadata=metadata)


class MultiStepQueryEngineWorkflow(Workflow):
    @staticmethod
    async def combine_queries(
        query_bundle: QueryBundle,
        prev_reasoning: str,
        index_summary: str,
        llm: LLM,
    ) -> QueryBundle:
        """Combine queries using StepDecomposeQueryTransform."""
        transform_metadata = {
            "prev_reasoning": prev_reasoning,
            "index_summary": index_summary,
        }
        return await StepDecomposeQueryTransform(
            llm=llm, step_decompose_query_prompt=QUERY_DECOMPOSE_SM_PROMPT
        ).arun(
            query_bundle, metadata=transform_metadata
        )

    @staticmethod
    def default_stop_fn(stop_dict: Dict) -> bool:
        """Stop function for multistep query combiner."""
        query_bundle = stop_dict.get("query_bundle")
        if query_bundle is None:
            raise ValueError("Response must be provided to stop function.")

        return "none" in query_bundle.query_str.lower()

    @step
    async def query_multistep(
        self, ctx: Context, ev: StartEvent
    ) -> QueryMultiStepEvent:
        """Execute multi-step query process."""
        history: list[dict[str, Any]] = []
        cur_steps = 0

        # use response
        final_response_metadata: Dict[str, Any] = {"sub_qa": [], "source_nodes": []}

        text_chunks = []
        source_nodes = []

        query = ev.get("query")
        await ctx.store.set("query", ev.get("query"))

        llm = Settings.llm
        stop_fn = self.default_stop_fn

        num_steps = ev.get("num_steps")
        query_engine = ev.get("query_engine")
        index_summary = ev.get("index_summary")
        enable_stateful = ev.get("enable_stateful")

        while True:
            if num_steps is not None and cur_steps >= num_steps:
                break

            prev_reasoning = json.dumps(history, ensure_ascii=False)
            updated_query_bundle = await self.combine_queries(
                QueryBundle(query_str=query),
                prev_reasoning,
                index_summary,
                llm,
            )

            # print(
            #     f"Created query for the step - {cur_steps} is: {updated_query_bundle}"
            # )

            stop_dict = {"query_bundle": updated_query_bundle}
            if stop_fn(stop_dict):
                break

            if enable_stateful:
                embedding_query = build_stateful_embedding_query(
                    original_question=query,
                    history=history,
                    step_query=updated_query_bundle.query_str,
                )
            else:
                embedding_query = f"Query: {updated_query_bundle.query_str}"
            updated_query_bundle.custom_embedding_strs = [embedding_query]

            cur_response = await query_engine.aquery(updated_query_bundle)
            answer_obj = parse_answer(cur_response.response)
            step_status = answer_obj["status"]
            step_answer = answer_obj["answer"]
            cur_response.response = step_answer

            # append to response builder
            cur_qa_text = (
                f"\nQuestion: {updated_query_bundle.query_str}\n"
                f"Status: {step_status}\n"
                f"Answer: {cur_response!s}"
            )
            text_chunks.append(cur_qa_text)
            for source_node in cur_response.source_nodes:
                source_nodes.append(source_node)
            # update metadata
            final_response_metadata["sub_qa"].append(
                (updated_query_bundle.query_str, {"status": step_status, "answer": step_answer})
            )

            ctx.write_event_to_stream(SubQaEvent(
                sub_qa=updated_query_bundle,
                cur_response=step_answer or "",
                cur_status=step_status,
                source_nodes=cur_response.source_nodes,
            ))

            history.append({
                "sub_query": updated_query_bundle.query_str,
                "status": step_status,
                "answer": step_answer,
            })
            cur_steps += 1

        nodes = [
            NodeWithScore(node=TextNode(text=text_chunk))
            for text_chunk in text_chunks
        ]
        return QueryMultiStepEvent(
            nodes=nodes,
            source_nodes=source_nodes,
            final_response_metadata=final_response_metadata,
        )

    @step
    async def synthesize(
        self, ctx: Context, ev: QueryMultiStepEvent
    ) -> StopEvent:
        """Synthesize the response."""
        response_synthesizer = get_response_synthesizer(text_qa_template=QA_SM_PROMPT_SEL)
        query = await ctx.store.get("query", default=None)
        final_response = await response_synthesizer.asynthesize(
            query=query,
            nodes=ev.nodes,
            additional_source_nodes=ev.source_nodes,
        )
        answer_obj = parse_answer(final_response.response) or {}
        final_response.response = answer_obj["answer"]
        final_response.metadata = ev.final_response_metadata
        final_response.metadata["final_status"] = answer_obj["status"]

        return StopEvent(result=final_response)
