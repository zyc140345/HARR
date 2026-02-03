import json
from typing import Any

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index_hacked.query_engine import parse_answer


def build_stateful_embedding_query(
    original_question: str,
    history: list[dict[str, Any]],
    step_query: str,
) -> str:
    prev_reasoning = json.dumps(history, ensure_ascii=False)
    return (
        f"Original Question: {original_question}\n"
        f"History: {prev_reasoning}\n"
        f"Query: {step_query}"
    )


class ToolInputSchema(BaseModel):
    """Standard tool schema matching ReActAgent tool invocation."""

    input: str


class StatefulQueryEngineTool(AsyncBaseTool):
    """Query engine tool that builds the embedding query from (original question + history + step query)."""

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        metadata: ToolMetadata,
        original_question: str,
        enable_stateful: bool = True,
        resolve_input_errors: bool = True,
    ) -> None:
        self._query_engine = query_engine
        metadata.fn_schema = ToolInputSchema
        self._metadata = metadata
        self._original_question = original_question
        self._enable_stateful = enable_stateful
        self._history = []
        self._resolve_input_errors = resolve_input_errors

    @property
    def query_engine(self) -> BaseQueryEngine:
        return self._query_engine

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def _build_query_bundle(self, step_query: str) -> QueryBundle:
        if self._enable_stateful:
            embedding_query = build_stateful_embedding_query(
                original_question=self._original_question,
                history=self._history,
                step_query=step_query,
            )
        else:
            embedding_query = f"Query: {step_query}"
        return QueryBundle(query_str=step_query, custom_embedding_strs=[embedding_query])

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        step_query = str(kwargs.get("input", "")).strip()
        query_bundle = self._build_query_bundle(step_query)
        kwargs["input"] = query_bundle

        response = self._query_engine.query(query_bundle)
        answer_obj = parse_answer(response.response)
        response.response = answer_obj["answer"]
        response.metadata["status"] = answer_obj["status"]
        self._history.append({"sub_query": step_query} | answer_obj)

        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.get_name(),
            raw_input=kwargs,
            raw_output=response,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        step_query = str(kwargs.get("input", "")).strip()
        query_bundle = self._build_query_bundle(step_query)
        kwargs["input"] = query_bundle

        response = await self._query_engine.aquery(query_bundle)
        answer_obj = parse_answer(response.response)
        response.response = answer_obj["answer"]
        response.metadata["status"] = answer_obj["status"]
        self._history.append({"sub_query": step_query} | answer_obj)

        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.get_name(),
            raw_input=kwargs,
            raw_output=response,
        )
