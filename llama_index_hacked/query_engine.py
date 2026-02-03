import re
import json
import torch
import torch.nn.functional as F

from json import JSONDecodeError
from typing import List, Optional, Any, Type
from llama_index.core import BasePromptTemplate, Settings
from llama_index.core import instrumentation as instrument
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.query_engine import RetrieverQueryEngine as _RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager, EventPayload, CBEventType
from llama_index.core.response_synthesizers import BaseSynthesizer, ResponseMode, get_response_synthesizer
from prompt import QA_SM_PROMPT_SEL

dispatcher = instrument.get_dispatcher(__name__)


ANSWER_PATTERN = re.compile(
    r"""
    Thought:\s*(?P<thought>[\s\S]*?)
    \n\s*JSON:\s*(?P<json>\{[\s\S]*?})
    (?=\s*$|\n)
    """,
    re.IGNORECASE | re.VERBOSE,
)
DEFAULT_ANSWER = {"status": "AMBIGUOUS", "answer": ""}
VALID_STATUS = {"FOUND", "NOT_FOUND", "AMBIGUOUS"}


def parse_answer(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return DEFAULT_ANSWER.copy()

    text = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "").strip()

    m = ANSWER_PATTERN.search(text)
    if not m:
        return DEFAULT_ANSWER.copy()

    try:
        obj = json.loads(m.group("json"))
    except JSONDecodeError:
        return DEFAULT_ANSWER.copy()

    status = str(obj.get("status", "")).strip().upper()
    if status not in VALID_STATUS:
        return DEFAULT_ANSWER.copy()

    answer = str(obj.get("answer", "")).strip()

    return {"status": status, "answer": answer}


class RetrieverQueryEngine(_RetrieverQueryEngine):
    def __init__(
        self,
        retriever: BaseRetriever,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        similarity_top_k: int = 3,
        temperature: float = 0.7,
    ) -> None:
        super().__init__(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
            callback_manager=callback_manager,
        )
        self._similarity_top_k = similarity_top_k
        self._temperature = temperature
        self._train = False

    @classmethod
    def from_args(
        cls,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        similarity_top_k: int = 3,
        temperature: float = 0.7,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.COMPACT,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        summary_template: Optional[BasePromptTemplate] = None,
        simple_template: Optional[BasePromptTemplate] = None,
        output_cls: Optional[Type[BaseModel]] = None,
        use_async: bool = False,
        streaming: bool = False,
        **kwargs: Any,
    ) -> "RetrieverQueryEngine":
        """
        Initialize a RetrieverQueryEngine object.

        Args:
            retriever (BaseRetriever): A retriever object.
            llm (Optional[LLM]): An instance of an LLM.
            response_synthesizer (Optional[BaseSynthesizer]): An instance of a response
                synthesizer.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of
                node postprocessors.
            callback_manager (Optional[CallbackManager]): A callback manager.
            similarity_top_k (Optional[int]): The number of most similar nodes to retrieve.
            temperature (Optional[float]): The temperature for sampling.
            response_mode (ResponseMode): A ResponseMode object.
            text_qa_template (Optional[BasePromptTemplate]): A BasePromptTemplate
                object.
            refine_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.
            summary_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.
            simple_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.
            output_cls (Optional[Type[BaseModel]]): The pydantic model to pass to the
                response synthesizer.
            use_async (bool): Whether to use async.
            streaming (bool): Whether to use streaming.

        """
        llm = llm or Settings.llm

        response_synthesizer = response_synthesizer or get_response_synthesizer(
            llm=llm,
            text_qa_template=text_qa_template or QA_SM_PROMPT_SEL,
            refine_template=refine_template,
            summary_template=summary_template,
            simple_template=simple_template,
            response_mode=response_mode,
            output_cls=output_cls,
            use_async=use_async,
            streaming=streaming,
        )

        callback_manager = callback_manager or Settings.callback_manager

        return cls(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager,
            node_postprocessors=node_postprocessors,
            similarity_top_k=similarity_top_k,
            temperature=temperature,
        )

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    @dispatcher.span
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = self.retrieve(query_bundle)
            if self._train:
                nodes, additional_source_nodes = self._sample(nodes)
            else:
                additional_source_nodes = nodes[self._similarity_top_k:]
                nodes = nodes[:self._similarity_top_k]
            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
                additional_source_nodes=additional_source_nodes,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @dispatcher.span
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = await self.aretrieve(query_bundle)
            if self._train:
                nodes, additional_source_nodes = self._sample(nodes)
            else:
                additional_source_nodes = nodes[self._similarity_top_k:]
                nodes = nodes[:self._similarity_top_k]
            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
                additional_source_nodes=additional_source_nodes,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    def _sample(self, nodes):
        with torch.no_grad():
            logits = torch.tensor([node.score for node in nodes])
            logits /= self._temperature + 1e-7
            prob_dist = F.softmax(logits, dim=-1)
            index = torch.multinomial(prob_dist, num_samples=self._similarity_top_k, replacement=False)
            index_set = set(index.tolist())
        additional_nodes = [node for i, node in enumerate(nodes) if i not in index_set]
        nodes = [nodes[i] for i in index.tolist()]
        return nodes, additional_nodes
