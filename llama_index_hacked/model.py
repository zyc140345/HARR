import asyncio
import math
import socket
import uuid

from typing import List, Optional, Any
from vllm import AsyncLLMEngine, PoolingParams
from vllm.inputs.data import TokensPrompt
from transformers import AutoTokenizer, Qwen2TokenizerFast
from llama_index.core.bridge.pydantic import Field
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from trl_hacked.vllm_client import VLLMClient
from prompt import STATEFUL_RETRIEVAL_INSTRUCTION


class Qwen3EmbeddingServer(BaseEmbedding):
    query_prompt: str = Field(default="Instruct: {}\n{}")

    client: VLLMClient = Field()
    instruction: str = Field()

    def __init__(
        self,
        base_url: str,
        instruction: str = STATEFUL_RETRIEVAL_INSTRUCTION,
        init_communicator: bool = False,
        **kwargs: Any,
    ) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", 0))
            group_port = int(sock.getsockname()[1])

        client = VLLMClient(
            base_url=base_url,
            group_port=group_port,
            connection_timeout=240.0
        )
        if init_communicator:
            client.init_communicator()

        super().__init__(
            client=client,
            instruction=instruction,
            **kwargs
        )

    @classmethod
    def class_name(cls) -> str:
        return "Qwen3Embedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        # For a query, use the "query" prompt for better retrieval performance
        task_prompt = self._get_detailed_instruct(query)
        embeddings = await self.client.aembed([task_prompt])
        return embeddings[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        # For a document, use default encoding
        embeddings = await self.client.aembed([text])
        return embeddings[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # For documents, use default encoding
        embeddings = await self.client.aembed(texts)
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        # For a query, use the "query" prompt for better retrieval performance
        task_prompt = self._get_detailed_instruct(query)
        return self.client.embed([task_prompt])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        # For a document, use default encoding
        return self.client.embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # For documents, use default encoding
        return self.client.embed(texts)

    def _get_detailed_instruct(self, query: str) -> str:
        return self.query_prompt.format(self.instruction, query)


class Qwen3EmbeddingColocate(BaseEmbedding):
    query_prompt: str = Field(default="Instruct: {}\n{}")

    llm_engine: AsyncLLMEngine = Field()
    loop: asyncio.AbstractEventLoop = Field()
    instruction: str = Field()

    def __init__(
        self,
        llm_engine: AsyncLLMEngine,
        loop: asyncio.AbstractEventLoop,
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            llm_engine=llm_engine,
            loop=loop,
            instruction=instruction,
            **kwargs
        )

    @classmethod
    def class_name(cls) -> str:
        return "Qwen3Embedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        # For a query, use the "query" prompt for better retrieval performance
        task_prompt = self._get_detailed_instruct(query)
        pooling_params = PoolingParams()
        request_id = str(uuid.uuid4())

        generator = self.llm_engine.encode(task_prompt, pooling_params, request_id)
        final_output = None
        async for request_output in generator:
            final_output = request_output

        return final_output.outputs[0].embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        # For a document, use default encoding
        pooling_params = PoolingParams()
        request_id = str(uuid.uuid4())

        generator = self.llm_engine.encode(text, pooling_params, request_id)
        final_output = None
        async for request_output in generator:
            final_output = request_output

        return final_output.outputs[0].embedding

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # For documents, use default encoding
        tasks = [self._aget_text_embedding(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.loop.run_until_complete(self._aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.loop.run_until_complete(self._aget_text_embedding(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.loop.run_until_complete(self._aget_text_embeddings(texts))

    def _get_detailed_instruct(self, query: str) -> str:
        return self.query_prompt.format(self.instruction, query)


class Qwen3Reranker(BaseNodePostprocessor):
    system_prompt: str = Field(default=(
        "Judge whether the Document meets the requirements based on the Query and "
        "the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    ))
    temperature: float = Field(default=0)
    max_tokens: int = Field(default=1)
    logprobs: int = Field(default=20)

    client: VLLMClient = Field()
    tokenizer: Qwen2TokenizerFast = Field()
    max_length: int = Field()
    suffix_tokens: List[int] = Field()
    true_token: int = Field()
    false_token: int = Field()
    allowed_token_ids: List[int] = Field()
    top_n: int = Field()
    instruction: str = Field()

    def __init__(
        self,
        base_url: str,
        group_port: int = 51217,
        max_length: int = 1024,
        top_n: int = 3,
        instruction: str = 'Given a web search query, retrieve relevant passages that answer the query',
        init_communicator: bool = False,
    ):
        # Client configuration
        client = VLLMClient(
            base_url=base_url,
            group_port=group_port,
            connection_timeout=240.0
        )
        if init_communicator:
            client.init_communicator()

        # Tokenizer configuration
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.apply_chat_template()

        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

        true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
        false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
        allowed_token_ids = [true_token, false_token]

        super().__init__(
            client=client,
            tokenizer=tokenizer,
            max_length=max_length,
            suffix_tokens=suffix_tokens,
            true_token=true_token,
            false_token=false_token,
            allowed_token_ids=allowed_token_ids,
            top_n=top_n,
            instruction=instruction
        )

    @classmethod
    def class_name(cls) -> str:
        return "Qwen3Reranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        # Format inputs for reranker
        pairs = [(query_bundle.query_str, node.text) for node in nodes]
        # Process inputs and compute scores
        inputs = self._process_inputs(pairs)
        scores = self._compute_logits(inputs)
        # Combine documents with scores and sort by score (descending)
        doc_scores = [NodeWithScore(node=node.node, score=score) for node, score in zip(nodes, scores)]
        doc_scores.sort(key=lambda x: x.score, reverse=True)
        return doc_scores[:self.top_n]

    def _format_instruction(self, query, doc):
        text = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return text

    def _process_inputs(self, pairs):
        messages = [self._format_instruction(query, doc) for query, doc in pairs]
        messages = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:self.max_length - len(self.suffix_tokens)] + self.suffix_tokens for ele in messages]
        messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
        return messages

    def _compute_logits(self, messages):
        token_ids, logprobs = self.client.generate(
            messages, temperature=self.temperature, max_tokens=self.max_tokens,
            logprobs=self.logprobs, allowed_token_ids=self.allowed_token_ids
        )
        scores = []
        for logits in logprobs:
            final_logits = logits[-1]
            if self.true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self.true_token]
            if self.false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self.false_token]
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)
        return scores


def get_remote_llm(
    api_base: str,
    api_key: str,
    model: str,
    context_window: int,
    enable_thinking: bool = False,
    seed: int = None
):
    if 'qwen3' in model.lower():
        additional_kwargs = {
            "extra_body": {
                "enable_thinking": enable_thinking,
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            }
        }
    elif 'gpt-oss' in model.lower():
        additional_kwargs = {
            "extra_body": {"reasoning_effort": "medium" if enable_thinking else "low"}
        }
    else:
        additional_kwargs = {}

    if seed is not None:
        additional_kwargs["seed"] = seed

    return OpenAILike(
        model=model,
        api_base=api_base,
        api_key=api_key,
        context_window=context_window,
        is_chat_model=True,
        is_function_calling_model=False,
        additional_kwargs=additional_kwargs
    )
