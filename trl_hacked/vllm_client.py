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

import atexit
import logging
import socket
import time
import torch

from typing import Optional, Tuple
from urllib.parse import urlparse
from torch import nn
from trl.import_utils import is_vllm_available
from trl_hacked.trainer.utils import is_httpx_available


if is_httpx_available():
    import httpx
    from httpx import ConnectError


if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup


logger = logging.getLogger(__name__)


class VLLMClient:
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        base_url (`str` or `None`, *optional*, defaults to `None`):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `host` and `server_port` are
            ignored.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server. Ignored if `base_url` is provided.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server. Ignored if `base_url` is provided.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.init_communicator()
        >>> client.update_model_params(model)
        ```

        There are several ways to initialize the client:

        ```python
        VLLMClient(base_url="http://localhost:8000")
        VLLMClient(base_url="http://192.168.1.100:8000")
        VLLMClient(host="localhost", server_port=8000)
        VLLMClient(host="192.168.1.100", server_port=8000)
        ```
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not is_httpx_available():
            raise ImportError("httpx is not installed. Please install it with `pip install httpx`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        self.session = httpx.Client(timeout=connection_timeout)
        self.asession = httpx.AsyncClient(timeout=connection_timeout)

        if base_url is not None:
            # Parse the base_url to extract host and port
            parsed_url = urlparse(base_url)
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"
        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"{self.base_url}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = httpx.get(url)
            except httpx.RequestError as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.base_url} after {total_timeout} seconds. Make "
                        "sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    if "X-Forwarded-For" in response.headers:
                        self.host = response.headers["X-Forwarded-For"]
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
        logprobs: Optional[int] = None,
        allowed_token_ids: Optional[list[int]] = None
    ) -> Tuple[list[list[int]], Optional[list[list[dict[int, float]]]]]:
        """
        Generates model completions for the provided prompts synchronously.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            presence_penalty (`float`, *optional*, defaults to `0.0`):
                Parameter for presence penalty. 0.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.
            logprobs (`int` or `None`, *optional*, defaults to `None`):
                Number of log probabilities to return per output token.
            allowed_token_ids (`list[int]` or `None`, *optional*, defaults to `None`):
                If provided, the engine will construct a logits
                processor which only retains scores for the given token ids.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
            `Optional[list[list[dict[int, float]]]]`:
                List of lists of dicts mapping token IDs to their log probabilities at each generation step.
        """
        url = f"{self.base_url}/generate/"
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
                "logprobs": logprobs,
                "allowed_token_ids": allowed_token_ids
            },
        )
        if response.status_code == 200:
            response_json = response.json()
            completion_ids = response_json["completion_ids"]
            logprobs = [
                [{int(token_id): logprob for token_id, logprob in logprobs.items()} for logprobs in all_logprobs]
                for all_logprobs in response_json["logprobs"]
            ]
            return completion_ids, logprobs
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    async def agenerate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
        logprobs: Optional[int] = None,
        allowed_token_ids: Optional[list[int]] = None
    ) -> Tuple[list[list[int]], Optional[list[list[dict[int, float]]]]]:
        """
        Generates model completions for the provided prompts asynchronously.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            presence_penalty (`float`, *optional*, defaults to `0.0`):
                Parameter for presence penalty. 0.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.
            logprobs (`int` or `None`, *optional*, defaults to `None`):
                Number of log probabilities to return per output token.
            allowed_token_ids (`list[int]` or `None`, *optional*, defaults to `None`):
                If provided, the engine will construct a logits
                processor which only retains scores for the given token ids.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
            `Optional[list[list[dict[int, float]]]]`:
                List of lists of dicts mapping token IDs to their log probabilities at each generation step.
        """
        url = f"{self.base_url}/generate/"
        response = await self.asession.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
                "logprobs": logprobs,
                "allowed_token_ids": allowed_token_ids
            },
        )
        if response.status_code == 200:
            response_json = response.json()
            completion_ids = response_json["completion_ids"]
            logprobs = [
                [{int(token_id): logprob for token_id, logprob in logprobs.items()} for logprobs in all_logprobs]
                for all_logprobs in response_json["logprobs"]
            ]
            return completion_ids, logprobs
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def embed(self, prompts: list[str]) -> list[list[float]]:
        """
        Generate an embedding vector for each prompt synchronously.

        Args:
            prompts: `list[str]`:
                List of prompts for the model to generate embeddings.

        Returns:
            `list[list[float]]`:
                List of lists of embeddings for each prompt.
        """

        while True:
            try:
                url = f"{self.base_url}/embed/"
                response = self.session.post(
                    url,
                    json={
                        "prompts": prompts,
                    },
                )
                if response.status_code == 200:
                    return response.json()["embeddings"]
                else:
                    raise Exception(f"Request failed: {response.status_code}, {response.text}")
            except httpx.ReadError:
                logger.warning("ReadError occurred during embedding. Retrying...")

    async def aembed(self, prompts: list[str]) -> list[list[float]]:
        """
        Generate an embedding vector for each prompt asynchronously.

        Args:
            prompts: `list[str]`:
                List of prompts for the model to generate embeddings.

        Returns:
            `list[list[float]]`:
                List of lists of embeddings for each prompt.
        """

        while True:
            try:
                url = f"{self.base_url}/embed/"
                response = await self.asession.post(
                    url,
                    json={
                        "prompts": prompts,
                    },
                )
                if response.status_code == 200:
                    return response.json()["embeddings"]
                else:
                    raise Exception(f"Request failed: {response.status_code}, {response.text}")
            except httpx.ReadError:
                logger.warning("ReadError occurred during embedding. Retrying...")

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """
        # Get the world size from the server
        url = f"{self.base_url}/get_world_size/"
        response = httpx.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process

        # Initialize weight update group
        url = f"{self.base_url}/init_communicator/"
        # In the server side, the host is set to 0.0.0.0
        response = self.session.post(url, json={"host": "0.0.0.0", "port": self.group_port, "world_size": world_size})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Brief delay to allow server initialization. While not strictly required (client socket will retry on
        # connection failure), this prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=0)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.base_url}/update_named_param/"
        response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"{self.base_url}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"{self.base_url}/close_communicator/"

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")
