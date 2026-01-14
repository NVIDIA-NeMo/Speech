# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import socket
import subprocess
import time
import uuid
from threading import Thread
from typing import AsyncGenerator, List, Mapping, Optional

import psutil
import requests
from jinja2.exceptions import TemplateError
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import APITimeoutError, AsyncStream, BadRequestError
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from transformers import AsyncTextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer
from vllm.config import ModelConfig as vllmModelConfig

DEFAULT_GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}


class LLMUtilsMixin:
    """Utils for local LLM services."""

    def _maybe_add_user_message(self, messages: List[ChatCompletionMessageParam]) -> List[ChatCompletionMessageParam]:
        """
        Some LLMs like "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" requires a user turn after the system prompt,
        this function is used to add a dummy user turn if the system prompt is followed by an assistant turn.
        """
        if len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "assistant":
            message = {"role": "user", "content": "Hi"}
            messages.insert(1, message)
        elif len(messages) == 1 and messages[0]["role"] == "system":
            messages.append({"role": "user", "content": "Hi"})
        return messages

    def _maybe_merge_consecutive_user_turns(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        """
        Merge consecutive user turns into a single turn,
        since some LLMs like "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" do not support consecutive user turns.
        """
        if not messages:
            return messages

        merged_messages = []

        user_content = ""
        for message in messages:
            role = message["role"]
            if role != "user":
                # check if there's any preceeding user content, add them first
                if user_content:
                    merged_messages.append({"role": "user", "content": user_content})
                    user_content = ""
                merged_messages.append(message)
            else:
                if user_content:
                    user_content += "; " + message["content"]
                else:
                    user_content = message["content"]

        # add the last user content
        if user_content:
            merged_messages.append({"role": "user", "content": user_content})

        return merged_messages


class HuggingFaceLLMLocalService(LLMUtilsMixin):
    """
    HuggingFace LLM local service.
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        thinking_budget: int = 0,
        generation_kwargs: dict = None,
        apply_chat_template_kwargs: dict = None,
    ):
        self.device = device
        self.dtype = dtype
        self.thinking_budget = thinking_budget
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map=device, dtype=dtype, trust_remote_code=True
        )  # type: AutoModelForCausalLM

        self.generation_kwargs = generation_kwargs if generation_kwargs else DEFAULT_GENERATION_KWARGS
        logger.debug(f"LLM generation kwargs: {self.generation_kwargs}")

        self.apply_chat_template_kwargs = apply_chat_template_kwargs if apply_chat_template_kwargs else {}
        if "tokenize" in self.apply_chat_template_kwargs:
            if self.apply_chat_template_kwargs["tokenize"] is not False:
                logger.warning(
                    f"Found `tokenize=True` in apply_chat_template_kwargs={self.apply_chat_template_kwargs},"
                    "it will be ignored and forced to `False`"
                )
            self.apply_chat_template_kwargs.pop("tokenize")

        logger.debug(f"LLM apply_chat_template kwargs: {self.apply_chat_template_kwargs}")

    def _apply_chat_template(self, messages: List[ChatCompletionMessageParam]) -> str:
        """
        Apply the chat template to the messages.
        """
        return self.tokenizer.apply_chat_template(messages, tokenize=False, **self.apply_chat_template_kwargs)

    def _get_prompt_from_messages(self, messages: List[ChatCompletionMessageParam]) -> str:
        """
        Get the formatted prompt from the conversation history messages.
        This function also tries to fix the messages if the LLM cannot handle consecutive turns of the same role,
        or requires a user turn after the system prompt.
        """
        try:
            prompt = self._apply_chat_template(messages)
            return prompt
        except TemplateError as e:
            logger.warning(f"Got TemplateError: {e}.")

        logger.debug(f"Input LLM messages: {messages}")
        if len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "assistant":
            logger.warning("Trying to fix by adding dummy user message after system prompt...")
            try:
                messages = self._maybe_add_user_message(messages)
                logger.debug(f"LLM messages after adding dummy user message: {messages}")
                prompt = self._apply_chat_template(messages)
                return prompt
            except TemplateError as e:
                logger.warning(f"Got TemplateError: {e}. Trying to fix by merging consecutive turns if possible.")

        try:
            new_messages = self._maybe_merge_consecutive_user_turns(messages)
            logger.debug(f"LLM messages after merging consecutive user turns: {new_messages}")
            prompt = self._apply_chat_template(new_messages)
            # Update the messages in place if successful
            messages.clear()
            messages.extend(new_messages)
            return prompt
        except Exception as e:
            logger.warning(f"Got Exception: {e}, messages: {messages}")
            raise e

    async def generate_stream(
        self, messages: List[ChatCompletionMessageParam], **kwargs
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Generate a stream of chat completion chunks from the messages.
        """
        # Convert messages to prompt format
        prompt = self._get_prompt_from_messages(messages)

        logger.debug(f"LLM prompt: {prompt}")

        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)

        # Generate with streaming
        streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            **self.generation_kwargs,
        }

        # Start generation in background
        thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        # Stream the output
        async for text in streamer:
            # logger.debug(f"Streamer yielded text: {text}")
            chunk = ChatCompletionChunk(
                id="hf-" + str(uuid.uuid4()),
                choices=[{"delta": {"content": text}, "finish_reason": None, "index": 0}],
                created=int(time.time()),
                model=self.model.config._name_or_path,
                object="chat.completion.chunk",
            )
            yield chunk


class HuggingFaceLLMService(OpenAILLMService):
    """
    LLM service that hosts a HuggingFace model.
    """

    def __init__(
        self,
        *,
        model: str = "google/gemma-7b-it",
        device: str = "cuda",
        dtype: str = "bfloat16",
        thinking_budget: int = 0,
        generation_kwargs: dict = None,
        apply_chat_template_kwargs: dict = None,
        **kwargs,
    ):
        self._model_name = model
        self._device = device
        self._dtype = dtype
        self._thinking_budget = thinking_budget
        self._generation_kwargs = generation_kwargs if generation_kwargs is not None else DEFAULT_GENERATION_KWARGS
        self._apply_chat_template_kwargs = apply_chat_template_kwargs if apply_chat_template_kwargs is not None else {}
        super().__init__(model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """
        Create a HuggingFaceLLMLocalService client.
        """
        return HuggingFaceLLMLocalService(
            model=self._model_name,
            device=self._device,
            dtype=self._dtype,
            thinking_budget=self._thinking_budget,
            generation_kwargs=self._generation_kwargs,
            apply_chat_template_kwargs=self._apply_chat_template_kwargs,
        )

    async def _process_context(self, context: OpenAILLMContext):
        """Process a context through the LLM and push text frames.

        Args:
            context (OpenAILLMContext): The context to process, containing messages
                and other information needed for the LLM interaction.
        """
        await self.push_frame(LLMFullResponseStartFrame())
        cumulative_text = ""
        try:
            await self.start_ttfb_metrics()
            messages = context.get_messages()
            async for chunk in self._client.generate_stream(messages):
                if chunk.choices[0].delta.content:
                    await self.stop_ttfb_metrics()
                    text = chunk.choices[0].delta.content
                    cumulative_text += text
                    frame = LLMTextFrame(text)
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"Error in _process_context: {e}", exc_info=True)
            raise
        finally:
            cumulative_text = " ".join(cumulative_text.split()).strip()
            if not cumulative_text:
                logger.warning(f"LLM response is empty for context: {context}")
            await self.push_frame(LLMFullResponseEndFrame())

    async def get_chat_completions(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Create a streaming chat completion using HuggingFace model.

        Args:
            context (OpenAILLMContext): The context object containing tools configuration
                and other settings for the chat completion.
            messages (List[ChatCompletionMessageParam]): The list of messages comprising
                the conversation history and current request.

        Returns:
            AsyncGenerator[ChatCompletionChunk]: A streaming response of chat completion
                chunks that can be processed asynchronously.
        """
        messages = params_from_context["messages"]

        return self._client.generate_stream(messages)


class VLLMService(OpenAILLMService, LLMUtilsMixin):
    """
    LLM service that hosts a vLLM server.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        api_key="None",
        base_url="http://localhost:8000/v1",
        organization="None",
        project="None",
        default_headers: Optional[Mapping[str, str]] = None,
        params: Optional[OpenAILLMService.InputParams] = None,
        thinking_budget: int = 0,
        start_vllm_on_init: bool = False,
        vllm_server_params: Optional[str] = None,
        vllm_server_max_wait_time: int = 3600,  # 1 hour max wait time
        vllm_server_check_interval: int = 5,  # check server every 5 seconds
        **kwargs,
    ):
        self._device = device
        self._vllm_server_max_wait_time = vllm_server_max_wait_time
        self._vllm_server_check_interval = vllm_server_check_interval
        if start_vllm_on_init:
            base_url = self._start_vllm_server(model, vllm_server_params, base_url)

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
            params=params,
            **kwargs,
        )
        self._thinking_budget = thinking_budget
        self._vllm_server_params = vllm_server_params
        self._start_vllm_on_init = start_vllm_on_init

        # TODO: handle thinking budget
        logger.info(
            f"VLLMService initialized with model: {model}, api_key: {api_key}, base_url: {base_url},"
            f"params: {params}, thinking_budget: {thinking_budget}"
        )

    def _start_vllm_server(
        self, model: str, vllm_server_params: Optional[str] = None, base_url: Optional[str] = None
    ) -> str:
        """
        Start a vllm server and return the base url.
        """

        requested_port = None
        # If base_url is provided, extract port from it
        if base_url:
            try:
                # Extract port from base_url like "http://localhost:8003/v1"
                from urllib.parse import urlparse

                parsed_url = urlparse(base_url)
                if parsed_url.port:
                    requested_port = parsed_url.port
            except Exception as e:
                logger.warning(
                    f"Could not parse port from base_url {base_url}: {e}, using port from vllm_server_params"
                )

        # Parse port from vllm_server_params, default to 8000
        if vllm_server_params:
            params_list = vllm_server_params.split()
            for i, param in enumerate(params_list):
                if param == "--port" and i + 1 < len(params_list):
                    try:
                        param_port = int(params_list[i + 1])
                        if requested_port is None:
                            requested_port = param_port
                        else:
                            if param_port != requested_port:
                                logger.warning(
                                    f"Port {param_port} from vllm_server_params is different from base_url port"
                                    f"{requested_port}, using new port {param_port}"
                                )
                                requested_port = param_port
                        break
                    except ValueError:
                        logger.warning(f"Invalid port number: {params_list[i + 1]}, using default 8000")

        if requested_port is None:
            # try to use default port
            requested_port = 8000

        def find_available_port(start_port: int) -> int:
            """Find an available port starting from start_port"""
            for port in range(start_port, start_port + 100):  # Try up to 100 ports
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('localhost', port))
                        return port
                except OSError:
                    continue
            raise RuntimeError(f"Could not find an available port starting from {start_port}")

        def get_pid_on_port(port: int) -> Optional[int]:
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    return conn.pid
            return None

        def check_server_model(port: int, verbose: bool = False) -> tuple[bool, str]:
            """Check if server is running on port and return (is_running, model_name)"""
            try:
                response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
                if response.status_code == 200:
                    # get the PID for the server process
                    pid = get_pid_on_port(port)
                    if pid is not None and verbose:
                        logger.warning(
                            f"Found vLLM server process (PID: {pid}) on port {port}, you can use `lsof -i :{port}`"
                            "to find the process and kill it if you want to start a new server."
                        )
                    models_data = response.json()
                    if "data" in models_data and models_data["data"]:
                        served_model = models_data["data"][0].get("id", "")
                        return True, served_model
                    return True, ""
                return False, ""
            except (requests.exceptions.RequestException, requests.exceptions.Timeout):
                return False, ""

        # First, check if vLLM server is already running on the requested port
        is_running, served_model = check_server_model(requested_port, verbose=True)
        if is_running:
            if served_model == model:
                final_base_url = f"http://localhost:{requested_port}/v1"
                logger.info(f"vLLM server is already running at {final_base_url} with the correct model: {model}")
                return final_base_url
            else:
                logger.warning(
                    f"vLLM server on port {requested_port} is serving model '{served_model}' but we need '{model}'."
                    "Finding new port..."
                )

        # Find an available port for our model
        port = find_available_port(requested_port)
        if port != requested_port:
            logger.info(f"Using port {port} instead of requested port {requested_port}")

        final_base_url = f"http://localhost:{port}/v1"

        # Check if there's already a vLLM process running on the same port and model
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any('vllm' in arg and 'serve' in arg for arg in proc.info['cmdline']):
                    # Check if this process is using the same port and model
                    cmdline_str = ' '.join(proc.info['cmdline'])
                    if f"--port {port}" in cmdline_str:
                        # Extract the model from the command line
                        cmdline_parts = proc.info['cmdline']
                        model_index = -1
                        for i, arg in enumerate(cmdline_parts):
                            if arg == "serve" and i + 1 < len(cmdline_parts):
                                model_index = i + 1
                                break

                        if model_index != -1 and model_index < len(cmdline_parts):
                            running_model = cmdline_parts[model_index]
                            if running_model == model:
                                logger.info(
                                    f"Found existing vLLM server process (PID: {proc.info['pid']}) on port {port}"
                                    f"serving model {model}"
                                )
                                # Wait a bit and check if it's responding
                                time.sleep(2)
                                is_running, served_model = check_server_model(port)
                                if is_running and served_model == model:
                                    logger.info(
                                        f"Existing vLLM server is responding at {final_base_url} with correct model"
                                    )
                                    return final_base_url
                                else:
                                    logger.warning(
                                        f"Existing vLLM process found on port {port} but not responding correctly,"
                                        "will start new server"
                                    )
                            else:
                                logger.info(
                                    f"Found vLLM process on port {port} but serving different model '{running_model}'"
                                    f"(need '{model}'). Will start new server."
                                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # Build the command with the determined port
        cmd_parts = ["vllm", "serve", model]

        # Parse and modify vllm_server_params to use the correct port
        if vllm_server_params:
            # parse the vllm_server_params and add the port to the command
            params_list = vllm_server_params.split()
            modified_params = []
            i = 0
            while i < len(params_list):
                if params_list[i] == "--port" and i + 1 < len(params_list):
                    # Replace the port with our determined port
                    modified_params.extend(["--port", str(port)])
                    i += 2  # Skip the original port value
                else:
                    modified_params.append(params_list[i])
                    i += 1
            cmd_parts.extend(modified_params)
        else:
            # Add port if vllm_server_params is not provided
            cmd_parts.extend(["--port", str(port)])

        logger.info(f"Starting vLLM server with command: {' '.join(cmd_parts)}")
        logger.warning("It will take a while to download the model if it's not already downloaded.")
        # Set up environment variables for device configuration
        env = os.environ.copy()
        if self._device and self._device != "cpu":
            # Extract CUDA device number if it's in format "cuda:0", "cuda:1", etc.
            if self._device.startswith("cuda:"):
                device_id = self._device.split(":")[1]
                env["CUDA_VISIBLE_DEVICES"] = device_id
                logger.info(f"Setting CUDA_VISIBLE_DEVICES={device_id}")
            elif self._device == "cuda":
                # Use default CUDA device (don't set CUDA_VISIBLE_DEVICES)
                logger.info("Using default CUDA device")
            else:
                # For other device strings, try to extract device number
                logger.warning(f"Unknown device format: {self._device}, using as-is")
                env["CUDA_VISIBLE_DEVICES"] = self._device
        elif self._device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("Setting CUDA_VISIBLE_DEVICES='' to use CPU")

        try:
            # Start the vLLM server process with environment variables
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                preexec_fn=os.setsid if os.name != 'nt' else None,  # Create new process group
            )

            # Store the process for potential cleanup later
            self._vllm_process = process

            # Wait for server to start up
            max_wait_time = self._vllm_server_max_wait_time
            check_interval = self._vllm_server_check_interval
            waited_time = 0

            logger.info(f"Waiting for vLLM server to start on port {port}...")
            while waited_time < max_wait_time:
                is_running, served_model = check_server_model(port)
                if is_running and served_model == model:
                    logger.info(f"vLLM server started successfully at {final_base_url} serving model: {model}")
                    return final_base_url
                elif is_running and served_model != model:
                    logger.warning(
                        f"vLLM server started but serving wrong model '{served_model}' instead of '{model}'."
                        "Continuing to wait..."
                    )

                # Check if process is still running
                if process.poll() is not None:
                    # Process has terminated
                    stdout, stderr = process.communicate()
                    logger.error(f"vLLM server process terminated unexpectedly. stdout: {stdout}, stderr: {stderr}")
                    raise RuntimeError(f"Failed to start vLLM server: {stderr}")

                time.sleep(check_interval)
                waited_time += check_interval
                logger.debug(f"Still waiting for vLLM server on port {port}... ({waited_time}s)")

            # If we get here, server didn't start in time
            logger.error(f"vLLM server failed to start within {max_wait_time} seconds on port {port}")
            process.terminate()
            raise RuntimeError(f"vLLM server failed to start within {max_wait_time} seconds on port {port}")

        except FileNotFoundError:
            logger.error("vLLM not found. Please install vLLM: pip install vllm")
            raise RuntimeError("vLLM not found. Please install vLLM: pip install vllm")
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            self._stop_vllm_server()
            raise e

    def _stop_vllm_server(self):
        """Stop the vLLM server process if it's running."""
        if hasattr(self, '_vllm_process') and self._vllm_process:
            logger.info(f"Stopping vLLM server process {self._vllm_process.pid}")
            self._vllm_process.terminate()

    async def stop(self, frame: EndFrame):
        """Stop the LLM service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        self._stop_vllm_server()

    async def cancel(self, frame: CancelFrame):
        """Cancel the LLM service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        self._stop_vllm_server()

    async def get_chat_completions(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> AsyncStream[ChatCompletionChunk]:
        """Get streaming chat completions from OpenAI API.

        Args:
            context: The LLM context containing tools and configuration.
            messages: List of chat completion messages to send.

        Returns:
            Async stream of chat completion chunks.
        """

        params = self.build_chat_completion_params(params_from_context)
        messages = params_from_context["messages"]
        if self._retry_on_timeout:
            try:
                chunks = await asyncio.wait_for(
                    self._get_response_from_client(messages, params), timeout=self._retry_timeout_secs
                )
                return chunks
            except (APITimeoutError, asyncio.TimeoutError):
                # Retry, this time without a timeout so we get a response
                logger.debug(f"{self}: Retrying chat completion due to timeout")
                chunks = await self._get_response_from_client(messages, params)
                return chunks
        else:
            chunks = await self._get_response_from_client(messages, params)
            return chunks

    async def _get_response_from_client(
        self, messages: List[ChatCompletionMessageParam], params: dict
    ) -> AsyncStream[ChatCompletionChunk]:
        """Get a response from the client."""

        try:
            chunks = await self._client.chat.completions.create(**params)
        except BadRequestError as e:
            logger.error(f"Error in _get_response_from_client: {e}, trying to fix...")
            logger.debug(f"LLM messages before fixing: {messages}")
            messages = self._maybe_add_user_message(messages)
            messages = self._maybe_merge_consecutive_user_turns(messages)
            logger.debug(f"LLM messages after fixing: {messages}")
            params["messages"] = messages
            chunks = await self._client.chat.completions.create(**params)

        return chunks


def get_llm_service_from_config(config: DictConfig) -> OpenAILLMService:
    """Get an LLM service from the configuration."""
    backend = config.type

    # If backend is "auto", try to detect the best backend
    if backend == "auto":
        model_name = config.get("model")
        if not model_name:
            raise ValueError("Model name is required for LLM")

        try:
            _ = vllmModelConfig(model_name, trust_remote_code=True)
            backend = "vllm"
            logger.info(f"Auto-detected vLLM as the best backend for model {model_name}")
        except Exception as e:
            logger.info(
                f"The LLM doesn't seem to be supported by vLLM yet (error: {e}), using HuggingFace as the backend"
                f"for model: {model_name}. If you are sure that the LLM is supported by vLLM, you can set `type: vllm`"
                "in the config file to force using vLLM."
            )
            backend = "hf"

    assert backend in [
        "hf",
        "vllm",
        "auto",
    ], f"Invalid backend: {backend}, only `hf`, `vllm`, and `auto` are supported."

    if backend == "hf":
        llm_model = config.model
        llm_device = config.device
        llm_dtype = config.dtype
        llm_generation_kwargs = config.get("generation_kwargs", {})
        if llm_generation_kwargs is not None:
            llm_generation_kwargs = OmegaConf.to_container(llm_generation_kwargs, resolve=True)
        llm_apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", None)
        if llm_apply_chat_template_kwargs is not None:
            llm_apply_chat_template_kwargs = OmegaConf.to_container(llm_apply_chat_template_kwargs, resolve=True)
        llm_thinking_budget = config.get("thinking_budget", 0)
        return HuggingFaceLLMService(
            model=llm_model,
            device=llm_device,
            dtype=llm_dtype,
            generation_kwargs=llm_generation_kwargs,
            apply_chat_template_kwargs=llm_apply_chat_template_kwargs,
            thinking_budget=llm_thinking_budget,
        )
    elif backend == "vllm":
        llm_model = config.get("model", "vllm_server")
        llm_api_key = config.get("api_key", "None")
        llm_base_url = config.get("base_url", "http://localhost:8000/v1")
        llm_organization = config.get("organization", "None")
        llm_project = config.get("project", "None")
        llm_default_headers = config.get("default_headers", None)
        llm_params = config.get("vllm_generation_params", None)
        llm_dtype = config.dtype
        vllm_server_params = config.get("vllm_server_params", None)
        if vllm_server_params is not None:
            if "dtype" not in vllm_server_params:
                vllm_server_params = f"--dtype {llm_dtype} {vllm_server_params}"
                logger.info(f"Adding dtype {llm_dtype} to vllm_server_params: {vllm_server_params}")
        if llm_params is not None:
            # cast into OpenAILLMService.InputParams object
            llm_params = OmegaConf.to_container(llm_params, resolve=True)
            extra = llm_params.get("extra", None)
            # ensure extra is a dictionary
            if extra is None:
                llm_params["extra"] = {}
            elif not isinstance(extra, dict):
                raise ValueError(f"extra must be a dictionary, got {type(extra)}")
            llm_params = OpenAILLMService.InputParams(**llm_params)
        else:
            llm_params = OpenAILLMService.InputParams()
        llm_thinking_budget = config.get("thinking_budget", 0)
        return VLLMService(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
            organization=llm_organization,
            project=llm_project,
            default_headers=llm_default_headers,
            params=llm_params,
            thinking_budget=llm_thinking_budget,
            start_vllm_on_init=config.get("start_vllm_on_init", False),
            vllm_server_params=vllm_server_params,
        )
    else:
        raise ValueError(f"Invalid LLM backend: {backend}")
