# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Async vLLM engine wrapper for models that use custom input tensors.

Wraps vLLM's ``AsyncLLM`` for streaming step-by-step inference with
``custom_inputs`` (used by both DuplexSTT/LLM and EarTTS backends).
``engine_kind`` selects EarTTS-only runtime settings (guidance scale, attention
backend env); it does not imply an inheritance relationship between TTS and LLM.
"""

import os
import json
import torch
from typing import Any, AsyncGenerator, Literal
from dataclasses import dataclass
from enum import Enum

from vllm.v1.engine.async_llm import AsyncLLM
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.attention.selector import _cached_get_attn_backend

from nemo.utils import logging

class StreamStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    ABORTED = "aborted"
    FINISHED = "finished"


@dataclass
class GenerationResult:
    token_id: int
    is_finished: bool
    custom_outputs: dict[str, torch.Tensor] | None = None
    finish_reason: str | None = None
    total_tokens: int = 0


@dataclass
class RequestState:
    """State for a single generation request."""
    request_id: str
    status: StreamStatus
    generated_tokens: list
    generation_iterator: AsyncGenerator | None = None


class CustomInputAsyncVLLMEngine:
    """
    Wrapper for vLLM ``AsyncLLM`` with custom input tensor specifications.

    KV cache is managed entirely inside the AsyncLLM engine -- callers do not
    allocate, pass, or update cache objects.  The engine uses PagedAttention to
    manage GPU memory automatically.  Per-request state (generated tokens,
    generation iterator, status) is tracked in ``self.requests`` dicts.

    Provides:
    - Start/stop streaming with custom embedding inputs
    - Generate one token at a time via ``generate_next_token``
    - Abort / restart individual requests by ``request_id``
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 10240,
        max_num_batched_tokens: int = 768,
        gpu_memory_utilization: float = 0.8,
        trust_remote_code: bool = True,
        dtype: str = "bfloat16",
        skip_tokenizer_init: bool = False,
        engine_kind: Literal["llm", "eartts"] = "llm",
        **sampling_kwargs
    ):
        """
        Initialize the async vLLM engine wrapper.

        Args:
            model_path: Path to the vLLM-compatible model checkpoint (required)
            max_model_len: Maximum sequence length (default: 10240)
            max_num_batched_tokens: Maximum tokens processed per forward pass.
                Controls prefill chunk size and max concurrent decode streams.
                Default: 768.
            gpu_memory_utilization: GPU memory utilization ratio (default: 0.8)
            trust_remote_code: Whether to trust remote code (default: True)
            dtype: Data type for embeddings (default: "bfloat16")
            engine_kind: ``"llm"`` for DuplexSTT-style models; ``"eartts"`` applies
                EarTTS-specific ``guidance_scale`` and attention/runtime env during init.
            **sampling_kwargs: Additional vLLM sampling parameters.
                Note: vLLM is configured for greedy decoding internally
                (temperature=0, ignore_eos=True). Actual text sampling
                (top-p, repetition penalty) is applied post-hoc by
                ModelInterface._sample_text_token, not by vLLM.
        """
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.skip_tokenizer_init = skip_tokenizer_init
        self.engine_kind = engine_kind

        # Engine state
        self.engine: AsyncLLM | None = None

        # Request state tracking - supports multiple concurrent requests
        self.requests: dict[str, RequestState] = {}

        # vLLM sampling is disabled (skip_sampling=True) because we handle
        # token selection ourselves for both the LLM and EarTTS paths.
        # For LLM: text sampling (top-p, repetition penalty) is applied
        # post-hoc by ModelInterface._sample_text_token on the returned logits.
        # For EarTTS: acoustic tokens are produced by the model's own forward
        # pass (RVQ codebook prediction), not by vLLM's sampler.
        default_sampling = {
            "max_tokens": 100000, # Set very high to prevent stopping - use abort to stop explicitly
            "skip_sampling": True,
            "ignore_eos": True,
        }
        default_sampling.update(sampling_kwargs)
        self.sampling_params = SamplingParams(**default_sampling)

        if self.engine_kind == "eartts":
            guidance_scale = self._read_guidance_scale_from_config()
            self.sampling_params.guidance_scale = guidance_scale
            logging.info(
                f"CustomInputAsyncVLLMEngine initialized (engine_kind=eartts, "
                f"guidance_scale={guidance_scale}, model={model_path})"
            )
        else:
            logging.info(
                f"CustomInputAsyncVLLMEngine initialized (engine_kind=llm, model={model_path})"
            )

    def _read_guidance_scale_from_config(self) -> float:
        """Read guidance_scale from the converted vLLM model's config.json."""
        config_path = os.path.join(self.model_path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            value = cfg.get("guidance_scale", None)
            if value is not None:
                logging.info(f"Read guidance_scale={value} from {config_path}")
                return float(value)
        logging.warning(
            f"guidance_scale not found in {config_path}, using default 0.5."
        )
        return 0.5

    async def initialize(self):
        """Initialize the vLLM engine with custom input specifications."""
        if self.engine is not None:
            logging.info("Engine already initialized!")
            return

        logging.info("Initializing vLLM engine...")

        eartts_env = self.engine_kind == "eartts"
        if eartts_env:
            # Force TRITON_ATTN backend for EarTTS
            os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
            # TF32 matmul precision to match TTS training ("medium").
            # torch.set_float32_matmul_precision is process-local and does NOT
            # propagate to vLLM's spawned worker processes; this CUDA-level env
            # var is inherited by child processes.
            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
            _cached_get_attn_backend.cache_clear()

        try:
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_batched_tokens,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
                mamba_ssm_cache_dtype="float32",
                dtype=self.dtype,
                skip_tokenizer_init=self.skip_tokenizer_init,
                enable_prefix_caching=False
            )

            # Custom input/output specs are defined in the model's config file
            vllm_config = engine_args.create_engine_config()
            self.custom_input_specs = vllm_config.model_config.custom_input_specs

            # Initialize the engine
            self.engine = AsyncLLM.from_vllm_config(vllm_config)

            logging.info("Engine initialized with custom input specs:")
            for spec in self.custom_input_specs:
                logging.info(f"  - {spec}")
        finally:
            if eartts_env:
                os.environ.pop("VLLM_ATTENTION_BACKEND", None)
                os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
                _cached_get_attn_backend.cache_clear()

    def _get_safe_prompt_tokens(self, length: int = 10) -> list[int]:
        """Generate safe prompt tokens that won't cause immediate EOS."""
        if self.engine and hasattr(self.engine, 'tokenizer') and self.engine.tokenizer:
            bos_id = getattr(self.engine.tokenizer, 'bos_token_id', 1)
        else:
            bos_id = 1

        # Mix of BOS + safe alphanumeric tokens
        safe_tokens = [bos_id] + list(range(50, 59))  # tokens 50-58 are usually safe
        return (safe_tokens * ((length // len(safe_tokens)) + 1))[:length]

    async def start_generation(
        self,
        request_id: str = "speech_stream"
    ) -> bool:
        """
        Start a new streaming generation session.

        Args:
            request_id: Unique identifier for this generation request

        Returns:
            bool: True if generation started successfully
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized! Call initialize() first.")

        # Check if request already exists
        if request_id in self.requests:
            existing_state = self.requests[request_id]
            if existing_state.status == StreamStatus.ACTIVE:
                logging.warning(f"Request {request_id} is already active. Aborting it first.")
                await self.abort_generation(request_id)

        logging.info(f"Starting generation session with request_id: {request_id}")

        # Create new request state
        self.requests[request_id] = RequestState(
            request_id=request_id,
            status=StreamStatus.ACTIVE,
            generated_tokens=[],
            generation_iterator=None  # Will be created on first generate_next_token call
        )
        return True

    async def generate_next_token(self, input_tensors: list[torch.Tensor],
                                        prompt_token_ids: list[int] | None = None,
                                        request_id: str = "speech_stream") -> GenerationResult | None:
        """
        Generate the next token using the provided input embedding.

        Args:
            input_tensors: List of tensors for generating the next token
            prompt_token_ids: Optional list of token IDs for the system prompt
            request_id: Unique identifier for this generation request

        Returns:
            GenerationResult or None if generation is finished/aborted
        """
        if request_id not in self.requests:
            raise RuntimeError(f"Request {request_id} not found. Call start_generation() first.")

        request_state = self.requests[request_id]

        if request_state.status != StreamStatus.ACTIVE:
            logging.warning(f"Generation not active for request {request_id} (status: {request_state.status})")
            return None

        assert len(input_tensors) == len(self.custom_input_specs), f"Expected {len(self.custom_input_specs)} input tensors, got {len(input_tensors)}"

        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        custom_inputs = {}
        max_length = 1
        for i, spec in enumerate(self.custom_input_specs):
            input_dtype = spec.dtype
            if input_dtype is None:
                input_dtype = "float32"  # Default dtype
            if spec.dim is not None and spec.dim != input_tensors[i].shape[-1]:
                raise ValueError(f"Input tensor dimension mismatch for {spec.name}: expected {spec.dim}, got {input_tensors[i].shape[-1]}")
            custom_inputs[spec.name] = input_tensors[i].to(dtype=getattr(torch, input_dtype)).cpu()
            max_length = max(max_length, input_tensors[i].shape[0])

        try:
            # If this is the first call, initialize the generation
            if request_state.generation_iterator is None:
                # Create initial inputs with a single safe prompt token
                # this will not be used for generation, just to initialize the model state
                prompt_tokens = prompt_token_ids if prompt_token_ids is not None else self._get_safe_prompt_tokens(max_length)
                assert len(prompt_tokens) == max_length, f"Prompt tokens length {len(prompt_tokens)} does not match input length {max_length}"
                inputs = {
                    "prompt_token_ids": prompt_tokens,
                        "custom_inputs": custom_inputs
                }

                logging.info(f"Initializing generation for request {request_id} with first embedding")

                # Start generation
                request_state.generation_iterator = self.engine.generate(
                    inputs,
                    self.sampling_params,
                    request_id=request_id
                )

            # If this is not the first call, append the current embedding for processing
            elif request_state.generation_iterator is not None and len(request_state.generated_tokens) > 0:
                try:
                    await self.engine.append_request(
                        request_id=request_id,
                        custom_inputs=custom_inputs
                    )
                except ValueError as e:
                    if "not found" in str(e):
                        logging.warning(f"Request {request_id} was removed from vLLM engine. Marking as finished.")
                        request_state.status = StreamStatus.FINISHED
                        return None
                    else:
                        raise RuntimeError(f"Error appending to request {request_id}: {e}")

            # Get next output from the generation
            output = await request_state.generation_iterator.__anext__()

            # Extract new tokens
            current_tokens = output.outputs[0].token_ids
            if len(current_tokens) > len(request_state.generated_tokens):
                new_tokens = current_tokens[len(request_state.generated_tokens):]
                assert len(new_tokens) == 1, f"Expected exactly one new token, got {len(new_tokens)}"
                new_tokens = current_tokens[-1:]
                request_state.generated_tokens.extend(new_tokens)

                # Get the latest token
                latest_token = new_tokens[-1]

                # Check if finished
                if output.finished:
                    request_state.status = StreamStatus.FINISHED
                    finish_reason = output.outputs[0].finish_reason
                    logging.warning(f"Request {request_id} finished after {len(request_state.generated_tokens)} tokens. Reason: {finish_reason}")
                    return GenerationResult(
                        token_id=latest_token,
                        custom_outputs=output.outputs[0].custom_outputs if hasattr(output.outputs[0], 'custom_outputs') else None,
                        is_finished=True,
                        finish_reason=finish_reason,
                        total_tokens=len(request_state.generated_tokens)
                    )
                else:
                    return GenerationResult(
                        token_id=latest_token,
                        custom_outputs=output.outputs[0].custom_outputs if hasattr(output.outputs[0], 'custom_outputs') else None,
                        is_finished=False,
                        total_tokens=len(request_state.generated_tokens)
                    )
            else:
                # No new tokens generated
                if output.finished:
                    logging.warning("No new tokens but finished!")
                    logging.warning(output.outputs[0].finish_reason)
                    request_state.status = StreamStatus.FINISHED
                return None

        except StopAsyncIteration:
            # Generation ended
            request_state.status = StreamStatus.FINISHED
            return None
        except Exception as e:
            logging.error(f"Error in generate_next_token for request {request_id}: {e}")
            request_state.status = StreamStatus.FINISHED
            return None

    async def abort_generation(self, request_id: str = "speech_stream") -> bool:
        """
        Abort a specific generation request.

        Args:
            request_id: Unique identifier for the generation request to abort

        Returns:
            bool: True if abort was successful
        """
        if request_id not in self.requests:
            logging.warning(f"Request {request_id} not found")
            return False

        request_state = self.requests[request_id]

        if request_state.status != StreamStatus.ACTIVE:
            logging.info(f"Request {request_id} is {request_state.status.value}, cleaning up state")
            # Just remove the state, no need to abort
            del self.requests[request_id]
            return True

        try:
            await self.engine.abort(request_id)
            request_state.status = StreamStatus.ABORTED
            del self.requests[request_id]
            logging.info(f"Aborted generation for request: {request_id}")
            return True
        except Exception as e:
            logging.error(f"Error aborting generation for request {request_id}: {e}")
            return False

    async def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        # Abort all active requests
        for request_id, request_state in list(self.requests.items()):
            if request_state.status == StreamStatus.ACTIVE:
                await self.abort_generation(request_id)

        if self.engine is not None:
            logging.info("Shutting down engine...")
            self.engine.shutdown()
            self.engine = None
            logging.info("Engine shutdown complete.")

        # Clear all request states
        self.requests.clear()

    def get_status(self, request_id: str | None = None) -> dict[str, Any]:
        """Get current status information.

        Args:
            request_id: If provided, return status for specific request.
                       If None, return status for all requests.

        Returns:
            Status information dictionary
        """
        if request_id is not None:
            if request_id not in self.requests:
                return {"error": f"Request {request_id} not found"}

            request_state = self.requests[request_id]
            return {
                "request_id": request_id,
                "status": request_state.status.value,
                "tokens_generated": len(request_state.generated_tokens),
                "latest_tokens": request_state.generated_tokens[-5:] if request_state.generated_tokens else []
            }
        else:
            # Return summary of all requests
            return {
                "total_requests": len(self.requests),
                "requests": {
                    rid: {
                        "status": state.status.value,
                        "tokens_generated": len(state.generated_tokens)
                    }
                    for rid, state in self.requests.items()
                }
            }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


def create_engine(engine_type: str = "llm", **kwargs) -> CustomInputAsyncVLLMEngine:
    """
    Factory function to create a CustomInputAsyncVLLMEngine instance.

    Args:
        engine_type: ``"llm"`` or ``"eartts"`` (maps to ``engine_kind``).
        **kwargs: Passed to the engine (model_path, max_model_len, etc.).
    Returns:
        A configured ``CustomInputAsyncVLLMEngine``.
    """

    if engine_type == "eartts":
        return CustomInputAsyncVLLMEngine(engine_kind="eartts", **kwargs)
    elif engine_type == "llm":
        return CustomInputAsyncVLLMEngine(engine_kind="llm", **kwargs)
    else:
        raise ValueError(f"Unsupported engine_type: {engine_type}")

