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
Base class for vLLM-backed inference backends.

Provides shared vLLM engine lifecycle management (init, abort, restart,
shutdown) used by both the LLM backend (``VLLMLLM`` -- DuplexSTT text
token prediction) and the TTS backend (``VLLMEarTTS`` -- EarTTS audio
codec generation).
"""

from abc import abstractmethod
from typing import Any
import os
import torch

from nemo.utils import logging
from nemo.collections.speechlm2.inference.model_wrappers.backend.interface import ModelInterface


class VLLMModelBase(ModelInterface):
    """
    Base class for vLLM-backed model interfaces.

    Wraps a CustomInputAsyncVLLMEngine to provide streaming inference while
    conforming to the ModelInterface contract. Supports two usage modes:

    1. **Blocking component** (default): Call the model synchronously via
       ``__call__``. The async engine runs on an internal event loop.
       This is the mode used by the S2S streaming inference pipeline
       (``StreamingS2SPipeline`` / ``NemotronVoicechatInferenceWrapper``).

    2. **Async standalone server**: Use ``_async_inference()`` directly
       from your own async event loop for concurrent multi-stream serving
       (e.g., a WebSocket server handling multiple streams concurrently).

    Subclasses must implement:
        - ``_convert_ckpt(save_path)``: checkpoint conversion to vLLM format
        - ``__call__(...)``: synchronous inference entry point
        - ``_process_inputs_to_outputs(...)``: async core inference logic

    Requires vLLM from https://github.com/vklimkov-nvidia/vllm (branch vklimkov/voicechat).
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 1024,
        max_num_batched_tokens: int = 768,
        gpu_memory_utilization: float = 0.8,
        trust_remote_code: bool = True,
        dtype: str = "bfloat16",
        engine_path: str | None = None,
        pretrained_llm: str | None = None,
        special_token_ids: set[int] | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        model_type: str = "llm",
        **sampling_kwargs
    ):
        """
        Initialize vLLM model with engine creation, event loop setup, and warm-up.

        Args:
            model_path: Path to the vLLM-compatible model checkpoint
            max_model_len: Maximum sequence length
            max_num_batched_tokens: Maximum tokens per vLLM forward pass.
                Controls prefill chunk size and max concurrent decode streams.
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            trust_remote_code: Whether to trust remote code in model
            dtype: Data type for embeddings (e.g., "bfloat16", "float16")
            engine_path: Optional path to pre-converted vLLM model
            pretrained_llm: Optional path to pretrained LLM for conversion
            special_token_ids: Set of special token IDs (pad, eos, bos) that bypass
                sampling and always use greedy decoding.
            top_p: Top-p (nucleus) sampling threshold. Default: 1.0 (disabled).
            repetition_penalty: Penalty for repeated tokens. Default: 1.0 (disabled).
            temperature: Sampling temperature. Default: 1.0 (no scaling).
            model_type: Type of model for vLLM engine ("llm", "eartts", etc.)
            **sampling_kwargs: Additional vLLM sampling parameters.

        Note:
            vLLM internally runs greedy decoding (temperature=0, ignore_eos=True).
            Text sampling (top_p, repetition_penalty, temperature) is applied
            post-hoc by ``ModelInterface._sample_text_token`` on the logits
            returned by vLLM, not by vLLM's own sampler.
        """
        super().__init__(
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

        import asyncio
        from nemo.collections.speechlm2.inference.vllm.streaming_llm_engine import create_engine

        self.model_path = model_path
        self.pretrained_llm = pretrained_llm
        self._dtype = dtype
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Force greedy decoding in vLLM by setting temperature=0 if not specified
        if 'temperature' not in sampling_kwargs:
            sampling_kwargs['temperature'] = 0.0

        if engine_path is None:
            dir_name = os.path.basename(os.path.normpath(model_path))
            engine_path = "/tmp/" + dir_name + f"_vllm_converted_{model_type}"
            if os.path.exists(engine_path):
                logging.info(f"Found existing vLLM converted model at {engine_path}")
            else:
                self._convert_ckpt(save_path=engine_path)

        self.engine = create_engine(
            engine_type=model_type,
            model_path=engine_path,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            **sampling_kwargs
        )
        self._request_counter = 0

        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        logging.info("Initializing vLLM engine (this may take a moment)...")
        self._loop.run_until_complete(self.engine.initialize())

        if self.engine.engine.tokenizer is not None and not self.special_token_ids:
            self.special_token_ids = self._get_special_token_ids_from_vllm_tokenizer(self.engine.engine.tokenizer)

        logging.debug(f"Special token IDs: {self.special_token_ids}")
        logging.info("vLLM engine ready!")

    @abstractmethod
    def _convert_ckpt(self, save_path: str):
        """Convert existing checkpoint to vLLM format and save."""
        pass

    @staticmethod
    def _get_special_token_ids_from_vllm_tokenizer(tokenizer) -> set[int]:
        """
        Extract special token IDs from a vLLM tokenizer.
        Looks for: '<s>' (bos), '</s>' (eos), '<SPECIAL_12>' (pad).

        Args:
            tokenizer: A vLLM CachedTokenizer instance.

        Returns:
            Set of special token IDs.
        """
        special_ids = set()
        for token in ('<s>', '</s>', '<SPECIAL_12>'):
            try:
                tid = tokenizer.convert_tokens_to_ids(token)
                if isinstance(tid, int):
                    special_ids.add(tid)
            except (KeyError, AttributeError):
                logging.debug(f"Token '{token}' not found in vLLM tokenizer, skipping")
        return special_ids

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        return f"vllm_request_{self._request_counter}"

    async def _async_inference(
        self,
        inputs: torch.Tensor | list[torch.Tensor] | dict,
        request_id: str,
        **kwargs
    ) -> dict[str, Any]:
        """
        Async inference using the streaming engine.

        Checks request status (starting or restarting as needed) and
        delegates to the subclass ``_process_inputs_to_outputs``.

        Args:
            inputs: Model inputs (tensor for LLM, dict for EarTTS)
            request_id: Unique request identifier
            **kwargs: Passed through to ``_process_inputs_to_outputs``

        Returns:
            Dictionary with model-specific outputs
        """
        from nemo.collections.speechlm2.inference.vllm.streaming_llm_engine import StreamStatus

        if request_id not in self.engine.requests:
            await self.engine.start_generation(request_id=request_id)
        else:
            request_state = self.engine.requests[request_id]
            if request_state.status in (StreamStatus.FINISHED, StreamStatus.ABORTED):
                logging.warning(
                    f"Request {request_id} was {request_state.status.value}. "
                    f"Generated {len(request_state.generated_tokens)} tokens before stopping. "
                    "Cleaning up and restarting..."
                )
                try:
                    await self.engine.abort_generation(request_id)
                except Exception:
                    # The request already finished/aborted; abort_generation
                    # is just releasing engine-side resources.  If the engine
                    # already purged it, the call may raise -- harmless since
                    # we start a fresh generation immediately after.
                    pass
                await self.engine.start_generation(request_id=request_id)

        return await self._process_inputs_to_outputs(inputs, request_id, **kwargs)

    @abstractmethod
    async def _process_inputs_to_outputs(self, inputs, request_id: str, **kwargs) -> dict[str, Any]:
        """Process model inputs and return outputs. Subclasses must implement."""
        pass

    def to(self, device_or_dtype: torch.device | torch.dtype) -> 'VLLMModelBase':
        """
        Move model to specified device or convert to specified dtype.

        Note: vLLM manages device placement internally, this is for compatibility.
        """
        if isinstance(device_or_dtype, torch.device):
            self._device = device_or_dtype
        return self

    def eval(self) -> 'VLLMModelBase':
        """Set model to evaluation mode (vLLM is always in eval mode)."""
        return self

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self._device

    def abort_request(self, request_id: str) -> bool:
        """
        Abort a specific generation request.

        Args:
            request_id: Request identifier to abort

        Returns:
            bool: True if abort was successful
        """
        return self._loop.run_until_complete(
            self.engine.abort_generation(request_id)
        )

    def restart_request(self, request_id: str) -> bool:
        """
        Restart a finished or aborted generation request.

        Args:
            request_id: Request identifier to restart

        Returns:
            bool: True if restart was successful
        """
        if request_id in self.engine.requests:
            self.abort_request(request_id)

        return self._loop.run_until_complete(
            self.engine.start_generation(request_id=request_id)
        )

    def get_request_status(self, request_id: str | None = None) -> dict[str, Any]:
        """
        Get status of a specific request or all requests.

        Args:
            request_id: Optional request ID. If None, returns all requests.

        Returns:
            Status dictionary
        """
        return self.engine.get_status(request_id)

    def shutdown(self):
        """Shutdown the vLLM engine and cleanup resources."""
        self._loop.run_until_complete(self.engine.shutdown())

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            # __del__ may run during interpreter shutdown when globals
            # (self._loop, self.engine, asyncio) are already torn down.
            # Nothing useful to do with the error; suppress to avoid
            # noisy tracebacks on exit.
            pass
