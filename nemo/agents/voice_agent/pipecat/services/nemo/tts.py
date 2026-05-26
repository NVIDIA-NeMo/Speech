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
import concurrent.futures
import os
import queue as stdlib_queue
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, List, Optional

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMTextFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.tts_service import TTSService


def _disallow_mamba_kernels_in_graph() -> None:
    """Make Dynamo treat mamba-ssm / causal-conv1d Triton kernels as opaque.

    Custom Triton kernels with their own autotune wrappers can't be co-optimized
    by Inductor — without this, `torch.compile(model.decoder)` crashes with
    `KeyError: 'op5'` during Inductor lowering on the NemotronH SmallMamba
    checkpoint.  `torch._dynamo.disable` wraps each kernel so Dynamo breaks the
    graph at the call site, executes the kernel eagerly, and resumes tracing
    the rest of the backbone.

    Patches both the originating module and `nemotron_h_decoder` (which already
    imported these names at module load time).  Idempotent — safe to call from
    multiple workers.
    """
    import nemo.collections.tts.modules.nemotron_h_decoder as nhd
    from mamba_ssm.ops.triton import selective_state_update as _ssu_mod
    from mamba_ssm.ops.triton import ssd_combined as _ssd_mod
    from mamba_ssm.ops.triton import layernorm_gated as _rms_mod
    import causal_conv1d as _cc_mod

    targets = [
        (_ssu_mod, "selective_state_update", "selective_state_update"),
        (_ssd_mod, "mamba_chunk_scan_combined", "mamba_chunk_scan_combined"),
        (_ssd_mod, "mamba_split_conv1d_scan_combined", "mamba_split_conv1d_scan_combined"),
        (_rms_mod, "rmsnorm_fn", "rmsnorm_fn"),
        (_cc_mod, "causal_conv1d_fn", "causal_conv1d_fn"),
        (_cc_mod, "causal_conv1d_update", "causal_conv1d_update"),
    ]
    for src_mod, src_attr, dst_attr in targets:
        original = getattr(src_mod, src_attr)
        # Already-wrapped fns expose the original via __wrapped__; skip those.
        if getattr(original, "__wrapped__", None) is not None:
            continue
        wrapped = torch._dynamo.disable(original)
        setattr(src_mod, src_attr, wrapped)
        if hasattr(nhd, dst_attr):
            setattr(nhd, dst_attr, wrapped)

from nemo.agents.voice_agent.pipecat.services.nemo.audio_logger import AudioLogger
from nemo.agents.voice_agent.pipecat.utils.text.simple_text_aggregator import SimpleSegmentedTextAggregator
from nemo.agents.voice_agent.utils.tool_calling.mixins import ToolCallingMixin
from nemo.collections.tts.models import FastPitchModel, HifiGanModel


class BaseNemoTTSService(TTSService, ToolCallingMixin):
    """Text-to-Speech service using Nemo TTS models.

    This service works with any TTS model that exposes a generate(text) method
    that returns audio data. The TTS generation runs in a dedicated background thread to
    avoid blocking the main asyncio event loop, following the same pattern as NemoDiarService.

    Args:
        model: TTS model instance with a generate(text) method
        sample_rate: Audio sample rate in Hz (defaults to 22050)
        **kwargs: Additional arguments passed to TTSService
    """

    def __init__(
        self,
        *,
        model,
        device: str = "cuda",
        sample_rate: int = 22050,
        think_tokens: Optional[List[str]] = None,
        audio_logger: Optional[AudioLogger] = None,
        ignore_strings: Optional[List[str]] = None,
        streaming: bool = True,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        logger.info(f"Initializing TTS service with model: {model} and device: {device}")
        self._model_name = model
        self._device = device
        # Persistent single-worker executor.  ALL operations that touch the
        # torch.compile'd model (warmup, streaming_init, streaming_step in
        # _generate_audio) must run on this same thread so that CUDA graphs
        # captured by torch._inductor (stored in thread-local storage) are
        # visible at replay time.  Subclasses that use torch.compile with
        # CUDA graphs rely on this guarantee.  Created BEFORE _setup_model so
        # the subclass can submit warmup work to it.
        self._tts_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts-worker"
        )
        self._model = self._setup_model()
        self._think_tokens = think_tokens
        self._audio_logger = audio_logger
        if think_tokens is not None:
            assert (
                isinstance(think_tokens, list) and len(think_tokens) == 2
            ), f"think_tokens must be a list of two strings, but got type {type(think_tokens)}: {think_tokens}"
        self._ignore_strings = set(ignore_strings) if ignore_strings is not None else None
        # When True (default): background thread streams each chunk to the queue one-by-one;
        # run_tts awaits each chunk asynchronously, keeping the event loop free between chunks.
        # When False: background thread puts the generator object on the queue; run_tts
        # iterates it synchronously on the event loop (simpler, but blocks between chunks).
        self._streaming = streaming
        # Background processing infrastructure - no response handler needed
        self._tts_queue = asyncio.Queue()
        self._processing_task = None
        self._processing_running = False

        # Track pending requests with their response queues
        self._pending_requests = {}
        self._have_seen_think_tokens = False

    def reset(self):
        """Reset the TTS service."""
        self._text_aggregator.reset()

    def setup_tool_calling(self):
        """
        Setup the tool calling mixin by registering all available tools.
        """
        pass  # No tools by default

    def _setup_model(self):
        raise NotImplementedError("Subclass must implement _setup_model")

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        raise NotImplementedError("Subclass must implement _generate_audio")

    async def start(self, frame: StartFrame):
        """Handle service start."""
        await super().start(frame)

        # Initialize the model if not already done
        if not hasattr(self, "_model") or self._model is None:
            self._model = self._setup_model()

        # Only start background processing task - no response handler needed
        if not self._processing_task:
            self._processing_task = self.create_task(self._processing_task_handler())

    async def stop(self, frame: EndFrame):
        """Handle service stop."""
        await super().stop(frame)
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        """Handle service cancellation."""
        await super().cancel(frame)
        await self._stop_tasks()

    async def _stop_tasks(self):
        """Stop background processing tasks."""
        self._processing_running = False
        await self._tts_queue.put(None)  # Signal to stop processing

        if self._processing_task:
            await self.cancel_task(self._processing_task)
            self._processing_task = None

    def _tts_processor(self):
        """Background processor that handles TTS generation calls."""
        try:
            while self._processing_running:
                try:
                    future = asyncio.run_coroutine_threadsafe(self._tts_queue.get(), self.get_event_loop())
                    request = future.result()

                    if request is None:  # Stop signal
                        logger.debug("Received stop signal in TTS background processor")
                        break

                    text, request_id = request
                    logger.debug(f"Processing TTS request for text: [{text}]")

                    # Get the response queue for this request
                    response_queue = None
                    future = asyncio.run_coroutine_threadsafe(
                        self._get_response_queue(request_id), self.get_event_loop()
                    )
                    response_queue = future.result()

                    if response_queue is None:
                        logger.warning(f"No response queue found for request {request_id}")
                        continue

                    try:
                        if self._streaming:
                            # Streaming: iterate generator here in background thread,
                            # push each chunk to the queue so run_tts can yield between
                            # chunks without blocking the asyncio event loop.
                            for audio_chunk in self._generate_audio(text):
                                if request_id not in self._pending_requests:
                                    break  # request was cancelled
                                asyncio.run_coroutine_threadsafe(
                                    response_queue.put(('chunk', audio_chunk)), self.get_event_loop()
                                )
                            asyncio.run_coroutine_threadsafe(
                                response_queue.put(('done', None)), self.get_event_loop()
                            )
                        else:
                            # Non-streaming: send the generator object to run_tts,
                            # which iterates it synchronously on the event loop.
                            asyncio.run_coroutine_threadsafe(
                                response_queue.put(('result', self._generate_audio(text))), self.get_event_loop()
                            )
                    except Exception as e:
                        logger.exception(f"Error in TTS generation: {e}")
                        asyncio.run_coroutine_threadsafe(response_queue.put(('error', e)), self.get_event_loop())

                except Exception as e:
                    logger.exception(f"Error in background TTS processor: {e}")

        except Exception as e:
            logger.error(f"Background TTS processor fatal error: {e}")
        finally:
            logger.debug("Background TTS processor stopped")

    async def _get_response_queue(self, request_id: str):
        """Get the response queue for a specific request."""
        return self._pending_requests.get(request_id)

    async def _processing_task_handler(self):
        """Handler for background processing task."""
        try:
            self._processing_running = True
            logger.debug("Starting background TTS processing task")
            # Pin _tts_processor to the persistent _tts_executor's single worker
            # thread (NOT the default thread pool) so that CUDA graphs captured
            # during warmup on the same thread are replayable from _generate_audio.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._tts_executor, self._tts_processor)
        except asyncio.CancelledError:
            logger.debug("Background TTS processing task cancelled")
            self._processing_running = False
            raise
        finally:
            self._processing_running = False

    def _handle_think_tokens(self, text: str) -> Optional[str]:
        """
        Handle the thinking tokens for TTS.
        If the thinking tokens are not provided, return the text as it is.
        Otherwise:
            If both thinking tokens appear in the text, return the text after the end of thinking tokens.
            If the LLM is thinking, return None.
            If the LLM is done thinking, return the text after the end of thinking tokens.
            If the LLM starts thinking, return the text before the start of thinking tokens.
            If the LLM is not thinking, return the text as is.
        """
        if not self._think_tokens or not text:
            return text
        elif self._think_tokens[0] in text and self._think_tokens[1] in text:
            # LLM finishes thinking in one chunk or outputs dummy thinking tokens
            logger.debug(f"LLM finishes thinking: {text}")
            idx = text.index(self._think_tokens[1])
            # only return the text after the end of thinking tokens
            text = text[idx + len(self._think_tokens[1]) :]
            self._have_seen_think_tokens = False
            logger.debug(f"Returning text after thinking: {text}")
            return text
        elif self._have_seen_think_tokens:
            # LLM is thinking
            if self._think_tokens[1] not in text:
                logger.debug(f"LLM is still thinking: {text}")
                # LLM is still thinking
                return None
            else:
                # LLM is done thinking
                logger.debug(f"LLM is done thinking: {text}")
                idx = text.index(self._think_tokens[1])
                # only return the text after the end of thinking tokens
                text = text[idx + len(self._think_tokens[1]) :]
                self._have_seen_think_tokens = False
                logger.debug(f"Returning text after thinking: {text}")
                return text
        elif self._think_tokens[0] in text:
            # LLM now starts thinking
            logger.debug(f"LLM starts thinking: {text}")
            self._have_seen_think_tokens = True
            # return text before the start of thinking tokens
            idx = text.index(self._think_tokens[0])
            text = text[:idx]
            logger.debug(f"Returning text before thinking: {text}")
            return text
        else:
            # LLM is not thinking
            return text

    def _drop_special_tokens(self, text: str) -> Optional[str]:
        """
        Drop the special tokens from the text.
        """
        if self._ignore_strings is None:
            return text
        for ignore_string in self._ignore_strings:
            if ignore_string in text:
                logger.debug(f"Dropping string `{ignore_string}` from text: `{text}`")
                text = text.replace(ignore_string, "")
        return text

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using the Nemo TTS model."""

        if self._think_tokens is not None:
            text = self._handle_think_tokens(text)

        if not text:
            yield None
            return

        if self._ignore_strings is not None:
            text = self._drop_special_tokens(text)

        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            yield TTSStartedFrame()

            # Increment turn index at the start of agent speaking (only if speaker changed)
            if self._audio_logger is not None:
                self._audio_logger.increment_turn_index(speaker="agent")

            # Generate unique request ID
            request_id = str(uuid.uuid4())

            # Create response queue for this specific request
            request_queue = asyncio.Queue()
            self._pending_requests[request_id] = request_queue

            try:
                # Queue the TTS request for background processing
                await self._tts_queue.put((text, request_id))

                first_chunk = True
                all_audio_bytes = b""

                if self._streaming:
                    # Streaming mode: background thread pushes chunks one-by-one;
                    # await each so the event loop stays free between chunks.
                    while True:
                        result = await request_queue.get()
                        status, data = result

                        if status == 'error':
                            logger.error(f"{self} TTS generation error: {data}")
                            yield ErrorFrame(error=f"TTS generation error: {str(data)}")
                            return

                        if status == 'done':
                            break

                        # status == 'chunk'
                        audio_chunk = data
                        if audio_chunk is None:
                            break

                        if first_chunk:
                            first_chunk = False
                            if self._audio_logger is not None and self._audio_logger.first_audio_timestamp is None:
                                self._audio_logger.first_audio_timestamp = datetime.now()

                        audio_bytes = self._convert_to_bytes(audio_chunk)
                        all_audio_bytes += audio_bytes
                        chunk_size = self.chunk_size
                        for i in range(0, len(audio_bytes), chunk_size):
                            audio_chunk_bytes = audio_bytes[i : i + chunk_size]
                            if not audio_chunk_bytes:
                                break
                            yield TTSAudioRawFrame(audio=audio_chunk_bytes, sample_rate=self.sample_rate, num_channels=1)
                else:
                    # Non-streaming mode: background thread sends the generator object;
                    # iterate it synchronously here (blocks event loop between chunks).
                    result = await request_queue.get()
                    status, data = result

                    if status == 'error':
                        logger.error(f"{self} TTS generation error: {data}")
                        yield ErrorFrame(error=f"TTS generation error: {str(data)}")
                        return

                    # status == 'result'
                    for audio_chunk in data:
                        if first_chunk:
                            first_chunk = False
                            if self._audio_logger is not None and self._audio_logger.first_audio_timestamp is None:
                                self._audio_logger.first_audio_timestamp = datetime.now()

                        audio_bytes = self._convert_to_bytes(audio_chunk)
                        all_audio_bytes += audio_bytes
                        chunk_size = self.chunk_size
                        for i in range(0, len(audio_bytes), chunk_size):
                            audio_chunk_bytes = audio_bytes[i : i + chunk_size]
                            if not audio_chunk_bytes:
                                break
                            yield TTSAudioRawFrame(audio=audio_chunk_bytes, sample_rate=self.sample_rate, num_channels=1)

                # Log the complete audio if logger is available
                if self._audio_logger is not None and all_audio_bytes:
                    try:
                        self._audio_logger.log_agent_audio(
                            audio_data=all_audio_bytes,
                            text=text,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            additional_metadata={
                                "model": self._model_name,
                            },
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log agent audio: {e}")

                yield TTSStoppedFrame()

            finally:
                # Clean up the pending request
                if request_id in self._pending_requests:
                    del self._pending_requests[request_id]

        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
            error_message = f"TTS generation error: {str(e)}"
            yield ErrorFrame(error=error_message)

    def _convert_to_bytes(self, audio_data) -> bytes:
        """Convert various audio data formats to bytes."""
        if isinstance(audio_data, (bytes, bytearray)):
            return bytes(audio_data)

        if isinstance(audio_data, np.ndarray):
            # Ensure it's in the right format (16-bit PCM)
            if audio_data.dtype in [np.float32, np.float64]:
                # Convert float [-1, 1] to int16 [-32768, 32767]
                audio_data = np.clip(audio_data, -1.0, 1.0)  # Ensure values are in range
                audio_data = (audio_data * 32767).astype(np.int16)
            elif audio_data.dtype != np.int16:
                # Convert other integer types to int16
                audio_data = audio_data.astype(np.int16)
            return audio_data.tobytes()
        elif hasattr(audio_data, 'tobytes'):
            return audio_data.tobytes()
        else:
            return bytes(audio_data)


class NeMoFastPitchHiFiGANTTSService(BaseNemoTTSService):
    """Text-to-Speech service using NeMo FastPitch-Hifigan model.

    More info: https://huggingface.co/nvidia/tts_en_fastpitch

    Args:
        fastpitch_model: FastPitch model name
        hifigan_model: Hifigan model name
        device: Device to run on (default: 'cuda')
        **kwargs: Additional arguments passed to BaseNemoTTSService
    """

    def __init__(
        self,
        fastpitch_model: str = "nvidia/tts_en_fastpitch",
        hifigan_model: str = "nvidia/tts_hifigan",
        device: str = "cuda",
        **kwargs,
    ):
        model_name = f"{fastpitch_model}+{hifigan_model}"
        self._fastpitch_model_name = fastpitch_model
        self._hifigan_model_name = hifigan_model
        super().__init__(model=model_name, device=device, **kwargs)
        self.setup_tool_calling()

    def _setup_model(self):
        logger.info(
            f"Loading FastPitch model={self._fastpitch_model_name} and HiFiGAN model={self._hifigan_model_name}"
        )
        self._fastpitch_model = self._setup_fastpitch_model(self._fastpitch_model_name)
        self._hifigan_model = self._setup_hifigan_model(self._hifigan_model_name)
        return self._fastpitch_model, self._hifigan_model

    def _setup_fastpitch_model(self, model_name: str):
        if model_name.endswith(".nemo"):
            fastpitch_model = FastPitchModel.restore_from(model_name, map_location=torch.device(self._device))
        else:
            fastpitch_model = FastPitchModel.from_pretrained(model_name, map_location=torch.device(self._device))
        fastpitch_model.eval()
        return fastpitch_model

    def _setup_hifigan_model(self, model_name: str):
        if model_name.endswith(".nemo"):
            hifigan_model = HifiGanModel.restore_from(model_name, map_location=torch.device(self._device))
        else:
            hifigan_model = HifiGanModel.from_pretrained(model_name, map_location=torch.device(self._device))
        hifigan_model.eval()
        return hifigan_model

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        with torch.no_grad():
            parsed = self._fastpitch_model.parse(text)
            spectrogram = self._fastpitch_model.generate_spectrogram(tokens=parsed)
            audio = self._hifigan_model.convert_spectrogram_to_audio(spec=spectrogram)
            audio = audio.detach().view(-1).cpu().numpy()
            yield audio


class KokoroTTSService(BaseNemoTTSService):
    """Text-to-Speech service using Kokoro-82M model.

    Kokoro is an open-weight TTS model with 82 million parameters.
    More info: https://huggingface.co/hexgrad/Kokoro-82M

    Args:
        lang_code: Language code for the model (default: 'a' for American English)
        voice: Voice to use (default: 'af_heart')
        device: Device to run on (default: 'cuda')
        sample_rate: Audio sample rate in Hz (default: 24000 for Kokoro)
        download_all: Download all models for different languages (default: True)
        cache_models: Cache models on GPU for faster switching between languages (default: True)
        **kwargs: Additional arguments passed to BaseNemoTTSService
    """

    def __init__(
        self,
        model: str = "hexgrad/Kokoro-82M",
        lang_code: str = "a",
        voice: str = "af_heart",
        device: str = "cuda",
        sample_rate: int = 24000,
        speed: float = 1.0,
        download_all: bool = True,
        cache_models: bool = True,
        **kwargs,
    ):
        self._lang_code = lang_code
        self._voice = voice
        self._speed = speed
        assert speed > 0, "Speed must be greater than 0"
        self._original_speed = speed
        self._original_voice = voice
        self._gender = 'female' if voice[1] == 'f' else 'male'
        self._original_gender = self._gender
        self._original_lang_code = self._lang_code
        if download_all:
            self._model_maps = self._download_all_models(
                lang_code=["a", "b"], device=device, repo_id=model, cache_models=cache_models
            )
        else:
            self._model_maps = {}
        super().__init__(model=model, device=device, sample_rate=sample_rate, **kwargs)
        self.setup_tool_calling()

    def _setup_model(self, lang_code: Optional[str] = None, voice: Optional[str] = None):
        """Initialize the Kokoro pipeline."""
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "kokoro package is required for KokoroTTSService. Install it with: `pip install kokoro>=0.9.2`"
            )
        if lang_code is None:
            lang_code = self._lang_code
        if voice is None:
            voice = self._voice
        logger.info(f"Loading Kokoro TTS model with model={self._model_name}, lang_code={lang_code}, voice={voice}")
        if lang_code in self._model_maps:
            pipeline = self._model_maps[lang_code]
        else:
            pipeline = KPipeline(lang_code=lang_code, device=self._device, repo_id=self._model_name)
            self._model_maps[lang_code] = pipeline
        return pipeline

    def _download_all_models(
        self, lang_code: List[str] = ['a', 'b'], device="cuda", repo_id="hexgrad/Kokoro-82M", cache_models=True
    ):
        """Download all models for Kokoro TTS service."""
        logger.info(f"Downloading all models for Kokoro TTS service with lang_code={lang_code}")
        from kokoro import KPipeline

        model_maps = {}

        for lang in lang_code:
            pipeline = KPipeline(lang_code=lang, device=device, repo_id=repo_id)
            if cache_models:
                model_maps[lang] = pipeline
        torch.cuda.empty_cache()
        return model_maps

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        """Generate audio using the Kokoro pipeline.

        Args:
            text: Text to convert to speech

        Yields:
            Audio data as numpy arrays
        """
        try:
            # Generate audio using Kokoro pipeline
            generator = self._model(text, voice=self._voice, speed=self._speed)

            # The generator yields tuples of (gs, ps, audio)
            # We only need the audio component
            for i, (gs, ps, audio) in enumerate(generator):
                logger.debug(
                    f"Kokoro generated audio chunk {i}: gs={gs}, ps={ps},"
                    f"audio_shape={audio.shape if hasattr(audio, 'shape') else len(audio)}"
                )
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                # Kokoro returns audio as numpy array in float32 format [-1, 1]
                # The base class will handle conversion to int16 bytes
                yield audio

        except Exception as e:
            logger.error(f"Error generating audio with Kokoro: {e}")
            raise

    async def tool_tts_set_speed(self, params: FunctionCallParams, speed_lambda: float):
        """
        Set a specific speaking speed of the assistant's voice.
        This tool should be called only when the user specifies the speed explicitly,
        such as "speak twice as fast" or "speak half as slow" or "speak 1.5 times as fast".

        Inform user of the result of this tool call. After calling this tool, continue the previous
        response if it was unfinished and was interrupted by the user, otherwise start a new response
        and ask if the user needs help on anything else. Avoid repeating previous responses.

        Args:
            speed_lambda: positive float, the relative change of the speaking speed to the original speed.
                        E.g., 1.0 for original speed, 1.25 for 25% faster than original speed,
                        0.8 for 20% slower than original speed.

        """
        if speed_lambda <= 0:
            result = {
                "success": False,
                "message": f"Speed remains unchanged since the change is not a positive number: {speed_lambda}",
            }
            logger.debug(f"Speed remains unchanged since the change is not a positive number: {speed_lambda}")
        else:
            self._speed = speed_lambda * self._speed
            result = {
                "success": True,
                "message": f"Speed set to {speed_lambda} of the previous speed",
            }
            logger.debug(f"Speed set to {speed_lambda} of the previous speed {self._original_speed}")
        await params.result_callback(result)

    async def tool_tts_reset_speed(self, params: FunctionCallParams):
        """
        Reset the speaking speed to the original speed.

        Inform user of the result of this tool call. After calling this tool, continue the previous
        response if it was unfinished and was interrupted by the user, otherwise start a new response
        and ask if the user needs help on anything else. Avoid repeating previous responses.
        """
        self._speed = self._original_speed
        result = {"success": True, "message": "Speaking speed is reset to the original one"}
        logger.debug(f"Speaking speed is reset to the original speed {self._original_speed}")
        await params.result_callback(result)

    async def tool_tts_speak_faster(self, params: FunctionCallParams):
        """
        Speak faster by increasing the speaking speed 15% faster each time this function is called.

        Inform user of the result of this tool call. After calling this tool, continue the previous
        response if it was unfinished and was interrupted by the user, otherwise start a new response
        and ask if the user needs help on anything else. Avoid repeating previous responses.
        """
        speed_lambda = 1.15
        self._speed = speed_lambda * self._speed
        result = {
            "success": True,
            "message": f"Speaking speed is increased to {speed_lambda} of the previous speed",
        }
        logger.debug(f"Speed is set to {speed_lambda} of the previous speed, new speed is {self._speed}")
        await params.result_callback(result)

    async def tool_tts_speak_slower(self, params: FunctionCallParams):
        """
        Speak slower by decreasing the speaking speed 15% slower each time this function is called.

        Inform user of the result of this tool call. After calling this tool, continue the previous
        response if it was unfinished and was interrupted by the user, otherwise start a new response
        and ask if the user needs help on anything else. Avoid repeating previous responses.
        """
        speed_lambda = 0.85
        self._speed = speed_lambda * self._speed
        result = {
            "success": True,
            "message": f"Speaking speed is decreased to {speed_lambda} of the previous speed",
        }
        logger.debug(f"Speed is set to {speed_lambda} of the previous speed, new speed is {self._speed}")
        await params.result_callback(result)

    async def tool_tts_set_voice(self, params: FunctionCallParams, accent: str, gender: str):
        """
        Set the accent and gender of the assistant's voice.
        This tool should be called only when the user specifies the accent and/or gender explicitly.

        Inform user of the result of this tool call. After calling this tool, continue the previous
        response if it was unfinished and was interrupted by the user, otherwise start a new response
        and ask if the user needs help on anything else. Avoid repeating previous responses.

        Args:
            accent: Accent for the TTS model. Must be one of 'American English', 'British English'
                    or 'current' for keeping the current accent.
            gender: gender of the assistant's voice. Must be one of 'male', 'female',
                    or 'current' for keeping the current gender.
        """
        await params.llm.push_frame(LLMTextFrame("Just a moment."))

        lang_code = "a" if accent == "American English" else "b" if accent == "British English" else "current"
        new_lang_code = self._lang_code
        new_gender = self._gender
        if lang_code != 'current':
            new_lang_code = lang_code
        if gender != 'current':
            new_gender = gender

        if new_lang_code == 'a':
            new_voice = 'af_heart' if new_gender == 'female' else 'am_michael'
        elif new_lang_code == 'b':
            new_voice = 'bf_emma' if new_gender == 'female' else 'bm_george'
        else:
            await params.result_callback(
                {
                    "success": False,
                    "message": f"Invalid language code: {new_lang_code} or gender: {new_gender}",
                }
            )
            return

        new_model = await asyncio.to_thread(self._setup_model, new_lang_code, new_voice)
        self._model = new_model
        self._lang_code = new_lang_code
        self._gender = new_gender
        self._voice = new_voice
        logger.debug(f"Language and voice are set to {new_lang_code} and {new_voice}")
        await params.result_callback({"success": True, "message": "Voice has been updated."})

    async def tool_tts_reset_voice(self, params: FunctionCallParams):
        """
        Reset the accent and voice to the original ones.

        Inform user of the result of this tool call. After calling this tool, continue the previous
        response if it was unfinished and was interrupted by the user, otherwise start a new response
        and ask if the user needs help on anything else. Avoid repeating previous responses.

        """
        await params.llm.push_frame(LLMTextFrame("Of course."))

        new_model = await asyncio.to_thread(self._setup_model, self._original_lang_code, self._original_voice)
        self._model = new_model
        self._lang_code = self._original_lang_code
        self._gender = self._original_gender
        self._voice = self._original_voice
        logger.debug(
            f"Language and voice are reset to the original ones {self._original_lang_code} and {self._original_voice}"
        )
        await params.result_callback({"success": True, "message": "Voice has been reset to the original one."})

    def setup_tool_calling(self):
        """
        Setup the tool calling mixin by registering all available tools.
        """
        self.register_direct_function("tool_tts_reset_speed", self.tool_tts_reset_speed)
        self.register_direct_function("tool_tts_speak_faster", self.tool_tts_speak_faster)
        self.register_direct_function("tool_tts_speak_slower", self.tool_tts_speak_slower)
        self.register_direct_function("tool_tts_set_speed", self.tool_tts_set_speed)
        self.register_direct_function("tool_tts_set_voice", self.tool_tts_set_voice)
        self.register_direct_function("tool_tts_reset_voice", self.tool_tts_reset_voice)

    def reset(self):
        """
        Reset the voice and speed to the original ones.
        """
        self._text_aggregator.reset()
        self._speed = self._original_speed
        self._model = self._setup_model(self._original_lang_code, self._original_voice)
        self._lang_code = self._original_lang_code
        self._gender = self._original_gender
        self._voice = self._original_voice


class MagpieTTSService(BaseNemoTTSService):
    """Text-to-Speech service using Magpie TTS model.

    Magpie is a multilingual TTS model with 357 million parameters.
    More info: https://huggingface.co/nvidia/magpie_tts_multilingual_357m

    Args:
        model: Model name or path to the Magpie TTS model.
        language: Language code for the model (default: 'en' for English)
        speaker: Speaker to use for the model (default: 'Sofia')
        apply_TN: Whether to apply text normalization (default: False)
        device: Device to run on (default: 'cuda')
        **kwargs: Additional arguments passed to BaseNemoTTSService
    """

    SPEAKER_MAP = {"John": 0, "Sofia": 1, "Aria": 2, "Jason": 3, "Leo": 4}

    def __init__(
        self,
        model: str = "nvidia/magpie_tts_multilingual_357m",
        language: str = "en",
        speaker: str = "Sofia",
        apply_TN: bool = False,
        device: str = "cuda",
        **kwargs,
    ):
        if speaker not in self.SPEAKER_MAP:
            raise ValueError(f"Invalid speaker: {speaker}, must be one of {list(self.SPEAKER_MAP.keys())}")
        self._language = language
        self._current_speaker = speaker
        self._apply_TN = apply_TN
        super().__init__(model=model, device=device, **kwargs)
        self.setup_tool_calling()

    def _setup_model(self):
        from nemo.collections.tts.models import MagpieTTSModel

        if self._model_name.endswith(".nemo"):
            model = MagpieTTSModel.restore_from(self._model_name, map_location=torch.device(self._device))
        else:
            model = MagpieTTSModel.from_pretrained(self._model_name, map_location=torch.device(self._device))
        model.eval()

        text = "Warming up the Magpie TTS model, this will help the model to respond faster for later requests."
        with torch.no_grad():
            _, _ = model.do_tts(
                text,
                language=self._language,
                apply_TN=self._apply_TN,
                speaker_index=self.SPEAKER_MAP[self._current_speaker],
            )
        torch.cuda.empty_cache()
        return model

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        audio, audio_len = self._model.do_tts(
            text,
            language=self._language,
            apply_TN=self._apply_TN,
            speaker_index=self.SPEAKER_MAP[self._current_speaker],
        )
        audio_len = audio_len.view(-1).item()
        audio = audio.detach().view(-1).cpu().numpy()
        yield audio[:audio_len]

    def setup_tool_calling(self):
        """No tools for now for Magpie TTS service."""
        pass


class EasyMagpieTTSService(BaseNemoTTSService):
    """Text-to-Speech service using EasyMagpieTTS decoder-only model with true streaming audio output.

    Uses streaming_init + streaming_step: audio codes are generated one step at a time and
    decoded in chunks every decode_every_n_frames frames, yielding audio progressively instead
    of waiting for the full sentence to finish.
    """

    def __init__(
        self,
        *,
        model_path: str,
        codec_model_path: str,
        phoneme_tokenizer_path: str,
        context_text: str = "[NO TEXT CONTEXT]",
        context_audio_path: Optional[str] = None,
        decode_every_n_frames: int = 20,
        temperature: float = 0.7,
        topk: int = 80,
        use_cfg: bool = False,
        cfg_scale: float = 2.5,
        max_steps: int = 300,
        device: str = "cuda",
        codec_device: Optional[str] = None,
        streaming: bool = True,
        plugin_path: Optional[str] = None,
        **kwargs,
    ):
        self._model_path = model_path
        self._plugin_path = plugin_path
        self._codec_model_path = codec_model_path
        self._codec_device_override = codec_device
        self._phoneme_tokenizer_path = phoneme_tokenizer_path
        self._context_text = context_text
        self._context_audio_path = context_audio_path
        self._decode_every_n_frames = decode_every_n_frames
        self._temperature = temperature
        self._topk = topk
        self._use_cfg = use_cfg
        self._cfg_scale = cfg_scale
        self._max_steps = max_steps
        self._codec_sample_rate = 22050  # EasyMagpie codec always outputs at 22050 Hz (bandwidth extension)
        self._decode_codec_helper = None
        self._base_context_tensors = None
        self._base_streaming_state = None
        # The pipecat WebSocketTransport frontend player runs at PLAYER_SAMPLE_RATE=24000 Hz.
        # Audio is resampled from the codec's native 22050 Hz to 24000 Hz before sending.
        output_sample_rate = kwargs.pop("sample_rate", 22050)
        super().__init__(model=model_path, device=device, sample_rate=output_sample_rate, streaming=streaming, **kwargs)
        self.setup_tool_calling()

    def _setup_model(self):
        if "EASYMAGPIE_LT_BACKEND" not in os.environ:
            os.environ["EASYMAGPIE_LT_BACKEND"] = "trt"

        # TRT requires libcudart and libnvinfer to be loaded before tensorrt is imported.
        # We preload the CUDA 12 runtime and TRT libs (from nvidia/cuda_runtime package)
        # using RTLD_GLOBAL so that tensorrt_libs/__init__.py finds them already in memory
        # and uses these cu12 builds rather than any cu13 builds that may be on disk.
        import ctypes
        _site_packages = os.path.join(os.path.dirname(os.__file__), "site-packages")
        _cudart12_so = os.path.join(_site_packages, "nvidia", "cuda_runtime", "lib", "libcudart.so.12")
        if os.path.isfile(_cudart12_so):
            ctypes.CDLL(_cudart12_so, mode=ctypes.RTLD_GLOBAL)

        from nemo.collections.tts.models import AudioCodecModel
        from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
        from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
        from nemo.collections.tts.modules.magpietts_modules import CodecHelper
        from omegaconf import open_dict

        logger.info(f"Loading EasyMagpieTTS model from {self._model_path}")
        model_cfg = EasyMagpieTTSInferenceModel.restore_from(self._model_path, return_config=True)
        with open_dict(model_cfg):
            model_cfg.target = "nemo.collections.tts.models.easy_magpietts_inference.EasyMagpieTTSInferenceModel"
            model_cfg.codecmodel_path = self._codec_model_path
            model_cfg.train_ds = None
            model_cfg.validation_ds = None
            model_cfg.run_val_inference = False
            model_cfg.use_utmos = False
            model_cfg.use_meta_init_for_decoder = True
            if getattr(model_cfg, "phoneme_tokenizer", None) is not None:
                model_cfg.phoneme_tokenizer.tokenizer_path = self._phoneme_tokenizer_path

        model = EasyMagpieTTSInferenceModel.restore_from(
            self._model_path,
            override_config_path=model_cfg,
            map_location=torch.device("cpu"),
            # strict=False lets us load DisableTextEmb-style checkpoints whose
            # state dict legitimately omits the backbone's input embedding tensor
            # (it's never used at inference because text tokens go through a
            # separate text embedding outside the backbone).
            strict=False,
        )
        model.use_kv_cache_for_inference = True
        model.eval().to(self._device).float()

        # Nemotron-H SmallMamba checkpoint requires the mamba-ssm /
        # causal-conv1d Triton kernels to be marked opaque to Dynamo, otherwise
        # `torch.compile(model.decoder)` crashes with `KeyError: 'op5'` during
        # Inductor lowering.  Idempotent and safe even if torch.compile is
        # later disabled.
        #
        # The server process must be launched with
        #   LD_PRELOAD=/home/subhankarg/.local/lib/libc_single_threaded_stub.so
        # so the GLIBC 2.32 `__libc_single_threaded` symbol used by mamba-ssm's
        # prebuilt .so resolves.  See easymagpie_minimal_streaming_demo.py
        # header for the full rationale.
        #
        # NOTE on precision: the Gradio demo wraps streaming_step in
        # `torch.autocast(dtype=bfloat16)` and additionally calls
        # `model.decoder.half()` to get an extra ~5-10 ms/step.  This service
        # explicitly disables autocast (line ~1132, ~1362) to match the FP32
        # fused TRT LT engine, so we leave the decoder at FP32 here — halving
        # would produce dtype mismatches against the FP32 input embeddings and
        # break SDPA.  If you ever introduce autocast on this path, the
        # `.half()` can come back.
        is_nemotron_h = getattr(model, "decoder_type", None) == "nemotron_h"
        if is_nemotron_h:
            logger.info("Detected nemotron_h decoder — applying SmallMamba-specific setup")
            _disallow_mamba_kernels_in_graph()

        # Compile the backbone decoder with Triton autotuning but WITHOUT CUDA
        # graph capture.  CUDA graphs are stored in torch._inductor's thread-local
        # storage; even when warmup and inference are pinned to the same thread,
        # variable-length text means new shapes are common and each new shape
        # triggers an expensive (~40s) graph capture on the first request that
        # hits it.  Triton kernel autotuning alone gives the bulk of the speedup
        # without that cost.
        model.decoder = torch.compile(model.decoder, dynamic=True, mode="max-autotune-no-cudagraphs")

        lt_backend = os.environ.get("EASYMAGPIE_LT_BACKEND", "torch").strip().lower()
        if lt_backend == "compile":
            logger.info("Compiling local_transformer with torch.compile(mode='reduce-overhead')")
            compiled_lt = torch.compile(model.local_transformer, mode="reduce-overhead")
            model.local_transformer = compiled_lt
            model._lt_helper.local_transformer = compiled_lt
        elif lt_backend == "trt_fused":
            if self._plugin_path:
                os.environ.setdefault("EASYMAGPIE_CATEGORICAL_PLUGIN_PATH", self._plugin_path)
            model._lt_helper._fused_temperature = self._temperature
            model._lt_helper._fused_topk = self._topk
            logger.info(
                f"trt_fused LT backend active "
                f"(temperature={self._temperature}, topk={self._topk})"
            )

        # Build a separate codec instance dedicated to decoding
        logger.info("Building decode codec helper...")
        codec_model = AudioCodecModel.restore_from(
            self._codec_model_path, strict=False, map_location=torch.device("cpu")
        )
        if hasattr(codec_model, "discriminator"):
            del codec_model.discriminator
        codec_model.freeze()
        _codec_device = self._codec_device_override if self._codec_device_override else self._device
        if _codec_device != self._device:
            logger.info(f"Codec decoder on separate device {_codec_device} (TTS model on {self._device}) — true GPU overlap enabled")
        codec_model = codec_model.to(_codec_device).eval()

        codec_converter = None
        if model._codec_converter is not None:
            vq_new = deepcopy(model._codec_converter.vector_quantizer_new).to(_codec_device).eval()
            codec_converter = VectorQuantizerIndexConverter(
                vector_quantizer_original=codec_model.vector_quantizer,
                vector_quantizer_new=vq_new,
            ).to(_codec_device).eval()

        self._decode_codec_helper = CodecHelper(codec_model=codec_model, codec_converter=codec_converter)
        self._codec_device = _codec_device
        logger.info("Decode codec helper built")

        # Pre-compute and cache context tensors (reused on every sentence)
        self._base_context_tensors = self._build_context_inputs(model)
        logger.info("Context tensors pre-computed and cached")

        # Run warmup AND streaming_init on the persistent _tts_executor's worker
        # thread so torch.compile's Triton compile-worker IPC and any thread-local
        # state are bound to the same thread that will run _generate_audio later.
        # (CUDA graphs are disabled via the no-cudagraphs compile mode — see above —
        # but keeping warmup on the worker thread still avoids the
        # ThreadPoolExecutor-vs-main-thread deadlock we saw with cross-thread
        # compile-worker IPC.)
        _device_type = "cuda" if "cuda" in self._device else "cpu"

        def _warmup_and_init():
            self._run_warmup(model)
            with torch.inference_mode():
                base_state = self._call_streaming_init(model, device_type=_device_type)
            base_state_dynamic = self._clone_state(base_state)
            base_state_static = self._convert_kv_to_static_cache(
                model, base_state, max_gen_steps=self._max_steps + 50
            )
            return base_state_static, base_state_dynamic

        (
            self._base_streaming_state,
            self._base_streaming_state_dynamic,
        ) = self._tts_executor.submit(_warmup_and_init).result()
        logger.info("Warmup + base streaming_init done on TTS worker thread")

        # Thread pool for overlapping codec decoding with autoregressive generation.
        # Codec runs on cuda:0 (separate device) — this thread is independent of
        # the TTS worker thread; CUDA graphs aren't used on the codec.
        self._decode_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        logger.info("Decode executor initialized")

        return model

    def _build_context_inputs(self, model):
        """Encode context text/audio into tensors once at init. Reused for every streaming_init."""
        device = next(model.parameters()).device
        context_text = self._context_text or "[NO TEXT CONTEXT]"
        text_ids = model.tokenizer.encode(context_text, tokenizer_name=model.text_conditioning_tokenizer_name)
        context_text_tokens = torch.tensor([text_ids], dtype=torch.long, device=device)
        context_text_lens = torch.tensor([len(text_ids)], dtype=torch.long, device=device)

        if self._context_audio_path:
            context_audio = model._load_audio_for_inference(self._context_audio_path, model.sample_rate)
            context_audio = model._adjust_audio_to_duration_for_inference(
                context_audio, model.sample_rate, 5.0, model.codec_model_samples_per_frame
            )
            context_audio = context_audio.to(device)
            context_audio_lens = torch.tensor([context_audio.size(1)], dtype=torch.long, device=device)
            with torch.inference_mode():
                context_audio_codes, context_audio_codes_lens = model._codec_helper.audio_to_codes(
                    context_audio, context_audio_lens
                )
        else:
            context_audio_codes = torch.zeros(
                1, model.data_num_audio_codebooks, 0, dtype=torch.long, device=device
            )
            context_audio_codes_lens = torch.zeros(1, dtype=torch.long, device=device)

        return context_audio_codes, context_audio_codes_lens, context_text_tokens, context_text_lens

    def _call_streaming_init(self, model, device_type: str = "cuda"):
        """Run streaming_init with the cached context tensors and return the state.

        Must be called inside torch.inference_mode(). Pass device_type='cuda' when
        Runs in FP32 (no autocast) to match the FP32 fused TRT LT engine — keeping
        the full AR pipeline in one precision avoids dtype mismatches that can
        degrade audio quality.
        """
        ctx = self._base_context_tensors
        with torch.autocast(device_type=device_type, enabled=False):
            state = model.streaming_init(
                context_audio_codes=ctx[0],
                context_audio_codes_lens=ctx[1],
                context_text_tokens=ctx[2],
                context_text_tokens_lens=ctx[3],
                use_cfg=self._use_cfg,
                cfg_scale=self._cfg_scale,
                use_local_transformer=True,
                temperature=self._temperature,
                topk=self._topk,
            )
        return state

    @staticmethod
    def _convert_kv_to_static_cache(model, state, max_gen_steps: int):
        """Replace DynamicCache in state with a pre-allocated StaticCache.

        DynamicCache grows via torch.cat each step (changes tensor address),
        which is incompatible with CUDA graph capture.  StaticCache writes
        in-place at cache_position so shapes stay static and CUDA graphs work.

        NemotronH backbones use a HybridMambaAttentionDynamicCache (mix of
        attention KV + Mamba2 SSM state) that doesn't have a `.layers` list.
        StaticCache only covers the attention path; the Mamba2 state is already
        fixed-shape inside the hybrid cache.  For these checkpoints we leave the
        cache untouched — torch.compile can still trace it (slightly less optimal
        per-step but functionally correct).
        """
        from transformers import StaticCache
        dyn = state.past_key_values
        if not hasattr(dyn, "layers"):
            return state  # HybridMambaAttentionDynamicCache (NemotronH) — keep as-is
        context_len = state.cache_seq_len
        max_cache_len = context_len + max_gen_steps
        device = state.config.device
        # Use the actual KV dtype (fp16 under autocast) so StaticCache buffers
        # match what streaming_step writes during generation.
        dtype = dyn.layers[0].keys.dtype

        static = StaticCache(
            config=model.transformer_backend_config,
            max_cache_len=max_cache_len,
        )
        first = dyn.layers[0]
        static.early_initialization(
            batch_size=first.keys.shape[0],
            num_heads=first.keys.shape[1],
            head_dim=first.keys.shape[3],
            dtype=dtype,
            device=device,
        )
        for dyn_layer, static_layer in zip(dyn.layers, static.layers):
            static_layer.keys[:, :, :context_len, :] = dyn_layer.keys
            static_layer.values[:, :, :context_len, :] = dyn_layer.values

        state.past_key_values = static
        return state

    @staticmethod
    def _snapshot_state_for_reset(state):
        """Snapshot a StreamingState for later in-place restore.

        Used to reset _live_streaming_state at the start of every TTS call without
        allocating new tensors.  Tensor addresses in the live state stay stable
        across calls — required for torch.compile + CUDA graphs (graphs capture
        specific GPU addresses; a new allocation invalidates the graph and
        triggers a costly re-capture).

        For each field we store either the int/bool value, a clone of the tensor,
        or for StaticCache: clones of all layer keys/values tensors.  StreamingConfig
        is treated as immutable and stored by reference.
        """
        from dataclasses import fields as dc_fields

        snap = {}
        for f in dc_fields(state):
            if f.name == "config":
                snap[f.name] = getattr(state, f.name)
                continue
            v = getattr(state, f.name)
            if v is None:
                snap[f.name] = None
            elif isinstance(v, torch.Tensor):
                snap[f.name] = v.clone()
            elif isinstance(v, list):
                snap[f.name] = list(v)  # shallow copy (lists usually start empty)
            else:
                # StaticCache or primitive (int, bool, ...).  Detect StaticCache by
                # presence of `.layers`.
                layers = getattr(v, "layers", None)
                if layers is not None and hasattr(layers[0], "keys"):
                    snap[f.name] = [
                        (layer.keys.clone(), layer.values.clone()) for layer in layers
                    ]
                else:
                    snap[f.name] = v
        return snap

    @staticmethod
    def _reset_state_in_place(state, snap):
        """Restore state to match snap IN-PLACE (preserves tensor addresses).

        Tensor fields are reset via .copy_(), StaticCache buffers via per-layer
        .copy_(), lists via .clear() + .extend(), primitives via setattr.
        """
        from dataclasses import fields as dc_fields

        for f in dc_fields(state):
            if f.name == "config":
                continue
            live_v = getattr(state, f.name)
            snap_v = snap[f.name]

            if live_v is None and snap_v is None:
                continue
            if live_v is None or snap_v is None:
                # Shape changed — must reassign (rare; no addresses to preserve).
                setattr(state, f.name, snap_v)
                continue

            if isinstance(live_v, torch.Tensor) and isinstance(snap_v, torch.Tensor):
                # Shape match → in-place .copy_() preserves the tensor's GPU
                # address (matters for fields the compiled decoder is captured
                # against).  Shape mismatch → the field was REASSIGNED inside
                # streaming_step (e.g. last_hidden goes from (1, ctx_len, d) to
                # (1, 1, d) after the first step), so a fresh clone is the only
                # safe restore.  These fields aren't compiled-graph inputs;
                # they're streaming_step bookkeeping — overhead is microseconds.
                if live_v.shape == snap_v.shape and live_v.dtype == snap_v.dtype:
                    live_v.copy_(snap_v)
                else:
                    setattr(state, f.name, snap_v.clone())
            elif isinstance(live_v, list) and isinstance(snap_v, list):
                live_v.clear()
                live_v.extend(snap_v)
            elif hasattr(live_v, "layers") and isinstance(snap_v, list):
                # StaticCache: copy each layer's keys/values in-place.
                for layer, (k_snap, v_snap) in zip(live_v.layers, snap_v):
                    layer.keys.copy_(k_snap)
                    layer.values.copy_(v_snap)
            else:
                setattr(state, f.name, snap_v)

    @staticmethod
    def _clone_state(state):
        """Clone a StreamingState without deepcopy.

        deepcopy of DynamicCache (HF transformers 4.38+) can deadlock when called
        from a background thread because it interacts with torch.compile's internal
        guard/compilation machinery. This function clones all tensors explicitly and
        copies primitive values directly, bypassing deepcopy entirely.
        """
        from dataclasses import fields as dc_fields

        def _clone_val(v):
            if isinstance(v, torch.Tensor):
                return v.clone()
            if isinstance(v, list):
                return [_clone_val(x) for x in v]
            if isinstance(v, tuple):
                return tuple(_clone_val(x) for x in v)
            # HF transformers DynamicCache (past_key_values)
            try:
                from transformers.cache_utils import DynamicCache
                if isinstance(v, DynamicCache):
                    new_cache = DynamicCache()
                    for layer in v.layers:
                        from transformers.cache_utils import DynamicLayer
                        new_layer = DynamicLayer()
                        if getattr(layer, 'is_initialized', False):
                            new_layer.dtype = layer.dtype
                            new_layer.device = layer.device
                            new_layer.keys = layer.keys.clone()
                            new_layer.values = layer.values.clone()
                            new_layer.is_initialized = True
                        new_cache.layers.append(new_layer)
                    new_cache._seen_tokens = getattr(v, '_seen_tokens', 0)
                    return new_cache
            except ImportError:
                pass
            # NemotronH hybrid cache (HybridMambaAttentionDynamicCache) — has
            # attention KV tensors + Mamba2 SSM state tensors but no `.layers`
            # list compatible with DynamicCache.  deepcopy works here because
            # the deepcopy-deadlock issue is specific to DynamicCache's
            # torch.compile guard interaction, not the hybrid cache.
            if type(v).__name__ == "HybridMambaAttentionDynamicCache":
                return deepcopy(v)
            # For legacy tuple-based KV cache (pre-4.38 transformers)
            return v  # primitives (int, bool, float, str, torch.device, etc.)

        # Clone StreamingConfig (immutable after init — shallow copy is safe for
        # primitives and tensors are cloned via _clone_val)
        from nemo.collections.tts.models.easy_magpietts_inference import StreamingConfig
        cfg = state.config
        new_cfg = StreamingConfig(
            **{f.name: _clone_val(getattr(cfg, f.name)) for f in dc_fields(cfg)}
        )

        # Clone StreamingState
        from nemo.collections.tts.models.easy_magpietts_inference import StreamingState
        return StreamingState(
            **{f.name: _clone_val(getattr(state, f.name)) for f in dc_fields(state)
               if f.name != 'config'},
            config=new_cfg,
        )

    def _run_warmup(self, model):
        device = next(model.parameters()).device
        main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
        # Use the bot's greeting as warmup text so the CUDA graphs captured during
        # warmup cover the exact shapes the first user-facing TTS call will need.
        # Otherwise the first real request pays a ~40s graph-capture penalty for
        # the new text length.
        token_ids = model.tokenizer.encode(
            "Hi, I'm Lisa, your helpful AI assistant. How can I help you today?",
            tokenizer_name=main_tokenizer_name,
        )
        token_ids.append(model.eos_id)
        pending = deque(token_ids)
        steps = 0
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        with torch.inference_mode():
            state = self._call_streaming_init(model, device_type=device_type)
            # StaticCache required for CUDA graph capture (DynamicCache grows via
            # torch.cat which changes tensor addresses, incompatible with CUDA graphs).
            # Use max_gen_steps matching _generate_audio so torch.compile sees the same
            # StaticCache buffer shape during warmup as during inference — different shapes
            # trigger a retrace from the background thread which deadlocks the compile worker.
            state = self._convert_kv_to_static_cache(model, state, max_gen_steps=self._max_steps + 50)
            with torch.autocast(device_type=device_type, enabled=False):
                # Cap at 80 steps — letting warmup run the FULL greeting caused
                # a deadlock (every thread stuck on futex after the first real
                # TTS request), likely from CUDA graph capture state interacting
                # badly with later AR steps near EOS.  80 steps is enough to
                # warm Triton kernels and capture the steady-state graphs.
                while not bool(state.finished.all()) and steps < 80:
                    tok = pending.popleft() if pending else None
                    text_tokens = torch.tensor([tok], dtype=torch.long, device=device) if tok is not None else None
                    # Signal new step to CUDA graph replay (required when replaying
                    # compiled graphs with changing dynamic values like cache_position).
                    torch.compiler.cudagraph_mark_step_begin()
                    state, _, _ = model.streaming_step(state=state, text_tokens=text_tokens, use_inference_mode=True)
                    steps += 1
        torch.cuda.empty_cache()

    def _decode_new_chunk(self, accumulated_codes, last_emitted_sample_idx):
        codec_dev = getattr(self, "_codec_device", None)
        if codec_dev is not None:
            # Move to codec device BEFORE _prepare_codes_for_decode so every op in this
            # executor thread runs on codec_dev, never touching the TTS AR device's default
            # CUDA stream.  _prepare_codes_for_decode is pure tensor ops (no model params)
            # so it runs correctly on whichever device the input tensor is on.
            accumulated_codes = accumulated_codes.to(codec_dev)
        codes_lens = torch.tensor(
            [accumulated_codes.size(-1)], dtype=torch.long, device=accumulated_codes.device
        )
        pred_codes, pred_codes_lens = self._model._prepare_codes_for_decode(accumulated_codes, codes_lens)
        audio, audio_len, _ = self._decode_codec_helper.codes_to_audio(pred_codes, pred_codes_lens)
        full_wav = audio[0, : audio_len[0]].detach().float().cpu().numpy()
        if full_wav.shape[0] <= last_emitted_sample_idx:
            return np.zeros((0,), dtype=np.float32), last_emitted_sample_idx
        new_samples = full_wav[last_emitted_sample_idx:]
        if self._codec_sample_rate != self.sample_rate:
            import soxr
            new_samples = soxr.resample(new_samples, self._codec_sample_rate, self.sample_rate)
        return new_samples, full_wav.shape[0]

    def _decode_new_chunk_no_grad(self, accumulated_codes, last_emitted_sample_idx):
        """Wrapper that calls _decode_new_chunk inside torch.no_grad so the
        codec decode can safely run in a ThreadPoolExecutor thread (which
        doesn't inherit the torch.inference_mode context from the caller)."""
        with torch.no_grad():
            return self._decode_new_chunk(accumulated_codes, last_emitted_sample_idx)

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        model = self._model
        device = next(model.parameters()).device

        main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
        token_ids = model.tokenizer.encode(text, tokenizer_name=main_tokenizer_name)
        token_ids.append(model.eos_id)
        pending = deque(token_ids)

        accumulated_codes = None
        last_emitted_sample_idx = 0
        steps = 0
        num_decoding_steps = 0
        _max_codec_frames = 64
        _spf = getattr(model, 'codec_model_samples_per_frame', 640)
        pending_decode_future = None

        device_type = "cuda" if "cuda" in self._device else "cpu"
        # Full FP32 path: no autocast, so the backbone, KV cache, and FP32 fused
        # TRT LT engine all operate in matching precision.  Slower than FP16 but
        # numerically faithful to the original eager model.
        with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=False):
            # Clone the per-request state from the DynamicCache base then convert
            # to a StaticCache.  StaticCache is required because DynamicCache grows
            # via torch.cat (changing tensor addresses each step), which prevents
            # torch.compile from caching kernels for the inner attention call.
            state = self._clone_state(self._base_streaming_state_dynamic)
            state = self._convert_kv_to_static_cache(
                model, state, max_gen_steps=self._max_steps + 50
            )

            while not bool(state.finished.all()) and steps < self._max_steps:
                tok = pending.popleft() if pending else None
                text_tokens = torch.tensor([tok], dtype=torch.long, device=device) if tok is not None else None

                # Signal new step to CUDA graph replay so the compiled graph
                # correctly updates dynamic values (e.g. cache_position) each step.
                torch.compiler.cudagraph_mark_step_begin()
                state, audio_codes, _ = model.streaming_step(
                    state=state, text_tokens=text_tokens, use_inference_mode=True
                )
                steps += 1

                if audio_codes is not None:
                    audio_codes = audio_codes.to(device=device, dtype=torch.long)
                    accumulated_codes = (
                        audio_codes if accumulated_codes is None
                        else torch.cat([accumulated_codes, audio_codes], dim=-1)
                    )
                    num_decoding_steps += 1  # count only audio-producing steps so first batch has same size as subsequent

                if accumulated_codes is not None and num_decoding_steps >= self._decode_every_n_frames:
                    num_decoding_steps = 0

                    if pending_decode_future is not None:
                        chunk, last_emitted_sample_idx = pending_decode_future.result()
                        if chunk.size > 0:
                            yield chunk

                    if last_emitted_sample_idx > 0 and accumulated_codes.size(-1) > _max_codec_frames:
                        _trim = accumulated_codes.size(-1) - _max_codec_frames
                        last_emitted_sample_idx = max(0, last_emitted_sample_idx - _trim * _spf)
                        accumulated_codes = accumulated_codes[..., _trim:]

                    pending_decode_future = self._decode_executor.submit(
                        self._decode_new_chunk_no_grad,
                        accumulated_codes,
                        last_emitted_sample_idx,
                    )

        if pending_decode_future is not None:
            chunk, last_emitted_sample_idx = pending_decode_future.result()
            if chunk.size > 0:
                yield chunk
            if last_emitted_sample_idx > 0 and accumulated_codes is not None and accumulated_codes.size(-1) > _max_codec_frames:
                _trim = accumulated_codes.size(-1) - _max_codec_frames
                last_emitted_sample_idx = max(0, last_emitted_sample_idx - _trim * _spf)
                accumulated_codes = accumulated_codes[..., _trim:]

        if accumulated_codes is not None:
            chunk, _ = self._decode_new_chunk_no_grad(accumulated_codes, last_emitted_sample_idx)
            if chunk.size > 0:
                yield chunk

    def setup_tool_calling(self):
        pass


class EasyMagpieSmallMambaVllmService(EasyMagpieTTSService):
    """vLLM-backed variant of :class:`EasyMagpieTTSService`.

    The AR backbone (NemotronH SmallMamba) + Local Transformer run in a
    separate ``vllm_omni_env`` sidecar process and are reached over HTTP.
    See ``HANDOFF_vllm_smallmamba_streaming.md`` for the rationale (vLLM 0.19.1
    cannot coexist with the legacy nemo_virtual_environment that this agent
    uses).

    What still runs locally:
      * Phoneme tokenizer (text -> subword IDs to send to the sidecar).
      * Codec **encoder** (context audio -> 16-codebook codes to send).
      * Codec **decoder** (audio code frames from the sidecar -> waveform).
      * The existing chunked-decode + sliding-window emit loop from the
        parent class.

    What's replaced:
      * The autoregressive loop -- ``_generate_audio`` no longer calls
        ``model.streaming_step``. Instead it POSTs to ``/tts/stream`` on the
        sidecar and yields audio chunks as code frames arrive.

    v1 reuses the parent's ``_setup_model`` so we still load the AR weights
    onto the local GPU even though they're unused on this code path. Trades
    GPU memory for code reuse (tokenizer + codec helpers + codes-prep
    method are all already wired by the parent). Future work: a thinner
    setup that skips the AR backbone load entirely.
    """

    def __init__(
        self,
        *,
        vllm_server_url: str,
        max_frames_per_request: int = 300,
        **kwargs,
    ) -> None:
        # Defer the vLLM client construction until _setup_model (which runs
        # on the TTS executor thread) so a misconfigured sidecar surfaces in
        # the agent's startup logs alongside the AR model load, not silently
        # at the first user utterance.
        self._vllm_server_url = vllm_server_url
        self._vllm_client = None
        self._max_frames_per_request = max_frames_per_request
        # Pre-flattened context inputs we'll send to the sidecar every call.
        self._sidecar_ctx_codes: Optional[list[list[int]]] = None
        self._sidecar_ctx_text: Optional[list[int]] = None
        super().__init__(**kwargs)

    def _run_warmup(self, model):
        """Skip the parent's warmup loop entirely.

        The parent's ``_run_warmup`` runs ~10-20 streaming_step iterations,
        which on the SmallMamba checkpoint triggers a per-static-shape
        TensorRT engine build for the local transformer at each new shape
        (each build takes ~30 s, total ~5-10 min). On this code path we
        never call ``streaming_step`` -- the AR loop happens entirely in
        the vLLM sidecar -- so these engines are dead weight. Skipping the
        warmup cuts service startup from ~10 min to ~80 s.

        Note: the agent's local-side ``streaming_state`` and TRT engine
        cache remain unbuilt. If you ever want to fall back to the local
        AR path, remove this override.
        """
        return

    def _setup_model(self):
        # Re-use the parent's full setup. This loads the AR backbone, the LT,
        # the codec helpers, builds the cached context tensors. The
        # warmup-loop step has been short-circuited above; ``streaming_init``
        # still runs (cheap, ~30 s).
        model = super()._setup_model()

        # Build the HTTP client now so a missing sidecar fails at startup,
        # not at the first user utterance.
        from nemo.agents.voice_agent.pipecat.services.nemo.easymagpie_vllm_client import (
            build_client_from_url,
        )
        logger.info(
            f"Connecting to vLLM sidecar at {self._vllm_server_url} ..."
        )
        self._vllm_client = build_client_from_url(self._vllm_server_url, ping_first=True)
        logger.info("vLLM sidecar reachable.")

        # Wire the CAS encoder on the sidecar with our BPE subword vocab.
        # Without this the CAS encoder's subword->char map is empty and
        # the first /tts/stream POST triggers a CUDA out-of-range assert.
        # Uses the SAME tokenizer that _build_context_inputs encoded the
        # context_text with, namely ``text_conditioning_tokenizer_name``.
        ctx_tok_name = model.text_conditioning_tokenizer_name
        ctx_tokenizer = model.tokenizer.tokenizers[ctx_tok_name]
        self._vllm_client.wire_tokenizer(
            subword_vocab=ctx_tokenizer.get_vocab(),
            bos_id=model.bos_id,
            eos_id=model.eos_id,
            cfg_unk_token_id=model.cfg_unk_token_id,
            subword_padding_idx=model.tokenizer.pad,
        )
        logger.info(
            "Sidecar CAS encoder wired with %d-token subword vocab.",
            len(ctx_tokenizer.get_vocab()),
        )

        # The parent's _build_context_inputs already encoded the context audio
        # into 16-codebook codes and the context text into subword IDs.
        # Flatten to plain Python lists for HTTP serialization. These are
        # identical across utterances, so we cache them.
        ctx_codes, ctx_codes_lens, ctx_text_tokens, ctx_text_lens = self._base_context_tensors
        logger.info(
            "[em-vllm] raw ctx_codes shape=%s dtype=%s min=%d max=%d  ctx_text shape=%s",
            tuple(ctx_codes.shape), ctx_codes.dtype,
            int(ctx_codes.min()), int(ctx_codes.max()),
            tuple(ctx_text_tokens.shape),
        )
        # Apply the same context-codes preparation the production
        # ``prepare_context_tensors`` path does (and that the A4'
        # equivalence test mirrored):
        #   1. codec_converter map (codec-native vocab -> model vocab)
        #   2. add_special_tokens (BOS prefix + EOS suffix)
        #   3. stack_codes (B, C=8, T) -> (B, C*S=16, T/S)
        # The vLLM subclass's build_prefill_combined_embeddings expects the
        # output of step 3. Without these the audio-code IDs exceed the
        # codebook vocab and trigger an async CUDA out-of-range assert.
        from nemo.collections.tts.modules.magpietts_modules import add_special_tokens
        if getattr(model, "_codec_converter", None) is not None:
            ctx_codes = model._codec_converter.convert_original_to_new(
                audio_tokens=ctx_codes, audio_lens=ctx_codes_lens,
            ).long()
        ctx_codes, ctx_codes_lens = add_special_tokens(
            codes=ctx_codes, codes_len=ctx_codes_lens,
            bos_id=model.context_audio_bos_id,
            eos_id=model.context_audio_eos_id,
        )
        ctx_codes, ctx_codes_lens = model.stack_codes(
            ctx_codes, ctx_codes_lens,
            model.context_audio_bos_id, model.context_audio_eos_id,
            model.frame_stacking_factor, model.num_audio_codebooks,
        )
        logger.info(
            "[em-vllm] prepared ctx_codes shape=%s min=%d max=%d",
            tuple(ctx_codes.shape), int(ctx_codes.min()), int(ctx_codes.max()),
        )
        codes_cpu = ctx_codes.detach().to("cpu").long()[0]   # (16, T_ctx_stacked)
        self._sidecar_ctx_codes = codes_cpu.tolist()
        # ctx_text_tokens shape: (1, L). Send as list[int].
        text_cpu = ctx_text_tokens.detach().to("cpu").long()[0]
        self._sidecar_ctx_text = text_cpu.tolist()
        logger.info(
            f"Cached sidecar context: codes ({len(self._sidecar_ctx_codes)} tables x "
            f"{len(self._sidecar_ctx_codes[0])} frames) + text ({len(self._sidecar_ctx_text)} tokens)"
        )
        return model

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        """Stream audio for ``text`` by:
        1. Tokenizing ``text`` with the local model's tokenizer.
        2. POSTing to the sidecar with the cached context + the per-utterance
           text token IDs.
        3. Accumulating returned audio code frames, calling the parent's
           ``_decode_new_chunk`` every ``decode_every_n_frames`` frames to
           emit audio chunks.
        """
        model = self._model
        device = next(model.parameters()).device

        # Tokenize the utterance with the model's main phoneme tokenizer
        # (the one production ``_generate_audio`` uses on its
        # ``streaming_step(text_tokens=...)`` path). The sidecar's
        # ``embed_input_ids`` hook consumes one of these IDs per AR step,
        # looks it up via ``phoneme_embeddings[0]``, and adds it to the
        # audio-code embedding -- matching production's additive
        # ``next_input = audio_emb + phoneme_emb`` layout.
        main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
        phoneme_token_ids = model.tokenizer.encode(
            text, tokenizer_name=main_tokenizer_name,
        )
        phoneme_token_ids.append(model.eos_id)
        # Diagnostic so we can spot out-of-range IDs without re-running.
        logger.info(
            f"[em-vllm] tokenizer={main_tokenizer_name} phoneme stream: "
            f"len={len(phoneme_token_ids)} "
            f"min={int(min(phoneme_token_ids))} "
            f"max={int(max(phoneme_token_ids))} "
            f"eos_id={int(model.eos_id)} "
            f"text_tokenizer_keys={list(model.cfg.text_tokenizers.keys())}"
        )

        # The sidecar's CAS-encoder path takes the cached context-text
        # token IDs (built from speaker/language tag at setup time). It
        # is NOT the place to send per-utterance phonemes -- those go
        # via ``phoneme_token_ids`` and a separate embedding table.
        sidecar_text = list(self._sidecar_ctx_text)

        accumulated_codes: Optional[torch.Tensor] = None
        last_emitted_sample_idx = 0
        num_decoding_steps = 0
        _max_codec_frames = 64
        _spf = getattr(model, 'codec_model_samples_per_frame', 640)
        pending_decode_future = None

        with torch.inference_mode():
            # Iterate over audio code frames streamed from the sidecar.
            # max_frames is sized to give the AR a little headroom past
            # the EOS so the tail audio (silence / decay) lands; the
            # sidecar's _streaming_eos_seen flag (set when the EOS
            # phoneme token is consumed) provides the tighter early-exit
            # signal in practice.
            # Pull the model-specific constants production's streaming_init
            # reads from the .nemo config. The sidecar's state machine needs
            # these to replicate production semantics.
            #   - audio_bos_id / audio_eos_id: AudioToken special IDs
            #   - phoneme_bos_id / phoneme_eos_id: from phoneme_tokenizer
            #   - streaming_speech_delay / streaming_phonemes_delay: from
            #     selected training_mode (e.g. "streaming_3_5" -> 3/5).
            training_mode = model.mode_name_to_mode[model.default_inference_mode]
            for frame_codes in self._vllm_client.stream_frames(
                context_audio_codes=self._sidecar_ctx_codes,
                context_text_token_ids=sidecar_text,
                phoneme_token_ids=phoneme_token_ids,
                text_eos_id=int(model.eos_id),
                audio_bos_id=int(model.audio_bos_id),
                audio_eos_id=int(model.audio_eos_id),
                phoneme_bos_id=int(model.phoneme_tokenizer.bos_token_id),
                phoneme_eos_id=int(model.phoneme_tokenizer.eos_token_id),
                streaming_speech_delay=int(training_mode.streaming_speech_delay),
                streaming_phonemes_delay=int(training_mode.streaming_phonemes_delay),
                # Sidecar audio_eos detection is the real early-exit; this
                # is just a safety cap. Use a generous per-token budget so
                # natural-length English utterances reach EOS organically.
                # Roughly 8 stacked frames per subword (with frame_stacking
                # factor 2, that's ~16 codec frames = ~0.64 s of audio at
                # 25 fps -- plenty for any subword's worst case).
                max_frames=min(
                    self._max_frames_per_request,
                    len(phoneme_token_ids) * 8 + 80,
                ),
            ):
                # Each frame is a list of 16 ints. Stack into a (1, 16, 1)
                # tensor and append along the frame (time) axis.
                frame_t = torch.tensor(
                    frame_codes, dtype=torch.long, device=device,
                ).view(1, -1, 1)
                accumulated_codes = (
                    frame_t if accumulated_codes is None
                    else torch.cat([accumulated_codes, frame_t], dim=-1)
                )
                num_decoding_steps += 1

                if num_decoding_steps >= self._decode_every_n_frames:
                    num_decoding_steps = 0

                    if pending_decode_future is not None:
                        chunk, last_emitted_sample_idx = pending_decode_future.result()
                        if chunk.size > 0:
                            yield chunk

                    if last_emitted_sample_idx > 0 and accumulated_codes.size(-1) > _max_codec_frames:
                        _trim = accumulated_codes.size(-1) - _max_codec_frames
                        last_emitted_sample_idx = max(
                            0, last_emitted_sample_idx - _trim * _spf
                        )
                        accumulated_codes = accumulated_codes[..., _trim:]

                    pending_decode_future = self._decode_executor.submit(
                        self._decode_new_chunk_no_grad,
                        accumulated_codes,
                        last_emitted_sample_idx,
                    )

        # Drain any pending decode + emit the tail.
        if pending_decode_future is not None:
            chunk, last_emitted_sample_idx = pending_decode_future.result()
            if chunk.size > 0:
                yield chunk
            if last_emitted_sample_idx > 0 and accumulated_codes is not None \
                    and accumulated_codes.size(-1) > _max_codec_frames:
                _trim = accumulated_codes.size(-1) - _max_codec_frames
                last_emitted_sample_idx = max(
                    0, last_emitted_sample_idx - _trim * _spf
                )
                accumulated_codes = accumulated_codes[..., _trim:]

        if accumulated_codes is not None:
            chunk, _ = self._decode_new_chunk_no_grad(
                accumulated_codes, last_emitted_sample_idx
            )
            if chunk.size > 0:
                yield chunk


def get_tts_service_from_config(config: DictConfig, audio_logger: Optional[AudioLogger] = None) -> BaseNemoTTSService:
    """Get the TTS service from the configuration.

    Args:
        config: The DictConfig object containing the TTS configuration.
        audio_logger: The audio logger to use for audio logging.
    Returns:
        The TTS service.
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    model = config.get("model", None)
    device = config.get("device", "cuda")
    if config.get("type", None) != "nemo":
        raise ValueError(f"Invalid TTS type: {config.get('type', None)}, only 'nemo' is supported")
    if model is None:
        raise ValueError("Model is required for Nemo TTS service")

    text_aggregator = SimpleSegmentedTextAggregator(
        punctuation_marks=config.get("extra_separator", None),
        ignore_marks=config.get("ignore_strings", None),
        min_sentence_length=config.get("min_sentence_length", 5),
        use_legacy_eos_detection=config.get("use_legacy_eos_detection", False),
    )

    if model == "fastpitch-hifigan":
        return NeMoFastPitchHiFiGANTTSService(
            fastpitch_model=config.get("main_model_id", None),
            hifigan_model=config.get("sub_model_id", None),
            device=device,
            text_aggregator=text_aggregator,
            think_tokens=config.get("think_tokens", None),
            audio_logger=audio_logger,
            ignore_strings=config.get("ignore_strings", None),
        )
    elif model == "magpie":
        return MagpieTTSService(
            model=config.get("main_model_id", None),
            language=config.get("language", "en"),
            speaker=config.get("speaker", "Sofia"),
            apply_TN=config.get("apply_TN", False),
            device=device,
            text_aggregator=text_aggregator,
            think_tokens=config.get("think_tokens", None),
            audio_logger=audio_logger,
            ignore_strings=config.get("ignore_strings", None),
        )
    elif model == "kokoro":
        return KokoroTTSService(
            model=config.get("main_model_id", "hexgrad/Kokoro-82M"),
            voice=config.get("sub_model_id", "af_heart"),
            device=device,
            speed=config.get("speed", 1.0),
            text_aggregator=text_aggregator,
            think_tokens=config.get("think_tokens", None),
            sample_rate=24000,
            audio_logger=audio_logger,
            ignore_strings=config.get("ignore_strings", None),
        )
    elif model == "easy_magpie_smallmamba_vllm":
        # SmallMamba via a separate vLLM sidecar (vllm_omni_env).  Tokenizer
        # + codec live locally; the AR loop is HTTP-streamed from the sidecar.
        return EasyMagpieSmallMambaVllmService(
            vllm_server_url=config.get("vllm_server_url",
                                       "http://127.0.0.1:18765"),
            max_frames_per_request=config.get("max_steps", 300),
            model_path=config.get("main_model_id"),
            codec_model_path=config.get("codec_model_path"),
            phoneme_tokenizer_path=config.get("phoneme_tokenizer_path"),
            context_text=config.get("context_text", "[NO TEXT CONTEXT]"),
            context_audio_path=config.get("context_audio_path", None),
            decode_every_n_frames=config.get("decode_every_n_frames", 6),
            temperature=config.get("temperature", 0.7),
            topk=config.get("topk", 80),
            use_cfg=config.get("use_cfg", False),
            cfg_scale=config.get("cfg_scale", 2.5),
            max_steps=config.get("max_steps", 300),
            device=device,
            codec_device=config.get("codec_device", None),
            streaming=config.get("streaming", True),
            plugin_path=config.get("plugin_path", None),
            text_aggregator=text_aggregator,
            think_tokens=config.get("think_tokens", None),
            audio_logger=audio_logger,
            ignore_strings=config.get("ignore_strings", None),
        )
    elif model in ("easy_magpie", "easy_magpie_smallmamba"):
        # Both names route to EasyMagpieTTSService; the SmallMamba-specific
        # setup (Mamba kernel disable, fp16 decoder) is auto-applied inside
        # _setup_model based on the loaded checkpoint's decoder_type.
        return EasyMagpieTTSService(
            model_path=config.get("main_model_id"),
            codec_model_path=config.get("codec_model_path"),
            phoneme_tokenizer_path=config.get("phoneme_tokenizer_path"),
            context_text=config.get("context_text", "[NO TEXT CONTEXT]"),
            context_audio_path=config.get("context_audio_path", None),
            decode_every_n_frames=config.get("decode_every_n_frames", 6),
            temperature=config.get("temperature", 0.7),
            topk=config.get("topk", 80),
            use_cfg=config.get("use_cfg", False),
            cfg_scale=config.get("cfg_scale", 2.5),
            max_steps=config.get("max_steps", 300),
            device=device,
            codec_device=config.get("codec_device", None),
            streaming=config.get("streaming", True),
            plugin_path=config.get("plugin_path", None),
            text_aggregator=text_aggregator,
            think_tokens=config.get("think_tokens", None),
            audio_logger=audio_logger,
            ignore_strings=config.get("ignore_strings", None),
        )
    else:
        raise ValueError(
            f"Invalid model: {model}, only 'fastpitch-hifigan', 'magpie', "
            "'kokoro', 'easy_magpie', 'easy_magpie_smallmamba', and "
            "'easy_magpie_smallmamba_vllm' are supported"
        )
