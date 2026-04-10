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
import inspect
import os
import queue as stdlib_queue
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from copy import deepcopy  # kept for backward compat, replaced by _clone_state in EasyMagpieTTSService
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
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        logger.info(f"Initializing TTS service with model: {model} and device: {device}")
        self._model_name = model
        self._device = device
        self._model = self._setup_model()
        self._think_tokens = think_tokens
        self._audio_logger = audio_logger
        if think_tokens is not None:
            assert (
                isinstance(think_tokens, list) and len(think_tokens) == 2
            ), f"think_tokens must be a list of two strings, but got type {type(think_tokens)}: {think_tokens}"
        self._ignore_strings = set(ignore_strings) if ignore_strings is not None else None
        # Background processing infrastructure - stdlib queues to avoid asyncio deadlocks
        self._tts_queue = stdlib_queue.Queue()
        # Single-worker executor to overlap codec decode with next generation batch
        self._decode_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="em_decode"
        )
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

    def can_generate_metrics(self) -> bool:
        """If the TTS service can generate metrics."""
        return True

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
        self._tts_queue.put(None)  # Signal to stop processing

        if self._processing_task:
            await self.cancel_task(self._processing_task)
            self._processing_task = None

    def _tts_processor(self):
        """Background processor that handles TTS generation calls."""
        try:
            while self._processing_running:
                try:
                    request = self._tts_queue.get()  # blocks until a request arrives

                    if request is None:  # Stop signal
                        logger.debug("Received stop signal in TTS background processor")
                        break

                    text, request_id = request
                    logger.debug(f"Processing TTS request for text: [{text}]")

                    # Get the response queue for this request (direct dict lookup — thread-safe in CPython)
                    response_queue = self._pending_requests.get(request_id)

                    if response_queue is None:
                        logger.warning(f"No response queue found for request {request_id}")
                        continue

                    # Process TTS generation — push each chunk directly into a stdlib
                    # queue as it's produced. run_tts awaits each item via
                    # asyncio.to_thread(q.get), which never blocks the event loop and
                    # never deadlocks regardless of what pipecat does with the consumer.
                    try:
                        response_queue.put(('streaming_start', None))
                        chunk_idx = 0
                        for chunk in self._generate_audio(text):
                            logger.debug(f"[TTS] putting chunk {chunk_idx}, size={chunk.shape}")
                            response_queue.put(('chunk', chunk))
                            chunk_idx += 1
                        logger.debug(f"[TTS] all {chunk_idx} chunks generated, sending done")
                        response_queue.put(('streaming_done', None))
                    except Exception as e:
                        import traceback
                        logger.error(f"Error in TTS generation: {e}\n{traceback.format_exc()}")
                        response_queue.put(('error', e))

                except Exception as e:
                    logger.error(f"Error in background TTS processor: {e}")

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
            await asyncio.to_thread(self._tts_processor)
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
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            # Increment turn index at the start of agent speaking (only if speaker changed)
            if self._audio_logger is not None:
                self._audio_logger.increment_turn_index(speaker="agent")

            # Generate unique request ID

            request_id = str(uuid.uuid4())

            # Create response queue for this specific request (stdlib queue — no event loop needed)
            request_queue = stdlib_queue.Queue()
            self._pending_requests[request_id] = request_queue

            try:
                pipeline_start = time.time()
                pipeline_chunks_sent = 0
                # Queue the TTS request for background processing
                self._tts_queue.put((text, request_id))

                # Wait for the first message (streaming_start or error)
                result = await asyncio.to_thread(request_queue.get)
                status, data = result

                if status == 'error':
                    logger.error(f"{self} TTS generation error: {data}")
                    yield ErrorFrame(error=f"TTS generation error: {str(data)}")
                    return

                if status == 'streaming_start':
                    # Chunks arrive one at a time from the background thread
                    await self.start_tts_usage_metrics(text)
                    all_audio_bytes = b""
                    if self._audio_logger is not None and self._audio_logger.first_audio_timestamp is None:
                        self._audio_logger.first_audio_timestamp = datetime.now()

                    first_chunk = True
                    while True:
                        result = await asyncio.to_thread(request_queue.get)
                        chunk_status, chunk_data = result

                        if chunk_status == 'streaming_done':
                            break
                        if chunk_status == 'error':
                            logger.error(f"{self} TTS generation error: {chunk_data}")
                            yield ErrorFrame(error=f"TTS generation error: {str(chunk_data)}")
                            return

                        audio_chunk = chunk_data
                        if audio_chunk is None:
                            break

                        if first_chunk:
                            await self.stop_ttfb_metrics()
                            first_chunk = False
                            logger.info(
                                f"[TIMING][TTS] first_chunk_received={time.time() - pipeline_start:.3f}s "
                                f"text_len={len(text)}"
                            )
                            if self._audio_logger is not None:
                                tts_start_time = self._audio_logger.get_time_from_start_of_session()
                        pipeline_chunks_sent += 1

                        audio_bytes = self._convert_to_bytes(audio_chunk)
                        all_audio_bytes += audio_bytes
                        chunk_size = self.chunk_size
                        for i in range(0, len(audio_bytes), chunk_size):
                            audio_chunk_bytes = audio_bytes[i : i + chunk_size]
                            if not audio_chunk_bytes:
                                break
                            frame = TTSAudioRawFrame(
                                audio=audio_chunk_bytes, sample_rate=self.sample_rate, num_channels=1
                            )
                            yield frame
                else:
                    # Legacy: single-result response (non-streaming models like FastPitch)
                    audio_result = data
                    if audio_result is None:
                        logger.error(f"{self} TTS model returned None for text: [{text}]")
                        yield ErrorFrame(error="TTS generation failed - no audio returned")
                        return

                    await self.start_tts_usage_metrics(text)
                    all_audio_bytes = b""
                    if self._audio_logger is not None and self._audio_logger.first_audio_timestamp is None:
                        self._audio_logger.first_audio_timestamp = datetime.now()

                    await self.stop_ttfb_metrics()
                    if self._audio_logger is not None:
                        tts_start_time = self._audio_logger.get_time_from_start_of_session()
                    audio_bytes = self._convert_to_bytes(audio_result)
                    all_audio_bytes = audio_bytes
                    chunk_size = self.chunk_size
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i : i + chunk_size]
                        if not chunk:
                            break
                        frame = TTSAudioRawFrame(audio=chunk, sample_rate=self.sample_rate, num_channels=1)
                        yield frame

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
                            tts_generation_time=tts_start_time,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log agent audio: {e}")

                logger.info(
                    f"[TIMING][TTS] DONE: pipeline_total={time.time() - pipeline_start:.3f}s "
                    f"chunks_sent={pipeline_chunks_sent} audio_bytes={len(all_audio_bytes)}"
                )
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
            logger.debug(f"[ KOKORO ] Generating audio for text: {text}")
            gen_start = time.time()
            # Generate audio using Kokoro pipeline
            generator = self._model(text, voice=self._voice, speed=self._speed)
            pipeline_init_time = time.time() - gen_start
            logger.debug(f"[TIMING][KOKORO] pipeline_init={pipeline_init_time*1000:.1f}ms")

            first_chunk_time = None
            chunk_count = 0
            total_audio_samples = 0

            # The generator yields tuples of (gs, ps, audio)
            for i, (gs, ps, audio) in enumerate(generator):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    logger.info(
                        f"[TIMING][KOKORO] TTFC={first_chunk_time - gen_start:.3f}s "
                        f"text_len={len(text)}"
                    )
                logger.debug(
                    f"Kokoro generated audio chunk {i}: gs={gs}, ps={ps},"
                    f"audio_shape={audio.shape if hasattr(audio, 'shape') else len(audio)}"
                )
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                total_audio_samples += len(audio)
                chunk_count += 1
                yield audio

            total_time = time.time() - gen_start
            total_audio_s = total_audio_samples / self.sample_rate
            rtf = total_time / total_audio_s if total_audio_s > 0 else float('inf')
            logger.info(
                f"[TIMING][KOKORO] DONE: total_gen={total_time:.3f}s "
                f"audio_dur={total_audio_s:.3f}s RTF={rtf:.3f}x chunks={chunk_count}"
            )

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

    Reference: easymagpie_flask_ws_demo.py (branch magpietts_decoderonly_2601_trtllm)
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
        use_cfg: bool = True,
        cfg_scale: float = 2.5,
        max_steps: int = 300,
        device: str = "cuda",
        trt_backbone_engine_dir: Optional[str] = None,
        trt_lt_engine_path: Optional[str] = None,
        trt_lt_run_device: Optional[str] = None,
        compile_backbone: bool = False,
        **kwargs,
    ):
        self._model_path = model_path
        self._codec_model_path = codec_model_path
        self._phoneme_tokenizer_path = phoneme_tokenizer_path
        self._context_text = context_text
        self._context_audio_path = context_audio_path
        self._decode_every_n_frames = decode_every_n_frames
        self._temperature = temperature
        self._topk = topk
        self._use_cfg = use_cfg
        self._cfg_scale = cfg_scale
        self._max_steps = max_steps
        self._trt_backbone_engine_dir = trt_backbone_engine_dir
        self._trt_lt_engine_path = trt_lt_engine_path
        self._compile_backbone = compile_backbone
        # Run the LT engine on a separate device to avoid Myelin context conflict
        # with TRT-LLM backbone (which loads libnvinfer_plugin_tensorrt_llm.so via
        # RTLD_GLOBAL, corrupting Myelin kernels on the same device).
        # When TRT backbone is NOT used (HF/compile backbone), there is no Myelin
        # conflict so LT TRT can run on the same device as the model.
        if trt_lt_run_device is not None:
            self._trt_lt_run_device = trt_lt_run_device
        elif (trt_backbone_engine_dir is not None
              and trt_lt_engine_path is not None
              and device.strip() not in ("cuda", "cuda:0")):
            # TRT-LLM backbone on cuda:1 → LT TRT must move to cuda:0 to avoid Myelin conflict
            self._trt_lt_run_device = "cuda:0"
        else:
            # HF/compile backbone (no Myelin conflict) or cuda:0 → LT TRT on same device
            self._trt_lt_run_device = device
        self._codec_sample_rate = 22050  # EasyMagpie codec always outputs at 22050 Hz (bandwidth extension)
        self._decode_codec_helper = None
        self._base_context_tensors = None
        self._base_streaming_state = None
        # The pipecat WebSocketTransport frontend player runs at PLAYER_SAMPLE_RATE=24000 Hz.
        # Audio is resampled from the codec's native 22050 Hz to 24000 Hz before sending.
        output_sample_rate = kwargs.pop("sample_rate", 24000)
        super().__init__(model=model_path, device=device, sample_rate=output_sample_rate, **kwargs)
        self.setup_tool_calling()

    def _setup_model(self):
        # Set LT backend and engine path env vars before the model is loaded.
        # trt_direct: pre-built TRT fp16 engine (fastest, requires exported lt.engine).
        # compile: torch.compile/inductor fallback (no extra deps, ~1.12x speedup).
        if self._trt_lt_engine_path is not None and "EASYMAGPIE_LT_BACKEND" not in os.environ:
            os.environ["EASYMAGPIE_LT_BACKEND"] = "trt_direct"
            os.environ["EASYMAGPIE_LT_ENGINE_PATH"] = self._trt_lt_engine_path
            logger.info(f"TRT direct LT mode: engine={self._trt_lt_engine_path}")
        elif "EASYMAGPIE_LT_BACKEND" not in os.environ:
            # Default to "compile" (torch.compile/inductor) for ~1.12x LT speedup with no extra deps.
            os.environ["EASYMAGPIE_LT_BACKEND"] = "compile"

        # Enable LT timing without CUDA syncs (wall-clock approximation, no overhead).
        # SYNC_CUDA=0: skip the ~288 torch.cuda.synchronize() calls per step so timing
        # doesn't dominate wall-clock; gives approximate backbone/LT split for profiling.
        if "EASYMAGPIE_STREAMING_TIMING" not in os.environ:
            os.environ["EASYMAGPIE_STREAMING_TIMING"] = "1"
        if "EASYMAGPIE_STREAMING_TIMING_SYNC_CUDA" not in os.environ:
            os.environ["EASYMAGPIE_STREAMING_TIMING_SYNC_CUDA"] = "0"

        from nemo.collections.tts.models import AudioCodecModel
        from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
        from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
        from nemo.collections.tts.modules.magpietts_modules import CodecHelper
        from omegaconf import open_dict

        # Pre-load the LT TRT engine BEFORE the backbone only when using the TRT-LLM backbone.
        # Reason: TRT backbone init imports trt_backbone_runner which calls
        # _register_trt_llm_plugins(), loading libnvinfer_plugin_tensorrt_llm.so with
        # RTLD_GLOBAL. This corrupts the Myelin kernel context for the vanilla TRT LT engine
        # if the LT engine is loaded AFTER. By loading LT first (clean TRT state), it works.
        #
        # For HF backbone (use_trt_backbone=False): no Myelin concern, so we defer LT TRT
        # loading until AFTER torch.compile to avoid TRT's cross-device CUDA initialization
        # from corrupting the cuda:1 PyTorch context before compile runs.
        use_trt_backbone = self._trt_backbone_engine_dir is not None
        _prefetched_lt_runner = None
        if self._trt_lt_engine_path is not None and use_trt_backbone:
            from nemo.collections.tts.models.trt_lt_runner import TRTLocalTransformerRunner
            lt_run_device = self._trt_lt_run_device
            logger.info(f"Pre-loading LT TRT engine (before TRT backbone) on {lt_run_device}: {self._trt_lt_engine_path}")
            _prefetched_lt_runner = TRTLocalTransformerRunner(
                engine_path=self._trt_lt_engine_path,
                device=torch.device(self._device),
                run_device=torch.device(lt_run_device),
            )

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
            strict=not use_trt_backbone,  # strict=False: checkpoint has decoder weights TRT doesn't need
        )
        model.use_kv_cache_for_inference = True
        model.eval().to(self._device).float()

        # Inject pre-loaded LT TRT runner so _run_local_transformer uses it directly
        # without trying to load it lazily (which would happen after TRT-LLM plugins
        # are registered and would break the Myelin kernel context).
        if _prefetched_lt_runner is not None and hasattr(model, '_lt_helper'):
            model._lt_helper._lt_trt_direct_runner = _prefetched_lt_runner
            model._lt_helper._lt_trt_logged = True
            logger.info("LT TRT runner injected into model._lt_helper")

        # Apply torch.compile to the HF backbone decoder when requested.
        # Note: compile_backbone=True only makes sense when trt_backbone_engine_dir is None
        # (i.e. using HF backbone). With TRT backbone the decoder is not a PyTorch module.
        # mode="default" is used to avoid CUDA graph conflicts with the dynamic KV cache.
        # mode="reduce-overhead" (CUDA graphs) crashes because the KV cache grows between calls.
        if self._compile_backbone and not use_trt_backbone:
            if hasattr(model, 'decoder') and model.decoder is not None:
                logger.info("Applying torch.compile to HF backbone decoder (mode='default')...")
                model.decoder = torch.compile(model.decoder, mode="default", dynamic=True)
                logger.info("Backbone compiled")

        # For HF backbone: load LT TRT AFTER torch.compile to prevent TRT's cross-device
        # CUDA initialization from corrupting the PyTorch cuda:1 context before compile.
        if self._trt_lt_engine_path is not None and not use_trt_backbone and _prefetched_lt_runner is None:
            from nemo.collections.tts.models.trt_lt_runner import TRTLocalTransformerRunner
            lt_run_device = self._trt_lt_run_device
            logger.info(f"Loading LT TRT engine (after torch.compile) on {lt_run_device}: {self._trt_lt_engine_path}")
            _prefetched_lt_runner = TRTLocalTransformerRunner(
                engine_path=self._trt_lt_engine_path,
                device=torch.device(self._device),
                run_device=torch.device(lt_run_device),
            )
            # Inject the deferred runner now (the earlier injection block ran when it was None)
            if hasattr(model, '_lt_helper'):
                model._lt_helper._lt_trt_direct_runner = _prefetched_lt_runner
                model._lt_helper._lt_trt_logged = True
                logger.info("LT TRT runner injected into model._lt_helper (deferred)")

        # Build a separate codec instance dedicated to decoding
        logger.info("Building decode codec helper...")
        codec_model = AudioCodecModel.restore_from(
            self._codec_model_path, strict=False, map_location=torch.device("cpu")
        )
        if hasattr(codec_model, "discriminator"):
            del codec_model.discriminator
        codec_model.freeze()
        # Run codec decode on the same device as the TTS model (cuda:1) to avoid concurrent
        # CUDA execution with LT TRT on cuda:0. TRT backbone + codec both on cuda:1 is safe
        # since they run on separate streams and TRT-LLM manages its own context.
        _codec_device = self._device
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

        # Warm-up: prime CUDA kernels and KV-cache path
        self._run_warmup(model)

        # Cache the streaming_init state so _generate_audio never calls streaming_init again —
        # each call clones this instead, avoiding the full context-encoding forward pass.
        # Must be built under inference_mode + autocast so the KV-cache tensors have bfloat16
        # dtype, matching the autocast context used during _generate_audio. A dtype mismatch
        # would cause torch.compile to retrace (slow) or deadlock on the background thread.
        _device_type = "cuda" if "cuda" in self._device else "cpu"
        with torch.inference_mode():
            self._base_streaming_state = self._call_streaming_init(model, device_type=_device_type)
        logger.info("Cached base streaming_init state")

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
        called inside torch.autocast so the returned state has bfloat16 KV-cache
        tensors that match the inference autocast context.
        """
        ctx = self._base_context_tensors
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
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
        """Run streaming_init + N streaming_step calls to compile CUDA kernels and warm KV-cache."""
        logger.info("Warming up EasyMagpieTTS model...")
        device = next(model.parameters()).device
        main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
        token_ids = model.tokenizer.encode("This is a warmup.", tokenizer_name=main_tokenizer_name)
        token_ids.append(model.eos_id)
        pending = deque(token_ids)
        steps = 0

        device_type = "cuda" if "cuda" in str(device) else "cpu"
        # _call_streaming_init adds autocast internally, so we only need inference_mode here
        with torch.inference_mode():
            state = self._call_streaming_init(model, device_type=device_type)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                while not bool(state.finished.all()) and steps < 80:
                    tok = pending.popleft() if pending else None
                    text_tokens = torch.tensor([tok], dtype=torch.long, device=device) if tok is not None else None
                    state, _, _ = model.streaming_step(state=state, text_tokens=text_tokens, use_inference_mode=True)
                    steps += 1

        torch.cuda.empty_cache()
        logger.info("EasyMagpieTTS warm-up complete")

    def _decode_new_chunk(self, accumulated_codes, last_emitted_sample_idx):
        """Decode all accumulated codes; return only the new samples since last emission."""
        codes_lens = torch.tensor(
            [accumulated_codes.size(-1)], dtype=torch.long, device=accumulated_codes.device
        )
        logger.debug(
            f"[DIAG] _decode_new_chunk: acc_codes.shape={tuple(accumulated_codes.shape)} "
            f"contiguous={accumulated_codes.is_contiguous()} codes_lens={codes_lens.tolist()} "
            f"last_emitted={last_emitted_sample_idx}"
        )
        pred_codes, pred_codes_lens = self._model._prepare_codes_for_decode(accumulated_codes, codes_lens)
        logger.debug(
            f"[DIAG] post-prepare: pred_codes.shape={tuple(pred_codes.shape)} "
            f"pred_codes_lens={pred_codes_lens.tolist()}"
        )
        codec_start = time.time()
        # Move codes to the codec device (may differ from TTS model device).
        codec_dev = getattr(self, "_codec_device", None)
        if codec_dev is not None:
            pred_codes = pred_codes.to(codec_dev)
            pred_codes_lens = pred_codes_lens.to(codec_dev)
        audio, audio_len, _ = self._decode_codec_helper.codes_to_audio(pred_codes, pred_codes_lens)
        codec_ms = (time.time() - codec_start) * 1000
        num_frames = accumulated_codes.size(-1)
        logger.info(
            f"[TIMING][EM][CODEC] codes_to_audio: frames={num_frames} "
            f"codec_decode={codec_ms:.1f}ms"
        )
        full_wav = audio[0, : audio_len[0]].detach().float().cpu().numpy()
        # Log actual samples-per-frame on first call to detect codec output rate.
        if last_emitted_sample_idx == 0 and num_frames > 0:
            actual_spf = full_wav.shape[0] / num_frames
            logger.info(f"[DIAG][CODEC] actual samples_per_frame={actual_spf:.1f} "
                        f"(full_wav={full_wav.shape[0]} / frames={num_frames}) "
                        f"min={full_wav.min():.3f} max={full_wav.max():.3f} "
                        f"codec_sr={self._codec_sample_rate} out_sr={self.sample_rate}")
            # Dump raw codec output for inspection
            import scipy.io.wavfile as wavio
            wavio.write("/tmp/codec_raw_output.wav", self._codec_sample_rate,
                        (full_wav * 32767).astype(np.int16))
        if full_wav.shape[0] <= last_emitted_sample_idx:
            return np.zeros((0,), dtype=np.float32), last_emitted_sample_idx
        new_samples = full_wav[last_emitted_sample_idx:]
        # Resample from codec's native rate to the output sample rate if they differ.
        if self._codec_sample_rate != self.sample_rate:
            import soxr
            new_samples = soxr.resample(new_samples, self._codec_sample_rate, self.sample_rate)
        # Cursor is always tracked in codec native samples so the slice math stays correct.
        return new_samples, full_wav.shape[0]

    def _decode_new_chunk_no_grad(self, accumulated_codes, last_emitted_sample_idx):
        """Wrapper that calls _decode_new_chunk inside torch.no_grad so the
        codec decode can safely run in a ThreadPoolExecutor thread (which
        doesn't inherit the torch.inference_mode context from the caller)."""
        with torch.no_grad():
            return self._decode_new_chunk(accumulated_codes, last_emitted_sample_idx)

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        """Decode and yield audio every decode_every_n_frames codec frames.

        Accumulates audio codes from streaming_step and decodes using cumulative
        codes + a cursor (same pattern as _decode_new_audio_chunk in
        easymagpie_flask_ws_demo.py).

        Decode is pipelined with generation: while the LM runs the NEXT batch of
        streaming_steps, the codec decode for the CURRENT batch runs in a separate
        thread (different CUDA stream). The resulting chunk is yielded as soon as
        both the decode AND the next batch of steps are both done, hiding up to
        ~0.28 s of decode latency per chunk.

        NOTE: accumulated_codes is never reset — it is kept cumulative so that
        _decode_new_chunk always works against the full code history.
        """
        logger.debug(f"Generating audio for text: {text}")
        model = self._model
        device = next(model.parameters()).device

        main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
        token_ids = model.tokenizer.encode(text, tokenizer_name=main_tokenizer_name)
        token_ids.append(model.eos_id)
        pending = deque(token_ids)

        logger.debug("[EM] Cloning base streaming state...")
        state = self._clone_state(self._base_streaming_state)
        logger.debug("[EM] Base streaming state cloned, starting generation loop")

        accumulated_codes = None
        last_emitted_sample_idx = 0
        frames_since_last_decode = 0
        steps = 0
        num_decoding_steps = 0

        # Sliding window: cap accumulated_codes to prevent codec integer overflow.
        # The spectral codec decoder fails with storage/padding overflow when the
        # cumulative frame count grows too large (~90-102 frames in practice).
        # We keep at most _max_codec_frames frames and adjust the audio cursor.
        _max_codec_frames = 64
        _spf = getattr(model, 'codec_model_samples_per_frame', 640)

        # Pipelined decode state: (Future, emit_idx_at_submit_time)
        pending_decode_future = None
        pending_decode_emit_idx = 0

        # Timing state
        gen_start = time.time()
        first_chunk_time = None
        chunk_count = 0
        total_audio_samples = 0
        step_times = []
        lt_times_ms = []    # LT wall-clock per step (from last_ar_timing_ms, no-sync approx)
        # pipeline_wait_ms: time the LM loop spent blocked waiting for a decode future
        # (non-zero only when codec is slower than decode_every_n_frames steps)
        pipeline_wait_ms_list = []

        # autocast runs LM matmuls in bfloat16 for ~1.5x speedup without changing
        # stored weight/state dtypes (avoids dtype mismatch in KV-cache).
        device_type = "cuda" if "cuda" in self._device else "cpu"
        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            while not bool(state.finished.all()) and steps < self._max_steps:
                if pending:
                    tok = pending.popleft()
                    text_tokens = torch.tensor([tok], dtype=torch.long, device=device)
                else:
                    text_tokens = None

                step_start = time.time()
                if steps == 0:
                    logger.debug("[EM] Starting first streaming_step...")
                state, audio_codes, _ = model.streaming_step(
                    state=state, text_tokens=text_tokens, use_inference_mode=True
                )
                step_ms = (time.time() - step_start) * 1000
                step_times.append(step_ms / 1000)
                # Collect LT timing (populated when EASYMAGPIE_STREAMING_TIMING=1)
                _lt_helper = getattr(model, '_lt_helper', None)
                if _lt_helper is not None:
                    lt_ms = _lt_helper.last_ar_timing_ms.get('lt_call_total', 0.0)
                    if lt_ms > 0:
                        lt_times_ms.append(lt_ms)
                if steps == 0:
                    logger.debug(f"[EM] First streaming_step done in {step_ms:.1f}ms")
                steps += 1
                num_decoding_steps += 1

                if audio_codes is not None:
                    audio_codes = audio_codes.to(device=device, dtype=torch.long)
                    accumulated_codes = (
                        audio_codes if accumulated_codes is None
                        else torch.cat([accumulated_codes, audio_codes], dim=-1)
                    )
                    frames_since_last_decode += audio_codes.size(-1)

                if accumulated_codes is not None and num_decoding_steps >= self._decode_every_n_frames:
                    decode_start = time.time()
                    frames_since_last_decode = 0
                    num_decoding_steps = 0

                    if pending_decode_future is not None:
                        # Previous decode ran while we were doing this batch's steps.
                        # Retrieve its result (likely already done) and yield it.
                        chunk, last_emitted_sample_idx = pending_decode_future.result()
                        wait_ms = (time.time() - decode_start) * 1000
                        pipeline_wait_ms_list.append(wait_ms)
                        logger.debug(
                            f"[ EMMagpieTTS ] step={steps} yield pipelined chunk "
                            f"size={chunk.size} pipeline_wait={wait_ms:.1f}ms"
                        )
                        if chunk.size > 0:
                            if first_chunk_time is None:
                                first_chunk_time = time.time()
                                logger.info(
                                    f"[TIMING][EM] TTFC={first_chunk_time - gen_start:.3f}s "
                                    f"after {steps} steps text_len={len(text)}"
                                )
                            total_audio_samples += chunk.size
                            chunk_count += 1
                            yield chunk

                    # Sliding-window trim: keep at most _max_codec_frames to prevent
                    # integer overflow in the codec decoder for long sentences.
                    # Only trim after at least one decode result has been retrieved
                    # (last_emitted_sample_idx > 0), so the cursor is valid.
                    if last_emitted_sample_idx > 0 and accumulated_codes is not None and accumulated_codes.size(-1) > _max_codec_frames:
                        _trim = accumulated_codes.size(-1) - _max_codec_frames
                        last_emitted_sample_idx = max(0, last_emitted_sample_idx - _trim * _spf)
                        accumulated_codes = accumulated_codes[..., _trim:]

                    # Submit the decode for the current accumulated_codes.
                    # accumulated_codes is a cumulative tensor; torch.cat above creates
                    # a new tensor each time, so passing the reference is safe here.
                    pending_decode_future = self._decode_executor.submit(
                        self._decode_new_chunk_no_grad,
                        accumulated_codes,
                        last_emitted_sample_idx,
                    )
                    pending_decode_emit_idx = last_emitted_sample_idx

        # Generation done — flush any in-flight decode.
        if pending_decode_future is not None:
            chunk, last_emitted_sample_idx = pending_decode_future.result()
            logger.debug(f"[ EMMagpieTTS ] flush pipelined chunk size={chunk.size}")
            if chunk.size > 0:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    logger.info(
                        f"[TIMING][EM] TTFC={first_chunk_time - gen_start:.3f}s "
                        f"(flush) after {steps} steps text_len={len(text)}"
                    )
                total_audio_samples += chunk.size
                chunk_count += 1
                yield chunk
            # Trim after flush so the final decode stays within safe bounds.
            if last_emitted_sample_idx > 0 and accumulated_codes is not None and accumulated_codes.size(-1) > _max_codec_frames:
                _trim = accumulated_codes.size(-1) - _max_codec_frames
                last_emitted_sample_idx = max(0, last_emitted_sample_idx - _trim * _spf)
                accumulated_codes = accumulated_codes[..., _trim:]

        # Final decode for any codes accumulated after the last batch boundary.
        if accumulated_codes is not None:
            chunk, _ = self._decode_new_chunk_no_grad(accumulated_codes, last_emitted_sample_idx)
            if chunk.size > 0:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    logger.info(
                        f"[TIMING][EM] TTFC={first_chunk_time - gen_start:.3f}s "
                        f"(final) after {steps} steps text_len={len(text)}"
                    )
                total_audio_samples += chunk.size
                chunk_count += 1
                yield chunk

        total_time = time.time() - gen_start
        total_audio_s = total_audio_samples / self.sample_rate
        rtf = total_time / total_audio_s if total_audio_s > 0 else float('inf')
        avg_step_ms = (sum(step_times) / len(step_times) * 1000) if step_times else 0
        avg_pipeline_wait_ms = (sum(pipeline_wait_ms_list) / len(pipeline_wait_ms_list)) if pipeline_wait_ms_list else 0
        avg_lt_ms = (sum(lt_times_ms) / len(lt_times_ms)) if lt_times_ms else 0.0
        avg_backbone_ms = avg_step_ms - avg_lt_ms if avg_lt_ms > 0 else 0.0
        lt_pct = (avg_lt_ms / avg_step_ms * 100) if avg_step_ms > 0 and avg_lt_ms > 0 else 0.0
        logger.info(
            f"[TIMING][EM] DONE: total_gen={total_time:.3f}s audio_dur={total_audio_s:.3f}s "
            f"RTF={rtf:.3f}x steps={steps} avg_step={avg_step_ms:.1f}ms "
            f"[backbone≈{avg_backbone_ms:.1f}ms LT≈{avg_lt_ms:.1f}ms LT%={lt_pct:.0f}%] "
            f"chunks={chunk_count} avg_pipeline_wait={avg_pipeline_wait_ms:.1f}ms"
        )

    def setup_tool_calling(self):
        pass


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
    elif model == "easy_magpie":
        return EasyMagpieTTSService(
            model_path=config.get("main_model_id"),
            codec_model_path=config.get("codec_model_path"),
            phoneme_tokenizer_path=config.get("phoneme_tokenizer_path"),
            context_text=config.get("context_text", "[NO TEXT CONTEXT]"),
            context_audio_path=config.get("context_audio_path", None),
            decode_every_n_frames=config.get("decode_every_n_frames", 6),
            temperature=config.get("temperature", 0.7),
            topk=config.get("topk", 80),
            use_cfg=config.get("use_cfg", True),
            cfg_scale=config.get("cfg_scale", 2.5),
            max_steps=config.get("max_steps", 300),
            device=device,
            trt_backbone_engine_dir=config.get("trt_backbone_engine_dir", None),
            trt_lt_engine_path=config.get("trt_lt_engine_path", None),
            trt_lt_run_device=config.get("trt_lt_run_device", None),
            compile_backbone=config.get("compile_backbone", False),
            text_aggregator=text_aggregator,
            think_tokens=config.get("think_tokens", None),
            audio_logger=audio_logger,
            ignore_strings=config.get("ignore_strings", None),
        )
    else:
        raise ValueError(f"Invalid model: {model}, only 'fastpitch-hifigan', 'magpie', 'kokoro' and 'easy_magpie' are supported")
