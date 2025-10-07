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
import inspect
from collections.abc import AsyncGenerator
from typing import Iterator, List, Optional

import numpy as np
import torch
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from nemo.collections.tts.models import FastPitchModel, HifiGanModel


class BaseNemoTTSService(TTSService):
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
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._model_name = model
        self._device = device
        self._model = self._setup_model()
        self._think_tokens = think_tokens
        if think_tokens is not None:
            assert (
                isinstance(think_tokens, list) and len(think_tokens) == 2
            ), f"think_tokens must be a list of two strings: {think_tokens}"

        # Background processing infrastructure - no response handler needed
        self._tts_queue = asyncio.Queue()
        self._processing_task = None
        self._processing_running = False

        # Track pending requests with their response queues
        self._pending_requests = {}
        self._have_seen_think_tokens = False

    def _setup_model(self):
        raise NotImplementedError("Subclass must implement _setup_model")

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        raise NotImplementedError("Subclass must implement _generate_audio")

    def can_generate_metrics(self) -> bool:
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

                    # Process TTS generation
                    try:
                        audio_result = self._generate_audio(text)

                        # Send result directly to the waiting request
                        asyncio.run_coroutine_threadsafe(
                            response_queue.put(('success', audio_result)), self.get_event_loop()
                        )
                    except Exception as e:
                        logger.error(f"Error in TTS generation: {e}")
                        # Send error directly to the waiting request
                        asyncio.run_coroutine_threadsafe(response_queue.put(('error', e)), self.get_event_loop())

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
        if not self._think_tokens:
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

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using the Nemo TTS model."""
        text = self._handle_think_tokens(text)

        if not text:
            yield None
            return

        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            # Generate unique request ID
            import uuid

            request_id = str(uuid.uuid4())

            # Create response queue for this specific request
            request_queue = asyncio.Queue()
            self._pending_requests[request_id] = request_queue

            try:
                # Queue the TTS request for background processing
                await self._tts_queue.put((text, request_id))

                # Wait for the result directly from our request queue
                result = await request_queue.get()
                status, data = result

                if status == 'error':
                    logger.error(f"{self} TTS generation error: {data}")
                    yield ErrorFrame(error=f"TTS generation error: {str(data)}")
                    return

                audio_result = data
                if audio_result is None:
                    logger.error(f"{self} TTS model returned None for text: [{text}]")
                    yield ErrorFrame(error="TTS generation failed - no audio returned")
                    return

                await self.start_tts_usage_metrics(text)

                # Process the audio result (same as before)
                if (
                    inspect.isgenerator(audio_result)
                    or hasattr(audio_result, '__iter__')
                    and hasattr(audio_result, '__next__')
                ):
                    # Handle generator case
                    first_chunk = True
                    for audio_chunk in audio_result:
                        if first_chunk:
                            await self.stop_ttfb_metrics()
                            first_chunk = False

                        if audio_chunk is None:
                            break

                        audio_bytes = self._convert_to_bytes(audio_chunk)
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
                    # Handle single result case
                    await self.stop_ttfb_metrics()
                    audio_bytes = self._convert_to_bytes(audio_result)

                    chunk_size = self.chunk_size
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i : i + chunk_size]
                        if not chunk:
                            break

                        frame = TTSAudioRawFrame(audio=chunk, sample_rate=self.sample_rate, num_channels=1)
                        yield frame

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

        # Handle numpy arrays
        try:
            import numpy as np

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
        except ImportError:
            # Fallback if numpy is not available
            if hasattr(audio_data, 'tobytes'):
                return audio_data.tobytes()
            else:
                return bytes(audio_data)


class NeMoFastPitchHiFiGANTTSService(BaseNemoTTSService):
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

    def _setup_model(self):
        print("Loading model...")
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
