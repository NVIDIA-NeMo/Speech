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
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Iterator, List, Optional

import numpy as np
import torch
from loguru import logger
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
from nemo.agents.voice_agent.pipecat.utils.tool_calling.mixins import ToolCallingMixin
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
        audio_logger: Optional[AudioLogger] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._model_name = model
        self._device = device
        self._model = self._setup_model()
        self._think_tokens = think_tokens
        self._audio_logger = audio_logger
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

                # Collect all audio for logging
                all_audio_bytes = b""
                # Capture the start time when TTS begins (not when it ends)
                if self._audio_logger is not None and self._audio_logger.first_audio_timestamp is None:
                    self._audio_logger.first_audio_timestamp = datetime.now()

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
                            # Capture start time on first chunk
                            if self._audio_logger is not None:
                                tts_start_time = self._audio_logger.get_time_from_start_of_session()

                        if audio_chunk is None:
                            break

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
                    # Handle single result case
                    await self.stop_ttfb_metrics()
                    # Capture start time for single result
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


class KokoroTTSService(BaseNemoTTSService, ToolCallingMixin):
    """Text-to-Speech service using Kokoro-82M model.

    Kokoro is an open-weight TTS model with 82 million parameters.
    More info: https://huggingface.co/hexgrad/Kokoro-82M

    Args:
        lang_code: Language code for the model (default: 'a' for American English)
        voice: Voice to use (default: 'af_heart')
        device: Device to run on (default: 'cuda')
        sample_rate: Audio sample rate in Hz (default: 24000 for Kokoro)
        **kwargs: Additional arguments passed to BaseNemoTTSService
    """

    def __init__(
        self,
        lang_code: str = "a",
        voice: str = "af_heart",
        device: str = "cuda",
        sample_rate: int = 24000,
        speed: float = 1.0,
        **kwargs,
    ):
        self._lang_code = lang_code
        self._voice = voice
        self._speed = speed
        assert speed > 0, "Speed must be greater than 0"
        model_name = f"kokoro-{lang_code}-{voice}"
        self._speed_lambda = 1.0
        self._original_speed = speed
        self._original_voice = voice
        self._gender = 'female' if voice[1] == 'f' else 'male'
        self._original_gender = self._gender
        self._original_lang_code = self._lang_code
        super().__init__(model=model_name, device=device, sample_rate=sample_rate, **kwargs)
        self.setup_tool_calling()

    def _setup_model(self, lang_code: Optional[str] = None, voice: Optional[str] = None):
        """Initialize the Kokoro pipeline."""
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "kokoro package is required for KokoroTTSService. " "Install it with: pip install kokoro>=0.9.2"
            )
        if lang_code is None:
            lang_code = self._lang_code
        if voice is None:
            voice = self._voice
        logger.info(f"Loading Kokoro TTS model with lang_code={lang_code}, voice={voice}")
        pipeline = KPipeline(lang_code=lang_code)
        return pipeline

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

        After calling this tool, continue the previous response if it was unfinished and was
        interrupted by calling this tool, otherwise start a new response.

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
            self._speed = speed_lambda * self._original_speed
            result = {
                "success": True,
                "message": f"Speed set to {speed_lambda} of the original speed",
            }
            logger.debug(f"Speed set to {speed_lambda} of the original speed {self._original_speed}")
        await params.result_callback(result)

    async def tool_tts_reset_speed(self, params: FunctionCallParams):
        """
        Reset the speaking speed to the original speed.

        After calling this tool, continue the previous response if it was unfinished and was
        interrupted by calling this tool, otherwise start a new response.
        """
        self._speed = self._original_speed
        result = {"success": True, "message": "Speaking speed is reset to the original one"}
        logger.debug(f"Speaking speed is reset to the original speed {self._original_speed}")
        await params.result_callback(result)

    async def tool_tts_speak_faster(self, params: FunctionCallParams):
        """
        Speak faster by increasing the speaking speed 15% faster each time this function is called.

        After calling this tool, continue the previous response if it was unfinished and was
        interrupted by calling this tool, otherwise start a new response.
        """
        self._speed_lambda = self._speed_lambda + 0.15
        self._speed = self._speed_lambda * self._original_speed
        result = {
            "success": True,
            "message": f"Speaking speed is increased to {self._speed_lambda} of the original speed",
        }
        logger.debug(f"Speed is set to {self._speed_lambda} of the original speed {self._original_speed}")
        await params.result_callback(result)

    async def tool_tts_speak_slower(self, params: FunctionCallParams):
        """
        Speak slower by decreasing the speaking speed 15% slower each time this function is called.

        After calling this tool, continue the previous response if it was unfinished and was
        interrupted by calling this tool, otherwise start a new response.
        """
        self._speed_lambda = self._speed_lambda - 0.15
        if self._speed_lambda < 0.1:
            self._speed = 0.1 * self._original_speed
            result = {
                "success": True,
                "message": "Speaking speed is decreased to the minimum of 0.1 of the original speed",
            }
            logger.debug(f"Speed is set to the minimum of 0.1 of the original speed {self._original_speed}")
        else:
            self._speed = self._speed_lambda * self._original_speed
            result = {
                "success": True,
                "message": f"Speaking speed is decreased to {self._speed_lambda} of the original speed",
            }
            logger.debug(f"Speed is set to {self._speed_lambda} of the original speed {self._original_speed}")
        await params.result_callback(result)

    async def tool_tts_set_voice(self, params: FunctionCallParams, accent: str, gender: str):
        """
        Set the accent and gender of the assistant's voice.
        This tool should be called only when the user specifies the accent and/or gender explicitly.

        After calling this tool, continue the previous response if it was unfinished and was
        interrupted by calling this tool, otherwise start a new response.

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
        await params.result_callback({"success": True, "message": "Done. Language and voice are set to the new ones."})

    async def tool_tts_reset_voice(self, params: FunctionCallParams):
        """
        Reset the accent and voice to the original ones.

        After calling this tool, continue the previous response if it was unfinished and was
        interrupted by calling this tool, otherwise start a new response.
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
        await params.result_callback(
            {"success": True, "message": "Done. Language and voice are reset to the original ones."}
        )

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
