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
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pydantic import BaseModel

from nemo.agents.voice_agent.pipecat.frames.frames import DiarResultFrame
from nemo.agents.voice_agent.pipecat.services.nemo.legacy_diar import DiarizationConfig, NeMoLegacyDiarService


class NeMoDiarInputParams(BaseModel):
    threshold: Optional[float] = (
        0.4  # threshold value used to determine if a speaker exists or not, setting it to a lower value will increase the sensitivity of the diarization model
    )
    language: Optional[Language] = Language.EN_US
    frame_len_in_secs: Optional[float] = 0.08  # 80ms for FastConformer model
    config_path: Optional[str] = None  # path to the Niva ASR config file
    raw_audio_frame_len_in_secs: Optional[float] = 0.016  # 16ms for websocket transport
    buffer_size: Optional[int] = (
        30  # number of audio frames to buffer, 1 frame is 16ms, streaming Sortformer was trained with 6*0.08=0.48s chunks
    )


class NemoDiarService(STTService):
    def __init__(
        self,
        *,
        model: Optional[str] = "",
        device: Optional[str] = "cuda:0",
        sample_rate: Optional[int] = 16000,
        params: Optional[NeMoDiarInputParams] = None,
        use_vad: bool = True,
        audio_passthrough: bool = True,
        backend: Optional[str] = "legacy",
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__(audio_passthrough=audio_passthrough, **kwargs)

        self._enabled = enabled
        self._queue = asyncio.Queue()
        self._response_queue = asyncio.Queue()  # Add response queue
        self._processing_task = None  # Add processing task
        self._response_task = None  # Add response task
        self._device = device
        self._sample_rate = sample_rate
        self._audio_passthrough = audio_passthrough
        params.buffer_size = params.frame_len_in_secs // params.raw_audio_frame_len_in_secs
        self._params = params
        self._model_name = model
        self._use_vad = use_vad
        self._backend = backend
        if not params:
            raise ValueError("params is required")

        self._load_model()

        self._vad_user_speaking = False
        self._audio_buffer = []
        self._current_speaker_id = None
        self._processing_running = False

        if not self._use_vad:
            self._vad_user_speaking = True

    def _load_model(self):
        if not self._enabled or not self._model_name:
            self._model = None
            self._enabled = False
            return

        if self._backend == "legacy":
            cfg = DiarizationConfig()
            cfg.device = self._device
            self._model = NeMoLegacyDiarService(
                cfg, self._model_name, frame_len_in_secs=self._params.frame_len_in_secs, sample_rate=self.sample_rate
            )
        else:
            raise ValueError(f"Invalid backend: {self._backend}")
        logger.info(f"Diarization service initialized on device: {self._device}")

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Handle service start."""
        await super().start(frame)

        # Initialize the model if not already done
        if not hasattr(self, "_model"):
            self._load_model()

        # Start background processing task
        if not self._processing_task:
            self._processing_task = self.create_task(self._processing_task_handler())

        # Start response handling task
        if not self._response_task:
            self._response_task = self.create_task(self._response_task_handler())

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
        await self._queue.put(None)  # Signal to stop processing
        if self._processing_task:
            await self.cancel_task(self._processing_task)
            self._processing_task = None

        if self._response_task:
            await self.cancel_task(self._response_task)
            self._response_task = None

    def _diarization_processor(self):
        """Background processor that handles diarization calls."""
        try:
            while self._processing_running:
                try:
                    # Get audio from queue - blocking call that will be interrupted by cancellation
                    future = asyncio.run_coroutine_threadsafe(self._queue.get(), self.get_event_loop())
                    audio = future.result()

                    if audio is None:  # Stop signal
                        logger.debug("Received stop signal in background processor")
                        break

                    # Process diarization
                    diar_result = self._model.diarize(audio)

                    # Send result back to async loop
                    asyncio.run_coroutine_threadsafe(self._response_queue.put(diar_result), self.get_event_loop())

                except Exception as e:
                    logger.error(f"Error in background diarization processor: {e}")
                    # Send error back to async loop
                    asyncio.run_coroutine_threadsafe(self._response_queue.put(('error', e)), self.get_event_loop())

        except Exception as e:
            logger.error(f"Background diarization processor fatal error: {e}")
        finally:
            logger.debug("Background diarization processor stopped")

    async def _processing_task_handler(self):
        """Handler for background processing task."""
        try:
            self._processing_running = True
            logger.debug("Starting background processing task")
            await asyncio.to_thread(self._diarization_processor)
        except asyncio.CancelledError:
            logger.debug("Background processing task cancelled")
            self._processing_running = False
            raise
        finally:
            self._processing_running = False

    async def _handle_diarization_result(self, diar_result):
        """Handle diarization result from background processing."""
        try:
            if diar_result is None:
                return
            dominant_speaker_id = self._get_dominant_speaker_id(diar_result)
            # logger.debug(f"Dominant speaker ID: {dominant_speaker_id}")
            if dominant_speaker_id is not None and dominant_speaker_id != self._current_speaker_id:
                self._current_speaker_id = dominant_speaker_id
                logger.debug(f"Pushing DiarResultFrame with speaker {dominant_speaker_id}")
                await self.push_frame(DiarResultFrame(dominant_speaker_id, stream_id="default"))
        except Exception as e:
            logger.error(f"Error handling diarization result: {e}")
            await self.push_frame(
                ErrorFrame(
                    str(e),
                    time_now_iso8601(),
                )
            )

    async def _response_task_handler(self):
        """Handler for processing diarization results."""
        logger.debug("Response task handler started")
        try:
            while True:
                try:
                    result = await self._response_queue.get()

                    if isinstance(result, tuple) and result[0] == 'error':
                        # Handle error from background processing
                        error = result[1]
                        logger.error(f"Error in NeMo Diarization processing: {error}")
                        await self.push_frame(
                            ErrorFrame(
                                str(error),
                                time_now_iso8601(),
                            )
                        )
                    else:
                        # Handle successful diarization result
                        await self._handle_diarization_result(result)

                except Exception as e:
                    logger.error(f"Error in response task handler: {e}")
        except asyncio.CancelledError:
            logger.debug("Response task handler cancelled")
            raise

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data and generate transcription frames.

        Args:
            audio: Raw audio bytes to transcribe

        Yields:
            Frame: Transcription frames containing the results
        """
        if self._vad_user_speaking and self._enabled:
            self._audio_buffer.append(audio)
            if len(self._audio_buffer) >= self._params.buffer_size:
                await self.start_ttfb_metrics()
                await self.start_processing_metrics()
                audio = b"".join(self._audio_buffer)
                self._audio_buffer = []
                # Queue audio for background processing
                await self._queue.put(audio)
        yield None

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Optional[str] = None):
        """Handle a transcription result.

        Args:
            transcript: The transcribed text
            is_final: Whether this is a final transcription
            language: The language of the transcription
        """
        pass  # Base implementation - can be extended for specific handling needs

    async def set_language(self, language: Language):
        """Update the service's recognition language.

        Args:
            language: New language for recognition
        """
        if self._params:
            self._params.language = language
        else:
            self._params = NeMoDiarInputParams(language=language)

        logger.info(f"Switching STT language to: {language}")

    async def set_model(self, model: str):
        """Update the service's model.

        Args:
            model: New model name/path to use
        """
        await super().set_model(model)
        self._model_name = model
        self._load_model()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process audio data and generate transcription frames.

        Args:
            audio: Raw audio bytes to transcribe

        Yields:
            Frame: Transcription frames containing the results
        """
        if not self._enabled:
            # if diarization is disabled, just pass the frame through
            await self.push_frame(frame, direction)
            return

        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_user_speaking = True
            self._audio_buffer = []
            logger.debug("VADUserStartedSpeakingFrame received")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_user_speaking = False
            logger.debug("VADUserStoppedSpeakingFrame received")
            self._current_speaker_id = None
            self._audio_buffer = []

    def reset(self):
        self._current_speaker_id = None
        self._audio_buffer = []
        self._vad_user_speaking = False
        self._model.reset_state()

    def _get_dominant_speaker_id(self, spk_pred: np.ndarray):
        spk_pred = (spk_pred > self._params.threshold).astype(int)
        dominant_speaker_id = None
        if spk_pred.sum() > 0:
            # get the dominant speaker id
            # Filter to only keep frames that have any speaker probability > 0.0
            valid_frame_mask = spk_pred.sum(axis=1) > 0

            # Filter diar_result to only keep valid frames
            filtered_diar_result = spk_pred[valid_frame_mask]  # ndarray of shape [num_valid_frames, num_speakers]

            # Get the primary speaker for each valid frame
            primary_spk = np.argmax(filtered_diar_result, axis=1)  # ndarray of shape [num_valid_frames]
            # logger.debug(f"Primary speaker for valid frames: {primary_spk}")

            # count the number of different speakers in the primary speaker sequence
            num_speakers = len(np.unique(primary_spk))
            # logger.debug(f"Number of different speakers: {num_speakers}")

            # If there are multiple speakers, get the dominant one
            if num_speakers > 1:
                # Count occurrences of each speaker
                speaker_counts = np.bincount(primary_spk)
                dominant_speaker_id = np.argmax(speaker_counts)
            else:
                # Only one speaker, return that speaker ID
                dominant_speaker_id = primary_spk[0]
        return dominant_speaker_id
