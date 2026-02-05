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
import time
from datetime import datetime
from typing import AsyncGenerator, List, Optional

from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pydantic import BaseModel

from nemo.agents.voice_agent.pipecat.services.nemo.audio_logger import AudioLogger
from nemo.agents.voice_agent.pipecat.services.nemo.streaming_asr import NemoStreamingASRService

ASR_EOU_MODELS = ["nvidia/parakeet_realtime_eou_120m-v1"]

try:
    # disable nemo logging
    from nemo.utils import logging

    level = logging.getEffectiveLevel()
    logging.setLevel(logging.CRITICAL)


except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use NVIDIA NeMo STT, you need to `pip install "nemo_toolkit[all]"`.')
    raise Exception(f"Missing module: {e}")


class NeMoSTTInputParams(BaseModel):
    """Input parameters for NeMo STT service."""

    language: Optional[Language] = Language.EN_US
    att_context_size: Optional[List] = [70, 1]
    frame_len_in_secs: Optional[float] = 0.08  # 80ms for FastConformer model
    config_path: Optional[str] = None  # path to the Niva ASR config file
    raw_audio_frame_len_in_secs: Optional[float] = 0.016  # 16ms for websocket transport
    buffer_size: int = 5  # number of raw audio frames to buffer, 1 frame is 16ms


class NemoSTTService(STTService):
    """NeMo Speech-to-Text service for Pipecat integration."""

    def __init__(
        self,
        *,
        model: Optional[str] = "nnvidia/parakeet_realtime_eou_120m-v1",
        device: Optional[str] = "cuda:0",
        sample_rate: Optional[int] = 16000,
        params: Optional[NeMoSTTInputParams] = None,
        has_turn_taking: Optional[bool] = None,  # if None, it will be set by the model name
        backend: Optional[str] = "legacy",
        decoder_type: Optional[str] = "rnnt",
        audio_logger: Optional[AudioLogger] = None,
        ignore_eou_eob: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._queue = asyncio.Queue()
        self._sample_rate = sample_rate

        self._params = params
        self._model_name = model
        self._ignore_eou_eob = ignore_eou_eob
        self._input_sample_rate = None
        if has_turn_taking is None:
            has_turn_taking = True if model in ASR_EOU_MODELS else False
            logger.info(f"Setting has_turn_taking to `{has_turn_taking}` based on model name: `{model}`")
        self._has_turn_taking = has_turn_taking and not self._ignore_eou_eob
        self._backend = backend
        self._decoder_type = decoder_type
        self._audio_logger = audio_logger
        self._is_vad_active = False
        if not params:
            raise ValueError("params is required")

        self._device = device

        self._load_model()

        self._bytes_per_buffer = int(
            self._params.buffer_size * self._params.raw_audio_frame_len_in_secs * sample_rate * 2
        )
        self._audio_buffer = bytearray()
        self.user_is_speaking = False
        self._has_logged_audio_chunk = False
        self._audio_timestamps = []
        logger.info(f"Initialized NeMo STT service with model `{model}` and params `{self._params}`")

    def _load_model(self):
        if self._backend == "legacy":
            self._model = NemoStreamingASRService(
                self._model_name,
                self._params.att_context_size,
                device=self._device,
                decoder_type=self._decoder_type,
                frame_len_in_secs=self._params.frame_len_in_secs,
                ignore_eou_eob=self._ignore_eou_eob,
            )
        else:
            raise ValueError(f"Invalid ASR backend: {self._backend}")

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    def _reset_stt_state(self):
        """Reset the state of the STT service."""
        if isinstance(self._model, NemoStreamingASRService):
            logger.debug("Resetting state of the model")
            self._model.reset_state()
        self._audio_buffer = []
        self._audio_timestamps = []
        self.user_is_speaking = False
        self._has_logged_audio_chunk = False
        self._is_vad_active = False

    async def start(self, frame: StartFrame):
        """Handle service start.

        Args:
            frame: StartFrame containing initial configuration
        """
        await super().start(frame)

        # Initialize the model if not already done
        if not hasattr(self, "_model"):
            self._load_model()

        # Reset the state of the STT service
        self._reset_stt_state()

    async def stop(self, frame: EndFrame):
        """Handle service stop.

        Args:
            frame: EndFrame that triggered this method
        """
        await super().stop(frame)
        # Clear any internal state if needed
        await self._queue.put(None)  # Signal to stop processing
        # Reset the state of the STT service
        self._reset_stt_state()

    async def cancel(self, frame: CancelFrame):
        """Handle service cancellation.

        Args:
            frame: CancelFrame that triggered this method
        """
        await super().cancel(frame)
        # Clear any internal state
        await self._queue.put(None)  # Signal to stop processing
        self._queue = asyncio.Queue()  # Reset the queue
        # Reset the state of the STT service
        self._reset_stt_state()

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        """Process an audio frame for speech recognition.

        If the service is muted, this method does nothing. Otherwise, it
        processes the audio frame and runs speech-to-text on it, yielding
        transcription results. If the frame has a user_id, it is stored
        for later use in transcription.

        Args:
            frame: The audio frame to process.
            direction: The direction of frame processing.
        """
        if self._muted:
            return

        # UserAudioRawFrame contains a user_id (e.g. Daily, Livekit)
        if hasattr(frame, "user_id"):
            self._user_id = frame.user_id
        # AudioRawFrame does not have a user_id (e.g. SmallWebRTCTransport, websockets)
        else:
            self._user_id = ""

        if not frame.audio:
            # Ignoring in case we don't have audio to transcribe.
            logger.warning(f"Empty audio frame received for STT service: {self.name} {frame.num_frames}")
            return

        await self.process_generator(self.run_stt(frame.audio, frame))

    async def run_stt(self, audio: bytes, audio_frame: Optional[AudioRawFrame] = None) -> AsyncGenerator[Frame, None]:
        """Process audio data and generate transcription frames.

        Args:
            audio: Raw audio bytes to transcribe

        Yields:
            Frame: Transcription frames containing the results
        """
        timestamp_now = datetime.now()
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        if self._audio_logger is not None and self._audio_logger.first_audio_timestamp is None:
            self._audio_logger.first_audio_timestamp = timestamp_now

        try:
            is_final = False
            user_has_finished = False
            transcription = None
            self._audio_buffer.extend(audio)
            if audio_frame and hasattr(audio_frame, 'timestamp'):
                self._audio_timestamps.append(audio_frame.timestamp)
            else:
                self._audio_timestamps.append(timestamp_now)

            if not self._has_logged_audio_chunk:
                # convert bytes to seconds
                import numpy as np

                audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                audio_seconds = len(audio_array) / self._sample_rate
                logger.debug(f"Received audio chunk length: {len(audio)} bytes, seconds: {audio_seconds}")
                self._has_logged_audio_chunk = True

            if len(self._audio_buffer) >= self._bytes_per_buffer:
                audio_chunk_bytes = bytes(self._audio_buffer[: self._bytes_per_buffer])
                self._audio_buffer = self._audio_buffer[self._bytes_per_buffer :]

                last_audio_timestamp = self._audio_timestamps[: self._params.buffer_size][-1]
                self._audio_timestamps = self._audio_timestamps[self._params.buffer_size :]

                # Append to continuous user audio buffer for stereo conversation recording
                if self._audio_logger is not None:
                    self._audio_logger.append_continuous_user_audio(audio_chunk_bytes)

                # Run ASR inference in thread pool to avoid blocking event loop
                start_time = time.time()
                asr_result = await asyncio.to_thread(self._model.transcribe, audio_chunk_bytes)
                end_time = time.time()
                logger.debug(f"ASR inference time: {end_time - start_time} seconds")
                transcription = asr_result.text
                is_final = asr_result.is_final

                delay = datetime.now() - last_audio_timestamp
                logger.debug(
                    f"Transcription: {transcription}, audio timestamp: {last_audio_timestamp}, delay: {delay}"
                )
                if self._audio_logger is not None:
                    if self._is_vad_active:
                        is_first_frame = False
                        self._audio_logger.turn_audio_buffer.append(audio_chunk_bytes)
                        # Accumulate transcriptions for turn-based logging
                        if transcription:
                            self._audio_logger.turn_transcription_buffer.append(transcription)
                            self._audio_logger.stage_turn_audio_and_transcription(
                                timestamp_now=timestamp_now,
                                is_first_frame=is_first_frame,
                                additional_metadata={
                                    "model": self._model_name,
                                    "backend": self._backend,
                                },
                            )
                eou_latency = asr_result.eou_latency
                eob_latency = asr_result.eob_latency
                eou_prob = asr_result.eou_prob
                eob_prob = asr_result.eob_prob
                if eou_latency is not None:
                    logger.debug(
                        f"EOU latency: {eou_latency: .4f} seconds. EOU probability: {eou_prob: .2f}."
                        f"Processing time: {asr_result.processing_time: .4f} seconds."
                    )
                    user_has_finished = True
                if eob_latency is not None:
                    logger.debug(
                        f"EOB latency: {eob_latency: .4f} seconds. EOB probability: {eob_prob: .2f}."
                        f"Processing time: {asr_result.processing_time: .4f} seconds."
                    )
                    user_has_finished = True
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()

            if transcription:
                logger.debug(f"Transcription (is_final={is_final}): `{transcription}`")
                self.user_is_speaking = True if not user_has_finished else False

                # Get the language from params or default to EN_US
                language = self._params.language if self._params else Language.EN_US

                # Create and yield the transcription frame
                if self._has_turn_taking:
                    # if turn taking is enabled, we push interim transcription frames
                    # and let the turn taking service handle the final transcription
                    frame_type = InterimTranscriptionFrame
                else:
                    # otherwise, we use the is_final flag to determine the frame type
                    frame_type = TranscriptionFrame if is_final else InterimTranscriptionFrame

                # Yield the frame instead of pushing it to avoid blocking
                yield frame_type(
                    transcription,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result={"text": transcription},
                )

        except Exception as e:
            logger.error(f"Error in NeMo STT processing: {e}")
            yield ErrorFrame(
                str(e),
                time_now_iso8601(),
            )

    async def set_language(self, language: Language):
        """Update the service's recognition language.

        Args:
            language: New language for recognition
        """
        if self._params:
            self._params.language = language
        else:
            self._params = NeMoSTTInputParams(language=language)

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
        """Process incoming frames and handle VAD events."""
        if isinstance(frame, VADUserStoppedSpeakingFrame) and isinstance(self._model, NemoStreamingASRService):
            # manualy reset the state of the model when end of utterance is detected by VAD
            logger.debug("Resetting state of the model due to VADUserStoppedSpeakingFrame")
            if self.user_is_speaking:
                logger.debug(
                    "[EOU missing] STT failed to detect end of utterance before VAD detected user stopped speaking"
                )
            self._model.reset_state()
            self._is_vad_active = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._is_vad_active = True

        await super().process_frame(frame, direction)
