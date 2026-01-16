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

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import yaml
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from nemo.agents.voice_agent.pipecat.frames.frames import DiarResultFrame
from nemo.agents.voice_agent.pipecat.services.nemo.audio_logger import AudioLogger


class NeMoTurnTakingService(FrameProcessor):
    """Service for handling turn-taking in voice conversations with backchannel detection."""

    def __init__(
        self,
        backchannel_phrases: Union[str, List[str]] = None,
        eou_string: str = "<EOU>",
        eob_string: str = "<EOB>",
        language: Language = Language.EN_US,
        use_vad: bool = True,
        use_diar: bool = False,
        max_buffer_size: int = 3,
        bot_stop_delay: float = 0.5,
        audio_logger: Optional[AudioLogger] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eou_string = eou_string
        self.eob_string = eob_string
        self.language = language
        self.use_vad = use_vad
        self.use_diar = use_diar
        self.max_buffer_size = max_buffer_size

        self.backchannel_phrases = self._load_backchannel_phrases(backchannel_phrases)
        self.backchannel_phrases_nopc = set([self.clean_text(phrase) for phrase in self.backchannel_phrases])
        self.bot_stop_delay = bot_stop_delay
        # internal data
        self._current_speaker_id = None
        self._prev_speaker_id = None
        self._bot_stop_time = None
        self._bot_speaking = False
        self._vad_user_speaking = False
        self._have_sent_user_started_speaking = False
        self._user_speaking_buffer = ""
        self._audio_logger = audio_logger
        if not self.use_vad:
            # if vad is not used, we assume the user is always speaking
            self._vad_user_speaking = True

    def _load_backchannel_phrases(self, backchannel_phrases: Optional[Union[str, List[str]]] = None):
        if not backchannel_phrases:
            return []

        if isinstance(backchannel_phrases, str) and Path(backchannel_phrases).is_file():
            logger.info(f"Loading backchannel phrases from file: {backchannel_phrases}")
            if not Path(backchannel_phrases).exists():
                raise FileNotFoundError(f"Backchannel phrases file not found: {backchannel_phrases}")
            with open(backchannel_phrases, "r") as f:
                backchannel_phrases = yaml.safe_load(f)
            if not isinstance(backchannel_phrases, list):
                raise ValueError(f"Backchannel phrases must be a list, got {type(backchannel_phrases)}")
            logger.info(f"Loaded {len(backchannel_phrases)} backchannel phrases from file: {backchannel_phrases}")
        elif isinstance(backchannel_phrases, list):
            logger.info(f"Using backchannel phrases from list: {backchannel_phrases}")
        else:
            raise ValueError(f"Invalid backchannel phrases: {backchannel_phrases}")
        return backchannel_phrases

    def clean_text(self, text: str) -> str:
        """
        Clean the text so that it can be used for backchannel detection.
        """
        if self.language != Language.EN_US:
            raise ValueError(f"Language {self.language} not supported, currently only English is supported.")
        for eou_string in [self.eou_string, self.eob_string]:
            if text.endswith(eou_string):
                text = text[: -len(eou_string)].strip()
        text = text.lower()
        valid_chars = "abcdefghijklmnopqrstuvwxyz'"
        text = ''.join([c for c in text if c in valid_chars or c.isspace() or c == "'"])
        return " ".join(text.split()).strip()

    def is_backchannel(self, text: str) -> bool:
        """
        Check if the text is a backchannel phrase.
        """
        if text.startswith("<speaker_"):
            # if the text starts with a speaker tag, we remove it
            text = text[len("<speaker_0>") :]
        text = self.clean_text(text)
        return text in self.backchannel_phrases_nopc

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle turn-taking logic."""
        await super().process_frame(frame, direction)

        if self._bot_stop_time is not None:
            # check if the bot has stopped speaking for more than the delay
            if time.time() - self._bot_stop_time > self.bot_stop_delay:
                # set the _bot_speaking flag to False to actually consider the bot as stopped speaking
                logger.debug(
                    f"Bot stopped speaking for more than {self.bot_stop_delay} seconds, setting _bot_speaking to False"
                )
                self._bot_stop_time = None
                self._bot_speaking = False

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            await self._handle_transcription(frame, direction)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame, direction)
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("BotStartedSpeakingFrame received")
            self._bot_speaking = True
            # Capture the actual start time when audio starts playing
            # This is more accurate than capturing during TTS generation
            if self._audio_logger:
                self._audio_logger.set_agent_turn_start_time()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("BotStoppedSpeakingFrame received")
            self._bot_stop_time = time.time()
            if self.bot_stop_delay is None or self.bot_stop_delay <= 0:
                # only set the flag if the delay is not set or is 0
                self._bot_speaking = False
                logger.debug("Setting _bot_speaking to False")
        elif isinstance(frame, DiarResultFrame):
            logger.debug("DiarResultFrame received")
            await self._handle_diar_result(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_backchannel_text(self, text: str):
        # ignore the backchannel string while bot is speaking
        # push the backchannel string upstream, not downstream
        await self.push_frame(
            TranscriptionFrame(
                text=f"({text})",
                user_id="",
                timestamp=time_now_iso8601(),
                language=self.language if self.language else Language.EN_US,
                result={"text": f"Backchannel detected: {text}"},
            ),
            direction=FrameDirection.UPSTREAM,
        )

    async def _handle_transcription(
        self, frame: TranscriptionFrame | InterimTranscriptionFrame, direction: FrameDirection
    ):
        text_segment = frame.text
        if self._vad_user_speaking:
            self._user_speaking_buffer += text_segment
            has_eou = self._user_speaking_buffer.endswith(self.eou_string)
            has_eob = self._user_speaking_buffer.endswith(self.eob_string)
            if has_eou:
                # EOU detected, user is done speaking - push completed text and interrupt bot
                logger.debug(f"<EOU> Detected: `{self._user_speaking_buffer}`")
                completed_text = self._user_speaking_buffer[: -len(self.eou_string)].strip()
                if self._bot_speaking and self.is_backchannel(completed_text):
                    logger.debug(f"<EOU> detected for a backchannel phrase while bot is speaking: `{completed_text}`")
                    await self._handle_backchannel_text(completed_text)
                    if self._audio_logger:
                        if self._audio_logger.staged_metadata is None:
                            self._audio_logger.staged_metadata = {"is_backchannel": True, "start_time": datetime.now()}
                        else:
                            self._audio_logger.staged_metadata["is_backchannel"] = True

                else:
                    await self._handle_completed_text(completed_text, direction)
                    await self._handle_user_interruption(UserStoppedSpeakingFrame())
                self._user_speaking_buffer = ""
                self._have_sent_user_started_speaking = False  # user is done speaking, so we reset the flag
            elif has_eob and self._bot_speaking:
                logger.debug(f"<EOB> detected while bot is speaking: `{self._user_speaking_buffer}`")
                await self._handle_backchannel_text(str(self._user_speaking_buffer))
                if self._audio_logger:
                    if self._audio_logger.staged_metadata is None:
                        self._audio_logger.staged_metadata = {"is_backchannel": True, "start_time": datetime.now()}
                    else:
                        self._audio_logger.staged_metadata["is_backchannel"] = True
                self._user_speaking_buffer = ""
                self._have_sent_user_started_speaking = False  # user is done speaking, so we reset the flag
            else:
                # if bot is not speaking, the backchannel string is not considered a backchannel phrase
                # user is still speaking, so we append the text segment to the buffer
                logger.debug(f"User is speaking: `{self._user_speaking_buffer}`")
                if has_eob:
                    logger.debug(
                        f"{self.eob_string} detected but ignored (bot NOT speaking): "
                        f"`{self._user_speaking_buffer}`"
                    )
                    self._user_speaking_buffer = self._user_speaking_buffer[: -len(self.eob_string)].strip()
                completed_words = self._user_speaking_buffer.strip().split()[
                    :-1
                ]  # assume the last word is not completed
                if len(completed_words) >= self.max_buffer_size:
                    completed_text = " ".join(completed_words)
                    await self._handle_completed_text(completed_text, direction, is_final=False)

        else:
            # if vad is not detecting user speaking
            logger.debug(
                f"VAD is not detecting user speaking, but still received text segment from STT: `{text_segment}`"
            )
            is_backchannel = self.is_backchannel(text_segment)
            if text_segment.endswith(self.eob_string):
                is_backchannel = True
                logger.debug(f"Dropping EOB token: `{text_segment}`")
                text_segment = text_segment[: -len(self.eob_string)].strip()
            elif text_segment.endswith(self.eou_string):
                logger.debug(f"Dropping EOU token: `{text_segment}`")
                text_segment = text_segment[: -len(self.eou_string)].strip()

            if not text_segment.strip():
                return
            if is_backchannel and self._bot_speaking:
                logger.debug(f"Backchannel detected while bot is speaking: `{text_segment}`")
                # push the backchannel string upstream, not downstream
                curr_text = str(self._user_speaking_buffer + text_segment)
                self._user_speaking_buffer = ""
                if self._audio_logger:
                    if self._audio_logger.staged_metadata is None:
                        self._audio_logger.staged_metadata = {"is_backchannel": True, "start_time": datetime.now()}
                    else:
                        self._audio_logger.staged_metadata["is_backchannel"] = True
                await self.push_frame(
                    TranscriptionFrame(
                        text=f"({curr_text})",
                        user_id="",
                        timestamp=time_now_iso8601(),
                        language=self.language if self.language else Language.EN_US,
                        result={"text": f"Backchannel detected: {self._user_speaking_buffer+text_segment}"},
                    ),
                    direction=FrameDirection.UPSTREAM,
                )

            else:
                # if the text segment is not empty and have non-space characters, we append it to the buffer
                self._user_speaking_buffer += text_segment
                if self.is_backchannel(self._user_speaking_buffer):
                    logger.debug(f"Backchannel detected: `{self._user_speaking_buffer}`")
                    self._user_speaking_buffer = ""
                    self._have_sent_user_started_speaking = False
                    return
                logger.debug(f"Appending text segment to user speaking buffer: `{self._user_speaking_buffer}`")

    async def _handle_completed_text(self, completed_text: str, direction: FrameDirection, is_final: bool = True):
        if not self._have_sent_user_started_speaking:
            # if we haven't sent the user started speaking frame, we send it now
            # so that the bot can be interrupted and be ready to respond to the new user turn
            await self._handle_user_interruption(UserStartedSpeakingFrame())
            self._have_sent_user_started_speaking = True

        completed_text = completed_text.strip()
        completed_text = completed_text.replace(self.eou_string, "").replace(self.eob_string, "")

        if self.use_diar and not completed_text.startswith("<speaker_") and self._prev_speaker_id is not None:
            # Add the previous speaker tag to the beginning of the text
            completed_text = f"<speaker_{self._prev_speaker_id}> {completed_text}"

        frame_type = TranscriptionFrame if is_final else InterimTranscriptionFrame
        text_frame = frame_type(
            text=completed_text,
            user_id="",  # No speaker ID in this implementation
            timestamp=time_now_iso8601(),
            language=self.language if self.language else Language.EN_US,
            result={"text": completed_text},
        )
        logger.debug(f"Pushing text frame: {text_frame}")
        await self.push_frame(text_frame, direction)

    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame, direction: FrameDirection):
        self._vad_user_speaking = True
        logger.debug("NeMoTurnTakingService: VADUserStartedSpeakingFrame")
        await self.push_frame(frame, direction)

    def _contains_only_speaker_tags(self, text: str) -> bool:
        """
        Check if the text contains only speaker tags.
        """
        return text.strip().startswith("<speaker_") and text.strip().endswith(">")

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame, direction: FrameDirection):
        """
        Handle the user stopped speaking frame.

        If the buffer is not empty:
            - If bot is not speaking: push completed text regardless of backchannel
            - If bot is speaking: ignore backchannel strings
        If the buffer is empty, do nothing.
        """
        if self.use_vad:
            self._vad_user_speaking = False
        logger.debug("NeMoTurnTakingService: VADUserStoppedSpeakingFrame")
        await self.push_frame(frame, direction)

        # if user buffer only contains speaker tags, we don't push the completed text frame
        if self._contains_only_speaker_tags(self._user_speaking_buffer):
            logger.debug(f"User buffer only contains speaker tags: `{self._user_speaking_buffer}`, ignoring")
            return

        is_backchannel = self.is_backchannel(self._user_speaking_buffer)
        if not self._user_speaking_buffer:
            return
        if not self._bot_speaking or not is_backchannel:
            logger.debug(f"Bot talking: {self._bot_speaking}, backchannel: {is_backchannel}")
            logger.debug(f"Pushing completed text frame for VAD user stopped speaking: {self._user_speaking_buffer}")
            await self._handle_completed_text(self._user_speaking_buffer, direction)
            self._user_speaking_buffer = ""
            if self._have_sent_user_started_speaking:
                await self._handle_user_interruption(UserStoppedSpeakingFrame())
                self._have_sent_user_started_speaking = False
        elif is_backchannel:
            logger.debug(f"Backchannel detected: `{self._user_speaking_buffer}`")
            if self._audio_logger:
                self._audio_logger.save_user_audio(is_backchannel=True)
                logger.debug(
                    f"[TurnTakingService] Saved backchannel audio (VAD stopped): {self._user_speaking_buffer}"
                )

            await self.push_frame(
                TranscriptionFrame(
                    text=f"({self._user_speaking_buffer})",
                    user_id="",
                    timestamp=time_now_iso8601(),
                    language=self.language if self.language else Language.EN_US,
                    result={"text": f"Backchannel detected: {self._user_speaking_buffer}"},
                ),
                direction=FrameDirection.UPSTREAM,
            )
            self._user_speaking_buffer = ""
            self._have_sent_user_started_speaking = False

    async def _handle_user_interruption(self, frame: Frame):
        # Adapted from BaseInputTransport._handle_user_interruption
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("User started speaking")
            await self.push_frame(frame)
            await self.push_frame(StartInterruptionFrame(), direction=FrameDirection.DOWNSTREAM)
            # Record cutoff time for agent audio when TTS is interrupted
            if self._audio_logger and self._bot_speaking:
                self._audio_logger.set_agent_cutoff_time()
            # Increment turn index when user starts speaking (only if speaker changed)
            if self._audio_logger:
                self._audio_logger.increment_turn_index(speaker="user")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("User stopped speaking")
            await self.push_frame(frame)
        else:
            logger.debug(f"Unknown frame type for _handle_user_interruption: {type(frame)}")

    async def _handle_diar_result(self, frame: DiarResultFrame, direction: FrameDirection):
        if not self.use_diar:
            logger.debug("Diarization is disabled, skipping")
            return

        new_speaker_id = frame.diar_result  # speaker id of the dominant speaker

        # logger.debug(f"Dominant speaker ID: {dominant_speaker_id}")
        self._prev_speaker_id = self._current_speaker_id
        last_speaker_id = self._current_speaker_id

        if not self._user_speaking_buffer.startswith("<speaker_"):
            # add speaker tag <speaker_{speaker_id}> to the beginning of the current utterance
            self._user_speaking_buffer = f"<speaker_{new_speaker_id}> {self._user_speaking_buffer}"
        elif last_speaker_id != new_speaker_id:
            # change the speaker tag to the dominant speaker id
            self._user_speaking_buffer = self._user_speaking_buffer[len("<speaker_0>") :]
            self._user_speaking_buffer = f"<speaker_{new_speaker_id}> {self._user_speaking_buffer}"
        logger.debug(f"Speaker changed from {last_speaker_id} to {new_speaker_id}")
        self._current_speaker_id = new_speaker_id
