# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Adapted from https://github.com/pipecat-ai/pipecat/blob/v0.0.98/examples/foundational/07s-interruptible-google-audio-in.py

from loguru import logger
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMRunFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator
from pipecat.processors.frame_processor import FrameProcessor


class UserAudioBuffer(FrameProcessor):
    def __init__(
        self,
        *,
        context: LLMContext,
        user_context_aggregator: LLMUserContextAggregator,
        pre_cache_duration_secs: float = 0.2,
        use_transcript: bool = False,
        text_prompt: str = "Follow instructions or answer questions in the audio.",
        text_prompt_for_transcript: str = "Here is the pseudo-transcript of the audio for reference: ",
        raw_audio_frame_len_in_secs: float = 0.016,
        keep_only_last_audio_turn: bool = True,
    ) -> None:
        """
        Args:
            context: The LLM context to add the audio frames to.
            user_context_aggregator: The user context aggregator to push the LLM run frame to.
            pre_cache_duration_secs: The duration of the audio to cache before the user starts speaking.
            use_transcript: Whether to add the transcript as auxiliary user text input to the LLM context.
            text_prompt: The prompt to add to the LLM context when the user starts speaking.
            text_prompt_for_transcript: The prompt to add to the LLM context when the user stops speaking and the transcript is available.
            raw_audio_frame_len_in_secs: The length of the audio frame in seconds.
            keep_only_last_audio_turn: Whether to keep only the last audio turn.
        """
        super().__init__()
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_frames = []
        self._start_secs = pre_cache_duration_secs  # this should match VAD start_secs
        self._user_speaking = False
        self._transcript_buffer = []
        self._use_transcript = use_transcript
        self._text_prompt = text_prompt
        self._text_prompt_for_transcript = text_prompt_for_transcript
        self._raw_audio_frame_len_in_secs = raw_audio_frame_len_in_secs
        self._previsous_user_text = ""
        self._keep_only_last_audio_turn = keep_only_last_audio_turn

    @property
    def buffer_duration(self) -> float:
        """
        Get the duration of the audio frames.
        """
        if not self._audio_frames:
            return 0.0
        return self._raw_audio_frame_len_in_secs * len(self._audio_frames)

    def _finalize_transcript(self) -> str:
        """
        Finalize the transcript.
        """
        return ' '.join(self._transcript_buffer).strip()

    def _replace_previous_audio_turns(self) -> None:
        """
        Replace the old audio turns with the `_previsous_user_text`.
        """
        last_audio_turn_idx = None
        idx = len(self._context.messages) - 1
        seen_user_turns = 0
        while idx >= 0:
            # skip the current user turn
            if self._context.messages[idx]["role"] == "user" and isinstance(
                self._context.messages[idx]["content"], list
            ):
                last_audio_turn_idx = idx
                break
            idx -= 1
            seen_user_turns += 1

        if last_audio_turn_idx is not None:
            self._context.messages[last_audio_turn_idx]["content"] = self._previsous_user_text
            logger.debug(f"Replaced old audio turn with: {self._previsous_user_text}")
            logger.debug(f"New context: {self._context.messages}")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self._transcript_buffer.append(frame.text)
            logger.debug(f"Added transcript to buffer: `{frame.text}`. New buffer: {self._transcript_buffer}")
            return
        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.debug(f"User started speaking")
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            logger.debug(f"User stopped speaking")
            if self._keep_only_last_audio_turn:
                self._replace_previous_audio_turns()
            text = self._text_prompt
            current_transcript = self._finalize_transcript()
            if self._use_transcript and self._transcript_buffer:
                text += f"\n{self._text_prompt_for_transcript}\n{current_transcript}"
            self._previsous_user_text = (
                f"{self._text_prompt}\n{self._text_prompt_for_transcript}\n`{current_transcript}`".strip()
            )
            await self._context.add_audio_frames_message(audio_frames=self._audio_frames, text=text)
            logger.debug(
                f"Adding audio frames message to context: {text}. Audio duration: {self.buffer_duration} seconds."
            )
            await self._user_context_aggregator.push_frame(LLMRunFrame())
            self._audio_frames = []
            self._transcript_buffer = []
        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                self._audio_frames.append(frame)
            else:
                # Append the audio frame to our buffer. Treat the buffer as a ring buffer, dropping the oldest
                # frames as necessary. Assume all audio frames have the same duration.
                self._audio_frames.append(frame)
                while self.buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)

        await self.push_frame(frame, direction)

    def reset(self) -> None:
        """
        Reset the audio buffer.
        """
        self._audio_frames = []
        self._transcript_buffer = []
        self._user_speaking = False
