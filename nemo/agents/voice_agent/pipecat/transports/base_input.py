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


from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.transports.base_input import BaseInputTransport as _BaseInputTransport


class BaseInputTransport(_BaseInputTransport):
    async def _handle_vad(self, audio_frame: InputAudioRawFrame, vad_state: VADState):
        """Handle Voice Activity Detection results and generate appropriate frames."""
        new_vad_state = await self._vad_analyze(audio_frame)
        if new_vad_state != vad_state and new_vad_state != VADState.STARTING and new_vad_state != VADState.STOPPING:
            frame = None
            # If the turn analyser is enabled, this will prevent:
            # - Creating the UserStoppedSpeakingFrame
            # - Creating the UserStartedSpeakingFrame multiple times
            can_create_user_frames = (
                self._params.turn_analyzer is None or not self._params.turn_analyzer.speech_triggered
            ) and self._params.can_create_user_frames

            if new_vad_state == VADState.SPEAKING:
                await self.push_frame(VADUserStartedSpeakingFrame())
                if can_create_user_frames:
                    frame = UserStartedSpeakingFrame()
                else:
                    logger.debug("base_input: VAD state changed to SPEAKING but can_create_user_frames is False")
            elif new_vad_state == VADState.QUIET:
                await self.push_frame(VADUserStoppedSpeakingFrame())
                if can_create_user_frames:
                    frame = UserStoppedSpeakingFrame()
                else:
                    logger.debug("base_input: VAD state changed to QUIET but can_create_user_frames is False")

            if frame:
                await self._handle_user_interruption(frame)

            vad_state = new_vad_state
        return vad_state
