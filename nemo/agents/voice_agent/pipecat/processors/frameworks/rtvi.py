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
from pipecat.frames.frames import Frame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, TTSTextFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frameworks.rtvi import (
    RTVIBotLLMStartedMessage,
    RTVIBotLLMStoppedMessage,
    RTVIBotTranscriptionMessage,
    RTVIBotTTSTextMessage,
)
from pipecat.processors.frameworks.rtvi import RTVIObserver as _RTVIObserver
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVITextMessageData
from pipecat.transports.base_output import BaseOutputTransport


class RTVIObserver(_RTVIObserver):
    """
    An observer that processes RTVI frames and pushes them to the transport.
    """

    def __init__(self, rtvi: RTVIProcessor, *args, **kwargs):
        super().__init__(rtvi, *args, **kwargs)

    async def on_push_frame(self, data: FramePushed):
        """Process a frame being pushed through the pipeline.

        Args:
            data: Frame push event data containing source, frame, direction, and timestamp.
        """
        src = data.source
        frame: Frame = data.frame

        if frame.id in self._frames_seen:
            return

        if not self._params.bot_llm_enabled:
            if isinstance(frame, LLMFullResponseStartFrame):
                await self.send_rtvi_message(RTVIBotLLMStartedMessage())
                self._frames_seen.add(frame.id)
            elif isinstance(frame, LLMFullResponseEndFrame):
                await self.send_rtvi_message(RTVIBotLLMStoppedMessage())
                self._frames_seen.add(frame.id)
            elif isinstance(frame, TTSTextFrame) and isinstance(src, BaseOutputTransport):
                message = RTVIBotTTSTextMessage(data=RTVITextMessageData(text=frame.text))
                await self.send_rtvi_message(message)
                await self._push_bot_transcription(frame.text)
                self._frames_seen.add(frame.id)
            else:
                await super().on_push_frame(data)
        else:
            await super().on_push_frame(data)

    async def _push_bot_transcription(self, text: str):
        """Push accumulated bot transcription as a message."""
        if len(text.strip()) > 0:
            message = RTVIBotTranscriptionMessage(data=RTVITextMessageData(text=text))
            logger.debug(f"Pushing bot transcription: `{text}`")
            await self.send_rtvi_message(message)
