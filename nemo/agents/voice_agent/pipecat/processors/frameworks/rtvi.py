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

from typing import Optional

from loguru import logger
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.frameworks.rtvi import RTVIBotLLMTextMessage, RTVIBotTranscriptionMessage
from pipecat.processors.frameworks.rtvi import RTVIObserver as _RTVIObserver
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVITextMessageData

from nemo.agents.voice_agent.pipecat.utils.text.simple_text_aggregator import SimpleSegmentedTextAggregator


class RTVIObserver(_RTVIObserver):
    def __init__(
        self, rtvi: RTVIProcessor, text_aggregator: Optional[SimpleSegmentedTextAggregator] = None, *args, **kwargs
    ):
        super().__init__(rtvi, *args, **kwargs)
        self._text_aggregator = text_aggregator if text_aggregator else SimpleSegmentedTextAggregator("?!:.")

    async def _handle_llm_text_frame(self, frame: LLMTextFrame):
        """Handle LLM text output frames."""
        message = RTVIBotLLMTextMessage(data=RTVITextMessageData(text=frame.text))
        await self.push_transport_message_urgent(message)

        completed_text = await self._text_aggregator.aggregate(frame.text)
        if completed_text:
            await self._push_bot_transcription(completed_text)

    async def _push_bot_transcription(self, text: str):
        """Push accumulated bot transcription as a message."""
        if len(text.strip()) > 0:
            message = RTVIBotTranscriptionMessage(data=RTVITextMessageData(text=text))
            logger.debug(f"Pushing bot transcription: `{text}`")
            await self.push_transport_message_urgent(message)
            self._bot_transcription = ""
