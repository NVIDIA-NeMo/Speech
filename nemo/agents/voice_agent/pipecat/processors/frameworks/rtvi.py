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
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, TTSTextFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frameworks.rtvi import RTVIObserver as _RTVIObserver
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.transports.base_output import BaseOutputTransport


class RTVIObserver(_RTVIObserver):
    """
    An observer that processes RTVI frames and pushes them to the transport.
    """

    TRANSPORT_OUTPUT_FRAMES = (TTSTextFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame)

    def __init__(self, rtvi: RTVIProcessor, *args, **kwargs):
        super().__init__(rtvi, *args, **kwargs)
        self.transport_output_seen = set()

    async def on_push_frame(self, data: FramePushed):
        """Process a frame being pushed through the pipeline.

        Args:
            data: Frame push event data containing source, frame, direction, and timestamp.
        """
        src = data.source
        frame: Frame = data.frame

        if isinstance(frame, BotStoppedSpeakingFrame) and isinstance(src, BaseOutputTransport):
            logger.debug(
                f"Bot stopped speaking in RTVIObserver: {frame}, seen: {frame.id in self.transport_output_seen}"
            )
        if isinstance(frame, self.TRANSPORT_OUTPUT_FRAMES) and isinstance(src, BaseOutputTransport):
            if frame.id not in self.transport_output_seen and frame.id in self._frames_seen:
                self._frames_seen.remove(frame.id)

        await super().on_push_frame(data)

        if isinstance(frame, self.TRANSPORT_OUTPUT_FRAMES) and isinstance(src, BaseOutputTransport):
            self.transport_output_seen.add(frame.id)

    def reset(self):
        """Reset the observer."""
        self.transport_output_seen.clear()
