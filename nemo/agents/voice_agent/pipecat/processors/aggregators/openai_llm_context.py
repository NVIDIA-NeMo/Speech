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

import asyncio
import base64
import io
import wave
from pathlib import Path
from typing import Optional

from loguru import logger
from pipecat.frames.frames import AudioRawFrame
from pipecat.processors.aggregators.llm_context import LLMContextMessage
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext as BaseOpenAILLMContext


class OpenAILLMContext(BaseOpenAILLMContext):

    @staticmethod
    async def create_audio_message(
        *,
        role: str = "user",
        audio_frames: list[AudioRawFrame],
        text: str = "Audio follows",
        audio_output_path: Optional[Path] = None,
    ) -> LLMContextMessage:
        """Create a context message containing audio.

        Args:
            role: The role of this message (defaults to "user").
            audio_frames: List of audio frame objects to include.
            text: Optional text to include with the audio.
        """
        content = [{"type": "text", "text": text}]

        def encode_audio() -> str:
            """Sync WAV encode; must not be async — ``asyncio.to_thread`` only awaits sync callables."""
            sample_rate = audio_frames[0].sample_rate
            num_channels = audio_frames[0].num_channels

            data = b"".join(frame.audio for frame in audio_frames)

            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(num_channels)
                    wf.setframerate(sample_rate)
                    wf.writeframes(data)

                wav_bytes = buffer.getvalue()
                encoded_audio = base64.b64encode(wav_bytes).decode("utf-8")

            if audio_output_path is not None:
                audio_output_path.parent.mkdir(parents=True, exist_ok=True)
                audio_output_path.write_bytes(wav_bytes)
                logger.debug("Wrote LLM user audio WAV to {}", audio_output_path)

            return encoded_audio

        encoded_audio = await asyncio.to_thread(encode_audio)

        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": encoded_audio, "format": "wav"},
            }
        )

        return {"role": role, "content": content}

    async def add_audio_frames_message(self, *, audio_frames: list[AudioRawFrame], text: str) -> None:
        message = await self.create_audio_message(audio_frames=audio_frames, text=text)
        self.add_message(message)
