# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import torch

from nemo.collections.asr.inference.streaming.framing.mono_stream import MonoStream
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.streaming.framing.stream import Stream


class SilencePaddedStream(Stream):
    """Wraps a ``MonoStream`` and appends silence frames after the real audio
    to reach a target duration.

    The pipeline's ``run()`` loop sees a single, longer stream — no frame
    mutation or side-channel silence injection is needed.  ``MultiStream``
    keeps the stream alive until the final silence frame sets ``is_last=True``.
    """

    def __init__(
        self,
        inner: MonoStream,
        chunk_size_in_secs: float,
        pad_to_sec: float | None = None,
        pad_by_sec: float | None = None,
        pad_ratio: float | None = None,
    ):
        super().__init__(inner.stream_id)
        self.inner = inner
        self.chunk_size_in_secs = chunk_size_in_secs
        self.pad_to_sec = pad_to_sec
        self.pad_by_sec = pad_by_sec
        self.pad_ratio = pad_ratio
        self._inner_exhausted = False
        self._silence_frames_remaining = 0

    def load_audio(self, audio, options=None):
        self.inner.load_audio(audio, options)
        audio_secs = self.inner.n_samples / self.inner.rate
        remaining = self._padding_secs(audio_secs)
        self._silence_frames_remaining = (
            max(1, round(remaining / self.chunk_size_in_secs)) if remaining > 0 else 0
        )

    def _padding_secs(self, elapsed: float) -> float:
        if self.pad_to_sec is not None:
            return max(0.0, self.pad_to_sec - elapsed)
        if self.pad_ratio is not None:
            return elapsed * self.pad_ratio
        if self.pad_by_sec is not None:
            return self.pad_by_sec
        return 0.0

    def __iter__(self):
        self.inner.__iter__()
        self._inner_exhausted = False
        return self

    def __next__(self) -> list[Frame]:
        if not self._inner_exhausted:
            frames = next(self.inner)
            frame = frames[0]
            if frame.is_last and self._silence_frames_remaining > 0:
                modified = Frame(
                    samples=frame.samples,
                    stream_id=frame.stream_id,
                    is_first=frame.is_first,
                    is_last=False,
                    length=frame.length,
                    options=frame.options,
                )
                self._inner_exhausted = True
                return [modified]
            return frames

        if self._silence_frames_remaining > 0:
            self._silence_frames_remaining -= 1
            return [Frame(
                samples=torch.zeros(self.inner.frame_size),
                stream_id=self.stream_id,
                is_first=False,
                is_last=(self._silence_frames_remaining == 0),
                length=self.inner.frame_size,
            )]

        raise StopIteration
