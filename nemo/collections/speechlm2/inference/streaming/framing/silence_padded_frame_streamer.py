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

from nemo.collections.asr.inference.streaming.framing.mono_stream import MonoStream
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedFrameStreamer
from nemo.collections.speechlm2.inference.streaming.framing.silence_padded_stream import SilencePaddedStream


class SilencePaddedContinuousBatchedFrameStreamer(ContinuousBatchedFrameStreamer):
    """``ContinuousBatchedFrameStreamer`` that optionally wraps each
    ``MonoStream`` in a :class:`SilencePaddedStream` so extra silence
    frames are yielded transparently at the end of each audio file.

    When no padding is configured the behaviour is identical to the base
    class.
    """

    def __init__(
        self,
        *,
        pad_to_sec: float | None = None,
        pad_by_sec: float | None = None,
        pad_ratio: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_to_sec = pad_to_sec
        self.pad_by_sec = pad_by_sec
        self.pad_ratio = pad_ratio

    @property
    def _needs_padding(self) -> bool:
        return any(x is not None for x in (self.pad_to_sec, self.pad_by_sec, self.pad_ratio))

    def add_stream(self) -> None:
        if self.stream_id >= self.n_audio_files:
            return

        inner = MonoStream(
            self.sample_rate,
            self.frame_size_in_secs,
            stream_id=self.stream_id,
            pad_last_frame=self.pad_last_frame,
        )

        if self._needs_padding:
            stream = SilencePaddedStream(
                inner,
                chunk_size_in_secs=self.frame_size_in_secs,
                pad_to_sec=self.pad_to_sec,
                pad_by_sec=self.pad_by_sec,
                pad_ratio=self.pad_ratio,
            )
        else:
            stream = inner

        audio_filepath = self.audio_filepaths[self.stream_id]
        self.sid2filepath[self.stream_id] = audio_filepath
        self.elapsed_durations[self.stream_id] = 0.0
        stream.load_audio(audio_filepath, self.options[self.stream_id])

        self.multi_streamer.add_stream(stream, stream_id=self.stream_id)
        self.stream_id += 1
        self.update_progress_bar()
