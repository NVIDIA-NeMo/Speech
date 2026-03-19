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


import torch
from nemo.collections.asr.inference.streaming.framing.request import Frame, RequestOptions
from nemo.collections.asr.inference.streaming.framing.stream import Stream
from nemo.collections.asr.inference.utils.audio_io import read_audio


class MonoStream(Stream):
    """
    Streamer for mono wav files.
    Iterates over the frames of the audio file
    """

    def __init__(self, rate: int, frame_size_in_secs: float, stream_id: int, pad_last_frame: bool = False):
        """
        Initialize the MonoStream
        Args:
            rate (int): sampling rate
            frame_size_in_secs (int): frame length in seconds
            stream_id (int): stream id
        """

        self.rate = rate
        self.frame_size = int(frame_size_in_secs * rate)
        self.pad_last_frame = pad_last_frame

        self.samples = None
        self.n_samples = None
        self.options = None
        super().__init__(stream_id)

    def load_audio(self, audio: str | torch.Tensor, options: RequestOptions | None = None) -> None:
        """
        Load the audio file either from a file or from a torch tensor
        Args:
            audio (str | torch.Tensor): audio file path or torch tensor of audio samples
            options (RequestOptions | None): optional options for the request
        """
        if isinstance(audio, str):
            # Read the audio file and convert to mono
            self.samples = read_audio(audio, target_sr=self.rate, mono=True)
        else:
            self.samples = audio
        self.n_samples = len(self.samples)
        self.frame_count = 0  # Reset frame count
        self.options = options

    def __iter__(self):
        """Returns the frame iterator object"""
        self.start = 0
        self.frame_count = 0
        return self

    def __next__(self) -> list[Frame]:
        """
        Get the next frame in the stream
        Returns:
            list[Frame]: The next frame in the stream
        """
        if self.samples is None:
            raise RuntimeError("No audio samples loaded. Please call load_audio() first.")

        if self.start < self.n_samples:

            end = min(self.start + self.frame_size, self.n_samples)

            # Check if this is the last frame
            is_end = False
            chunk_length = end - self.start
            if (end - self.start < self.frame_size) or (end == self.n_samples):
                is_end = True

            # Pad the last frame if needed
            if not is_end:
                chunk_samples = self.samples[self.start : end]
            else:
                if self.pad_last_frame:
                    chunk_samples = torch.zeros(self.frame_size)
                    chunk_samples[:chunk_length] = self.samples[self.start : end]
                else:
                    chunk_samples = self.samples[self.start : end]

            # Package the frame
            is_first = self.frame_count == 0
            frame = Frame(
                samples=chunk_samples,
                stream_id=self.stream_id,
                is_first=is_first,
                is_last=is_end,
                length=chunk_length,
                options=self.options if is_first else None,
            )

            self.frame_count += 1
            self.start += frame.size

            return [frame]

        # End of stream
        raise StopIteration
