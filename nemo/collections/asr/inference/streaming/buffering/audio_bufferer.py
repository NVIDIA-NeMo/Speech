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
from torch import Tensor
from nemo.collections.asr.inference.streaming.framing.request import Frame


class AudioBufferer:
    """
    Audio bufferer class
    It buffers the audio chunks and maintains the buffer.
    """

    def __init__(self, sample_rate: int, buffer_size_in_secs: float):
        """
        Args:
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
        """
        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.sample_buffer = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.left_padding = self.buffer_size

    def reset(self) -> None:
        """
        Reset the buffer to zero
        """
        self.sample_buffer.zero_()
        self.left_padding = self.buffer_size

    def update(self, frame: Frame) -> None:
        """
        Update the buffer with the new frame
        Args:
            frame (Frame): frame to update the buffer with
        """
        if frame.size > self.buffer_size:
            raise RuntimeError(f"Frame size ({frame.size}) exceeds buffer size ({self.buffer_size})")

        shift = frame.size
        self.sample_buffer = torch.roll(self.sample_buffer, -shift)
        self.sample_buffer[-shift:].copy_(frame.samples)
        self.left_padding = max(0, self.left_padding - shift)

    def get_buffer(self) -> Tensor:
        """
        Get the current buffer
        Returns:
            Tensor: current state of the buffer
        """
        return self.sample_buffer.clone()

    def get_left_padding(self) -> int:
        """
        Get the left padding
        Returns:
            int: left padding
        """
        return self.left_padding


class BatchedAudioBufferer:
    """
    Batched audio bufferer class
    It buffers the audio chunks from multiple streams and returns the buffers.
    """

    def __init__(self, sample_rate: int, buffer_size_in_secs: float):
        """
        Args:
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
        """
        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.bufferers = {}

    def reset(self) -> None:
        """
        Reset bufferers
        """
        self.bufferers = {}

    def rm_bufferer(self, stream_id: int) -> None:
        """
        Remove bufferer for the given stream id
        Args:
            stream_id (int): stream id
        """
        self.bufferers.pop(stream_id, None)

    def update(self, frames: list[Frame]) -> tuple[list[Tensor], list[int]]:
        """
        Update the bufferers with the new frames.
        Frames can come from different streams (audios), so we need to maintain a bufferer for each stream
        Args:
            frames (list[Frame]): list of frames
        Returns:
            tuple[list[Tensor], list[int]]:
                buffers: list of buffered audio tensors, one per input frame
                left_paddings: list of left paddings, one per input frame
        """
        buffers, left_paddings = [], []
        for frame in frames:
            bufferer = self.bufferers.get(frame.stream_id, None)

            if bufferer is None:
                bufferer = AudioBufferer(self.sample_rate, self.buffer_size_in_secs)
                self.bufferers[frame.stream_id] = bufferer

            bufferer.update(frame)
            buffers.append(bufferer.get_buffer())
            left_paddings.append(bufferer.get_left_padding())

            if frame.is_last:
                self.rm_bufferer(frame.stream_id)

        return buffers, left_paddings
