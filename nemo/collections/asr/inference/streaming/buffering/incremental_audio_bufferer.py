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


class IncrementalAudioBufferer:
    """
    Incremental Audio bufferer class
    It buffers the audio chunks and maintains the buffer.
    """

    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        chunk_size_in_secs: float,
        overlap_size_in_secs: float,
    ) -> None:
        """
        Args:
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
            chunk_size_in_secs (float): chunk size in seconds
            overlap_size_in_secs (float): overlap size in seconds
        """
        self.sample_rate = sample_rate
        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.chunk_size = int(chunk_size_in_secs * sample_rate)
        self.overlap_size = int(overlap_size_in_secs * sample_rate)

        # Ensure overlap is within buffer bounds to keep drop_size non-negative and meaningful.
        if not (0 <= self.overlap_size <= self.buffer_size):
            raise ValueError(
                f"Overlap size in samples ({self.overlap_size}) must satisfy "
                f"0 <= overlap_size <= buffer_size ({self.buffer_size})."
            )

        if self.buffer_size % self.chunk_size != 0:
            raise ValueError(f"Buffer size ({self.buffer_size}) must be divisible by chunk size ({self.chunk_size})")

        if self.overlap_size % self.chunk_size != 0:
            raise ValueError(f"Overlap size ({self.overlap_size}) must be divisible by chunk size ({self.chunk_size})")

        self.drop_size = self.buffer_size - self.overlap_size
        self.sample_buffer = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.remaining_capacity = self.buffer_size
        self.head = 0

    def is_full(self) -> bool:
        """
        Check if the buffer is full
        Returns:
            bool: True if the buffer is full, False otherwise
        """
        return self.remaining_capacity == 0

    def update(self, frame: Frame) -> None:
        """
        Update the buffer with the new frame
        Args:
            frame (Frame): frame to update the buffer with
        """
        if frame.size > self.buffer_size:
            raise RuntimeError(f"Frame size ({frame.size}) exceeds buffer size ({self.buffer_size})")

        if self.is_full():
            # Drop the oldest chunk to make space for the new chunk
            self.sample_buffer[0 : self.drop_size].zero_()
            self.sample_buffer = torch.roll(self.sample_buffer, -self.drop_size)
            self.head -= self.drop_size
            self.remaining_capacity += self.drop_size

        self.sample_buffer[self.head : self.head + frame.size].copy_(frame.samples)
        self.head += frame.size
        self.remaining_capacity = max(0, self.remaining_capacity - frame.size)


class BatchedIncrementalAudioBufferer:
    """
    Batched incremental audio bufferer class
    It buffers the audio chunks from multiple streams and returns the buffers.
    """

    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        chunk_size_in_secs: float,
        overlap_size_in_secs: float,
    ) -> None:
        """
        Args:
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
            chunk_size_in_secs (float): chunk size in seconds
            overlap_size_in_secs (float): overlap size in seconds
        """
        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.chunk_size_in_secs = chunk_size_in_secs
        self.overlap_size_in_secs = overlap_size_in_secs
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

    def is_full(self, stream_id: int) -> bool | None:
        """
        Check if the buffer is full for the given stream id
        Returns:
            bool | None: True if the buffer is full, False otherwise
        """
        if stream_id not in self.bufferers:
            return None
        return self.bufferers[stream_id].is_full()

    def update(self, frames: list[Frame]) -> tuple[list[Tensor], list[int]]:
        """
        Update the bufferers with the new frames.
        Frames can come from different streams (audios), so we need to maintain a bufferer for each stream
        Args:
            frames (list[Frame]): list of frames
        Returns:
            tuple[list[Tensor], list[int]]:
                buffers: list of buffered audio tensors, one per input frame
                paddings: list of paddings, one per input frame
        """
        buffers, paddings = [], []
        for frame in frames:
            bufferer = self.bufferers.get(frame.stream_id, None)

            if bufferer is None:
                bufferer = IncrementalAudioBufferer(
                    self.sample_rate,
                    self.buffer_size_in_secs,
                    self.chunk_size_in_secs,
                    self.overlap_size_in_secs,
                )
                self.bufferers[frame.stream_id] = bufferer

            bufferer.update(frame)
            buffers.append(bufferer.sample_buffer.clone())
            paddings.append(bufferer.remaining_capacity)

            if frame.is_last:
                self.rm_bufferer(frame.stream_id)

        return buffers, paddings
