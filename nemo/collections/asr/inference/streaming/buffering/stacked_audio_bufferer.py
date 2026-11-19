# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


class StackedAudioBufferer:
    """
    Audio bufferer that keeps the buffers of all running streams in one [num_slots, buffer_size] tensor
    on the given device. A batch of frames is buffered with a few vectorized operations, and the
    result is a batch of buffers instead of a list of per-stream tensors.
    It buffers exactly like BatchedAudioBufferer.
    """

    def __init__(self, sample_rate: int, buffer_size_in_secs: float, device: torch.device | str, num_slots: int = 64):
        """
        Args:
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
            device (torch.device | str): device that holds the buffers
            num_slots (int): initial number of streams that fit, the capacity doubles when it is exceeded
        """
        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.device = torch.device(device)
        self.reset(num_slots)

    def reset(self, num_slots: int | None = None) -> None:
        """
        Drop all streams and zero the buffers
        Args:
            num_slots (int | None): number of slots to allocate, None keeps the current capacity
        """
        if num_slots is None:
            num_slots = self.buffer.shape[0]
        self.buffer = torch.zeros(num_slots, self.buffer_size, dtype=torch.float32, device=self.device)
        self.left_padding = torch.full((num_slots,), self.buffer_size, dtype=torch.int64, device=self.device)
        self.stream_id2slot = {}
        self.free_slots = list(range(num_slots - 1, -1, -1))

    def _grow(self) -> None:
        """Double the number of slots"""
        old = self.buffer.shape[0]
        new = max(1, 2 * old)
        self.buffer = torch.cat([self.buffer, torch.zeros(new - old, self.buffer_size, device=self.device)])
        self.left_padding = torch.cat(
            [self.left_padding, torch.full((new - old,), self.buffer_size, dtype=torch.int64, device=self.device)]
        )
        self.free_slots = list(range(new - 1, old - 1, -1))

    def _get_slot(self, stream_id: int) -> int:
        """Get the slot of the stream, a new stream takes a free slot with a zero buffer"""
        slot = self.stream_id2slot.get(stream_id)
        if slot is None:
            if not self.free_slots:
                self._grow()
            slot = self.free_slots.pop()
            self.buffer[slot].zero_()
            self.left_padding[slot] = self.buffer_size
            self.stream_id2slot[stream_id] = slot
        return slot

    def rm_stream(self, stream_id: int) -> None:
        """
        Release the slot of the given stream
        Args:
            stream_id (int): stream id
        """
        slot = self.stream_id2slot.pop(stream_id, None)
        if slot is not None:
            self.free_slots.append(slot)

    def update(self, frames: list[Frame]) -> tuple[Tensor, Tensor]:
        """
        Update the buffers with the new frames.
        Every frame must come from a different stream and have the same size.
        Args:
            frames (list[Frame]): list of frames
        Returns:
            tuple[Tensor, Tensor]:
                buffers: [len(frames), buffer_size] float tensor on the device, a copy of the state
                left_paddings: [len(frames)] int64 tensor on the device, one per input frame
        """
        if len(frames) == 0:
            return (
                torch.zeros(0, self.buffer_size, device=self.device),
                torch.zeros(0, dtype=torch.int64, device=self.device),
            )

        shift = frames[0].size
        if any(frame.size != shift for frame in frames):
            raise ValueError("All frames of a batch must have the same size, use pad_last_frame=True")
        if shift > self.buffer_size:
            raise RuntimeError(f"Frame size ({shift}) exceeds buffer size ({self.buffer_size})")
        if len({frame.stream_id for frame in frames}) != len(frames):
            raise ValueError("A batch of frames must have one frame per stream")

        slots = torch.tensor([self._get_slot(frame.stream_id) for frame in frames], device=self.device)
        samples = torch.stack([frame.samples for frame in frames]).to(self.device, dtype=torch.float32)

        buffers = torch.roll(self.buffer[slots], -shift, dims=1)
        buffers[:, -shift:] = samples
        self.buffer[slots] = buffers

        left_paddings = (self.left_padding[slots] - shift).clamp_(min=0)
        self.left_padding[slots] = left_paddings

        for frame in frames:
            if frame.is_last:
                self.rm_stream(frame.stream_id)

        return buffers, left_paddings
