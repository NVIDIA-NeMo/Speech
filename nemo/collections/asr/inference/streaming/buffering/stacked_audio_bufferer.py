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
    The buffer of a stream holds its last buffer_size samples, zero padded in front.
    """

    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        device: torch.device | str,
        num_slots: int = 64,
        buffer_size: int | None = None,
    ):
        """
        Args:
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
            device (torch.device | str): device that holds the buffers
            num_slots (int): initial number of streams that fit, the capacity doubles when it is exceeded
            buffer_size (int | None): buffer size in samples, overrides buffer_size_in_secs when given
        """
        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.buffer_size = int(buffer_size_in_secs * sample_rate) if buffer_size is None else buffer_size
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
            self.reset_slots([slot])
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

    def reset_slots(self, slots: list[int]) -> None:
        """
        Zero the buffers of the given slots
        Args:
            slots (list[int]): slot indices
        """
        slots = torch.tensor(slots, device=self.device, dtype=torch.long)
        self.buffer.index_fill_(0, slots, 0.0)
        self.left_padding.index_fill_(0, slots, self.buffer_size)

    def update(self, frames: list[Frame]) -> tuple[Tensor, Tensor]:
        """
        Update the buffers with the new frames, every stream gets a slot on its first frame
        and releases it on its last frame.
        Every frame must come from a different stream and have the same size.
        Args:
            frames (list[Frame]): list of frames
        Returns:
            tuple[Tensor, Tensor]:
                buffers: [len(frames), buffer_size] float tensor on the device, a copy of the state
                left_paddings: [len(frames)] int64 tensor on the device, one per input frame
        """
        self._check_frame_sizes(frames)
        if len({frame.stream_id for frame in frames}) != len(frames):
            raise ValueError("A batch of frames must have one frame per stream")

        outputs = self.update_slots([self._get_slot(frame.stream_id) for frame in frames], frames)

        for frame in frames:
            if frame.is_last:
                self.rm_stream(frame.stream_id)

        return outputs

    def update_slots(self, slots: list[int], frames: list[Frame]) -> tuple[Tensor, Tensor]:
        """
        Update the buffers of the given slots with the new frames, for callers that assign the slots themselves.
        Every frame must go to a different slot and have the same size.
        Args:
            slots (list[int]): slot index of every frame
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

        shift = self._check_frame_sizes(frames)
        if len(set(slots)) != len(frames):
            raise ValueError("A batch of frames must have one frame per stream")

        slots = torch.tensor(slots, device=self.device, dtype=torch.long)
        samples = torch.stack([frame.samples for frame in frames]).to(self.device, dtype=torch.float32)

        buffers = torch.cat([self.buffer.index_select(0, slots)[:, shift:], samples], dim=1)
        self.buffer.index_copy_(0, slots, buffers)

        left_paddings = (self.left_padding.index_select(0, slots) - shift).clamp_(min=0)
        self.left_padding.index_copy_(0, slots, left_paddings)

        return buffers, left_paddings

    def _check_frame_sizes(self, frames: list[Frame]) -> int:
        """Get the common size of the frames, 0 for an empty batch"""
        if len(frames) == 0:
            return 0
        size = frames[0].size
        if any(frame.size != size for frame in frames):
            raise ValueError(
                "All frames of a batch must have the same size; pad the last frame of a stream to the frame size "
                "and set `length` to the number of valid samples"
            )
        if size > self.buffer_size:
            raise RuntimeError(f"Frame size ({size}) exceeds buffer size ({self.buffer_size})")
        return size
