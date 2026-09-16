# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import random

import numpy as np
import torch
import torch.nn as nn

from nemo.collections.asr.parts.packed_sequence import PackedEncoderActivations, packed_encoder_position_ids
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import LengthsType, NeuralType, SpectrogramType


class SpecAugment(nn.Module, Typing):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
        Can be a positive integer or a float value in the range [0, 1].
        If positive integer value, defines maximum number of time steps
        to be cut in one segment.
        If a float value, defines maximum percentage of timesteps that
        are cut adaptively.
    use_vectorized_code - GPU-based implementation with batched masking and GPU rng,
        setting it to False reverts to the legacy implementation.
        Fast implementation is inspired by torchaudio:
        https://github.com/pytorch/audio/blob/ea437b31ce316ea3d66fe73768c0dcb94edb79ad/src/torchaudio/functional/functional.py#L816
    """

    FREQ_AXIS = 1  # Frequency axis in the spectrogram tensor
    TIME_AXIS = 2  # Time axis in the spectrogram tensor

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            "input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(
        self,
        freq_masks: int = 0,
        time_masks: int = 0,
        freq_width: int = 10,
        time_width: int | float = 10,
        rng: random.Random | None = None,
        mask_value: float = 0.0,
        use_vectorized_code: bool = True,
    ):
        super().__init__()

        self._rng = random.Random() if rng is None else rng

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        self.mask_value = mask_value
        self.use_vectorized_code = use_vectorized_code

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError("If `time_width` is a float value, must be in range [0, 1]")

            self.adaptive_temporal_width = True

    @typecheck()
    @torch.no_grad()
    def forward(self, input_spec, length):
        if self.use_vectorized_code:
            return self._forward_vectorized(input_spec, length)
        else:
            return self._forward_legacy(input_spec, length)

    @torch.no_grad()
    def forward_packed(self, input_spec: PackedEncoderActivations) -> PackedEncoderActivations:
        """Apply GPU-vectorized sequence-local masks directly to token-flat features.

        ``use_vectorized_code`` only preserves the historical dense backend choice;
        the new packed API always avoids the legacy Python loop.
        """
        data = _apply_packed_axis_masks(
            input_spec.data,
            input_spec,
            num_masks=self.time_masks,
            width=self.time_width,
            axis=self.TIME_AXIS,
            mask_value=self.mask_value,
        )
        data = _apply_packed_axis_masks(
            data,
            input_spec,
            num_masks=self.freq_masks,
            width=self.freq_width,
            axis=self.FREQ_AXIS,
            mask_value=self.mask_value,
        )
        return input_spec.with_data(data)

    def _forward_legacy(self, input_spec, length):
        batch_size, num_freq_bins, _ = input_spec.shape
        # Move lengths to CPU before repeated indexing
        lengths_cpu = length.cpu().numpy()
        # Generate a numpy boolean mask. `True` elements represent where the input spec will be augmented.
        fill_mask: np.array = np.full(shape=input_spec.shape, fill_value=False)
        freq_start_upper_bound = num_freq_bins - self.freq_width
        # Choose different mask ranges for each element of the batch
        for idx in range(batch_size):
            # Set freq masking
            for _ in range(self.freq_masks):
                start = self._rng.randint(0, freq_start_upper_bound)
                width = self._rng.randint(0, self.freq_width)
                fill_mask[idx, start : start + width, :] = True

            # Derive time width, sometimes based percentage of input length.
            if self.adaptive_temporal_width:
                time_max_width = max(1, int(lengths_cpu[idx] * self.time_width))
            else:
                time_max_width = self.time_width
            time_start_upper_bound = max(1, lengths_cpu[idx] - time_max_width)

            # Set time masking
            for _ in range(self.time_masks):
                start = self._rng.randint(0, time_start_upper_bound)
                width = self._rng.randint(0, time_max_width)
                fill_mask[idx, :, start : start + width] = True
        # Bring the mask to device and fill spec
        fill_mask = torch.from_numpy(fill_mask).to(input_spec.device)
        masked_spec = input_spec.masked_fill(mask=fill_mask, value=self.mask_value)
        return masked_spec

    def _forward_vectorized(self, input_spec: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        # time masks
        input_spec = self._apply_masks(
            input_spec=input_spec,
            num_masks=self.time_masks,
            length=length,
            width=self.time_width,
            axis=self.TIME_AXIS,
            mask_value=self.mask_value,
        )
        # freq masks
        input_spec = self._apply_masks(
            input_spec=input_spec,
            num_masks=self.freq_masks,
            length=length,
            width=self.freq_width,
            axis=self.FREQ_AXIS,
            mask_value=self.mask_value,
        )
        return input_spec

    def _apply_masks(
        self,
        input_spec: torch.Tensor,
        num_masks: int,
        length: torch.Tensor,
        width: int | float,
        mask_value: float,
        axis: int,
    ) -> torch.Tensor:

        assert axis in (
            self.FREQ_AXIS,
            self.TIME_AXIS,
        ), f"Axis can be only be equal to frequency \
            ({self.FREQ_AXIS}) or time ({self.TIME_AXIS}). Received: {axis=}"
        assert not (
            isinstance(width, float) and axis == self.FREQ_AXIS
        ), "Float width supported \
            only with time axis."

        batch_size = input_spec.shape[0]
        axis_length = input_spec.shape[axis]

        # If width is float then it is transformed into a tensor
        if axis == self.TIME_AXIS and isinstance(width, float):
            width = torch.clamp(width * length, max=axis_length).unsqueeze(1)

        # Generate [0-1) random numbers and then scale the tensors.
        # Use float32 dtype for begin/end mask markers before they are quantized to long.
        mask_width = torch.rand((batch_size, num_masks), device=input_spec.device, dtype=torch.float32) * width
        mask_width = mask_width.long()
        mask_start = torch.rand((batch_size, num_masks), device=input_spec.device, dtype=torch.float32)

        if axis == self.TIME_AXIS:
            # length can only be used for the time axis
            mask_start = mask_start * (length.unsqueeze(1) - mask_width)
        else:
            mask_start = mask_start * (axis_length - mask_width)

        mask_start = mask_start.long()
        mask_end = mask_start + mask_width

        # Create mask values using vectorized indexing
        indices = torch.arange(axis_length, device=input_spec.device)
        # Create a mask_tensor with all the indices.
        # The mask_tensor shape is (batch_size, num_masks, axis_length).
        mask_tensor = (indices >= mask_start.unsqueeze(-1)) & (indices < mask_end.unsqueeze(-1))

        # Reduce masks to one mask
        mask_tensor = mask_tensor.any(dim=1)

        # Create a final mask that aligns with the full tensor
        mask = torch.zeros_like(input_spec, dtype=torch.bool)
        if axis == self.TIME_AXIS:
            mask_ranges = mask_tensor[:, None, :]
        else:  # axis == self.FREQ_AXIS
            mask_ranges = mask_tensor[:, :, None]
        mask[:, :, :] = mask_ranges

        # Apply the mask value
        return input_spec.masked_fill(mask=mask, value=mask_value)


class SpecCutout(nn.Module, Typing):
    """
    Zeroes out(cuts) random rectangles in the spectrogram
    as described in (https://arxiv.org/abs/1708.04552).

    params:
    rect_masks - how many rectangular masks should be cut
    rect_freq - maximum size of cut rectangles along the frequency dimension
    rect_time - maximum size of cut rectangles along the time dimension
    """

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {"input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20, rng=None):
        super(SpecCutout, self).__init__()

        self._rng = random.Random() if rng is None else rng

        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    @typecheck()
    @torch.no_grad()
    def forward(self, input_spec):
        sh = input_spec.shape

        for idx in range(sh[0]):
            for i in range(self.rect_masks):
                rect_x = self._rng.randint(0, sh[1] - self.rect_freq)
                rect_y = self._rng.randint(0, sh[2] - self.rect_time)

                w_x = self._rng.randint(0, self.rect_freq)
                w_y = self._rng.randint(0, self.rect_time)

                input_spec[idx, rect_x : rect_x + w_x, rect_y : rect_y + w_y] = 0.0

        return input_spec

    @torch.no_grad()
    def forward_packed(self, input_spec: PackedEncoderActivations) -> PackedEncoderActivations:
        """Apply sequence-local rectangular masks to token-flat features."""
        if input_spec.total_tokens == 0 or self.rect_masks == 0:
            return input_spec
        batch_size = input_spec.batch_size
        data = input_spec.data
        freq_start = torch.zeros((batch_size, self.rect_masks), dtype=torch.long, device=data.device)
        freq_width = torch.zeros_like(freq_start)
        time_start = torch.zeros((batch_size, self.rect_masks), dtype=torch.long, device=data.device)
        time_width = torch.zeros_like(time_start)
        dense_time = input_spec.padded_length if input_spec.padded_length is not None else input_spec.max_seqlen
        for row in range(batch_size):
            for mask in range(self.rect_masks):
                freq_start[row, mask] = self._rng.randint(0, data.shape[1] - self.rect_freq)
                time_start[row, mask] = self._rng.randint(0, dense_time - self.rect_time)
                freq_width[row, mask] = self._rng.randint(0, self.rect_freq)
                time_width[row, mask] = self._rng.randint(0, self.rect_time)
        data = _apply_packed_range_masks(
            data,
            input_spec,
            freq_start=freq_start,
            freq_width=freq_width,
            time_start=time_start,
            time_width=time_width,
            mask_value=0.0,
            rectangular=True,
        )
        return input_spec.with_data(data)


def _apply_packed_axis_masks(data, packed, *, num_masks, width, axis, mask_value):
    if num_masks == 0 or packed.total_tokens == 0:
        return data
    batch_size = packed.batch_size
    axis_length = packed.max_seqlen if axis == SpecAugment.TIME_AXIS else data.shape[1]
    if axis == SpecAugment.TIME_AXIS and isinstance(width, float):
        width = torch.clamp(width * packed.lengths, max=axis_length).unsqueeze(1)
    mask_width = (torch.rand((batch_size, num_masks), device=data.device, dtype=torch.float32) * width).long()
    mask_start = torch.rand((batch_size, num_masks), device=data.device, dtype=torch.float32)
    if axis == SpecAugment.TIME_AXIS:
        mask_start = (mask_start * (packed.lengths.unsqueeze(1) - mask_width)).long()
        token_indices = torch.arange(packed.total_tokens, device=data.device)
        sequence_ids = torch.bucketize(token_indices, packed.cu_seqlens[1:], right=True)
        positions = (token_indices - packed.cu_seqlens[sequence_ids]).unsqueeze(1)
        mask = ((positions >= mask_start[sequence_ids]) & (positions < (mask_start + mask_width)[sequence_ids])).any(1)
        return data.masked_fill(mask.unsqueeze(1), mask_value)
    mask_start = (mask_start * (axis_length - mask_width)).long()
    bins = torch.arange(axis_length, device=data.device)
    mask = ((bins >= mask_start.unsqueeze(-1)) & (bins < (mask_start + mask_width).unsqueeze(-1))).any(1)
    token_indices = torch.arange(packed.total_tokens, device=data.device)
    sequence_ids = torch.bucketize(token_indices, packed.cu_seqlens[1:], right=True)
    return data.masked_fill(mask[sequence_ids], mask_value)


def _apply_packed_range_masks(
    data, packed, *, freq_start, freq_width, time_start, time_width, mask_value, rectangular=False
):
    if packed.total_tokens == 0:
        return data
    sequence_ids = torch.repeat_interleave(torch.arange(packed.batch_size, device=data.device), packed.lengths)
    positions = packed_encoder_position_ids(packed)
    bins = torch.arange(data.shape[1], device=data.device)
    time_mask = (positions[:, None] >= time_start[sequence_ids]) & (
        positions[:, None] < (time_start + time_width)[sequence_ids]
    )
    freq_mask = (bins >= freq_start.unsqueeze(-1)) & (bins < (freq_start + freq_width).unsqueeze(-1))
    freq_mask = freq_mask[sequence_ids].transpose(1, 2)
    if rectangular:
        mask = (time_mask.unsqueeze(1) & freq_mask).any(2)
    else:
        mask = time_mask.any(1).unsqueeze(1) | freq_mask.any(2)
    return data.masked_fill(mask, mask_value)
