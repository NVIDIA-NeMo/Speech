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
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from nemo.collections.asr.parts.packed_sequence import (
    PackedEncoderActivations,
    packed_encoder_position_ids,
    packed_sequence_ids,
)
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import LengthsType, NeuralType, SpectrogramType

_MASK_COUNT_KEYS = ("freq_masks", "time_masks", "min_freq_masks", "min_time_masks")


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
    time_masks_per_frames, min_freq_masks, min_time_masks, mask_schedule - the length-following mask policy
        and its schedule, off by default; see `SpectrogramAugmentation` for the description.
        The schedule reads the training step that `set_num_updates` stores.
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
        *,
        time_masks_per_frames: int | None = None,
        min_freq_masks: int = 0,
        min_time_masks: int = 0,
        mask_schedule: Sequence[Mapping[str, int]] | None = None,
    ):
        super().__init__()

        self._rng = random.Random() if rng is None else rng

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.short_recording_frame_duration = None
        self.short_recording_max_duration = None
        self.short_recording_max_mask_fraction = None

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

        self.time_masks_per_frames = time_masks_per_frames
        stages = self._length_following_stages(min_freq_masks, min_time_masks, mask_schedule)
        self._max_freq_masks = max((stage["freq_masks"] for stage in stages), default=0)
        self._max_time_masks = max((stage["time_masks"] for stage in stages), default=0)
        # Row 0 holds the constructor's counts, row i the counts from the i-th stage on.
        stage_counts = torch.tensor([[stage[key] for key in _MASK_COUNT_KEYS] for stage in stages], dtype=torch.long)
        stage_steps = torch.tensor([stage["step"] for stage in stages[1:]], dtype=torch.long)
        self.register_buffer("_stage_counts", stage_counts if stages else None, persistent=False)
        self.register_buffer("_stage_steps", stage_steps if stages else None, persistent=False)
        # The training step of the schedule, a device scalar so that a captured graph reads its current value.
        schedule_step = torch.zeros((), dtype=torch.long) if mask_schedule else None
        self.register_buffer("schedule_step", schedule_step, persistent=False)
        self._schedule_step_is_set = False

    def set_num_updates(self, num_updates: int | torch.Tensor) -> None:
        """Set the training step that `mask_schedule` selects its stage by.

        Args:
            num_updates: number of optimizer steps done so far, a Python integer or a scalar tensor.
        """
        if self.schedule_step is not None:
            self.schedule_step.fill_(num_updates)
            self._schedule_step_is_set = True

    def _length_following_stages(
        self, min_freq_masks: int, min_time_masks: int, mask_schedule: Sequence[Mapping[str, int]] | None
    ) -> list[dict[str, int]]:
        """Validate the options of the length-following policy and return the mask counts of every stage.

        The list is empty without the policy.
        """
        base = {
            "freq_masks": self.freq_masks,
            "time_masks": self.time_masks,
            "min_freq_masks": min_freq_masks,
            "min_time_masks": min_time_masks,
        }
        if self.time_masks_per_frames is None:
            if min_freq_masks != 0 or min_time_masks != 0 or mask_schedule:
                raise ValueError(
                    "min_freq_masks, min_time_masks and mask_schedule require time_masks_per_frames, "
                    f"got {min_freq_masks=}, {min_time_masks=}, {mask_schedule=}"
                )
            return []
        if not _is_int(self.time_masks_per_frames) or self.time_masks_per_frames < 1:
            raise ValueError(f"time_masks_per_frames must be a positive integer, got {self.time_masks_per_frames!r}")
        if not self.use_vectorized_code:
            raise ValueError("time_masks_per_frames requires the vectorized implementation, not the legacy loop")
        if mask_schedule is None:
            mask_schedule = ()
        if isinstance(mask_schedule, (str, Mapping)) or not isinstance(mask_schedule, Sequence):
            raise ValueError(f"mask_schedule must be a list of stages, got {mask_schedule!r}")
        stages = [base]
        for stage in mask_schedule:
            if not isinstance(stage, Mapping) or "step" not in stage or not set(stage) <= {"step", *_MASK_COUNT_KEYS}:
                raise ValueError(f"a mask_schedule stage takes 'step' and any of {_MASK_COUNT_KEYS}, got {stage!r}")
            # A count that a stage leaves out keeps its value of the stage before.
            stages.append({**stages[-1], **stage})
        steps = [stage["step"] for stage in stages[1:]]
        if not all(_is_int(step) for step in steps) or steps != sorted(steps):
            raise ValueError(f"the steps of mask_schedule must be non-decreasing integers, got {steps!r}")
        for stage in stages:
            if not all(_is_int(stage[key]) for key in _MASK_COUNT_KEYS):
                raise ValueError(f"mask counts must be integers, got {stage!r}")
            if not (
                0 <= stage["min_freq_masks"] <= stage["freq_masks"]
                and 0 <= stage["min_time_masks"] <= stage["time_masks"]
            ):
                raise ValueError(f"mask counts must satisfy 0 <= min_*_masks <= *_masks, got {stage!r}")
        for name, width, masks in (
            ("freq_width", self.freq_width, "freq_masks"),
            ("time_width", self.time_width, "time_masks"),
        ):
            if any(stage[masks] > 0 for stage in stages) and (not _is_int(width) or width < 1):
                raise ValueError(f"{name} must be a positive integer with time_masks_per_frames, got {width!r}")
        return stages

    def configure_short_recording_time_mask_cap(
        self,
        *,
        frame_duration_seconds: float,
        max_duration_seconds: float = 1.0,
        max_mask_fraction: float = 0.1,
    ) -> None:
        """Cap total time masking for recordings shorter than a duration threshold.

        Short rows use at most one non-empty time mask whose width cannot exceed
        ``floor(length * max_mask_fraction)``. Long rows retain the original mask
        count, width, and random-number path.
        """
        if frame_duration_seconds <= 0:
            raise ValueError("frame_duration_seconds must be positive.")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive.")
        if not 0 <= max_mask_fraction <= 1:
            raise ValueError("max_mask_fraction must be in [0, 1].")
        self.short_recording_frame_duration = float(frame_duration_seconds)
        self.short_recording_max_duration = float(max_duration_seconds)
        self.short_recording_max_mask_fraction = float(max_mask_fraction)

    def _time_mask_width_limits(self, lengths: torch.Tensor, width: int | float, num_masks: int):
        """Return per-mask width limits when the short-recording cap is active."""
        if self.short_recording_frame_duration is None or num_masks == 0:
            return width
        short_rows = (
            lengths.to(torch.float32) * self.short_recording_frame_duration < self.short_recording_max_duration
        )
        if isinstance(width, torch.Tensor):
            ordinary_limit = width.to(device=lengths.device, dtype=torch.float32)
            if ordinary_limit.ndim == 1:
                ordinary_limit = ordinary_limit.unsqueeze(1)
        elif isinstance(width, float):
            ordinary_limit = torch.clamp(width * lengths, min=0).to(torch.float32).unsqueeze(1)
        else:
            ordinary_limit = torch.full((lengths.numel(), 1), float(width), device=lengths.device, dtype=torch.float32)
        ordinary_limits = ordinary_limit.expand(-1, num_masks)
        short_limits = torch.zeros_like(ordinary_limits)
        total_budget = torch.floor(lengths.to(torch.float32) * self.short_recording_max_mask_fraction)
        short_limits[:, 0] = torch.minimum(ordinary_limit[:, 0], total_budget)
        return torch.where(short_rows.unsqueeze(1), short_limits, ordinary_limits)

    @typecheck()
    @torch.no_grad()
    def forward(self, input_spec, length):
        if self.time_masks_per_frames is not None:
            return self._forward_length_following(input_spec, length)
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
        if self.time_masks_per_frames is not None:
            data = input_spec.data
            # The padded width, where known, gives the masks of the padded forward for the same seed.
            max_length = input_spec.max_seqlen if input_spec.padded_length is None else input_spec.padded_length
            time_ranges, freq_mask = self._length_following_masks(
                input_spec.lengths, max_length=max_length, num_freq_bins=data.shape[1]
            )
            if freq_mask is not None:
                data = data.masked_fill(freq_mask[packed_sequence_ids(input_spec)], self.mask_value)
            if time_ranges is not None:
                time_mask = _covered(*time_ranges, offsets=input_spec.cu_seqlens[:-1].long(), total=data.shape[0])
                data = data.masked_fill(time_mask[:, None], self.mask_value)
            return input_spec.with_data(data)
        time_width = self._time_mask_width_limits(input_spec.lengths, self.time_width, self.time_masks)
        data = _apply_packed_axis_masks(
            input_spec.data,
            input_spec,
            num_masks=self.time_masks,
            width=time_width,
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

    def _forward_length_following(self, input_spec: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        batch_size, num_freq_bins, max_length = input_spec.shape
        length = length.to(device=input_spec.device, dtype=torch.long).clamp(min=0, max=max_length)
        time_ranges, freq_mask = self._length_following_masks(
            length, max_length=max_length, num_freq_bins=num_freq_bins
        )
        if freq_mask is not None:
            input_spec = input_spec.masked_fill(freq_mask[:, :, None], self.mask_value)
        if time_ranges is not None:
            # One spare column per row takes the end marker of a mask that reaches the last frame.
            offsets = torch.arange(batch_size, device=input_spec.device) * (max_length + 1)
            time_mask = _covered(*time_ranges, offsets=offsets, total=batch_size * (max_length + 1))
            time_mask = time_mask.view(batch_size, max_length + 1)[:, :max_length]
            input_spec = input_spec.masked_fill(time_mask[:, None, :], self.mask_value)
        return input_spec

    def _length_following_masks(
        self, lengths: torch.Tensor, *, max_length: int, num_freq_bins: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        """Draw the masks of the length-following policy.

        Returns the time masks as ``(starts, ends, valid)``, each of shape ``(B, slots)``,
        and the frequency mask of shape ``(B, num_freq_bins)``; an axis without masks gives None.
        No tensor value is read on the host,
        so every schedule stage runs the same sequence of device operations.
        """
        device, batch_size = lengths.device, lengths.numel()
        counts = self._stage_counts[0]
        if self.schedule_step is not None:
            if self.training and not self._schedule_step_is_set:
                raise RuntimeError(
                    "mask_schedule needs the training step: call set_num_updates(step) before every training step "
                    "(the NeMo ASR models do this with the trainer's global step)"
                )
            # index_select: indexing with a 0-dim tensor would read the stage on the host
            stage = (self.schedule_step >= self._stage_steps).sum()
            counts = self._stage_counts.index_select(0, stage.view(1))[0]
        max_freq_masks, max_time_masks, min_freq_masks, min_time_masks = counts.to(device).unbind(0)

        time_ranges = None
        slots = min(max_length, max(max_length // self.time_masks_per_frames, 2) * self._max_time_masks)
        if slots > 0:
            max_masks = torch.clamp(lengths // self.time_masks_per_frames, min=2) * max_time_masks
            max_masks = torch.minimum(max_masks, lengths)
            num_masks = _draw_counts(torch.minimum(min_time_masks, lengths), max_masks)
            # Distinct start positions inside every sequence: the top scores among its frames.
            scores = torch.rand(batch_size, max_length, device=device)
            scores.masked_fill_(torch.arange(max_length, device=device) >= lengths[:, None], -1.0)
            starts = scores.topk(slots, dim=1).indices
            widths = torch.randint(1, self.time_width + 1, (batch_size, slots), device=device)
            if self.short_recording_frame_duration is not None:
                seconds = lengths.to(torch.float32) * self.short_recording_frame_duration
                short_rows = seconds < self.short_recording_max_duration
                num_masks = torch.where(short_rows, num_masks.clamp(max=1), num_masks)
                budget = torch.floor(lengths.to(torch.float32) * self.short_recording_max_mask_fraction).long()
                widths = torch.where(short_rows[:, None], torch.minimum(widths, budget[:, None]), widths)
            ends = torch.minimum(starts + widths, lengths[:, None])
            valid = torch.arange(slots, device=device) < num_masks[:, None]
            time_ranges = (starts, ends, valid)

        freq_mask = None
        slots = min(num_freq_bins, self._max_freq_masks)
        if slots > 0:
            num_masks = _draw_counts(min_freq_masks.expand(batch_size), max_freq_masks.expand(batch_size))
            starts = torch.rand(batch_size, num_freq_bins, device=device).topk(slots, dim=1).indices
            widths = torch.randint(1, self.freq_width + 1, (batch_size, slots), device=device)
            valid = torch.arange(slots, device=device) < num_masks[:, None]
            bins = torch.arange(num_freq_bins, device=device)
            freq_mask = (bins >= starts[:, :, None]) & (bins < (starts + widths)[:, :, None]) & valid[:, :, None]
            freq_mask = freq_mask.any(dim=1)
        return time_ranges, freq_mask

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
            time_width_limits = self._time_mask_width_limits(
                torch.as_tensor([lengths_cpu[idx]]), time_max_width, self.time_masks
            )
            if isinstance(time_width_limits, torch.Tensor):
                time_width_limits = time_width_limits[0].tolist()
            else:
                time_width_limits = [time_width_limits] * self.time_masks

            # Draw with the historical bounds even for capped short rows.
            # randint uses rejection sampling, so changing its bounds would shift
            # the RNG state for later rows and subsequent batches.
            time_start_upper_bound = max(1, lengths_cpu[idx] - time_max_width)
            for mask_idx in range(self.time_masks):
                start = self._rng.randint(0, time_start_upper_bound)
                width = self._rng.randint(0, time_max_width)
                width = min(width, int(time_width_limits[mask_idx]))
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

        if axis == self.TIME_AXIS:
            width = self._time_mask_width_limits(length, width, num_masks)

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


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _draw_counts(min_counts: torch.Tensor, max_counts: torch.Tensor) -> torch.Tensor:
    """Draw one integer per row, uniformly from ``[min_counts, max_counts]``."""
    span = (max_counts - min_counts + 1).to(torch.float32)
    counts = min_counts + (torch.rand(max_counts.shape, device=max_counts.device) * span).long()
    return torch.minimum(counts, max_counts)


def _covered(
    starts: torch.Tensor, ends: torch.Tensor, valid: torch.Tensor, *, offsets: torch.Tensor, total: int
) -> torch.Tensor:
    """Mark the positions of a flat axis of ``total`` entries that some range ``[start, end)`` covers.

    Row ``i`` owns the positions from ``offsets[i]`` on;
    ``starts`` and ``ends``, shape ``(B, slots)``, count from there.
    A difference array keeps the memory at the size of the axis, whatever the number of ranges;
    a range with ``valid`` unset adds nothing.
    """
    valid = valid.to(torch.int32).flatten()
    coverage = torch.zeros(total + 1, dtype=torch.int32, device=starts.device)
    coverage.index_add_(0, (offsets[:, None] + starts).clamp(max=total).flatten(), valid)
    coverage.index_add_(0, (offsets[:, None] + ends).clamp(max=total).flatten(), -valid)
    return coverage.cumsum(0)[:total] > 0


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
