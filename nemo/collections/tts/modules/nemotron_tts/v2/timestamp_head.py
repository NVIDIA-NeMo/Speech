# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output-only word-end supervision at the TTS backbone's audio-frame rate."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


NONEND, END = range(2)
TIMESTAMP_LABELS = ('NONEND', 'END')
IGNORE_INDEX = -100


class TimestampHead(nn.Module):
    """Predict one binary word-end logit from each audio hidden state.

    This projection only reads hidden states; its predictions never feed back into
    audio generation. ``compute_loss`` accepts a mask whose true entries are usable
    audio frames and speech EOS, after the caller excludes unreliable ordinary frames.
    """

    def __init__(
        self,
        hidden_size: int,
        projection_size: int = 128,
        pos_weight: float = 1.0,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        if hidden_size <= 0 or projection_size <= 0:
            raise ValueError('Timestamp head dimensions must be positive')
        self.ignore_index = ignore_index
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_size),
            nn.GELU(),
            nn.Linear(projection_size, 1),
        )
        weight = torch.as_tensor(pos_weight, dtype=torch.float32)
        if isinstance(pos_weight, bool) or weight.ndim != 0 or not torch.isfinite(weight) or weight <= 0:
            raise ValueError('pos_weight must be a finite positive scalar')
        self.register_buffer('pos_weight', weight, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``(..., frames)`` without modifying the input."""
        return self.projection(hidden_states).squeeze(-1)

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute masked FP32 BCEWithLogits with a differentiable all-masked zero.

        Targets are NONEND=0 or END=1. ``pos_weight`` scales positive examples;
        the mean denominator is the number of supervised frames, as in PyTorch BCE.
        """
        if logits.shape != labels.shape:
            raise ValueError('Timestamp logits and labels must have the same shape')
        selected = labels != self.ignore_index
        if valid_mask is not None:
            if valid_mask.shape != labels.shape or valid_mask.dtype != torch.bool:
                raise ValueError('valid_mask must be boolean and have the same shape as labels')
            selected = selected & valid_mask
        selected_logits = logits[selected].float()
        selected_labels = labels[selected].float()
        if selected_labels.numel() == 0:
            return selected_logits.sum()
        return F.binary_cross_entropy_with_logits(selected_logits, selected_labels, pos_weight=self.pos_weight.float())


@dataclass(frozen=True)
class TimestampFrameLabels:
    """Frame targets and diagnostics for a single aligned audio turn.

    ``collision_count`` counts bins containing multiple internal word ends and
    ``collision_word_count`` counts extra ends beyond the first in those bins.
    Shared bins remain valid END targets because this head predicts presence.
    ``valid_mask`` excludes padding or the whole turn when any word has invalid
    timing. Ignored labels always contain ``IGNORE_INDEX``.
    """

    labels: torch.Tensor
    valid_mask: torch.Tensor
    collision_count: int
    invalid_word_count: int
    collision_word_count: int = 0


def word_intervals_to_frame_labels(
    words: Sequence[Mapping],
    num_frames: int,
    frame_seconds: float = 0.08,
    audio_duration: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> TimestampFrameLabels:
    """Label internal word ends; the caller separately supervises speech EOS.

    Each nonfinal word's end is the following word's onset. The first word starts
    at zero, including leading silence. This helper excludes the last word's end:
    the caller maps that alignment timestamp to check the decoder's EOS slot and
    supervises terminal END separately. A single-word turn has only NONEND targets
    here and receives its terminal END separately.

    An internal end rounds to the nearest frame boundary (ties round forward)
    and labels its preceding acoustic frame: an end at 0.16 seconds is END in
    frame 1 on the 80 ms grid. Consumers recover the event time as
    ``(frame_index + 1) * frame_seconds``. A positive end rounding to zero uses
    frame 0. Multiple ends in one bin stay one valid END with collision counts.

    Ends beyond the crop or at audio EOS create no internal event. A true internal end
    exactly on the crop edge closes the last retained frame. Cropping a word
    does not manufacture an end. ``audio_duration`` excludes padded bins; the last
    partial audio bin remains valid. Invalid or nonpositive word intervals mask
    the whole turn rather than teaching false NONEND labels.
    """
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 0:
        raise ValueError('num_frames must be a nonnegative integer')
    if not math.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError('frame_seconds must be finite and positive')
    if audio_duration is None:
        audio_duration = num_frames * frame_seconds
    if not math.isfinite(audio_duration) or audio_duration < 0:
        raise ValueError('audio_duration must be finite and nonnegative')

    num_valid = min(num_frames, math.ceil(_frame_coordinate(audio_duration, frame_seconds)))
    crop_duration = min(audio_duration, num_frames * frame_seconds)
    # Word metadata is already on the host. Avoid launching CUDA kernels for each word.
    labels = torch.full((num_frames,), IGNORE_INDEX, dtype=torch.long)
    valid_mask = torch.arange(num_frames) < num_valid
    labels[valid_mask] = NONEND
    occupancy = torch.zeros(num_frames, dtype=torch.long)
    invalid_word_count = 0
    for index, word in enumerate(words):
        try:
            start, end = float(word['start']), float(word['end'])
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_word_count += 1
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            invalid_word_count += 1
            continue
        if index == len(words) - 1 or not num_valid or end <= 0 or end > crop_duration or end >= audio_duration:
            continue
        boundary = math.floor(_frame_coordinate(end + frame_seconds / 2, frame_seconds))
        frame = min(max(boundary - 1, 0), num_valid - 1)
        labels[frame] = END
        occupancy[frame] += 1

    collisions = occupancy > 1
    collision_count = int(collisions.sum().item())
    collision_word_count = int((occupancy - 1).clamp_min(0).sum().item())
    if invalid_word_count:
        valid_mask[:] = False
    labels[~valid_mask] = IGNORE_INDEX
    if device is not None:
        labels, valid_mask = labels.to(device=device), valid_mask.to(device=device)
    return TimestampFrameLabels(labels, valid_mask, collision_count, invalid_word_count, collision_word_count)


def reconcile_timestamp_eos(labels, valid_mask, terminal_mask, eos_mask, audio_frame_mask, eligible_mask):
    """Supervise eligible speech EOS and count differences from aligned endpoints.

    Ordinary frame labels require valid teacher supervision and codec audio.
    Eligible speech EOS always receives END, including missing aligned endpoints.
    Alignment endpoints only affect diagnostics here; they never create a new
    event away from speech EOS. Inputs are left unchanged and counts stay on device.
    """
    masks = (valid_mask, terminal_mask, eos_mask, audio_frame_mask, eligible_mask)
    if any(mask.dtype != torch.bool or mask.shape != labels.shape or mask.device != labels.device for mask in masks):
        raise ValueError('Timestamp masks must be boolean, match labels.shape, and share its device')
    eos = eos_mask & eligible_mask
    mask = (valid_mask & audio_frame_mask & eligible_mask) | eos
    targets = labels.masked_fill(~mask, IGNORE_INDEX).masked_fill(eos, END)
    return (
        targets,
        mask,
        {
            'eos_alignment_total_count': eos.sum(),
            'eos_alignment_missing_count': (eos & ~terminal_mask).sum(),
            'eos_alignment_extra_count': (terminal_mask & eligible_mask & ~eos_mask).sum(),
        },
    )


def _frame_coordinate(seconds: float, frame_seconds: float) -> float:
    """Avoid spurious shared bins from floating-point roundoff at exact boundaries."""
    coordinate = seconds / frame_seconds
    nearest = round(coordinate)
    if abs(coordinate - nearest) <= 4 * math.ulp(coordinate):
        return float(nearest)
    return coordinate
