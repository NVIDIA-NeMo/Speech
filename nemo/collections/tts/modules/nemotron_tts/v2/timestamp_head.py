# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output-only word-boundary supervision at the TTS backbone's audio-frame rate."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


OUTSIDE, START, CONTINUE, END, SINGLE = range(5)
TIMESTAMP_LABELS = ('OUTSIDE', 'START', 'CONTINUE', 'END', 'SINGLE')
IGNORE_INDEX = -100


class TimestampHead(nn.Module):
    """Predict five independent word-boundary classes from each audio hidden state.

    This projection only reads hidden states; its predictions never feed back into
    audio generation. ``compute_loss`` accepts a mask whose true entries are usable
    audio frames, after the caller excludes unreliable teacher rows and control tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        projection_size: int = 128,
        class_weights: Optional[Sequence[float]] = None,
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
            nn.Linear(projection_size, len(TIMESTAMP_LABELS)),
        )
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        if weights is not None:
            if weights.shape != (len(TIMESTAMP_LABELS),) or not torch.isfinite(weights).all():
                raise ValueError('class_weights must contain five finite values')
            if (weights < 0).any() or not (weights > 0).any():
                raise ValueError('class_weights must be nonnegative with at least one positive value')
        self.register_buffer('class_weights', weights, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``(..., frames, 5)`` without modifying the input."""
        return self.projection(hidden_states)

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute masked FP32 cross entropy, including a differentiable empty zero.

        Weights use the same weighted-mean normalization as PyTorch cross entropy.
        A selection containing only zero-weight classes also returns zero.
        """
        if logits.shape[:-1] != labels.shape or logits.shape[-1] != len(TIMESTAMP_LABELS):
            raise ValueError('Timestamp logits must have shape labels.shape + (5,)')
        if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('Timestamp labels must have an integer dtype')
        selected = labels != self.ignore_index
        if valid_mask is not None:
            if valid_mask.shape != labels.shape or valid_mask.dtype != torch.bool:
                raise ValueError('valid_mask must be boolean and have the same shape as labels')
            selected = selected & valid_mask
        selected_logits = logits[selected].float()
        selected_labels = labels[selected].long()
        if selected_labels.numel() == 0:
            return selected_logits.sum()
        weights = None if self.class_weights is None else self.class_weights.float()
        losses = F.cross_entropy(selected_logits, selected_labels, weight=weights, reduction='none')
        denominator = losses.new_tensor(selected_labels.numel())
        if weights is not None:
            denominator = weights[selected_labels].sum()
        return losses.sum() / denominator.clamp_min(torch.finfo(losses.dtype).tiny)


@dataclass(frozen=True)
class TimestampFrameLabels:
    """Frame targets and diagnostics for a single aligned audio turn.

    ``collision_count`` counts distinct bins occupied by more than one word.
    ``collapsed_word_count`` counts positive words whose quantized duration is zero.
    ``valid_mask`` is false on collisions, cropped one-bin words, padding, or
    every bin when any word has invalid timing. Ignored labels always contain
    ``IGNORE_INDEX``.
    """

    labels: torch.Tensor
    valid_mask: torch.Tensor
    collision_count: int
    invalid_word_count: int
    collapsed_word_count: int = 0


def word_intervals_to_frame_labels(
    words: Sequence[Mapping],
    num_frames: int,
    frame_seconds: float = 0.08,
    audio_duration: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> TimestampFrameLabels:
    """Quantize reference word boundaries and label their disjoint frame spans.

    Each timestamp rounds to its nearest frame boundary (ties round forward), so
    adjacent nonempty spans with a shared boundary never collide. At 80 ms per
    frame each boundary moves by at most 40 ms. Words occupy their rounded spans;
    one-bin words are SINGLE and gaps remain OUTSIDE. Genuine overlapping spans
    are ambiguous and ignored. A positive word rounded to zero bins contributes
    SINGLE in its original midpoint bin if uncontested, without shifting its
    neighbors. A SINGLE label requires the complete word to lie inside the crop.

    Cropping retains the original quantized boundary classes without inventing a
    START or END at the crop.

    ``audio_duration`` excludes padded bins (the last partial bin remains valid).
    Missing, nonfinite, or nonpositive word timings invalidate the whole turn,
    since replacing their unknown locations with OUTSIDE would teach false labels.
    Negative starts and intervals beyond the crop are allowed and clipped.
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
    labels[valid_mask] = OUTSIDE
    occupancy = torch.zeros(num_frames, dtype=torch.long)
    cropped_word_mask = torch.zeros(num_frames, dtype=torch.bool)
    invalid_word_count = 0
    collapsed_word_count = 0
    for word in words:
        try:
            start, end = float(word['start']), float(word['end'])
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_word_count += 1
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            invalid_word_count += 1
            continue
        if not num_valid or end <= 0 or start >= crop_duration:
            continue
        complete_word = start >= 0 and end <= crop_duration
        first = math.floor(_frame_coordinate(start + frame_seconds / 2, frame_seconds))
        stop = math.floor(_frame_coordinate(end + frame_seconds / 2, frame_seconds))
        if first == stop:
            collapsed_word_count += 1
            midpoint_bin = math.floor((start + (end - start) / 2) / frame_seconds)
            midpoint_bin = min(max(midpoint_bin, 0), num_valid - 1)
            labels[midpoint_bin] = SINGLE
            occupancy[midpoint_bin] += 1
            if not complete_word:
                cropped_word_mask[midpoint_bin] = True
            continue
        last = stop - 1
        left, right = max(first, 0), min(last + 1, num_valid)
        if left >= right:
            continue
        labels[left:right] = CONTINUE
        if first == last:
            labels[left] = SINGLE
            if not complete_word:
                cropped_word_mask[left] = True
        else:
            if left == first:
                labels[left] = START
            if right - 1 == last:
                labels[right - 1] = END
        occupancy[left:right] += 1

    collisions = occupancy > 1
    collision_count = int(collisions.sum().item())
    valid_mask &= ~(collisions | cropped_word_mask)
    if invalid_word_count:
        valid_mask[:] = False
    labels[~valid_mask] = IGNORE_INDEX
    if device is not None:
        labels, valid_mask = labels.to(device=device), valid_mask.to(device=device)
    return TimestampFrameLabels(labels, valid_mask, collision_count, invalid_word_count, collapsed_word_count)


def _frame_coordinate(seconds: float, frame_seconds: float) -> float:
    """Avoid spurious shared bins from floating-point roundoff at exact boundaries."""
    coordinate = seconds / frame_seconds
    nearest = round(coordinate)
    if abs(coordinate - nearest) <= 4 * math.ulp(coordinate):
        return float(nearest)
    return coordinate
