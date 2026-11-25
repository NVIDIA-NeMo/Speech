# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output-only word-end and distilled CTC supervision for audio hidden states."""

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
        local_count_loss: Optional[Mapping] = None,
        cumulative_count_loss: Optional[Mapping] = None,
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
        count_config = {} if local_count_loss is None else local_count_loss
        if not isinstance(count_config, Mapping):
            raise ValueError('local_count_loss must be a mapping')
        self.local_count_loss_enabled = count_config.get('enabled', False)
        if not isinstance(self.local_count_loss_enabled, bool):
            raise ValueError('local_count_loss.enabled must be boolean')
        count_weight = count_config.get('weight', 2.0)
        if isinstance(count_weight, bool) or not isinstance(count_weight, (int, float)):
            raise ValueError('local_count_loss.weight must be finite and nonnegative')
        self.local_count_loss_weight = float(count_weight)
        if not math.isfinite(self.local_count_loss_weight) or self.local_count_loss_weight < 0:
            raise ValueError('local_count_loss.weight must be finite and nonnegative')
        self.local_count_window_size = count_config.get('window_size', 5)
        if (
            isinstance(self.local_count_window_size, bool)
            or not isinstance(self.local_count_window_size, int)
            or self.local_count_window_size < 2
        ):
            raise ValueError('local_count_loss.window_size must be an integer of at least two frames')
        cumulative_config = {} if cumulative_count_loss is None else cumulative_count_loss
        if not isinstance(cumulative_config, Mapping):
            raise ValueError('cumulative_count_loss must be a mapping')
        self.cumulative_count_loss_enabled = cumulative_config.get('enabled', False)
        if not isinstance(self.cumulative_count_loss_enabled, bool):
            raise ValueError('cumulative_count_loss.enabled must be boolean')
        cumulative_weight = cumulative_config.get('weight', 32.0)
        if isinstance(cumulative_weight, bool) or not isinstance(cumulative_weight, (int, float)):
            raise ValueError('cumulative_count_loss.weight must be finite and nonnegative')
        self.cumulative_count_loss_weight = float(cumulative_weight)
        if not math.isfinite(self.cumulative_count_loss_weight) or self.cumulative_count_loss_weight < 0:
            raise ValueError('cumulative_count_loss.weight must be finite and nonnegative')
        cumulative_delta = cumulative_config.get('delta', 1.0)
        if isinstance(cumulative_delta, bool) or not isinstance(cumulative_delta, (int, float)):
            raise ValueError('cumulative_count_loss.delta must be finite and positive')
        self.cumulative_count_delta = float(cumulative_delta)
        if not math.isfinite(self.cumulative_count_delta) or self.cumulative_count_delta <= 0:
            raise ValueError('cumulative_count_loss.delta must be finite and positive')

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
        selected = self._supervision_mask(logits, labels, valid_mask)
        selected_logits = logits[selected].float()
        selected_labels = labels[selected].float()
        if selected_labels.numel() == 0:
            return selected_logits.sum()
        return F.binary_cross_entropy_with_logits(selected_logits, selected_labels, pos_weight=self.pos_weight.float())

    def compute_local_count_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        eos_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Match expected and target END counts in fully supervised sliding windows.

        Return the unweighted FP32 mean squared count error. Invalid frames break
        windows; speech EOS can close a window but cannot occur inside it. Thus
        neighboring turns cannot cancel each other's count errors. Targets count
        binary events, including genuine adjacent ENDs and coalesced collisions.
        Sequences without a complete usable window receive a differentiable zero.
        """
        selected = self._supervision_mask(logits, labels, valid_mask)
        if logits.ndim == 0:
            raise ValueError('Local count loss requires a frame dimension')
        if eos_mask is not None and (eos_mask.shape != labels.shape or eos_mask.dtype != torch.bool):
            raise ValueError('eos_mask must be boolean and have the same shape as labels')
        safe_logits = logits.float().masked_fill(~selected, 0)
        window = self.local_count_window_size
        if logits.size(-1) < window:
            return safe_logits.sum() * 0
        safe_labels = labels.masked_fill(~selected, 0).float()
        error = (safe_logits.sigmoid() - safe_labels).masked_fill(~selected, 0)
        windows = selected.unfold(-1, window, 1).all(dim=-1)
        if eos_mask is not None:
            windows = windows & ~eos_mask.unfold(-1, window, 1)[..., :-1].any(dim=-1)
        count_error = error.unfold(-1, window, 1).sum(dim=-1).masked_fill(~windows, 0)
        return count_error.square().sum() / windows.sum().clamp_min(1)

    def compute_cumulative_count_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        eos_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Match the expected and target cumulative END counts with FP32 Huber loss.

        Each valid region starts at zero, includes its speech EOS, and resets
        after EOS or invalid frames. Divide each frame's Huber contribution by
        its valid region length before averaging over supervised frames, so long
        turns do not amplify the cumulative gradients. Counts remain unnormalized.
        Binary targets preserve adjacent ENDs and coalesced collisions. Empty
        supervision returns a differentiable zero.
        """
        selected = self._supervision_mask(logits, labels, valid_mask)
        if logits.ndim == 0:
            raise ValueError('Cumulative count loss requires a frame dimension')
        if eos_mask is not None and (eos_mask.shape != labels.shape or eos_mask.dtype != torch.bool):
            raise ValueError('eos_mask must be boolean and have the same shape as labels')
        safe_logits = logits.float().masked_fill(~selected, 0)
        if logits.size(-1) == 0:
            return safe_logits.sum() * 0
        error = (safe_logits.sigmoid() - labels.masked_fill(~selected, 0).float()).masked_fill(~selected, 0)
        prefix = error.cumsum(dim=-1)
        resets = ~selected
        if eos_mask is not None:
            resets[..., 1:] |= eos_mask[..., :-1]
        positions = torch.arange(logits.size(-1), device=logits.device)
        offsets = torch.where(resets, positions, 0).cummax(dim=-1).values
        previous = F.pad(prefix[..., :-1], (1, 0))
        trajectory_error = prefix - previous.gather(-1, offsets)
        losses = F.huber_loss(
            trajectory_error, torch.zeros_like(trajectory_error), reduction='none', delta=self.cumulative_count_delta
        )
        lengths = torch.zeros_like(prefix).scatter_add(-1, offsets, selected.float()).gather(-1, offsets)
        return (losses / lengths.clamp_min(1)).masked_fill(~selected, 0).sum() / selected.sum().clamp_min(1)

    def _supervision_mask(self, logits, labels, valid_mask):
        if logits.shape != labels.shape:
            raise ValueError('Timestamp logits and labels must have the same shape')
        selected = labels != self.ignore_index
        if valid_mask is not None:
            if valid_mask.shape != labels.shape or valid_mask.dtype != torch.bool:
                raise ValueError('valid_mask must be boolean and have the same shape as labels')
            selected = selected & valid_mask
        return selected


class CTCTimestampHead(nn.Module):
    """Predict two ordered CTC distributions per audio hidden state.

    ``compute_loss`` operates on one independent turn, excluding audio EOS. It
    checkpoints bounded projection chunks and keeps only transcript classes,
    blank, and an OTHER class for CTC. OTHER collects all excluded probability
    mass, preserving both the full-vocabulary CTC value and gradients. Teacher
    targets contain explicit token/blank probabilities plus their residual tail;
    distillation is the categorical KL of that coarsened distribution.
    """

    subframes = 2

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        blank_id: int,
        projection_size: int = 256,
        chunk_size: int = 128,
        ctc_weight: float = 1.0,
        distillation_weight: float = 1.0,
    ):
        super().__init__()
        for name, value in (
            ('hidden_size', hidden_size),
            ('vocab_size', vocab_size),
            ('projection_size', projection_size),
            ('chunk_size', chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(blank_id, bool) or not isinstance(blank_id, int) or not 0 <= blank_id < vocab_size:
            raise ValueError('blank_id must index the CTC vocabulary')
        for name, value in (('ctc_weight', ctc_weight), ('distillation_weight', distillation_weight)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        self.vocab_size, self.blank_id = vocab_size, blank_id
        self.chunk_size = chunk_size
        self.ctc_weight, self.distillation_weight = float(ctc_weight), float(distillation_weight)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_size),
            nn.GELU(),
            nn.Linear(projection_size, self.subframes * vocab_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return ``(..., audio_frames, 2, vocab_size)`` logits without future frames."""
        logits = self.projection(hidden_states.to(dtype=self.projection[1].weight.dtype))
        return logits.unflatten(-1, (self.subframes, self.vocab_size))

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        teacher: Optional[Mapping] = None,
        distillation_mask: Optional[torch.Tensor] = None,
        input_length: Optional[int] = None,
        input_start: int = 0,
        return_predictions: bool = False,
        word_intervals: Optional[Sequence[Mapping] | torch.Tensor] = None,
    ) -> dict:
        """Return weighted loss, raw FP32 CTC/KL, and feasibility diagnostics.

        Sparse teacher arrays ``indices``/``probabilities`` have shape
        ``(subframes, kept_classes)``; ``tail_probabilities`` holds the omitted
        mass. Their time grid must already match this turn's student subframes.
        CTC uses ``[input_start:input_length]`` so partially occupied boundary
        frames can be excluded. Distillation retains its explicit validity mask
        over ``[0:input_length]``. Infeasible CTC targets are reported and retain
        distillation supervision.
        ``return_predictions`` also returns detached full-vocabulary argmax IDs
        and their probabilities on ``[0:input_length]`` from the same projection
        chunks, without retaining full-vocabulary logits.

        ``teacher=None`` supports transcript-only CTC when distillation is off.
        Optional word intervals specify exclusive ``token_start``/``token_end``
        transcript ranges and ``start_frame``/``end_frame`` CTC-subframe ranges.
        Their additional CTC loss uses the full vocabulary normalization and is
        returned separately from ``loss``. Overlapping time windows are allowed.
        """
        from torch.utils.checkpoint import checkpoint

        if hidden_states.ndim != 2 or target_ids.ndim != 1:
            raise ValueError('CTC loss expects hidden_states (frames, hidden) and target_ids (tokens,)')
        total = hidden_states.size(0) * self.subframes
        length = total if input_length is None else input_length
        if isinstance(length, bool) or not isinstance(length, int) or not 0 <= length <= total:
            raise ValueError('input_length must be an integer within the available CTC subframes')
        if isinstance(input_start, bool) or not isinstance(input_start, int) or not 0 <= input_start <= length:
            raise ValueError('input_start must be an integer between zero and input_length')
        ctc_length = length - input_start
        targets = target_ids.detach().to(device='cpu', dtype=torch.long)
        if targets.numel() and ((targets < 0).any() or (targets >= self.vocab_size).any()):
            raise ValueError('CTC targets must index the CTC vocabulary')
        if targets.eq(self.blank_id).any():
            raise ValueError('CTC transcript targets cannot contain blank')
        if teacher is None:
            if self.distillation_weight:
                raise ValueError('CTC teacher probabilities are required when distillation_weight is nonzero')
            indices = torch.empty((length, 0), dtype=torch.long)
            probabilities = torch.empty((length, 0), dtype=torch.float32)
            tail = torch.zeros(length, dtype=torch.float32)
            if distillation_mask is not None:
                raise ValueError('distillation_mask requires CTC teacher probabilities')
            distillation_mask = torch.zeros(length, dtype=torch.bool)
        else:
            indices, probabilities, tail = (teacher[key] for key in ('indices', 'probabilities', 'tail_probabilities'))
        if (
            indices.ndim != 2
            or probabilities.shape != indices.shape
            or tail.shape != indices.shape[:1]
            or indices.size(0) < length
            or not (0 if teacher is None else 1) <= indices.size(1) <= self.vocab_size
        ):
            raise ValueError('Teacher CTC probabilities must cover the student subframes and vocabulary')
        if distillation_mask is None:
            distillation_mask = torch.ones(indices.size(0), dtype=torch.bool, device=indices.device)
        if distillation_mask.shape != indices.shape[:1] or distillation_mask.dtype != torch.bool:
            raise ValueError('distillation_mask must be boolean with one entry per teacher subframe')
        required = targets.numel() + int(targets[1:].eq(targets[:-1]).sum())
        feasible = ctc_length >= required
        device = hidden_states.device
        unique = targets.unique(sorted=True)
        compact_ids = torch.cat([torch.tensor([self.blank_id]), unique]).to(device=device)
        mapped_targets = torch.searchsorted(unique, targets) + 1
        compact_chunks, kl_chunks = [], []
        prediction_chunks, probability_chunks = [], []
        for start in range(0, (length + self.subframes - 1) // self.subframes, self.chunk_size):
            end = min(start + self.chunk_size, hidden_states.size(0))
            left, right = self.subframes * start, min(self.subframes * end, length)
            args = (
                hidden_states[start:end],
                compact_ids,
                indices[left:right].detach().to(device=device, dtype=torch.long),
                probabilities[left:right].detach().to(device=device, dtype=torch.float32),
                tail[left:right].detach().to(device=device, dtype=torch.float32),
                distillation_mask[left:right].to(device=device),
            )
            if return_predictions:
                args += (True,)
            if torch.is_grad_enabled():
                values = checkpoint(self._chunk_loss_inputs, *args, use_reentrant=False)
            else:
                values = self._chunk_loss_inputs(*args)
            compact, kl = values[:2]
            compact_chunks.append(compact)
            kl_chunks.append(kl)
            if return_predictions:
                prediction_chunks.append(values[2])
                probability_chunks.append(values[3])
        if compact_chunks:
            compact_log_probs = torch.cat(compact_chunks)
            zero = compact_log_probs.sum() * 0
            denominator = distillation_mask[:length].sum().to(device=device).clamp_min(1)
            distillation_loss = torch.stack(kl_chunks).sum() / denominator
            if feasible and ctc_length:
                ctc_loss = F.ctc_loss(
                    compact_log_probs[input_start:].unsqueeze(1),
                    mapped_targets,
                    torch.tensor([ctc_length]),
                    torch.tensor([targets.numel()]),
                    blank=0,
                    reduction='sum',
                ) / max(targets.numel(), 1)
            else:
                ctc_loss = zero
        else:
            zero = self(hidden_states[:0]).float().sum() * 0
            ctc_loss = distillation_loss = zero
        result = {
            'loss': self.ctc_weight * ctc_loss + self.distillation_weight * distillation_loss,
            'ctc_loss': ctc_loss,
            'distillation_loss': distillation_loss,
            'ctc_valid': feasible,
            'infeasible_count': int(not feasible),
            'supervised_subframes': ctc_length,
            'distilled_subframes': distillation_mask[:length].sum().to(device=device),
        }
        if word_intervals is not None:
            fields = ('token_start', 'token_end', 'start_frame', 'end_frame')
            if isinstance(word_intervals, torch.Tensor):
                if word_intervals.ndim != 2 or word_intervals.size(1) != len(fields):
                    raise ValueError(
                        'word_intervals must have four columns: token_start, token_end, start_frame, end_frame'
                    )
                word_intervals = [dict(zip(fields, row)) for row in word_intervals.detach().cpu().tolist()]
            intervals, skipped, infeasible = [], 0, 0
            for word in word_intervals:
                token_start, token_end, left, right = (word[key] for key in fields)
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (token_start, token_end, left, right)
                ):
                    raise ValueError('Word interval token and subframe indices must be integers')
                left, right = max(left, input_start), min(right, length)
                if not 0 <= token_start < token_end <= targets.numel() or right <= left:
                    skipped += 1
                    continue
                word_targets = targets[token_start:token_end]
                required = word_targets.numel() + int(word_targets[1:].eq(word_targets[:-1]).sum())
                if right - left < required:
                    infeasible += 1
                    continue
                intervals.append((left, right, token_start, token_end))

            # Group similar lengths and bound padded DP inputs for long turns.
            groups, group = [], []
            for interval in sorted(intervals, key=lambda item: item[1] - item[0]):
                padded_elements = (interval[1] - interval[0]) * (len(group) + 1) * (compact_ids.numel() + 1)
                if group and (len(group) == 64 or padded_elements > 8_000_000):
                    groups.append(group)
                    group = []
                group.append(interval)
            if group:
                groups.append(group)
            word_losses = []
            for group in groups:
                sequences = [compact_log_probs[left:right] for left, right, _, _ in group]
                log_probs = nn.utils.rnn.pad_sequence(sequences, padding_value=float('-inf'))
                input_lengths = torch.tensor([right - left for left, right, _, _ in group])
                target_lengths = torch.tensor([last - first for _, _, first, last in group])
                valid = torch.arange(log_probs.size(0), device=device)[:, None] < input_lengths.to(device)[None]
                blank_padding = log_probs.new_full((log_probs.size(-1),), float('-inf'))
                blank_padding[0] = 0
                log_probs = torch.where(valid[..., None], log_probs, blank_padding)
                word_targets = torch.cat([mapped_targets[first:last] for _, _, first, last in group])
                losses = F.ctc_loss(log_probs, word_targets, input_lengths, target_lengths, blank=0, reduction='none')
                word_losses.append((losses / target_lengths.to(device)).sum())
            result.update(
                word_ctc_loss=torch.stack(word_losses).sum() / len(intervals) if word_losses else zero,
                word_ctc_valid_intervals=len(intervals),
                word_ctc_infeasible_intervals=infeasible,
                word_ctc_skipped_intervals=skipped,
            )
        if return_predictions:
            result['predicted_ids'] = (
                torch.cat(prediction_chunks) if prediction_chunks else torch.empty(0, dtype=torch.long, device=device)
            )
            result['predicted_probabilities'] = (
                torch.cat(probability_chunks)
                if probability_chunks
                else torch.empty(0, dtype=torch.float32, device=device)
            )
        return result

    def _chunk_loss_inputs(
        self, hidden_states, compact_ids, indices, probabilities, tail, valid, return_predictions=False
    ):
        logits = self(hidden_states).flatten(0, 1)[: indices.size(0)].float()
        compact = logits.index_select(-1, compact_ids)
        if compact_ids.numel() < self.vocab_size:
            other = logits.index_fill(-1, compact_ids, float('-inf')).logsumexp(-1, keepdim=True)
            compact = torch.cat([compact, other], dim=-1)
        compact_log_probs = compact.log_softmax(-1)
        if indices.size(-1):
            normalizer = logits.logsumexp(-1, keepdim=True)
            selected_log_probs = logits.gather(-1, indices) - normalizer
            probabilities = probabilities.masked_fill(~valid.unsqueeze(-1), 0)
            tail = tail.masked_fill(~valid, 0)
            tiny = torch.finfo(torch.float32).tiny
            kl = (probabilities * (probabilities.clamp_min(tiny).log() - selected_log_probs)).sum(-1)
            if indices.size(-1) < self.vocab_size:
                remaining = logits.scatter(-1, indices, float('-inf')).logsumexp(-1) - normalizer.squeeze(-1)
                kl = kl + tail * (tail.clamp_min(tiny).log() - remaining)
            kl = kl.sum()
        else:
            normalizer = logits.logsumexp(-1, keepdim=True) if return_predictions else None
            kl = compact_log_probs.sum() * 0
        values = compact_log_probs, kl
        if return_predictions:
            with torch.no_grad():
                maxima, predicted = logits.max(-1)
                confidence = (maxima - normalizer.squeeze(-1)).exp()
            values += predicted, confidence
        return values


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


def add_timestamp_markers(audio, sample_rate, boundaries):
    """Return a mono FP32 diagnostic copy with short, tapered clicks at word ends."""
    import numpy as np

    marked = np.array(audio, dtype=np.float32, copy=True)
    if marked.ndim != 1 or isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError('Timestamp audio must be mono and sample_rate must be a positive integer')
    positions = set()
    for value in boundaries:
        try:
            seconds = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(seconds) and 0 <= seconds <= marked.size / sample_rate:
            positions.add(round(seconds * sample_rate))
    size = max(3, round(0.012 * sample_rate))
    phase = np.arange(size) * (2 * np.pi * min(1000, sample_rate / 4) / sample_rate)
    click = (0.15 * np.hanning(size) * np.sin(phase)).astype(np.float32)
    for position in sorted(positions):
        start = int(position) - size // 2
        left, right = max(0, start), min(marked.size, start + size)
        if right > left:
            marked[left:right] = np.clip(marked[left:right] + click[left - start : right - start], -1, 1)
    return marked


def plot_timestamp_alignment(diagnostic):
    """Plot teacher words, cumulative predicted word position, and END probability.

    Frame endpoints are physical audio times, including a possible zero-duration
    EOS row. Word positions follow the raw END predictions; an END frame still
    belongs to the word it closes. Only the display is cropped.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    words = diagnostic['words']
    times = np.asarray(diagnostic['frame_end_seconds'], dtype=np.float64)
    values = diagnostic['end_probabilities']
    if isinstance(values, torch.Tensor):
        values = values.detach().double().numpy()
    probabilities = np.asarray(values, dtype=np.float64)
    targets = np.asarray(diagnostic['targets'])
    valid = np.asarray(diagnostic['valid_mask'], dtype=bool)
    if times.ndim != 1 or any(array.shape != times.shape for array in (probabilities, targets, valid)):
        raise ValueError('Timestamp diagnostic arrays must be one-dimensional with matching shapes')
    if not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError('Frame endpoints must be finite and nondecreasing')
    times = np.clip(times, 0, float(diagnostic['audio_duration']))
    predicted = np.isfinite(probabilities) & (probabilities >= 0.5)
    positions = np.cumsum(predicted) - predicted
    expected = len(words)
    position_count = expected or (int(positions.max()) + 1 if positions.size else 1)
    visible = min(position_count, 80)
    later = position_count > visible
    labels = [f'{index}: {str(words[index]["word"])[:28]}' if words else str(index) for index in range(visible)]
    labels += (['later words'] if later else []) + (['extra'] if words else [])
    shown = min(times.size, 800)
    edges = np.r_[0.0, times[:shown]]
    centers = (edges[:-1] + edges[1:]) / 2
    reference_positions = np.searchsorted([float(word['end']) for word in words], centers, side='right')
    reference_valid = valid[:shown] & (reference_positions < expected)
    if words and shown:
        starts = np.asarray([float(word['start']) for word in words])
        reference_valid &= centers >= starts[np.minimum(reference_positions, expected - 1)]
    head_positions = np.minimum(positions[:shown], visible)
    if words:
        head_positions = np.where(positions[:shown] >= expected, len(labels) - 1, head_positions)
    figure, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True, gridspec_kw={'width_ratios': [1, 1, 0.6]})
    colormap = plt.get_cmap('viridis').copy()
    colormap.set_bad('lightgray')
    ticks = np.unique(
        np.r_[np.linspace(0, visible - 1, min(visible, 24)).astype(int), np.arange(visible, len(labels))]
    )
    for axis, title, index, selected in zip(
        axes[:2],
        ['ASR word alignment', 'Head word position (END count)'],
        [np.minimum(reference_positions, visible), head_positions],
        [reference_valid, np.isfinite(probabilities[:shown])],
    ):
        matrix = (index[:, None] == np.arange(len(labels))[None, :]).astype(np.float32)
        matrix[~selected] = np.nan
        if shown:
            axis.pcolormesh(np.arange(len(labels) + 1) - 0.5, edges, matrix, cmap=colormap, vmin=0, vmax=1)
        else:
            axis.text(0.5, 0.5, 'No timestamp frames', ha='center', transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel('Word occurrence')
        axis.set_xticks(ticks, [labels[index] for index in ticks], rotation=60, ha='right', fontsize=7)
        for tick in axis.get_xticklabels():
            tick.set_parse_math(False)
    if not words:
        axes[0].text(0.5, 0.5, 'Reference words unavailable', ha='center', transform=axes[0].transAxes)
    axes[0].set_ylabel('Speech time (s)')
    axes[2].plot(probabilities[:shown], times[:shown], label='P(END)')
    axes[2].axvline(0.5, color='gray', linestyle='--', label='Threshold')
    reference_ends = valid & (targets == END)
    axes[2].scatter(
        np.ones(reference_ends[:shown].sum()),
        times[:shown][reference_ends[:shown]],
        color='green',
        marker='|',
        label='Reference END',
    )
    axes[2].scatter(
        probabilities[:shown][predicted[:shown]],
        times[:shown][predicted[:shown]],
        color='darkorange',
        marker='x',
        label='Predicted END',
    )
    axes[2].set(
        xlim=(-0.05, 1.08),
        xlabel='Probability',
        title=f'Ends: {predicted.sum()} predicted / {reference_ends.sum()} reference',
    )
    axes[2].legend(loc='upper left', fontsize=7)
    axes[0].set_ylim(0, max(float(times[shown - 1]) if shown else 0, float(diagnostic['frame_seconds'])))
    figure.suptitle('Word-end alignment')
    cropped = f'Display: first {shown}/{times.size} frames, {visible}/{position_count} word positions. '
    figure.text(
        0.5, 0.01, cropped + 'Counts use all frames. Gray: reference supervision unavailable.', ha='center', fontsize=8
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    return figure


def decode_ctc_tokens(token_ids, blank_id, tokenizer, valid_mask=None):
    """Greedily decode a single turn, collapsing repeats before removing blanks.

    Ignored or invalid frames break runs. In particular, ``a, blank, a`` remains
    two tokens, whereas ``a, a`` collapses to one. The tokenizer is the teacher's
    dedicated CTC tokenizer, independently of the TTS conditioning tokenizer.
    """
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1:
            raise ValueError('CTC decoding requires one token ID per frame')
        token_ids = token_ids.detach().cpu().tolist()
    else:
        token_ids = list(token_ids)
    if valid_mask is None:
        valid_mask = [True] * len(token_ids)
    elif isinstance(valid_mask, torch.Tensor):
        if valid_mask.ndim != 1 or valid_mask.dtype != torch.bool:
            raise ValueError('CTC decoding requires a one-dimensional boolean validity mask')
        valid_mask = valid_mask.detach().cpu().tolist()
    else:
        valid_mask = list(valid_mask)
    if len(valid_mask) != len(token_ids):
        raise ValueError('CTC token IDs and validity mask must have the same length')
    previous, collapsed = blank_id, []
    for token, valid in zip(token_ids, valid_mask):
        if not valid or token == IGNORE_INDEX:
            previous = blank_id
            continue
        if token != previous and token != blank_id:
            collapsed.append(int(token))
        previous = token
    return tokenizer.ids_to_text(collapsed)


def plot_ctc_timestamp_alignment(diagnostic, tokenizer, max_frames=800, max_tokens=64):
    """Compare teacher and student greedy CTC emissions on the same audio clock.

    Rows represent vocabulary identities, ordered by first occurrence in the
    reference. Repeated reference pieces share a row; the plot does not infer
    token-occurrence alignment from a vocabulary classifier. Display limits do
    not truncate the decoded text or CER supplied by the caller.
    """
    from collections import Counter

    import matplotlib.pyplot as plt
    import numpy as np

    def array(name, dtype=None):
        value = diagnostic[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    starts, ends = array('frame_start_seconds', np.float64), array('frame_end_seconds', np.float64)
    valid = array('valid_mask', bool)
    teacher_ids = array('teacher_indices', np.int64)
    teacher_probabilities = array('teacher_probabilities', np.float64)
    tail = array('teacher_tail_probabilities', np.float64)
    student_ids = array('student_token_ids', np.int64)
    student_probabilities = array('student_probabilities', np.float64)
    frames = starts.size
    if starts.ndim != 1 or any(
        value.shape != starts.shape for value in (ends, valid, tail, student_ids, student_probabilities)
    ):
        raise ValueError('CTC diagnostic frame arrays must be one-dimensional with matching shapes')
    if teacher_ids.ndim != 2 or teacher_ids.shape != teacher_probabilities.shape or teacher_ids.shape[0] != frames:
        raise ValueError('CTC teacher IDs and probabilities must have shape (frames, retained classes)')
    if not teacher_ids.shape[1]:
        raise ValueError('CTC diagnostics require at least one retained teacher class')
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or (ends < starts).any():
        raise ValueError('CTC frame intervals must be finite and nonnegative in duration')
    if (np.diff(starts) < 0).any() or (starts[1:] < ends[:-1] - 1e-8).any():
        raise ValueError('CTC frame intervals must be ordered and nonoverlapping')
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames < 1
        or isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise ValueError('CTC display limits must be positive')

    blank_id = int(diagnostic['blank_id'])
    winner = teacher_probabilities.argmax(axis=-1)
    teacher_tokens = np.take_along_axis(teacher_ids, winner[:, None], axis=-1).reshape(-1)
    teacher_confidence = np.take_along_axis(teacher_probabilities, winner[:, None], axis=-1).reshape(-1)
    reference = list(dict.fromkeys(int(token) for token in diagnostic['reference_token_ids'] if token != blank_id))
    observed = Counter(
        int(token)
        for token in np.r_[teacher_tokens[valid], student_ids[valid]]
        if token not in (blank_id, IGNORE_INDEX)
    )
    identities = reference[:max_tokens]
    identities += [token for token, _ in observed.most_common() if token not in reference][
        : max(0, max_tokens - len(identities))
    ]
    identities = [blank_id, *identities]
    omitted = set(reference).union(observed).difference(identities)
    labels = ['<blank>'] + [
        f'{token}: {piece}' for token, piece in zip(identities[1:], tokenizer.ids_to_tokens(identities[1:]))
    ]
    if omitted:
        labels.append('other / undisplayed IDs')
    rows = {token: index for index, token in enumerate(identities)}
    shown = min(frames, max_frames)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, min(13, max(4.5, 0.22 * len(labels)))),
        sharex=True,
        gridspec_kw={'width_ratios': [1, 1, 0.7]},
    )
    colormap = plt.get_cmap('viridis').copy()
    colormap.set_bad('lightgray')
    for axis, title, tokens, confidence in zip(
        axes[:2],
        ['Frozen ASR teacher', 'Distilled CTC head'],
        [teacher_tokens, student_ids],
        [teacher_confidence, student_probabilities],
    ):
        matrix = np.zeros((len(labels), shown), dtype=np.float32)
        if shown:
            indices = [rows.get(int(token), len(labels) - 1) for token in tokens[:shown]]
            matrix[indices, np.arange(shown)] = confidence[:shown]
            matrix[:, ~valid[:shown]] = np.nan
            # Plot one rectangle per subframe, preserving cropped boundary bins.
            for left, right in zip(
                np.flatnonzero(np.r_[True, starts[1:shown] > ends[: shown - 1] + 1e-8]),
                np.r_[np.flatnonzero(starts[1:shown] > ends[: shown - 1] + 1e-8) + 1, shown],
            ):
                image = axis.pcolormesh(
                    np.r_[starts[left:right], ends[right - 1]],
                    np.arange(len(labels) + 1) - 0.5,
                    matrix[:, left:right],
                    cmap=colormap,
                    vmin=0,
                    vmax=1,
                )
            figure.colorbar(image, ax=axis, label='Winning-token probability', fraction=0.04, pad=0.02)
        else:
            axis.text(0.5, 0.5, 'No CTC frames', ha='center', transform=axis.transAxes)
        axis.set(title=title, xlabel='Speech time (s)', ylabel='CTC token identity')
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
        for tick in axis.get_yticklabels():
            tick.set_parse_math(False)
    centers = (starts[:shown] + ends[:shown]) / 2
    teacher_blank = np.where(teacher_ids == blank_id, teacher_probabilities, 0).sum(axis=-1)
    for values, label, style in (
        (teacher_confidence, 'Teacher winning token', '-'),
        (student_probabilities, 'Student winning token', '-'),
        (teacher_blank, 'Teacher P(blank)', '--'),
    ):
        axes[2].plot(centers, np.where(valid[:shown], values[:shown], np.nan), style, label=label)
    blank = valid[:shown] & (student_ids[:shown] == blank_id)
    axes[2].scatter(centers[blank], student_probabilities[:shown][blank], s=8, marker='x', label='Student blank wins')
    axes[2].set(title='CTC confidence', xlabel='Speech time (s)', ylabel='Probability', ylim=(-0.03, 1.03))
    axes[2].legend(fontsize=7, loc='lower left')
    if shown:
        axes[0].set_xlim(starts[0], max(ends[shown - 1], starts[0] + 0.04))
    figure.suptitle('Teacher and student greedy CTC emissions')
    caption = f'Display: {shown}/{frames} frames. Repeated token IDs share a row; these are not forced alignments.'
    figure.text(0.5, 0.10, caption, ha='center', fontsize=8)
    for key, label, y in (
        ('reference_text', 'TTS input', 0.075),
        ('teacher_text', 'Teacher', 0.05),
        ('student_text', 'Student', 0.025),
    ):
        text = str(diagnostic.get(key, '')).replace('\n', ' ')
        figure.text(0.5, y, f'{label}: {text[:180]}', ha='center', fontsize=8, parse_math=False)
    figure.tight_layout(rect=(0, 0.135, 1, 0.95))
    return figure


class TTSCtcTokenizer:
    """Expose the existing aggregated TTS vocabulary to an auxiliary CTC head.

    Encoding delegates to the selected input tokenizer with its global ID offset.
    Decoding accepts predictions from any tokenizer in the aggregate vocabulary.
    The CTC blank remains external and must be removed by the CTC decoder.
    """

    def __init__(self, tokenizer, tokenizer_name=None):
        self.tokenizer = tokenizer
        if tokenizer_name is None:
            if len(tokenizer.tokenizers) != 1:
                raise ValueError('Select a TTS tokenizer for CTC when multiple input tokenizers are configured')
            tokenizer_name = next(iter(tokenizer.tokenizers))
        if tokenizer_name not in tokenizer.tokenizers:
            raise ValueError(f'Unknown TTS CTC tokenizer: {tokenizer_name!r}')
        self.tokenizer_name = tokenizer_name
        self.vocab_size = len(tokenizer.tokens)
        self._names = sorted(tokenizer.tokenizer_offsets, key=tokenizer.tokenizer_offsets.get)
        self._offsets = [tokenizer.tokenizer_offsets[name] for name in self._names]

    def text_to_ids(self, text):
        """Encode the complete TTS input with the same global IDs as conditioning."""
        return self.tokenizer.encode(text, tokenizer_name=self.tokenizer_name)

    def ids_to_text(self, ids):
        """Decode contiguous runs with their owning native input tokenizer."""
        return ''.join(self.tokenizer.decode(list(tokens), tokenizer_name=name) for name, tokens in self._groups(ids))

    def ids_to_tokens(self, ids):
        """Return native token pieces in their original global prediction order."""
        pieces = []
        for name, tokens in self._groups(ids):
            native = self.tokenizer.tokenizers[name]
            local = [token - self.tokenizer.tokenizer_offsets[name] for token in tokens]
            convert = getattr(native, 'ids_to_tokens', None) or getattr(native, 'convert_ids_to_tokens', None)
            pieces.extend(convert(local) if convert is not None else [native.tokens[token] for token in local])
        return pieces

    def _groups(self, ids):
        from itertools import groupby

        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return groupby(map(int, ids), key=self._owner)

    def _owner(self, token):
        from bisect import bisect_right

        if not 0 <= token < self.vocab_size:
            raise ValueError('Remove CTC blank and ignored IDs before decoding the TTS vocabulary')
        return self._names[bisect_right(self._offsets, token) - 1]


def map_tts_ctc_word_intervals(tokenizer, text, token_ids, words, audio_duration):
    """Map complete native input BPE occurrences to forced-aligned word intervals.

    Offsets come from one full-text native tokenization and must reproduce the
    actual conditioning IDs exactly. No target is re-tokenized per word. BPEs
    spanning words merge their intervals; standalone punctuation joins the
    nearest lexical word. Crops and incompatible transcripts are skipped.
    """
    from bisect import bisect_left, bisect_right

    def result(error, segments=()):
        return dict(
            segments=list(segments),
            error=error,
            token_coverage=0.0 if error else 1.0,
            merged_word_groups=0 if error else len(words) - len(segments),
        )

    if not isinstance(text, str) or not text or not words:
        return result('empty_transcript')
    try:
        duration = float(audio_duration)
        times = [(float(word['start']), float(word['end'])) for word in words]
    except (KeyError, TypeError, ValueError, OverflowError):
        return result('invalid_word_times')
    if (
        not math.isfinite(duration)
        or duration <= 0
        or any(
            not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration + 1e-6
            for start, end in times
        )
        or any(right[0] < left[1] - 1e-6 for left, right in zip(times, times[1:]))
    ):
        return result('invalid_word_times')

    lexical, starts, ends = _lexical_text_offsets(text)
    word_ends, cursor = [], 0
    try:
        for word in words:
            unit, _, _ = _lexical_text_offsets(word['word'])
            if not unit:
                return result('nonlexical_teacher_word')
            if lexical[cursor : cursor + len(unit)] != unit:
                return result('transcript_mismatch')
            cursor += len(unit)
            word_ends.append(cursor)
    except (KeyError, TypeError):
        return result('transcript_mismatch')
    if cursor != len(lexical) or not lexical:
        return result('transcript_mismatch')

    native = tokenizer.tokenizer.tokenizers[tokenizer.tokenizer_name]
    try:
        encoded = native(text, return_offsets_mapping=True, add_special_tokens=False)
        offset = tokenizer.tokenizer.tokenizer_offsets[tokenizer.tokenizer_name]
        encoded_ids = [int(token) + offset for token in encoded['input_ids']]
        offsets = encoded['offset_mapping']
    except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
        return result('offsets_unavailable')
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if encoded_ids != list(token_ids):
        return result('input_token_mismatch')
    if not encoded_ids or not isinstance(offsets, Sequence) or len(offsets) != len(encoded_ids):
        return result('invalid_token_offsets')

    ownership, previous = [], (0, 0)
    for bounds in offsets:
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            return result('invalid_token_offsets')
        left, right = bounds
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in bounds) or not (
            0 <= left < right <= len(text) and left >= previous[0] and right >= previous[1]
        ):
            return result('invalid_token_offsets')
        previous = left, right
        first, last = bisect_right(ends, left), bisect_left(starts, right)
        if first == last:
            following = bisect_left(starts, right)
            preceding = following - 1
            nearest = (
                preceding
                if preceding >= 0 and (following == len(starts) or left - ends[preceding] <= starts[following] - right)
                else following
            )
            first = last = nearest
        else:
            last -= 1
        ownership.append((bisect_right(word_ends, first), bisect_right(word_ends, last)))

    segments, token_start, word_start = [], 0, 0
    word_end = ownership[0][1]
    if ownership[0][0] != 0:
        return result('uncovered_word')
    for index, (left, right) in enumerate(ownership[1:], 1):
        if left <= word_end:
            word_end = max(word_end, right)
            continue
        if left != word_end + 1:
            return result('uncovered_word')
        segments.append(
            dict(token_start=token_start, token_end=index, start=times[word_start][0], end=times[word_end][1])
        )
        token_start, word_start, word_end = index, left, right
    if word_end != len(words) - 1:
        return result('uncovered_word')
    segments.append(
        dict(token_start=token_start, token_end=len(encoded_ids), start=times[word_start][0], end=times[word_end][1])
    )
    return result(None, segments)


def _lexical_text_offsets(text):
    """Retain source-character spans while comparing casefolded Unicode lexemes."""
    import unicodedata

    characters, starts, ends = [], [], []
    for index, character in enumerate(text):
        for normalized in unicodedata.normalize('NFKD', character).casefold():
            if unicodedata.category(normalized)[0] in 'LMN':
                characters.append(normalized)
                starts.append(index)
                ends.append(index + 1)
    return ''.join(characters), starts, ends


def _frame_coordinate(seconds: float, frame_seconds: float) -> float:
    """Avoid spurious shared bins from floating-point roundoff at exact boundaries."""
    coordinate = seconds / frame_seconds
    nearest = round(coordinate)
    if abs(coordinate - nearest) <= 4 * math.ulp(coordinate):
        return float(nearest)
    return coordinate
