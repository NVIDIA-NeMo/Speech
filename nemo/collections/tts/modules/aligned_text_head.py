# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable text-head supervision from independently aligned speech turns."""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class AlignedTextTargets:
    """Supervision on the original acoustic grid, before causal shifts or EOS."""

    loss: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor
    valid_turn_count: torch.Tensor
    turn_count: torch.Tensor
    one_frame_tokens: torch.Tensor
    token_count: torch.Tensor


class AlignedTextHead(nn.Module):
    """Factorized BPE output head with bounded training-logit memory.

    ``forward`` is useful for one-frame inference. ``loss`` projects only valid
    training frames in checkpointed chunks, so it never retains a full
    conversation-by-vocabulary logits tensor for backward.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, projection_dim: int = 128):
        super().__init__()
        if min(hidden_dim, vocab_size, projection_dim) < 1:
            raise ValueError('Hidden, vocabulary and projection dimensions must be positive')
        self.input_projection = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.output_projection = nn.Linear(projection_dim, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.input_projection(hidden))

    def loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """Mean cross-entropy over the selected frames, with FP32 accumulation."""
        if hidden.ndim != 3 or targets.shape != hidden.shape[:2] or mask.shape != targets.shape:
            raise ValueError('Expected hidden [B,T,H] and matching targets/mask [B,T]')
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        selected_hidden = hidden[mask.bool()]
        selected_targets = targets[mask.bool()].long()
        if selected_targets.numel() == 0:
            return hidden.float().sum() * 0.0 + _zero_parameter_loss(self, hidden)
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, selected_targets.numel(), chunk_size):
            features = selected_hidden[start : start + chunk_size]
            labels = selected_targets[start : start + chunk_size]
            if torch.is_grad_enabled():
                chunk_loss = checkpoint(
                    self._chunk_loss, features, labels, use_reentrant=False, preserve_rng_state=False
                )
            else:
                chunk_loss = self._chunk_loss(features, labels)
            total = total + chunk_loss
        return total / selected_targets.numel()

    def _chunk_loss(self, features, labels):
        return F.cross_entropy(self(features).float(), labels, reduction='sum')


def align_text_turns(
    aligner: nn.Module,
    audio_features: torch.Tensor,
    text_tokens: torch.Tensor,
    text_lens: torch.Tensor,
    turn_indices: torch.Tensor,
    warmup_targets: torch.Tensor,
    embed_text: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pad_id: int,
    use_alignment: bool,
    max_turns_per_batch: int = 32,
    max_frames_per_batch: int = 8192,
) -> AlignedTextTargets:
    """Align compact text turns and restore their labels to the conversation grid.

    ``turn_indices`` has rows ``(sample, start, end_exclusive)`` on the acoustic
    grid. Text contains content tokens only: the caller owns BOS/EOS handling,
    channel delays and autoregressive shifts. Acoustic inputs and the existing
    text embeddings are detached from the TTS generator; the likelihood still
    trains every aligner parameter.

    Turns are sorted by audio length and packed under a padded-frame budget.
    No acoustic-score tensor is transferred to CPU. Only the small index/length
    table is copied once to organize batches. Invalid turns are excluded from
    both likelihood and text-head supervision. A single turn longer than the
    frame budget raises rather than truncating speech or changing its labels.

    Before the curriculum switch, valid spans retain ``warmup_targets``.
    Afterwards they contain BPE tokens repeated according to the MAS path.
    Outside valid spans, targets are PAD and the loss mask is false.
    """
    if audio_features.ndim != 3 or warmup_targets.shape != audio_features.shape[:2]:
        raise ValueError('Expected audio_features [B,T,C] and warmup_targets [B,T]')
    if text_tokens.ndim != 2 or text_lens.shape != (text_tokens.size(0),):
        raise ValueError('Expected compact text_tokens [turns,N] and text_lens [turns]')
    if turn_indices.shape != (text_tokens.size(0), 3):
        raise ValueError('turn_indices must have shape [turns,3]')
    if max_turns_per_batch < 1 or max_frames_per_batch < 1:
        raise ValueError('Turn and frame batch limits must be positive')
    if text_lens.dtype not in (torch.int32, torch.int64) or turn_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError('Text lengths and turn indices must contain integers')
    if text_tokens.device != audio_features.device or warmup_targets.device != audio_features.device:
        raise ValueError('Audio, compact text and warmup targets must be on the same device')

    metadata = torch.cat([turn_indices, text_lens.to(turn_indices.device).unsqueeze(1)], dim=1).detach().cpu().tolist()
    batch_size, audio_steps, channels = audio_features.shape
    valid_rows = []
    for row, (sample, start, end, num_text) in enumerate(metadata):
        if (
            0 <= sample < batch_size
            and 0 <= start < end <= audio_steps
            and 0 < num_text <= text_tokens.size(1)
            and end - start >= num_text
        ):
            if end - start > max_frames_per_batch:
                raise ValueError(
                    f'Turn {row} has {end - start} frames, exceeding max_frames_per_batch={max_frames_per_batch}'
                )
            valid_rows.append(row)
    # Overlapping targets would make restoration depend on the sorting order.
    previous_ends = {}
    for row in sorted(valid_rows, key=lambda index: metadata[index][:2]):
        sample, start, end, _ = metadata[row]
        if start < previous_ends.get(sample, 0):
            raise ValueError('Alignment turn spans must not overlap within a sample')
        previous_ends[sample] = end
    valid_rows.sort(key=lambda index: metadata[index][2] - metadata[index][1])
    batches = _turn_batches(valid_rows, metadata, max_turns_per_batch, max_frames_per_batch)

    targets = torch.full_like(warmup_targets, pad_id)
    loss_mask = torch.zeros_like(warmup_targets, dtype=torch.bool)
    weighted_loss = audio_features.new_zeros((), dtype=torch.float32)
    counts = torch.zeros(4, dtype=torch.long, device=audio_features.device)
    # Counts: valid turns, one-frame text tokens, text tokens, acoustic frames.
    for batch_rows in batches:
        max_audio = max(metadata[row][2] - metadata[row][1] for row in batch_rows)
        max_text = max(metadata[row][3] for row in batch_rows)
        packed_audio = audio_features.new_zeros(len(batch_rows), max_audio, channels)
        audio_lengths = []
        for packed_row, row in enumerate(batch_rows):
            sample, start, end, _ = metadata[row]
            packed_audio[packed_row, : end - start] = audio_features[sample, start:end].detach()
            audio_lengths.append(end - start)
        row_indices = torch.tensor(batch_rows, device=text_tokens.device, dtype=torch.long)
        packed_text = text_tokens.index_select(0, row_indices)[:, :max_text]
        packed_text_lens = text_lens.to(audio_features.device).index_select(0, row_indices)
        packed_audio_lens = torch.tensor(audio_lengths, device=audio_features.device, dtype=torch.long)
        with torch.no_grad():
            text_features = embed_text(packed_text, packed_text_lens).detach()
        result = aligner(packed_audio, text_features, packed_audio_lens, packed_text_lens)
        valid = result.valid_turns.to(dtype=torch.bool, device=audio_features.device)
        frame_count = (packed_audio_lens * valid).sum()
        weighted_loss = weighted_loss + result.loss * frame_count
        counts[0] += valid.sum()
        counts[1] += ((result.durations == 1) & valid[:, None]).sum()
        counts[2] += (packed_text_lens * valid).sum()
        counts[3] += frame_count
        for packed_row, row in enumerate(batch_rows):
            sample, start, end, _ = metadata[row]
            path = result.token_indices[packed_row, : end - start]
            valid_frames = valid[packed_row] & (path >= 0)
            if use_alignment:
                labels = packed_text[packed_row].gather(0, path.clamp_min(0))
            else:
                labels = warmup_targets[sample, start:end]
            targets[sample, start:end] = torch.where(valid_frames, labels, pad_id)
            loss_mask[sample, start:end] = valid_frames

    loss = weighted_loss / counts[3].clamp_min(1)
    if not batches:
        loss = loss + _zero_parameter_loss(aligner, audio_features)
    return AlignedTextTargets(
        loss=loss,
        targets=targets,
        loss_mask=loss_mask,
        valid_turn_count=counts[0],
        turn_count=torch.tensor(text_tokens.size(0), device=audio_features.device, dtype=torch.long),
        one_frame_tokens=counts[1],
        token_count=counts[2],
    )


def _turn_batches(rows, metadata, max_turns, max_frames):
    batches, current = [], []
    max_audio = 0
    for row in rows:
        audio_length = metadata[row][2] - metadata[row][1]
        next_max = max(max_audio, audio_length)
        if current and (len(current) >= max_turns or (len(current) + 1) * next_max > max_frames):
            batches.append(current)
            current, max_audio = [], 0
        current.append(row)
        max_audio = max(max_audio, audio_length)
    if current:
        batches.append(current)
    return batches


def _zero_parameter_loss(module, reference):
    return sum(
        (parameter.sum() * 0.0 for parameter in module.parameters()), reference.new_zeros((), dtype=torch.float32)
    )
