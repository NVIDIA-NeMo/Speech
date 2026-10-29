# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-only attention guidance from detached acoustic alignments.

The aligner supplies token *positions*, not BPE IDs. Only explicitly selected
audio queries are supervised. Their targets are mapped to the exact conditioning
key positions, respecting padding and causal availability. Losses use the full
attention normalization so matching a text-only slice cannot hide negligible
attention mass on text.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class AttentionAlignmentTurn:
    """A complete output turn on the decoder's absolute, pre-shifted timeline."""

    batch_index: int
    query_positions: torch.Tensor
    key_positions: torch.Tensor
    token_indices: torch.Tensor


@dataclass
class AttentionAlignmentGuidance:
    """No parameters or mutable loss accumulators are stored in this request.

    ``prior_strength=0`` preserves decoder outputs exactly while still computing
    the auxiliary loss. One is a soft text-band prior; intermediate values mix
    that prior with all ones. Heads not listed here always retain ordinary
    attention. ``heads=None`` supervises all heads.
    """

    turns: list[AttentionAlignmentTurn]
    prior_strength: float = 0.0
    band_width: float = 1.0
    heads: tuple[int, ...] | None = (0,)
    query_chunk_size: int = 128

    def __post_init__(self):
        if not math.isfinite(self.prior_strength) or not 0 <= self.prior_strength <= 1:
            raise ValueError('prior_strength must be finite and in [0, 1]')
        if not math.isfinite(self.band_width) or self.band_width <= 0:
            raise ValueError('band_width must be finite and positive')
        if type(self.query_chunk_size) is not int or self.query_chunk_size < 1:
            raise ValueError('query_chunk_size must be a positive integer')
        if self.heads is not None:
            self.heads = tuple(self.heads)
            if (
                not self.heads
                or any(type(head) is not int or head < 0 for head in self.heads)
                or len(set(self.heads)) != len(self.heads)
            ):
                raise ValueError('heads must be null or distinct nonnegative head indices')


def apply_attention_alignment(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ordinary_output: torch.Tensor,
    guidance: AttentionAlignmentGuidance,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Guide selected heads without retaining a conversation-square matrix.

    Q/K/V and ordinary output have shape ``[B,H,T,D]``; grouped K/V must have
    already been repeated to the query head count. Cached decoding is deliberately
    unsupported: this teacher is only available for complete training/validation
    examples. Chunk-level non-reentrant checkpointing bounds retained attention
    probabilities while preserving gradients through the guidance and prior.
    """
    if query.ndim != 4 or key.shape != value.shape or ordinary_output.shape != query.shape:
        raise ValueError('Expected Q/K/V and attention output [B,H,T,D]')
    if key.shape[:2] != query.shape[:2] or key.size(-1) != query.size(-1):
        raise ValueError('Attention guidance requires expanded K/V with matching query heads')
    if key.size(2) != query.size(2):
        raise ValueError('Attention guidance does not support cached decoding')
    if attention_mask is not None and attention_mask.ndim not in (2, 4):
        raise ValueError('Attention guidance requires a 2D padding or 4D attention mask')

    batch_size, num_heads, query_length, _ = query.shape
    head_indices = tuple(range(num_heads)) if guidance.heads is None else guidance.heads
    if any(head >= num_heads for head in head_indices):
        raise ValueError('An attention alignment head index exceeds the decoder head count')
    selected_heads = torch.tensor(head_indices, device=query.device, dtype=torch.long)
    # Retain the original tensor and arithmetic exactly when the prior is off.
    output = ordinary_output.clone() if guidance.prior_strength else ordinary_output
    loss_sum = query.float().sum() * 0.0 + key.float().sum() * 0.0
    count = 0
    seen_queries = set()
    for turn in guidance.turns:
        _validate_turn(turn, query, query_length, batch_size)
        if not turn.query_positions.numel() or not turn.key_positions.numel():
            continue
        positions = turn.query_positions.detach().long()
        keys = turn.key_positions.detach().long()
        centers = turn.token_indices.detach().long()
        target_exists = (centers >= 0) & (centers < keys.numel())
        center_keys = keys[centers.clamp(0, keys.numel() - 1)]
        target_exists = target_exists & (center_keys <= positions)
        if attention_mask is not None and attention_mask.ndim == 2:
            target_exists = target_exists & attention_mask[turn.batch_index, positions].bool()
        # Validate overlapping output spans once, outside checkpointed arithmetic.
        for position in positions.detach().cpu().tolist():
            identity = (turn.batch_index, position)
            if identity in seen_queries:
                raise ValueError('Attention alignment query spans must not overlap')
            seen_queries.add(identity)
        if attention_mask is not None:
            center_mask = _select_center_mask(attention_mask, turn.batch_index, selected_heads, positions, center_keys)
            target_exists = target_exists & _mask_visibility(center_mask).all(dim=0)
        positions = positions[target_exists]
        centers = centers[target_exists]
        if not positions.numel():
            continue
        selected_query = query[turn.batch_index].index_select(0, selected_heads)
        selected_key = key[turn.batch_index].index_select(0, selected_heads)
        selected_value = value[turn.batch_index].index_select(0, selected_heads)
        for start in range(0, positions.numel(), guidance.query_chunk_size):
            rows = positions[start : start + guidance.query_chunk_size]
            row_centers = centers[start : start + guidance.query_chunk_size]
            q = selected_query.index_select(1, rows)
            if attention_mask is None:
                # Keep checkpoint inputs compact; the causal query axis is
                # reconstructed inside the chunk rather than retained per row.
                mask = torch.ones((1, 1, key.size(2)), device=q.device, dtype=torch.bool)
            else:
                mask = _select_attention_mask(attention_mask, turn.batch_index, selected_heads, rows, key.size(2))
            args = (
                q,
                selected_key,
                selected_value,
                rows,
                keys,
                row_centers,
                mask,
                guidance.prior_strength,
                guidance.band_width,
                dropout_p,
            )
            if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, selected_key, selected_value)):
                chunk_loss, guided_output = checkpoint(_alignment_chunk, *args, use_reentrant=False)
            else:
                chunk_loss, guided_output = _alignment_chunk(*args)
            loss_sum = loss_sum + chunk_loss
            count += len(head_indices) * rows.numel()
            if guidance.prior_strength:
                output[turn.batch_index, selected_heads[:, None], rows[None, :], :] = guided_output
    return output, loss_sum / max(count, 1)


def _alignment_chunk(query, key, value, rows, text_keys, centers, mask, prior_strength, band_width, dropout_p):
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(query.size(-1))
        causal = torch.arange(key.size(1), device=query.device)[None, None, :] <= rows[None, :, None]
        visible = causal & _mask_visibility(mask)
        if mask.dtype != torch.bool:
            scores = scores + mask.float()
        scores = scores.masked_fill(~visible, -torch.inf)
        log_probs = F.log_softmax(scores, dim=-1)

        token_distance = torch.arange(text_keys.numel(), device=query.device)[None, :] - centers[:, None]
        target = torch.exp(-0.5 * (token_distance.float() / band_width).square())
        text_visible = visible.expand(query.size(0), -1, -1).index_select(-1, text_keys)
        target = target[None, :, :] * text_visible
        target = (target / target.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)).detach()
        text_log_probs = log_probs.index_select(-1, text_keys)
        loss = -(target * torch.where(target > 0, text_log_probs, 0.0)).sum()

        if not prior_strength:
            return loss, query.new_empty((0,))
        prior = torch.full_like(scores, 1e-6)
        prior = prior.scatter(-1, text_keys[None, None, :].expand_as(target), target.clamp_min(1e-6))
        prior = (1.0 - prior_strength) + prior_strength * prior
        probabilities = F.softmax(scores + prior.log(), dim=-1)
        probabilities = F.dropout(probabilities, p=dropout_p, training=dropout_p > 0)
        guided_output = torch.matmul(probabilities, value.float()).to(dtype=query.dtype)
        return loss, guided_output


def _validate_turn(turn, query, query_length, batch_size):
    if type(turn.batch_index) is not int or not 0 <= turn.batch_index < batch_size:
        raise ValueError('Attention alignment batch_index is out of range')
    for field in (turn.query_positions, turn.key_positions, turn.token_indices):
        if field.ndim != 1 or field.dtype not in (torch.int32, torch.int64) or field.device != query.device:
            raise ValueError(
                'Attention alignment positions and indices must be 1D integer tensors on the query device'
            )
    if turn.query_positions.shape != turn.token_indices.shape:
        raise ValueError('Each guided query requires one teacher token index')
    for positions in (turn.query_positions, turn.key_positions):
        if positions.numel() and (
            torch.any(positions < 0)
            or torch.any(positions >= query_length)
            or torch.any(positions[1:] <= positions[:-1])
        ):
            raise ValueError(
                'Attention alignment positions must be strictly increasing and within the decoder timeline'
            )


def _select_attention_mask(mask, batch_index, heads, rows, key_length):
    if mask.ndim == 2:
        return mask[batch_index, :key_length].bool()[None, None, :].expand(heads.numel(), rows.numel(), -1)
    batch_mask = mask[batch_index if mask.size(0) > 1 else 0]
    if batch_mask.size(0) == 1:
        batch_mask = batch_mask.expand(heads.numel(), -1, -1)
    else:
        batch_mask = batch_mask.index_select(0, heads)
    if batch_mask.size(1) == 1:
        return batch_mask[:, :, :key_length].expand(-1, rows.numel(), -1)
    return batch_mask.index_select(1, rows)[:, :, :key_length]


def _select_center_mask(mask, batch_index, heads, rows, centers):
    if mask.ndim == 2:
        return mask[batch_index, centers].bool()[None, :].expand(heads.numel(), -1)
    selected_batch = batch_index if mask.size(0) > 1 else 0
    selected_heads = heads if mask.size(1) > 1 else torch.zeros_like(heads)
    selected_rows = rows if mask.size(2) > 1 else torch.zeros_like(rows)
    return mask[selected_batch, selected_heads[:, None], selected_rows[None, :], centers[None, :]]


def _mask_visibility(mask):
    if mask.dtype == torch.bool:
        return mask
    if not mask.is_floating_point():
        return mask != 0
    return torch.isfinite(mask) & (mask > torch.finfo(mask.dtype).min / 2)
