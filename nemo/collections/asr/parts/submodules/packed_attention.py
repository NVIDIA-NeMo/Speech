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

"""Self-attention over sequence-packed activations with explicit per-utterance boundaries."""

import functools
import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from nemo.collections.asr.parts.packed_sequence import (
    PackedEncoderActivations,
    _lengths_to_cu_seqlens,
    _new_packed_encoder_activations,
    packed_encoder_position_ids,
    packed_sequence_ids,
)

__all__ = ["PackedAttentionContext", "packed_attention"]

# the sparse block size of FlexAttention; the document block mask is built at this granularity
BLOCK_SIZE = 128
# The FlexAttention kernels are compiled statically per shape,
# because with dynamic shapes Dynamo cannot trace the gather of the relative-position bias.
# The token axis and the relative-position width the kernels see are rounded up to powers of two
# from these floors, so the kernels of a run are bounded by the logarithms of the two ranges,
# not by the number of distinct batches, and the recompile limit below leaves room for them.
# Reaching the limit anyway is an error, not a fallback: past its limit Dynamo runs a function eagerly,
# and eager flex_attention materializes the dense scores over the padded token axis.
TOKEN_PADDING = 4096
LENGTH_PADDING = 128
RECOMPILE_LIMIT = 64


def _padded_size(value: int, floor: int) -> int:
    """Round ``value`` up to a power of two, at least ``floor`` (a power of two itself)."""
    return max(floor, 1 << max(value - 1, 0).bit_length())


@functools.cache
def _compiled_flex_attention() -> Callable:
    """flex_attention compiled statically per shape, at the first call so that importing the encoders stays cheap."""
    return torch.compile(flex_attention, dynamic=False)


def _recompile_limit_patch():
    """Dynamo settings for the per-shape FlexAttention kernels: room for RECOMPILE_LIMIT shapes, an error beyond."""
    # torch renamed `cache_size_limit` to `recompile_limit`; older versions have only the old name
    name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    settings = {name: max(getattr(torch._dynamo.config, name), RECOMPILE_LIMIT)}
    if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        settings["fail_on_recompile_limit_hit"] = True
    return torch._dynamo.config.patch(**settings)


class PackedAttentionContext:
    """Packing metadata of one batch, shared by all attention layers of a packed forward.

    The FlexAttention block mask is derived from the utterance boundaries,
    without evaluating a mask over the dense token grid:
    a key block is complete for a query block when both lie inside the same utterance,
    partial when it intersects an utterance that the query block touches,
    and skipped otherwise.

    Args:
        packed: the packed activations entering the encoder layers; only their sequence metadata is used.
    """

    def __init__(self, packed: PackedEncoderActivations):
        self.lengths = packed.lengths
        self.cu_seqlens = packed.cu_seqlens
        self.max_seqlen = packed.max_seqlen
        self.total_tokens = packed.total_tokens
        self.device = packed.data.device
        self.sequence_ids = packed_sequence_ids(packed)
        self.position_ids = packed_encoder_position_ids(packed)
        # the shapes the compiled attention kernels see, see TOKEN_PADDING and LENGTH_PADDING
        self.padded_total_tokens = _padded_size(self.total_tokens, TOKEN_PADDING)
        self.padded_max_seqlen = _padded_size(self.max_seqlen, LENGTH_PADDING)
        self._block_mask = None
        self._bias_rows = None
        self._sequence_slices = None

    @classmethod
    def from_lengths(cls, lengths: Tensor) -> "PackedAttentionContext":
        """Metadata for token-flat ids or states whose sequences have these lengths, in packing order."""
        lengths = lengths.to(torch.int64)
        total_tokens = int(lengths.sum().item())
        max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
        placeholder = lengths.new_empty((total_tokens, 0), dtype=torch.float32)
        return cls(_new_packed_encoder_activations(placeholder, lengths, _lengths_to_cu_seqlens(lengths), max_seqlen))

    @property
    def sequence_slices(self) -> tuple[tuple[int, int], ...]:
        """Host copies of the utterance boundaries, for the per-utterance reference path."""
        if self._sequence_slices is None:
            offsets = self.cu_seqlens.tolist()
            self._sequence_slices = tuple(zip(offsets[:-1], offsets[1:]))
        return self._sequence_slices

    def block_mask(self) -> BlockMask:
        """The document block mask over the padded token axis, built once per batch."""
        if self._block_mask is None:
            self._block_mask = document_block_mask(
                self.sequence_ids, self.cu_seqlens, self.total_tokens, padded_total_tokens=self.padded_total_tokens
            )
        return self._block_mask

    def bias_rows(self) -> Tensor:
        """The relative-position bias row of every position of the padded token axis.

        Tokens read their own row and the tail beyond the last token reads the last token's row,
        so the kernels gather the bias in bounds without a padded copy of it; the tail is masked.
        """
        if self._bias_rows is None:
            rows = torch.arange(self.padded_total_tokens, device=self.device)
            self._bias_rows = rows.clamp(max=max(self.total_tokens - 1, 0))
        return self._bias_rows


def document_block_mask(
    sequence_ids: Tensor, cu_seqlens: Tensor, total_tokens: int, padded_total_tokens: int | None = None
) -> BlockMask:
    """Build the FlexAttention block mask that confines attention to the utterance of every token.

    Args:
        sequence_ids: utterance index of every token, shape ``(T_total,)``.
        cu_seqlens: cumulative token counts, shape ``(B + 1,)``.
        total_tokens: number of tokens.
        padded_total_tokens: length of the token axis the kernel sees, ``total_tokens`` by default;
            the tail beyond ``total_tokens`` belongs to no utterance and its blocks are skipped.

    Returns:
        The block mask over the padded token axis.
    """
    if total_tokens <= 0:
        raise ValueError("document_block_mask needs at least one token.")
    if padded_total_tokens is None:
        padded_total_tokens = total_tokens
    if padded_total_tokens < total_tokens:
        raise ValueError("padded_total_tokens must not be smaller than total_tokens.")
    device = sequence_ids.device
    num_blocks = -(-padded_total_tokens // BLOCK_SIZE)
    # FlexAttention evaluates the mask on the block-rounded grid,
    # so the tail beyond the last token must index safely; the sentinel never equals a real utterance index.
    padded_ids = sequence_ids.new_full((num_blocks * BLOCK_SIZE,), -1)
    padded_ids[:total_tokens] = sequence_ids
    block_starts = torch.arange(num_blocks, device=device) * BLOCK_SIZE
    block_ends = torch.clamp(block_starts + BLOCK_SIZE - 1, max=total_tokens - 1)
    in_range = block_starts < total_tokens
    first_utterance = sequence_ids[torch.clamp(block_starts, max=total_tokens - 1)]
    last_utterance = sequence_ids[block_ends]
    offsets = cu_seqlens.to(torch.int64)
    # the token span of all utterances a query block touches
    span_start = offsets[first_utterance].view(-1, 1)
    span_end = offsets[last_utterance + 1].view(-1, 1)
    intersects = (block_starts.view(1, -1) < span_end) & (block_ends.view(1, -1) >= span_start)
    intersects &= in_range.view(-1, 1) & in_range.view(1, -1)
    # a complete block lies inside one utterance and holds no padded tail tokens;
    # the tail belongs to no utterance and must stay masked, so a partially filled block keeps its mask evaluation
    complete = block_ends == block_starts + BLOCK_SIZE - 1
    one_utterance = (first_utterance == last_utterance) & in_range & complete
    full = one_utterance.view(-1, 1) & one_utterance.view(1, -1) & (first_utterance.view(-1, 1) == first_utterance)
    partial = intersects & ~full

    def ordered(blocks: Tensor) -> tuple[Tensor, Tensor]:
        count = blocks.sum(dim=-1, dtype=torch.int32)
        indices = torch.argsort(blocks.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
        return count.view(1, 1, num_blocks), indices.view(1, 1, num_blocks, num_blocks)

    def mask_mod(b, h, q_idx, kv_idx):
        return padded_ids[q_idx] == padded_ids[kv_idx]

    kv_num_blocks, kv_indices = ordered(partial)
    full_kv_num_blocks, full_kv_indices = ordered(full)
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(padded_total_tokens, padded_total_tokens),
    )


def packed_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    context: PackedAttentionContext,
    *,
    rel_pos_bias: Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> Tensor:
    """Attend within utterances over token-flat queries, keys and values.

    On CUDA, attention without a position bias runs variable-length FlashAttention in half precision when available,
    and everything else runs FlexAttention with the document block mask of the context.
    Elsewhere the utterances are attended one at a time with dense attention,
    which is also the reference of the unit tests.

    Args:
        q: queries, shape ``(T_total, H, d_k)``.
        k: keys, shape ``(T_total, H, d_k)``.
        v: values, shape ``(T_total, H, d_k)``.
        context: packing metadata of the batch.
        rel_pos_bias: optional Transformer-XL position bias, shape ``(T_total, H, 2 * max_seqlen - 1)``,
            whose last axis is indexed by ``max_seqlen - 1 - (query position - key position)``;
            it is scaled like the content scores and added to them.
        dropout_p: dropout on the attention weights,
            supported by the variable-length FlashAttention path and the reference path but not by FlexAttention.
        scale: score scale, ``1 / sqrt(d_k)`` by default.

    Returns:
        Context vectors of shape ``(T_total, H, d_k)``.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if context.total_tokens == 0:
        # keep every projection in the autograd graph when a rank holds no valid token,
        # so distributed gradient reductions see the same parameters on all ranks
        anchor = q.sum() + k.sum()
        if rel_pos_bias is not None:
            anchor = anchor + rel_pos_bias.sum()
        return v + 0.0 * anchor.to(v.dtype)
    if q.is_cuda:
        if rel_pos_bias is None:
            flash = _flash_attention_varlen(q)
            if flash is not None:
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
                return flash(
                    q,
                    k,
                    v,
                    context.cu_seqlens,
                    context.cu_seqlens,
                    context.max_seqlen,
                    context.max_seqlen,
                    dropout_p=dropout_p,
                    softmax_scale=scale,
                    causal=False,
                )
        if dropout_p > 0.0:
            raise NotImplementedError(
                "FlexAttention has no dropout on the attention weights: "
                "set dropout_att: 0 on the encoder for rel_pos attention over packed sequences; "
                "abs_pos and rope attention keep their dropout through variable-length FlashAttention, "
                "which runs in float16 or bfloat16 (autocast) only."
            )
        return _flex_packed_attention(q, k, v, context, rel_pos_bias=rel_pos_bias, scale=scale)
    return _reference_packed_attention(q, k, v, context, rel_pos_bias=rel_pos_bias, dropout_p=dropout_p, scale=scale)


def _flash_attention_varlen(q: Tensor) -> Callable | None:
    """The variable-length FlashAttention provider for bias-free attention, or None when it does not apply."""
    # a function-level import: the modules package initializes the encoders, which import this module
    from nemo.collections.asr.modules.transformer_encoder_utils import _select_flash_attention_varlen

    head_dim = q.shape[-1]
    return _select_flash_attention_varlen(q, static_eligible=head_dim <= 256 and head_dim % 8 == 0)


def _flex_packed_attention(
    q: Tensor, k: Tensor, v: Tensor, context: PackedAttentionContext, *, rel_pos_bias: Tensor | None, scale: float
) -> Tensor:
    total_tokens = context.total_tokens
    padded_total = context.padded_total_tokens
    tail = padded_total - total_tokens

    def pad_tokens(tensor: Tensor) -> Tensor:
        return torch.nn.functional.pad(tensor, (0, 0) * (tensor.dim() - 1) + (0, tail)) if tail else tensor

    score_mod = None
    if rel_pos_bias is not None:
        bias = rel_pos_bias
        rows = context.bias_rows()
        # the bias covers the relative positions (W - 1) / 2 down to -(W - 1) / 2;
        # indices outside an utterance are masked, so clamping only keeps the gather in bounds,
        # and the rows map the padded token axis onto the bias, which is not padded
        center = (bias.shape[-1] - 1) // 2
        last_index = bias.shape[-1] - 1

        def score_mod(score, b, h, q_idx, kv_idx):
            rel = (center - q_idx + kv_idx).clamp(0, last_index)
            return score + bias[rows[q_idx], h, rel] * scale

    compiled = _compiled_flex_attention()
    with _recompile_limit_patch():
        out = compiled(
            pad_tokens(q).transpose(0, 1).unsqueeze(0),
            pad_tokens(k).transpose(0, 1).unsqueeze(0),
            pad_tokens(v).transpose(0, 1).unsqueeze(0),
            score_mod=score_mod,
            block_mask=context.block_mask(),
            scale=scale,
        )
    return out.squeeze(0).transpose(0, 1)[:total_tokens]


def _reference_packed_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    context: PackedAttentionContext,
    *,
    rel_pos_bias: Tensor | None,
    dropout_p: float,
    scale: float,
) -> Tensor:
    """Dense attention one utterance at a time, with the same bias convention as the kernels."""
    outputs = []
    center = (rel_pos_bias.shape[-1] - 1) // 2 if rel_pos_bias is not None else 0
    for start, end in context.sequence_slices:
        if end == start:
            continue
        qi = q[start:end].transpose(0, 1)
        ki = k[start:end].transpose(0, 1)
        vi = v[start:end].transpose(0, 1)
        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale
        if rel_pos_bias is not None:
            length = end - start
            rows = torch.arange(length, device=q.device).view(length, 1)
            cols = torch.arange(length, device=q.device).view(1, length)
            # advanced indices around the head slice move the heads to the last axis
            bias = rel_pos_bias[start:end][rows, :, center - rows + cols]
            scores = scores + bias.permute(2, 0, 1) * scale
        weights = torch.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            weights = torch.nn.functional.dropout(weights, p=dropout_p, training=True)
        outputs.append(torch.matmul(weights, vi).transpose(0, 1))
    return torch.cat(outputs, dim=0)
