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

"""Self- and cross-attention over sequence-packed activations with explicit per-sequence boundaries."""

import functools
import math
from collections.abc import Callable

import torch
import torch._dynamo
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from nemo.collections.asr.parts.packed_sequence import (
    PackedEncoderActivations,
    lengths_to_cu_seqlens,
    packed_encoder_position_ids,
    packed_sequence_ids,
)
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.triton.rel_pos_attention import (
        PACKED_REL_POS_ATTENTION_DTYPES,
        packed_rel_pos_attention,
    )

__all__ = [
    "PackedAttentionContext",
    "document_block_mask",
    "packed_attention",
    "packed_attention_supports_dropout",
    "packed_cross_attention",
]

# the sparse block size of FlexAttention; the document block mask is built at this granularity
BLOCK_SIZE = 128
# The head width the FlexAttention kernels see is rounded up to a power of two from this floor,
# the smallest width their Triton kernels take (tl.dot); the zero features change nothing and the output is sliced.
# The kernels round the width to the next power of two internally anyway, and torch's tuned kernel configurations
# are keyed by the powers of two 64, 128 and 256: any other width falls to a generic configuration
# whose shared memory exceeds the 101 KB of consumer Blackwell GPUs (torch 2.12, width 44, the FastConformer Small head).
FLEX_MIN_HEAD_DIM = 16
# FlexAttention is the relative-position path without Triton
# (with it, nemo.collections.asr.parts.triton.rel_pos_attention runs; float64 takes the reference path).
# Its kernels are compiled statically per shape
# (a fully dynamic compile cannot trace the computed gather of the relative-position bias;
# only the bias's token axis is marked dynamic, see _flex_packed_attention).
# The token axis and the relative-position width the kernels see are rounded up to powers of two
# from these floors, so the kernels of a run are bounded by the logarithms of the two ranges,
# not by the number of distinct batches, and the recompile limit below leaves room for them.
# Reaching the limit anyway is an error, not a fallback: past its limit Dynamo runs a function eagerly,
# and eager flex_attention materializes the dense scores over the padded token axis.
TOKEN_PADDING = 4096
LENGTH_PADDING = 128
RECOMPILE_LIMIT = 64


class PackedAttentionContext:
    """Packing metadata of one batch, shared by all attention layers of a packed forward.

    On CUDA with Triton the relative-position attention runs the kernel of
    :mod:`nemo.collections.asr.parts.triton.rel_pos_attention` over the utterance boundaries as they are.
    Without it, or with ``rel_pos_kernel`` off, FlexAttention runs with a block mask derived from the boundaries,
    without evaluating a mask over the dense token grid:
    a key block is complete for a query block when both lie inside the same utterance,
    partial when it intersects an utterance that the query block touches,
    and skipped otherwise.
    A limited attention context (``key_window``) narrows the keys of every token inside its utterance,
    and the block mask skips the key blocks outside of it.

    Args:
        packed: the packed activations entering the encoder layers; only their sequence metadata is used.
        key_window: a limited attention context, or None for the whole utterance:
            the first and the last key position that a query at each position of an utterance attends to,
            two integer tensors of shape ``(max_seqlen,)`` indexed by the query position, on any device,
            either of them None for a side without a limit (both None is the whole utterance).
            Positions beyond the utterance are cut off, and a token always attends to itself.
        rel_pos_kernel: whether relative-position attention runs the Triton kernel where it can (the default),
            or FlexAttention on CUDA whatever the dtype, the backend before the kernel,
            which computes in float32 and has no dropout on the attention weights.
            The kernel knows the utterance boundaries only, so a context with a key window never runs it.
    """

    def __init__(
        self,
        packed: PackedEncoderActivations,
        *,
        key_window: tuple[Tensor | None, Tensor | None] | None = None,
        rel_pos_kernel: bool = True,
    ):
        self.lengths = packed.lengths
        self.cu_seqlens = packed.cu_seqlens
        self.max_seqlen = packed.max_seqlen
        self.total_tokens = packed.total_tokens
        self.device = packed.data.device
        self.sequence_ids = packed_sequence_ids(packed)
        self.position_ids = packed_encoder_position_ids(packed)
        # the shapes the compiled FlexAttention kernels see, see TOKEN_PADDING and LENGTH_PADDING
        self.padded_total_tokens = _padded_size(self.total_tokens, TOKEN_PADDING)
        self.padded_max_seqlen = _padded_size(self.max_seqlen, LENGTH_PADDING)
        if key_window is not None:
            first_key, last_key = key_window
            for bound in (first_key, last_key):
                if bound is None:
                    continue
                if bound.shape != (self.max_seqlen,):
                    raise ValueError(
                        f"key_window bounds must have shape ({self.max_seqlen},), one entry per position "
                        f"of the longest utterance, got {tuple(bound.shape)}."
                    )
                if bound.dtype == torch.bool or bound.is_floating_point() or bound.is_complex():
                    raise TypeError(f"key_window bounds must be integer tensors, got {bound.dtype}.")
            if first_key is None and last_key is None:
                key_window = None
        self.key_window = key_window
        # whether the window limits both sides, so that no key is further from its query than the window reaches
        self.key_window_is_bounded = key_window is not None and key_window[0] is not None and key_window[1] is not None
        # the Triton kernel reads the relative-position table of the longest utterance (at least one row),
        # FlexAttention one of its rounded length; a key window is FlexAttention's, the kernel has none
        self.rel_pos_kernel = (
            rel_pos_kernel and TRITON_AVAILABLE and self.device.type == "cuda" and self.key_window is None
        )
        self.rel_pos_max_seqlen = max(self.max_seqlen, 1) if self.rel_pos_kernel else self.padded_max_seqlen
        self._block_mask = None
        self._bias_rows = None
        self._sequence_slices = None
        self._seq_lens = None
        self._key_spans = None

    @classmethod
    def from_lengths(cls, lengths: Tensor, *, rel_pos_kernel: bool = True) -> "PackedAttentionContext":
        """Metadata for token-flat ids or states whose sequences have these lengths, in packing order."""
        lengths = lengths.to(torch.int64)
        total_tokens = int(lengths.sum().item())
        max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
        placeholder = lengths.new_empty((total_tokens, 0), dtype=torch.float32)
        packed = PackedEncoderActivations.from_validated(
            placeholder, lengths, lengths_to_cu_seqlens(lengths), max_seqlen
        )
        return cls(packed, rel_pos_kernel=rel_pos_kernel)

    @property
    def sequence_slices(self) -> tuple[tuple[int, int], ...]:
        """Host copies of the utterance boundaries, for the per-utterance reference path."""
        if self._sequence_slices is None:
            offsets = self.cu_seqlens.tolist()
            self._sequence_slices = tuple(zip(offsets[:-1], offsets[1:]))
        return self._sequence_slices

    def key_spans(self) -> tuple[Tensor, Tensor]:
        """The keys of every token as a span ``[start, end)`` of the token axis, two tensors of shape ``(T_total,)``.

        The utterance of the token, narrowed to the key window when the context has one.
        """
        if self._key_spans is None:
            offsets = self.cu_seqlens.to(torch.int64)
            first = offsets[self.sequence_ids]
            start, end = first, offsets[self.sequence_ids + 1]
            if self.key_window is not None:
                first_key, last_key = self.key_window
                positions = self.position_ids
                if first_key is not None:
                    start = first + torch.minimum(first_key.to(self.device)[positions], positions).clamp(min=0)
                if last_key is not None:
                    end = torch.minimum(first + torch.maximum(last_key.to(self.device)[positions], positions) + 1, end)
            self._key_spans = (start, end)
        return self._key_spans

    @property
    def seq_starts(self) -> Tensor:
        """First token of every utterance, shape ``(B,)`` int32, for the Triton kernel."""
        return self.cu_seqlens[:-1]

    @property
    def seq_lens(self) -> Tensor:
        """Tokens of every utterance, shape ``(B,)`` int32, for the Triton kernel."""
        if self._seq_lens is None:
            self._seq_lens = self.lengths.to(torch.int32)
        return self._seq_lens

    def block_mask(self) -> BlockMask:
        """The block mask of the utterances and the key window over the padded token axis, built once per batch."""
        if self._block_mask is None:
            self._block_mask = document_block_mask(
                self.sequence_ids,
                self.cu_seqlens,
                self.total_tokens,
                padded_total_tokens=self.padded_total_tokens,
                key_spans=self.key_spans(),
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
    sequence_ids: Tensor,
    cu_seqlens: Tensor,
    total_tokens: int,
    padded_total_tokens: int | None = None,
    key_spans: tuple[Tensor, Tensor] | None = None,
) -> BlockMask:
    """Build the FlexAttention block mask that confines attention to the utterance of every token, or to its key span.

    Args:
        sequence_ids: utterance index of every token, shape ``(T_total,)``.
        cu_seqlens: cumulative token counts, shape ``(B + 1,)``.
        total_tokens: number of tokens.
        padded_total_tokens: length of the token axis the kernel sees, ``total_tokens`` by default;
            the tail beyond ``total_tokens`` belongs to no utterance and its blocks are skipped.
        key_spans: a narrower attention context than the utterance: the keys of every token
            as a span ``[start, end)`` of the token axis inside its utterance, two tensors of shape ``(T_total,)``
            (see ``PackedAttentionContext.key_spans``, which builds them; they are not checked here);
            the whole utterance by default.

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
    if key_spans is None:
        offsets = cu_seqlens.to(torch.int64)
        key_spans = (offsets[sequence_ids], offsets[sequence_ids + 1])
    # FlexAttention evaluates the mask on the block-rounded grid, so the tail beyond the last token
    # must index safely; the tail attends to itself, which no token's span reaches.
    tail = num_blocks * BLOCK_SIZE - total_tokens
    span_starts = torch.nn.functional.pad(key_spans[0], (0, tail), value=total_tokens)
    span_ends = torch.nn.functional.pad(key_spans[1], (0, tail), value=num_blocks * BLOCK_SIZE)
    # per query block the spans of its tokens; the tail has none, so it widens no hull and completes no block
    block_span_starts = torch.nn.functional.pad(key_spans[0], (0, tail), value=num_blocks * BLOCK_SIZE)
    block_span_starts = block_span_starts.view(num_blocks, BLOCK_SIZE)
    block_span_ends = torch.nn.functional.pad(key_spans[1], (0, tail), value=0).view(num_blocks, BLOCK_SIZE)
    block_starts = (torch.arange(num_blocks, device=device) * BLOCK_SIZE).view(1, -1)
    block_ends = block_starts + BLOCK_SIZE
    # a key block is needed when it intersects the hull of the key spans of a query block,
    intersects = (block_starts < block_span_ends.amax(dim=1, keepdim=True)) & (
        block_ends > block_span_starts.amin(dim=1, keepdim=True)
    )
    # and complete when every query of the block attends to all of its keys;
    # no span reaches into the tail, so a partially filled block keeps its mask evaluation, as queries and as keys
    full = (block_span_starts.amax(dim=1, keepdim=True) <= block_starts) & (
        block_span_ends.amin(dim=1, keepdim=True) >= block_ends
    )
    partial = intersects & ~full

    def ordered(blocks: Tensor) -> tuple[Tensor, Tensor]:
        count = blocks.sum(dim=-1, dtype=torch.int32)
        indices = torch.argsort(blocks.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
        return count.view(1, 1, num_blocks), indices.view(1, 1, num_blocks, num_blocks)

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx >= span_starts[q_idx]) & (kv_idx < span_ends[q_idx])

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
    pos_query: Tensor | None = None,
    pos_emb: Tensor | None = None,
    rel_pos_bias: Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> Tensor:
    """Attend within utterances over token-flat queries, keys and values.

    On CUDA, attention without a position term over the whole utterance runs variable-length FlashAttention
    in half precision when available,
    and attention over the whole utterance with the Transformer-XL position term given by its factors
    runs the Triton kernel of :mod:`nemo.collections.asr.parts.triton.rel_pos_attention`
    (float32, bfloat16 and float16, with dropout) where the context selects it (``context.rel_pos_kernel``);
    everything else, the materialized ``rel_pos_bias`` and a limited attention context (``context.key_window``)
    included, runs FlexAttention with the block mask of the context.
    Elsewhere, and for float64 (FlexAttention's kernels compute in float32 at most),
    the utterances are attended one at a time with dense attention,
    which is also the reference of the unit tests.

    Args:
        q: queries, shape ``(T_total, H, d_k)``.
        k: keys, shape ``(T_total, H, d_k)``.
        v: values, shape ``(T_total, H, d_k)``.
        context: packing metadata of the batch.
        pos_query: queries with their position bias added, shape ``(T_total, H, d_k)``;
            with ``pos_emb`` the Transformer-XL position term ``pos_query[i] . pos_emb[c - (i - j)]``
            of query position ``i`` and key position ``j`` is scaled like the content scores and added to them;
            both or neither.
        pos_emb: projected relative position embedding, shape ``(R, H, d_k)`` with odd ``R``,
            whose row ``c = (R - 1) / 2`` is the relative position zero;
            it covers at least ``context.rel_pos_max_seqlen - 1`` positions each way.
            With a key window bounded on both sides a shorter one serves, provided no key is further
            than ``c`` from its query, which is the caller's contract and not checked (it would take a device read).
        rel_pos_bias: the position term materialized over the relative positions instead of its factors,
            shape ``(T_total, H, R)`` with odd ``R`` and the row convention of ``pos_emb``
            (``R = 2 * max_seqlen - 1`` indexes it by ``max_seqlen - 1 - (i - j)``);
            the form of the FlexAttention and reference paths, which it keeps on.
        dropout_p: dropout on the attention weights, supported by every path but FlexAttention
            (see ``packed_attention_supports_dropout``).
        scale: score scale, ``1 / sqrt(d_k)`` by default.

    Returns:
        Context vectors of shape ``(T_total, H, d_k)``.
    """
    if (pos_query is None) != (pos_emb is None):
        raise ValueError("pos_query and pos_emb come together: give both or neither.")
    if rel_pos_bias is not None and pos_query is not None:
        raise ValueError("The position term comes either as rel_pos_bias or as pos_query and pos_emb, not both.")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    rel_positions = pos_emb.shape[0] if pos_emb is not None else None
    if rel_pos_bias is not None:
        rel_positions = rel_pos_bias.shape[-1]
    if rel_positions is not None and not context.key_window_is_bounded and rel_positions < 2 * context.max_seqlen - 1:
        raise ValueError(
            f"The position term covers {rel_positions} relative positions, but the utterances reach "
            f"{2 * context.max_seqlen - 1} (2 * max_seqlen - 1); only a key window bounded on both sides "
            "takes a narrower one."
        )
    if context.total_tokens == 0:
        # keep every projection in the autograd graph when a rank holds no valid token,
        # so distributed gradient reductions see the same parameters on all ranks
        anchor = q.sum() + k.sum()
        if pos_query is not None:
            anchor = anchor + pos_query.sum() + pos_emb.sum()
        if rel_pos_bias is not None:
            anchor = anchor + rel_pos_bias.sum()
        return v + 0.0 * anchor.to(v.dtype)
    if q.is_cuda and q.dtype != torch.float64:
        # variable-length FlashAttention knows the utterance boundaries only, not a window inside them
        if pos_query is None and rel_pos_bias is None and context.key_window is None:
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
        elif pos_query is not None and context.rel_pos_kernel and q.dtype in PACKED_REL_POS_ATTENTION_DTYPES:
            # the context's lengths and max_seqlen agree by construction: no device read for the check
            return packed_rel_pos_attention(
                q,
                k,
                v,
                pos_query,
                pos_emb,
                context.seq_starts,
                context.seq_lens,
                context.max_seqlen,
                dropout_p=dropout_p,
                scale=scale,
                check_lengths=False,
            )
        if dropout_p > 0.0:
            raise NotImplementedError(
                "FlexAttention has no dropout on the attention weights. "
                "rel_pos attention keeps its dropout through its Triton kernel "
                "(Triton installed, float32, bfloat16 or float16), "
                "abs_pos and rope attention through variable-length FlashAttention in float16 or bfloat16 (autocast), "
                "both with the full attention context only; "
                "otherwise, and for a limited attention context, set dropout_att: 0 on the encoder for packed sequences."
            )
        return _flex_packed_attention(
            q, k, v, context, rel_pos_bias=_materialized(pos_query, pos_emb, rel_pos_bias), scale=scale
        )
    return _reference_packed_attention(
        q,
        k,
        v,
        context,
        rel_pos_bias=_materialized(pos_query, pos_emb, rel_pos_bias),
        dropout_p=dropout_p,
        scale=scale,
    )


def packed_attention_supports_dropout(
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    *,
    rel_pos: bool = False,
    rel_pos_kernel: bool = True,
    limited_context: bool = False,
) -> bool:
    """Whether ``packed_attention`` applies dropout on this device, dtype and head width: the dispatch rule.

    Off CUDA, and on CUDA in float64, it always does (the dense reference path).
    On CUDA without a position term, only where variable-length FlashAttention runs:
    float16 or bfloat16, a head width that is a multiple of 8 up to 256, compute capability 8 or newer.
    With the Transformer-XL position term (``rel_pos``), only where its Triton kernel runs:
    selected (``rel_pos_kernel``, the switch of the ``PackedAttentionContext``), Triton installed,
    in float32, bfloat16 or float16.
    FlexAttention, the path otherwise and the path of every limited attention context (``limited_context``),
    has no dropout.
    """
    if device.type != 'cuda' or dtype == torch.float64:
        return True
    if limited_context:
        return False
    if rel_pos:
        return rel_pos_kernel and TRITON_AVAILABLE and dtype in PACKED_REL_POS_ATTENTION_DTYPES
    if dtype not in (torch.float16, torch.bfloat16) or head_dim > 256 or head_dim % 8 != 0:
        return False
    # a function-level import: the modules package initializes the encoders, which import this module
    from nemo.collections.asr.modules.transformer_encoder_utils import _get_flash_attention_varlen_for_device

    return _get_flash_attention_varlen_for_device(device) is not None


def packed_cross_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    query_context: PackedAttentionContext,
    key_context: PackedAttentionContext,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> Tensor:
    """Attend from token-flat queries to the token-flat keys of the same sequence, without a position bias.

    On CUDA this runs variable-length FlashAttention and raises when that does not apply
    (float32, a head width above 256 or not divisible by 8, a GPU before Ampere):
    the dense attention below would be several times slower than the padded attention.
    On the CPU the sequences are attended one at a time with dense attention,
    which is also the reference of the unit tests.

    Args:
        q: queries, shape ``(Q_total, H, d_k)``.
        k: keys, shape ``(K_total, H, d_k)``, packed in the same sequence order as the queries.
        v: values, shape ``(K_total, H, d_k)``.
        query_context: packing metadata of the queries.
        key_context: packing metadata of the keys; the same object as ``query_context`` in self-attention.
        causal: let every query see only the keys at its position or before; needs equal query and key lengths.
        dropout_p: dropout on the attention weights, in ``[0, 1)``.
        scale: positive score scale, ``1 / sqrt(d_k)`` by default.

    Returns:
        Context vectors of shape ``(Q_total, H, d_k)``.

    Raises:
        ValueError: for inputs that do not match their contexts and for scalars outside their ranges,
            checked before the dispatch so that every backend rejects the same calls.
        NotImplementedError: on CUDA when variable-length FlashAttention does not apply.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}.")
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(f"scale must be positive, got {scale}.")
    if query_context.lengths.numel() != key_context.lengths.numel():
        raise ValueError("query_context and key_context must describe the same number of sequences.")
    if q.shape[0] != query_context.total_tokens:
        raise ValueError(f"q has {q.shape[0]} rows, query_context describes {query_context.total_tokens} tokens.")
    if k.shape[0] != key_context.total_tokens or v.shape[0] != key_context.total_tokens:
        raise ValueError(
            f"k and v have {k.shape[0]} and {v.shape[0]} rows, key_context describes {key_context.total_tokens} tokens."
        )
    if causal and query_context is not key_context and not torch.equal(query_context.lengths, key_context.lengths):
        raise ValueError("causal packed attention needs equal query and key lengths per sequence.")
    if query_context.total_tokens == 0 or key_context.total_tokens == 0:
        # empty-slice sums keep the gradient edges to q, k and v without summing the inputs,
        # which overflows in float16 and turns the zero anchor into NaN
        anchor = q[:0].sum() + k[:0].sum() + v[:0].sum()
        return q.new_zeros(q.shape) + anchor.to(q.dtype)
    if q.is_cuda:
        flash = _flash_attention_varlen(q)
        if flash is None:
            raise NotImplementedError(
                "packed cross attention on CUDA runs variable-length FlashAttention only, which needs float16 or "
                "bfloat16 (train under autocast), a head width of at most 256 and divisible by 8, and a GPU of "
                f"compute capability 8.0 or newer; got {q.dtype} with head width {q.shape[-1]} on "
                f"{torch.cuda.get_device_name(q.device)}."
            )
        return flash(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            query_context.cu_seqlens,
            key_context.cu_seqlens,
            query_context.max_seqlen,
            key_context.max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=causal,
        )
    return _reference_packed_cross_attention(
        q, k, v, query_context, key_context, causal=causal, dropout_p=dropout_p, scale=scale
    )


def _padded_size(value: int, floor: int) -> int:
    """Round ``value`` up to a power of two, at least ``floor`` (a power of two itself)."""
    return max(floor, 1 << max(value - 1, 0).bit_length())


@functools.cache
def _compiled_flex_attention() -> Callable:
    """FlexAttention compiled statically per shape, at the first call so that importing the encoders stays cheap.

    The compile goes through a function of this module, not through ``flex_attention`` itself:
    Dynamo keeps a compiled function's entries on its code object and counts them against the recompile limit there,
    so the kernels of this module would otherwise share the cache, and the limit, with every other compile
    of ``flex_attention`` in the process (the Transformer encoders compile it with dynamic shapes),
    and the entries of one site would send the other to eager FlexAttention.
    """

    def packed_flex_attention(q, k, v, score_mod, block_mask, scale):
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, scale=scale)

    return torch.compile(packed_flex_attention, dynamic=False)


def _recompile_limit_patch():
    """Dynamo settings for the per-shape FlexAttention kernels: room for RECOMPILE_LIMIT shapes, an error beyond."""
    # torch renamed `cache_size_limit` to `recompile_limit`; older versions have only the old name
    name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    settings = {name: max(getattr(torch._dynamo.config, name), RECOMPILE_LIMIT)}
    if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        settings["fail_on_recompile_limit_hit"] = True
    return torch._dynamo.config.patch(**settings)


def _flash_attention_varlen(q: Tensor) -> Callable | None:
    """The variable-length FlashAttention provider for bias-free attention, or None when it does not apply."""
    # a function-level import: the modules package initializes the encoders, which import this module
    from nemo.collections.asr.modules.transformer_encoder_utils import _select_flash_attention_varlen

    head_dim = q.shape[-1]
    # the same width rule as packed_attention_supports_dropout
    return _select_flash_attention_varlen(q, static_eligible=head_dim <= 256 and head_dim % 8 == 0)


def _materialized(pos_query: Tensor | None, pos_emb: Tensor | None, rel_pos_bias: Tensor | None) -> Tensor | None:
    """The unscaled position term over all relative positions, shape ``(T_total, H, R)``, from either form."""
    if pos_query is None:
        return rel_pos_bias
    return torch.einsum("thd,rhd->thr", pos_query, pos_emb)


def _flex_packed_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    context: PackedAttentionContext,
    *,
    rel_pos_bias: Tensor | None,
    scale: float,
) -> Tensor:
    total_tokens = context.total_tokens
    padded_total = context.padded_total_tokens
    tail = padded_total - total_tokens
    head_dim = q.shape[-1]
    # the power-of-two head width the kernels are tuned for, see FLEX_MIN_HEAD_DIM;
    # zero features change neither the scores nor the context vectors, and the output is sliced back
    width_pad = _padded_size(head_dim, FLEX_MIN_HEAD_DIM) - head_dim

    def pad_tokens(tensor: Tensor) -> Tensor:
        return torch.nn.functional.pad(tensor, (0, width_pad, 0, 0, 0, tail)) if tail or width_pad else tensor

    score_mod = None
    if rel_pos_bias is not None:
        bias = rel_pos_bias
        rows = context.bias_rows()
        # the bias keeps its token axis (no padded copy), which the static compile would otherwise
        # specialize on, one kernel per distinct token total of a run; the mark keeps that axis symbolic
        torch._dynamo.maybe_mark_dynamic(bias, 0)
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
    return out.squeeze(0).transpose(0, 1)[:total_tokens, :, :head_dim]


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
    """Dense attention one utterance at a time, with the same position-term convention as the kernels."""
    outputs = []
    center = (rel_pos_bias.shape[-1] - 1) // 2 if rel_pos_bias is not None else 0
    span_starts, span_ends = context.key_spans() if context.key_window is not None else (None, None)
    for start, end in context.sequence_slices:
        if end == start:
            continue
        qi = q[start:end].transpose(0, 1)
        ki = k[start:end].transpose(0, 1)
        vi = v[start:end].transpose(0, 1)
        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale
        length = end - start
        rows = torch.arange(length, device=q.device).view(length, 1)
        cols = torch.arange(length, device=q.device).view(1, length)
        if rel_pos_bias is not None:
            # relative positions beyond a narrow bias lie outside the key window and are masked below;
            # advanced indices around the head slice move the heads to the last axis
            rel = (center - rows + cols).clamp(0, rel_pos_bias.shape[-1] - 1)
            bias = rel_pos_bias[start:end][rows, :, rel]
            scores = scores + bias.permute(2, 0, 1) * scale
        if span_starts is not None:
            keys = cols + start
            outside = (keys < span_starts[start:end].view(length, 1)) | (keys >= span_ends[start:end].view(length, 1))
            scores = scores.masked_fill(outside, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            weights = torch.nn.functional.dropout(weights, p=dropout_p, training=True)
        outputs.append(torch.matmul(weights, vi).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _reference_packed_cross_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    query_context: PackedAttentionContext,
    key_context: PackedAttentionContext,
    *,
    causal: bool,
    dropout_p: float,
    scale: float,
) -> Tensor:
    """Dense attention one sequence at a time from its queries to its keys."""
    outputs = []
    for (q_start, q_end), (k_start, k_end) in zip(query_context.sequence_slices, key_context.sequence_slices):
        if q_end == q_start:
            continue
        if k_end == k_start:
            # a zero context that keeps the gradient edges to q, k and v, like the empty-batch case
            anchor = q[:0].sum() + k[:0].sum() + v[:0].sum()
            outputs.append(q.new_zeros((q_end - q_start,) + tuple(q.shape[1:])) + anchor)
            continue
        qi = q[q_start:q_end].transpose(0, 1)
        ki = k[k_start:k_end].transpose(0, 1)
        vi = v[k_start:k_end].transpose(0, 1)
        # scaled before the product like the padded attention, so that half-precision scores stay finite
        root = math.sqrt(scale)
        scores = torch.matmul(qi * root, (ki * root).transpose(-2, -1))
        if causal:
            visible = torch.ones(q_end - q_start, k_end - k_start, dtype=torch.bool, device=q.device).tril()
            scores = scores.masked_fill(~visible, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            weights = torch.nn.functional.dropout(weights, p=dropout_p, training=True)
        outputs.append(torch.matmul(weights, vi).transpose(0, 1))
    return torch.cat(outputs, dim=0)
