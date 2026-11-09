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

"""Transformer-XL self-attention over sequence-packed activations, with dropout on the attention weights.

The utterances of a batch are concatenated along the token axis;
``seq_starts`` and ``seq_lens`` say where each one lies, so gaps between them are allowed,
and a program of the kernels never reads or writes a row outside its utterance.
Queries, keys and values have shape ``(T_total, H, D)``.
The score of query position ``i`` and key position ``j`` of one utterance is
``scale * q_i . k_j + scale * pos_query_i . pos_emb[c - (i - j)]`` with ``c = max_seqlen - 1``,
the convention of ``RelPositionMultiHeadAttention`` (``rel_shift`` selects the same entries in the padded forward).

The position term ``(T_total, H, 2 * max_seqlen - 1)`` is computed from ``pos_query`` and ``pos_emb``
inside the custom ops, in the forward and again in the backward,
so it is never saved for the backward and no layer retains one;
the kernels read it through 64-bit offsets, since ``H * T_total * (2 * max_seqlen - 1)`` passes 2**31
at large packed batches.
Its buffer is ``(H, T_total, R)`` with ``R`` rounded up to a multiple of eight,
because an odd leading dimension sends the GEMMs to slow unaligned kernels;
the kernels take the strides of the ``(T_total, H, R)`` view.

Dropout on the attention weights is a Philox hash of the global query row and the local key column,
with the head mixed into the seed, so the backward recomputes the mask instead of storing it.
The seed is a one-element device tensor read inside the kernels:
drawn on the device for every call, so a CUDA-graph replay draws a fresh mask,
or given by the caller for a reproducible run.
The gradient of the position term is row-owned (every query row writes its own row), without atomics,
so the backward is deterministic.

One head dimension serves queries, keys, values and the position embedding, as in RETURNN's kernel
(``returnn/torch/util/rel_pos_att_triton.py`` of https://github.com/rwth-i6/returnn), which this follows.
The backward recomputes ``delta = sum_j p_ij dp_ij`` in a pass of its own instead of the flash-attention shortcut
``rowsum(out * d_out)``, which holds only for an exactly accumulated ``out``;
with bf16 storage the shortcut's rounding error enters the position-term gradient as a per-row bias.
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["PACKED_REL_POS_ATTENTION_DTYPES", "packed_rel_pos_attention"]

# the dtypes the kernels read and write; the softmax and the accumulations run in float32 either way
PACKED_REL_POS_ATTENTION_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def packed_rel_pos_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pos_query: Tensor,
    pos_emb: Tensor,
    seq_starts: Tensor,
    seq_lens: Tensor,
    max_seqlen: int,
    *,
    dropout_p: float = 0.0,
    dropout_seed: Tensor | None = None,
    scale: float | None = None,
) -> Tensor:
    """Attend within utterances over token-flat queries, keys and values with the Transformer-XL position term.

    Args:
        q: queries with the content bias added, shape ``(T_total, H, D)``; float32, bfloat16 or float16.
        k: keys, shape ``(T_total, H, D)``, dtype of ``q``.
        v: values, shape ``(T_total, H, D)``, dtype of ``q``.
        pos_query: queries with the position bias added, shape ``(T_total, H, D)``, dtype of ``q``.
        pos_emb: projected relative position embedding, shape ``(R, H, D)`` with odd ``R``, dtype of ``q``;
            row ``(R - 1) / 2 - d`` holds the relative position ``d``,
            so the table must cover at least ``max_seqlen - 1`` positions each way (``R >= 2 * max_seqlen - 1``).
        seq_starts: first row of every utterance, shape ``(B,)``, integer.
        seq_lens: rows of every utterance, shape ``(B,)``, integer; the rows between the utterances are never read.
        max_seqlen: longest utterance, a host integer; the lengths are not checked against it (no device read),
            and an utterance beyond it attends within ``max_seqlen`` positions with its later query rows left zero.
        dropout_p: dropout probability on the attention weights, below one.
        dropout_seed: one-element integer tensor with the Philox seed for a reproducible mask;
            drawn on the device of ``q`` by default, which gives a fresh mask on every CUDA-graph replay.
        scale: factor of both score terms, ``1 / sqrt(D)`` by default.

    Returns:
        Context vectors of shape ``(T_total, H, D)`` and the dtype of ``q``; rows outside every utterance are zero.
    """
    if q.ndim != 3:
        raise ValueError(f"q must have shape (T_total, H, D), got {tuple(q.shape)}.")
    if not q.is_cuda:
        raise RuntimeError("The packed relative-position attention kernel requires CUDA tensors.")
    if q.dtype not in PACKED_REL_POS_ATTENTION_DTYPES:
        raise TypeError(f"q must be float32, bfloat16 or float16, got {q.dtype}.")
    for name, tensor in (("k", k), ("v", v), ("pos_query", pos_query)):
        if tensor.shape != q.shape:
            raise ValueError(f"{name} must have the shape of q {tuple(q.shape)}, got {tuple(tensor.shape)}.")
    for name, tensor in (("k", k), ("v", v), ("pos_query", pos_query), ("pos_emb", pos_emb)):
        if tensor.dtype != q.dtype or tensor.device != q.device:
            raise TypeError(f"{name} must be {q.dtype} on {q.device}, got {tensor.dtype} on {tensor.device}.")
    total, heads, head_dim = q.shape
    if pos_emb.ndim != 3 or pos_emb.shape[1:] != (heads, head_dim):
        raise ValueError(f"pos_emb must have shape (R, {heads}, {head_dim}), got {tuple(pos_emb.shape)}.")
    width = pos_emb.shape[0]
    if width % 2 == 0:
        raise ValueError(f"pos_emb must have an odd number of rows, got {width}.")
    if max_seqlen < 0 or width < 2 * max_seqlen - 1:
        raise ValueError(f"pos_emb with {width} rows covers no utterance of max_seqlen {max_seqlen} frames.")
    if seq_starts.shape != seq_lens.shape or seq_starts.ndim != 1:
        raise ValueError(
            f"seq_starts and seq_lens must have shape (B,), got {tuple(seq_starts.shape)} and {tuple(seq_lens.shape)}."
        )
    for name, tensor in (("seq_starts", seq_starts), ("seq_lens", seq_lens)):
        if tensor.dtype not in (torch.int32, torch.int64) or tensor.device != q.device:
            raise TypeError(f"{name} must be int32 or int64 on {q.device}, got {tensor.dtype} on {tensor.device}.")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}.")
    if total * heads * head_dim >= 2**31:
        raise ValueError(
            f"q with {total} x {heads} x {head_dim} elements exceeds the 32-bit row offsets of the kernels."
        )
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    if total == 0 or max_seqlen == 0:
        return torch.zeros_like(q)
    if width > 2 * max_seqlen - 1:
        center = (width - 1) // 2
        pos_emb = pos_emb[center - (max_seqlen - 1) : center + max_seqlen]
    if dropout_seed is not None and (
        dropout_seed.numel() != 1 or dropout_seed.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            f"dropout_seed must be one integer, got {dropout_seed.dtype} of shape {tuple(dropout_seed.shape)}."
        )
    if dropout_seed is not None:
        dropout_seed = dropout_seed.to(device=q.device, dtype=torch.int32)
    elif dropout_p > 0.0:
        dropout_seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32, device=q.device)
    else:
        dropout_seed = torch.zeros((1,), dtype=torch.int32, device=q.device)
    out, _ = torch.ops.nemo_asr.packed_rel_pos_attention_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        pos_query.contiguous(),
        pos_emb.contiguous(),
        seq_starts.to(torch.int32),
        seq_lens.to(torch.int32),
        max_seqlen,
        dropout_p,
        dropout_seed,
        scale,
    )
    return out


def _keep_mask(
    total: int, heads: int, max_seqlen: int, *, dropout_p: float, dropout_seed: Tensor, device: torch.device
) -> Tensor:
    """The attention weights the kernels keep under dropout, for exact comparisons with a dense reference.

    Args:
        total: rows of the packed batch.
        heads: attention heads.
        max_seqlen: longest utterance.
        dropout_p: dropout probability on the attention weights.
        dropout_seed: the one-element integer seed tensor the kernels ran with.
        device: the CUDA device.

    Returns:
        Boolean mask of shape ``(T_total, H, max_seqlen)``:
        entry ``(i, h, j)`` says whether query row ``i`` keeps the weight of the key at local position ``j``.
    """
    mask = torch.zeros((total, heads, max_seqlen), dtype=torch.int8, device=device)
    if total and heads and max_seqlen:
        _keep_mask_kernel[(total, heads)](
            mask,
            dropout_seed.to(device=device, dtype=torch.int32),
            dropout_p,
            max_seqlen,
            mask.stride(0),
            mask.stride(1),
            BLOCK_N=triton.next_power_of_2(max_seqlen),
        )
    return mask.bool()


@torch.library.custom_op("nemo_asr::packed_rel_pos_attention_fwd", mutates_args=(), device_types="cuda")
def _attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pos_query: Tensor,
    pos_emb: Tensor,
    seq_starts: Tensor,
    seq_lens: Tensor,
    max_seqlen: int,
    dropout_p: float,
    dropout_seed: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor]:
    # the inputs are made contiguous here as well: a compiled graph may feed the op any layout
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    pos_emb = _padded_rows(pos_emb.contiguous())
    position_term = _position_term(pos_query.contiguous(), pos_emb, scale)
    out, lse = _forward(
        q, k, v, _term_view(position_term), seq_starts, seq_lens, max_seqlen, dropout_p, dropout_seed, scale
    )
    return out, lse


@_attention_fwd.register_fake
def _attention_fwd_fake(q, k, v, pos_query, pos_emb, seq_starts, seq_lens, max_seqlen, dropout_p, dropout_seed, scale):
    del k, v, pos_query, pos_emb, seq_starts, seq_lens, max_seqlen, dropout_p, dropout_seed, scale
    # new_empty, not empty_like: the outputs are contiguous whatever the traced input layout
    return q.new_empty(tuple(q.shape)), q.new_empty(tuple(q.shape[:2]), dtype=torch.float32)


@torch.library.custom_op("nemo_asr::packed_rel_pos_attention_bwd", mutates_args=(), device_types="cuda")
def _attention_bwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pos_query: Tensor,
    pos_emb: Tensor,
    seq_starts: Tensor,
    seq_lens: Tensor,
    max_seqlen: int,
    lse: Tensor,
    d_out: Tensor,
    dropout_p: float,
    dropout_seed: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    q, k, v, pos_query = q.contiguous(), k.contiguous(), v.contiguous(), pos_query.contiguous()
    total, heads, head_dim = q.shape
    width = pos_emb.shape[0]
    pos_emb = _padded_rows(pos_emb.contiguous())
    position_term = _position_term(pos_query, pos_emb, scale)
    # the position-term gradient takes the layout of the term; zeroed as a whole, the padding columns stay zero
    d_position_term = torch.zeros_like(position_term, dtype=torch.float32)
    dq, dk, dv = _backward(
        q,
        k,
        v,
        _term_view(position_term),
        seq_starts,
        seq_lens,
        max_seqlen,
        lse,
        d_out.contiguous(),
        dropout_p,
        dropout_seed,
        scale,
        _term_view(d_position_term),
    )
    del position_term
    # the chain rule through the position term, GEMMs in the io dtype on the row-padded buffers
    # (the zero padding columns contribute nothing), with the scale in their epilogue
    d_position_term = d_position_term.to(q.dtype)
    d_pos_query = torch.empty((heads, total, head_dim), dtype=q.dtype, device=q.device)
    torch.baddbmm(d_pos_query, d_position_term, pos_emb.transpose(0, 1), beta=0.0, alpha=scale, out=d_pos_query)
    d_pos_emb = torch.empty((heads, pos_emb.shape[0], head_dim), dtype=q.dtype, device=q.device)
    torch.baddbmm(
        d_pos_emb, d_position_term.transpose(1, 2), pos_query.transpose(0, 1), beta=0.0, alpha=scale, out=d_pos_emb
    )
    return (
        dq.to(q.dtype),
        dk.to(q.dtype),
        dv.to(q.dtype),
        d_pos_query.permute(1, 0, 2).contiguous(),
        d_pos_emb[:, :width].permute(1, 0, 2).contiguous(),
    )


@_attention_bwd.register_fake
def _attention_bwd_fake(
    q, k, v, pos_query, pos_emb, seq_starts, seq_lens, max_seqlen, lse, d_out, dropout_p, dropout_seed, scale
):
    del seq_starts, seq_lens, max_seqlen, lse, d_out, dropout_p, dropout_seed, scale
    return (
        q.new_empty(tuple(q.shape)),
        k.new_empty(tuple(k.shape)),
        v.new_empty(tuple(v.shape)),
        pos_query.new_empty(tuple(pos_query.shape)),
        pos_emb.new_empty(tuple(pos_emb.shape)),
    )


def _setup_context(ctx, inputs, output):
    q, k, v, pos_query, pos_emb, seq_starts, seq_lens, max_seqlen, dropout_p, dropout_seed, scale = inputs
    _, lse = output
    ctx.save_for_backward(q, k, v, pos_query, pos_emb, seq_starts, seq_lens, lse, dropout_seed)
    ctx.max_seqlen, ctx.dropout_p, ctx.scale = max_seqlen, dropout_p, scale


def _autograd_backward(ctx, d_out, d_lse):
    del d_lse  # the log-sum-exp is an internal output of the forward
    q, k, v, pos_query, pos_emb, seq_starts, seq_lens, lse, dropout_seed = ctx.saved_tensors
    dq, dk, dv, d_pos_query, d_pos_emb = torch.ops.nemo_asr.packed_rel_pos_attention_bwd(
        q,
        k,
        v,
        pos_query,
        pos_emb,
        seq_starts,
        seq_lens,
        ctx.max_seqlen,
        lse,
        d_out,
        ctx.dropout_p,
        dropout_seed,
        ctx.scale,
    )
    return dq, dk, dv, d_pos_query, d_pos_emb, None, None, None, None, None, None


torch.library.register_autograd(
    "nemo_asr::packed_rel_pos_attention_fwd", _autograd_backward, setup_context=_setup_context
)


def _padded_rows(pos_emb: Tensor) -> Tensor:
    """``(R, H, D)`` to ``(R rounded up to a multiple of eight, H, D)`` with zero rows appended."""
    width = pos_emb.shape[0]
    padded = pos_emb.new_zeros((-(-width // 8) * 8,) + tuple(pos_emb.shape[1:]))
    padded[:width] = pos_emb
    return padded


def _position_term(pos_query: Tensor, pos_emb: Tensor, scale: float) -> Tensor:
    """The scaled position term as an ``(H, T_total, R)`` buffer from ``(T_total, H, D)`` and ``(R, H, D)`` factors."""
    total, heads, _ = pos_query.shape
    buffer = torch.empty((heads, total, pos_emb.shape[0]), dtype=pos_query.dtype, device=pos_query.device)
    # beta 0: the uninitialized buffer is not read
    torch.baddbmm(buffer, pos_query.transpose(0, 1), pos_emb.permute(1, 2, 0), beta=0.0, alpha=scale, out=buffer)
    return buffer


def _term_view(buffer: Tensor) -> Tensor:
    """The ``(T_total, H, R)`` view the kernels index, of an ``(H, T_total, R)`` buffer."""
    return buffer.permute(1, 0, 2)


def _head_dim_block(head_dim: int) -> int:
    return max(16, triton.next_power_of_2(head_dim))


def _forward(q, k, v, position_term, seq_starts, seq_lens, max_seqlen, dropout_p, dropout_seed, scale):
    total, heads, head_dim = q.shape
    # zeros, not empty: the kernels write the rows of the utterances only
    out = torch.zeros_like(q)
    lse = torch.empty((total, heads), dtype=torch.float32, device=q.device)
    n_batch = seq_starts.numel()
    if n_batch and max_seqlen:
        # 64 x 128 tiles with eight warps and two pipeline stages (tuned at D = 128 in bf16 on a GB300);
        # rows above 256 bytes (float32 at D = 128) take 64 key columns to fit the shared memory
        head_dim_block = _head_dim_block(head_dim)
        block_m, block_n = 64, (128 if head_dim_block * q.element_size() <= 256 else 64)
        _forward_kernel[(n_batch * heads, triton.cdiv(max_seqlen, block_m))](
            q,
            k,
            v,
            position_term,
            out,
            lse,
            seq_starts,
            seq_lens,
            dropout_seed,
            dropout_p,
            scale,
            q.stride(0),
            q.stride(1),
            position_term.stride(0),
            position_term.stride(1),
            2 * max_seqlen - 1,
            max_seqlen - 1,
            max_seqlen,
            H=heads,
            D=head_dim,
            D_PAD=head_dim_block,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            DROPOUT=dropout_p > 0.0,
            IEEE=q.dtype == torch.float32,
            num_warps=8,
            num_stages=2,
        )
    return out, lse


def _backward(
    q,
    k,
    v,
    position_term,
    seq_starts,
    seq_lens,
    max_seqlen,
    lse,
    d_out,
    dropout_p,
    dropout_seed,
    scale,
    d_position_term,
):
    total, heads, head_dim = q.shape
    delta = torch.zeros((total, heads), dtype=torch.float32, device=q.device)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    n_batch = seq_starts.numel()
    if n_batch and max_seqlen:
        head_dim_block = _head_dim_block(head_dim)
        # smaller tiles for wide heads (the shared memory of the backward tiles)
        block_m, block_n = (64, 64) if head_dim_block <= 64 else (32, 32)
        common = dict(
            H=heads,
            D=head_dim,
            D_PAD=head_dim_block,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            DROPOUT=dropout_p > 0.0,
            IEEE=q.dtype == torch.float32,
        )
        strides = (q.stride(0), q.stride(1), position_term.stride(0), position_term.stride(1))
        inputs = (q, k, v, position_term, d_out, lse)
        _delta_kernel[(n_batch * heads, triton.cdiv(max_seqlen, block_m))](
            *inputs,
            delta,
            seq_starts,
            seq_lens,
            dropout_seed,
            dropout_p,
            scale,
            *strides,
            2 * max_seqlen - 1,
            max_seqlen - 1,
            max_seqlen,
            **common,
        )
        _dkv_kernel[(n_batch * heads, triton.cdiv(max_seqlen, block_n))](
            *inputs,
            delta,
            dk,
            dv,
            seq_starts,
            seq_lens,
            dropout_seed,
            dropout_p,
            scale,
            *strides,
            2 * max_seqlen - 1,
            max_seqlen - 1,
            max_seqlen,
            **common,
        )
        _dq_kernel[(n_batch * heads, triton.cdiv(max_seqlen, block_m))](
            *inputs,
            delta,
            dq,
            d_position_term,
            seq_starts,
            seq_lens,
            dropout_seed,
            dropout_p,
            scale,
            *strides,
            d_position_term.stride(0),
            d_position_term.stride(1),
            2 * max_seqlen - 1,
            max_seqlen - 1,
            max_seqlen,
            **common,
        )
    return dq, dk, dv


@triton.jit
def _load_rows(ptr, rows, h, stride_t, stride_h, offs_d, row_mask, D: tl.constexpr, D_PAD: tl.constexpr):
    """A ``(rows, D_PAD)`` tile of a ``(T_total, H, D)`` tensor at head ``h``, zero outside the rows and past ``D``."""
    ptrs = ptr + rows[:, None] * stride_t + h * stride_h + offs_d[None, :]
    if D_PAD == D:
        mask = row_mask[:, None]
    else:
        mask = row_mask[:, None] & (offs_d < D)[None, :]
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def _store_rows(ptr, rows, h, stride_t, stride_h, offs_d, row_mask, tile, D: tl.constexpr, D_PAD: tl.constexpr):
    ptrs = ptr + rows[:, None] * stride_t + h * stride_h + offs_d[None, :]
    if D_PAD == D:
        mask = row_mask[:, None]
    else:
        mask = row_mask[:, None] & (offs_d < D)[None, :]
    tl.store(ptrs, tile, mask=mask)


@triton.jit
def _dot(a, b, IEEE: tl.constexpr):
    if IEEE:
        product = tl.dot(a, b, input_precision="ieee")
    else:
        product = tl.dot(a, b)
    return product


@triton.jit
def _term_offsets(q_rows, h, rel, stride_bt, stride_bh):
    """64-bit offsets into the position term, whose extent passes 2**31 at large packed batches."""
    return q_rows[:, None].to(tl.int64) * stride_bt + h.to(tl.int64) * stride_bh + rel


@triton.jit
def _keep(dropout_seed, h, q_rows, offs_n, drop_stride, dropout_p):
    """The kept attention weights of the (global query row, local key column) pairs; 32-bit Philox offsets.

    The head enters the seed through a literal stride (Triton takes no Python globals in a kernel).
    """
    offsets = q_rows[:, None] * drop_stride + offs_n[None, :]
    return tl.rand(tl.load(dropout_seed) + h * 1000003, offsets) >= dropout_p


@triton.jit
def _forward_kernel(
    Q,
    K,
    V,
    BD,
    Out,
    Lse,
    SeqStarts,
    SeqLens,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_bt,
    stride_bh,
    R,
    center,
    drop_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    seq_start = tl.load(SeqStarts + b)
    seq_len = tl.load(SeqLens + b)
    if pid_m * BLOCK_M >= seq_len:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_PAD)
    q_rows = seq_start + offs_m
    q_mask = offs_m < seq_len
    q = _load_rows(Q, q_rows, h, stride_qt, stride_qh, offs_d, q_mask, D, D_PAD)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_PAD], dtype=tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        k_rows = seq_start + offs_n
        k = _load_rows(K, k_rows, h, stride_qt, stride_qh, offs_d, n_mask, D, D_PAD)
        s = _dot(q, tl.trans(k), IEEE) * scale
        rel = center + offs_n[None, :] - offs_m[:, None]
        valid = q_mask[:, None] & n_mask[None, :] & (rel >= 0) & (rel < R)
        bd = tl.load(BD + _term_offsets(q_rows, h, rel, stride_bt, stride_bh), mask=valid, other=0.0)
        s = tl.where(valid, s + bd, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # a row without a valid score yet has m_new = -inf; exp(-inf - -inf) would be NaN
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if DROPOUT:
            keep = _keep(Seed, h, q_rows, offs_n, drop_stride, dropout_p)
            p_use = tl.where(keep, p / (1.0 - dropout_p), 0.0)
        else:
            p_use = p
        v_tile = _load_rows(V, k_rows, h, stride_qt, stride_qh, offs_d, n_mask, D, D_PAD)
        acc += _dot(p_use.to(v_tile.dtype), v_tile, IEEE)
        m_i = m_new
    out = acc / l_i[:, None]
    _store_rows(Out, q_rows, h, stride_qt, stride_qh, offs_d, q_mask, out, D, D_PAD)
    tl.store(Lse + q_rows * H + h, m_i + tl.log(l_i), mask=q_mask)


@triton.jit
def _keep_mask_kernel(Mask, Seed, dropout_p, max_seqlen, stride_mt, stride_mh, BLOCK_N: tl.constexpr):
    i = tl.program_id(0)
    h = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    keep = tl.rand(tl.load(Seed) + h * 1000003, i * max_seqlen + offs_n) >= dropout_p
    tl.store(Mask + i * stride_mt + h * stride_mh + offs_n, keep.to(tl.int8), mask=offs_n < max_seqlen)


@triton.jit
def _delta_kernel(
    Q,
    K,
    V,
    BD,
    DO,
    Lse,
    Delta,
    SeqStarts,
    SeqLens,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_bt,
    stride_bh,
    R,
    center,
    drop_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    # delta_i = sum_j p_ij dp_ij from the recomputed weights, see the module docstring
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    seq_start = tl.load(SeqStarts + b)
    seq_len = tl.load(SeqLens + b)
    if pid_m * BLOCK_M >= seq_len:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_PAD)
    m_mask = offs_m < seq_len
    q_rows = seq_start + offs_m
    q = _load_rows(Q, q_rows, h, stride_qt, stride_qh, offs_d, m_mask, D, D_PAD)
    do = _load_rows(DO, q_rows, h, stride_qt, stride_qh, offs_d, m_mask, D, D_PAD)
    lse = tl.load(Lse + q_rows * H + h, mask=m_mask, other=0.0)
    delta = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        k_rows = seq_start + offs_n
        k = _load_rows(K, k_rows, h, stride_qt, stride_qh, offs_d, n_mask, D, D_PAD)
        v = _load_rows(V, k_rows, h, stride_qt, stride_qh, offs_d, n_mask, D, D_PAD)
        s = _dot(q, tl.trans(k), IEEE) * scale
        dp = _dot(do, tl.trans(v), IEEE)
        rel = center + offs_n[None, :] - offs_m[:, None]
        valid = m_mask[:, None] & n_mask[None, :] & (rel >= 0) & (rel < R)
        bd = tl.load(BD + _term_offsets(q_rows, h, rel, stride_bt, stride_bh), mask=valid, other=0.0)
        s = tl.where(valid, s + bd, float("-inf"))
        p = tl.exp(s - lse[:, None])
        if DROPOUT:
            keep = _keep(Seed, h, q_rows, offs_n, drop_stride, dropout_p)
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        delta += tl.sum(tl.where(valid, p * dp, 0.0), 1)
    tl.store(Delta + q_rows * H + h, delta, mask=m_mask)


@triton.jit
def _dkv_kernel(
    Q,
    K,
    V,
    BD,
    DO,
    Lse,
    Delta,
    DK,
    DV,
    SeqStarts,
    SeqLens,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_bt,
    stride_bh,
    R,
    center,
    drop_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    seq_start = tl.load(SeqStarts + b)
    seq_len = tl.load(SeqLens + b)
    if pid_n * BLOCK_N >= seq_len:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_PAD)
    n_mask = offs_n < seq_len
    k_rows = seq_start + offs_n
    k = _load_rows(K, k_rows, h, stride_qt, stride_qh, offs_d, n_mask, D, D_PAD)
    v = _load_rows(V, k_rows, h, stride_qt, stride_qh, offs_d, n_mask, D, D_PAD)
    dk = tl.zeros([BLOCK_N, D_PAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_PAD], dtype=tl.float32)
    for start_m in range(0, seq_len, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < seq_len
        q_rows = seq_start + offs_m
        q = _load_rows(Q, q_rows, h, stride_qt, stride_qh, offs_d, m_mask, D, D_PAD)
        do = _load_rows(DO, q_rows, h, stride_qt, stride_qh, offs_d, m_mask, D, D_PAD)
        lse = tl.load(Lse + q_rows * H + h, mask=m_mask, other=0.0)
        delta = tl.load(Delta + q_rows * H + h, mask=m_mask, other=0.0)
        s = _dot(q, tl.trans(k), IEEE) * scale
        rel = center + offs_n[None, :] - offs_m[:, None]
        valid = m_mask[:, None] & n_mask[None, :] & (rel >= 0) & (rel < R)
        bd = tl.load(BD + _term_offsets(q_rows, h, rel, stride_bt, stride_bh), mask=valid, other=0.0)
        s = tl.where(valid, s + bd, float("-inf"))
        p = tl.exp(s - lse[:, None])
        if DROPOUT:
            keep = _keep(Seed, h, q_rows, offs_n, drop_stride, dropout_p)
            p_use = tl.where(keep, p / (1.0 - dropout_p), 0.0)
        else:
            p_use = p
        dv += _dot(tl.trans(p_use.to(do.dtype)), do, IEEE)
        dp = _dot(do, tl.trans(v), IEEE)
        if DROPOUT:
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        # the scale is folded into ds before the dot, as in the flash-attention backward:
        # Triton 3.8 compiled a scaled dot (dot(trans(ds), q) * scale) in bf16 and fp16 to a non-accumulating update
        ds = tl.where(valid, p * (dp - delta[:, None]) * scale, 0.0)
        dk += _dot(tl.trans(ds.to(q.dtype)), q, IEEE)
    _store_rows(DK, k_rows, h, stride_qt, stride_qh, offs_d, n_mask, dk, D, D_PAD)
    _store_rows(DV, k_rows, h, stride_qt, stride_qh, offs_d, n_mask, dv, D, D_PAD)


@triton.jit
def _dq_kernel(
    Q,
    K,
    V,
    BD,
    DO,
    Lse,
    Delta,
    DQ,
    DBD,
    SeqStarts,
    SeqLens,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_bt,
    stride_bh,
    stride_dbt,
    stride_dbh,
    R,
    center,
    drop_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    seq_start = tl.load(SeqStarts + b)
    seq_len = tl.load(SeqLens + b)
    if pid_m * BLOCK_M >= seq_len:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_PAD)
    m_mask = offs_m < seq_len
    q_rows = seq_start + offs_m
    q = _load_rows(Q, q_rows, h, stride_qt, stride_qh, offs_d, m_mask, D, D_PAD)
    do = _load_rows(DO, q_rows, h, stride_qt, stride_qh, offs_d, m_mask, D, D_PAD)
    lse = tl.load(Lse + q_rows * H + h, mask=m_mask, other=0.0)
    delta = tl.load(Delta + q_rows * H + h, mask=m_mask, other=0.0)
    dq = tl.zeros([BLOCK_M, D_PAD], dtype=tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        k_rows = seq_start + offs_n
        k = _load_rows(K, k_rows, h, stride_qt, stride_qh, offs_d, n_mask, D, D_PAD)
        v = _load_rows(V, k_rows, h, stride_qt, stride_qh, offs_d, n_mask, D, D_PAD)
        s = _dot(q, tl.trans(k), IEEE) * scale
        rel = center + offs_n[None, :] - offs_m[:, None]
        valid = m_mask[:, None] & n_mask[None, :] & (rel >= 0) & (rel < R)
        bd = tl.load(BD + _term_offsets(q_rows, h, rel, stride_bt, stride_bh), mask=valid, other=0.0)
        s = tl.where(valid, s + bd, float("-inf"))
        p = tl.exp(s - lse[:, None])
        dp = _dot(do, tl.trans(v), IEEE)
        if DROPOUT:
            keep = _keep(Seed, h, q_rows, offs_n, drop_stride, dropout_p)
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
        dq += _dot((ds * scale).to(k.dtype), k, IEEE)
        # the position-term gradient is ds itself, row-owned: every (i, j) of a row has its own entry
        tl.store(DBD + _term_offsets(q_rows, h, rel, stride_dbt, stride_dbh), ds, mask=valid)
    _store_rows(DQ, q_rows, h, stride_qt, stride_qh, offs_d, m_mask, dq, D, D_PAD)
