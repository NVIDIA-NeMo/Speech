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

"""Depthwise 1-D convolution over sequence-packed activations with explicit utterance boundaries.

The convolution module of the sequence-packed Conformer convolves every utterance along time
with one filter per channel.
Its input ``(T_total, C)`` holds the utterances one after another without padding,
and every utterance is given by its start row and its length (device tensors),
so the kernels read the rows of an utterance only, never a neighbour's,
and no gapped or padded copy of the activations is formed.
With ``K`` taps and ``pad_left`` frames of zero padding before the first frame of an utterance
(``K - 1 - pad_left`` after its last one, so the output has as many rows as the input)::

    out[r, c] = bias[c] + sum_k w[c, k] * x[r + k - pad_left, c]    the tap row inside the utterance of r
    dx[r, c]  = sum_k w[c, k] * d_out[r - k + pad_left, c]          the correlation with the flipped filter
    dw[c, k]  = sum_r d_out[r, c] * x[r + k - pad_left, c]
    db[c]     = sum_r d_out[r, c]

The kernels follow the register-window depthwise kernels of rwth-i6/returnn (``returnn/triton/depthwise_conv.py``):
a program owns ``BLOCK_R`` consecutive output rows of one utterance and ``BLOCK_C`` channels,
loads each of its ``BLOCK_R + K - 1`` input rows once as a float32 vector
and adds it into the register accumulators of the output rows it serves,
one fused multiply-add per element and tap, the taps of every row in increasing order.

The forward and the input gradient run one program per (utterance, row block, channel block)
and return at once for a row block past the utterance's length,
so the grid is bounded by the longest utterance
and the kernels follow new contents and new lengths within that bound on a CUDA-graph replay.
The filter and bias gradients reduce over all rows: a fixed number of row splits,
each walked in blocks that end at the utterance boundaries
(the utterance of a row is found by a binary search over the starts),
with one float32 accumulator per tap, into a (split, tap, channel) buffer summed afterwards,
so the result is deterministic and the scratch does not grow with the rows.
Rows outside every utterance (gaps between utterances, the slack of a capacity buffer) are neither read nor written.
The offsets into the activations are 64-bit.
The two custom ops have fake implementations, so a traced or captured training step sees opaque ops.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["PACKED_DEPTHWISE_CONV_DTYPES", "packed_depthwise_conv1d"]

PACKED_DEPTHWISE_CONV_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

# a program owns 32 output rows and 256 channels over four warps, two channels per thread:
# the row accumulators of the window and the taps live in registers
# (about 150 to 200 per thread at 32 taps, no spills on a GB300)
BLOCK_R = 32
BLOCK_C = 256
NUM_WARPS = 4
# row splits of the filter gradient; its (split, tap, channel) float32 scratch is summed afterwards
FILTER_GRAD_SPLITS = 128


@triton.jit
def _window_row(
    acc,
    taps,
    in_base,
    c_mask,
    t0,
    length,
    pad,
    n_chan,
    j: tl.constexpr,
    FLIP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    KW: tl.constexpr,
):
    """Load window row ``j`` once (the utterance's input row ``t0 + j - pad``, zero outside ``[0, length)``)
    and add it to the accumulators of the output rows ``i`` it serves,
    through tap ``j - i`` (``KW - 1 - (j - i)`` with FLIP).
    The static index arithmetic stays inline: Triton binds a constexpr name once."""
    t_in = t0 + j - pad
    x = tl.load(in_base + t_in.to(tl.int64) * n_chan, mask=c_mask & (t_in >= 0) & (t_in < length), other=0.0)
    x = x.to(tl.float32)
    for i in tl.static_range(BLOCK_R):
        if j - i >= 0:
            if j - i < KW:
                acc = acc[:i] + (acc[i] + x * taps[(KW - 1 - (j - i)) if FLIP else (j - i)],) + acc[i + 1 :]
    return acc


@triton.jit
def _window_kernel(
    In,
    Taps,
    Bias,
    Starts,
    Lens,
    n_blocks_t,
    n_chan,
    pad,
    Out,
    FLIP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KW: tl.constexpr,
):
    """out[r, c] = bias[c] + sum_k taps[k, c] * in[r + k - pad, c] over the rows of one utterance
    (with FLIP: out[r, c] = sum_k taps[k, c] * in[r - k + (KW - 1 - pad), c]),
    one program per (utterance, output row block, channel block); ``Taps`` is the filter as (tap, channel).

    Window row ``j`` is the utterance's input row ``t0 + j - pad`` for ``j`` in ``[0, BLOCK_R + KW - 1)``;
    it serves the output rows ``i`` with tap ``j - i`` (``KW - 1 - (j - i)`` with FLIP),
    and the rows are visited in the order that adds the taps of every output row in increasing order."""
    pid = tl.program_id(0)
    b = pid // n_blocks_t
    t0 = (pid % n_blocks_t) * BLOCK_R
    length = tl.load(Lens + b)
    if t0 < length:
        start = tl.load(Starts + b)
        offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = offs_c < n_chan
        taps = ()
        for k in tl.static_range(KW):
            taps = taps + (tl.load(Taps + k * n_chan + offs_c, mask=c_mask, other=0.0).to(tl.float32),)
        if HAS_BIAS:
            bias = tl.load(Bias + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        else:
            bias = taps[0]
        base = start.to(tl.int64) * n_chan + offs_c
        acc = (tl.zeros((BLOCK_C,), dtype=tl.float32),) * BLOCK_R
        for jj in tl.static_range(BLOCK_R + KW - 1):
            acc = _window_row(
                acc,
                taps,
                In + base,
                c_mask,
                t0,
                length,
                pad,
                n_chan,
                j=(BLOCK_R + KW - 2 - jj) if FLIP else jj,
                FLIP=FLIP,
                BLOCK_R=BLOCK_R,
                KW=KW,
            )
        n_valid = length - t0
        for i in tl.static_range(BLOCK_R):
            v = acc[i]
            if HAS_BIAS:
                v = v + bias
            tl.store(
                Out + base + (t0 + i).to(tl.int64) * n_chan, v.to(Out.dtype.element_ty), mask=c_mask & (i < n_valid)
            )


@triton.jit
def _filter_row(
    acc,
    gs,
    x_base,
    c_mask,
    t0,
    length,
    pad_left,
    n_chan,
    j: tl.constexpr,
    BLOCK_R: tl.constexpr,
    KW: tl.constexpr,
):
    """Load window row ``j`` of a block of :func:`_filter_grad_kernel` once
    (the utterance's input row ``t0 + j - pad_left``, zero outside ``[0, length)``)
    and add its products with the block's output gradient rows to the accumulators of the taps it serves."""
    t_in = t0 + j - pad_left
    x = tl.load(x_base + t_in.to(tl.int64) * n_chan, mask=c_mask & (t_in >= 0) & (t_in < length), other=0.0)
    x = x.to(tl.float32)
    for i in tl.static_range(BLOCK_R):
        if j - i >= 0:
            if j - i < KW:
                acc = acc[: j - i] + (acc[j - i] + gs[i] * x,) + acc[j - i + 1 :]
    return acc


@triton.jit
def _filter_grad_kernel(
    X,
    DOut,
    Starts,
    Lens,
    n_rows,
    n_batch,
    n_chan,
    pad_left,
    rows_per_prog,
    Partial,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KW: tl.constexpr,
    KW_P: tl.constexpr,
):
    """Partial[s, k, c] = sum over the rows r of split s of d_out[r, c] * x[r + k - pad_left, c],
    the tap row inside the utterance of r, plus the d_out row sum as tap KW for the bias;
    one program per (row split, channel block).

    The program walks its rows in blocks of up to BLOCK_R rows that end at the utterance boundaries:
    the utterance of its first row is the last one that starts at or before it
    (an empty utterance shares its start with its successor),
    and the walk moves on to the next utterances as the rows pass their ends.
    Each block stages its d_out rows and loads every input row of its window once;
    the blocks add into the taps in increasing row order.
    Rows outside every utterance (a gap before an utterance, the slack past the last one) add nothing."""
    pid_s = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < n_chan
    acc = (tl.zeros((BLOCK_C,), dtype=tl.float32),) * KW_P
    row_start = pid_s * rows_per_prog
    row_end = tl.minimum(row_start + rows_per_prog, n_rows)
    lo_b = tl.zeros_like(row_start)
    hi_b = lo_b + n_batch
    while lo_b + 1 < hi_b:
        mid = (lo_b + hi_b) // 2
        if tl.load(Starts + mid) <= row_start:
            lo_b = mid
        else:
            hi_b = mid
    b = lo_b
    start = tl.load(Starts + b)
    end = start + tl.load(Lens + b)
    r0 = row_start
    while r0 < row_end:
        while (r0 >= end) & (b + 1 < n_batch):
            b += 1
            start = tl.load(Starts + b)
            end = start + tl.load(Lens + b)
        # a block past the last utterance's end loads zeros and consumes the rest of the split
        content = r0 < end
        n_valid = tl.where(content, tl.minimum(tl.minimum(BLOCK_R, end - r0), row_end - r0), row_end - r0)
        t0 = r0 - start
        length = end - start
        base = start.to(tl.int64) * n_chan + offs_c
        g_mask = c_mask & content
        gs = ()
        for i in tl.static_range(BLOCK_R):
            # a row before the utterance's start (a gap in front of it) lies outside every utterance
            g_row = g_mask & (i < n_valid) & (t0 + i >= 0)
            g = tl.load(DOut + base + (t0 + i).to(tl.int64) * n_chan, mask=g_row, other=0.0)
            gs = gs + (g.to(tl.float32),)
        for j in tl.static_range(BLOCK_R + KW - 1):
            acc = _filter_row(acc, gs, X + base, c_mask, t0, length, pad_left, n_chan, j=j, BLOCK_R=BLOCK_R, KW=KW)
        if HAS_BIAS:
            for i in tl.static_range(BLOCK_R):
                acc = acc[:KW] + (acc[KW] + gs[i],)
        r0 += n_valid
    p_base = Partial + pid_s.to(tl.int64) * KW_P * n_chan + offs_c
    for k in tl.static_range(KW_P):
        tl.store(p_base + k * n_chan, acc[k], mask=c_mask)


def _forward(x, weight, bias, seq_starts, seq_lens, max_seqlen: int, pad_left: int) -> Tensor:
    """Launch the forward window kernel; rows outside every utterance are zero."""
    n_chan = x.shape[1]
    taps = weight.t().contiguous()
    out = torch.zeros_like(x)
    n_blocks_t = triton.cdiv(max_seqlen, BLOCK_R)
    grid = (seq_lens.shape[0] * n_blocks_t, triton.cdiv(n_chan, BLOCK_C))
    _window_kernel[grid](
        x,
        taps,
        bias if bias is not None else taps,
        seq_starts,
        seq_lens,
        n_blocks_t,
        n_chan,
        pad_left,
        out,
        FLIP=False,
        HAS_BIAS=bias is not None,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        KW=weight.shape[1],
        num_warps=NUM_WARPS,
    )
    return out


def _backward(x, weight, d_out, seq_starts, seq_lens, max_seqlen: int, pad_left: int, has_bias: bool):
    """Launch the input-gradient window kernel and the filter-gradient kernel.

    Returns:
        the input gradient (rows outside every utterance are zero) and the float32 sums ``(K + has_bias, C)``
        the filter gradient (the first ``K`` rows, as (tap, channel)) and the bias gradient (row ``K``) are read from.
    """
    n_rows, n_chan = x.shape
    width = weight.shape[1]
    taps = weight.t().contiguous()
    dx = torch.zeros_like(x)
    n_blocks_t = triton.cdiv(max_seqlen, BLOCK_R)
    grid = (seq_lens.shape[0] * n_blocks_t, triton.cdiv(n_chan, BLOCK_C))
    _window_kernel[grid](
        d_out,
        taps,
        taps,
        seq_starts,
        seq_lens,
        n_blocks_t,
        n_chan,
        width - 1 - pad_left,
        dx,
        FLIP=True,
        HAS_BIAS=False,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        KW=width,
        num_warps=NUM_WARPS,
    )
    rows_per_prog = max(triton.cdiv(n_rows, FILTER_GRAD_SPLITS), 1)
    n_splits = max(triton.cdiv(n_rows, rows_per_prog), 1)
    partial = torch.empty((n_splits, width + int(has_bias), n_chan), dtype=torch.float32, device=x.device)
    _filter_grad_kernel[(n_splits, triton.cdiv(n_chan, BLOCK_C))](
        x,
        d_out,
        seq_starts,
        seq_lens,
        n_rows,
        seq_lens.shape[0],
        n_chan,
        pad_left,
        rows_per_prog,
        partial,
        HAS_BIAS=has_bias,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        KW=width,
        KW_P=width + int(has_bias),
        num_warps=NUM_WARPS,
    )
    return dx, partial.sum(dim=0)


@torch.library.custom_op("nemo_asr::packed_depthwise_conv1d_fwd", mutates_args=(), device_types="cuda")
def _fwd_op(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    seq_starts: Tensor,
    seq_lens: Tensor,
    max_seqlen: int,
    pad_left: int,
) -> Tensor:
    # a compiled graph may feed the op any layout
    bias = bias.contiguous() if bias is not None else None
    return _forward(x.contiguous(), weight.contiguous(), bias, seq_starts, seq_lens, max_seqlen, pad_left)


@_fwd_op.register_fake
def _fwd_fake(x, weight, bias, seq_starts, seq_lens, max_seqlen, pad_left):
    del weight, bias, seq_starts, seq_lens, max_seqlen, pad_left
    return x.new_empty(x.shape)


@torch.library.custom_op("nemo_asr::packed_depthwise_conv1d_bwd", mutates_args=(), device_types="cuda")
def _bwd_op(
    x: Tensor,
    weight: Tensor,
    d_out: Tensor,
    seq_starts: Tensor,
    seq_lens: Tensor,
    max_seqlen: int,
    pad_left: int,
    has_bias: bool,
) -> tuple[Tensor, Tensor]:
    return _backward(
        x.contiguous(), weight.contiguous(), d_out.contiguous(), seq_starts, seq_lens, max_seqlen, pad_left, has_bias
    )


@_bwd_op.register_fake
def _bwd_fake(x, weight, d_out, seq_starts, seq_lens, max_seqlen, pad_left, has_bias):
    del d_out, seq_starts, seq_lens, max_seqlen, pad_left
    return x.new_empty(x.shape), x.new_empty((weight.shape[1] + int(has_bias), weight.shape[0]), dtype=torch.float32)


def _setup_context(ctx, inputs, output):
    del output
    x, weight, bias, seq_starts, seq_lens, max_seqlen, pad_left = inputs
    ctx.save_for_backward(x, weight, seq_starts, seq_lens)
    ctx.bias_dtype = bias.dtype if bias is not None else None
    ctx.max_seqlen = max_seqlen
    ctx.pad_left = pad_left


def _autograd_backward(ctx, d_out):
    x, weight, seq_starts, seq_lens = ctx.saved_tensors
    width = weight.shape[1]
    has_bias = ctx.bias_dtype is not None
    dx, summed = torch.ops.nemo_asr.packed_depthwise_conv1d_bwd(
        x, weight, d_out, seq_starts, seq_lens, ctx.max_seqlen, ctx.pad_left, has_bias
    )
    d_weight = summed[:width].t().contiguous().to(weight.dtype)
    d_bias = summed[width].to(ctx.bias_dtype) if has_bias else None
    return dx, d_weight, d_bias, None, None, None, None


torch.library.register_autograd(
    "nemo_asr::packed_depthwise_conv1d_fwd", _autograd_backward, setup_context=_setup_context
)


def packed_depthwise_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    seq_starts: Tensor,
    seq_lens: Tensor,
    max_seqlen: int,
    *,
    pad_left: int,
) -> Tensor:
    """Depthwise convolution of every utterance of a packed buffer along time, with the output rows of the input.

    Args:
        x: packed activations ``(T_total, C)`` on CUDA, in float32, bfloat16 or float16.
        weight: one filter per channel, ``(C, K)``, in the dtype of ``x``.
        bias: ``(C,)`` in the dtype of ``x``, or None.
        seq_starts: first row of every utterance, ``(B,)``, int32 or int64.
        seq_lens: rows of every utterance, ``(B,)``, int32 or int64; the utterances must not overlap.
        max_seqlen: the longest utterance (a host integer, it sizes the grid); on a CUDA-graph replay the kernels
            follow new starts and lengths up to this bound.
        pad_left: frames of zero padding before the first frame of an utterance, in ``[0, K)``;
            ``K - 1 - pad_left`` frames follow its last one.

    Returns:
        ``(T_total, C)`` in the dtype of ``x``, accumulated in float32; a row outside every utterance is zero.
        The gradients for ``x``, ``weight`` and ``bias`` are those of the per-utterance convolution;
        rows of the output gradient outside every utterance do not enter them.
    """
    if x.ndim != 2:
        raise ValueError(f"x must have shape (T_total, C), got {tuple(x.shape)}.")
    if not x.is_cuda:
        raise ValueError("packed_depthwise_conv1d runs on CUDA tensors only.")
    if x.dtype not in PACKED_DEPTHWISE_CONV_DTYPES:
        raise ValueError(f"x must be float32, bfloat16 or float16, got {x.dtype}.")
    if weight.ndim != 2 or weight.shape[0] != x.shape[1]:
        raise ValueError(f"weight must have shape (C, K) with C = {x.shape[1]}, got {tuple(weight.shape)}.")
    width = weight.shape[1]
    if weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("weight must have the dtype and device of x.")
    if bias is not None and (bias.shape != (x.shape[1],) or bias.dtype != x.dtype or bias.device != x.device):
        raise ValueError("bias must have shape (C,) and the dtype and device of x.")
    if seq_starts.ndim != 1 or seq_starts.shape != seq_lens.shape:
        raise ValueError("seq_starts and seq_lens must be 1-D of the same shape.")
    for name, bounds in (("seq_starts", seq_starts), ("seq_lens", seq_lens)):
        if bounds.dtype not in (torch.int32, torch.int64) or bounds.device != x.device:
            raise ValueError(
                f"{name} must be int32 or int64 on the device of x, got {bounds.dtype} on {bounds.device}."
            )
    if not 0 <= pad_left < width:
        raise ValueError(f"pad_left must be in [0, {width}), got {pad_left}.")
    if int(max_seqlen) != max_seqlen or max_seqlen < 0:
        raise ValueError(f"max_seqlen must be a non-negative integer, got {max_seqlen}.")
    if x.shape[0] == 0 or max_seqlen == 0:
        # nothing to convolve; the parameters keep a zero gradient edge
        anchor = weight.sum() if bias is None else weight.sum() + bias.sum()
        return x * 0.0 + anchor.to(x.dtype) * 0.0
    return torch.ops.nemo_asr.packed_depthwise_conv1d_fwd(
        x.contiguous(),
        weight.contiguous(),
        bias.contiguous() if bias is not None else None,
        seq_starts.to(torch.int32).contiguous(),
        seq_lens.to(torch.int32).contiguous(),
        int(max_seqlen),
        int(pad_left),
    )
