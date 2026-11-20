# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Activation and dropout of a feed-forward hidden in one kernel, with the pre-activation as the only saved tensor.

``linear2(dropout(activation(linear1(x))))`` as separate modules saves three tensors of the hidden's width for the backward:
the activation's input or output,
the dropout mask,
and the dropout output as the input of the second linear layer.
``activation_dropout_linear`` computes the activation, the mask and the second linear layer together
and saves the pre-activation and the mask's seed alone;
its backward recomputes the dropped-out activation from them for the weight gradient
and the activation's derivative and the mask for the input gradient,
so one tensor of the hidden's width stays alive per feed-forward block.
``activation_dropout`` is the elementwise part on its own.

The mask is a Philox hash of the element index (``tl.rand``)
with a seed read from a one-element device tensor;
the seed is drawn on the device for every call, so a CUDA-graph replay draws a fresh mask,
and a caller may pass a seed for a reproducible mask.
With ``broadcast=True`` every sequence has one mask of the width of the last axis,
shared by all its positions, as ``BroadcastDropout`` draws them:
the sequences of padded states ``(..., T, D)`` are the leading indices,
the sequence of a row of sequence-packed states ``(T_total, D)`` comes from ``sequence_ids``.
The kernels read the kept features from the ``(sequences, D)`` matrix of booleans
``activation_dropout_mask((sequences, D), p, seed, device)`` at the row's sequence and the element's column,
instead of hashing every element.
The activation and its derivative are evaluated in float32 and the results stored in the input dtype,
so the output is rounded once where the separate modules round after the activation and after the scaling.

The activations are the ones a feed-forward block takes from ``activation_registry``
that have a closed form here: identity, ReLU, squared ReLU, SiLU (Swish) and the exact (erf) GELU.
``fused_activation_name`` maps an activation module to its name, or to None for the others.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.common.parts.utils import SquaredReLU

__all__ = [
    "ACTIVATION_DROPOUT_DTYPES",
    "FUSED_ACTIVATIONS",
    "activation_dropout",
    "activation_dropout_linear",
    "activation_dropout_mask",
    "fused_activation_name",
]

# the dtypes the kernels read and write; the activation runs in float32 either way
ACTIVATION_DROPOUT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
# the activations the kernels implement, by name, with the compile-time id the kernels dispatch on
FUSED_ACTIVATIONS = {"identity": 0, "relu": 1, "squared_relu": 2, "silu": 3, "gelu": 4}
BLOCK = 1024


def fused_activation_name(activation: torch.nn.Module) -> str | None:
    """The name in ``FUSED_ACTIVATIONS`` of an activation module, or None when the kernels have no form of it.

    The module's exact class decides, so a subclass with another ``forward`` is not fused.
    """
    if type(activation) is torch.nn.Identity:
        return "identity"
    if type(activation) is torch.nn.ReLU:
        return "relu"
    if type(activation) is SquaredReLU:
        return "squared_relu"
    if type(activation) in (torch.nn.SiLU, Swish):
        return "silu"
    if type(activation) is torch.nn.GELU and activation.approximate == "none":
        return "gelu"
    return None


def activation_dropout(
    x: Tensor,
    activation: str,
    p: float,
    *,
    broadcast: bool = False,
    sequence_ids: Tensor | None = None,
    num_sequences: int | None = None,
    seed: Tensor | None = None,
) -> Tensor:
    """``dropout(activation(x), p)`` of a training step, with ``x`` and the mask's seed saved for the backward.

    The identity activation saves the seed (and the broadcast mask) alone: its backward is the dropout of the gradient.

    Args:
        x: the pre-activation, any shape; float32, bfloat16 or float16 on CUDA.
            With ``broadcast`` padded states ``(..., T, D)`` with at least one batch axis,
            or with ``sequence_ids`` sequence-packed states ``(T_total, D)``.
        activation: a name in ``FUSED_ACTIVATIONS`` (``identity``, ``relu``, ``squared_relu``, ``silu``, ``gelu``).
        p: dropout probability in ``[0, 1)``; the kept elements are scaled by ``1 / (1 - p)``.
        broadcast: one mask over the last axis per sequence, shared by all positions of the sequence,
            instead of a mask per element.
        sequence_ids: the sequence of every row of sequence-packed states, an integer tensor of ``T_total``
            on the device of ``x``; needs ``broadcast`` and ``num_sequences``.
            The ids are not range-checked (that would cost a device sync): a row whose id is outside
            ``[0, num_sequences)`` is dropped whole.
        num_sequences: the number of sequences the ids index.
        seed: one-element integer tensor with the Philox seed of the mask, taken modulo ``2**31``;
            drawn on the device of ``x`` by default, which gives a fresh mask on every CUDA-graph replay.

    Returns:
        A tensor of the shape and dtype of ``x``.
    """
    seed = _checked_seed(x, activation, p, seed)
    keep, sequence_ids, rows_per_sequence = _broadcast_mask(x, p, seed, broadcast, sequence_ids, num_sequences)
    return torch.ops.nemo_asr.activation_dropout_fwd(
        x, FUSED_ACTIVATIONS[activation], p, seed, keep, sequence_ids, rows_per_sequence
    )


def activation_dropout_linear(
    x: Tensor,
    activation: str,
    p: float,
    weight: Tensor,
    bias: Tensor | None,
    *,
    broadcast: bool = False,
    sequence_ids: Tensor | None = None,
    num_sequences: int | None = None,
    seed: Tensor | None = None,
) -> Tensor:
    """``linear(dropout(activation(x), p), weight, bias)`` of a training step, with ``x`` saved for the backward.

    The linear layer's input, the dropped-out activation, is recomputed from ``x`` and the mask's seed
    in the backward instead of being saved.

    Args:
        x: the pre-activation, shape ``(..., D_in)``; float32, bfloat16 or float16 on CUDA.
            With ``broadcast`` padded states ``(..., T, D_in)`` with at least one batch axis,
            or with ``sequence_ids`` sequence-packed states ``(T_total, D_in)``.
        activation: a name in ``FUSED_ACTIVATIONS``.
        p: dropout probability in ``[0, 1)``.
        weight: the linear layer's weight, shape ``(D_out, D_in)``.
        bias: the linear layer's bias, shape ``(D_out,)``, or None.
        broadcast: one mask over ``D_in`` per sequence, shared by all its positions, instead of a mask per element.
        sequence_ids: the sequence of every row of sequence-packed states, on the device of ``x``;
            needs ``broadcast`` and ``num_sequences``; see ``activation_dropout`` for ids outside the range.
        num_sequences: the number of sequences the ids index.
        seed: one-element integer tensor with the Philox seed of the mask, taken modulo ``2**31``,
            drawn on the device by default.

    Returns:
        A tensor of shape ``(..., D_out)`` in the dtype the linear layer produces.
    """
    seed = _checked_seed(x, activation, p, seed)
    if weight.ndim != 2 or weight.shape[1] != x.shape[-1]:
        raise ValueError(f"weight must have shape (D_out, {x.shape[-1]}), got {tuple(weight.shape)}.")
    keep, sequence_ids, rows_per_sequence = _broadcast_mask(x, p, seed, broadcast, sequence_ids, num_sequences)
    return _ActivationDropoutLinear.apply(
        x, FUSED_ACTIVATIONS[activation], p, seed, keep, sequence_ids, rows_per_sequence, weight, bias
    )


def activation_dropout_mask(shape: tuple[int, ...], p: float, seed: Tensor, device: torch.device) -> Tensor:
    """The kept elements of ``activation_dropout`` for this seed, a bool tensor of ``shape``, for exact comparisons.

    For the per-element mask ``shape`` is the shape of ``x``;
    for the broadcast mask it is ``(sequences, D)``, the mask of every sequence in its row.
    """
    seed = seed.to(device=device, dtype=torch.int32)
    mask = torch.empty(shape, dtype=torch.bool, device=device)
    numel = mask.numel()
    if numel:
        # the kernels read and write the booleans as bytes: Triton's one-bit pointers are not reliable
        _mask_kernel[(triton.cdiv(numel, BLOCK),)](mask.view(torch.uint8), seed, numel, p, BLOCK=BLOCK)
    return mask


def _checked_seed(x: Tensor, activation: str, p: float, seed: Tensor | None) -> Tensor:
    """Validate the inputs of the kernels and return the int32 seed tensor on the device of ``x``."""
    if not x.is_cuda:
        raise RuntimeError("activation_dropout requires a CUDA tensor.")
    if x.dtype not in ACTIVATION_DROPOUT_DTYPES:
        raise TypeError(f"x must be float32, bfloat16 or float16, got {x.dtype}.")
    if activation not in FUSED_ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}; the kernels implement {sorted(FUSED_ACTIVATIONS)}")
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p must be in [0, 1), got {p}.")
    if seed is not None and (seed.numel() != 1 or seed.dtype not in (torch.int32, torch.int64)):
        raise ValueError(f"seed must be one integer, got {seed.dtype} of shape {tuple(seed.shape)}.")
    if seed is not None:
        # the kernels read a 32-bit seed; the mask keeps the range without a device read
        return (seed.to(device=x.device, dtype=torch.int64) & (2**31 - 1)).to(torch.int32)
    if p > 0.0:
        return torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32, device=x.device)
    return torch.zeros((1,), dtype=torch.int32, device=x.device)


def _broadcast_mask(
    x: Tensor, p: float, seed: Tensor, broadcast: bool, sequence_ids: Tensor | None, num_sequences: int | None
) -> tuple[Tensor | None, Tensor | None, int]:
    """The ``(sequences, D)`` mask of a broadcast call with the row-to-sequence map, or none for the per-element mask.

    Returns the kept features, the sequence ids of the rows (None for padded states, whose rows map by position)
    and the number of rows per sequence (the time axis of padded states, 1 with sequence ids).
    """
    if not broadcast:
        if sequence_ids is not None or num_sequences is not None:
            raise ValueError("sequence_ids and num_sequences need broadcast=True.")
        return None, None, 1
    width = x.shape[-1] if x.ndim else 1
    if sequence_ids is not None:
        if x.ndim != 2 or sequence_ids.ndim != 1 or sequence_ids.shape[0] != x.shape[0]:
            raise ValueError(
                f"sequence_ids give the sequence of every row of (T_total, D) states, got ids of shape"
                f" {tuple(sequence_ids.shape)} for states of shape {tuple(x.shape)}."
            )
        if not isinstance(num_sequences, int) or isinstance(num_sequences, bool) or num_sequences < 0:
            raise ValueError(
                f"num_sequences, the number of sequences the ids index, is needed with sequence_ids, got {num_sequences!r}."
            )
        if sequence_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"sequence_ids must be int32 or int64, got {sequence_ids.dtype}.")
        if sequence_ids.device != x.device:
            raise ValueError(f"sequence_ids must be on the device of x, got {sequence_ids.device} and {x.device}.")
        if num_sequences == 0 and x.shape[0]:
            raise ValueError(f"{x.shape[0]} rows cannot belong to zero sequences.")
        # the kernel reads one id per row; a row whose id is outside [0, num_sequences) is dropped whole,
        # since a range check would cost a device sync
        sequence_ids = sequence_ids.contiguous()
        sequences, rows_per_sequence = num_sequences, 1
    else:
        if num_sequences is not None:
            raise ValueError("num_sequences needs sequence_ids.")
        if x.ndim < 3:
            raise ValueError(
                f"a broadcast mask needs padded states of shape (..., T, D) with a batch axis, got {tuple(x.shape)};"
                " sequence-packed states (T_total, D) need their packing context (sequence_ids)"
            )
        rows_per_sequence = x.shape[-2]
        sequences = x.numel() // (rows_per_sequence * width) if rows_per_sequence and width else 0
    if p <= 0.0:
        return None, None, 1
    return activation_dropout_mask((sequences, width), p, seed, x.device), sequence_ids, rows_per_sequence


class _ActivationDropoutLinear(torch.autograd.Function):
    """The second linear layer over the dropped-out activation, recomputed in the backward from the pre-activation."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: Tensor,
        activation: int,
        p: float,
        seed: Tensor,
        keep: Tensor | None,
        sequence_ids: Tensor | None,
        rows_per_sequence: int,
        weight: Tensor,
        bias: Tensor | None,
    ):
        hidden = torch.ops.nemo_asr.activation_dropout_fwd(
            x, activation, p, seed, keep, sequence_ids, rows_per_sequence
        )
        ctx.save_for_backward(x, seed, keep, sequence_ids, weight)
        ctx.activation, ctx.p, ctx.rows_per_sequence, ctx.has_bias = activation, p, rows_per_sequence, bias is not None
        return torch.nn.functional.linear(hidden, weight, bias)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, d_out: Tensor):
        x, seed, keep, sequence_ids, weight = ctx.saved_tensors
        needs_x, _, _, _, _, _, _, needs_weight, needs_bias = ctx.needs_input_grad
        d_out_rows = d_out.reshape(-1, d_out.shape[-1])
        # the matmuls run under the autocast state of the forward,
        # so a float32 weight meets a half-precision gradient as in the backward of a linear layer under autocast
        d_weight = d_bias = d_x = None
        if needs_weight:
            hidden = torch.ops.nemo_asr.activation_dropout_fwd(
                x, ctx.activation, ctx.p, seed, keep, sequence_ids, ctx.rows_per_sequence
            )
            d_weight = torch.matmul(d_out_rows.transpose(0, 1), hidden.reshape(-1, hidden.shape[-1]))
            d_weight = d_weight.to(weight.dtype)
        if needs_bias and ctx.has_bias:
            d_bias = d_out_rows.sum(dim=0).to(weight.dtype)
        if needs_x:
            d_hidden = torch.matmul(d_out, weight).to(x.dtype)
            d_x = torch.ops.nemo_asr.activation_dropout_bwd(
                x, d_hidden, ctx.activation, ctx.p, seed, keep, sequence_ids, ctx.rows_per_sequence
            )
        return d_x, None, None, None, None, None, None, d_weight, d_bias


def _launch(kernel, x: Tensor, keep: Tensor | None, sequence_ids: Tensor | None, rows_per_sequence: int, **tensors):
    """Launch a kernel over ``x``: flat tiles for the per-element mask, one program per row and column tile otherwise."""
    numel = x.numel()
    if not numel:
        return
    width = x.shape[-1] if x.ndim else 1
    common = dict(numel=numel, width=width, rows_per_sequence=rows_per_sequence, BLOCK=BLOCK, **tensors)
    if keep is None:
        kernel[(triton.cdiv(numel, BLOCK),)](
            Keep=None, SeqIds=None, num_sequences=0, BROADCAST=False, HAS_IDS=False, **common
        )
        return
    if keep.dtype != torch.bool or keep.ndim != 2 or keep.shape[1] != width:
        raise ValueError(
            f"keep must be a (sequences, {width}) bool matrix, got {keep.dtype} of shape {tuple(keep.shape)}."
        )
    rows = numel // width
    if sequence_ids is None and (
        rows_per_sequence <= 0 or rows % rows_per_sequence or rows // rows_per_sequence != keep.shape[0]
    ):
        raise ValueError(f"{rows} rows of {rows_per_sequence} per sequence do not match the {keep.shape[0]} masks.")
    grid = (rows, triton.cdiv(width, BLOCK))
    kernel[grid](
        Keep=keep.view(torch.uint8),
        SeqIds=sequence_ids,
        num_sequences=keep.shape[0],
        BROADCAST=True,
        HAS_IDS=sequence_ids is not None,
        **common,
    )


@torch.library.custom_op("nemo_asr::activation_dropout_fwd", mutates_args=(), device_types="cuda")
def _activation_dropout_fwd(
    x: Tensor,
    activation: int,
    p: float,
    seed: Tensor,
    keep: Tensor | None,
    sequence_ids: Tensor | None,
    rows_per_sequence: int,
) -> Tensor:
    # the input is made contiguous here as well: a compiled graph may feed the op any layout
    x = x.contiguous()
    out = torch.empty_like(x)
    _launch(
        _forward_kernel,
        x,
        keep,
        sequence_ids,
        rows_per_sequence,
        X=x,
        Out=out,
        Seed=seed,
        p=p,
        scale=1.0 / (1.0 - p),
        ACT=activation,
        DROPOUT=p > 0.0,
    )
    return out


@_activation_dropout_fwd.register_fake
def _activation_dropout_fwd_fake(x, activation, p, seed, keep, sequence_ids, rows_per_sequence):
    del activation, p, seed, keep, sequence_ids, rows_per_sequence
    # new_empty, not empty_like: the output is contiguous whatever the traced input layout
    return x.new_empty(tuple(x.shape))


@torch.library.custom_op("nemo_asr::activation_dropout_bwd", mutates_args=(), device_types="cuda")
def _activation_dropout_bwd(
    x: Tensor,
    d_out: Tensor,
    activation: int,
    p: float,
    seed: Tensor,
    keep: Tensor | None,
    sequence_ids: Tensor | None,
    rows_per_sequence: int,
) -> Tensor:
    x, d_out = x.contiguous(), d_out.contiguous()
    d_x = torch.empty_like(x)
    _launch(
        _backward_kernel,
        x,
        keep,
        sequence_ids,
        rows_per_sequence,
        X=x,
        DOut=d_out,
        DX=d_x,
        Seed=seed,
        p=p,
        scale=1.0 / (1.0 - p),
        ACT=activation,
        DROPOUT=p > 0.0,
    )
    return d_x


@_activation_dropout_bwd.register_fake
def _activation_dropout_bwd_fake(x, d_out, activation, p, seed, keep, sequence_ids, rows_per_sequence):
    del d_out, activation, p, seed, keep, sequence_ids, rows_per_sequence
    return x.new_empty(tuple(x.shape))


def _setup_context(ctx, inputs, output):
    del output
    x, activation, p, seed, keep, sequence_ids, rows_per_sequence = inputs
    # the identity's derivative is one: its backward needs the mask alone
    ctx.save_for_backward(None if activation == FUSED_ACTIVATIONS["identity"] else x, seed, keep, sequence_ids)
    ctx.activation, ctx.p, ctx.rows_per_sequence = activation, p, rows_per_sequence


def _autograd_backward(ctx, d_out):
    x, seed, keep, sequence_ids = ctx.saved_tensors
    if x is None:
        d_x = torch.ops.nemo_asr.activation_dropout_fwd(
            d_out, ctx.activation, ctx.p, seed, keep, sequence_ids, ctx.rows_per_sequence
        )
    else:
        d_x = torch.ops.nemo_asr.activation_dropout_bwd(
            x, d_out, ctx.activation, ctx.p, seed, keep, sequence_ids, ctx.rows_per_sequence
        )
    return d_x, None, None, None, None, None, None


torch.library.register_autograd("nemo_asr::activation_dropout_fwd", _autograd_backward, setup_context=_setup_context)


@triton.jit
def _activation(x, ACT: tl.constexpr):
    """The activation of a float32 tile and its derivative."""
    if ACT == 0:
        y = x
        dy = tl.full(x.shape, 1.0, tl.float32)
    elif ACT == 1:
        y = tl.maximum(x, 0.0)
        dy = tl.where(x > 0.0, 1.0, 0.0)
    elif ACT == 2:
        r = tl.maximum(x, 0.0)
        y = r * r
        dy = 2.0 * r
    elif ACT == 3:
        s = tl.sigmoid(x)
        y = x * s
        dy = s * (1.0 + x * (1.0 - s))
    else:
        tl.static_assert(ACT == 4, "unknown activation id")
        # the exact GELU, x * Phi(x), with the normal density in its derivative
        cdf = 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
        y = x * cdf
        dy = cdf + x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
    return y, dy


@triton.jit
def _tile(
    Seed,
    Keep,
    SeqIds,
    numel,
    width,
    rows_per_sequence,
    num_sequences,
    p,
    DROPOUT: tl.constexpr,
    BROADCAST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The element offsets of this program, their bounds mask and their kept flags.

    Per-element masks run over flat tiles, with the Philox hash of the element index as the mask.
    Broadcast masks run one program per row and column tile and read the row's sequence mask:
    the sequence is the row's id, or its row index divided by the rows per sequence for padded states.
    """
    if BROADCAST:
        # the broadcast path always carries a mask: _launch takes the flat path for dropout zero
        tl.static_assert(DROPOUT, "the broadcast path needs a mask")
        row = tl.program_id(0).to(tl.int64)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        offsets = row * width + columns
        mask = columns < width
        if HAS_IDS:
            sequence = tl.load(SeqIds + row).to(tl.int64)
        else:
            sequence = row // rows_per_sequence
        # a row whose id lies outside the mask matrix is dropped whole instead of read out of bounds
        in_range = (sequence >= 0) & (sequence < num_sequences)
        keep = tl.load(Keep + sequence * width + columns, mask=mask & in_range, other=0) != 0
    else:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        if DROPOUT:
            keep = tl.rand(tl.load(Seed), offsets) >= p
        else:
            keep = mask
    return offsets, mask, keep


@triton.jit
def _forward_kernel(
    X,
    Out,
    Seed,
    Keep,
    SeqIds,
    numel,
    width,
    rows_per_sequence,
    num_sequences,
    p,
    scale,
    ACT: tl.constexpr,
    DROPOUT: tl.constexpr,
    BROADCAST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets, mask, keep = _tile(
        Seed, Keep, SeqIds, numel, width, rows_per_sequence, num_sequences, p, DROPOUT, BROADCAST, HAS_IDS, BLOCK
    )
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    y, _ = _activation(x, ACT)
    if DROPOUT:
        y = tl.where(keep, y * scale, 0.0)
    tl.store(Out + offsets, y.to(Out.dtype.element_ty), mask=mask)


@triton.jit
def _backward_kernel(
    X,
    DOut,
    DX,
    Seed,
    Keep,
    SeqIds,
    numel,
    width,
    rows_per_sequence,
    num_sequences,
    p,
    scale,
    ACT: tl.constexpr,
    DROPOUT: tl.constexpr,
    BROADCAST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets, mask, keep = _tile(
        Seed, Keep, SeqIds, numel, width, rows_per_sequence, num_sequences, p, DROPOUT, BROADCAST, HAS_IDS, BLOCK
    )
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    d_out = tl.load(DOut + offsets, mask=mask, other=0.0).to(tl.float32)
    _, dy = _activation(x, ACT)
    d_x = dy * d_out
    if DROPOUT:
        d_x = tl.where(keep, d_x * scale, 0.0)
    tl.store(DX + offsets, d_x.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def _mask_kernel(Mask, Seed, numel, p, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    tl.store(Mask + offsets, (tl.rand(tl.load(Seed), offsets) >= p).to(tl.uint8), mask=mask)
