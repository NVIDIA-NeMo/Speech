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
With ``broadcast=True`` the mask has the width of the last axis alone and every position of the leading axes
(batch and time) shares it, as RETURNN's ``rf_dropout_broadcast`` draws its feed-forward and residual masks:
the kernels read the kept features from a vector of ``x.shape[-1]`` booleans,
``activation_dropout_mask((x.shape[-1],), p, seed, device)``, instead of hashing every element.
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
    x: Tensor, activation: str, p: float, *, broadcast: bool = False, seed: Tensor | None = None
) -> Tensor:
    """``dropout(activation(x), p)`` of a training step, with ``x`` and the mask's seed saved for the backward.

    The identity activation saves the seed (and the broadcast mask) alone: its backward is the dropout of the gradient.

    Args:
        x: the pre-activation, any shape; float32, bfloat16 or float16 on CUDA.
        activation: a name in ``FUSED_ACTIVATIONS`` (``identity``, ``relu``, ``squared_relu``, ``silu``, ``gelu``).
        p: dropout probability in ``[0, 1)``; the kept elements are scaled by ``1 / (1 - p)``.
        broadcast: one mask over the last axis, shared by every position of the leading axes,
            instead of a mask per element.
        seed: one-element integer tensor with the Philox seed of the mask, taken modulo ``2**31``;
            drawn on the device of ``x`` by default, which gives a fresh mask on every CUDA-graph replay.

    Returns:
        A tensor of the shape and dtype of ``x``.
    """
    seed = _checked_seed(x, activation, p, seed, broadcast)
    keep = _broadcast_keep(x, p, seed) if broadcast else None
    return torch.ops.nemo_asr.activation_dropout_fwd(x, FUSED_ACTIVATIONS[activation], p, seed, keep)


def activation_dropout_linear(
    x: Tensor,
    activation: str,
    p: float,
    weight: Tensor,
    bias: Tensor | None,
    *,
    broadcast: bool = False,
    seed: Tensor | None = None,
) -> Tensor:
    """``linear(dropout(activation(x), p), weight, bias)`` of a training step, with ``x`` saved for the backward.

    The linear layer's input, the dropped-out activation, is recomputed from ``x`` and the mask's seed
    in the backward instead of being saved.

    Args:
        x: the pre-activation, shape ``(..., D_in)``; float32, bfloat16 or float16 on CUDA.
        activation: a name in ``FUSED_ACTIVATIONS``.
        p: dropout probability in ``[0, 1)``.
        weight: the linear layer's weight, shape ``(D_out, D_in)``.
        bias: the linear layer's bias, shape ``(D_out,)``, or None.
        broadcast: one mask over ``D_in``, shared by every row of ``x``, instead of a mask per element.
        seed: one-element integer tensor with the Philox seed of the mask, taken modulo ``2**31``,
            drawn on the device by default.

    Returns:
        A tensor of shape ``(..., D_out)`` in the dtype the linear layer produces.
    """
    seed = _checked_seed(x, activation, p, seed, broadcast)
    if weight.ndim != 2 or weight.shape[1] != x.shape[-1]:
        raise ValueError(f"weight must have shape (D_out, {x.shape[-1]}), got {tuple(weight.shape)}.")
    keep = _broadcast_keep(x, p, seed) if broadcast else None
    return _ActivationDropoutLinear.apply(x, FUSED_ACTIVATIONS[activation], p, seed, keep, weight, bias)


def activation_dropout_mask(shape: tuple[int, ...], p: float, seed: Tensor, device: torch.device) -> Tensor:
    """The kept elements of ``activation_dropout`` for this seed, a bool tensor of ``shape``, for exact comparisons."""
    seed = seed.to(device=device, dtype=torch.int32)
    mask = torch.empty(shape, dtype=torch.bool, device=device)
    numel = mask.numel()
    if numel:
        _mask_kernel[(triton.cdiv(numel, BLOCK),)](mask, seed, numel, p, BLOCK=BLOCK)
    return mask


def _checked_seed(x: Tensor, activation: str, p: float, seed: Tensor | None, broadcast: bool = False) -> Tensor:
    """Validate the inputs of the kernels and return the int32 seed tensor on the device of ``x``."""
    if not x.is_cuda:
        raise RuntimeError("activation_dropout requires a CUDA tensor.")
    if x.dtype not in ACTIVATION_DROPOUT_DTYPES:
        raise TypeError(f"x must be float32, bfloat16 or float16, got {x.dtype}.")
    if broadcast and x.ndim == 0:
        raise ValueError("a broadcast mask needs an axis to span; x has none.")
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


def _broadcast_keep(x: Tensor, p: float, seed: Tensor) -> Tensor | None:
    """The kept features of a broadcast mask, ``x.shape[-1]`` booleans, or None without dropout."""
    if p <= 0.0:
        return None
    return activation_dropout_mask((x.shape[-1],), p, seed, x.device)


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
        weight: Tensor,
        bias: Tensor | None,
    ):
        hidden = torch.ops.nemo_asr.activation_dropout_fwd(x, activation, p, seed, keep)
        ctx.save_for_backward(x, seed, keep, weight)
        ctx.activation, ctx.p, ctx.has_bias = activation, p, bias is not None
        return torch.nn.functional.linear(hidden, weight, bias)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, d_out: Tensor):
        x, seed, keep, weight = ctx.saved_tensors
        needs_x, _, _, _, _, needs_weight, needs_bias = ctx.needs_input_grad
        d_out_rows = d_out.reshape(-1, d_out.shape[-1])
        # the matmuls run under the autocast state of the forward,
        # so a float32 weight meets a half-precision gradient as in the backward of a linear layer under autocast
        d_weight = d_bias = d_x = None
        if needs_weight:
            hidden = torch.ops.nemo_asr.activation_dropout_fwd(x, ctx.activation, ctx.p, seed, keep)
            d_weight = torch.matmul(d_out_rows.transpose(0, 1), hidden.reshape(-1, hidden.shape[-1]))
            d_weight = d_weight.to(weight.dtype)
        if needs_bias and ctx.has_bias:
            d_bias = d_out_rows.sum(dim=0).to(weight.dtype)
        if needs_x:
            d_hidden = torch.matmul(d_out, weight).to(x.dtype)
            d_x = torch.ops.nemo_asr.activation_dropout_bwd(x, d_hidden, ctx.activation, ctx.p, seed, keep)
        return d_x, None, None, None, None, d_weight, d_bias


def _kernel_mask(x: Tensor, keep: Tensor | None) -> dict:
    """The mask arguments of the kernels: the broadcast vector and its width, or the per-element hash."""
    if keep is None:
        return dict(Keep=None, width=1, BROADCAST=False, ALIGNED=True)
    width = x.shape[-1]
    if keep.numel() != width or keep.dtype != torch.bool:
        raise ValueError(f"keep must hold {width} booleans, got {keep.dtype} of shape {tuple(keep.shape)}.")
    return dict(Keep=keep, width=width, BROADCAST=True, ALIGNED=width % BLOCK == 0)


@torch.library.custom_op("nemo_asr::activation_dropout_fwd", mutates_args=(), device_types="cuda")
def _activation_dropout_fwd(x: Tensor, activation: int, p: float, seed: Tensor, keep: Tensor | None) -> Tensor:
    # the input is made contiguous here as well: a compiled graph may feed the op any layout
    x = x.contiguous()
    out = torch.empty_like(x)
    numel = x.numel()
    if numel:
        _forward_kernel[(triton.cdiv(numel, BLOCK),)](
            X=x,
            Out=out,
            Seed=seed,
            numel=numel,
            p=p,
            scale=1.0 / (1.0 - p),
            ACT=activation,
            DROPOUT=p > 0.0,
            BLOCK=BLOCK,
            **_kernel_mask(x, keep),
        )
    return out


@_activation_dropout_fwd.register_fake
def _activation_dropout_fwd_fake(x, activation, p, seed, keep):
    del activation, p, seed, keep
    # new_empty, not empty_like: the output is contiguous whatever the traced input layout
    return x.new_empty(tuple(x.shape))


@torch.library.custom_op("nemo_asr::activation_dropout_bwd", mutates_args=(), device_types="cuda")
def _activation_dropout_bwd(
    x: Tensor, d_out: Tensor, activation: int, p: float, seed: Tensor, keep: Tensor | None
) -> Tensor:
    x, d_out = x.contiguous(), d_out.contiguous()
    d_x = torch.empty_like(x)
    numel = x.numel()
    if numel:
        _backward_kernel[(triton.cdiv(numel, BLOCK),)](
            X=x,
            DOut=d_out,
            DX=d_x,
            Seed=seed,
            numel=numel,
            p=p,
            scale=1.0 / (1.0 - p),
            ACT=activation,
            DROPOUT=p > 0.0,
            BLOCK=BLOCK,
            **_kernel_mask(x, keep),
        )
    return d_x


@_activation_dropout_bwd.register_fake
def _activation_dropout_bwd_fake(x, d_out, activation, p, seed, keep):
    del d_out, activation, p, seed, keep
    return x.new_empty(tuple(x.shape))


def _setup_context(ctx, inputs, output):
    del output
    x, activation, p, seed, keep = inputs
    # the identity's derivative is one: its backward needs the mask alone
    ctx.save_for_backward(None if activation == FUSED_ACTIVATIONS["identity"] else x, seed, keep)
    ctx.activation, ctx.p = activation, p


def _autograd_backward(ctx, d_out):
    x, seed, keep = ctx.saved_tensors
    if x is None:
        d_x = torch.ops.nemo_asr.activation_dropout_fwd(d_out, ctx.activation, ctx.p, seed, keep)
    else:
        d_x = torch.ops.nemo_asr.activation_dropout_bwd(x, d_out, ctx.activation, ctx.p, seed, keep)
    return d_x, None, None, None, None


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
def _keep(Seed, Keep, offsets, width, p, BROADCAST: tl.constexpr, ALIGNED: tl.constexpr, BLOCK: tl.constexpr):
    """The kept elements of the tile: the Philox hash of the element index, or the broadcast mask at its column."""
    if BROADCAST:
        # the tile's first column from one 64-bit modulo per program; the rest counts on in 32 bits
        first = ((tl.program_id(0).to(tl.int64) * BLOCK) % width).to(tl.int32)
        columns = first + tl.arange(0, BLOCK)
        if not ALIGNED:
            # a tile that crosses the end of a row wraps; a width that divides the tile never does
            columns = columns % width
        return tl.load(Keep + columns)
    return tl.rand(tl.load(Seed), offsets) >= p


@triton.jit
def _forward_kernel(
    X,
    Out,
    Seed,
    Keep,
    numel,
    width,
    p,
    scale,
    ACT: tl.constexpr,
    DROPOUT: tl.constexpr,
    BROADCAST: tl.constexpr,
    ALIGNED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    y, _ = _activation(x, ACT)
    if DROPOUT:
        y = tl.where(_keep(Seed, Keep, offsets, width, p, BROADCAST, ALIGNED, BLOCK), y * scale, 0.0)
    tl.store(Out + offsets, y.to(Out.dtype.element_ty), mask=mask)


@triton.jit
def _backward_kernel(
    X,
    DOut,
    DX,
    Seed,
    Keep,
    numel,
    width,
    p,
    scale,
    ACT: tl.constexpr,
    DROPOUT: tl.constexpr,
    BROADCAST: tl.constexpr,
    ALIGNED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    d_out = tl.load(DOut + offsets, mask=mask, other=0.0).to(tl.float32)
    _, dy = _activation(x, ACT)
    d_x = dy * d_out
    if DROPOUT:
        d_x = tl.where(_keep(Seed, Keep, offsets, width, p, BROADCAST, ALIGNED, BLOCK), d_x * scale, 0.0)
    tl.store(DX + offsets, d_x.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def _mask_kernel(Mask, Seed, numel, p, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    tl.store(Mask + offsets, (tl.rand(tl.load(Seed), offsets) >= p).to(tl.int1), mask=mask)
