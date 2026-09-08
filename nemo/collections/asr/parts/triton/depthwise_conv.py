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

"""Channels-last depthwise conv2d with fused bias and per-sample length masking.

Each output position reads a KERNEL_H x KERNEL_W window. With taps (kh, kw):

    in_y = y * STRIDE_H - PAD_H + kh
    in_x = x * STRIDE_W - PAD_W + kw

    out[n, y, x, c] = bias[c] + sum over (kh, kw) of  weight[c, kh, kw] * input[n, in_y, in_x, c]

Input and output are NHWC. ``weight`` keeps nn.Conv2d's own ``(channels, 1, KERNEL_H, KERNEL_W)``
layout which the kernels index as ``(channels, taps)`` with ``tap = kh * KERNEL_W + kw``.

Kernel size, stride and padding are compile-time constants, and the start and end edges pad
independently, so this covers any depthwise geometry ``nn.Conv2d`` expresses.

A program computes POSITION_BLOCK windows at once, CHANNEL_BLOCK channels wide. A *position* is
the output pixel a window produces, its (batch, y, x) flattened onto one axis as
``batch * height * width + y * width + x``; the tile is [POSITION_BLOCK, CHANNEL_BLOCK], windows
down the rows and channels across the columns.

The loop runs over taps, not over windows. At each tap every window in the tile picks up one
term:

    acc += weight[tap] * the input pixel each window reads through that tap

``weight[tap]`` is a single value per channel, shared by all POSITION_BLOCK windows; the pixel it
multiplies differs per window, because the windows sit STRIDE_H and STRIDE_W apart. The tile holds
finished outputs once every tap has been applied.

Windows overlap whenever KERNEL_H exceeds STRIDE_H, so pixels read for one window are reread by
its neighbour, and keeping a block of windows in flight turns those rereads into cache hits.
Channels being contiguous in NHWC makes each row of the tile a single wide load.

The bias is added to the tile before the store, and ``in_lengths``/``out_lengths`` enter the load
and store masks that already bound the reads.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _forward_configs():
    return [
        triton.Config({"POSITION_BLOCK": positions, "CHANNEL_BLOCK": channels}, num_warps=warps)
        for positions in (2, 4, 8)
        for channels in (128, 256)
        for warps in (2, 4)
    ]


def _weight_grad_configs():
    # The accumulator is [TAPS_PADDED, CHANNEL_BLOCK]; positions are reduced inside the
    # loop, so POSITION_BLOCK only sets how many output positions each iteration reads.
    return [
        triton.Config({"POSITION_BLOCK": positions, "CHANNEL_BLOCK": channels, "TILE_SPLITS": splits}, num_warps=warps)
        for positions in (16, 32)
        for channels in (64, 128, 256)
        for splits in (512, 2048)
        for warps in (2, 4)
    ]


@triton.jit
def _decode_position(position, extent_y, extent_x):
    """Split a flat position back into (batch, y, x)."""
    per_image = extent_y * extent_x
    return position // per_image, (position % per_image) // extent_x, position % extent_x


@triton.jit
def _load_tap(weight_ptr, channel, channel_mask, tap: tl.constexpr, TAPS: tl.constexpr):
    """weight is (channels, KERNEL_H * KERNEL_W), channel-major as nn.Conv2d stores it."""
    return tl.load(weight_ptr + channel * TAPS + tap, mask=channel_mask, other=0.0).to(tl.float32)


@triton.jit
def _load_input_at_tap(
    input_ptr,
    batch,
    out_y,
    out_x,
    channel,
    position_mask,
    channel_mask,
    in_height,
    in_width,
    channels,
    valid_height,
    kh: tl.constexpr,
    kw: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
):
    """The input each output position reads through tap (kh, kw); zero at out of bounds and past
    ``valid_height``."""
    in_y = out_y * STRIDE_H - PAD_H + kh
    in_x = out_x * STRIDE_W - PAD_W + kw
    valid = (in_y >= 0) & (in_y < in_height) & (in_y < valid_height) & (in_x >= 0) & (in_x < in_width) & position_mask
    offset = batch * (in_height * in_width * channels) + in_y * (in_width * channels) + in_x * channels
    return tl.load(
        input_ptr + offset[:, None] + channel[None, :], mask=valid[:, None] & channel_mask[None, :], other=0.0
    ).to(tl.float32)


@triton.autotune(configs=_forward_configs(), key=["channels", "in_width"])
@triton.jit
def _forward_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_length_ptr,
    out_length_ptr,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    num_output_positions,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    TAPS: tl.constexpr,
    POSITION_BLOCK: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
):
    position = tl.program_id(0) * POSITION_BLOCK + tl.arange(0, POSITION_BLOCK)
    channel = tl.program_id(1) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    position_mask = position < num_output_positions
    channel_mask = channel < channels

    batch, out_y, out_x = _decode_position(position, out_height, out_width)
    valid_in_height = tl.load(in_length_ptr + batch, mask=position_mask, other=0)
    valid_out_height = tl.load(out_length_ptr + batch, mask=position_mask, other=0)

    total = tl.load(bias_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)[None, :] + tl.zeros(
        [POSITION_BLOCK, CHANNEL_BLOCK], tl.float32
    )
    for kh in tl.static_range(KERNEL_H):
        for kw in tl.static_range(KERNEL_W):
            total += _load_tap(weight_ptr, channel, channel_mask, kh * KERNEL_W + kw, TAPS) * _load_input_at_tap(
                input_ptr,
                batch,
                out_y,
                out_x,
                channel,
                position_mask,
                channel_mask,
                in_height,
                in_width,
                channels,
                valid_in_height,
                kh,
                kw,
                STRIDE_H,
                STRIDE_W,
                PAD_H,
                PAD_W,
            )

    # zeros are stored past each sample's valid height
    total = tl.where((out_y < valid_out_height)[:, None], total, 0.0)
    tl.store(
        output_ptr + (position * channels)[:, None] + channel[None, :],
        total.to(output_ptr.dtype.element_ty),
        mask=position_mask[:, None] & channel_mask[None, :],
    )


@triton.autotune(configs=_forward_configs(), key=["channels", "in_width"])
@triton.jit
def _input_grad_kernel(
    grad_output_ptr,
    weight_ptr,
    grad_input_ptr,
    in_length_ptr,
    out_length_ptr,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    num_input_positions,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    TAPS: tl.constexpr,
    POSITION_BLOCK: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
):
    """Gradient with respect to the input.

    Each input pixel fed several outputs in the forward pass, so its gradient is the sum of
    what comes back from each, weighted by the tap that carried it. A program owns a block of
    input pixels and walks the taps, accumulating those contributions.
    """
    position = tl.program_id(0) * POSITION_BLOCK + tl.arange(0, POSITION_BLOCK)
    channel = tl.program_id(1) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    position_mask = position < num_input_positions
    channel_mask = channel < channels

    batch, in_y, in_x = _decode_position(position, in_height, in_width)
    valid_in_height = tl.load(in_length_ptr + batch, mask=position_mask, other=0)
    valid_out_height = tl.load(out_length_ptr + batch, mask=position_mask, other=0)

    total = tl.zeros([POSITION_BLOCK, CHANNEL_BLOCK], tl.float32)
    for kh in tl.static_range(KERNEL_H):
        reach_y = in_y + PAD_H - kh
        out_y = reach_y // STRIDE_H
        row_ok = (reach_y % STRIDE_H == 0) & (out_y >= 0) & (out_y < out_height) & (out_y < valid_out_height)
        for kw in tl.static_range(KERNEL_W):
            reach_x = in_x + PAD_W - kw
            out_x = reach_x // STRIDE_W
            valid = row_ok & (reach_x % STRIDE_W == 0) & (out_x >= 0) & (out_x < out_width) & position_mask
            offset = batch * (out_height * out_width * channels) + out_y * (out_width * channels) + out_x * channels
            grad = tl.load(
                grad_output_ptr + offset[:, None] + channel[None, :],
                mask=valid[:, None] & channel_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            total += _load_tap(weight_ptr, channel, channel_mask, kh * KERNEL_W + kw, TAPS) * grad

    total = tl.where((in_y < valid_in_height)[:, None], total, 0.0)
    tl.store(
        grad_input_ptr + (position * channels)[:, None] + channel[None, :],
        total.to(grad_input_ptr.dtype.element_ty),
        mask=position_mask[:, None] & channel_mask[None, :],
    )


@triton.autotune(
    configs=_weight_grad_configs(),
    key=["channels", "in_width"],
    reset_to_zero=["grad_weight_ptr", "grad_bias_ptr"],
)
@triton.jit
def _weight_grad_kernel(
    input_ptr,
    grad_output_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    in_length_ptr,
    out_length_ptr,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    num_output_positions,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    TAPS: tl.constexpr,
    TAPS_PADDED: tl.constexpr,
    POSITION_BLOCK: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
    TILE_SPLITS: tl.constexpr,
):
    """Weight and bias gradients, summed over every output position.

    One program handles all KERNEL_H x KERNEL_W taps for a block of channels, so each
    ``grad`` value it loads is used by all of them. Triton cannot index into a block with a
    runtime value, so the per-tap accumulators share one [TAPS_PADDED, CHANNEL_BLOCK] tile
    and each tap is picked out with a one-hot ``tl.where``.
    """
    channel = tl.program_id(0) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    channel_mask = channel < channels
    tap_index = tl.arange(0, TAPS_PADDED)

    grad_weight_total = tl.zeros([TAPS_PADDED, CHANNEL_BLOCK], tl.float32)
    grad_bias_total = tl.zeros([CHANNEL_BLOCK], tl.float32)

    for base in range(tl.program_id(1) * POSITION_BLOCK, num_output_positions, TILE_SPLITS * POSITION_BLOCK):
        position = base + tl.arange(0, POSITION_BLOCK)
        position_mask = position < num_output_positions
        batch, out_y, out_x = _decode_position(position, out_height, out_width)
        position_mask = position_mask & (out_y < tl.load(out_length_ptr + batch, mask=position_mask, other=0))
        valid_in_height = tl.load(in_length_ptr + batch, mask=position_mask, other=0)
        grad = tl.load(
            grad_output_ptr + (position * channels)[:, None] + channel[None, :],
            mask=position_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_bias_total += tl.sum(grad, 0)

        for kh in tl.static_range(KERNEL_H):
            for kw in tl.static_range(KERNEL_W):
                tap_total = tl.sum(
                    grad
                    * _load_input_at_tap(
                        input_ptr,
                        batch,
                        out_y,
                        out_x,
                        channel,
                        position_mask,
                        channel_mask,
                        in_height,
                        in_width,
                        channels,
                        valid_in_height,
                        kh,
                        kw,
                        STRIDE_H,
                        STRIDE_W,
                        PAD_H,
                        PAD_W,
                    ),
                    0,
                )
                grad_weight_total += tl.where((tap_index == kh * KERNEL_W + kw)[:, None], tap_total[None, :], 0.0)

    # grad_weight is (TAPS, channels), so each atomic row is a contiguous channel run
    tl.atomic_add(
        grad_weight_ptr + tap_index[:, None] * channels + channel[None, :],
        grad_weight_total,
        mask=(tap_index < TAPS)[:, None] & channel_mask[None, :],
    )
    tl.atomic_add(grad_bias_ptr + channel, grad_bias_total, mask=channel_mask)


class _DepthwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight, bias, in_lengths, out_lengths, geometry, autocast_dtype):
        (
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        ) = geometry
        dtype = features.dtype if autocast_dtype is None else autocast_dtype
        features_cast = features.contiguous().to(dtype)
        weight_cast, bias_cast = weight.to(dtype), bias.to(dtype)
        num_output_positions = batch_size * out_height * out_width
        output = torch.empty((batch_size, out_height, out_width, channels), device=features.device, dtype=dtype)

        def grid(meta):
            return (
                triton.cdiv(num_output_positions, meta["POSITION_BLOCK"]),
                triton.cdiv(channels, meta["CHANNEL_BLOCK"]),
            )

        _forward_kernel[grid](
            features_cast,
            weight_cast,
            bias_cast,
            output,
            in_lengths,
            out_lengths,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            num_output_positions,
            KERNEL_H=kernel_h,
            KERNEL_W=kernel_w,
            STRIDE_H=stride_h,
            STRIDE_W=stride_w,
            PAD_H=pad_h,
            PAD_W=pad_w,
            TAPS=kernel_h * kernel_w,
        )
        # Cloned because callers own the lengths and edit them in place; backward needs the originals.
        ctx.save_for_backward(features_cast, weight_cast, in_lengths.clone(), out_lengths.clone())
        ctx.geometry = geometry
        ctx.param_dtypes = (weight.dtype, bias.dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, weight, in_lengths, out_lengths = ctx.saved_tensors
        (
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        ) = ctx.geometry
        taps = kernel_h * kernel_w
        grad_output = grad_output.contiguous()
        num_input_positions = batch_size * in_height * in_width
        num_output_positions = batch_size * out_height * out_width
        grad_input = torch.empty(
            (batch_size, in_height, in_width, channels), device=grad_output.device, dtype=grad_output.dtype
        )
        grad_weight = torch.zeros((taps, channels), device=grad_output.device, dtype=torch.float32)
        grad_bias = torch.zeros((channels,), device=grad_output.device, dtype=torch.float32)

        def input_grid(meta):
            return (
                triton.cdiv(num_input_positions, meta["POSITION_BLOCK"]),
                triton.cdiv(channels, meta["CHANNEL_BLOCK"]),
            )

        def weight_grid(meta):
            return triton.cdiv(channels, meta["CHANNEL_BLOCK"]), meta["TILE_SPLITS"]

        geometry_constexprs = dict(
            KERNEL_H=kernel_h,
            KERNEL_W=kernel_w,
            STRIDE_H=stride_h,
            STRIDE_W=stride_w,
            PAD_H=pad_h,
            PAD_W=pad_w,
            TAPS=taps,
        )
        _input_grad_kernel[input_grid](
            grad_output,
            weight,
            grad_input,
            in_lengths,
            out_lengths,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            num_input_positions,
            **geometry_constexprs,
        )
        _weight_grad_kernel[weight_grid](
            features,
            grad_output,
            grad_weight,
            grad_bias,
            in_lengths,
            out_lengths,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            num_output_positions,
            TAPS_PADDED=triton.next_power_of_2(taps),
            **geometry_constexprs,
        )

        weight_dtype, bias_dtype = ctx.param_dtypes
        return (
            grad_input,
            grad_weight.t().reshape(channels, 1, kernel_h, kernel_w).to(weight_dtype),
            grad_bias.to(bias_dtype),
            None,  # in_lengths
            None,  # out_lengths
            None,  # geometry
            None,  # autocast_dtype
        )


def dw_conv2d(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int | tuple[int, int],
    pad_start: int | tuple[int, int],
    pad_end: int | tuple[int, int],
    in_lengths: torch.Tensor,
    out_lengths: torch.Tensor,
) -> torch.Tensor:
    """Channels-last depthwise conv2d with the bias and the length masking fused in.

    ``in_lengths``/``out_lengths`` fold into the load and store masks that already exist for
    bounds. Height is the masked axis.

    Args:
        features: ``(batch, in_height, in_width, channels)``, channels-last.
        weight: ``(channels, 1, kernel_h, kernel_w)``.
        bias: ``(channels,)``.
        stride: int or ``(stride_h, stride_w)``.
        pad_start: int or ``(pad_h, pad_w)``, padding at the start of each axis.
        pad_end: int or ``(pad_h, pad_w)``, padding at the end.
        in_lengths: ``(batch,)`` integer valid input heights.
        out_lengths: ``(batch,)`` integer valid output heights.

    Returns:
        ``(batch, out_height, out_width, channels)``.

    Raises:
        RuntimeError: if the inputs are not on a CUDA device.
    """
    if not features.is_cuda:
        raise RuntimeError("The depthwise convolution kernel requires CUDA tensors")
    batch_size, in_height, in_width, channels = features.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (pad_start, pad_start) if isinstance(pad_start, int) else pad_start
    end_h, end_w = (pad_end, pad_end) if isinstance(pad_end, int) else pad_end
    out_height = (in_height + pad_h + end_h - kernel_h) // stride_h + 1
    out_width = (in_width + pad_w + end_w - kernel_w) // stride_w + 1
    geometry = (
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    )
    # Autocast only rewrites aten ops, never a Triton launch, so the cast is done by hand.
    autocast_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else None
    with torch.autocast("cuda", enabled=False):
        return _DepthwiseConv2d.apply(features, weight, bias, in_lengths, out_lengths, geometry, autocast_dtype)
