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

"""Fused conv -> ReLU -> depthwise for FastConformer's ``dw_striding`` subsampling.

Both convolutions are 3x3 stride 2 over (time, frequency). With taps (kt, kf) and start padding P:

    relu_out[c, t, f] = relu( conv_bias[c] + sum over (kt, kf) of W_conv[c, kt, kf] * mel[2t + kt - P, 2f + kf - P] )
    out[c, t, f]      = depth_bias[c] + sum over (kt, kf) of W_depth[c, kt, kf] * relu_out[c, 2t + kt - P, 2f + kf - P]

One program emits two adjacent frequency bins for CHANNEL_BLOCK channels, running

    conv0:      mel 7x11      ->  relu_out 3x5
    depthwise:  relu_out 3x5  ->  two output bins

where each extent follows from the stage after it: n consecutive outputs along an axis need
KERNEL + (n-1) * STRIDE inputs, applied at each stage.

conv0 is a matrix product. Each of the 15 relu_out values is a 9-term dot product of conv0's taps
with 9 mel values, and every channel sums the same 9 mel values, differing only in its taps:

    conv_taps [CHANNEL_BLOCK x 9] @ feats [9 x 15] = relu_out [CHANNEL_BLOCK x 15]

The depthwise is not: each channel reads only its own slice of ``relu_out``, so no operand is
shared across channels and there is nothing to contract. ``taps_bin0`` and ``taps_bin1`` hold its
weights positioned at the columns each bin reads, zero elsewhere, so a bin is the window
multiplied by one vector and summed.

Only the two output values reach memory, channels-last for the 1x1 convolution that follows.
``relu_out`` is the largest tensor in the stage and is never written.

Backward mirrors it:

    grad_relu_out     = (grad_bin0*taps_bin0 + grad_bin1*taps_bin1) * (relu_out > 0)
    grad_conv_weight += dot(grad_relu_out, feats^T)
    acc_bin0         += grad_bin0 * relu_out          outer products -> grad depthwise
    acc_bin1         += grad_bin1 * relu_out

``mel`` is a leaf, so it gets no gradient. ``grad_conv_bias`` comes out of the same dot as
``grad_conv_weight``, in its column TAPS; see the backward kernel.

The 9 taps and 15 positions are padded to PAD, the next power of two, as ``tl.arange`` and
``tl.dot`` require, and the surplus is masked.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The geometry both kernels hardcode in their index arithmetic. Names, not knobs. They must be
# ``tl.constexpr``: @triton.jit rejects a plain int global. ``.value`` unwraps one for the
# tensor arithmetic that would otherwise return a wrapped constexpr.
KERNEL = tl.constexpr(3)
STRIDE = tl.constexpr(2)
NUM_BINS = tl.constexpr(2)  # frequency bins one program emits together
WINDOW_COLS = tl.constexpr(KERNEL.value + (NUM_BINS.value - 1) * STRIDE.value)  # columns they span
WINDOW_SIZE = tl.constexpr(KERNEL.value * WINDOW_COLS.value)
TAPS = tl.constexpr(KERNEL.value * KERNEL.value)
PAD = tl.constexpr(1 << (max(TAPS.value, WINDOW_SIZE.value) - 1).bit_length())  # a power of two


def _forward_configs():
    return [
        triton.Config({"CHANNEL_BLOCK": block, "TIME_ROWS": rows}, num_warps=warps)
        for block in (64, 128)
        for rows in (4, 8)
        for warps in (2, 4)
    ]


def _backward_configs():
    return [
        triton.Config({"CHANNEL_BLOCK": block, "TIME_ROWS": rows, "TILE_SPLITS": splits}, num_warps=warps)
        for block in (64, 128)
        for rows in (1, 2, 4)
        for splits in (1024, 4096)
        for warps in (2, 4)
    ]


@triton.jit
def _window_cells(out_row, bin0_col, relu_out_len, relu_out_freq, pad_start):
    """Where each window position sits in `relu_out`, and whether that cell exists."""
    window_pos = tl.arange(0, PAD)
    row = STRIDE * out_row + window_pos // WINDOW_COLS - pad_start
    col = STRIDE * bin0_col + window_pos % WINDOW_COLS - pad_start
    return (
        row,
        col,
        (row >= 0) & (row < relu_out_len) & (col >= 0) & (col < relu_out_freq) & (window_pos < WINDOW_SIZE),
    )


@triton.jit
def _load_params(conv_weight_ptr, conv_bias_ptr, taps_bin0_ptr, taps_bin1_ptr, channel, channel_mask):
    """conv0's taps and bias, and each bin's depthwise taps."""
    tap = tl.arange(0, PAD)
    window_pos = tl.arange(0, PAD)
    conv_taps = tl.load(
        conv_weight_ptr + channel[:, None] * TAPS + tap[None, :],
        mask=(tap < TAPS)[None, :] & channel_mask[:, None],
        other=0.0,
    )
    conv_bias = tl.load(conv_bias_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
    taps_bin0 = tl.load(
        taps_bin0_ptr + channel[:, None] * PAD + window_pos[None, :], mask=channel_mask[:, None], other=0.0
    ).to(tl.float32)
    taps_bin1 = tl.load(
        taps_bin1_ptr + channel[:, None] * PAD + window_pos[None, :], mask=channel_mask[:, None], other=0.0
    ).to(tl.float32)
    return conv_taps, conv_bias, taps_bin0, taps_bin1


@triton.jit
def _load_window(
    mel_ptr, batch_base, mel_time_stride, relu_out_row, relu_out_col, pos_ok, valid_time, mel_freq, pad_start
):
    """The window's mel patch as [tap, window position], both padded to PAD. The GEMM's rhs.

    Indexed directly into the unpadded mel. The mask that keeps those reads in bounds also
    zeroes anything at or past ``valid_time``, this utterance's own length.
    """
    tap = tl.arange(0, PAD)
    mel_row = STRIDE * relu_out_row[None, :] + (tap // KERNEL)[:, None] - pad_start
    mel_col = STRIDE * relu_out_col[None, :] + (tap % KERNEL)[:, None] - pad_start
    in_range = (mel_row >= 0) & (mel_row < valid_time) & (mel_col >= 0) & (mel_col < mel_freq)
    return tl.load(
        mel_ptr + batch_base + mel_row * mel_time_stride + mel_col,
        mask=(tap < TAPS)[:, None] & pos_ok[None, :] & in_range,
        other=0.0,
    )


@triton.autotune(configs=_forward_configs(), key=["channels", "out_freq"])
@triton.jit
def _forward_kernel(
    mel_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    taps_bin0_ptr,
    taps_bin1_ptr,
    depth_bias_ptr,
    output_ptr,
    mel_len_ptr,
    relu_out_len_ptr,
    out_len_ptr,
    pad_start,
    channels,
    out_time,
    out_freq,
    mel_freq,
    relu_out_freq,
    mel_batch_stride,
    mel_time_stride,
    out_batch_stride,
    out_time_stride,
    out_freq_stride,
    TIME_ROWS: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
):
    # CUDA caps the three grid axes at 2**31 - 1, 65535 and 65535. Only the tile count grows
    # with audio length, so it takes the first; channel tiles and batch stay far below 65535.
    tile = tl.program_id(0)
    batch = tl.program_id(2)
    freq_tiles = tl.cdiv(out_freq, NUM_BINS)
    first_out_row = (tile // freq_tiles) * TIME_ROWS
    bin0_col = (tile % freq_tiles) * NUM_BINS
    channel = tl.program_id(1) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    channel_mask = channel < channels

    conv_taps, conv_bias, taps_bin0, taps_bin1 = _load_params(
        conv_weight_ptr, conv_bias_ptr, taps_bin0_ptr, taps_bin1_ptr, channel, channel_mask
    )
    depth_bias = tl.load(depth_bias_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)

    batch_base = batch * mel_batch_stride
    # Each stage has its own length. `relu_out` is masked at the post-conv0 length because
    # relu(conv_bias) is non-zero, so padded time would otherwise carry the bias forward.
    this_mel_len = tl.load(mel_len_ptr + batch)
    this_relu_out_len = tl.load(relu_out_len_ptr + batch)
    this_out_len = tl.load(out_len_ptr + batch)

    # Tiles past the utterance skip their work but must still store zeros: the downstream
    # `pw1` contracts over every time position for its weight gradient.
    tile_live = first_out_row < this_out_len

    for step in tl.static_range(TIME_ROWS):
        if tile_live:
            relu_out_row, relu_out_col, pos_ok = _window_cells(
                first_out_row + step, bin0_col, this_relu_out_len, relu_out_freq, pad_start
            )
            feats = _load_window(
                mel_ptr,
                batch_base,
                mel_time_stride,
                relu_out_row,
                relu_out_col,
                pos_ok,
                this_mel_len,
                mel_freq,
                pad_start,
            )
            relu_out = tl.dot(conv_taps, feats) + conv_bias[:, None]
            relu_out = tl.where(pos_ok[None, :], tl.maximum(relu_out, 0.0), 0.0)

            out_bin0 = depth_bias + tl.sum(relu_out * taps_bin0, 1)
            out_bin1 = depth_bias + tl.sum(relu_out * taps_bin1, 1)
        else:
            out_bin0 = tl.zeros([CHANNEL_BLOCK], tl.float32)
            out_bin1 = tl.zeros([CHANNEL_BLOCK], tl.float32)
        # zeros are stored past the utterance, so mask the value, not the store
        in_length = first_out_row + step < this_out_len
        out_bin0 = tl.where(in_length, out_bin0, 0.0)
        out_bin1 = tl.where(in_length, out_bin1, 0.0)
        time_valid = channel_mask & (first_out_row + step < out_time)
        out_base = (
            output_ptr
            + batch * out_batch_stride
            + (first_out_row + step) * out_time_stride
            + bin0_col * out_freq_stride
            + channel
        )
        # Each tile covers NUM_BINS bins, so bin0 is always in range. When out_freq is odd the
        # last tile's bin1 is one past the end.
        tl.store(out_base, out_bin0.to(output_ptr.dtype.element_ty), mask=time_valid)
        tl.store(
            out_base + out_freq_stride,
            out_bin1.to(output_ptr.dtype.element_ty),
            mask=time_valid & (bin0_col + 1 < out_freq),
        )


# reset_to_zero: this kernel accumulates with atomic_add, and autotune re-runs each config.
@triton.autotune(
    configs=_backward_configs(),
    key=["channels", "out_freq"],
    reset_to_zero=[
        "grad_conv_weight_ptr",
        "grad_conv_bias_ptr",
        "acc_bin0_ptr",
        "acc_bin1_ptr",
        "grad_depth_bias_ptr",
    ],
)
@triton.jit
def _backward_kernel(
    mel_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    taps_bin0_ptr,
    taps_bin1_ptr,
    grad_output_ptr,
    grad_conv_weight_ptr,
    grad_conv_bias_ptr,
    acc_bin0_ptr,
    acc_bin1_ptr,
    grad_depth_bias_ptr,
    mel_len_ptr,
    relu_out_len_ptr,
    out_len_ptr,
    pad_start,
    channels,
    out_time,
    out_freq,
    mel_freq,
    relu_out_freq,
    batch_size,
    mel_batch_stride,
    mel_time_stride,
    grad_batch_stride,
    grad_time_stride,
    grad_freq_stride,
    TIME_ROWS: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
    TILE_SPLITS: tl.constexpr,
):
    channel = tl.program_id(0) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    channel_mask = channel < channels
    tap = tl.arange(0, PAD)
    window_pos = tl.arange(0, PAD)
    conv_taps, conv_bias, taps_bin0, taps_bin1 = _load_params(
        conv_weight_ptr, conv_bias_ptr, taps_bin0_ptr, taps_bin1_ptr, channel, channel_mask
    )

    # Columns 0..TAPS-1 accumulate the conv0 weight gradients, column TAPS accumulates the bias gradient.
    grad_conv_weight = tl.zeros([CHANNEL_BLOCK, PAD], tl.float32)
    acc_bin0 = tl.zeros([CHANNEL_BLOCK, PAD], tl.float32)
    acc_bin1 = tl.zeros([CHANNEL_BLOCK, PAD], tl.float32)
    grad_depth_bias = tl.zeros([CHANNEL_BLOCK], tl.float32)

    freq_tiles = tl.cdiv(out_freq, NUM_BINS)
    tiles_per_batch = tl.cdiv(out_time, TIME_ROWS) * freq_tiles
    for tile in range(tl.program_id(1), batch_size * tiles_per_batch, TILE_SPLITS):
        batch = tile // tiles_per_batch
        tile_in_batch = tile % tiles_per_batch
        first_out_row = (tile_in_batch // freq_tiles) * TIME_ROWS
        bin0_col = (tile_in_batch % freq_tiles) * NUM_BINS
        batch_base = batch * mel_batch_stride
        this_mel_len = tl.load(mel_len_ptr + batch)
        this_relu_out_len = tl.load(relu_out_len_ptr + batch)
        this_out_len = tl.load(out_len_ptr + batch)

        # Tiles past the utterance contribute nothing to any gradient.
        if first_out_row < this_out_len:
            for step in tl.static_range(TIME_ROWS):
                relu_out_row, relu_out_col, pos_ok = _window_cells(
                    first_out_row + step, bin0_col, this_relu_out_len, relu_out_freq, pad_start
                )
                feats = _load_window(
                    mel_ptr,
                    batch_base,
                    mel_time_stride,
                    relu_out_row,
                    relu_out_col,
                    pos_ok,
                    this_mel_len,
                    mel_freq,
                    pad_start,
                )
                # The bias enters conv0 as `bias * 1`, making it a tap whose input is always 1,
                # so its gradient is the same sum as any other tap's. Row TAPS of `feats` is
                # padding, so ones there produce the bias gradient in column TAPS.
                # conv_taps is zero in that row, so the conv0 recompute is unaffected.
                feats = tl.where((tap == TAPS)[:, None] & pos_ok[None, :], tl.full((1, 1), 1.0, feats.dtype), feats)
                pre_relu = tl.dot(conv_taps, feats) + conv_bias[:, None]
                live = pos_ok[None, :] & (pre_relu > 0.0)
                relu_out = tl.where(live, pre_relu, 0.0)

                time_valid = channel_mask & (first_out_row + step < out_time) & (first_out_row + step < this_out_len)
                grad_base = (
                    grad_output_ptr
                    + batch * grad_batch_stride
                    + (first_out_row + step) * grad_time_stride
                    + bin0_col * grad_freq_stride
                    + channel
                )
                grad_bin0 = tl.load(grad_base, mask=time_valid, other=0.0).to(tl.float32)
                grad_bin1 = tl.load(
                    grad_base + grad_freq_stride, mask=time_valid & (bin0_col + 1 < out_freq), other=0.0
                ).to(tl.float32)

                grad_depth_bias += grad_bin0 + grad_bin1
                acc_bin0 += grad_bin0[:, None] * relu_out
                acc_bin1 += grad_bin1[:, None] * relu_out

                grad_relu_out = tl.where(live, grad_bin0[:, None] * taps_bin0 + grad_bin1[:, None] * taps_bin1, 0.0)
                grad_conv_weight += tl.dot(grad_relu_out.to(feats.dtype), tl.trans(feats))

    grad_conv_bias = tl.sum(tl.where((tap == TAPS)[None, :], grad_conv_weight, 0.0), 1)
    tl.atomic_add(
        grad_conv_weight_ptr + channel[:, None] * PAD + tap[None, :],
        grad_conv_weight,
        mask=channel_mask[:, None] & (tap < TAPS)[None, :],
    )
    tl.atomic_add(acc_bin0_ptr + channel[:, None] * PAD + window_pos[None, :], acc_bin0, mask=channel_mask[:, None])
    tl.atomic_add(acc_bin1_ptr + channel[:, None] * PAD + window_pos[None, :], acc_bin1, mask=channel_mask[:, None])
    tl.atomic_add(grad_conv_bias_ptr + channel, grad_conv_bias, mask=channel_mask)
    tl.atomic_add(grad_depth_bias_ptr + channel, grad_depth_bias, mask=channel_mask)


def _downsampled_length(length, pad_total):
    """Output extent of one strided convolution stage, floor mode. Takes ints or int tensors."""
    return (length + pad_total - KERNEL.value) // STRIDE.value + 1


def _as_window(row):
    """Read a flat PAD-wide row back as the KERNEL x WINDOW_COLS window it holds."""
    return row[:, :WINDOW_SIZE].reshape(-1, KERNEL, WINDOW_COLS)


def _as_row(window):
    """Flatten a window into the PAD-wide row the kernel loads; PAD is the next power of two."""
    row = window.new_zeros(window.shape[0], PAD)
    row[:, :WINDOW_SIZE] = window.reshape(window.shape[0], WINDOW_SIZE)
    return row


def _build_bin_taps(depth_weight):
    """The depthwise taps laid out across the window, once per output bin.

    The two bins a program emits sit STRIDE columns apart, so their KERNEL x KERNEL patches
    together span the KERNEL x WINDOW_COLS window:

        columns   0  1  2  3  4
        bin0      [-----]
        bin1            [-----]

    Both bins read the same taps. Each returned vector holds them at its own columns and zero
    elsewhere, so a bin's output is the whole window multiplied by its vector and summed. The
    zeros stand in for a slice, which the flattened window tile cannot express.
    """
    channels = depth_weight.shape[0]
    taps = depth_weight.reshape(channels, KERNEL, KERNEL)
    bin0 = depth_weight.new_zeros(channels, KERNEL, WINDOW_COLS)
    bin1 = depth_weight.new_zeros(channels, KERNEL, WINDOW_COLS)
    bin0[..., :KERNEL] = taps
    bin1[..., STRIDE:] = taps
    return _as_row(bin0), _as_row(bin1)


class _FusedSubsampling(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mel,
        conv_weight,
        conv_bias,
        depth_weight,
        depth_bias,
        mel_lengths,
        relu_out_lengths,
        out_lengths,
        pad_start,
        dims,
        dtype,
    ):
        out_time, out_freq, relu_out_freq = dims
        batch_size, _, _, mel_freq = mel.shape
        channels = conv_weight.shape[0]
        mel = mel.contiguous().to(dtype)
        taps_bin0, taps_bin1 = _build_bin_taps(depth_weight)
        conv_weight_cast, conv_bias_cast = conv_weight.to(dtype), conv_bias.to(dtype)
        taps_bin0, taps_bin1 = taps_bin0.to(dtype), taps_bin1.to(dtype)
        output = torch.empty((batch_size, out_time, out_freq, channels), device=mel.device, dtype=dtype)

        def grid(meta):
            return (
                triton.cdiv(out_time, meta["TIME_ROWS"]) * triton.cdiv(out_freq, NUM_BINS),
                triton.cdiv(channels, meta["CHANNEL_BLOCK"]),
                batch_size,
            )

        _forward_kernel[grid](
            mel,
            conv_weight_cast,
            conv_bias_cast,
            taps_bin0,
            taps_bin1,
            depth_bias.to(dtype),
            output,
            mel_lengths,
            relu_out_lengths,
            out_lengths,
            pad_start,
            channels,
            out_time,
            out_freq,
            mel_freq,
            relu_out_freq,
            mel.stride(0),
            mel.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
        )
        # Cloned because callers own the lengths and edit them in place; backward needs the originals.
        ctx.save_for_backward(
            mel,
            conv_weight_cast,
            conv_bias_cast,
            taps_bin0,
            taps_bin1,
            mel_lengths.clone(),
            relu_out_lengths.clone(),
            out_lengths.clone(),
        )
        ctx.shapes = (batch_size, channels, mel_freq, relu_out_freq, out_time, out_freq, pad_start)
        ctx.param_dtypes = (conv_weight.dtype, conv_bias.dtype, depth_weight.dtype, depth_bias.dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (mel, conv_weight, conv_bias, taps_bin0, taps_bin1, mel_lengths, relu_out_lengths, out_lengths) = (
            ctx.saved_tensors
        )
        (batch_size, channels, mel_freq, relu_out_freq, out_time, out_freq, pad_start) = ctx.shapes
        grad_output = grad_output.contiguous()
        device = grad_output.device
        grad_conv_weight = torch.zeros((channels, PAD), device=device, dtype=torch.float32)
        grad_conv_bias = torch.zeros((channels,), device=device, dtype=torch.float32)
        acc_bin0 = torch.zeros((channels, PAD), device=device, dtype=torch.float32)
        acc_bin1 = torch.zeros((channels, PAD), device=device, dtype=torch.float32)
        grad_depth_bias = torch.zeros((channels,), device=device, dtype=torch.float32)

        def grid(meta):
            tiles = batch_size * triton.cdiv(out_time, meta["TIME_ROWS"]) * triton.cdiv(out_freq, NUM_BINS)
            return triton.cdiv(channels, meta["CHANNEL_BLOCK"]), min(meta["TILE_SPLITS"], tiles)

        _backward_kernel[grid](
            mel,
            conv_weight,
            conv_bias,
            taps_bin0,
            taps_bin1,
            grad_output,
            grad_conv_weight,
            grad_conv_bias,
            acc_bin0,
            acc_bin1,
            grad_depth_bias,
            mel_lengths,
            relu_out_lengths,
            out_lengths,
            pad_start,
            channels,
            out_time,
            out_freq,
            mel_freq,
            relu_out_freq,
            batch_size,
            mel.stride(0),
            mel.stride(2),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
        )
        # Undo `_build_bin_taps`: each bin accumulated into the columns it read.
        grad_depth_weight = (_as_window(acc_bin0)[..., :KERNEL] + _as_window(acc_bin1)[..., STRIDE:]).reshape(
            channels, TAPS
        )
        conv_w_dtype, conv_b_dtype, depth_w_dtype, depth_b_dtype = ctx.param_dtypes
        return (
            None,  # mel
            grad_conv_weight[:, :TAPS].view(channels, 1, KERNEL, KERNEL).to(conv_w_dtype),
            grad_conv_bias.to(conv_b_dtype),
            grad_depth_weight.view(channels, 1, KERNEL, KERNEL).to(depth_w_dtype),
            grad_depth_bias.to(depth_b_dtype),
            None,  # mel_lengths
            None,  # relu_out_lengths
            None,  # out_lengths
            None,  # pad_start
            None,  # dims
            None,  # dtype
        )


def fused_conv_relu_dw(
    mel: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    depth_weight: torch.Tensor,
    depth_bias: torch.Tensor,
    pad_start: int,
    pad_end: int,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``conv -> ReLU -> depthwise``, with the between-stage length masking folded in.

    Both convolutions are 3x3 stride 2. Padding is index arithmetic inside the kernel, so
    symmetric and causal (asymmetric ``(2, 1)``) padding take the same path.

    Args:
        mel: ``(batch, 1, time, freq)`` features, channel-first and unpadded.
        conv_weight: first convolution weight, ``(channels, 1, 3, 3)``.
        conv_bias: first convolution bias, ``(channels,)``.
        depth_weight: depthwise convolution weight, ``(channels, 1, 3, 3)``.
        depth_bias: depthwise convolution bias, ``(channels,)``.
        pad_start: padding at the start of both axes.
        pad_end: padding at the end; only affects the output extent.
        lengths: ``(batch,)`` valid time steps.

    Returns:
        ``(output, out_lengths)`` where output is ``(batch, out_time, out_freq, channels)``

    Raises:
        RuntimeError: if the inputs are not on a CUDA device.
    """
    if not mel.is_cuda:
        raise RuntimeError("The fused subsampling kernel requires CUDA tensors")
    batch_size, _, mel_time, mel_freq = mel.shape
    pad_total = pad_start + pad_end
    # Both convolutions downsample both axes.
    relu_out_time = _downsampled_length(mel_time, pad_total)
    relu_out_freq = _downsampled_length(mel_freq, pad_total)
    dims = (
        _downsampled_length(relu_out_time, pad_total),
        _downsampled_length(relu_out_freq, pad_total),
        relu_out_freq,
    )
    relu_out_lengths = _downsampled_length(lengths, pad_total)
    out_lengths = _downsampled_length(relu_out_lengths, pad_total)
    # Autocast only rewrites aten ops, never a Triton launch, so the cast is done by hand.
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else mel.dtype
    with torch.autocast("cuda", enabled=False):
        output = _FusedSubsampling.apply(
            mel,
            conv_weight,
            conv_bias,
            depth_weight,
            depth_bias,
            lengths,
            relu_out_lengths,
            out_lengths,
            pad_start,
            dims,
            dtype,
        )
    return output, out_lengths
