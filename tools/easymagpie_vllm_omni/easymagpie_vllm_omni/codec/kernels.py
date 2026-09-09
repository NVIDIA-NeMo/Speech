# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Initial packed CUDA kernels for the stateful codec layers.

These kernels establish the packed-sequence semantics and remove the Python
per-request loop. They are intentionally conservative first versions; tuning
tile sizes and fusing HalfSnake are separate performance work.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _packed_half_snake_kernel(
    input_ptr,
    alpha_ptr,
    output_ptr,
    numel,
    channels: tl.constexpr,
    snake_channels: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    channel = offsets % channels
    values = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    alpha = tl.load(alpha_ptr + channel, mask=mask & (channel < snake_channels), other=1.0).to(tl.float32)
    sine = tl.sin(alpha * values)
    periodic = values + sine * sine / (alpha + 1.0e-9)
    leaky = tl.where(values >= 0.0, values, values * 0.01)
    tl.store(output_ptr + offsets, tl.where(channel < snake_channels, periodic, leaky), mask=mask)


def packed_half_snake(inputs: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Fused HalfSnake and leaky-ReLU over packed ``[tokens, channels]`` input."""
    inputs = inputs.contiguous()
    channels = inputs.shape[1]
    snake_channels = alpha.numel()
    outputs = torch.empty_like(inputs)
    block = 256
    _packed_half_snake_kernel[(triton.cdiv(inputs.numel(), block),)](
        inputs,
        alpha,
        outputs,
        inputs.numel(),
        channels,
        snake_channels,
        BLOCK=block,
    )
    return outputs


@triton.jit
def _gather_packed_state_inputs_kernel(
    input_ptr,
    state_ptr,
    cache_indices_ptr,
    has_initial_ptr,
    output_ptr,
    stride_state_page: tl.constexpr,
    channels: tl.constexpr,
    history: tl.constexpr,
    sequence_rows: tl.constexpr,
    IS_DECODE: tl.constexpr,
    numel,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    channel = offsets % channels
    packed_row = offsets // channels
    joined_rows = history + sequence_rows
    sequence = packed_row // joined_rows
    row = packed_row % joined_rows
    page = tl.load(cache_indices_ptr + sequence, mask=mask)
    if IS_DECODE:
        has_initial = 1
    else:
        has_initial = tl.load(has_initial_ptr + sequence, mask=mask)
    previous = tl.load(
        state_ptr + page * stride_state_page + row * channels + channel,
        mask=mask & (row < history) & (has_initial != 0),
        other=0.0,
    )
    input_row = sequence * sequence_rows + row - history
    current = tl.load(
        input_ptr + input_row * channels + channel,
        mask=mask & (row >= history),
        other=0.0,
    )
    tl.store(output_ptr + offsets, tl.where(row < history, previous, current), mask=mask)


def gather_packed_state_inputs(
    inputs: torch.Tensor,
    state: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial: torch.Tensor,
    *,
    history: int,
    is_decode: bool,
) -> torch.Tensor:
    """Gather cached history and append uniform packed inputs in one kernel."""
    inputs = inputs.contiguous()
    batch_size = cache_indices.numel()
    sequence_rows = inputs.shape[0] // batch_size
    channels = inputs.shape[1]
    outputs = torch.empty(
        (batch_size, history + sequence_rows, channels),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    block = 256
    _gather_packed_state_inputs_kernel[(triton.cdiv(outputs.numel(), block),)](
        inputs,
        state,
        cache_indices,
        has_initial,
        outputs,
        state.stride(0),
        channels,
        history,
        sequence_rows,
        IS_DECODE=is_decode,
        numel=outputs.numel(),
        BLOCK=block,
    )
    return outputs


@triton.jit
def _packed_causal_conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    state_ptr,
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_ptr,
    output_ptr,
    stride_state_page: tl.constexpr,
    input_channels: tl.constexpr,
    output_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    time_factor: tl.constexpr,
    IS_DECODE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    token_offsets = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    output_offsets = tl.program_id(2) * BLOCK_O + tl.arange(0, BLOCK_O)
    input_offsets = tl.arange(0, BLOCK_I)

    if IS_DECODE:
        base_start = seq_idx
        base_end = seq_idx + 1
        has_initial = 1
    else:
        base_start = tl.load(query_start_loc_ptr + seq_idx)
        base_end = tl.load(query_start_loc_ptr + seq_idx + 1)
        has_initial = tl.load(has_initial_ptr + seq_idx)
    sequence_length = (base_end - base_start) * time_factor
    sequence_start = base_start * time_factor
    page = tl.load(cache_indices_ptr + seq_idx)
    history = kernel_size - 1

    accumulator = tl.zeros((BLOCK_T, BLOCK_O), dtype=tl.float32)
    for kernel_offset in range(kernel_size):
        source_offsets = token_offsets - (history - kernel_offset)
        current_mask = source_offsets >= 0
        for input_block in range(tl.cdiv(input_channels, BLOCK_I)):
            channels = input_block * BLOCK_I + input_offsets
            x_offsets = (sequence_start + source_offsets[:, None]) * input_channels + channels[None, :]
            current = tl.load(
                x_ptr + x_offsets,
                mask=(token_offsets[:, None] < sequence_length)
                & current_mask[:, None]
                & (channels[None, :] < input_channels),
                other=0.0,
            )
            state_row = history + source_offsets
            state_offsets = page * stride_state_page + state_row[:, None] * input_channels + channels[None, :]
            previous = tl.load(
                state_ptr + state_offsets,
                mask=(token_offsets[:, None] < sequence_length)
                & (~current_mask[:, None])
                & (channels[None, :] < input_channels)
                & (has_initial != 0),
                other=0.0,
            )
            values = tl.where(current_mask[:, None], current, previous)
            weight_offsets = (
                output_offsets[None, :] * input_channels * kernel_size
                + channels[:, None] * kernel_size
                + kernel_offset
            )
            weights = tl.load(
                weight_ptr + weight_offsets,
                mask=(channels[:, None] < input_channels) & (output_offsets[None, :] < output_channels),
                other=0.0,
            )
            accumulator += tl.dot(values, weights, input_precision="ieee")

    if HAS_BIAS:
        bias = tl.load(bias_ptr + output_offsets, mask=output_offsets < output_channels, other=0.0)
        accumulator += bias[None, :]
    output_indices = (sequence_start + token_offsets[:, None]) * output_channels + output_offsets[None, :]
    tl.store(
        output_ptr + output_indices,
        accumulator,
        mask=(token_offsets[:, None] < sequence_length) & (output_offsets[None, :] < output_channels),
    )


@triton.jit
def _update_packed_state_kernel(
    x_ptr,
    state_ptr,
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_ptr,
    stride_state_page: tl.constexpr,
    channels: tl.constexpr,
    history: tl.constexpr,
    time_factor: tl.constexpr,
    IS_DECODE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    channel_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    history_offsets = tl.arange(0, BLOCK_H)
    if IS_DECODE:
        base_start = seq_idx
        base_end = seq_idx + 1
        has_initial = 1
    else:
        base_start = tl.load(query_start_loc_ptr + seq_idx)
        base_end = tl.load(query_start_loc_ptr + seq_idx + 1)
        has_initial = tl.load(has_initial_ptr + seq_idx)
    sequence_start = base_start * time_factor
    sequence_length = (base_end - base_start) * time_factor
    page = tl.load(cache_indices_ptr + seq_idx)

    source_offsets = sequence_length - history + history_offsets
    current_offsets = (sequence_start + source_offsets[:, None]) * channels + channel_offsets[None, :]
    current = tl.load(
        x_ptr + current_offsets,
        mask=(history_offsets[:, None] < history)
        & (source_offsets[:, None] >= 0)
        & (channel_offsets[None, :] < channels),
        other=0.0,
    )
    old_rows = history + source_offsets
    old_offsets = page * stride_state_page + old_rows[:, None] * channels + channel_offsets[None, :]
    previous = tl.load(
        state_ptr + old_offsets,
        mask=(history_offsets[:, None] < history)
        & (source_offsets[:, None] < 0)
        & (channel_offsets[None, :] < channels)
        & (has_initial != 0),
        other=0.0,
    )
    values = tl.where(source_offsets[:, None] >= 0, current, previous)
    output_offsets = page * stride_state_page + history_offsets[:, None] * channels + channel_offsets[None, :]
    tl.store(
        state_ptr + output_offsets,
        values,
        mask=(history_offsets[:, None] < history) & (channel_offsets[None, :] < channels),
    )


@triton.jit
def _packed_causal_deconv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    state_ptr,
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_ptr,
    output_ptr,
    stride_state_page: tl.constexpr,
    input_channels: tl.constexpr,
    output_channels: tl.constexpr,
    stride: tl.constexpr,
    input_time_factor: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    IS_DECODE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    output_token_offsets = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    output_offsets = tl.program_id(2) * BLOCK_O + tl.arange(0, BLOCK_O)
    if IS_DECODE:
        base_start = seq_idx
        base_end = seq_idx + 1
        has_initial = 1
    else:
        base_start = tl.load(query_start_loc_ptr + seq_idx)
        base_end = tl.load(query_start_loc_ptr + seq_idx + 1)
        has_initial = tl.load(has_initial_ptr + seq_idx)
    input_start = base_start * input_time_factor
    input_length = (base_end - base_start) * input_time_factor
    output_length = input_length * stride
    page = tl.load(cache_indices_ptr + seq_idx)

    input_token = output_token_offsets // stride
    phase = output_token_offsets % stride
    group = output_offsets // output_channels_per_group
    output_in_group = output_offsets % output_channels_per_group
    input_group_start = group * input_channels_per_group
    accumulator = tl.zeros((BLOCK_T, BLOCK_O), dtype=tl.float32)

    for input_in_group in range(input_channels_per_group):
        input_channel = input_group_start + input_in_group
        current_offsets = (input_start + input_token[:, None]) * input_channels + input_channel[None, :]
        current = tl.load(
            x_ptr + current_offsets,
            mask=(output_token_offsets[:, None] < output_length)
            & (output_offsets[None, :] < output_channels)
            & (input_channel[None, :] < input_channels),
            other=0.0,
        )
        previous_token = input_token - 1
        previous_offsets = (input_start + previous_token[:, None]) * input_channels + input_channel[None, :]
        previous_current = tl.load(
            x_ptr + previous_offsets,
            mask=(output_token_offsets[:, None] < output_length)
            & (previous_token[:, None] >= 0)
            & (output_offsets[None, :] < output_channels)
            & (input_channel[None, :] < input_channels),
            other=0.0,
        )
        state_offsets = page * stride_state_page + input_channel
        previous_state = tl.load(
            state_ptr + state_offsets,
            mask=(output_offsets < output_channels) & (input_channel < input_channels) & (has_initial != 0),
            other=0.0,
        )
        previous = tl.where(previous_token[:, None] >= 0, previous_current, previous_state[None, :])

        current_weight_offsets = (
            input_channel[None, :] * output_channels_per_group * (2 * stride)
            + output_in_group[None, :] * (2 * stride)
            + phase[:, None]
        )
        previous_weight_offsets = current_weight_offsets + stride
        current_weights = tl.load(
            weight_ptr + current_weight_offsets,
            mask=(output_token_offsets[:, None] < output_length) & (output_offsets[None, :] < output_channels),
            other=0.0,
        )
        previous_weights = tl.load(
            weight_ptr + previous_weight_offsets,
            mask=(output_token_offsets[:, None] < output_length) & (output_offsets[None, :] < output_channels),
            other=0.0,
        )
        accumulator += current * current_weights + previous * previous_weights

    if HAS_BIAS:
        bias = tl.load(bias_ptr + output_offsets, mask=output_offsets < output_channels, other=0.0)
        accumulator += bias[None, :]
    output_start = base_start * input_time_factor * stride
    output_indices = (output_start + output_token_offsets[:, None]) * output_channels + output_offsets[None, :]
    tl.store(
        output_ptr + output_indices,
        accumulator,
        mask=(output_token_offsets[:, None] < output_length) & (output_offsets[None, :] < output_channels),
    )


def _max_sequence_length(query_start_loc: torch.Tensor, time_factor: int) -> int:
    # The Mamba metadata builder already has this value on the CPU; this
    # correctness kernel recomputes it until the custom metadata subclass lands.
    return int(torch.diff(query_start_loc).max().item()) * time_factor


def update_packed_state(
    inputs: torch.Tensor,
    state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial: torch.Tensor,
    *,
    channels: int,
    history: int,
    time_factor: int,
    is_decode: bool,
) -> None:
    """Update fixed-history state pages from packed layer inputs."""
    num_sequences = cache_indices.numel() if is_decode else query_start_loc.numel() - 1
    metadata_placeholder = cache_indices
    query_start_loc_ptr = metadata_placeholder if is_decode else query_start_loc
    has_initial_ptr = metadata_placeholder if is_decode else has_initial
    _update_packed_state_kernel[(num_sequences, triton.cdiv(channels, 64))](
        inputs,
        state,
        query_start_loc_ptr,
        cache_indices,
        has_initial_ptr,
        state.stride(0),
        channels,
        history,
        time_factor,
        IS_DECODE=is_decode,
        BLOCK_C=64,
        BLOCK_H=triton.next_power_of_2(history),
    )


def packed_causal_conv1d(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial: torch.Tensor,
    *,
    time_factor: int,
    max_query_len: int | None = None,
    is_decode: bool = False,
) -> torch.Tensor:
    """Packed dense causal Conv1d and in-place fixed-history update."""
    if not inputs.is_cuda:
        raise ValueError("packed_causal_conv1d is a CUDA kernel")
    inputs = inputs.contiguous()
    weight = weight.contiguous()
    output_channels, input_channels, kernel_size = weight.shape
    outputs = torch.empty((inputs.shape[0], output_channels), dtype=inputs.dtype, device=inputs.device)
    num_sequences = cache_indices.numel() if is_decode else query_start_loc.numel() - 1
    if is_decode:
        max_length = time_factor
    elif max_query_len is not None:
        max_length = max_query_len * time_factor
    else:
        max_length = _max_sequence_length(query_start_loc, time_factor)
    if is_decode and inputs.shape[0] != num_sequences * time_factor:
        raise ValueError(f"decode input has {inputs.shape[0]} rows; expected {num_sequences} * {time_factor}")
    metadata_placeholder = cache_indices
    query_start_loc_ptr = metadata_placeholder if is_decode else query_start_loc
    has_initial_ptr = metadata_placeholder if is_decode else has_initial
    block_t, block_i, block_o = 16, 32, 32
    grid = (
        num_sequences,
        triton.cdiv(max_length, block_t),
        triton.cdiv(output_channels, block_o),
    )
    _packed_causal_conv1d_kernel[grid](
        inputs,
        weight,
        bias,
        state,
        query_start_loc_ptr,
        cache_indices,
        has_initial_ptr,
        outputs,
        state.stride(0),
        input_channels,
        output_channels,
        kernel_size,
        time_factor,
        IS_DECODE=is_decode,
        HAS_BIAS=bias is not None,
        BLOCK_T=block_t,
        BLOCK_I=block_i,
        BLOCK_O=block_o,
    )
    update_packed_state(
        inputs,
        state,
        query_start_loc,
        cache_indices,
        has_initial,
        channels=input_channels,
        history=kernel_size - 1,
        time_factor=time_factor,
        is_decode=is_decode,
    )
    return outputs


def packed_causal_conv_transpose1d(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial: torch.Tensor,
    *,
    stride: int,
    time_factor: int,
    output_channels: int,
    max_query_len: int | None = None,
    is_decode: bool = False,
) -> torch.Tensor:
    """Packed causal grouped ConvTranspose1d and one-frame state update."""
    inputs = inputs.contiguous()
    weight = weight.contiguous()
    input_channels = inputs.shape[1]
    groups = output_channels
    input_channels_per_group = input_channels // groups
    output_channels_per_group = output_channels // groups
    outputs = torch.empty(
        (inputs.shape[0] * stride, output_channels),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    num_sequences = cache_indices.numel() if is_decode else query_start_loc.numel() - 1
    if is_decode:
        max_length = time_factor * stride
    elif max_query_len is not None:
        max_length = max_query_len * time_factor * stride
    else:
        max_length = _max_sequence_length(query_start_loc, time_factor) * stride
    if is_decode and inputs.shape[0] != num_sequences * time_factor:
        raise ValueError(f"decode input has {inputs.shape[0]} rows; expected {num_sequences} * {time_factor}")
    metadata_placeholder = cache_indices
    query_start_loc_ptr = metadata_placeholder if is_decode else query_start_loc
    has_initial_ptr = metadata_placeholder if is_decode else has_initial
    block_t, block_o = 32, 32
    grid = (
        num_sequences,
        triton.cdiv(max_length, block_t),
        triton.cdiv(output_channels, block_o),
    )
    _packed_causal_deconv1d_kernel[grid](
        inputs,
        weight,
        bias,
        state,
        query_start_loc_ptr,
        cache_indices,
        has_initial_ptr,
        outputs,
        state.stride(0),
        input_channels,
        output_channels,
        stride,
        time_factor,
        input_channels_per_group,
        output_channels_per_group,
        IS_DECODE=is_decode,
        HAS_BIAS=bias is not None,
        BLOCK_T=block_t,
        BLOCK_O=block_o,
    )
    update_packed_state(
        inputs,
        state,
        query_start_loc,
        cache_indices,
        has_initial,
        channels=input_channels,
        history=1,
        time_factor=time_factor,
        is_decode=is_decode,
    )
    return outputs
