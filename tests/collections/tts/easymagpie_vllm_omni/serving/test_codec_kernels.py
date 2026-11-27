# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
import torch.nn.functional as F

from easymagpie_vllm_omni.codec import kernels
from easymagpie_vllm_omni.codec.kernels import (
    gather_packed_state_inputs,
    packed_causal_conv1d,
    packed_causal_conv_transpose1d,
    packed_half_snake,
)
from easymagpie_vllm_omni.codec.packed import CODEC_STATE_ELEMENTS


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")


def test_packed_half_snake() -> None:
    torch.manual_seed(17)
    inputs = torch.randn(113, 32, device="cuda")
    alpha = torch.rand(1, 16, 1, device="cuda") + 0.25
    actual = packed_half_snake(inputs, alpha)
    snake_in = inputs[:, :16]
    scale = alpha.reshape(1, -1)
    expected = torch.cat(
        (snake_in + torch.sin(scale * snake_in).square() / (scale + 1e-9), F.leaky_relu(inputs[:, 16:])),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_packed_causal_conv1d() -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    factor = 2
    lengths = [3, 2]
    sequences = [torch.randn(length * factor, 4, device=device) for length in lengths]
    packed = torch.cat(sequences)
    weight = torch.randn(8, 4, 3, device=device)
    bias = torch.randn(8, device=device)
    state = torch.zeros(2, CODEC_STATE_ELEMENTS, device=device)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    has_initial = torch.zeros(2, dtype=torch.bool, device=device)

    actual = packed_causal_conv1d(
        packed,
        weight,
        bias,
        state,
        query_start_loc,
        cache_indices,
        has_initial,
        time_factor=factor,
    )
    # Triton uses IEEE dot products; compare against a float64 CPU oracle rather than CUDA TF32.
    expected = torch.cat(
        [
            F.conv1d(
                F.pad(sequence.T[None].double().cpu(), (2, 0)),
                weight.double().cpu(),
                bias.double().cpu(),
            )[0].T
            for sequence in sequences
        ]
    ).to(device=device, dtype=actual.dtype)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for index, sequence in enumerate(sequences):
        torch.testing.assert_close(state[index, : 2 * 4].view(2, 4), sequence[-2:])


def test_packed_causal_conv_transpose1d() -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    factor = 2
    stride = 2
    lengths = [3, 2]
    sequences = [torch.randn(length * factor, 8, device=device) for length in lengths]
    packed = torch.cat(sequences)
    conv = torch.nn.ConvTranspose1d(8, 4, 2 * stride, stride=stride, groups=4).to(device)
    state = torch.zeros(2, CODEC_STATE_ELEMENTS, device=device)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    has_initial = torch.zeros(2, dtype=torch.bool, device=device)

    actual = packed_causal_conv_transpose1d(
        packed,
        conv.weight,
        conv.bias,
        state,
        query_start_loc,
        cache_indices,
        has_initial,
        stride=stride,
        time_factor=factor,
        output_channels=4,
    )
    expected = torch.cat([conv(sequence.T[None])[0, :, :-stride].T for sequence in sequences])
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    for index, sequence in enumerate(sequences):
        torch.testing.assert_close(state[index, :8], sequence[-1])


@pytest.mark.parametrize("is_decode", [False, True])
def test_gather_packed_state_inputs_varying_chunk_lengths_compile_once(is_decode: bool) -> None:
    """Gather stays exact and compiles once across codec chunk lengths.

    A new chunk length must not trigger a recompilation during serving. ``channels=16`` keeps ``numel`` in one
    specialization class, so only ``sequence_rows`` could add compiled variants.

    Args:
        is_decode: Whether the gather runs in decode mode, which always reads the cached history.
    """
    torch.manual_seed(23)
    device = torch.device("cuda")
    channels, history, batch = 16, 3, 2
    state = torch.randn(4, history * channels + 5, device=device)
    cache_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    has_initial = torch.tensor([True, False], device=device)
    # One cache entry per compiled specialization.
    kernel_cache = kernels._gather_packed_state_inputs_kernel.device_caches[torch.cuda.current_device()][0]
    compiled_before = len(kernel_cache)

    # Startup, steady and tail codec chunk lengths, including a single row.
    for sequence_rows in (12, 16, 1, 7, 5):
        inputs = torch.randn(batch * sequence_rows, channels, device=device)
        actual = gather_packed_state_inputs(
            inputs, state, cache_indices, has_initial, history=history, is_decode=is_decode
        )
        expected = []
        for index in range(batch):
            previous = state[cache_indices[index], : history * channels].reshape(history, channels)
            if not is_decode and not has_initial[index]:
                previous = torch.zeros_like(previous)
            expected.append(torch.cat((previous, inputs[index * sequence_rows : (index + 1) * sequence_rows])))
        torch.testing.assert_close(actual, torch.stack(expected), atol=0, rtol=0)

    assert len(kernel_cache) - compiled_before <= 1
