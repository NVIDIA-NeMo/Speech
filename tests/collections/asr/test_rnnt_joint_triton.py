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

import pytest
import torch

from nemo.collections.asr.parts.triton.rnnt_joint import (
    _PackedJoint,
    _PackedLogProbs,
    lattice_layout,
    packed_joint_logprobs,
    packed_scatter,
)
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

CUDA_TRITON_AVAILABLE = TRITON_AVAILABLE and torch.cuda.is_available()


def _joint_hidden_state(encoder, predictor, activation, dropout_p=0.0):
    """Dense ``[B, T, U + 1, H]`` joint state from the packed kernels.

    Giving every sample the full source and transcript extent makes the packed row order the same
    as a dense reshape, so the kernels can be compared elementwise against an eager broadcast add.
    """
    batch, source_steps, hidden_size = encoder.shape
    target_states = predictor.shape[1]
    source_lengths = torch.full((batch,), source_steps, device=encoder.device, dtype=torch.int32)
    target_lengths = torch.full((batch,), target_states - 1, device=encoder.device, dtype=torch.int32)
    offsets, states, total_rows = lattice_layout(source_lengths, target_lengths, source_steps, target_states)
    seed = source_lengths
    if dropout_p > 0.0:
        seed = torch.randint(0, 2**31 - 1, (1,), device=encoder.device, dtype=torch.int32)
    predictor_mask = torch.ones(batch, target_states, device=encoder.device, dtype=torch.bool)
    hidden = _PackedJoint.apply(
        encoder, predictor, offsets, states, source_lengths, seed, predictor_mask, 0, total_rows, activation, dropout_p
    )
    return hidden.view(batch, source_steps, target_states, hidden_size)


def _scatter_row_numbers(source_lengths, target_lengths, frames, target_states):
    """Scatter each row's own number into the planes, so the layout can be read back off them."""
    offsets, states, total_rows = lattice_layout(source_lengths, target_lengths, frames, target_states)
    numbers = torch.arange(total_rows, device="cuda", dtype=torch.float32) + 1.0
    planes = [torch.zeros(len(source_lengths) * frames * target_states, device="cuda") for _ in range(2)]
    shape = (len(source_lengths), frames, target_states)
    planes = packed_scatter(numbers, numbers, planes[0], planes[1], offsets, states, 0, shape)
    return planes[0], total_rows


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.gpu
def test_packed_rows_run_frame_major_within_a_sample():
    """Rows advance through transcript states first, then frames, with samples laid end to end.

    The scatter derives each address from the row rather than from a stored index, so this is what
    pins that addressing down.
    """
    source_lengths = torch.tensor([3, 1, 2], device="cuda")
    target_lengths = torch.tensor([2, 0, 1], device="cuda")
    plane, total_rows = _scatter_row_numbers(source_lengths, target_lengths, 3, 3)

    # sample 0 fills its 3x3 plane, sample 1 contributes one row, sample 2 a 2x2 corner
    occupied = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19, 21, 22], device="cuda")
    assert total_rows == occupied.numel()
    torch.testing.assert_close(plane[occupied], torch.arange(total_rows, device="cuda", dtype=torch.float32) + 1.0)
    empty = torch.ones_like(plane, dtype=torch.bool)
    empty[occupied] = False
    assert torch.equal(plane[empty], torch.zeros_like(plane[empty]))


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.gpu
def test_packed_rows_of_a_full_lattice_are_a_dense_reshape():
    """With every sample at full extent nothing is padded, so packed order is dense order."""
    batch, frames, target_states = 4, 5, 3
    source_lengths = torch.full((batch,), frames, device="cuda")
    target_lengths = torch.full((batch,), target_states - 1, device="cuda")
    plane, total_rows = _scatter_row_numbers(source_lengths, target_lengths, frames, target_states)

    torch.testing.assert_close(plane, torch.arange(total_rows, device="cuda", dtype=torch.float32) + 1.0)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.gpu
def test_flash_rnnt_join_matches_the_eager_broadcast_add(activation, dtype):
    """The fused kernel must compute the same function as an eager broadcast-add plus activation.

    It keeps the pre-activation in FP32 where eager rounds it to the input dtype, so in BF16 the
    two agree only to about one unit in the last place -- and for ReLU a rounded pre-activation can
    cross zero, flipping the derivative mask outright. Both effects are inherent to the precision
    rather than to the kernel, so compare at matched dtype with a precision-appropriate tolerance.
    """
    torch.manual_seed(7)
    # An odd hidden size leaves a partial block, which is where a mishandled tail would show.
    batch, source_steps, target_states, hidden_size = 3, 7, 5, 129
    encoder = torch.randn(batch, source_steps, hidden_size, device="cuda", dtype=dtype)
    predictor = torch.randn(batch, target_states, hidden_size, device="cuda", dtype=dtype)
    upstream = torch.randn(batch, source_steps, target_states, hidden_size, device="cuda", dtype=dtype)

    def forward_and_gradients(join):
        leaf_encoder = encoder.detach().clone().requires_grad_(True)
        leaf_predictor = predictor.detach().clone().requires_grad_(True)
        hidden = join(leaf_encoder, leaf_predictor)
        return hidden, torch.autograd.grad(hidden, (leaf_encoder, leaf_predictor), upstream)

    def eager(leaf_encoder, leaf_predictor):
        activate = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}[activation]
        return activate(leaf_encoder.unsqueeze(2) + leaf_predictor.unsqueeze(1))

    fused_hidden, fused_gradients = forward_and_gradients(lambda e, p: _joint_hidden_state(e, p, activation))
    eager_hidden, eager_gradients = forward_and_gradients(eager)

    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (3e-2, 3e-2)
    torch.testing.assert_close(fused_hidden, eager_hidden, atol=atol, rtol=rtol)
    for actual, expected in zip(fused_gradients, eager_gradients):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("hidden_size", [6, 8, 1536])
@pytest.mark.gpu
def test_joint_dropout_mask_agrees_across_kernels(hidden_size):
    """The forward and both backward kernels must reach the same verdict on every element.

    Nothing stores the mask and the three kernels walk the joint grid along different axes, so each
    redraws it from the element's position. Under relu with a positive pre-activation the activation
    derivative is exactly 1, so a backward reduces to the mask weighted by its grad_output along the
    axis it collapses. A grad_output that is one at a single index and zero elsewhere therefore
    leaves exactly that slice of the backward's mask, which compares against the forward's directly
    rather than through a sum that a compensating disagreement could still satisfy.

    ``hidden_size`` covers a size that is not a multiple of four, one block, and several blocks.
    """
    torch.manual_seed(53)
    dropout_p = 0.25
    batch, source_steps, target_states = 2, 5, 4
    # Ones against zeros makes every pre-activation exactly one, so relu's derivative is exactly one
    # and each gradient below is the mask itself, scaled.
    encoder = torch.ones(batch, source_steps, hidden_size, device="cuda", requires_grad=True)
    predictor = torch.zeros(batch, target_states, hidden_size, device="cuda", requires_grad=True)

    hidden = _joint_hidden_state(encoder, predictor, "relu", dropout_p)
    forward_mask = hidden != 0.0

    for target_index in range(target_states):
        encoder.grad = None
        probe = torch.zeros_like(hidden)
        probe[:, :, target_index] = 1.0
        hidden.backward(probe, retain_graph=True)
        torch.testing.assert_close(encoder.grad != 0.0, forward_mask[:, :, target_index])

    for source_index in range(source_steps):
        predictor.grad = None
        probe = torch.zeros_like(hidden)
        probe[:, source_index] = 1.0
        hidden.backward(probe, retain_graph=True)
        torch.testing.assert_close(predictor.grad != 0.0, forward_mask[:, source_index])

    survivors = hidden[hidden != 0.0]
    torch.testing.assert_close(survivors, torch.full_like(survivors, 1.0 / (1.0 - dropout_p)))
    assert 0.6 < survivors.numel() / hidden.numel() < 0.9


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.gpu
def test_joint_dropout_replays_each_tile_once(monkeypatch):
    torch.manual_seed(137)
    # Every pre-activation equals one, so the hidden values reveal every dropout decision.
    encoder = torch.ones(1, 4, 8, device="cuda", requires_grad=True)
    predictor = torch.zeros(1, 3, 8, device="cuda", requires_grad=True)
    calls = {}
    apply = _PackedJoint.apply

    def record(*args):
        hidden = apply(*args)
        start = args[7]  # First packed row of this tile.
        calls.setdefault(start, []).append(hidden.detach().clone())
        return hidden

    monkeypatch.setattr(_PackedJoint, "apply", staticmethod(record))
    target, blank = packed_joint_logprobs(
        encoder,
        predictor,
        targets=torch.tensor([[0, 1]], device="cuda"),
        source_lengths=torch.tensor([4], device="cuda"),
        target_lengths=torch.tensor([2], device="cuda"),
        output_weight=torch.randn(3, 8, device="cuda", requires_grad=True),
        output_bias=torch.zeros(3, device="cuda"),
        activation="relu",
        dropout_p=0.25,
        blank=2,
        clamp=0.0,
        max_joint_rows=4,
        predictor_mask=torch.ones(1, 3, dtype=torch.bool, device="cuda"),
    )
    assert {start: len(values) for start, values in calls.items()} == {0: 1, 4: 1, 8: 1}
    (target.sum() + blank.sum()).backward()
    assert {start: len(values) for start, values in calls.items()} == {0: 2, 4: 2, 8: 2}
    forward_values = torch.cat([forward for forward, _ in calls.values()])
    assert (forward_values == 0).any() and (forward_values != 0).any()
    for forward, replay in calls.values():
        assert torch.equal(forward, replay)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.gpu
def test_packed_logprobs_large_common_offset():
    logits = torch.tensor([[0, 0, 0, 0, 0], [0, 8, 8, 0, 0]], device="cuda", dtype=torch.float32) + 1e8
    reference = logits.clone().requires_grad_()
    targets = torch.tensor([[1]], device="cuda")
    offsets = torch.tensor([0, 2], device="cuda")
    states = torch.tensor([2], device="cuda")
    target, blank = _PackedLogProbs.apply(logits.requires_grad_(), targets, offsets, states, 0, 4, 0.0, None)
    logprobs = reference.log_softmax(dim=-1)
    torch.testing.assert_close(target[0], logprobs[0, 1])
    torch.testing.assert_close(blank, logprobs[:, 4])
    (target[0] + blank.sum()).backward()
    (logprobs[0, 1] + logprobs[:, 4].sum()).backward()
    torch.testing.assert_close(logits.grad, reference.grad)
