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

import numpy as np
import pytest
import torch

from nemo.collections.asr.losses.rnnt import NUMBA_RNNT_AVAILABLE
from nemo.collections.asr.parts.triton.rnnt_loss import rnnt_loss_triton
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

CUDA_TRITON_AVAILABLE = TRITON_AVAILABLE and torch.cuda.is_available()


@pytest.mark.unit
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE or not NUMBA_RNNT_AVAILABLE,
    reason="CUDA, Triton, and Numba RNN-T are required",
)
@pytest.mark.parametrize(
    "frames,tokens,target_bias,fastemit_lambda",
    [
        (1, 0, 0, 0.0),
        (1, 7, 0, 0.0),
        (7, 0, 0, 0.0),
        (9, 5, 0, 0.0),
        (3, 32, 0, 0.0),
        (4, 513, 6, 0.0),
        (9, 5, 0, 0.01),
    ],
)
def test_rnnt_loss_and_gradients(frames, tokens, target_bias, fastemit_lambda):
    """Check ragged boundaries, cross-warp neighbours, padding, and incoming gradient scales."""
    from nemo.collections.asr.parts.numba.rnnt_loss import rnnt_numpy

    torch.manual_seed(17)
    source_lengths = [frames, 1, max(1, frames - 1)]
    target_lengths = [tokens, 0, max(0, tokens - 1)]
    scores = torch.randn(3, frames, tokens + 1, 3)
    # Keep the wide, multi-position-per-thread case focused on indexing rather than long-path FP32 rounding.
    scores[..., 1] += target_bias
    scores = scores.log_softmax(-1)
    # Blank is channel 0; every target label is channel 1.
    expected, expected_grads = rnnt_numpy.transduce_batch(
        scores.numpy(),
        labels=np.ones((3, tokens), dtype=np.int64),
        flen=source_lengths,
        glen=target_lengths,
        blank=0,
        fastemit_lambda=fastemit_lambda,
    )
    expected = torch.tensor(expected, dtype=torch.float32)
    weights = torch.tensor([0.3, -0.7, 1.5])
    expected_grads = torch.from_numpy(expected_grads) * weights[:, None, None, None]

    # Flash passes a sliced target plane and a contiguous blank plane.
    target_plane = scores[..., 1].contiguous().cuda()
    target = target_plane[..., :-1]
    blank = scores[..., 0].contiguous().cuda()
    for b, (t, u) in enumerate(zip(source_lengths, target_lengths)):
        # Poison padding to expose an accidental dependency on an invalid cell.
        blank[b, t:, :] = float("nan")
        blank[b, :, u + 1 :] = float("nan")
        target[b, t:, :] = float("nan")
        target[b, :, u:] = float("nan")
    target.requires_grad_()
    blank.requires_grad_()
    actual = rnnt_loss_triton(
        target,
        blank,
        torch.tensor(source_lengths, device="cuda"),
        torch.tensor(target_lengths, device="cuda"),
        fastemit_lambda=fastemit_lambda,
    )
    (actual * weights.cuda()).sum().backward()
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-5)
    torch.testing.assert_close(target.grad.cpu(), expected_grads[:, :, :tokens, 1], atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(blank.grad.cpu(), expected_grads[..., 0], atol=2e-4, rtol=2e-4)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_rnnt_loss_repeated_backward_preserves_occupancies():
    target = torch.full((2, 4, 5), -2.0, device="cuda", requires_grad=True)
    blank = torch.full((2, 4, 6), -1.0, device="cuda", requires_grad=True)
    scale = torch.empty(2, device="cuda")
    losses = rnnt_loss_triton(
        target,
        blank,
        torch.tensor([4, 2], device="cuda"),
        torch.tensor([5, 1], device="cuda"),
        fastemit_lambda=0.01,
        loss_grad_scale=scale,
    )
    weights = torch.tensor([0.3, -0.7], device="cuda")
    first = torch.autograd.grad(losses, (target, blank), weights, retain_graph=True)
    torch.testing.assert_close(scale, weights, atol=0, rtol=0)
    second = torch.autograd.grad(losses, (target, blank), -weights)
    for initial, repeated in zip(first, second):
        torch.testing.assert_close(repeated, -initial, atol=0, rtol=0)
    torch.testing.assert_close(scale, -weights, atol=0, rtol=0)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_rnnt_loss_invalid_lengths_are_masked():
    target = torch.full((4, 3, 2), -2.0, device="cuda", requires_grad=True)
    blank = torch.full((4, 3, 3), -1.0, device="cuda", requires_grad=True)
    losses = rnnt_loss_triton(
        target,
        blank,
        torch.tensor([0, 4, 3, 3], device="cuda"),
        torch.tensor([1, 1, -1, 3], device="cuda"),
    )
    assert torch.isposinf(losses).all()
    losses.sum().backward()
    assert torch.equal(target.grad, torch.zeros_like(target))
    assert torch.equal(blank.grad, torch.zeros_like(blank))
