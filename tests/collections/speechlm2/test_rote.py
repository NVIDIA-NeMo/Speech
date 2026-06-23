# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math

import pytest
import torch

from nemo.collections.speechlm2.modules.rote import RotaryTimeEmbedding, rotate_half


def test_defaults():
    """Defaults follow the Audio Flamingo Next audio-encoder config (theta=1200, partial_rotary_factor=0.2)."""
    dim = 100
    rote = RotaryTimeEmbedding(dim)
    assert rote.theta == 1200.0
    assert rote.rotary_fraction == 0.2
    assert rote.rotary_dim == 20
    assert rote.inv_freq.shape == (10,)

    # Rotated width is rounded down to an even number: int(10 * 0.5) = 5 -> 4.
    assert RotaryTimeEmbedding(dim=10, rotary_fraction=0.5).rotary_dim == 4

    # rotary_fraction=1.0 recovers full-width rotation.
    full = RotaryTimeEmbedding(dim, rotary_fraction=1.0)
    assert full.rotary_dim == dim
    assert full.inv_freq.shape == (dim // 2,)


def test_rotate_half():
    """GPT-J convention: ``[x0, x1, x2, x3] -> [-x1, x0, -x3, x2]``."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]])
    assert torch.equal(rotate_half(x), expected)


def test_known_value():
    """Check the rotation against an explicit per-pair 2x2 rotation, locking in the
    ``angle = -tau * 2pi * (1/theta^(2k/rotary_dim))`` formula, the sign, and the GPT-J pairing."""
    dim, theta, tau = 4, 100.0, 0.5
    rote = RotaryTimeEmbedding(dim, theta=theta, rotary_fraction=1.0)

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (B=1, T=1, C=4)
    times = torch.tensor([[tau]])
    out = rote(x, times)

    # Two channel pairs, each rotated by angle_k = -tau * 2pi / theta^(2k/dim), k in {0, 1}.
    angles = [-tau * 2.0 * math.pi / (theta ** (2 * k / dim)) for k in range(dim // 2)]
    expected = torch.empty_like(x)
    for k, a in enumerate(angles):
        xa, xb = x[0, 0, 2 * k], x[0, 0, 2 * k + 1]
        c, s = math.cos(a), math.sin(a)
        # [[c, -s], [s, c]] @ [xa, xb]
        expected[0, 0, 2 * k] = xa * c - xb * s
        expected[0, 0, 2 * k + 1] = xa * s + xb * c

    assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_shape_dtype_preserved(dtype):
    """Output keeps the input shape and dtype (the internal angle math runs in fp32, so the
    passthrough channels round-trip exactly for fp32 and the lower-precision dtypes)."""
    dim = 80
    rote = RotaryTimeEmbedding(dim)
    x = torch.randn(2, 5, dim, dtype=dtype)
    times = torch.arange(5, dtype=torch.float32).unsqueeze(0).expand(2, -1)
    out = rote(x, times)
    assert out.shape == x.shape
    assert out.dtype == dtype
    # Partial rotation: channels beyond rotary_dim pass through unchanged.
    assert torch.equal(out[..., rote.rotary_dim :], x[..., rote.rotary_dim :])
