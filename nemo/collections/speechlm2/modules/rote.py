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

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent channel pairs: ``[x0, x1, x2, x3] -> [-x1, x0, -x3, x2]`` (GPT-J convention)."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


class RotaryTimeEmbedding(nn.Module):
    """Rotary Time Embedding (RoTE), as defined in OMCAT (Goel et al., 2024, https://arxiv.org/abs/2410.12109).

    RoTE is RoPE with the token index replaced by the absolute timestamp in seconds:
    ``θ ← −τ·2π`` instead of ``θ ← −i·2π``. Each channel pair rotates at its own frequency
    (the "clock-hands" analogy from the paper): for channel pair ``k`` and timestamp ``τ`` in
    seconds, the rotation angle is ``−τ · 2π · (1 / theta^(2k/rotary_dim))``.

    Args:
        dim: Feature dimension of the ``(Batch, Time, Channel)`` input (channel dimension ``C``). Should be equal to encoder output dim size.
        theta: Base of the geometric frequency progression (RoPE ``rope_theta``). Defaults to the Audio Flamingo Next value ``1200.0``.
        rotary_fraction: Fraction of ``dim`` to rotate (RoPE ``partial_rotary_factor``). The rotated
            width is rounded down to an even number. Defaults to the Audio Flamingo Next value ``0.2``.
    """

    def __init__(self, dim: int, theta: float = 1200.0, rotary_fraction: float = 0.2):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even to split into rotated pairs, got dim={dim}.")
        if not 0.0 < rotary_fraction <= 1.0:
            raise ValueError(f"rotary_fraction must be in (0, 1], got rotary_fraction={rotary_fraction}.")
        rotary_dim = int(dim * rotary_fraction)
        rotary_dim -= rotary_dim % 2
        if rotary_dim < 2:
            raise ValueError(
                f"rotary_fraction={rotary_fraction} yields rotary_dim={rotary_dim} for dim={dim}; "
                f"need at least 2 channels to rotate."
            )
        self.dim = dim
        self.theta = theta
        self.rotary_fraction = rotary_fraction
        self.rotary_dim = rotary_dim
        # Following Audio Flamingo Next, the exponent normalizer is the rotated width `rotary_dim` (not the full `dim`).
        inv_freq = -(2.0 * math.pi) / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        # Derived (not trained) and recomputable from config.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, times: Tensor) -> Tensor:
        """Rotate ``x`` by absolute per-frame ``times`` (seconds).

        Args:
            x: Feature embeddings of shape ``(B, T, C)`` with ``C == dim`` (channel-last).
            times: Per-frame absolute time in seconds, shape ``(B, T)`` (broadcastable to ``x[..., 0]``).

        Returns:
            Tensor of the same shape and dtype as ``x``, rotated by the time-dependent angle.
        """
        ori_dtype = x.dtype
        # OMCAT runs this in fp64, do we need it or fp32 enough?
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            times = times.float()
            # From Audio Flamingo Next): rotate the first `rotary_dim` channels (others are unchanged)
            x_rot, x_pass = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
            freqs = times.unsqueeze(-1) * self.inv_freq.to(device=x.device, dtype=torch.float32)
            emb = torch.repeat_interleave(freqs, 2, dim=-1)  # interleaved [f0, f0, f1, f1, ...]
            cos, sin = emb.cos(), emb.sin()
            out_rot = x_rot * cos + rotate_half(x_rot) * sin
            out = torch.cat((out_rot, x_pass), dim=-1)
        return out.to(ori_dtype)
