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
from typing import Optional

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent channel pairs: ``[x0, x1, x2, x3] -> [-x1, x0, -x3, x2]``.

    Matches the ``rotate_half`` used by OmniVinci
    """
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


class RotaryTimeEmbedding(nn.Module):
    """Apply ROtary Time Embedding to feature embeddings which encodes absolute time into them.
    Audio Flamingo Next uses a diffrent implementation of RoTE.
    
    Args:
        dim: Feature dimension of the embeddings ROTE is applied to (the channel
            dimension ``C`` of the ``(Batch, Time, Channel)`` input).
        theta: Base used for the geometric progression of inverse frequencies
            (a.k.a. ``rope_theta``). If ``None`` (default), it is equals to``max_time / 2π`` (OmniVinci) when ``max_time`` is set,
            otherwise the Audio Flamingo next value is used : 1200.0. Pass a value to
            override.
        rotary_fraction: Fraction of channels to rotate, in ``(0, 1]``. The first
            ``rot_dim`` channels (rounded down to an even number) are rotated and
            the remaining channels are passed through unchanged.
        max_time: Optional maximum expected time in seconds. When set, per-frame
            ``times`` are normalized as ``times / max_time * 2π`` before computing
            angles (not optional in OmniVinci). The fastest channel
            then completes exactly one rotation over ``max_time`` and every slower
            channel less than one, monotonic phase across
            ``[0, max_time]``. When ``None`` (default), raw seconds are used as-is
            (as in RoPE: fast channels wrap, slow channels disambiguate).
    """

    def __init__(
        self, dim: int, theta: Optional[float] = None, rotary_fraction: float = 0.2, max_time: Optional[float] = None
    ):
        super().__init__()
        if not 0.0 < rotary_fraction <= 1.0:
            raise ValueError(f"rotary_fraction must be in (0, 1], got {rotary_fraction}.")
        if max_time is not None and max_time <= 0.0:
            raise ValueError(f"max_time must be positive, got {max_time}.")
        self.dim = dim
        self.rotary_fraction = rotary_fraction  # default to 0.2 in Audio Flamingo Next
        self.max_time = max_time
        # OmniVinci uses (theta = max_time / 2π) with max_time=40s
        # Audio Flamingo Next uses theta=1200 (default)
        if theta is None:
            theta = (self.max_time / (2.0 * math.pi)) if self.max_time is not None else 1200.0
        self.theta = theta
        # Number of channels actually rotated; must be even so it splits into pairs.
        rot_dim = int(dim * self.rotary_fraction)
        rot_dim -= rot_dim % 2
        if rot_dim < 2:
            raise ValueError(
                f"rotary_fraction={self.rotary_fraction} and dim={dim} yield rot_dim={rot_dim}; "
                "need at least 2 channels to rotate."
            )
        self.rot_dim = rot_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))
        # Derived (not trained) and recomputable from config
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, times: Tensor) -> Tensor:
        """Rotate ``x`` according to per-frame ``times``.

        Args:
            x: Feature embeddings of shape ``(B, T, C)`` (channel-last).
            times: Per-frame absolute time in seconds, shape ``(B, T)`` (or any
                shape broadcastable to ``x[..., 0]``).

        Returns:
            Tensor of the same shape and dtype as ``x``, with the first
            ``rot_dim`` channels rotated by the time-dependent angle and the rest
            passed through unchanged.
        """
        ori_dtype = x.dtype
        # OmniVinci uses fp64, but fp32 is ample for this bounded angle math and cheaper on GPU.
        # maybe we can use bfloat16 here and reput autocast?
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()           # do we need it?
            x_rot, x_pass = x[..., : self.rot_dim], x[..., self.rot_dim :]

            times = times.float()   # do we need it?
            if self.max_time is not None:
                # OmniVinci normalization: map [0, max_time] -> [0, 2π] so the fastest channel
                # (inv_freq[0] == 1) completes exactly one rotation over max_time
                times = times / self.max_time * (2.0 * math.pi)

            # angles: (..., T, rot_dim/2) -> (..., T, rot_dim)
            freqs = times.unsqueeze(-1) * self.inv_freq.to(device=x.device, dtype=torch.float32)
            # Interleave each frequency twice ([f0, f0, f1, f1, ...])
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            cos, sin = emb.cos(), emb.sin()

            out = torch.cat((x_rot * cos + rotate_half(x_rot) * sin, x_pass), dim=-1)
        return out.to(ori_dtype)
