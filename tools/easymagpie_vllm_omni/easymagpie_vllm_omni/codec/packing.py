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
"""EasyMagpie acoustic-code packing helpers."""

from __future__ import annotations

import torch


def unstack_acoustic_codes(
    codes: torch.Tensor,
    *,
    num_codebooks: int,
    frame_stacking_factor: int,
) -> torch.Tensor:
    """Convert predictor rows ``[..., T, C*S]`` to codec rows ``[..., T*S, C]``.

    EasyMagpie stacks adjacent codec frames inside each codebook. For ``S=2``
    a predictor row is ordered as ``c0_t0, c0_t1, c1_t0, c1_t1, ...``. This
    function restores time-major 8-codebook frames without moving data before
    the final packed reshape.
    """
    expected = num_codebooks * frame_stacking_factor
    if codes.dim() < 2 or codes.shape[-1] != expected:
        raise ValueError(f"expected [..., T, {expected}] stacked codes, got {tuple(codes.shape)}")

    leading_shape = codes.shape[:-2]
    frames = codes.shape[-2]
    return (
        codes.unflatten(-1, (num_codebooks, frame_stacking_factor))
        .transpose(-2, -1)
        .reshape(*leading_shape, frames * frame_stacking_factor, num_codebooks)
    )
