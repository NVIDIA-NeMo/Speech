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

"""
Regression test for a silent precision-loss bug in
nemo.collections.tts.modules.transformer.PositionalEmbedding.

PyTorch Lightning's "bf16-true" precision setting calls torch.set_default_dtype(torch.bfloat16)
for the duration of model construction (see nemo/collections/speechlm2/parts/precision.py's
fp32_precision() docstring, which documents this exact PTL behavior as a known hazard).

PositionalEmbedding.__init__ builds its `inv_freq` buffer with
    torch.arange(0.0, demb, 2.0) / demb
and no explicit dtype. Under a bf16 default dtype, this computes the RoPE/sinusoidal
frequency table itself in bfloat16 -- losing precision in a calculation (division then
power-of-10000) that is numerically sensitive -- instead of computing in float32 and
only casting the *result* to the training dtype.
"""

import pytest
import torch

from nemo.collections.tts.modules.transformer import PositionalEmbedding


@pytest.mark.unit
class TestPositionalEmbeddingDtype:
    def test_inv_freq_buffer_is_float32_under_bf16_default_dtype(self):
        """inv_freq must always be computed/stored in float32, regardless of the
        torch default dtype active at construction time (as happens under PTL's
        bf16-true precision)."""
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            pos_emb = PositionalEmbedding(demb=512)
        finally:
            torch.set_default_dtype(default_dtype)

        assert pos_emb.inv_freq.dtype == torch.float32, (
            f"inv_freq should be computed in float32 to avoid precision loss, "
            f"got {pos_emb.inv_freq.dtype}"
        )

    def test_inv_freq_values_match_float32_reference_under_bf16_default_dtype(self):
        """Even when the dtype happens to come out as float32, make sure the actual
        values match a float32-computed reference -- i.e. no part of the arithmetic
        silently ran in bfloat16 before being cast back up."""
        default_dtype = torch.get_default_dtype()

        torch.set_default_dtype(torch.float32)
        reference = PositionalEmbedding(demb=512).inv_freq.clone()

        torch.set_default_dtype(torch.bfloat16)
        try:
            under_test = PositionalEmbedding(demb=512)
        finally:
            torch.set_default_dtype(default_dtype)

        assert under_test.inv_freq.dtype == torch.float32
        torch.testing.assert_close(under_test.inv_freq, reference, rtol=1e-6, atol=1e-6)
