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

import pytest
import torch

import nemo.collections.asr.modules.ggemm_transformer_encoder as ggemm_module
from nemo.collections.asr.parts.packed_sequence import PackedEncoderActivations, pack_encoder_output
from tests.collections.asr.test_parallel_expert_encoder_two_branch import (
    _MEL_FEATURES,
    _N_SPK,
    build_toy_packed_pe_encoder,
)


@pytest.mark.unit
def test_canonical_pee_packed_path_matches_dense_without_legacy_grouped_runtime():
    torch.manual_seed(0)
    encoder = build_toy_packed_pe_encoder().eval()
    mels = torch.randn(2, _MEL_FEATURES, 40)
    lengths = torch.tensor([40, 17])
    targets = torch.zeros(2, 5, _N_SPK)

    with torch.no_grad():
        dense, dense_lengths = encoder(mels, lengths, spk_targets=targets)
        packed = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)

    valid = torch.arange(dense.shape[-1])[None, :] < dense_lengths[:, None]
    torch.testing.assert_close(packed.data, dense.transpose(1, 2)[valid], rtol=1e-5, atol=1e-6)
    assert not hasattr(encoder, "pee")
    assert not hasattr(encoder, "sequence_packed_execution_mode")


@pytest.mark.unit
def test_canonical_pee_accepts_token_flat_mels():
    torch.manual_seed(0)
    encoder = build_toy_packed_pe_encoder().eval()
    mels = torch.randn(2, _MEL_FEATURES, 40)
    lengths = torch.tensor([40, 17])
    mels.masked_fill_(torch.arange(mels.shape[-1])[None, None, :] >= lengths[:, None, None], 0.0)
    packed_mels = pack_encoder_output(mels.transpose(1, 2), lengths)
    targets = torch.zeros(2, 5, _N_SPK)

    with torch.no_grad():
        dense_input = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)
        packed_input = encoder.forward_sequence_packed(packed_mels, spk_targets=targets)

    assert isinstance(packed_input, PackedEncoderActivations)
    assert torch.equal(packed_input.lengths, dense_input.lengths)
    torch.testing.assert_close(packed_input.data, dense_input.data, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_grouped_biasless_linear_uses_bmm_without_zero_bias_allocation(monkeypatch):
    """Keep direct coverage for the independent grouped-transformer primitive."""
    linears = torch.nn.ModuleList([torch.nn.Linear(4, 6, bias=False) for _ in range(3)])
    inputs = torch.randn(3, 5, 4, requires_grad=True)
    expected = torch.stack([linear(inputs[index]) for index, linear in enumerate(linears)])

    def reject_baddbmm(*args, **kwargs):
        raise AssertionError("an all-biasless group must use bmm")

    monkeypatch.setattr(torch, "baddbmm", reject_baddbmm)
    actual = ggemm_module._grouped_linear(inputs, linears)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()

    assert inputs.grad is not None
    assert all(linear.weight.grad is not None for linear in linears)
