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
from torch.utils._pytree import tree_flatten

from nemo.collections.asr.parts.packed_sequence import PackedEncoderActivations, pack_encoder_output
from tests.collections.asr.test_parallel_expert_encoder_two_branch import (
    _MEL_FEATURES,
    _N_SPK,
    build_toy_packed_pe_encoder,
)


def test_packed_encoder_activations_is_registered_as_pytree():
    packed = pack_encoder_output(torch.randn(2, 4, 3), torch.tensor([4, 2]))

    leaves, _ = tree_flatten(packed)

    assert all(leaf is not packed for leaf in leaves)
    assert any(leaf is packed.data for leaf in leaves)
    assert any(leaf is packed.lengths for leaf in leaves)
    assert any(leaf is packed.cu_seqlens for leaf in leaves)


def test_packed_output_with_data_reuses_validated_metadata_and_preserves_gradients():
    packed = pack_encoder_output(torch.randn(2, 4, 3), torch.tensor([4, 2]))
    replacement = torch.randn(6, 5, requires_grad=True)

    updated = packed.with_data(replacement)

    assert updated.lengths is packed.lengths
    assert updated.cu_seqlens is packed.cu_seqlens
    assert updated.max_seqlen == packed.max_seqlen
    updated.data.square().sum().backward()
    assert replacement.grad is not None
    with pytest.raises(ValueError, match="replacement data"):
        packed.with_data(torch.randn(5, 3))


def test_canonical_pee_packed_output_preserves_compact_metadata():
    torch.manual_seed(0)
    encoder = build_toy_packed_pe_encoder().eval()
    mels = torch.randn(2, _MEL_FEATURES, 24)
    lengths = torch.tensor([24, 11])
    targets = torch.zeros(2, 3, _N_SPK)

    with torch.no_grad():
        output = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)

    assert isinstance(output, PackedEncoderActivations)
    assert output.total_tokens == int(output.lengths.sum())
    assert output.cu_seqlens.tolist() == [0, *output.lengths.cumsum(0).tolist()]


def test_canonical_pee_dense_contract_is_unchanged_after_packed_use():
    encoder = build_toy_packed_pe_encoder().eval()
    mels = torch.randn(2, _MEL_FEATURES, 24)
    lengths = torch.tensor([24, 11])
    targets = torch.zeros(2, 3, _N_SPK)
    state_keys = set(encoder.state_dict())

    with torch.no_grad():
        packed = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)
        dense, dense_lengths = encoder(mels, lengths, spk_targets=targets)

    restored = torch.cat(
        [dense[index, :, : int(length)].transpose(0, 1) for index, length in enumerate(dense_lengths)]
    )
    torch.testing.assert_close(packed.data, restored, rtol=1e-5, atol=1e-6)
    assert set(encoder.state_dict()) == state_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE ASR-gradient parity requires CUDA")
def test_canonical_pee_packed_matches_dense_trainable_asr_gradients():
    torch.manual_seed(0)
    dense_encoder = build_toy_packed_pe_encoder(freeze_asr=False, freeze_diar=True).cuda().eval()
    packed_encoder = build_toy_packed_pe_encoder(freeze_asr=False, freeze_diar=True).cuda().eval()
    packed_encoder.load_state_dict(dense_encoder.state_dict(), strict=True)
    dense_mels = torch.randn(2, _MEL_FEATURES, 32, device="cuda", requires_grad=True)
    packed_mels = dense_mels.detach().clone().requires_grad_()
    lengths = torch.tensor([32, 17], device="cuda")
    targets = torch.zeros(2, 4, _N_SPK, device="cuda")

    dense, output_lengths = dense_encoder(dense_mels, lengths, spk_targets=targets)
    packed = packed_encoder.forward_sequence_packed(packed_mels, lengths, spk_targets=targets)
    valid = torch.arange(dense.shape[-1], device="cuda")[None, :] < output_lengths[:, None]
    dense.transpose(1, 2)[valid].float().square().mean().backward()
    packed.data.float().square().mean().backward()

    torch.testing.assert_close(packed_mels.grad, dense_mels.grad, rtol=2e-3, atol=2e-4)
    for name, dense_parameter in dense_encoder.named_parameters():
        if not name.startswith(("asr_encoder.", "asr_norm.")) or not dense_parameter.requires_grad:
            continue
        packed_grad = dict(packed_encoder.named_parameters())[name].grad
        assert dense_parameter.grad is not None and packed_grad is not None
        torch.testing.assert_close(packed_grad, dense_parameter.grad, rtol=2e-3, atol=2e-4)
