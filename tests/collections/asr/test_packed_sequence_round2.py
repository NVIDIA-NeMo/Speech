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

import copy

import pytest
import torch
from torch.utils._pytree import tree_flatten

from nemo.collections.asr.modules.moe_transformer_encoder import MoEFeedForward, MoETransformerEncoder
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


@pytest.mark.parametrize(("top_k", "router_type"), [(1, "switch"), (2, "omni")])
def test_moe_packed_routing_statistics_auxiliary_loss_and_reset(top_k, router_type):
    torch.manual_seed(0)
    encoder = MoETransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=1,
        subsampling_factor=2,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        self_attention_model="rope",
        moe_num_experts=4,
        moe_top_k=top_k,
        moe_router_type=router_type,
        sync_max_audio_length=False,
    ).train()
    lengths = torch.tensor([7, 3, 1])

    with torch.no_grad():
        encoder.forward_sequence_packed(torch.randn(3, 7, 32), lengths, bypass_pre_encode=True)

    ffn = encoder.layers[0].ffn
    assert isinstance(ffn, MoEFeedForward)
    num_tokens = int(lengths.sum())
    assert ffn._num_tokens == num_tokens
    assert int(ffn._expert_counts.sum()) == num_tokens * top_k
    torch.testing.assert_close(ffn._gate_prob_sum.sum(), torch.tensor(float(num_tokens)))
    expected_aux = (
        ffn.num_experts * (ffn._expert_counts.float() / num_tokens * (ffn._gate_prob_sum / num_tokens)).sum()
    )
    torch.testing.assert_close(ffn._aux_loss.float(), expected_aux.float())
    torch.testing.assert_close(encoder._cum_counts[0], ffn._expert_counts)
    torch.testing.assert_close(encoder._cum_prob_sum[0], ffn._gate_prob_sum)
    assert int(encoder._cum_tokens[0]) == num_tokens

    metrics = encoder.get_moe_metrics(distributed=False, reset=True)
    assert metrics is not None
    assert not encoder._cum_counts.any()
    assert not encoder._cum_prob_sum.any()
    assert not encoder._cum_tokens.any()


def test_moe_packed_auxiliary_loss_is_padding_neutral_while_legacy_contract_is_unchanged():
    torch.manual_seed(0)
    encoder = MoETransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=1,
        subsampling_factor=2,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        self_attention_model="rope",
        moe_num_experts=4,
        moe_top_k=2,
        sync_max_audio_length=False,
    ).train()
    inputs = torch.randn(2, 6, 32)
    lengths = torch.tensor([6, 2])

    with torch.no_grad():
        encoder(inputs, lengths, bypass_pre_encode=True)
        legacy_tokens = encoder.layers[0].ffn._num_tokens
        legacy_auxiliary_loss = encoder.get_moe_auxiliary_loss()
        encoder.forward_sequence_packed(inputs, lengths, bypass_pre_encode=True)
        packed_tokens = encoder.layers[0].ffn._num_tokens
        packed_auxiliary_loss = encoder.get_moe_auxiliary_loss()

    assert legacy_tokens == inputs.shape[0] * inputs.shape[1]
    assert packed_tokens == int(lengths.sum())
    assert torch.isfinite(legacy_auxiliary_loss)
    assert torch.isfinite(packed_auxiliary_loss)


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
    packed_encoder = copy.deepcopy(dense_encoder)
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
