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

import torch

from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.collections.speechlm2.modules.perception import IndependentDualEncoder


def make_encoder(d_model: int, *, subsampling_factor: int = 2) -> TransformerEncoder:
    return TransformerEncoder(
        feat_in=4,
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        subsampling="feature_stacking",
        subsampling_factor=subsampling_factor,
        ff_expansion=1.0,
        self_attention_model="rope",
        pos_emb_max_len=64,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        sync_max_audio_length=False,
    )


def test_independent_dual_encoder_chunks_only_frozen_auxiliary_branch():
    torch.manual_seed(7)
    asr = make_encoder(32)
    auxiliary = make_encoder(32)
    dual = IndependentDualEncoder(
        asr,
        auxiliary,
        frame_shift_seconds=0.01,
        asr_chunk_size_seconds=None,
        auxiliary_chunk_size_seconds=0.04,
        freeze_auxiliary=True,
    ).train()

    features = torch.randn(2, 4, 17)
    lengths = torch.tensor([17, 10], dtype=torch.int64)
    packed_features = pack_encoder_output(features.transpose(1, 2), lengths)
    with torch.no_grad():
        asr_reference = asr.forward_sequence_packed(packed_features, packed_features.lengths)

    auxiliary_calls = []
    auxiliary_forward = auxiliary.forward_sequence_packed

    def record_auxiliary_call(audio_signal, length, bypass_pre_encode=False, **kwargs):
        auxiliary_calls.append((length.detach().clone(), bypass_pre_encode))
        return auxiliary_forward(audio_signal, length, bypass_pre_encode=bypass_pre_encode, **kwargs)

    auxiliary.forward_sequence_packed = record_auxiliary_call
    output = dual.forward_sequence_packed(packed_features, packed_features.lengths)

    assert output.lengths.tolist() == [9, 5]
    assert output.data.shape == (14, 64)
    torch.testing.assert_close(output.data[:, :32], asr_reference.data)
    assert len(auxiliary_calls) == 1
    assert auxiliary_calls[0][0].tolist() == [2, 2, 2, 2, 1, 2, 2, 1]
    assert auxiliary_calls[0][1] is True
    assert not auxiliary.training
    assert all(not parameter.requires_grad for parameter in auxiliary.parameters())

    output.data.square().mean().backward()
    assert any(parameter.grad is not None for parameter in asr.parameters())
    assert all(parameter.grad is None for parameter in auxiliary.parameters())


def test_independent_dual_encoder_rejects_mismatched_frame_rates():
    asr = make_encoder(32, subsampling_factor=2)
    auxiliary = make_encoder(32, subsampling_factor=4)
    try:
        IndependentDualEncoder(asr, auxiliary, frame_shift_seconds=0.01)
    except ValueError as error:
        assert "subsampling_factor" in str(error)
    else:
        raise AssertionError("Expected mismatched subsampling factors to be rejected.")
