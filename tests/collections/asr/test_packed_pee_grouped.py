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
from omegaconf import OmegaConf

from nemo.collections.asr.parts.packed_sequence import PackedEncoderActivations, pack_encoder_output
from tests.collections.asr.test_parallel_expert_encoder_two_branch import (
    _MEL_FEATURES,
    _N_SPK,
    build_toy_packed_pe_encoder,
    toy_packed_diarization_model_cfg,
    toy_transformer_asr_encoder_cfg,
)


@pytest.mark.unit
def test_packed_pee_reaches_salm_perception_dispatch():
    from nemo.collections.speechlm2.modules.perception import AudioPerceptionModule
    from nemo.collections.speechlm2.parts.encoder_chunking import _encode_perception_unpadded

    perception = AudioPerceptionModule(
        OmegaConf.create(
            {
                "preprocessor": {
                    "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                    "features": _MEL_FEATURES,
                },
                "encoder": {
                    "_target_": "nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoder",
                    "_recursive_": False,
                    "asr_encoder_type": "transformer",
                    "asr_encoder_cfg": toy_transformer_asr_encoder_cfg(),
                    "diarization_model_cfg": toy_packed_diarization_model_cfg(),
                },
                "modality_adapter": {
                    "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                    "output_dim": 32,
                },
                "rote": None,
            }
        )
    ).eval()
    inputs = {"processed_signal": torch.randn(2, _MEL_FEATURES, 40), "processed_signal_length": torch.tensor([40, 17])}
    with torch.no_grad():
        expected = _encode_perception_unpadded(perception, inputs, sequence_packed=False)
        actual = _encode_perception_unpadded(perception, inputs, sequence_packed=True)
    assert perception.supports_sequence_packed_output
    assert len(actual) == len(expected) == 2
    for packed, dense in zip(actual, expected):
        torch.testing.assert_close(packed, dense, rtol=1e-4, atol=1e-5)


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
