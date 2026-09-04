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
import io
import tarfile

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output, unpack_encoder_output
from tests.collections.asr.test_packed_transformer_encoder import _make_moe_encoder
from tests.collections.asr.test_parallel_expert_encoder_two_branch import (
    _MEL_FEATURES,
    _N_SPK,
    build_toy_packed_pe_encoder,
    toy_packed_diarization_model_cfg,
    toy_transformer_asr_encoder_cfg,
)


def test_previous_moe_state_loads_strictly_before_and_after_packed_use():
    previous = _make_moe_encoder()
    previous_state = copy.deepcopy(previous.state_dict())
    restored = _make_moe_encoder()

    restored.load_state_dict(previous_state, strict=True)
    with torch.no_grad():
        restored.forward_sequence_packed(torch.randn(2, 8, 12), torch.tensor([12, 5]))

    assert set(restored.state_dict()) == set(previous_state)


def test_sequence_packed_training_dropout_is_finite_and_reproducible_within_path():
    encoder = TransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        subsampling_factor=2,
        drop_rate=0.2,
        dropout_pre_encoder=0.2,
        dropout_emb=0.2,
        self_attention_model="rope",
        sync_max_audio_length=False,
    ).train()
    audio = torch.randn(3, 8, 12)
    lengths = torch.tensor([12, 7, 3])

    with torch.no_grad():
        torch.manual_seed(17)
        first = encoder.forward_sequence_packed(audio, lengths).data
        torch.manual_seed(17)
        second = encoder.forward_sequence_packed(audio, lengths).data

    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_synthetic_canonical_pee_nemo_archive_loads_strictly_and_enables_packed_path(tmp_path):
    torch.manual_seed(0)
    source = build_toy_packed_pe_encoder().eval()
    cfg = OmegaConf.create(
        {
            "target": "nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT",
            "asr_encoder_type": "transformer",
            "asr_encoder_cfg": toy_transformer_asr_encoder_cfg(),
            "diarization_model_cfg": toy_packed_diarization_model_cfg(),
            "asr_normalize_type": "per_feature",
            "speaker_feature_config_version": 1,
            "speaker_feature_mode": "continuous",
            "speaker_activity_threshold": None,
            "sync_max_audio_length": False,
        }
    )
    config_bytes = OmegaConf.to_yaml(cfg).encode()
    weights = io.BytesIO()
    torch.save({f"encoder.{key}": value for key, value in source.state_dict().items()}, weights)
    archive = tmp_path / "synthetic_canonical_pee.nemo"
    with tarfile.open(archive, "w") as tar:
        config_info = tarfile.TarInfo("model_config.yaml")
        config_info.size = len(config_bytes)
        tar.addfile(config_info, io.BytesIO(config_bytes))
        weight_bytes = weights.getvalue()
        weight_info = tarfile.TarInfo("model_weights.ckpt")
        weight_info.size = len(weight_bytes)
        tar.addfile(weight_info, io.BytesIO(weight_bytes))

    restored = ParallelExpertEncoderPT.load_from_nemo(str(archive), strict=True).eval()

    assert set(restored.state_dict()) == set(source.state_dict())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    with torch.no_grad():
        packed = restored.forward_sequence_packed(
            torch.randn(2, _MEL_FEATURES, 24),
            torch.tensor([24, 11]),
        )
    assert packed.total_tokens == int(packed.lengths.sum())


@pytest.mark.parametrize("target_mode", ["none", "mixed", "external"])
def test_pee_packed_fusion_matches_dense_routing_modes(target_mode):
    torch.manual_seed(0)
    encoder = build_toy_packed_pe_encoder().eval()
    mels = torch.randn(3, _MEL_FEATURES, 40)
    lengths = torch.tensor([40, 23, 9])
    targets = None
    if target_mode != "none":
        targets = torch.zeros(3, 5, _N_SPK)
        targets[0, :, 0] = 1.0
        targets[2, :, 2] = 1.0
        if target_mode == "mixed":
            targets[1] = -1.0

    with torch.no_grad():
        legacy, output_lengths = encoder(mels, lengths, spk_targets=targets)
        packed = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)

    restored = unpack_encoder_output(packed, total_length=legacy.shape[-1])
    valid = torch.arange(legacy.shape[-1])[None, :] < output_lengths[:, None]
    torch.testing.assert_close(restored[valid], legacy.transpose(1, 2)[valid], rtol=1e-4, atol=1e-5)


def test_pee_packed_speaker_threshold_edges_match_legacy():
    encoder = build_toy_packed_pe_encoder(speaker_activity_threshold=0.5).eval()
    lengths = torch.tensor([3, 2])
    padded = torch.randn(2, 3, encoder.d_model)
    packed = pack_encoder_output(padded, lengths)
    threshold = encoder.speaker_activity_threshold
    targets = torch.full((2, 3, _N_SPK), threshold)
    targets[:, 0, 0] = threshold - 1e-6
    targets[:, 1, 1] = threshold + 1e-6

    with torch.no_grad():
        legacy = encoder._fuse_diar_and_asr(padded.transpose(1, 2), targets).transpose(1, 2)
        compact = encoder._fuse_diar_and_asr_packed(packed, targets)

    restored = unpack_encoder_output(compact, total_length=3)
    valid = torch.arange(3)[None, :] < lengths[:, None]
    torch.testing.assert_close(restored[valid], legacy[valid])


def test_pee_packed_rejects_mismatched_branch_metadata(monkeypatch):
    encoder = build_toy_packed_pe_encoder().eval()
    diar = pack_encoder_output(torch.randn(2, 3, _N_SPK), torch.tensor([3, 1]))
    asr = pack_encoder_output(torch.randn(2, 3, encoder.d_model), torch.tensor([3, 2]))
    monkeypatch.setattr(encoder, "_run_diarization_packed", lambda features: diar)
    monkeypatch.setattr(encoder, "_run_asr_packed", lambda features: asr)

    with pytest.raises(RuntimeError, match="metadata diverged"):
        encoder.forward_sequence_packed(torch.randn(2, _MEL_FEATURES, 8), torch.tensor([8, 4]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE packed gradient parity requires CUDA")
def test_pee_packed_matches_dense_input_and_parameter_gradients():
    torch.manual_seed(0)
    dense_encoder = build_toy_packed_pe_encoder(freeze_asr=False, freeze_diar=True).cuda().eval()
    packed_encoder = copy.deepcopy(dense_encoder)
    dense_mels = torch.randn(2, _MEL_FEATURES, 32, device="cuda", requires_grad=True)
    packed_mels = dense_mels.detach().clone().requires_grad_()
    lengths = torch.tensor([32, 17], device="cuda")
    targets = torch.zeros(2, 4, _N_SPK, device="cuda")
    targets[0, :, 0] = 1.0

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
