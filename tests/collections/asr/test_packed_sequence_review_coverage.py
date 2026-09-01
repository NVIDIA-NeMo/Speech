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

from nemo.collections.asr.modules.parallel_expert_encoder_ggemm import (
    ParallelExpertEncoderPT,
    _validate_packed_expert_lengths,
)
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output, unpack_encoder_output
from tests.collections.asr.test_packed_transformer_encoder import _make_moe_encoder
from tests.collections.asr.test_parallel_expert_encoder_ggemm import (
    _MEL_FEATURES,
    _N_SPK,
    build_toy_pe_encoder,
    toy_sortformer_modules_cfg,
    toy_sound_expert_cfg,
    toy_speaker_expert_cfg,
    toy_speech_expert_cfg,
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


def test_synthetic_legacy_shaped_pee_nemo_archive_loads_strictly_and_enables_packed_path(tmp_path):
    torch.manual_seed(0)
    source = build_toy_pe_encoder().eval()
    cfg = OmegaConf.create(
        {
            "target": "nemo.collections.asr.modules.parallel_expert_encoder_ggemm.ParallelExpertEncoderPT",
            "speech_expert_cfg": toy_speech_expert_cfg(),
            "speaker_expert_cfg": toy_speaker_expert_cfg(),
            "sound_expert_cfg": toy_sound_expert_cfg(),
            "sortformer_modules_cfg": toy_sortformer_modules_cfg(),
            "asr_normalize_type": "per_feature",
            "merge_sound_expert_to_asr": True,
        }
    )
    config_bytes = OmegaConf.to_yaml(cfg).encode()
    weights = io.BytesIO()
    torch.save({f"encoder.{key}": value for key, value in source.state_dict().items()}, weights)
    archive = tmp_path / "synthetic_legacy_shaped_pee.nemo"
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


@pytest.mark.parametrize(
    ("target_mode", "always_run_diarization"), [("none", True), ("mixed", True), ("external", False)]
)
def test_pee_packed_fusion_matches_legacy_routing_modes(target_mode, always_run_diarization):
    torch.manual_seed(0)
    encoder = build_toy_pe_encoder(always_run_diarization=always_run_diarization).eval()
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
    torch.testing.assert_close(restored[valid], legacy.transpose(1, 2)[valid], rtol=1e-5, atol=1e-6)


def test_pee_packed_speaker_threshold_edges_match_legacy():
    encoder = build_toy_pe_encoder().eval()
    lengths = torch.tensor([3, 2])
    padded = torch.randn(2, 3, encoder.d_model)
    packed = pack_encoder_output(padded, lengths)
    threshold = encoder.speaker_activity_threshold
    targets = torch.full((2, 3, _N_SPK), threshold)
    targets[:, 0, 0] = threshold - 1e-6
    targets[:, 1, 1] = threshold + 1e-6

    with torch.no_grad():
        legacy = encoder._fuse_diar_and_asr(padded.transpose(1, 2), targets).transpose(1, 2)
        compact = encoder._fuse_diar_and_asr_sequence_packed(packed, targets)

    restored = unpack_encoder_output(compact, total_length=3)
    valid = torch.arange(3)[None, :] < lengths[:, None]
    torch.testing.assert_close(restored[valid], legacy[valid])


def test_pee_packed_rejects_mismatched_expert_lengths():
    outputs = {
        "speech": pack_encoder_output(torch.randn(2, 3, 4), torch.tensor([3, 2])),
        "sound": pack_encoder_output(torch.randn(2, 3, 4), torch.tensor([3, 1])),
    }

    with pytest.raises(ValueError, match="do not match"):
        _validate_packed_expert_lengths(outputs)


def test_pee_packed_training_uses_checkpointable_path_even_with_legacy_fused_flag(monkeypatch):
    encoder = build_toy_pe_encoder(fused_forward_in_training=True).train()
    original = encoder._forward_all_sequence_packed_training
    calls = 0

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(encoder, "_forward_all_sequence_packed_training", tracked)
    with torch.no_grad():
        encoder.forward_sequence_packed(
            torch.randn(1, _MEL_FEATURES, 24),
            torch.tensor([24]),
            spk_targets=torch.zeros(1, 3, _N_SPK),
        )

    assert calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE packed gradient parity requires CUDA")
def test_pee_packed_matches_legacy_input_and_parameter_gradients():
    torch.manual_seed(0)
    legacy_encoder = build_toy_pe_encoder(freeze_speaker=True, freeze_sound=True).cuda().eval()
    packed_encoder = copy.deepcopy(legacy_encoder)
    legacy_mels = torch.randn(2, _MEL_FEATURES, 32, device="cuda", requires_grad=True)
    packed_mels = legacy_mels.detach().clone().requires_grad_()
    lengths = torch.tensor([32, 17], device="cuda")
    targets = torch.zeros(2, 4, _N_SPK, device="cuda")
    targets[0, :, 0] = 1.0

    legacy, output_lengths = legacy_encoder(legacy_mels, lengths, spk_targets=targets)
    packed = packed_encoder.forward_sequence_packed(packed_mels, lengths, spk_targets=targets)
    valid = torch.arange(legacy.shape[-1], device="cuda")[None, :] < output_lengths[:, None]
    legacy.transpose(1, 2)[valid].float().square().mean().backward()
    packed.data.float().square().mean().backward()

    torch.testing.assert_close(packed_mels.grad, legacy_mels.grad, rtol=2e-3, atol=2e-4)
    for name in ("asr_norm.weight", "pee.experts.speech.layers.0.attn.w_qkv.weight"):
        legacy_grad = dict(legacy_encoder.named_parameters())[name].grad
        packed_grad = dict(packed_encoder.named_parameters())[name].grad
        torch.testing.assert_close(packed_grad, legacy_grad, rtol=2e-3, atol=2e-4)
