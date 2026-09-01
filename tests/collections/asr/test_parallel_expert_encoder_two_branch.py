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

import importlib.util
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn

import nemo.collections.asr.modules.parallel_expert_encoder as pee_module
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.modules.parallel_expert_encoder import (
    ParallelExpertEncoder,
    ParallelExpertEncoderPT,
    _clone_config,
    _default_dtype,
    _disable_dist_feature_sync,
)
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output, unpack_encoder_output
from nemo.collections.asr.parts.preprocessing.features import normalize_batch, normalize_packed_batch

_PEE = getattr(ParallelExpertEncoder, "__wrapped__", ParallelExpertEncoder)

_MEL_FEATURES = 128
_ASR_D_MODEL = 32
_DIAR_FC_D_MODEL = 32
_DIAR_TF_D_MODEL = 16
_N_SPK = 4
_SUBSAMPLING_FACTOR = 8


def toy_asr_encoder_cfg() -> DictConfig:
    return DictConfig(
        {
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "feat_in": _MEL_FEATURES,
            "feat_out": -1,
            "n_layers": 1,
            "d_model": _ASR_D_MODEL,
            "subsampling": "dw_striding",
            "subsampling_factor": _SUBSAMPLING_FACTOR,
            "subsampling_conv_channels": 16,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": 4,
            "att_context_size": [-1, -1],
            "conv_kernel_size": 9,
            "dropout": 0.0,
            "dropout_pre_encoder": 0.0,
            "dropout_emb": 0.0,
            "dropout_att": 0.0,
        }
    )


def toy_transformer_asr_encoder_cfg() -> DictConfig:
    return DictConfig(
        {
            "_target_": "nemo.collections.asr.modules.transformer_encoder.TransformerEncoder",
            "feat_in": _MEL_FEATURES,
            "d_model": _ASR_D_MODEL,
            "n_heads": 2,
            "n_layers": 1,
            "subsampling": "feature_stacking",
            "subsampling_factor": _SUBSAMPLING_FACTOR,
            "drop_rate": 0.0,
            "dropout_pre_encoder": 0.0,
            "dropout_emb": 0.0,
            "qkv_bias": False,
            "qk_norm": True,
            "ff_expansion": 2.0,
            "pre_block_norm": True,
            "self_attention_model": "rope",
            "attn_mode": "full",
            "sync_max_audio_length": False,
        }
    )


def toy_diarization_model_cfg() -> DictConfig:
    defaults = {"fc_d_model": _DIAR_FC_D_MODEL, "tf_d_model": _DIAR_TF_D_MODEL}
    return DictConfig(
        {
            "target": "nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel",
            "sample_rate": 16000,
            "pil_weight": 0.5,
            "ats_weight": 0.5,
            "max_num_of_spks": _N_SPK,
            "streaming_mode": False,
            "async_streaming": False,
            "model_defaults": DictConfig(defaults),
            "preprocessor": DictConfig(
                {
                    "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                    "normalize": "per_feature",
                    "window_size": 0.025,
                    "sample_rate": 16000,
                    "window_stride": 0.01,
                    "window": "hann",
                    "features": _MEL_FEATURES,
                    "n_fft": 512,
                    "frame_splicing": 1,
                    "dither": 0.00001,
                }
            ),
            "encoder": DictConfig(
                {
                    "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                    "feat_in": _MEL_FEATURES,
                    "feat_out": -1,
                    "n_layers": 1,
                    "d_model": _DIAR_FC_D_MODEL,
                    "subsampling": "dw_striding",
                    "subsampling_factor": _SUBSAMPLING_FACTOR,
                    "subsampling_conv_channels": 16,
                    "causal_downsampling": False,
                    "ff_expansion_factor": 4,
                    "self_attention_model": "rel_pos",
                    "n_heads": 4,
                    "att_context_size": [-1, -1],
                    "conv_kernel_size": 9,
                    "conv_norm_type": "batch_norm",
                    "dropout": 0.0,
                    "dropout_pre_encoder": 0.0,
                    "dropout_emb": 0.0,
                    "dropout_att": 0.0,
                }
            ),
            "transformer_encoder": DictConfig(
                {
                    "_target_": "nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder",
                    "num_layers": 1,
                    "hidden_size": _DIAR_TF_D_MODEL,
                    "inner_size": 32,
                    "num_attention_heads": 4,
                    "attn_score_dropout": 0.0,
                    "attn_layer_dropout": 0.0,
                    "ffn_dropout": 0.0,
                    "hidden_act": "relu",
                    "pre_ln": False,
                    "pre_ln_final_layer_norm": True,
                }
            ),
            "sortformer_modules": DictConfig(
                {
                    "_target_": "nemo.collections.asr.modules.sortformer_modules.SortformerModules",
                    "num_spks": _N_SPK,
                    "dropout_rate": 0.0,
                    "fc_d_model": _DIAR_FC_D_MODEL,
                    "tf_d_model": _DIAR_TF_D_MODEL,
                }
            ),
            "loss": DictConfig(
                {
                    "_target_": "nemo.collections.asr.losses.bce_loss.BCELoss",
                    "weight": None,
                    "reduction": "mean",
                }
            ),
        }
    )


def toy_packed_diarization_model_cfg() -> DictConfig:
    cfg = toy_diarization_model_cfg()
    cfg.encoder = toy_transformer_asr_encoder_cfg()
    cfg.encoder.d_model = _DIAR_FC_D_MODEL
    cfg.encoder.qk_norm = False
    cfg.transformer_encoder.num_layers = 0
    cfg.transformer_encoder.pre_ln = False
    return cfg


def build_toy_pe_encoder(**overrides) -> ParallelExpertEncoder:
    kwargs = {
        "asr_encoder_cfg": toy_asr_encoder_cfg(),
        "diarization_model_cfg": toy_diarization_model_cfg(),
        "asr_normalize_type": "per_feature",
        "online_inference_length": 500,
    }
    kwargs.update(overrides)
    return ParallelExpertEncoder(**kwargs)


def build_toy_packed_pe_encoder(**overrides) -> ParallelExpertEncoder:
    """Construct the canonical PEE with native packed-capable branch encoders."""
    kwargs = {
        "asr_encoder_type": "transformer",
        "asr_encoder_cfg": toy_transformer_asr_encoder_cfg(),
        "diarization_model_cfg": toy_packed_diarization_model_cfg(),
    }
    kwargs.update(overrides)
    return build_toy_pe_encoder(**kwargs)


def bundle_config(**overrides) -> DictConfig:
    config = {
        "target": "nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT",
        "asr_encoder_cfg": toy_asr_encoder_cfg(),
        "diarization_model_cfg": toy_diarization_model_cfg(),
        "asr_normalize_type": "per_feature",
    }
    config.update(overrides)
    return OmegaConf.create(config)


def write_bundle(path, config, state=None):
    config_bytes = OmegaConf.to_yaml(config).encode()
    weights = io.BytesIO()
    torch.save(state or {}, weights)
    with tarfile.open(path, "w") as archive:
        config_info = tarfile.TarInfo("model_config.yaml")
        config_info.size = len(config_bytes)
        archive.addfile(config_info, io.BytesIO(config_bytes))
        weight_bytes = weights.getvalue()
        weight_info = tarfile.TarInfo("model_weights.ckpt")
        weight_info.size = len(weight_bytes)
        archive.addfile(weight_info, io.BytesIO(weight_bytes))


@pytest.mark.unit
def test_clone_config_is_deep_and_handles_none():
    config = OmegaConf.create({"a": {"b": 1}})
    clone = _clone_config(config)
    clone.a.b = 2
    assert config.a.b == 1
    assert _clone_config(None) is None


@pytest.mark.unit
@pytest.mark.parametrize("target_dtype", [torch.float64, torch.float16])
def test_default_dtype_sets_and_restores(target_dtype):
    previous = torch.get_default_dtype()
    with _default_dtype(target_dtype):
        assert torch.get_default_dtype() == target_dtype
    assert torch.get_default_dtype() == previous


@pytest.mark.unit
def test_disable_dist_feature_sync_noop_when_uninitialized():
    assert not dist.is_initialized()
    original = dist.is_initialized
    with _disable_dist_feature_sync():
        pass
    assert dist.is_initialized is original


@pytest.mark.unit
def test_static_helpers_align_and_cast():
    diar = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    aligned = ParallelExpertEncoder._align_diar_frames(diar, 5)
    assert aligned.shape == (1, 5, 3)
    assert torch.equal(aligned[:, -1], diar[:, -1])

    module = nn.Linear(4, 4).to(torch.float64)
    cast = ParallelExpertEncoder._match_module_io(torch.zeros(2, 4), module)
    assert cast.dtype == torch.float64


@pytest.mark.unit
def test_pe_encoder_builds_two_real_branches_and_freezes_diarizer():
    encoder = build_toy_pe_encoder()
    assert isinstance(encoder.asr_encoder, ConformerEncoder)
    assert encoder.asr_encoder_type == "fastconformer"
    assert isinstance(encoder.diarization_model, SortformerEncLabelModel)
    assert encoder.d_model == _ASR_D_MODEL
    assert encoder.subsampling_factor == _SUBSAMPLING_FACTOR
    assert encoder.n_spk == _N_SPK
    assert encoder.diar_normalize_type == "per_feature"
    assert all(not parameter.requires_grad for parameter in encoder.diarization_model.parameters())
    assert any(parameter.requires_grad for parameter in encoder.asr_encoder.parameters())

    encoder.train()
    assert encoder.asr_encoder.training
    assert not encoder.diarization_model.training


@pytest.mark.unit
def test_pe_encoder_selects_native_transformer_asr_branch():
    encoder = build_toy_pe_encoder(
        asr_encoder_type="transformer",
        asr_encoder_cfg=toy_transformer_asr_encoder_cfg(),
    )
    assert isinstance(encoder.asr_encoder, TransformerEncoder)
    assert encoder.asr_encoder_type == "transformer"
    assert encoder.d_model == _ASR_D_MODEL
    assert encoder.subsampling_factor == _SUBSAMPLING_FACTOR


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asr_encoder_type", "asr_encoder_cfg", "expected_class"),
    [
        ("fastconformer", toy_transformer_asr_encoder_cfg, "ConformerEncoder"),
        ("transformer", toy_asr_encoder_cfg, "TransformerEncoder"),
    ],
)
def test_pe_encoder_rejects_asr_encoder_type_config_mismatch(asr_encoder_type, asr_encoder_cfg, expected_class):
    with pytest.raises(TypeError, match=rf"requires .*{expected_class}"):
        build_toy_pe_encoder(
            asr_encoder_type=asr_encoder_type,
            asr_encoder_cfg=asr_encoder_cfg(),
        )


@pytest.mark.unit
def test_pe_encoder_rejects_unknown_asr_encoder_type():
    with pytest.raises(ValueError, match="asr_encoder_type must be one of"):
        build_toy_pe_encoder(asr_encoder_type="auto")


@pytest.mark.unit
def test_freeze_asr_keeps_both_frozen_branches_in_eval():
    encoder = build_toy_pe_encoder(freeze_asr=True)
    encoder.train()
    assert not encoder.asr_encoder.training
    assert not encoder.diarization_model.training
    assert all(not parameter.requires_grad for parameter in encoder.asr_encoder.parameters())


@pytest.mark.unit
def test_pe_encoder_rejects_incompatible_branch_frame_rates():
    diarization_config = toy_diarization_model_cfg()
    diarization_config.encoder.subsampling_factor = 4
    with pytest.raises(ValueError, match="embedded diarization encoder subsampling factor"):
        build_toy_pe_encoder(diarization_model_cfg=diarization_config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asr_encoder_type", "asr_encoder_cfg"),
    [
        ("fastconformer", toy_asr_encoder_cfg),
        ("transformer", toy_transformer_asr_encoder_cfg),
    ],
)
def test_offline_forward_runs_both_branches(asr_encoder_type, asr_encoder_cfg):
    encoder = build_toy_pe_encoder(
        asr_encoder_type=asr_encoder_type,
        asr_encoder_cfg=asr_encoder_cfg(),
    ).eval()
    mels = torch.randn(2, _MEL_FEATURES, 160)
    lengths = torch.tensor([160, 137])
    with torch.no_grad():
        output, output_lengths = encoder(mels, lengths)
    assert output.shape[:2] == (2, _ASR_D_MODEL)
    assert output.shape[-1] == int(output_lengths.max())
    assert torch.isfinite(output).all()


@pytest.mark.unit
def test_offline_sortformer_receives_per_feature_normalized_mels(monkeypatch):
    encoder = build_toy_pe_encoder().eval()
    mels = 4.0 * torch.randn(2, _MEL_FEATURES, 80) + 17.0
    lengths = torch.tensor([80, 53])
    observed = []
    original_frontend = encoder.diarization_model.frontend_encoder

    def tracked_frontend(**kwargs):
        observed.append(kwargs["processed_signal"].detach().clone())
        return original_frontend(**kwargs)

    monkeypatch.setattr(encoder.diarization_model, "frontend_encoder", tracked_frontend)
    with torch.no_grad():
        encoder._run_diarization(mels, lengths)

    expected, _, _ = normalize_batch(mels, lengths, normalize_type="per_feature")
    torch.testing.assert_close(observed[0], expected)


@pytest.mark.unit
def test_mixed_missing_rttm_rows_use_sortformer_predictions(monkeypatch):
    encoder = build_toy_pe_encoder().eval()
    mels = torch.randn(3, _MEL_FEATURES, 80)
    lengths = torch.tensor([80, 72, 64])
    diarization = torch.rand(3, 10, _N_SPK)
    asr_states = torch.randn(3, _ASR_D_MODEL, 10)
    asr_lengths = torch.tensor([10, 9, 8])
    monkeypatch.setattr(encoder, "_run_diarization", lambda *_: diarization)
    monkeypatch.setattr(encoder, "_run_asr", lambda *_: (asr_states, asr_lengths))

    targets = torch.zeros(3, 10, _N_SPK)
    targets[0, :, 0] = 1.0
    targets[1] = encoder.missing_rttm_target
    targets[2, :, 2] = 1.0
    expected = encoder._fuse_diar_and_asr(
        asr_states,
        targets,
        diarization_preds=diarization,
        use_diarization=torch.tensor([False, True, False]),
    )
    actual, actual_lengths = encoder(mels, lengths, spk_targets=targets)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_lengths, asr_lengths)


@pytest.mark.unit
@pytest.mark.parametrize(("training", "world_size"), [(True, 1), (False, 2)])
def test_all_rttm_rows_still_run_sortformer_in_collective_safe_paths(monkeypatch, training, world_size):
    encoder = build_toy_pe_encoder().train(training)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: world_size > 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
    mels = torch.randn(2, _MEL_FEATURES, 80)
    lengths = torch.tensor([80, 64])
    diarization = torch.rand(2, 10, _N_SPK)
    asr_states = torch.randn(2, _ASR_D_MODEL, 10)
    asr_lengths = torch.tensor([10, 8])
    diarization_calls = 0

    def run_diarization(*_):
        nonlocal diarization_calls
        diarization_calls += 1
        return diarization

    monkeypatch.setattr(encoder, "_run_diarization", run_diarization)
    monkeypatch.setattr(encoder, "_run_asr", lambda *_: (asr_states, asr_lengths))

    targets = torch.zeros(2, 10, _N_SPK)
    targets[0, :, 0] = 1.0
    targets[1, :, 1] = 1.0
    expected = encoder._fuse_diar_and_asr(asr_states, targets)
    actual, actual_lengths = encoder(mels, lengths, spk_targets=targets)

    assert diarization_calls == 1
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_lengths, asr_lengths)


@pytest.mark.unit
def test_all_rttm_rows_can_skip_sortformer_in_single_process_eval(monkeypatch):
    encoder = build_toy_pe_encoder().eval()
    mels = torch.randn(2, _MEL_FEATURES, 80)
    lengths = torch.tensor([80, 64])
    asr_states = torch.randn(2, _ASR_D_MODEL, 10)
    asr_lengths = torch.tensor([10, 8])
    monkeypatch.setattr(
        encoder,
        "_run_diarization",
        lambda *_: (_ for _ in ()).throw(AssertionError("single-process eval unexpectedly ran Sortformer")),
    )
    monkeypatch.setattr(encoder, "_run_asr", lambda *_: (asr_states, asr_lengths))

    targets = torch.zeros(2, 10, _N_SPK)
    targets[0, :, 0] = 1.0
    targets[1, :, 1] = 1.0
    expected = encoder._fuse_diar_and_asr(asr_states, targets)
    actual, actual_lengths = encoder(mels, lengths, spk_targets=targets)

    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_lengths, asr_lengths)


@pytest.mark.unit
def test_all_rttm_rows_still_run_sortformer_in_packed_training_path(monkeypatch):
    encoder = build_toy_pe_encoder().train()
    lengths = torch.tensor([80, 64])
    packed_input = pack_encoder_output(torch.randn(2, 80, _MEL_FEATURES), lengths)
    output_lengths = torch.tensor([10, 8])
    asr_states = pack_encoder_output(torch.randn(2, 10, _ASR_D_MODEL), output_lengths)
    diarization = pack_encoder_output(torch.rand(2, 10, _N_SPK), output_lengths)
    diarization_calls = 0

    def run_diarization(_):
        nonlocal diarization_calls
        diarization_calls += 1
        return diarization

    monkeypatch.setattr(encoder, "_run_diarization_packed", run_diarization)
    monkeypatch.setattr(encoder, "_run_asr_packed", lambda _: asr_states)

    targets = torch.zeros(2, 10, _N_SPK)
    targets[0, :, 0] = 1.0
    targets[1, :, 1] = 1.0
    expected = encoder._fuse_diar_and_asr_packed(asr_states, targets)
    actual = encoder.forward_sequence_packed(packed_input, spk_targets=targets)

    assert diarization_calls == 1
    torch.testing.assert_close(actual.data, expected.data)
    assert torch.equal(actual.lengths, expected.lengths)


@pytest.mark.unit
def test_speaker_threshold_and_kernel_scale_are_preserved():
    encoder = build_toy_pe_encoder(speaker_activity_threshold=0.5, spk_kernel_scale=0.25).eval()
    asr_states = torch.randn(1, _ASR_D_MODEL, 3)
    targets = torch.full((1, 3, _N_SPK), 0.5)
    targets[:, 1, 0] = 0.5001
    fused = encoder._fuse_diar_and_asr(asr_states, targets)

    normalized = encoder.asr_norm(asr_states.transpose(1, 2))
    binary = (targets > 0.5).to(normalized.dtype)
    infusion = encoder.diar_norm(binary) @ encoder.diar_kernel
    expected = (normalized + 0.25 * infusion).transpose(1, 2)
    torch.testing.assert_close(fused, expected)


@pytest.mark.unit
def test_high_resolution_diarization_is_pooled_to_asr_grid():
    diarization_config = toy_diarization_model_cfg()
    diarization_config.high_resolution = True
    diarization_config.output_subsampling_factor = 1
    encoder = build_toy_pe_encoder(diarization_model_cfg=diarization_config).eval()
    lengths = torch.tensor([80, 53])
    mels = torch.randn(2, _MEL_FEATURES, 80)
    packed_input = pack_encoder_output(mels.transpose(1, 2), lengths)

    with torch.no_grad():
        padded = encoder._run_diarization(mels, lengths)
        packed = encoder._run_diarization_packed(packed_input)
        asr = encoder._run_asr_packed(packed_input)

    assert torch.equal(packed.lengths, asr.lengths)
    assert padded.shape[1] == asr.max_seqlen
    restored = unpack_encoder_output(packed, total_length=padded.shape[1])
    valid = torch.arange(padded.shape[1])[None, :] < packed.lengths[:, None]
    torch.testing.assert_close(restored[valid], padded[valid], rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_high_resolution_fusion_pools_offline_but_not_aligned_online_predictions():
    diarization_config = toy_diarization_model_cfg()
    diarization_config.high_resolution = True
    diarization_config.output_subsampling_factor = 1
    encoder = build_toy_pe_encoder(diarization_model_cfg=diarization_config).eval()
    asr_states = torch.randn(1, _ASR_D_MODEL, 3)
    fine_predictions = torch.sigmoid(torch.arange(24 * _N_SPK).reshape(1, 24, _N_SPK).float() / 17.0 - 2.0)
    aligned_predictions = encoder.diarization_model.sortformer_modules.downsample_preds(
        fine_predictions, _SUBSAMPLING_FACTOR
    )

    offline_fused = encoder._fuse_diar_and_asr(asr_states, fine_predictions)
    online_fused = encoder._fuse_diar_and_asr(asr_states, aligned_predictions)

    assert aligned_predictions.shape[1] == asr_states.shape[-1]
    torch.testing.assert_close(offline_fused, online_fused)


@pytest.mark.unit
def test_packed_diarization_supports_optional_post_encoder():
    diarization_config = toy_packed_diarization_model_cfg()
    diarization_config.transformer_encoder = None
    encoder = build_toy_pe_encoder(diarization_model_cfg=diarization_config).eval()
    lengths = torch.tensor([80, 53])
    mels = torch.randn(2, _MEL_FEATURES, 80)

    with torch.no_grad():
        predictions = encoder._run_diarization_packed(pack_encoder_output(mels.transpose(1, 2), lengths))

    assert predictions.lengths.tolist() == [10, 7]
    assert torch.isfinite(predictions.data).all()


@pytest.mark.unit
def test_packed_fallback_matches_padded_forward_for_dense_and_packed_inputs():
    torch.manual_seed(0)
    encoder = build_toy_pe_encoder().eval()
    lengths = torch.tensor([80, 53])
    mels = torch.randn(2, _MEL_FEATURES, 80)
    mels[1, :, 53:] = 0.0
    targets = torch.zeros(2, 10, _N_SPK)
    targets[0, :, 0] = 1.0
    targets[1] = -1.0

    with torch.no_grad():
        padded, output_lengths = encoder(mels, lengths, spk_targets=targets)
        packed_from_dense = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)
        packed_input = pack_encoder_output(mels.transpose(1, 2), lengths)
        packed_from_packed = encoder.forward_sequence_packed(packed_input, spk_targets=targets)

    restored = unpack_encoder_output(packed_from_dense, total_length=padded.shape[-1])
    valid = torch.arange(padded.shape[-1])[None, :] < output_lengths[:, None]
    torch.testing.assert_close(restored[valid], padded.transpose(1, 2)[valid], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(packed_from_packed.data, packed_from_dense.data, rtol=1e-5, atol=1e-6)
    assert torch.equal(packed_from_packed.lengths, packed_from_dense.lengths)


@pytest.mark.unit
def test_native_packed_path_normalizes_diar_and_asr_independently_without_unpack(
    monkeypatch,
):
    encoder = build_toy_packed_pe_encoder().eval()
    lengths = torch.tensor([80, 53])
    mels = torch.randn(2, _MEL_FEATURES, 80)
    packed = pack_encoder_output(mels.transpose(1, 2), lengths)

    calls = []
    original_forward = encoder._forward_packed_branch

    def tracked_forward(branch, features, chunk_size_seconds):
        calls.append((branch, features.data.detach().clone(), chunk_size_seconds))
        return original_forward(branch, features, chunk_size_seconds)

    monkeypatch.setattr(encoder, "_forward_packed_branch", tracked_forward)
    monkeypatch.setattr(
        pee_module,
        "unpack_encoder_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native packed path unpacked")),
    )
    with torch.no_grad():
        output = encoder.forward_sequence_packed(packed)

    assert [branch for branch, _, _ in calls] == [
        encoder.diarization_model.encoder,
        encoder.asr_encoder,
    ]
    expected_diar = normalize_packed_batch(packed, "per_feature")
    torch.testing.assert_close(calls[0][1], expected_diar.data)
    torch.testing.assert_close(calls[1][1], expected_diar.data)
    assert torch.equal(output.lengths, torch.tensor([10, 7]))
    assert torch.isfinite(output.data).all()


@pytest.mark.unit
def test_native_packed_output_matches_padded_two_branch_path():
    torch.manual_seed(0)
    encoder = build_toy_packed_pe_encoder().eval()
    lengths = torch.tensor([80, 53])
    mels = torch.randn(2, _MEL_FEATURES, 80)
    mels[1, :, 53:] = 0.0
    with torch.no_grad():
        padded, padded_lengths = encoder(mels, lengths)
        packed = encoder.forward_sequence_packed(mels, lengths)
    restored = unpack_encoder_output(packed, total_length=padded.shape[-1])
    valid = torch.arange(padded.shape[-1])[None, :] < padded_lengths[:, None]
    torch.testing.assert_close(restored[valid], padded.transpose(1, 2)[valid], rtol=1e-4, atol=1e-5)


@pytest.mark.unit
def test_native_packed_branches_share_chunk_size_after_feature_stacking(monkeypatch):
    encoder = build_toy_packed_pe_encoder(
        frame_shift_seconds=0.01,
        chunk_size_seconds=0.16,
    ).eval()
    packed = pack_encoder_output(torch.randn(2, 80, _MEL_FEATURES), torch.tensor([80, 65]))
    diar_calls = []
    asr_calls = []
    diar_forward = encoder.diarization_model.encoder.forward_sequence_packed
    asr_forward = encoder.asr_encoder.forward_sequence_packed

    def tracked_diar(audio_signal, length, *args, **kwargs):
        diar_calls.append(
            (
                audio_signal.lengths.detach().cpu().tolist(),
                kwargs.get("bypass_pre_encode", False),
            )
        )
        return diar_forward(audio_signal, length, *args, **kwargs)

    def tracked_asr(audio_signal, length, *args, **kwargs):
        asr_calls.append(
            (
                audio_signal.lengths.detach().cpu().tolist(),
                kwargs.get("bypass_pre_encode", False),
            )
        )
        return asr_forward(audio_signal, length, *args, **kwargs)

    monkeypatch.setattr(encoder.diarization_model.encoder, "forward_sequence_packed", tracked_diar)
    monkeypatch.setattr(encoder.asr_encoder, "forward_sequence_packed", tracked_asr)
    with torch.no_grad():
        output = encoder.forward_sequence_packed(packed)

    expected_chunks = [([2, 2, 2, 2, 2, 2, 2, 2, 2, 1], True)]
    assert diar_calls == expected_chunks
    assert asr_calls == expected_chunks
    assert output.lengths.tolist() == [10, 9]


@pytest.mark.unit
def test_packed_fallback_rejects_online_scope():
    encoder = build_toy_pe_encoder().eval()
    with encoder.online_inference(), pytest.raises(RuntimeError, match="offline API"):
        encoder.forward_sequence_packed(torch.randn(1, _MEL_FEATURES, 32), torch.tensor([32]))


@pytest.mark.unit
def test_activation_checkpointing_wraps_trainable_asr_layers_and_packed_backward():
    encoder = build_toy_packed_pe_encoder().train()
    encoder.set_activation_checkpointing(True)
    encoder.set_activation_checkpointing(True)

    assert getattr(encoder.asr_encoder.pre_encode, "_checkpoint_wrapped_module", None) is not None
    assert all(getattr(layer, "_checkpoint_wrapped_module", None) is not None for layer in encoder.asr_encoder.layers)
    assert all(
        getattr(layer, "_checkpoint_wrapped_module", None) is None
        for layer in encoder.diarization_model.encoder.layers
    )

    mels = torch.randn(1, 64, _MEL_FEATURES, requires_grad=True)
    packed = pack_encoder_output(mels, torch.tensor([64]))
    output = encoder._run_asr_packed(packed)
    output.data.square().mean().backward()
    assert mels.grad is not None
    assert torch.isfinite(mels.grad).all()


def dispatch_stub(enabled):
    encoder = _PEE.__new__(_PEE)
    nn.Module.__init__(encoder)
    encoder.online_inference_length = 10
    encoder.online_inference_enabled = enabled
    encoder._forward = lambda **kwargs: ("offline", None)
    encoder._forward_online = lambda **kwargs: ("online", None)
    return encoder


@pytest.mark.unit
def test_online_inference_context_controls_dispatch_and_restores_state():
    encoder = dispatch_stub(False)
    audio = torch.zeros(1, 8, 20)
    length = torch.tensor([20])
    assert encoder(audio, length)[0] == "offline"
    with encoder.online_inference():
        assert encoder(audio, length)[0] == "online"
    assert encoder(audio, length)[0] == "offline"


@pytest.mark.unit
def test_online_inference_runs_two_real_branches_with_conformer_io():
    encoder = build_toy_pe_encoder(
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    encoder._suppress_online_pbar = True
    mels = torch.randn(1, _MEL_FEATURES, 160)
    lengths = torch.tensor([160])

    with torch.no_grad(), encoder.online_inference():
        output, output_lengths = encoder(mels, lengths)

    assert output.shape == (1, _ASR_D_MODEL, int(output_lengths[0]))
    assert torch.isfinite(output).all()

    targets = torch.zeros(1, output.shape[-1], _N_SPK)
    targets[:, :, 0] = 1.0
    with torch.no_grad(), encoder.online_inference():
        external_output, external_lengths = encoder(mels, lengths, spk_targets=targets)
    assert external_output.shape == output.shape
    assert torch.equal(external_lengths, output_lengths)


@pytest.mark.unit
def test_online_inference_passes_normalized_time_major_mels_to_sortformer(monkeypatch):
    """Sortformer receives normalized time-major features before its pre-encoder."""
    encoder = build_toy_pe_encoder(
        diarization_model_cfg=toy_packed_diarization_model_cfg(),
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    encoder._suppress_online_pbar = True
    original = encoder.diarization_model.forward_streaming_step
    observed_signals = []

    def checked_forward_streaming_step(**kwargs):
        processed_signal = kwargs["processed_signal"]
        observed_signals.append(processed_signal.detach().clone())
        assert processed_signal.shape[2] == _MEL_FEATURES
        return original(**kwargs)

    monkeypatch.setattr(
        encoder.diarization_model,
        "forward_streaming_step",
        checked_forward_streaming_step,
    )
    mels = 4.0 * torch.randn(1, _MEL_FEATURES, 160) + 17.0
    lengths = torch.tensor([160])

    with torch.no_grad(), encoder.online_inference():
        encoder(mels, lengths)

    expected, _, _ = normalize_batch(mels, lengths, normalize_type="per_feature")
    assert observed_signals
    expected_first_chunk = expected[:, :, : observed_signals[0].shape[1]].transpose(1, 2)
    torch.testing.assert_close(observed_signals[0], expected_first_chunk)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_online_inference_uses_activation_device_after_nested_parent_move():
    """A nested Lightning Sortformer must not retain its construction device."""
    parent = nn.Module()
    parent.encoder = build_toy_pe_encoder(
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    parent.encoder._suppress_online_pbar = True
    parent.to("cuda")

    mels = torch.randn(1, _MEL_FEATURES, 40, device="cuda")
    lengths = torch.tensor([40], device="cuda")
    with torch.no_grad(), parent.encoder.online_inference():
        output, output_lengths = parent.encoder(mels, lengths)

    assert output.device.type == "cuda"
    assert output_lengths.device.type == "cuda"
    assert torch.isfinite(output).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asr_encoder_type", "asr_encoder_cfg"),
    [
        ("fastconformer", toy_asr_encoder_cfg),
        ("transformer", toy_transformer_asr_encoder_cfg),
    ],
)
def test_online_inference_matches_independent_valid_prefixes_for_unequal_audio(asr_encoder_type, asr_encoder_cfg):
    encoder = build_toy_pe_encoder(
        asr_encoder_type=asr_encoder_type,
        asr_encoder_cfg=asr_encoder_cfg(),
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    encoder._suppress_online_pbar = True
    mels = torch.randn(2, _MEL_FEATURES, 321)
    lengths = torch.tensor([321, 173])

    with torch.no_grad(), encoder.online_inference():
        batched_output, batched_lengths = encoder(mels, lengths)
        first_output, first_length = encoder(mels[:1, :, :321], lengths[:1])
        second_output, second_length = encoder(mels[1:2, :, :173], lengths[1:])

    expected_lengths = torch.cat([first_length, second_length])
    assert torch.equal(batched_lengths, expected_lengths)
    assert batched_output.shape == (2, _ASR_D_MODEL, int(expected_lengths.max()))
    torch.testing.assert_close(batched_output[0, :, : first_length[0]], first_output[0])
    torch.testing.assert_close(batched_output[1, :, : second_length[0]], second_output[0])


@pytest.mark.unit
def test_transformer_online_inference_preserves_partial_feature_stack():
    encoder = build_toy_pe_encoder(
        asr_encoder_type="transformer",
        asr_encoder_cfg=toy_transformer_asr_encoder_cfg(),
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    encoder._suppress_online_pbar = True
    mels = torch.randn(1, _MEL_FEATURES, 161)
    lengths = torch.tensor([161])
    targets = torch.zeros(1, 21, _N_SPK)
    targets[:, :, 0] = 1.0

    with torch.no_grad():
        _, expected_lengths = encoder._run_asr(mels, lengths)
        with encoder.online_inference():
            output, output_lengths = encoder(mels, lengths, spk_targets=targets)

    assert torch.equal(output_lengths, expected_lengths)
    assert output.shape == (1, _ASR_D_MODEL, int(expected_lengths[0]))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asr_encoder_type", "asr_encoder_cfg", "write_type_flag"),
    [
        ("fastconformer", toy_asr_encoder_cfg, False),
        ("transformer", toy_transformer_asr_encoder_cfg, True),
    ],
)
def test_strict_two_branch_bundle_loading(tmp_path, asr_encoder_type, asr_encoder_cfg, write_type_flag):
    source = build_toy_pe_encoder(
        asr_encoder_type=asr_encoder_type,
        asr_encoder_cfg=asr_encoder_cfg(),
    ).eval()
    state = {f"encoder.{key}": value for key, value in source.state_dict().items()}
    archive = tmp_path / "two_branch.nemo"
    config_overrides = {"asr_encoder_cfg": asr_encoder_cfg()}
    if write_type_flag:
        config_overrides["asr_encoder_type"] = asr_encoder_type
    write_bundle(archive, bundle_config(**config_overrides), state)

    restored = ParallelExpertEncoderPT.load_from_nemo(str(archive), strict=True).eval()
    assert restored.asr_encoder_type == asr_encoder_type
    assert set(restored.state_dict()) == set(source.state_dict())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.unit
def test_inline_config_reconstructs_architecture_without_standalone_weights():
    config = bundle_config(asr_chunk_size_seconds=30.0, diar_chunk_size_seconds=45.0)

    restored = ParallelExpertEncoderPT.from_inline_config(config).eval()

    assert restored.asr_chunk_size_seconds == 30.0
    assert restored.diar_chunk_size_seconds == 45.0
    assert restored.diar_normalize_type == "per_feature"
    assert restored._bundle_config.diar_normalize_type == "per_feature"
    assert restored.speaker_feature_mode == "thresholded"
    assert restored.speaker_activity_threshold == 0.5
    assert restored._bundle_config.speaker_feature_config_version == 1
    assert restored._bundle_config.speaker_feature_mode == "thresholded"
    assert restored._bundle_config.speaker_activity_threshold == 0.5
    assert restored._bundle_config.target == config.target


@pytest.mark.unit
def test_legacy_canonical_bundle_requires_explicit_speaker_contract(tmp_path):
    config = bundle_config(target="nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT")
    with pytest.raises(ValueError, match="cannot be inferred safely"):
        ParallelExpertEncoderPT.from_inline_config(config)

    source = build_toy_pe_encoder().eval()
    state = {f"encoder.{key}": value for key, value in source.state_dict().items()}
    archive = tmp_path / "legacy-canonical.nemo"
    write_bundle(archive, config, state)
    overrides = {
        "speaker_feature_config_version": 1,
        "speaker_feature_mode": "continuous",
        "speaker_activity_threshold": None,
        "sync_max_audio_length": False,
    }
    restored = ParallelExpertEncoderPT.load_from_nemo(
        str(archive),
        strict=True,
        config_overrides=overrides,
    ).eval()

    assert restored.speaker_feature_mode == "continuous"
    assert restored.speaker_activity_threshold is None
    assert restored._bundle_config.speaker_feature_config_version == 1
    assert restored._bundle_config.speaker_feature_mode == "continuous"
    assert restored._bundle_config.speaker_activity_threshold is None
    assert restored._bundle_config.sync_max_audio_length is False


@pytest.mark.unit
def test_sync_max_audio_length_is_disabled_by_portable_config():
    config = bundle_config(sync_max_audio_length=False)
    config.asr_encoder_cfg.sync_max_audio_length = True
    config.diarization_model_cfg.encoder.sync_max_audio_length = True

    restored = ParallelExpertEncoderPT.from_inline_config(config).eval()

    assert restored.sync_max_audio_length is False
    assert restored.asr_encoder.sync_max_audio_length is False
    assert restored.diarization_model.encoder.sync_max_audio_length is False
    assert restored._bundle_config.sync_max_audio_length is False


@pytest.mark.unit
def test_explicit_portable_speaker_contract_is_independent_of_target_name():
    config = bundle_config(
        target="nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT",
        speaker_feature_config_version=1,
        speaker_feature_mode="thresholded",
        speaker_activity_threshold=0.5,
    )

    restored = ParallelExpertEncoderPT.from_inline_config(config).eval()

    assert restored.speaker_feature_mode == "thresholded"
    assert restored.speaker_activity_threshold == 0.5


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"speaker_feature_config_version": 1},
            "requires an explicit speaker_feature_mode",
        ),
        (
            {
                "speaker_feature_config_version": 1,
                "speaker_feature_mode": "continuous",
                "speaker_activity_threshold": 0.5,
            },
            "continuous.*speaker_activity_threshold=None",
        ),
        (
            {
                "speaker_feature_config_version": 1,
                "speaker_feature_mode": "thresholded",
                "speaker_activity_threshold": None,
            },
            "thresholded.*non-null",
        ),
        (
            {
                "speaker_feature_config_version": 2,
                "speaker_feature_mode": "thresholded",
                "speaker_activity_threshold": 0.5,
            },
            "Unsupported speaker_feature_config_version",
        ),
    ],
)
def test_invalid_speaker_feature_contract_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        ParallelExpertEncoderPT.from_inline_config(bundle_config(**overrides))


@pytest.mark.unit
def test_current_pee_loader_dispatches_two_branch_inline_schema():
    from nemo.collections.asr.modules.parallel_expert_encoder import (
        ParallelExpertEncoderPT as CurrentParallelExpertEncoderPT,
    )

    restored = CurrentParallelExpertEncoderPT.from_inline_config(bundle_config()).eval()

    assert isinstance(restored, ParallelExpertEncoder)
    assert restored.asr_encoder_type == "fastconformer"


@pytest.mark.unit
def test_current_pee_loader_rejects_ambiguous_inline_schema():
    from nemo.collections.asr.modules.parallel_expert_encoder import (
        ParallelExpertEncoderPT as CurrentParallelExpertEncoderPT,
    )

    config = OmegaConf.merge(
        bundle_config(),
        {
            "speech_expert_cfg": {"_target_": "example.Speech"},
            "speaker_expert_cfg": {"_target_": "example.Speaker"},
            "sound_expert_cfg": {"_target_": "example.Sound"},
            "sortformer_modules_cfg": {"_target_": "example.Sortformer"},
        },
    )
    with pytest.raises(ValueError, match="ambiguously contains both"):
        CurrentParallelExpertEncoderPT.from_inline_config(config)


@pytest.mark.unit
def test_exported_inline_config_round_trips_consolidated_weights():
    to_hf_path = Path(__file__).parents[3] / "examples" / "speechlm2" / "to_hf.py"
    spec = importlib.util.spec_from_file_location("to_hf_for_pee_roundtrip", to_hf_path)
    to_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(to_hf)

    source = ParallelExpertEncoderPT(bundle_config()).encoder.eval()
    source.asr_chunk_size_seconds = 30.0
    source.diar_chunk_size_seconds = 45.0
    root = SimpleNamespace(
        cfg={"pe_encoder_path": "/models/placeholderParallelExpertEncoder.nemo"},
        perception=SimpleNamespace(encoder=source),
    )

    exported = to_hf._hf_export_config(root, "bfloat16")
    restored = ParallelExpertEncoderPT.from_inline_config(exported["pe_encoder_config"]).eval()
    restored.load_state_dict(source.state_dict(), strict=True)

    assert set(restored.state_dict()) == set(source.state_dict())
    assert restored.asr_chunk_size_seconds == 30.0
    assert restored.diar_chunk_size_seconds == 45.0
    assert restored.diar_normalize_type == "per_feature"
    assert restored.speaker_feature_mode == "thresholded"
    assert restored.speaker_activity_threshold == 0.5
    assert exported["pe_encoder_config"]["speaker_feature_config_version"] == 1
    assert exported["pe_encoder_config"]["speaker_feature_mode"] == "thresholded"
    assert exported["pe_encoder_config"]["speaker_activity_threshold"] == 0.5
    assert exported["pe_encoder_config"]["sync_max_audio_length"] is False
    mels = torch.randn(1, _MEL_FEATURES, 161)
    lengths = torch.tensor([161])
    with torch.no_grad():
        expected, expected_lengths = source(mels, lengths)
        actual, actual_lengths = restored(mels, lengths)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_lengths, expected_lengths)


@pytest.mark.unit
def test_legacy_three_expert_bundle_is_rejected(tmp_path):
    archive = tmp_path / "legacy.nemo"
    legacy = OmegaConf.create(
        {
            "target": "nemo.collections.asr.modules.parallel_expert_encoder_two_branch.ParallelExpertEncoderPT",
            "speech_expert_cfg": {"_target_": "legacy.Speech"},
            "speaker_expert_cfg": {"_target_": "legacy.Speaker"},
            "sound_expert_cfg": {"_target_": "legacy.Sound"},
            "sortformer_modules_cfg": {"_target_": "legacy.Sortformer"},
        }
    )
    write_bundle(archive, legacy)
    with pytest.raises(ValueError, match="Legacy three-expert"):
        ParallelExpertEncoderPT.load_from_nemo(str(archive), strict=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            "nemo.collections.asr.modules.parallel_expert_encoder_two_branch.ParallelExpertEncoderPT",
            True,
        ),
        ("ParallelExpertEncoderPT", True),
        ("nemo.collections.asr.models.SomethingElse", False),
    ],
)
def test_is_pe_nemo_uses_target_and_two_branch_schema(tmp_path, target, expected):
    archive = tmp_path / "bundle.nemo"
    config = bundle_config()
    config.target = target
    write_bundle(archive, config)
    assert ParallelExpertEncoderPT.is_pe_nemo(str(archive)) is expected


@pytest.mark.unit
def test_training_loader_dispatches_two_branch_bundle_by_schema(tmp_path):
    from nemo.collections.speechlm2.parts.pretrained import _resolve_parallel_expert_encoder_class

    archive = tmp_path / "bundle.nemo"
    write_bundle(archive, bundle_config())

    assert _resolve_parallel_expert_encoder_class(str(archive)) is ParallelExpertEncoderPT


@pytest.mark.unit
def test_save_to_nemo_guard_rails(tmp_path):
    with pytest.raises(TypeError):
        ParallelExpertEncoderPT.save_to_nemo(
            nn.Linear(2, 2),
            str(tmp_path / "out.nemo"),
            template_bundle_path=str(tmp_path / "template.nemo"),
        )
    fake_encoder = _PEE.__new__(_PEE)
    with pytest.raises(FileNotFoundError):
        ParallelExpertEncoderPT.save_to_nemo(
            fake_encoder,
            str(tmp_path / "out.nemo"),
            template_bundle_path=str(tmp_path / "missing.nemo"),
        )
