# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Tests for init_from_checkpoint functionality in speechlm2.

Unit tests use simple nn.Module subclasses (no HF downloads, no CUDA).
Integration tests use real SALM / SALMAutomodel (require HF config download;
SALMAutomodel tests also require CUDA).
"""
import os
from unittest.mock import patch

import pytest
import torch
from omegaconf import DictConfig
from safetensors.torch import save_file

from nemo.collections.speechlm2.parts.pretrained import (
    _is_dcp_checkpoint,
    init_from_training_checkpoint,
    maybe_load_pretrained_models,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleModel(torch.nn.Module):
    """Tiny model used for fast, self-contained unit tests."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.norm = torch.nn.LayerNorm(4)


class ConfigurableModel(torch.nn.Module):
    """SimpleModel that carries a ``cfg`` attribute, like SALM/SALMAutomodel."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = DictConfig(cfg)
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.norm = torch.nn.LayerNorm(4)


def _save_ckpt(model, path):
    """Save model state_dict in Lightning checkpoint format."""
    torch.save({"state_dict": model.state_dict()}, path)


def _assert_state_dicts_equal(sd1, sd2):
    assert set(sd1.keys()) == set(sd2.keys()), f"Key mismatch: {set(sd1.keys()) ^ set(sd2.keys())}"
    for key in sd1:
        assert torch.equal(sd1[key].cpu(), sd2[key].cpu()), f"Tensor mismatch at key: {key}"


def _assert_state_dicts_not_equal(sd1, sd2):
    """Assert at least one tensor differs (sanity-check before loading)."""
    assert set(sd1.keys()) == set(sd2.keys())
    any_diff = any(not torch.equal(sd1[k].cpu(), sd2[k].cpu()) for k in sd1)
    assert any_diff, "Expected state dicts to differ before checkpoint loading"


# ---------------------------------------------------------------------------
# _is_dcp_checkpoint
# ---------------------------------------------------------------------------


class TestIsDcpCheckpoint:
    def test_with_metadata(self, tmp_path):
        ckpt_dir = tmp_path / "step=100.ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / ".metadata").touch()
        assert _is_dcp_checkpoint(str(ckpt_dir))

    def test_without_metadata(self, tmp_path):
        ckpt_dir = tmp_path / "step=100.ckpt"
        ckpt_dir.mkdir()
        assert not _is_dcp_checkpoint(str(ckpt_dir))

    def test_regular_file(self, tmp_path):
        ckpt_file = tmp_path / "step=100.ckpt"
        ckpt_file.touch()
        assert not _is_dcp_checkpoint(str(ckpt_file))

    def test_nonexistent_path(self, tmp_path):
        assert not _is_dcp_checkpoint(str(tmp_path / "nonexistent"))

    def test_hf_dir_without_metadata(self, tmp_path):
        """HF directory (model.safetensors) should NOT be detected as DCP."""
        hf_dir = tmp_path / "hf_model"
        hf_dir.mkdir()
        (hf_dir / "model.safetensors").touch()
        assert not _is_dcp_checkpoint(str(hf_dir))


# ---------------------------------------------------------------------------
# init_from_training_checkpoint — non-DCP paths
# ---------------------------------------------------------------------------


class TestInitFromTrainingCheckpoint:
    def test_none_is_noop(self):
        model = SimpleModel()
        original = model.linear.weight.clone()
        init_from_training_checkpoint(model, None)
        assert torch.equal(model.linear.weight, original)

    def test_single_file_ckpt(self, tmp_path):
        source = SimpleModel()
        torch.nn.init.ones_(source.linear.weight)

        ckpt_path = str(tmp_path / "model.ckpt")
        _save_ckpt(source, ckpt_path)

        target = SimpleModel()  # different random init
        _assert_state_dicts_not_equal(source.state_dict(), target.state_dict())

        init_from_training_checkpoint(target, ckpt_path)
        _assert_state_dicts_equal(target.state_dict(), source.state_dict())

    def test_hf_directory(self, tmp_path):
        source = SimpleModel()
        torch.nn.init.ones_(source.linear.weight)

        hf_dir = tmp_path / "hf_model"
        hf_dir.mkdir()
        save_file(source.state_dict(), str(hf_dir / "model.safetensors"))

        target = SimpleModel()
        _assert_state_dicts_not_equal(source.state_dict(), target.state_dict())

        init_from_training_checkpoint(target, str(hf_dir))
        _assert_state_dicts_equal(target.state_dict(), source.state_dict())


# ---------------------------------------------------------------------------
# init_from_training_checkpoint — DCP path (mocked)
# ---------------------------------------------------------------------------


class TestInitFromTrainingCheckpointDCP:
    def test_dcp_calls_distributed_load(self, tmp_path):
        """Verify DCP checkpoint triggers torch.distributed.checkpoint.load."""
        ckpt_dir = tmp_path / "step=100.ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / ".metadata").touch()

        model = SimpleModel()

        with patch("nemo.collections.speechlm2.parts.pretrained.torch.distributed.checkpoint.load") as mock_load:
            init_from_training_checkpoint(model, str(ckpt_dir))

            mock_load.assert_called_once()
            args, kwargs = mock_load.call_args
            state_dict_wrapper = args[0]
            assert "state_dict" in state_dict_wrapper
            assert kwargs["checkpoint_id"] == str(ckpt_dir)

    def test_dcp_state_dict_has_model_keys(self, tmp_path):
        """The state dict passed to dcp.load should contain model parameter keys."""
        ckpt_dir = tmp_path / "step=100.ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / ".metadata").touch()

        model = SimpleModel()

        with patch("nemo.collections.speechlm2.parts.pretrained.torch.distributed.checkpoint.load") as mock_load:
            init_from_training_checkpoint(model, str(ckpt_dir))

            state_dict_wrapper = mock_load.call_args[0][0]
            model_sd = state_dict_wrapper["state_dict"]
            assert "linear.weight" in model_sd
            assert "norm.weight" in model_sd


# ---------------------------------------------------------------------------
# maybe_load_pretrained_models — init_from_checkpoint config key
# ---------------------------------------------------------------------------


class TestMaybeLoadPretrainedModels:
    def test_init_from_checkpoint_loads_weights(self, tmp_path):
        source = ConfigurableModel(cfg={})
        torch.nn.init.ones_(source.linear.weight)
        ckpt_path = str(tmp_path / "model.ckpt")
        _save_ckpt(source, ckpt_path)

        target = ConfigurableModel(cfg={"init_from_checkpoint": ckpt_path})
        _assert_state_dicts_not_equal(source.state_dict(), target.state_dict())

        maybe_load_pretrained_models(target)
        _assert_state_dicts_equal(target.state_dict(), source.state_dict())

    def test_init_from_checkpoint_null_is_noop(self):
        model = ConfigurableModel(cfg={"init_from_checkpoint": None})
        original = model.linear.weight.clone()
        maybe_load_pretrained_models(model)
        assert torch.equal(model.linear.weight, original)

    def test_init_from_checkpoint_key_missing_is_noop(self):
        model = ConfigurableModel(cfg={})
        original = model.linear.weight.clone()
        maybe_load_pretrained_models(model)
        assert torch.equal(model.linear.weight, original)

    def test_pretrained_s2s_model_still_works(self, tmp_path):
        """Backward compat: pretrained_s2s_model should still be recognized."""
        source = ConfigurableModel(cfg={})
        torch.nn.init.ones_(source.linear.weight)
        ckpt_path = str(tmp_path / "model.ckpt")
        _save_ckpt(source, ckpt_path)

        target = ConfigurableModel(cfg={"pretrained_s2s_model": ckpt_path})
        maybe_load_pretrained_models(target)
        _assert_state_dicts_equal(target.state_dict(), source.state_dict())


# ---------------------------------------------------------------------------
# SALM integration test
# ---------------------------------------------------------------------------

AUDIO_LOCATOR_TAG = "<|audioplaceholder|>"
SALM_PERCEPTION_CFG = {
    "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
    "output_dim": 2048,
    "encoder": {
        "_target_": "nemo.collections.asr.modules.ConformerEncoder",
        "att_context_size": [-1, -1],
        "causal_downsampling": False,
        "conv_context_size": None,
        "conv_kernel_size": 9,
        "conv_norm_type": "batch_norm",
        "d_model": 1024,
        "dropout": 0.1,
        "dropout_att": 0.1,
        "dropout_emb": 0.0,
        "dropout_pre_encoder": 0.1,
        "feat_in": 128,
        "feat_out": -1,
        "ff_expansion_factor": 4,
        "n_heads": 8,
        "n_layers": 2,
        "pos_emb_max_len": 5000,
        "self_attention_model": "rel_pos",
        "subsampling": "dw_striding",
        "subsampling_conv_channels": 256,
        "subsampling_factor": 8,
    },
    "modality_adapter": {
        "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
        "d_model": 1024,
    },
    "preprocessor": {
        "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
        "dither": 1e-05,
        "features": 128,
        "frame_splicing": 1,
        "log": True,
        "n_fft": 512,
        "normalize": "per_feature",
        "pad_to": 0,
        "pad_value": 0.0,
        "sample_rate": 16000,
        "window": "hann",
        "window_size": 0.025,
        "window_stride": 0.01,
    },
}


def _resolve_pretrained_salm():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        return {
            "pretrained_llm": "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1",
            "pretrained_asr": "/home/TestData/speechlm/pretrained_models/canary-1b-flash.nemo",
        }
    return {
        "pretrained_asr": "nvidia/canary-1b-flash",
        "pretrained_llm": "TinyLlama/TinyLlama_v1.1",
    }


def _resolve_pretrained_automodel():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        return {
            "pretrained_llm": "/home/TestData/speechlm/pretrained_models/Qwen--Qwen3-1.7B",
            "pretrained_asr": "/home/TestData/speechlm/pretrained_models/canary-1b-flash.nemo",
        }
    return {
        "pretrained_asr": "nvidia/canary-1b-flash",
        "pretrained_llm": "Qwen/Qwen3-1.7B",
    }


def _make_salm_cfg(**overrides):
    cfg = {
        **_resolve_pretrained_salm(),
        "pretrained_weights": False,
        "prompt_format": "llama2",
        "audio_locator_tag": AUDIO_LOCATOR_TAG,
        "perception": SALM_PERCEPTION_CFG,
        "optimizer": {"_target_": "torch.optim.AdamW"},
    }
    cfg.update(overrides)
    return cfg


def _make_automodel_cfg(**overrides):
    cfg = {
        **_resolve_pretrained_automodel(),
        "pretrained_weights": False,
        "prompt_format": "qwen",
        "audio_locator_tag": AUDIO_LOCATOR_TAG,
        "perception": SALM_PERCEPTION_CFG,
        "optimizer": {"_target_": "torch.optim.AdamW"},
        "torch_dtype": "bfloat16",
    }
    cfg.update(overrides)
    return cfg


def test_salm_init_from_checkpoint(tmp_path):
    from nemo.collections.speechlm2.models import SALM

    # Create source model and save checkpoint
    model1 = SALM(_make_salm_cfg())
    expected_sd = {k: v.clone().cpu() for k, v in model1.state_dict().items()}
    ckpt_path = str(tmp_path / "source.ckpt")
    _save_ckpt(model1, ckpt_path)
    del model1

    # Create target model — init_from_checkpoint overrides the random init
    model2 = SALM(_make_salm_cfg(init_from_checkpoint=ckpt_path))
    _assert_state_dicts_equal(model2.state_dict(), expected_sd)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SALMAutomodel requires CUDA")
def test_salm_automodel_init_from_checkpoint(tmp_path):
    from nemo.collections.speechlm2.models import SALMAutomodel

    # Create source model and save checkpoint
    model1 = SALMAutomodel(_make_automodel_cfg())
    model1.configure_model()
    expected_sd = {k: v.clone().cpu() for k, v in model1.state_dict().items()}
    ckpt_path = str(tmp_path / "source.ckpt")
    _save_ckpt(model1, ckpt_path)
    del model1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create target model — configure_model loads checkpoint via maybe_load_pretrained_models
    model2 = SALMAutomodel(_make_automodel_cfg(init_from_checkpoint=ckpt_path))
    model2.configure_model()
    _assert_state_dicts_equal(model2.state_dict(), expected_sd)
