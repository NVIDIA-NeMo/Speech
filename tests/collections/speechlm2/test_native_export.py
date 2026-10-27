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
"""Native-DCP canonical export precision and retained MTP metadata."""

import json
from types import SimpleNamespace

import torch
from examples.speechlm2.to_hf import (
    HfExportConfig,
    _hf_export_config,
    prepare_native_export_model_config,
    save_hf_checkpoint,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from omegaconf import DictConfig
from safetensors.torch import load_file
from transformers import PretrainedConfig


def model_fixture():
    model = torch.nn.Module()
    model.cfg = DictConfig(
        {
            "pretrained_llm": "unused",
            "retain_native_mtp": True,
            "preserve_parameter_dtypes": True,
            "automodel_freeze_config": {"freeze_parameters": ["model.layers.0.norm.weight"]},
            "perception_freeze_config": {"freeze_modules": [{"glob": "*"}]},
            "mtp": {
                "enabled": False,
                "num_nextn_predict_layers": 2,
                "use_repeated_layer": True,
                "hybrid_override_pattern": "*E",
            },
        }
    )
    model.llm = torch.nn.Module()
    model.llm.config = PretrainedConfig(num_nextn_predict_layers=1, mtp_hybrid_override_pattern="*E")
    model.llm.mtp = torch.nn.Linear(2, 2)
    model.llm.mtp_config = SimpleNamespace(num_layers=2, use_repeated_layer=True)
    model.llm.state_dict_adapter = NemotronV3StateDictAdapter(
        model.llm.config,
        MoEConfig(
            dim=2,
            inter_dim=4,
            moe_inter_dim=4,
            n_routed_experts=2,
            n_activated_experts=1,
            n_shared_experts=1,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            norm_topk_prob=False,
        ),
        BackendConfig(),
    )
    return model


def test_canonical_native_export_preserves_exact_fp32_values(tmp_path):
    model = model_fixture()
    state = {
        "llm.model.layers.0.mixer._fp32_params.A_log": torch.tensor([0.1234567], dtype=torch.float32),
        "llm.model.layers.1.mixer.gate.e_score_correction_bias": torch.tensor([0.1234567], dtype=torch.float32),
        "perception.proj.weight": torch.tensor([[0.25]], dtype=torch.float32),
    }
    cfg = HfExportConfig(
        class_path="unused",
        ckpt_path="unused",
        ckpt_config="unused",
        output_dir=str(tmp_path),
        checkpoint_format="nemo_rl",
    )
    save_hf_checkpoint(model, state, cfg)
    loaded = load_file(tmp_path / "model.safetensors")
    for name in list(state)[:2]:
        assert loaded[name].dtype == torch.float32
        torch.testing.assert_close(loaded[name], state[name].cpu(), rtol=0, atol=0)
    assert loaded["perception.proj.weight"].dtype == torch.bfloat16
    metadata = json.loads((tmp_path / "config.json").read_text())
    assert metadata["mtp"] == {
        "enabled": True,
        "num_nextn_predict_layers": 2,
        "use_repeated_layer": True,
        "hybrid_override_pattern": "*E",
    }
    assert "retain_native_mtp" not in metadata
    assert "preserve_parameter_dtypes" not in metadata
    assert "automodel_freeze_config" not in metadata
    assert "perception_freeze_config" not in metadata
    assert "automodel_freeze_config" in model.cfg
    assert model.cfg.mtp.enabled is False


def test_native_mtp_retention_without_head_exports_disabled(tmp_path):
    model = model_fixture()
    model.llm.mtp = None
    model.llm.mtp_config = None
    model.llm.config.num_nextn_predict_layers = 0
    model.llm.proj = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    cfg = HfExportConfig(
        class_path="unused",
        ckpt_path="unused",
        ckpt_config="unused",
        output_dir=str(tmp_path),
        checkpoint_format="nemo_rl",
    )
    save_hf_checkpoint(model, model.state_dict(), cfg)
    metadata = json.loads((tmp_path / "config.json").read_text())
    assert metadata["mtp"]["enabled"] is False
    assert "retain_native_mtp" not in metadata
    loaded = load_file(tmp_path / "model.safetensors")
    assert set(loaded) == {"llm.proj.weight"}
    torch.testing.assert_close(loaded["llm.proj.weight"], model.llm.proj.weight.detach().cpu(), rtol=0, atol=0)


def test_native_mtp_retention_accepts_null_mtp_block():
    model = model_fixture()
    model.cfg.mtp = None
    metadata = _hf_export_config(model, "bfloat16")
    assert metadata["mtp"]["enabled"] is True
    assert metadata["mtp"]["num_nextn_predict_layers"] == 2


def test_native_export_config_accepts_null_optional_blocks():
    prepared = prepare_native_export_model_config({"mtp": None, "automodel_backend": None})
    assert prepared["mtp"] == {"enabled": False}
    assert prepared["automodel_backend"] == {"enable_hf_state_dict_adapter": True}


def test_entrypoint_native_export_config_preserves_backend_and_mtp_options():
    original = {
        "automodel_backend": {"attn": "sdpa", "enable_hf_state_dict_adapter": False},
        "automodel_freeze_config": {"freeze_parameters": ["model.layers.0.norm.weight"]},
        "perception_freeze_config": {"freeze_modules": [{"glob": "*"}]},
        "mtp": {
            "enabled": True,
            "num_nextn_predict_layers": 2,
            "use_repeated_layer": True,
            "hybrid_override_pattern": "*E",
        },
    }
    prepared = prepare_native_export_model_config(original)
    assert prepared["automodel_backend"] == {"attn": "sdpa", "enable_hf_state_dict_adapter": True}
    assert prepared["mtp"]["enabled"] is False
    assert prepared["mtp"]["num_nextn_predict_layers"] == 2
    assert prepared["mtp"]["use_repeated_layer"] is True
    assert prepared["retain_native_mtp"] is True
    assert "preserve_parameter_dtypes" not in prepared
    assert "preserve_parameter_dtypes" not in original
    assert "automodel_freeze_config" not in prepared
    assert "perception_freeze_config" not in prepared
    assert "automodel_freeze_config" in original
    assert "perception_freeze_config" in original
    assert original["automodel_backend"]["enable_hf_state_dict_adapter"] is False
    assert original["mtp"]["enabled"] is True
