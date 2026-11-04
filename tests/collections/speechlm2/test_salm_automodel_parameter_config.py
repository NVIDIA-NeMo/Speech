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
"""Perception storage precision under an explicit FreezeConfig."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

import nemo.collections.speechlm2.models.salm_automodel as salm_module

FreezeConfig = pytest.importorskip("nemo_automodel.components.utils.model_utils").FreezeConfig
if "unfreeze_parameters" not in FreezeConfig.__dataclass_fields__:
    pytest.skip("requires Automodel FreezeConfig exact parameter selectors", allow_module_level=True)

VALUE = 0.1234567


class _Perception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        self.encoder = torch.nn.Linear(2, 2, bias=False)
        self.register_buffer("window", torch.ones(2))
        with torch.no_grad():
            self.proj.weight.fill_(VALUE)
            self.encoder.weight.fill_(VALUE)

    def set_activation_checkpointing(self, _enabled):
        return None


class _LLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 2})()
        self.mtp = None


def _configure(monkeypatch, freeze_config, compute_dtype=None):
    model = salm_module.SALMAutomodel.__new__(salm_module.SALMAutomodel)
    torch.nn.Module.__init__(model)
    model.cfg = DictConfig(
        {
            "pretrained_llm": "unused",
            "pretrained_asr": "unused",
            "torch_dtype": "float32" if freeze_config is not None else "bfloat16",
            "perception_freeze_config": freeze_config,
        }
    )
    model._trainer = None
    model._use_fsdp = False
    model._use_tp = False
    model.setup_moe_options = lambda: None
    monkeypatch.setattr(salm_module, "load_pretrained_automodel_llm", lambda *_args, **_kwargs: _LLM())
    monkeypatch.setattr(
        salm_module, "setup_speech_encoder", lambda model, **_kwargs: setattr(model, "perception", _Perception())
    )
    monkeypatch.setattr(salm_module, "update_perception_output_dim", lambda _model: None)
    monkeypatch.setattr(salm_module, "maybe_load_pretrained_models", lambda _model: None)
    salm_module.SALMAutomodel.configure_model(
        model, perception_mixed_precision=SimpleNamespace(param_dtype=compute_dtype)
    )
    return model.perception


@pytest.mark.parametrize("compute_dtype", [None, torch.bfloat16])
def test_selected_fp32_perception_parameters_keep_exact_values(monkeypatch, compute_dtype):
    perception = _configure(
        monkeypatch,
        {"freeze_modules": [{"glob": "*"}], "unfreeze_parameters": ["proj.weight"]},
        compute_dtype,
    )
    expected = torch.full((2, 2), VALUE)

    assert perception.proj.weight.requires_grad
    assert torch.equal(perception.proj.weight, expected)
    assert not perception.encoder.weight.requires_grad
    if compute_dtype == torch.bfloat16:
        assert perception.encoder.weight.dtype == torch.bfloat16
    else:
        assert torch.equal(perception.encoder.weight, expected)
    assert perception.window.dtype == (compute_dtype or torch.float32)


def test_perception_storage_is_unchanged_without_freeze_config_or_mesh(monkeypatch):
    perception = _configure(monkeypatch, None)

    assert all(t.dtype == torch.float32 for t in (*perception.parameters(), *perception.buffers()))
