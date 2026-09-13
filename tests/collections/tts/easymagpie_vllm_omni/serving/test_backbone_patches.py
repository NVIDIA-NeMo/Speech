# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for narrow vLLM Nemotron-H compatibility patches."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from easymagpie_vllm_omni.backbone_patches import patch_shared_expert_activation  # noqa: E402
from vllm.model_executor.layers.activation import ReLUSquaredActivation  # noqa: E402


class NemotronHMoE:
    def __init__(self, activation):
        self.shared_experts = SimpleNamespace(act_fn=activation)


def _backbone(activation: str, current_activation):
    layer = SimpleNamespace(mixer=NemotronHMoE(current_activation))
    return SimpleNamespace(config=SimpleNamespace(mlp_hidden_act=activation), layers=[layer])


def _relu_squared_without_vllm_context():
    activation = ReLUSquaredActivation.__new__(ReLUSquaredActivation)
    torch.nn.Module.__init__(activation)
    return activation


def test_shared_expert_activation_is_read_from_config():
    backbone = _backbone("silu", _relu_squared_without_vllm_context())

    assert patch_shared_expert_activation(backbone) == 1
    torch.testing.assert_close(
        backbone.layers[0].mixer.shared_experts.act_fn(torch.tensor([-1.0, 0.0, 1.0])),
        torch.nn.functional.silu(torch.tensor([-1.0, 0.0, 1.0])),
    )
    assert patch_shared_expert_activation(backbone) == 0


def test_shared_expert_patch_rejects_unknown_upstream_implementation():
    backbone = _backbone("silu", torch.nn.Identity())

    with pytest.raises(RuntimeError, match="implementation changed"):
        patch_shared_expert_activation(backbone)
