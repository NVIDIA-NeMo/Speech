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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel


def _fake_model(*, global_step: int, every_steps: int = 100, detailed_every_steps: int = 500):
    return SimpleNamespace(
        cfg=DictConfig(
            {
                "moe_metrics": {
                    "enabled": True,
                    "mode": "detailed",
                    "every_steps": every_steps,
                    "detailed_every_steps": detailed_every_steps,
                    "top_k_experts": 5,
                }
            }
        ),
        global_step=global_step,
        llm=object(),
        _get_moe_dp_group=MagicMock(return_value="dp-group"),
        log_dict=MagicMock(),
    )


def test_moe_metrics_skip_collective_between_intervals(monkeypatch):
    import nemo_automodel.components.moe.load_balance_metrics as metrics

    collect = MagicMock()
    monkeypatch.setattr(metrics, "collect_expert_loads", collect)
    model = _fake_model(global_step=99)

    SALMAutomodel.maybe_log_moe_metrics(model)

    collect.assert_not_called()
    model._get_moe_dp_group.assert_not_called()
    model.log_dict.assert_not_called()


@pytest.mark.parametrize(
    ("global_step", "expected_metric"),
    [(100, "brief"), (500, "detailed")],
)
def test_moe_metrics_use_global_step_for_collection_and_detail_cadence(monkeypatch, global_step, expected_metric):
    import nemo_automodel.components.moe.load_balance_metrics as metrics

    collect = MagicMock(return_value={"layer": torch.tensor([1.0])})
    brief = MagicMock(return_value={"moe/mode": 0.0})
    detailed = MagicMock(return_value={"moe/mode": 1.0})
    monkeypatch.setattr(metrics, "collect_expert_loads", collect)
    monkeypatch.setattr(metrics, "compute_brief_metrics", brief)
    monkeypatch.setattr(metrics, "compute_detailed_metrics", detailed)
    model = _fake_model(global_step=global_step)

    SALMAutomodel.maybe_log_moe_metrics(model)

    collect.assert_called_once_with(model.llm, dp_group="dp-group")
    if expected_metric == "brief":
        brief.assert_called_once()
        detailed.assert_not_called()
    else:
        detailed.assert_called_once()
        brief.assert_not_called()
    model.log_dict.assert_called_once()


def test_moe_metrics_reject_nonpositive_interval():
    model = _fake_model(global_step=0, every_steps=0)

    with pytest.raises(ValueError, match="moe_metrics.every_steps must be positive"):
        SALMAutomodel.maybe_log_moe_metrics(model)


def test_moe_aux_loss_scale_is_independent_of_dp_size(monkeypatch):
    from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(128.0))
    model = SimpleNamespace(_get_moe_dp_group=MagicMock())

    SALMAutomodel._configure_moe_aux_loss_scaler(model)

    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(1.0)
    model._get_moe_dp_group.assert_not_called()
