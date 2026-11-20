# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import pytest
import torch.nn
from omegaconf import DictConfig

import nemo.core.optim.lr_scheduler
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, freeze_and_subset


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.conv = torch.nn.Conv1d(1, 1, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.conv(x)
        return x


def test_freezing_params():
    model = DummyModel().train()
    assert model.linear.weight.requires_grad
    assert model.linear.bias.requires_grad
    assert model.conv.weight.requires_grad
    assert model.conv.bias.requires_grad
    params = freeze_and_subset(model.named_parameters(), exclude_patterns=[r"linear\..+"])
    list(params)  # execute generator
    assert not model.linear.weight.requires_grad
    assert not model.linear.bias.requires_grad
    assert model.conv.weight.requires_grad
    assert model.conv.bias.requires_grad


def test_keeping_unfrozen_params():
    model = DummyModel().train()
    assert model.linear.weight.requires_grad
    assert model.linear.bias.requires_grad
    assert model.conv.weight.requires_grad
    assert model.conv.bias.requires_grad
    params = freeze_and_subset(
        model.named_parameters(), exclude_patterns=[r"linear\..+"], keep_patterns=[r"linear.bias"]
    )
    list(params)  # execute generator
    assert not model.linear.weight.requires_grad
    assert model.linear.bias.requires_grad
    assert model.conv.weight.requires_grad
    assert model.conv.bias.requires_grad


def test_configure_optimizers():
    model = DummyModel()
    model.cfg = DictConfig(
        {
            "optimizer": {"_target_": "torch.optim.adamw.AdamW"},
            "freeze_params": [r"conv\..+"],
        }
    )
    ans = configure_optimizers(model)
    assert ans.keys() == {"optimizer"}
    assert isinstance(ans["optimizer"], torch.optim.AdamW)
    parameters = ans["optimizer"].param_groups[0]['params']
    assert len(parameters) == 2
    assert parameters[0] == model.linear.weight
    assert parameters[1] == model.linear.bias


def test_configure_optimizers_with_lr_scheduler():
    model = DummyModel()
    model.cfg = DictConfig(
        {
            "optimizer": {"_target_": "torch.optim.adamw.AdamW"},
            "lr_scheduler": {
                "_target_": "nemo.core.optim.lr_scheduler.CosineAnnealing",
                "warmup_steps": 0,
                "min_lr": 1e-6,
                "max_steps": 100000,
            },
        }
    )
    ans = configure_optimizers(model)
    assert ans.keys() == {"optimizer", "lr_scheduler"}
    assert isinstance(ans["optimizer"], torch.optim.AdamW)
    assert isinstance(ans["lr_scheduler"]["scheduler"], nemo.core.optim.lr_scheduler.CosineAnnealing)


def test_fused_adam_dtype_config_preserves_source_and_applies_compatibility(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    from nemo.collections.speechlm2.parts.optim_setup import _optimizer_config_with_torch_dtypes

    apply_patches = Mock()
    monkeypatch.setitem(
        sys.modules, "nemo_automodel.shared.te_patches", SimpleNamespace(apply_te_patches=apply_patches)
    )
    config = DictConfig(
        {
            "_target_": "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam",
            "master_weight_dtype": "torch.float32",
            "exp_avg_dtype": "bfloat16",
            "exp_avg_sq_dtype": "bfloat16",
            "lr": 9e-5,
        }
    )
    resolved = _optimizer_config_with_torch_dtypes(config)
    assert resolved["master_weight_dtype"] is torch.float32
    assert resolved["exp_avg_dtype"] is torch.bfloat16
    assert resolved["exp_avg_sq_dtype"] is torch.bfloat16
    assert resolved["lr"] == 9e-5
    assert config.master_weight_dtype == "torch.float32"
    assert config.exp_avg_dtype == "bfloat16"
    apply_patches.assert_called_once_with()


def test_fused_adam_dtype_config_rejects_non_dtype_attributes(monkeypatch):
    import sys
    from types import SimpleNamespace

    import pytest

    from nemo.collections.speechlm2.parts.optim_setup import _optimizer_config_with_torch_dtypes

    monkeypatch.setitem(
        sys.modules, "nemo_automodel.shared.te_patches", SimpleNamespace(apply_te_patches=lambda: None)
    )
    config = DictConfig(
        {"_target_": "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam", "exp_avg_dtype": "torch.Tensor"}
    )
    with pytest.raises(ValueError, match="Invalid exp_avg_dtype"):
        _optimizer_config_with_torch_dtypes(config)


def test_non_te_optimizer_config_is_unchanged():
    from nemo.collections.speechlm2.parts.optim_setup import _optimizer_config_with_torch_dtypes

    config = DictConfig({"_target_": "torch.optim.adamw.AdamW", "lr": 1e-4})
    assert _optimizer_config_with_torch_dtypes(config) is config


def test_grouped_fused_adam_resolves_dtypes_and_applies_patch_before_construction(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    from nemo.collections.speechlm2.parts import optim_setup

    events = []
    apply_patches = Mock(side_effect=lambda: events.append("patch"))
    monkeypatch.setitem(
        sys.modules, "nemo_automodel.shared.te_patches", SimpleNamespace(apply_te_patches=apply_patches)
    )
    model = DummyModel()
    model.norm = torch.nn.LayerNorm(1)
    model.cfg = DictConfig(
        {
            "optimizer": {
                "_target_": "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam",
                "master_weight_dtype": "torch.float32",
                "exp_avg_dtype": "bfloat16",
                "exp_avg_sq_dtype": "bfloat16",
                "weight_decay": 0.2,
            },
            "freeze_params": [r"conv\..+"],
        }
    )
    original_config = model.cfg.copy()

    def instantiate(config, groups, **kwargs):
        events.append("construct")
        assert events == ["patch", "construct"]
        assert config["master_weight_dtype"] is torch.float32
        assert config["exp_avg_dtype"] is torch.bfloat16
        assert config["exp_avg_sq_dtype"] is torch.bfloat16
        assert [g["weight_decay"] for g in groups] == [0.2, 0.0]
        assert [id(p) for p in groups[0]["params"]] == [id(model.linear.weight)]
        assert {id(p) for p in groups[1]["params"]} == {
            id(model.linear.bias),
            id(model.norm.weight),
            id(model.norm.bias),
        }
        return torch.optim.AdamW(groups)

    monkeypatch.setattr(optim_setup, "safe_instantiate", instantiate)
    result = optim_setup.configure_optimizers_exclude_norm_from_wd(model)
    assert isinstance(result["optimizer"], torch.optim.AdamW)
    assert model.cfg == original_config
    apply_patches.assert_called_once_with()


def test_keep_patterns_override_module_level_freezing():
    # e.g. LoRA setup freezes a whole backbone via requires_grad=False before
    # the optimizer is built; prevent_freeze_params must still re-enable a subset.
    model = DummyModel().train()
    model.linear.weight.requires_grad = False
    model.linear.bias.requires_grad = False
    params = list(freeze_and_subset(model.named_parameters(), exclude_patterns=[], keep_patterns=[r"linear\.weight"]))
    assert model.linear.weight.requires_grad
    assert not model.linear.bias.requires_grad
    assert any(p is model.linear.weight for p in params)
    assert not any(p is model.linear.bias for p in params)


def test_build_param_groups_applies_first_matching_multiplier():
    from nemo.collections.speechlm2.parts.optim_setup import build_param_groups

    model = DummyModel()
    groups = build_param_groups(model.named_parameters(), {r"linear\.weight": 10.0, r"linear\..+": 2.0}, base_lr=1e-3)
    assert len(groups) == 3
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    assert lr_of[id(model.linear.weight)] == pytest.approx(1e-2)
    assert lr_of[id(model.linear.bias)] == pytest.approx(2e-3)
    assert lr_of[id(model.conv.weight)] == pytest.approx(1e-3)
    assert lr_of[id(model.conv.bias)] == pytest.approx(1e-3)


def test_configure_optimizers_with_lr_multipliers():
    model = DummyModel()
    model.cfg = DictConfig(
        {
            "optimizer": {"_target_": "torch.optim.adamw.AdamW", "lr": 1e-4},
            "freeze_params": [r"conv\.bias"],
            "lr_multipliers": {r"linear\..+": 5.0},
        }
    )
    opt = configure_optimizers(model)["optimizer"]
    lr_of = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    assert lr_of[id(model.linear.weight)] == pytest.approx(5e-4)
    assert lr_of[id(model.linear.bias)] == pytest.approx(5e-4)
    assert lr_of[id(model.conv.weight)] == pytest.approx(1e-4)
    assert id(model.conv.bias) not in lr_of
