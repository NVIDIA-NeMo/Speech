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
