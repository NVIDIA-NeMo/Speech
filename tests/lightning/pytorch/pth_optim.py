# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import Mock

import lightning.pytorch as pl
import pytest
import torch
from torch.optim import SGD

from nemo.lightning.pytorch.optim.base import LRSchedulerModule
from nemo.lightning.pytorch.optim.pytorch import PytorchOptimizerModule


class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.layer(x)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def optimizer_fn():
    return Mock(side_effect=lambda params: SGD(params, lr=0.01, weight_decay=0.1))


@pytest.fixture
def lr_scheduler():
    return Mock(spec=LRSchedulerModule)


@pytest.fixture
def optimizer_module(optimizer_fn, lr_scheduler):
    return PytorchOptimizerModule(optimizer_fn, lr_scheduler)


def test_optimizer_module_initialization(optimizer_module, optimizer_fn, lr_scheduler):
    assert optimizer_module.optimizer_fn == optimizer_fn
    assert optimizer_module.lr_scheduler == lr_scheduler
    assert callable(optimizer_module.no_weight_decay_cond)
    assert optimizer_module.lr_mult == 1.0


def test_optimizer_creation(dummy_model, optimizer_module):
    optimizer = optimizer_module.optimizers(dummy_model)
    assert isinstance(optimizer, list)
    assert len(optimizer) > 0
    assert isinstance(optimizer[0], torch.optim.Optimizer)


def test_connect_method(dummy_model, optimizer_module):
    optimizer_module.connect(dummy_model)
