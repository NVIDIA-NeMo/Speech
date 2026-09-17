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

from contextlib import nullcontext
from unittest.mock import MagicMock

import torch

from nemo.utils.oomptimizer import instantiate_profiling_models


class _DummyModel:
    instances = []

    def __init__(self, config):
        self.config = config
        _DummyModel.instances.append(self)

    def to(self, device):
        self.device = device
        return self


def setup_function():
    _DummyModel.instances = []


def test_instantiate_profiling_models_single_copy():
    trainer = MagicMock()
    trainer.init_module.return_value = nullcontext()
    device = torch.device("cpu")

    model, overhead = instantiate_profiling_models(
        _DummyModel, {"hidden_size": 4}, trainer, device, simulate_ddp=False
    )

    assert len(_DummyModel.instances) == 1
    assert model is _DummyModel.instances[0]
    assert overhead == []


def test_instantiate_profiling_models_simulates_ddp():
    trainer = MagicMock()
    trainer.init_module.return_value = nullcontext()
    device = torch.device("cpu")

    model, overhead = instantiate_profiling_models(
        _DummyModel, {"hidden_size": 4}, trainer, device, simulate_ddp=True
    )

    assert len(_DummyModel.instances) == 2
    assert model is _DummyModel.instances[-1]
    assert overhead == [_DummyModel.instances[0]]
