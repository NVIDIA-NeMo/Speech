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

import os
import tempfile

import pytest
import torch

from nemo.core.classes.module import NeuralModule


class TempModule(NeuralModule):

    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Linear(10, 10, bias=False)
        self.layer2 = torch.nn.Linear(10, 10, bias=False)


class TestNeuralModule:

    @pytest.mark.unit
    def test_num_weights(self):
        module = TempModule()
        assert module.num_weights == 200

    @pytest.mark.unit
    def test_freeze(self):
        module = TempModule()
        module.freeze()
        for p in module.parameters():
            assert not p.requires_grad

    @pytest.mark.unit
    def test_unfreeze(self):
        module = TempModule()
        module.freeze()
        module.unfreeze()
        for p in module.parameters():
            assert p.requires_grad

    @pytest.mark.unit
    def test_as_frozen(self):
        module = TempModule()

        for p in module.parameters():
            assert p.requires_grad

        with module.as_frozen():
            for p in module.parameters():
                assert not p.requires_grad

        for p in module.parameters():
            assert p.requires_grad

    @pytest.mark.unit
    def test_partial_unfreeze(self):
        module = TempModule()

        for param in module.layer1.parameters():
            param.requires_grad = False

        module.freeze()

        for param in module.layer1.parameters():
            assert not param.requires_grad

        assert module._frozen_grad_map is not None
        assert len(module._frozen_grad_map) == 2
        assert module._frozen_grad_map['layer1.weight'] is False

        module.unfreeze(partial=True)

        # layer1 should still be frozen due to partial unfreeze
        assert module.layer1.weight.requires_grad is False
        assert not hasattr(module, '_frozen_grad_map')
