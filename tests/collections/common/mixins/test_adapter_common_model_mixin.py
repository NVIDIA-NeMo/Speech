# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import tempfile
from typing import Tuple

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nemo.collections.common.parts.adapter_modules import LinearAdapter
from nemo.core import ModelPT, NeuralModule
from nemo.core.classes.mixins import adapter_mixin_strategies, adapter_mixins
from nemo.core.classes.mixins.adapter_mixins import AdapterModelPTMixin, AdapterModuleMixin
from nemo.utils import logging, logging_mode


class MockLinearAdapter1(LinearAdapter):
    pass


class MockLinearAdapter2(LinearAdapter):
    pass


class CommonModule(NeuralModule):
    """ Define a default neural module (without adapter support)"""

    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Linear(50, 50)
        self.bn = torch.nn.BatchNorm1d(50)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = x
        return out

    def num_params(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num


class CommonModuleAdapter(CommonModule, AdapterModuleMixin):
    """ Subclass the DefaultModule, adding adapter module support"""

    def forward(self, x):
        x = super().forward(x)

        if self.is_adapter_available():
            # For testing purposes, cache the adapter names
            self._adapter_names = self.get_enabled_adapters()
            # call forward over model adapters, summing them up
            x = self.forward_enabled_adapters(x)

        return x

    def get_accepted_adapter_types(self,) -> 'Set[type]':
        types = super().get_accepted_adapter_types()

        if len(types) == 0:
            self.set_accepted_adapter_types(['nemo.collections.common.parts.adapter_modules.LinearAdapter'])
            types = self.get_accepted_adapter_types()
        return types


def get_adapter_cfg(in_features=50, dim=100, norm_pos='pre'):
    cfg = {
        '_target_': 'nemo.collections.common.parts.adapter_modules.LinearAdapter',
        'in_features': in_features,
        'dim': dim,
        'norm_position': norm_pos,
    }
    return cfg


def get_classpath(cls):
    return f'{cls.__module__}.{cls.__name__}'


if adapter_mixins.get_registered_adapter(CommonModule) is None:
    adapter_mixins.register_adapter(CommonModule, CommonModuleAdapter)


class TestCommonAdapterModuleMixin:
    @pytest.mark.unit
    def test_get_accepted_adapter_types(self):

        model = CommonModuleAdapter()
        original_num_params = model.num_weights

        assert not hasattr(model, '_accepted_adapter_types')

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        # Adding adapter will implicitly try to get accepted adapters, initializing the set
        assert hasattr(model, '_accepted_adapter_types')

        types = model.get_accepted_adapter_types()
        types = list(types)
        assert len(types) == 1
        assert types[0].__name__ == 'LinearAdapter'

    @pytest.mark.unit
    def test_set_accepted_adapter_types_reset_types(self):
        model = CommonModuleAdapter()
        original_num_params = model.num_weights

        assert not hasattr(model, '_accepted_adapter_types')

        # Implicitly sets some types
        model.get_accepted_adapter_types()

        # Adding adapter will implicitly try to get accepted adapters, initializing the set
        assert hasattr(model, '_accepted_adapter_types')

        types = model.get_accepted_adapter_types()
        types = list(types)
        assert len(types) == 1
        assert types[0].__name__ == 'LinearAdapter'

        # Reset type now
        model.set_accepted_adapter_types([])

        assert hasattr(model, '_accepted_adapter_types')
        types = model._accepted_adapter_types
        assert len(types) == 0

        # Since types are empty, get_types will set the default types
        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_set_accepted_adapter_types_invalid_class(self):
        model = CommonModuleAdapter()
        original_num_params = model.num_weights

        assert not hasattr(model, '_accepted_adapter_types')

        # Explicitly set the accepted types to be the subclasses
        model.set_accepted_adapter_types(
            [
                get_classpath(MockLinearAdapter1),  # Pass string class path
                MockLinearAdapter2,  # Pass actual class itself
            ]
        )

        # Should throw error because the base class is now no longer in accepted list
        # and the get_types method does not fill in the default
        with pytest.raises(ValueError):
            model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
