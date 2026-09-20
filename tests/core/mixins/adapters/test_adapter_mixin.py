# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import torch

from nemo.core import NeuralModule
from nemo.core.classes.mixins import adapter_mixin_strategies, adapter_mixins
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin


class DefaultModule(NeuralModule):
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


class DefaultModuleAdapter(DefaultModule, AdapterModuleMixin):
    def forward(self, x):
        x = super(DefaultModuleAdapter, self).forward(x)

        if self.is_adapter_available():
            # For testing purposes, cache the adapter names
            self._adapter_names = self.get_enabled_adapters()
            # call forward over model adapters, summing them up
            x = self.forward_enabled_adapters(x)

        return x


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


if adapter_mixins.get_registered_adapter(DefaultModule) is None:
    adapter_mixins.register_adapter(DefaultModule, DefaultModuleAdapter)


class TestAdapterMixin:
    @pytest.mark.unit
    def test_module_registered_adapter_by_class_path(self):
        classpath = get_classpath(DefaultModule)
        adapter_meta = adapter_mixins.get_registered_adapter(classpath)
        assert adapter_meta is not None
        assert adapter_meta.base_class == DefaultModule
        assert adapter_meta.adapter_class == DefaultModuleAdapter

    @pytest.mark.unit
    def test_module_registered_adapter_by_class(self):
        adapter_meta = adapter_mixins.get_registered_adapter(DefaultModule)
        assert adapter_meta is not None
        assert adapter_meta.base_class == DefaultModule
        assert adapter_meta.adapter_class == DefaultModuleAdapter

    @pytest.mark.unit
    def test_module_registered_adapter_by_adapter_class(self):
        adapter_meta = adapter_mixins.get_registered_adapter(DefaultModuleAdapter)
        assert adapter_meta is not None
        assert adapter_meta.base_class == DefaultModule
        assert adapter_meta.adapter_class == DefaultModuleAdapter

    @pytest.mark.unit
    def test_single_adapter(self):
        model = DefaultModuleAdapter()
        original_num_params = model.num_params()

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_params()
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_multiple_adapter(self):
        model = DefaultModuleAdapter()
        original_num_params = model.num_params()

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_params()
        assert new_num_params > original_num_params

        original_num_params = new_num_params
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg())
        new_num_params = model.num_params()
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_forward_linear_pre(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        model = DefaultModuleAdapter()
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_output = model(x)

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    def test_forward_linear_post(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        model = DefaultModuleAdapter()
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg(norm_pos='post'))
        new_output = model(x)

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    def test_multi_adapter_forward(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        model = DefaultModuleAdapter()
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg())
        new_output = model(x)

        assert model._adapter_names == ['adapter_0', 'adapter_1']
        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    def test_multi_adapter_partial_forward(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        model = DefaultModuleAdapter()
        origial_output = model(x)

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg())

        model.set_enabled_adapters(name='adapter_0', enabled=False)
        new_output = model(x)

        assert model._adapter_names == ['adapter_1']
        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    def test_forward_unfrozen_adapters(self):
        model = DefaultModuleAdapter()
        original_num_params = model.num_params()

        dim = 10
        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg(dim=dim))
        model.freeze()
        model.unfreeze_enabled_adapters()

        assert original_num_params == 2650

        original_params = 0
        adapter_params = 0
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                assert param.requires_grad is False
                original_params += param.numel()
            else:
                assert param.requires_grad is True
                adapter_params += param.numel()

        for mname, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                assert module.track_running_stats is False

        assert original_params > adapter_params

    @pytest.mark.unit
    def test_forward_linear_no_strategy(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        model = DefaultModuleAdapter()
        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

        # delete the strategy
        adapter_module = model.adapter_layer[model.get_enabled_adapters()[0]]
        del adapter_module.adapter_strategy

        with pytest.raises(AttributeError):
            _ = model(x)

    @pytest.mark.unit
    def test_forward_linear_replaced_strategy(self):
        class MultiplyAdapterStrategy(adapter_mixin_strategies.AbstractAdapterStrategy):
            def forward(self, input: torch.Tensor, adapter: torch.nn.Module, *, module: AdapterModuleMixin):
                out = adapter(input)
                return input * out

        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        model = DefaultModuleAdapter()
        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())

        # modify the strategy
        adapter_module = model.adapter_layer[model.get_enabled_adapters()[0]]
        adapter_module.adapter_strategy = MultiplyAdapterStrategy()

        out = model(x)
        # result of adapter is zero tensor, output multiplied by adapter result should be zero
        assert (out > 0.0).any() == torch.tensor(False)
