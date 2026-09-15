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
from nemo.core.classes.mixins import AdapterModuleMixin, access_mixins, adapter_mixin_strategies, adapter_mixins
from nemo.utils import config_utils


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


class TestAdapterStrategy:
    @pytest.mark.unit
    def test_ResidualAddAdapterStrategyConfig(self):
        IGNORED_ARGS = ['_target_']

        result = config_utils.assert_dataclass_signature_match(
            adapter_mixin_strategies.ResidualAddAdapterStrategy,
            adapter_mixin_strategies.ResidualAddAdapterStrategyConfig,
            ignore_args=IGNORED_ARGS,
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_strategy_default(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        module = DefaultModuleAdapter()
        module.add_adapter(name='temp', cfg=get_adapter_cfg())
        adapter = module.adapter_layer[module.get_enabled_adapters()[0]]

        # update the strategy
        adapter_strategy = adapter_mixin_strategies.ResidualAddAdapterStrategy()
        adapter.adapter_strategy = adapter_strategy

        with torch.no_grad():
            assert adapter_strategy.stochastic_depth == 0.0
            out = adapter_strategy.forward(x, adapter, module=module)
            assert (out - x).abs().mean() < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('stochastic_depth', [0.0, 1.0])
    def test_strategy_stochasic_depth(self, stochastic_depth):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        module = DefaultModuleAdapter()
        module.add_adapter(name='temp', cfg=get_adapter_cfg())

        # extract adapter
        adapter = module.adapter_layer[module.get_enabled_adapters()[0]]
        # reinitialize the final layer of the adapter module (so that it is not zero init)
        adapter.module[-1].weight.data += 1

        # get just module output
        module.set_enabled_adapters('temp', enabled=False)
        module_out = module(x)

        # get module + adapter output
        module.set_enabled_adapters('temp', enabled=True)
        module_adapter_out = module(x)

        assert (
            module_out - module_adapter_out
        ).abs().sum() > 0  # results should not be the same after adapter forward now

        adapter_strategy = adapter_mixin_strategies.ResidualAddAdapterStrategy(stochastic_depth=stochastic_depth)
        adapter.adapter_strategy = adapter_strategy

        module.eval()
        with torch.inference_mode():  # stochastic depth disabled, no grad tracking
            assert adapter.adapter_strategy.stochastic_depth == stochastic_depth

            out = adapter_strategy.forward(module_out, adapter, module=module)
            assert (out - module_adapter_out).abs().mean() < 1e-5

        module.train()
        with torch.inference_mode():  # stochastic depth enabled, but no grad tracking during training mode
            out = adapter_strategy.forward(module_out, adapter, module=module)

            if stochastic_depth == 0.0:
                check = module_adapter_out
            else:
                check = module_out
            assert (out - check).abs().mean() < 1e-5

    @pytest.mark.unit
    def test_strategy_l2_lambda(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        module = DefaultModuleAdapter()
        module.add_adapter(name='temp', cfg=get_adapter_cfg())
        module.train()
        adapter = module.adapter_layer[module.get_enabled_adapters()[0]]

        # update the strategy
        adapter_strategy = adapter_mixin_strategies.ResidualAddAdapterStrategy(l2_lambda=0.01)
        adapter.adapter_strategy = adapter_strategy

        with torch.no_grad():
            access_mixins.AccessMixin.reset_registry(module)
            assert access_mixins.AccessMixin.is_access_enabled() is False

            assert adapter_strategy.stochastic_depth == 0.0
            assert adapter_strategy.l2_lambda > 0.0

            out = adapter_strategy.forward(x, adapter, module=module)
            assert (out - x).abs().mean() < 1e-5

            # extract losses
            assert access_mixins.AccessMixin.is_access_enabled() is True
            auxiliary_losses = access_mixins.AccessMixin.get_module_registry(module)
            loss = list(auxiliary_losses.values())[0]
            assert 'adapter_loss' in loss
            assert loss['adapter_loss'][0] == torch.tensor(0.0)  # initially adapter is 0 init, no loss required.
