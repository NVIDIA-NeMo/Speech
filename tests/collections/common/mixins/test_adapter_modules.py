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

import pytest
import torch

from nemo.collections.common.parts import adapter_modules
from nemo.core.classes.mixins import adapter_mixin_strategies
from nemo.utils import config_utils


class TestAdapterModules:
    @pytest.mark.unit
    def test_linear_adapter_config(self):
        IGNORED_ARGS = ['_target_']

        result = config_utils.assert_dataclass_signature_match(
            adapter_modules.LinearAdapter, adapter_modules.LinearAdapterConfig, ignore_args=IGNORED_ARGS
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_linear_adapter_init(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        adapter = adapter_modules.LinearAdapter(in_features=50, dim=5)

        with torch.no_grad():
            assert adapter.module[-1].weight.sum() == 0
            if hasattr(adapter.module[-1], 'bias') and adapter.module[-1].bias is not None:
                assert adapter.module[-1].bias.sum() == 0

            out = adapter(x)
            assert out.sum().abs() <= 1e-8

    @pytest.mark.unit
    def test_linear_adapter_dropout(self):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        adapter = adapter_modules.LinearAdapter(in_features=50, dim=5, dropout=0.5)

        with torch.no_grad():
            assert adapter.module[-1].weight.sum() == 0
            if hasattr(adapter.module[-1], 'bias') and adapter.module[-1].bias is not None:
                assert adapter.module[-1].bias.sum() == 0

            out = adapter(x)
            assert out.sum().abs() <= 1e-8

    @pytest.mark.unit
    @pytest.mark.parametrize('norm_position', ['pre', 'post'])
    def test_linear_adapter_norm_position(self, norm_position):
        torch.random.manual_seed(0)
        x = torch.randn(2, 50)

        adapter = adapter_modules.LinearAdapter(in_features=50, dim=5, norm_position=norm_position)

        with torch.no_grad():
            assert adapter.module[-1].weight.sum() == 0
            if hasattr(adapter.module[-1], 'bias') and adapter.module[-1].bias is not None:
                assert adapter.module[-1].bias.sum() == 0

            out = adapter(x)
            assert out.sum().abs() <= 1e-8

    @pytest.mark.unit
    def test_linear_adapter_strategy(self):
        adapter = adapter_modules.LinearAdapter(in_features=50, dim=5)
        assert hasattr(adapter, 'adapter_strategy')
        assert adapter.adapter_strategy is not None
        # assert default strategy is set
        assert isinstance(adapter.adapter_strategy, adapter_mixin_strategies.ResidualAddAdapterStrategy)
