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

from dataclasses import dataclass, is_dataclass
from typing import Any, Optional

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn as nn

from nemo.collections.common.parts.utils import activation_registry
from nemo.core.classes.mixins import access_mixins, adapter_mixin_strategies


class AdapterModuleUtil(access_mixins.AccessMixin):
    """
    Base class of Adapter Modules, providing common functionality to all Adapter Modules.
    """

    def setup_adapter_strategy(self, adapter_strategy: Optional[adapter_mixin_strategies.AbstractAdapterStrategy]):
        """
        Setup adapter strategy of this class, enabling dynamic change in the way the adapter output is
        merged with the input.

        When called successfully, will assign the variable `adapter_strategy` to the module.

        Args:
            adapter_strategy: Can be a None or an implementation of AbstractAdapterStrategy.
        """
        # set default adapter strategy
        if adapter_strategy is None:
            adapter_strategy = self.get_default_strategy_config()

        if is_dataclass(adapter_strategy):
            adapter_strategy = OmegaConf.structured(adapter_strategy)
            OmegaConf.set_struct(adapter_strategy, False)

        # The config must have the `_target_` field pointing to the actual adapter strategy class
        # which will load that strategy dynamically to this module.
        if isinstance(adapter_strategy, dict) or OmegaConf.is_config(adapter_strategy):
            self.adapter_strategy = instantiate(adapter_strategy)
        elif isinstance(adapter_strategy, adapter_mixin_strategies.AbstractAdapterStrategy):
            self.adapter_strategy = adapter_strategy
        else:
            raise AttributeError(f'`adapter_strategy` provided is invalid : {adapter_strategy}')

    def get_default_strategy_config(self) -> 'dataclass':
        """
        Returns a default adapter module strategy.
        """
        return adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()


class LinearAdapter(nn.Module, AdapterModuleUtil):

    """
    Simple Linear Feedforward Adapter module with LayerNorm and singe hidden layer with activation function.
    Note: The adapter explicitly initializes its final layer with all zeros in order to avoid affecting the
    original model when all adapters are disabled.

    Args:
        in_features: Input dimension of the module. Note that for adapters, input_dim == output_dim.
        dim: Hidden dimension of the feed forward network.
        activation: Str name for an activation function.
        norm_position: Str, can be `pre` or `post`. Defaults to `pre`. Determines whether the normalization
            will occur in the first layer or the last layer. Certain architectures may prefer one over the other.
        dropout: float value, whether to perform dropout on the output of the last layer of the adapter.
        adapter_strategy: By default, ResidualAddAdapterStrategyConfig. An adapter composition function object.
    """

    def __init__(
        self,
        in_features: int,
        dim: int,
        activation: str = 'swish',
        norm_position: str = 'pre',
        dropout: float = 0.0,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__()

        activation = activation_registry[activation]()
        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        assert norm_position in ['pre', 'post']
        self.norm_position = norm_position

        if norm_position == 'pre':
            self.module = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, dim, bias=False),
                activation,
                nn.Linear(dim, in_features, bias=False),
            )

        elif norm_position == 'post':
            self.module = nn.Sequential(
                nn.Linear(in_features, dim, bias=False),
                activation,
                nn.Linear(dim, in_features, bias=False),
                nn.LayerNorm(in_features),
            )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Final layer initializations must be 0
        if self.norm_position == 'pre':
            self.module[-1].weight.data *= 0

        elif self.norm_position == 'post':
            self.module[-1].weight.data *= 0
            self.module[-1].bias.data *= 0

    def forward(self, x):
        x = self.module(x)

        # Add dropout if available
        if self.dropout is not None:
            x = self.dropout(x)

        return x


@dataclass
class LinearAdapterConfig:
    in_features: int
    dim: int
    activation: str = 'swish'
    norm_position: str = 'pre'
    dropout: float = 0.0
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(LinearAdapter.__module__, LinearAdapter.__name__)
