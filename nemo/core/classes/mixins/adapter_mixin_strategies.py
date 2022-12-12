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

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch

from nemo.core.classes.mixins import AccessMixin


class AbstractAdapterStrategy(ABC):
    def forward(self, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'):
        """
        Forward method that defines how the output of the adapter should be merged with the input, or if it
        should be merged at all.

        Also provides the module that called this strategy - thereby allowing access to all other
        adapters in the calling module. This can be useful if one adapter is a meta adapter, that
        combines the outputs of various adapters. In such a case, the input can be forwarded across
        all other adapters, collecting their outputs, and those outputs can then be merged via some
        strategy. For example, refer to :

        - [AdapterFusion: Non-Destructive Task Composition for Transfer Learning](https://arxiv.org/abs/2005.00247)
        - [Exploiting Adapters for Cross-lingual Low-resource Speech Recognition](https://arxiv.org/abs/2105.11905)

        Args:
            input: Original output tensor of the module, or the output of the previous adapter (if more than
                one adapters are enabled).
            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.

        Returns:
            The result tensor, after one of the active adapters has finished its forward passes.
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ReturnResultAdapterStrategy(AbstractAdapterStrategy):
    """
    An implementation of an adapter strategy that simply returns the result of the adapter.
    Supports stochastic
    """

    def forward(self, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'):
        """
        A basic strategy, which simply returns the result of the adapter's calculation as the output.

        Args:
            input: Original output tensor of the module, or the output of the previous adapter (if more than
                one adapters are enabled).
            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.

        Returns:
            The result tensor, after one of the active adapters has finished its forward passes.
        """
        result = self.compute_output(input, adapter, module=module)

        return result

    def compute_output(
        self,
        input: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], Dict[str, Any]],
        adapter: torch.nn.Module,
        *,
        module: 'AdapterModuleMixin',
    ) -> torch.Tensor:
        """
        Compute the output of a single adapter to some input.

        Args:
            input: Original output tensor of the module, or the output of the previous adapter (if more than
                one adapters are enabled).
            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.

        Returns:
            The result tensor, after one of the active adapters has finished its forward passes.
        """
        if isinstance(input, (list, tuple)):
            out = adapter(*input)
        elif isinstance(input, dict):
            out = adapter(**input)
        else:
            out = adapter(input)
        return out


@dataclass
class ReturnResultAdapterStrategyConfig:
    _target_: str = "{0}.{1}".format(
        ReturnResultAdapterStrategy.__module__, ReturnResultAdapterStrategy.__name__
    )  # mandatory field


class ResidualAddAdapterStrategy(AbstractAdapterStrategy):
    """
    An implementation of residual addition of an adapter module with its input.
    Supports stochastic depth regularization.
    """

    def __init__(self, stochastic_depth: float = 0.0, l2_lambda: float = 0.0):
        """
        An implementation of residual addition of an adapter module with its input.
        Performs output = input + adapter(input).

        Args:
            stochastic_depth: float, when greater than one, can optionally dropout the output of
                the adapter's forward pass.
            l2_lambda: L2 norm of the difference between the original input to the function, and the adapter's
                output result. Disabled if set to 0.0.
        """
        super().__init__()
        self.stochastic_depth = stochastic_depth
        self.l2_lambda = l2_lambda

    def forward(self, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'):
        """
        A basic strategy, comprising of a residual connection over the input, after forward pass by
        the underlying adapter.

        Args:
            input: Original output tensor of the module, or the output of the previous adapter (if more than
                one adapters are enabled).
            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.

        Returns:
            The result tensor, after one of the active adapters has finished its forward passes.
        """
        out = self.compute_output(input, adapter, module=module)

        # If not in training mode, or probability of stochastic depth is 0, skip step.
        p = self.stochastic_depth
        if not module.training or p == 0.0:
            pass
        else:
            out = self.apply_stochastic_depth(out, input, adapter, module=module)

        # Return the residual connection output = input + adapter(input)
        result = input + out

        # If l2_lambda is activated, register the loss value
        self.compute_auxiliary_losses(result, input, adapter, module=module)

        return result

    def compute_output(
        self, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'
    ) -> torch.Tensor:
        """
        Compute the output of a single adapter to some input.

        Args:
            input: Original output tensor of the module, or the output of the previous adapter (if more than
                one adapters are enabled).
            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.

        Returns:
            The result tensor, after one of the active adapters has finished its forward passes.
        """
        out = adapter(input)
        return out

    def apply_stochastic_depth(
        self, output: torch.Tensor, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'
    ):
        """
        Compute and apply stochastic depth if probability is greater than 0.

        Args:
            output: The result tensor, after one of the active adapters has finished its forward passes.
            input: Original output tensor of the module, or the output of the previous adapter (if more than
                one adapters are enabled).
            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.

        Returns:
            The result tensor, after stochastic depth has been potentially applied to it.
        """
        # Perform stochastic depth if needed.
        p = self.stochastic_depth
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Stochastic depth probability has to be between 0 and 1, but got {p}")

        # Apply stochastic depth to the output of adapter.
        keep_prob = 1.0 - p
        shape = [1] * output.ndim
        noise = torch.empty(shape, dtype=output.dtype, device=output.device)
        noise = noise.bernoulli_(keep_prob)
        if keep_prob > 0.0:  # Done to normalize activation for inference mode
            noise.div_(keep_prob)

        output = noise * output

        return output

    def compute_auxiliary_losses(
        self, output: torch.Tensor, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'
    ):
        """
        Compute any auxiliary losses and preserve it in the tensor registry.

        Args:
            output: The result tensor, after one of the active adapters has finished its forward passes.
            input: Original output tensor of the module, or the output of the previous adapter (if more than
                one adapters are enabled).
            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.
        """
        if module.training and self.l2_lambda > 0.0:
            if not isinstance(adapter, AccessMixin):
                raise ValueError(f"Module {adapter.__class__.__name__} does not implement AccessMixin !")

            # Only add auxiliary loss if adapter has trainable parameters that require gradients
            if next(adapter.parameters()).requires_grad is True:
                # Check if globally allowed to compute aux loss
                compute_aux_loss = adapter.access_cfg.get('compute_adapter_loss', True)

                if compute_aux_loss:
                    # if l2 lambda is enabled, also enable AccessMixin
                    adapter.set_access_enabled(access_enabled=True)

                    l2_loss = self.l2_lambda * (input - output).square().reshape(input.size(0), -1).sum(dim=-1).mean()
                    adapter.register_accessible_tensor(name='adapter_loss', tensor=l2_loss)


@dataclass
class ResidualAddAdapterStrategyConfig:
    stochastic_depth: float = 0.0
    l2_lambda: float = 0.0

    _target_: str = "{0}.{1}".format(
        ResidualAddAdapterStrategy.__module__, ResidualAddAdapterStrategy.__name__
    )  # mandatory field
