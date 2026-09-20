# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

from omegaconf import MISSING, OmegaConf

__all__ = [
    'OptimizerParams',
    'AdamParams',
    'NovogradParams',
    'SGDParams',
    'AdadeltaParams',
    'AdamaxParams',
    'AdagradParams',
    'AdamWParams',
    'RMSpropParams',
    'RpropParams',
]


@dataclass
class OptimizerParams:
    """
    Base Optimizer params with no values. User can chose it to explicitly override via
    command line arguments
    """

    lr: Optional[float] = MISSING


@dataclass
class SGDParams(OptimizerParams):
    """
    Default configuration for Adam optimizer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html?highlight=sgd#torch.optim.SGD
    """

    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False


@dataclass
class AdamParams(OptimizerParams):
    """
    Default configuration for Adam optimizer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html?highlight=adam#torch.optim.Adam
    """

    # betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class AdamWParams(OptimizerParams):
    """
    Default configuration for AdamW optimizer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class AdadeltaParams(OptimizerParams):
    """
    Default configuration for Adadelta optimizer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.Adadelta
    """

    rho: float = 0.9
    eps: float = 1e-6
    weight_decay: float = 0


@dataclass
class AdamaxParams(OptimizerParams):
    """
    Default configuration for Adamax optimizer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.Adamax
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class AdagradParams(OptimizerParams):
    """
    Default configuration for Adagrad optimizer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.Adagrad
    """

    lr_decay: float = 0
    weight_decay: float = 0
    initial_accumulator_value: float = 0
    eps: float = 1e-10


@dataclass
class RMSpropParams(OptimizerParams):
    """
    Default configuration for RMSprop optimizer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop
    """

    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0
    momentum: float = 0
    centered: bool = False


@dataclass
class RpropParams(OptimizerParams):
    """
    Default configuration for RpropParams optimizer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.Rprop
    """

    etas: Tuple[float, float] = (0.5, 1.2)
    step_sizes: Tuple[float, float] = (1e-6, 50)


@dataclass
class NovogradParams(OptimizerParams):
    """
    Configuration of the Novograd optimizer.

    It has been proposed  in "Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks"
    (https://arxiv.org/abs/1905.11286)

    Args:
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
    """

    betas: Tuple[float, float] = (0.95, 0.98)
    eps: float = 1e-8
    weight_decay: float = 0
    grad_averaging: bool = False
    amsgrad: bool = False
    luc: bool = False
    luc_trust: float = 1e-3
    luc_eps: float = 1e-8


@dataclass
class AdafactorParams(OptimizerParams):
    """
    Configuration of the Adafactor optimizer.

    It has been proposed  in "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
    (https://arxiv.org/abs/1804.04235)

    Args:
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): coefficients used for computing
            running averages of gradient and its square (default: None)
        eps (Tuple [float, float] optional)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (float, optional): scale parameter (default: False)
        relative_step (bool, optional): whether to use relative step sizes (default: False)
        warmup_init (bool, optional): whether to warmup the learning rate linearly (default: False)
    """

    beta1: float = None
    eps: Tuple[float, float] = (1e-30, 1e-3)
    clip_threshold: float = 1.0
    decay_rate: float = 0.8
    weight_decay: float = 0
    scale_parameter: bool = True
    relative_step: bool = False
    warmup_init: bool = False


def register_optimizer_params(name: str, optimizer_params: OptimizerParams):
    """
    Checks if the optimizer param name exists in the registry, and if it doesnt, adds it.

    This allows custom optimizer params to be added and called by name during instantiation.

    Args:
        name: Name of the optimizer. Will be used as key to retrieve the optimizer.
        optimizer_params: Optimizer class
    """
    if name in AVAILABLE_OPTIMIZER_PARAMS:
        raise ValueError(f"Cannot override pre-existing optimizers. Conflicting optimizer name = {name}")

    AVAILABLE_OPTIMIZER_PARAMS[name] = optimizer_params


def get_optimizer_config(name: str, **kwargs: Optional[Dict[str, Any]]) -> OptimizerParams:
    """
    Convenience method to obtain a OptimizerParams class and partially instantiate it with optimizer kwargs.

    Args:
        name: Name of the OptimizerParams in the registry.
        kwargs: Optional kwargs of the optimizer used during instantiation.

    Returns:
        a partially instantiated OptimizerParams
    """
    if name is None:
        return kwargs

    if name not in AVAILABLE_OPTIMIZER_PARAMS:
        raise ValueError(
            f"Cannot resolve optimizer parameters '{name}'. Available optimizer parameters are : "
            f"{AVAILABLE_OPTIMIZER_PARAMS.keys()}"
        )

    scheduler_params = AVAILABLE_OPTIMIZER_PARAMS[name]

    if kwargs is not None and len(kwargs) != 0:
        kwargs = OmegaConf.create(kwargs)
        OmegaConf.merge(scheduler_params(), kwargs)

    scheduler_params = partial(scheduler_params, **kwargs)
    return scheduler_params


AVAILABLE_OPTIMIZER_PARAMS = {
    'optim_params': OptimizerParams,
    'adam_params': AdamParams,
    'novograd_params': NovogradParams,
    'sgd_params': SGDParams,
    'adadelta_params': AdadeltaParams,
    'adamax_params': AdamaxParams,
    'adagrad_params': AdagradParams,
    'adamw_params': AdamWParams,
    'rmsprop_params': RMSpropParams,
    'rprop_params': RpropParams,
    'adafactor_params': AdafactorParams,
}
