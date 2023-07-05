# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import copy
from functools import partial
from typing import Any, Dict, Optional, Union

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.optim import adadelta, adagrad, adamax, rmsprop, rprop
from torch.optim.optimizer import Optimizer

from nemo.core.config import OptimizerParams, get_optimizer_config, register_optimizer_params
from nemo.core.optim.adafactor import Adafactor
from nemo.core.optim.novograd import Novograd
from nemo.utils import logging
from nemo.utils.model_utils import maybe_update_config_version

AVAILABLE_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'adadelta': adadelta.Adadelta,
    'adamax': adamax.Adamax,
    'adagrad': adagrad.Adagrad,
    'rmsprop': rmsprop.RMSprop,
    'rprop': rprop.Rprop,
    'novograd': Novograd,
    'adafactor': Adafactor,
}

try:
    from apex.optimizers import FusedAdam, FusedLAMB

    HAVE_APEX = True

    AVAILABLE_OPTIMIZERS['lamb'] = FusedLAMB
    AVAILABLE_OPTIMIZERS['fused_adam'] = FusedAdam
except ModuleNotFoundError:
    HAVE_APEX = False

HAVE_APEX_DISTRIBUTED_ADAM = False
if HAVE_APEX:
    try:
        # Try importing wrapper for Apex distributed Adam optimizer
        from nemo.core.optim.distributed_adam import MegatronDistributedFusedAdam

        HAVE_APEX_DISTRIBUTED_ADAM = True

        AVAILABLE_OPTIMIZERS['distributed_fused_adam'] = MegatronDistributedFusedAdam
    except (ImportError, ModuleNotFoundError):
        logging.warning("Could not import distributed_fused_adam optimizer from Apex")

__all__ = ['get_optimizer', 'register_optimizer', 'parse_optimizer_args']


def parse_optimizer_args(
    optimizer_name: str, optimizer_kwargs: Union[DictConfig, Dict[str, Any]]
) -> Union[Dict[str, Any], DictConfig]:
    """
    Parses a list of strings, of the format "key=value" or "key2=val1,val2,..."
    into a dictionary of type {key=value, key2=[val1, val2], ...}

    This dictionary is then used to instantiate the chosen Optimizer.

    Args:
        optimizer_name: string name of the optimizer, used for auto resolution of params
        optimizer_kwargs: Either a list of strings in a specified format,
            or a dictionary. If a dictionary is provided, it is assumed the dictionary
            is the final parsed value, and simply returned.
            If a list of strings is provided, each item in the list is parsed into a
            new dictionary.

    Returns:
        A dictionary
    """
    kwargs = {}

    if optimizer_kwargs is None:
        return kwargs

    optimizer_kwargs = copy.deepcopy(optimizer_kwargs)
    optimizer_kwargs = maybe_update_config_version(optimizer_kwargs)

    if isinstance(optimizer_kwargs, DictConfig):
        optimizer_kwargs = OmegaConf.to_container(optimizer_kwargs, resolve=True)

    # If it is a dictionary, perform stepwise resolution
    if hasattr(optimizer_kwargs, 'keys'):
        # Attempt class path resolution
        if '_target_' in optimizer_kwargs:  # captures (target, _target_)
            optimizer_kwargs_config = OmegaConf.create(optimizer_kwargs)
            optimizer_instance = hydra.utils.instantiate(optimizer_kwargs_config)  # type: DictConfig
            optimizer_instance = vars(optimizer_instance)
            return optimizer_instance

        # If class path was not provided, perhaps `name` is provided for resolution
        if 'name' in optimizer_kwargs:
            # If `auto` is passed as name for resolution of optimizer name,
            # then lookup optimizer name and resolve its parameter config
            if optimizer_kwargs['name'] == 'auto':
                optimizer_params_name = "{}_params".format(optimizer_name)
                optimizer_kwargs.pop('name')
            else:
                optimizer_params_name = optimizer_kwargs.pop('name')

            # Override arguments provided in the config yaml file
            if 'params' in optimizer_kwargs:
                # If optimizer kwarg overrides are wrapped in yaml `params`
                optimizer_params_override = optimizer_kwargs.get('params')
            else:
                # If the kwargs themselves are a DictConfig
                optimizer_params_override = optimizer_kwargs

            if isinstance(optimizer_params_override, DictConfig):
                optimizer_params_override = OmegaConf.to_container(optimizer_params_override, resolve=True)

            optimizer_params_cls = get_optimizer_config(optimizer_params_name, **optimizer_params_override)

            # If we are provided just a Config object, simply return the dictionary of that object
            if optimizer_params_name is None:
                optimizer_params = vars(optimizer_params_cls)
                return optimizer_params

            else:
                # If we are provided a partial class instantiation of a Config,
                # Instantiate it and retrieve its vars as a dictionary
                optimizer_params = optimizer_params_cls()  # instantiate the parameters object
                optimizer_params = vars(optimizer_params)
                return optimizer_params

        # simply return the dictionary that was provided
        return optimizer_kwargs

    return kwargs


def register_optimizer(name: str, optimizer: Optimizer, optimizer_params: OptimizerParams):
    """
    Checks if the optimizer name exists in the registry, and if it doesnt, adds it.

    This allows custom optimizers to be added and called by name during instantiation.

    Args:
        name: Name of the optimizer. Will be used as key to retrieve the optimizer.
        optimizer: Optimizer class
        optimizer_params: The parameters as a dataclass of the optimizer
    """
    if name in AVAILABLE_OPTIMIZERS:
        raise ValueError(f"Cannot override pre-existing optimizers. Conflicting optimizer name = {name}")

    AVAILABLE_OPTIMIZERS[name] = optimizer

    optim_name = "{}_params".format(optimizer.__name__)
    register_optimizer_params(name=optim_name, optimizer_params=optimizer_params)


def get_optimizer(name: str, **kwargs: Optional[Dict[str, Any]]) -> Optimizer:
    """
    Convenience method to obtain an Optimizer class and partially instantiate it with optimizer kwargs.

    Args:
        name: Name of the Optimizer in the registry.
        kwargs: Optional kwargs of the optimizer used during instantiation.

    Returns:
        a partially instantiated Optimizer
    """
    if name not in AVAILABLE_OPTIMIZERS:
        raise ValueError(
            f"Cannot resolve optimizer '{name}'. Available optimizers are : " f"{AVAILABLE_OPTIMIZERS.keys()}"
        )
    if name == 'fused_adam':
        if not torch.cuda.is_available():
            raise ValueError(f'CUDA must be available to use fused_adam.')

    optimizer = AVAILABLE_OPTIMIZERS[name]
    optimizer = partial(optimizer, **kwargs)
    return optimizer
