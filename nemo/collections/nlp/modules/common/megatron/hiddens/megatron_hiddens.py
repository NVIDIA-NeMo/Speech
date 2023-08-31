# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
"""
In order to register external hidden transforms and losses please use the following methods:
* register_hidden_loss(cls_name: str, class_path: str)
* register_hidden_transform(cls_name: str, class_path: str)

See example config in: examples/nlp/language_modeling/conf/megatron_hiddens_base_config.yaml
"""

import functools
import itertools
from typing import List

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.modules.common.megatron.hiddens.megatron_hidden_loss import MegatronBaseHiddenLoss
from nemo.collections.nlp.modules.common.megatron.hiddens.megatron_hidden_transform import MegatronBaseHiddenTransform
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path

try:
    from megatron.core import ModelParallelConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    # fake missing classes with None attributes
    ModelParallelConfig = ApexGuardDefaults()

    HAVE_MEGATRON_CORE = False

__all__ = [
    "MegatronHiddensModule",
    "get_registered_hiddens",
    "register_hidden_loss",
    "register_hidden_transform",
    "get_hiddens_module",
]

# a registry of all hidden transforms (maps name to class path)
_LOSS_CLASS_REGISTRY = {
    "a_mim": "nemo.collections.nlp.modules.common.megatron.hiddens.megatron_hidden_loss.MegatronAMIMHiddenLoss",
    "vae": "nemo.collections.nlp.modules.common.megatron.hiddens.megatron_hidden_loss.MegatronVAEHiddenLoss",
}

# a registry of all hidden losses (maps name to class path)
_TRANSFORM_CLASS_REGISTRY = {
    "cond_gaussian": "nemo.collections.nlp.modules.common.megatron.hiddens.megatron_hidden_transform.MegatronGaussianHiddenTransform",
}


def get_registered_hiddens():
    """
    Return:
        A dictionary with all registered hidden transforms and losses.

    Example:
        {
            "loss": ["a_mim", "vae"],
            "transform": ["cond_gaussian"],
        }
    """
    return {
        "loss": list(_LOSS_CLASS_REGISTRY.keys()),
        "transform": list(_TRANSFORM_CLASS_REGISTRY.keys()),
    }


def register_hidden_loss(cls_name: str, class_path: str):
    """
    Register a hidden loss.

    
    Args:
        cls_name: name of the class
        class_path: path to the class (e.g., "nemo.collections.nlp.modules.common.megatron.hiddens.megatron_hidden_transform.MegatronGaussianHiddenTransform")
    """
    if cls_name in _LOSS_CLASS_REGISTRY:
        raise ValueError(f"Cannot register duplicate hidden loss ({cls_name})")
    _LOSS_CLASS_REGISTRY[cls_name] = class_path
    logging.info(f"Registered hidden loss {cls_name} at {class_path}")


def register_hidden_transform(cls_name: str, class_path: str):
    """
    Register a hidden transform.
    
    Args:
        cls_name: name of the class
        class_path: path to the class (e.g., "nemo.collections.nlp.modules.common.megatron.hiddens.megatron_hidden_transform.MegatronGaussianHiddenTransform")
    """
    if cls_name in _TRANSFORM_CLASS_REGISTRY:
        raise ValueError(f"Cannot register duplicate hidden transform ({cls_name})")
    _TRANSFORM_CLASS_REGISTRY[cls_name] = class_path
    logging.info(f"Registered hidden transform {cls_name} at {class_path}")


def get_hiddens_module(cfg=None, model_parallel_cfg: ModelParallelConfig = None):
    """Build a MegatronHiddensModule from a configuration cfg"""
    # Build a hiddens module if config is provided.
    if cfg is None:
        return None

    logging.info(f"NOTE: Adding hiddens transforms and losses")

    # build all hidden transforms. We support a list or a dictionary of transforms (list enforces order)
    transform_cfg = cfg.get("transform", [])
    if isinstance(transform_cfg, (DictConfig, dict)):
        transform_cfg = [transform_cfg]
    hidden_transforms = []
    # here we expect transform_cfg to be a list of dictionaries
    for cur_list_cfg in transform_cfg:
        for name, cur_cfg in cur_list_cfg.items():
            cls_kwargs = OmegaConf.to_container(cur_cfg)
            cls_kwargs["model_parallel_cfg"] = model_parallel_cfg
            if not "cls_name" in cls_kwargs:
                raise KeyError(f"Missing 'cls_name' in hidden transform {name}")

            cls_name = cls_kwargs.pop("cls_name")
            # add name based on dictionary if not given in conf
            if "name" not in cls_kwargs:
                cls_kwargs["name"] = name
            if cls_name not in _TRANSFORM_CLASS_REGISTRY:
                raise KeyError(f"Unknown hidden transform {cls_name}, available: {_TRANSFORM_CLASS_REGISTRY.keys()}")
            try:
                cur_transform = import_class_by_path(_TRANSFORM_CLASS_REGISTRY[cls_name])(**cls_kwargs)
            except Exception as e:
                logging.error(f"Failed to build hidden transform {name} with cfg={cur_cfg}")
                raise e

            hidden_transforms.append(cur_transform)
            logging.info(f"Added transform {name} with cfg={cur_cfg}")

    # build all hidden losses
    loss_cfg = cfg.get("loss", [])
    if isinstance(loss_cfg, (DictConfig, dict)):
        loss_cfg = [loss_cfg]
    hidden_loss_transforms = []
    # here we expect loss_cfg to be a list of dictionaries
    for cur_list_cfg in loss_cfg:
        for name, cur_cfg in cur_list_cfg.items():
            cls_kwargs = OmegaConf.to_container(cur_cfg)
            if not "cls_name" in cls_kwargs:
                raise KeyError(f"Missing 'cls_name' in hidden loss {name}")

            cls_name = cls_kwargs.pop("cls_name")
            # add name based on dictionary if not given in conf
            if "name" not in cls_kwargs:
                cls_kwargs["name"] = name
            if cls_name not in _LOSS_CLASS_REGISTRY:
                raise KeyError(f"Unknown hidden loss {cls_name}, available: {_LOSS_CLASS_REGISTRY.keys()}")
            try:
                cur_loss = import_class_by_path(_LOSS_CLASS_REGISTRY[cls_name])(**cls_kwargs)
            except Exception as e:
                logging.error(f"Failed to build hidden loss {name} with cfg={cur_cfg}")
                raise e
            hidden_loss_transforms.append(cur_loss)
            logging.info(f"Added loss {name} with cfg={cur_cfg}")

    enc_output_name = cfg.get("enc_output_name", "hiddens")

    return MegatronHiddensModule(
        hidden_transforms=hidden_transforms,
        hidden_loss_transforms=hidden_loss_transforms,
        enc_output_name=enc_output_name,
    )


class MegatronHiddensModule(torch.nn.Module):
    """
    This class jointly handles the hidden transforms and hidden loss transforms.
    It helps in validating, and applying the transforms.
    """

    def __init__(
        self,
        hidden_transforms: List[MegatronBaseHiddenLoss] = [],
        hidden_loss_transforms: List[MegatronBaseHiddenTransform] = [],
        enc_output_name: str = "hiddens",  # name (key) of the encoder output
        tokens_loss_weight: float = 1.0,  # weight of the tokens loss
        loss_prefix: str = "hiddens_",  # if not None or "", add this prefix to all loss names
    ):
        super().__init__()
        self.hidden_transforms = hidden_transforms
        self.hidden_loss_transforms = hidden_loss_transforms
        self.enc_output_name = enc_output_name
        self.tokens_loss_weight = tokens_loss_weight
        self.loss_prefix = loss_prefix

        # register all hidden / loss transforms as submodules to support learned parameters
        if not all([isinstance(ht, MegatronBaseHiddenLoss) for ht in self.hidden_loss_transforms]):
            raise TypeError(
                f"hidden_loss_transforms should be a list of MegatronBaseHiddenLoss, but got {hidden_loss_transforms}"
            )
        self.hidden_loss_transforms = torch.nn.ModuleList(self.hidden_loss_transforms)
        if not all([isinstance(ht, MegatronBaseHiddenTransform) for ht in self.hidden_transforms]):
            raise TypeError(
                f"hidden_transforms should be a list of MegatronBaseHiddenTransform, but got {hidden_transforms}"
            )
        self.hidden_transforms = torch.nn.ModuleList(self.hidden_transforms)

        # validate the inputs and outputs of all hidden transforms (make sure there are no duplicate output names)
        duplicate_names = {}
        # initialize with available outputs from hidden transforms with hiddens and mask as default
        hidden_outputs = set(["hiddens", "hiddens_mask", "enc_output"])
        for ht in self.hidden_transforms:
            # validate that all required inputs are available by order of hidden transforms
            cur_input_names = set(ht.input_names)
            if not cur_input_names.issubset(hidden_outputs):
                raise ValueError(
                    f"Hidden transform {ht.name} requires inputs {cur_input_names - hidden_outputs} that are not available"
                )

            # collect all duplicate output names
            cur_hidden_outputs = set(ht.output_names)
            if not cur_hidden_outputs.isdisjoint(hidden_outputs):
                duplicate_names[ht.name] = list(cur_hidden_outputs.intersection(hidden_outputs))

            hidden_outputs.update(cur_hidden_outputs)

        # fail here reporting all duplicate output names
        if duplicate_names:
            raise ValueError(
                f"Hidden transforms have duplicate outputs {{name: [duplicate outputs]}} = {duplicate_names}"
            )

        # validate that all loss transforms are supported by output of hidden transforms ("hiddens" is given by default)
        loss_inputs = set(itertools.chain(*[lt.input_names for lt in self.hidden_loss_transforms]))
        if not loss_inputs.issubset(hidden_outputs):
            loss_inputs_dict = {lt.name: lt.input_names for lt in self.hidden_loss_transforms}
            raise ValueError(
                f"Loss transforms inputs = {loss_inputs - hidden_outputs} are not supported by hidden transforms with hidden_outputs = {hidden_outputs}, expected inputs per loss = {loss_inputs_dict}"
            )

    @functools.cached_property
    def hidden_outputs(self):
        """Get the hidden outputs from all the hidden transforms"""
        all_output_names = [ht.output_names for ht in self.hidden_transforms] + [["hiddens", "hiddens_mask"]]
        output_names = set().union(*all_output_names)

        return list(output_names)

    @functools.cached_property
    def loss_inputs(self):
        """Get the loss inputs from all the loss transforms"""
        loss_inputs = set().union(*[lt.input_names for lt in self.hidden_loss_transforms])
        return list(loss_inputs)

    def apply_hidden_transforms(self, inputs, batch_data=None):
        """
        Apply hidden transforms
        Args:
            inputs: a dictionary of inputs, with "hiddens" as the default key for hidden states
            batch_data: a dictionary of batch data (e.g. "input_features"), optional
        
        Returns:
            outputs: a dictionary of outputs, collecting 
        """
        outputs = inputs.copy()
        for hidden_transform in self.hidden_transforms:
            # make sure to collect all outputs from hidden transforms
            outputs.update(hidden_transform.transform(outputs, batch_data=batch_data))

        # update final encoder output
        outputs["enc_output"] = outputs[self.enc_output_name]

        return outputs

    def apply_loss_transforms(self, outputs, batch_data=None):
        """
        Apply loss transforms
        Args:
            outputs: a dictionary of outputs (after hidden transforms)
            batch_data: a dictionary of batch data (e.g. "target_ids"), optional
        
        Returns:
            loss_dict: a dictionary of all losses, 
                {
                    loss: joint loss (float),
                    <name>_*: loss values from loss transforms, could be loss, or loss elements
                }
        """
        loss_dict = {}
        joint_loss = 0.0
        for i, loss_transform in enumerate(self.hidden_loss_transforms):
            cur_loss_dict = loss_transform.loss(outputs, batch_data=batch_data)
            joint_loss = joint_loss + cur_loss_dict["weighted_loss"]
            cur_loss_dict.pop("weighted_loss")
            # add name to loss values
            if loss_transform.name:
                cur_loss_dict = {f"{loss_transform.name}_{k}": v for k, v in cur_loss_dict.items()}

            # check if cur_loss keys are unique - we do not allow to override keys
            dup_keys = set(cur_loss_dict.keys()).intersection(set(loss_dict.keys()))
            if len(dup_keys):
                raise ValueError(
                    f"Loss transform ({i}) {loss_transform} is trying to override the following loss keys {list(dup_keys)}"
                )
            # update loss dict
            loss_dict.update(cur_loss_dict)

        # joint weighted loss (float)
        loss_dict["loss"] = joint_loss

        # add prefix to all loss keys (default to 'hiddens_')
        if self.loss_prefix:
            loss_dict = {f"{self.loss_prefix}{k}": v for k, v in loss_dict.items()}

        # add tokens loss weight (to be used by caller, or be ignored)
        loss_dict["tokens_loss_weight"] = torch.tensor(self.tokens_loss_weight).to(joint_loss)

        return loss_dict
