# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Mapping, Sequence

import hydra
import torch
from lightning.pytorch.plugins import HalfPrecision
from omegaconf import DictConfig, OmegaConf
from typing_extensions import override


def resolve_trainer_cfg(trainer_cfg: DictConfig) -> DictConfig:
    """
    Resolves and processes a trainer configuration.

    This function handles specific trainer configuration details:
    - For half precision setups, replaces precision settings with custom plugins
    - Instantiates strategy objects from mapping configurations
    - Instantiates custom callbacks from sequences

    Args:
        trainer_cfg: A DictConfig containing trainer configuration parameters

    Returns:
        A processed DictConfig with resolved configuration values
    """
    trainer_cfg = OmegaConf.to_container(trainer_cfg, resolve=True)

    # Avoids downcasting 'audio' tensors in 'true' half precision setups.
    precision = trainer_cfg.get("precision")
    if precision in ("fp16-true", "bf16-true"):
        trainer_cfg.pop("precision", None)
        trainer_cfg["plugins"] = [HalfPrecisionForAudio(precision)]

    # Allows customizable strategies (eg ModelParallelStrategy) in YAML configs.
    if (strategy := trainer_cfg.get("strategy", None)) is not None and isinstance(strategy, Mapping):
        trainer_cfg["strategy"] = hydra.utils.instantiate(strategy)

    # Allows to add custom callbacks (e.g. NsysCallback) from YAML config.
    if (cbs := trainer_cfg.get("callbacks", None)) is not None and isinstance(cbs, Sequence):
        resolved = []
        for cb in cbs:
            resolved.append(hydra.utils.instantiate(cb))
        trainer_cfg["callbacks"] = resolved

    return trainer_cfg


class HalfPrecisionForAudio(HalfPrecision):
    """
    Adjusted Pytorch Lightning plugin for training with half precision.
    It avoids downcasting audio to bfloat16 when the mini-batch is a dict
    with 'audio' string in the keys corresponding to audio tensors.
    """

    @override
    def convert_input(self, data: Any) -> Any:
        """
        Converts input data to the appropriate precision format, preserving audio tensor precision.

        This method overrides the parent class implementation to avoid downcasting tensors
        with 'audio' in their dictionary keys. It processes input data recursively when
        encountering nested dictionaries.

        Args:
            data: The input data to convert (can be tensor, dict, or other types)

        Returns:
            The converted data with appropriate precision for each element
        """
        if not isinstance(data, dict):
            return super().convert_input(data)

        def _convert(v):
            if isinstance(v, dict):
                ans = {}
                for k, v in v.items():
                    if "audio" not in k or not torch.is_tensor(v):
                        v = _convert(v)
                    ans[k] = v
                return ans
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                return v.to(self._desired_input_dtype)
            return v  # any other type

        return _convert(data)
