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

from contextlib import nullcontext
from typing import Any, ContextManager, Mapping, Sequence

import hydra
import torch
from lightning.pytorch.plugins import HalfPrecision
from lightning.pytorch.plugins.precision.precision import Precision
from lightning_utilities import apply_to_collection
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from typing_extensions import override

from lightning.fabric.plugins.precision.utils import _convert_fp_tensor


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
    elif precision in ("fp16-automodel", "bf16-automodel"):
        trainer_cfg.pop("precision", None)
        trainer_cfg["plugins"] = [AutomodelPrecision(precision)]

    # Allows customizable strategies (eg ModelParallelStrategy) in YAML configs.
    if (strategy := trainer_cfg.get("strategy", None)) is not None and isinstance(strategy, Mapping):
        trainer_cfg["strategy"] = hydra.utils.instantiate(strategy)
        # Convert dict-valued nemo_automodel configs to proper dataclass instances.
        # This must happen AFTER Hydra instantiation because Hydra's recursive
        # processing chokes on dataclass fields with Union types (e.g. MoEParallelizerConfig).
        _resolve_automodel_configs(trainer_cfg["strategy"])

    # Allows to add custom callbacks (e.g. NsysCallback) from YAML config.
    if (cbs := trainer_cfg.get("callbacks", None)) is not None and isinstance(cbs, Sequence):
        resolved = []
        for cb in cbs:
            resolved.append(hydra.utils.instantiate(cb))
        trainer_cfg["callbacks"] = resolved

    return trainer_cfg


def _resolve_automodel_configs(strategy) -> None:
    """Convert plain dicts for ``distributed_config`` and ``moe_config`` to nemo_automodel objects.

    When :class:`AutomodelParallelStrategy` is specified in YAML, ``distributed_config``
    and ``moe_config`` arrive as plain dicts (Hydra passes them through as-is).
    This function converts them to proper dataclass instances on the
    already-instantiated strategy object.

    Does nothing if the strategy doesn't have these attributes or if they are
    already proper objects (not dicts).
    """
    if isinstance(getattr(strategy, '_distributed_config', None), Mapping):
        from nemo_automodel.components.distributed.config import FSDP2Config

        cfg = strategy._distributed_config
        # Instantiate any nested _target_ dicts (e.g. a custom mp_policy)
        resolved = {}
        for k, v in cfg.items():
            if isinstance(v, Mapping) and "_target_" in v:
                resolved[k] = hydra.utils.instantiate(v)
            else:
                resolved[k] = v
        strategy._distributed_config = FSDP2Config(**resolved)

    if isinstance(getattr(strategy, '_moe_config', None), Mapping):
        from nemo_automodel.components.moe.config import MoEParallelizerConfig

        strategy._moe_config = MoEParallelizerConfig(**strategy._moe_config)


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

        return _convert_audio_preserving(data, self._desired_input_dtype)


class AutomodelPrecision(Precision):
    """Precision plugin for Automodel-based training.

    Unlike Lightning's :class:`HalfPrecision`, this does **not** call
    :func:`torch.set_default_dtype` and does **not** use :func:`torch.autocast`.
    Parameter casting is assumed to have been handled by ``configure_model()``.
    It's recommended to use this class together with ``flashoptim`` optimizers.

    This ensures that Automodel's fp32 escaping (``Float32RMSNorm``, Gate
    softmax) and FlashOptim's master-weight correction terms are never
    silently downcast by a global dtype override.

    Opt in by setting ``trainer.precision: bf16-automodel`` in the YAML config.
    """

    precision: str = "bf16-automodel"

    def __init__(self, precision: str = "bf16-automodel") -> None:
        self.precision = precision
        self._desired_input_dtype = torch.bfloat16 if "bf16" in precision else torch.float16

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # Lightning calls convert_module AFTER configure_model().  When FSDP2
        # is in use, parameters are already DTensors and casting was done
        # inside configure_model() (before fully_shard) using .to(dtype)
        # — so this is a no-op.
        return module

    @override
    def forward_context(self) -> ContextManager:
        return nullcontext()

    @override
    def convert_input(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)

        return _convert_audio_preserving(data, self._desired_input_dtype)


def _convert_audio_preserving(data: dict, dtype: torch.dtype) -> dict:
    """Convert dict batch to *dtype*, keeping tensors whose key contains ``'audio'`` in fp32."""

    def _convert(v):
        if isinstance(v, dict):
            ans = {}
            for k, v in v.items():
                if "audio" not in k or not torch.is_tensor(v):
                    v = _convert(v)
                ans[k] = v
            return ans
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            return v.to(dtype)
        return v

    return _convert(data)

