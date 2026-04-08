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
# pylint: disable=C0116

from contextlib import nullcontext
from typing import Any, ContextManager, Mapping, Sequence

import hydra
import torch
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor
from lightning.pytorch.plugins import HalfPrecision
from lightning.pytorch.plugins.precision.precision import Precision
from lightning_utilities import apply_to_collection
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from typing_extensions import override


_FLASH_PRECISION_ALIASES = {
    "fp16-flash": "fp16-flash",
    "bf16-flash": "bf16-flash",
    # Temporary backward-compatible aliases retained during migration.
    "fp16-automodel": "fp16-flash",
    "bf16-automodel": "bf16-flash",
}


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

    # Avoids downcasting 'audio' tensors in half precision setups and enables
    # the specialized flash precision plugin without mutating global dtype state.
    precision = trainer_cfg.get("precision")
    if precision in ("fp16-true", "bf16-true"):
        trainer_cfg.pop("precision", None)
        trainer_cfg["plugins"] = [HalfPrecisionForAudio(precision)]
    elif (flash_precision := _normalize_flash_precision(precision)) is not None:
        trainer_cfg.pop("precision", None)
        trainer_cfg["plugins"] = [FlashPrecision(flash_precision)]

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


class FlashPrecision(Precision):
    """Precision plugin for flash optimizer training.

    Unlike Lightning's :class:`HalfPrecision`, this does **not** call
    :func:`torch.set_default_dtype` and does **not** use :func:`torch.autocast`.
    It's recommended to use this class together with ``flashoptim`` optimizers.

    This ensures that model-specific fp32 escapes (for example custom norms or
    gating layers) and FlashOptim's master-weight correction terms are never
    silently downcast by a global dtype override.

    Note: it won't downcast your model's weights to half precision if some of them
    have already been downcast (manual downcasting) or if the model is using DTensor
    (in that case you have to downcast them yourself, typically in configure_model()).

    Opt in by setting ``trainer.precision: bf16-flash`` in the YAML config.
    """

    precision: str = "bf16-flash"

    def __init__(self, precision: str = "bf16-flash") -> None:
        normalized = _normalize_flash_precision(precision) or precision
        self.precision = normalized
        self._desired_input_dtype = torch.bfloat16 if "bf16" in normalized else torch.float16

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # Some models manage dtype explicitly inside configure_model() and may
        # intentionally keep select parameters in fp32. Only cast modules that
        # are still entirely plain fp32 tensors.
        if _should_skip_flash_module_conversion(module):
            return module

        from flashoptim import cast_model

        cast_model(module, dtype=self._desired_input_dtype)
        return module

    @override
    def forward_context(self) -> ContextManager:
        return nullcontext()

    @override
    def convert_input(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return apply_to_collection(
                data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype
            )

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


def _normalize_flash_precision(precision: str | None) -> str | None:
    if precision is None:
        return None

    return _FLASH_PRECISION_ALIASES.get(precision)


def _should_skip_flash_module_conversion(module: torch.nn.Module) -> bool:
    """Return True when a module should keep its existing parameter dtype policy."""

    saw_fp_tensor = False
    for tensor in _iter_module_tensors(module):
        if not torch.is_floating_point(tensor):
            continue

        saw_fp_tensor = True
        if _is_distributed_tensor(tensor):
            return True
        if tensor.dtype != torch.float32:
            return True

    return not saw_fp_tensor


def _iter_module_tensors(module: torch.nn.Module):
    yield from module.parameters()
    yield from module.buffers()


def _is_distributed_tensor(tensor: Tensor) -> bool:
    if hasattr(tensor, "device_mesh") or hasattr(tensor, "placements"):
        return True

    try:
        from torch.distributed.tensor import DTensor
    except (ImportError, ModuleNotFoundError):
        return False

    return isinstance(tensor, DTensor)
