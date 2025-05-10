# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Generator, Literal, TypeVar, Union

import torch
from lightning.pytorch.plugins.precision import Precision
from torch.nn import Module
from torch.optim import Optimizer

from nemo.utils import logging

AnyT = TypeVar("AnyT")


def get_optim_config(optimizer: Optimizer):
    """Extract optimizer configurations from a Megatron optimizer.

    Args:
        optimizer: A torch.optim.Optimizer instance

    Yields:
        Optimizer configurations
    """
    extract_config = lambda x: x.config
    try:
        from megatron.core.optimizer import ChainedOptimizer

        if isinstance(optimizer.mcore_optimizer, ChainedOptimizer):
            opts = optimizer.mcore_optimizer.chained_optimizers
        else:
            opts = [optimizer.mcore_optimizer]
        yield from map(extract_config, opts)
    except:
        raise ValueError("Failed to extract optimizer config from module.")


@dataclass
class DtypeConfig:
    """Configuration class for mixed precision training settings.

    Contains settings for FP32/FP16/BF16 training, FP8 training.
    """

    fp32: bool = False
    fp16: bool = False
    bf16: bool = False
    params_dtype: torch.dtype = None
    pipeline_dtype: torch.dtype = None
    autocast_dtype: torch.dtype = None
    autocast_enabled: bool = False
    grad_reduce_in_fp32: bool = True
    # fp8 related
    fp8: str = None
    fp8_recipe: str = "delayed"
    first_last_layers_bf16: bool = False
    fp8_margin: int = 0
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"
    fp8_wgrad: bool = True
    fp8_dot_product_attention: bool = False
    fp8_multi_head_attention: bool = False
    fp8_param: bool = True
    fp8_param_gather: bool = True
    # FP16 Loss scaling
    loss_scale: float = (None,)
    initial_loss_scale: float = (None,)
    min_loss_scale: float = (None,)
    loss_scale_window: float = (None,)
    hysteresis: float = (None,)
    num_layers_at_start_in_bf16: int = 0
    num_layers_at_end_in_bf16: int = 0


class MegatronMixedPrecision(Precision):
    """Plugin for mixed precision training with Megatron models.

    Handles conversion of model parameters and inputs/outputs between different precisions,
    and manages mixed precision training settings.
    """

    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed", "32"],
        params_dtype: torch.dtype = None,
        pipeline_dtype: torch.dtype = None,
        autocast_dtype: torch.dtype = None,
        autocast_enabled: bool = False,
        grad_reduce_in_fp32: bool = True,
        # fp8 related,
        fp8: str = None,
        fp8_recipe: str = "delayed",  # "tensorwise", "delayed", "mxfp8" (for Blackwell only)
        first_last_layers_bf16: bool = False,
        fp8_margin: int = 0,
        fp8_amax_history_len: int = 1,
        fp8_amax_compute_algo: str = "most_recent",
        fp8_wgrad: bool = True,
        fp8_dot_product_attention: bool = False,
        fp8_multi_head_attention: bool = False,
        fp8_params: bool = None,
        fp8_param_gather: bool = None,
        fp16_loss_scale: float = None,
        fp16_initial_loss_scale: float = 4294967296,
        fp16_min_loss_scale: float = 1.0,
        fp16_loss_scale_window: int = 1000,
        fp16_hysteresis: int = 2,
        num_layers_at_start_in_bf16: int = 0,
        num_layers_at_end_in_bf16: int = 0,
    ) -> None:
        if fp8_params is not None:
            logging.warning(
                "fp8_params is deprecated and will be removed in a future release, use fp8_param_gather instead"
            )
            if fp8_param_gather is not None and fp8_param_gather != fp8_params:
                raise ValueError(
                    "Getting conflicting values for fp8_params and fp8_param_gather. Please only set fp8_param_gather."
                )
            fp8_param_gather = fp8_params
        elif fp8_param_gather is None:
            fp8_param_gather = False

        if isinstance(precision, int):
            precision = str(precision)

        dtype = torch.bfloat16 if precision in ['bf16', 'bf16-mixed'] else torch.float32
        self.dtype_config = DtypeConfig(
            fp32=precision in ['fp32', '32'],
            fp16=precision in ['fp16', 'fp16-mixed', '16', '16-mixed'],
            bf16=precision in ['bf16', 'bf16-mixed'],
            params_dtype=params_dtype or torch.float32,
            pipeline_dtype=pipeline_dtype or dtype,
            autocast_dtype=autocast_dtype or dtype,
            autocast_enabled=autocast_enabled,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            fp8=fp8,
            fp8_recipe=fp8_recipe,
            first_last_layers_bf16=first_last_layers_bf16,
            fp8_margin=fp8_margin,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_wgrad=fp8_wgrad,
            fp8_dot_product_attention=fp8_dot_product_attention,
            fp8_multi_head_attention=fp8_multi_head_attention,
            fp8_param=fp8_param_gather,
            fp8_param_gather=fp8_param_gather,
            num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
            # fp16 loss scale
            loss_scale=fp16_loss_scale,
            initial_loss_scale=fp16_initial_loss_scale,
            min_loss_scale=fp16_min_loss_scale,
            loss_scale_window=fp16_loss_scale_window,
            hysteresis=fp16_hysteresis,
        )
        super().__init__()
        if self.dtype_config.fp16:
            self.precision = "16-mixed"
        elif self.dtype_config.bf16:
            self.precision = "bf16-mixed"
        else:
            self.precision = "32-true"

    def convert_module(self, module: Module) -> Module:
        """Convert the module parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        from megatron.core.transformer.module import Float16Module
        from megatron.core.utils import get_model_config

        if self.dtype_config.fp16 or self.dtype_config.bf16:
            # Patch config options
            config = get_model_config(module.module)
            config.fp16 = self.dtype_config.fp16
            config.bf16 = self.dtype_config.bf16
            # Avoid rewrapping the module if it's already of type Float16Module
            if hasattr(module, "module"):
                if not isinstance(module.module, Float16Module):
                    module.module = Float16Module(config, module.module)
            elif not isinstance(module, Float16Module):
                module = Float16Module(config, module)

        return module

    def convert_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Convert the optimizer parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        for optim_config in get_optim_config(optimizer):
            assert optim_config.bf16 == self.dtype_config.bf16, "BF16 model/optim config mismatch"
            assert optim_config.fp16 == self.dtype_config.fp16, "FP16 model/optim config mismatch"
        return optimizer

    def convert_input(self, data: AnyT) -> AnyT:
        """Convert model inputs (forward) to the floating point precision type of this plugin.

        Note: MegatronStrategy will take care of only doing this when:
            parallel_state.is_pipeline_first_stage()

        """
        return data

    def convert_output(self, data: AnyT) -> AnyT:
        """Convert outputs to the floating point precision type expected after model's forward.

        Note: MegatronStrategy will take care of only doing this when:
            parallel_state.is_pipeline_last_stage()

        """
        return data

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """No explicit precision casting. Inputs are supposed to be manually casted."""
        try:
            yield
        finally:
            pass

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm=None,
    ) -> None:
        """Clip gradients. Raises error if clip_val > 0, otherwise it is a no-op.

        Args:
            optimizer: The optimizer to clip gradients for
            clip_val: The value to clip gradients to
            gradient_clip_algorithm: The algorithm to use for clipping

        Raises:
            ValueError: If clip_val > 0 since gradient clipping is handled by Mcore's optimizer
        """
        if clip_val > 0.0:
            raise ValueError(
                "Gradient clipping is handled in Mcore's optimizer. Use the clip_grad attribute in OptimizerConfig."
            )

    def clip_grad_by_value(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        """Clip gradients by value - it is a no-op.

        Args:
            optimizer: The optimizer to clip gradients for
            clip_val: The value to clip gradients to
        """
        return

    def clip_grad_by_norm(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        """Clip gradients by norm - it is a no-op.

        Args:
            optimizer: The optimizer to clip gradients for
            clip_val: The value to clip gradients to
        """
        return


def update_config_with_dtype_overrides(dtype_config, config):
    """Update a config object with dtype settings from dtype_config.

    Args:
        dtype_config: Source of dtype settings
        config: Config object to update

    Returns:
        Updated config object
    """
    if hasattr(config, "__io__"):
        config.__io__ = update_config_with_dtype_overrides(dtype_config, config.__io__)
    for field in fields(dtype_config):
        if not hasattr(config, field.name):
            continue
        # If we overwrote a value, log a debug message.
        old_val = getattr(config, field.name)
        new_val = getattr(dtype_config, field.name)
        if old_val != new_val:
            setattr(config, field.name, new_val)
            logging.debug(f"Overwrote {type(config).__name__}.{field.name}  {old_val} -> {new_val}")
    return config


__all__ = ["MegatronMixedPrecision"]
