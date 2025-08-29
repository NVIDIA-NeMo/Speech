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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Literal, Optional, TypeVar

import torch
from lightning.fabric.plugins.precision import MixedPrecision
from torch import nn
from torch.optim import Optimizer

from nemo.lightning.fabric.conversion import to_fabric
from nemo.lightning.pytorch.plugins.mixed_precision import (
    DtypeConfig,
    MegatronMixedPrecision,
    get_optim_config,
    update_config_with_dtype_overrides,
)
from nemo.utils import logging

if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig

AnyT = TypeVar("AnyT")
ConfigT = TypeVar("ConfigT", bound="ModelParallelConfig")


class FabricMegatronMixedPrecision(MixedPrecision):
    """Fabric plugin for mixed precision training with Megatron models.

    Handles precision conversions and mixed precision training settings
    in the Fabric training framework.
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
        fp8_recipe: Optional[str] = None,
        first_last_layers_bf16: bool = False,
        num_layers_at_start_in_bf16: int = 0,
        num_layers_at_end_in_bf16: int = 0,
        reuse_grad_buf_for_mxfp8_param_ag: bool = False,
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
            num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
            reuse_grad_buf_for_mxfp8_param_ag=reuse_grad_buf_for_mxfp8_param_ag,
            fp8_margin=fp8_margin,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_wgrad=fp8_wgrad,
            fp8_dot_product_attention=fp8_dot_product_attention,
            fp8_multi_head_attention=fp8_multi_head_attention,
            fp8_param=fp8_param_gather,
            fp8_param_gather=fp8_param_gather,
            # fp16 loss scale
            loss_scale=fp16_loss_scale,
            initial_loss_scale=fp16_initial_loss_scale,
            min_loss_scale=fp16_min_loss_scale,
            loss_scale_window=fp16_loss_scale_window,
            hysteresis=fp16_hysteresis,
        )
        if self.dtype_config.fp16:
            self.precision = "16-mixed"
        elif self.dtype_config.bf16:
            self.precision = "bf16-mixed"
        else:
            self.precision = "32-true"
        self.scaler = None

    def convert_input(self, data: AnyT) -> AnyT:
        """Convert model inputs (forward) to the floating point precision type of this plugin.

        Note: MegatronStrategy will take care of only doing this when:
            mpu.is_pipeline_first_stage()

        """
        return data

    def convert_output(self, data: AnyT) -> AnyT:
        """Convert outputs to the floating point precision type expected after model's forward.

        Note: MegatronStrategy will take care of only doing this when:
            mpu.is_pipeline_first_stage()

        """
        return data

    def convert_config(self, config: ConfigT) -> ConfigT:
        """Convert the config to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.
        """
        return update_config_with_dtype_overrides(self.dtype_config, config)

    def convert_module(self, module: nn.Module) -> nn.Module:
        """Convert the module parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        if not hasattr(module, "module"):
            return module

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

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """No explicit precision casting. Inputs are supposed to be manually casted."""
        try:
            yield
        finally:
            pass


@to_fabric.register(MegatronMixedPrecision)
def _convert_megatron_mixed_precision(plugin: MegatronMixedPrecision) -> FabricMegatronMixedPrecision:
    return FabricMegatronMixedPrecision(
        precision=plugin.precision,
    )
