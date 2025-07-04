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


import nemo_run as run
import torch

from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision


@run.cli.factory
def bf16_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using BF16.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for BF16 mixed precision training
    """
    return run.Config(
        MegatronMixedPrecision,
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )


@run.cli.factory
def fp16_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using FP16.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for FP16 mixed precision training
    """
    return run.Config(
        MegatronMixedPrecision,
        precision="16-mixed",
        params_dtype=torch.half,
        pipeline_dtype=torch.half,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )


def bf16_with_fp8_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using BF16 with FP8.

    Note: FP8 recipes are experimental and have not been tested for training convergence.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for BF16 with FP8 mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_recipe = "delayed"
    cfg.fp8_margin = 0
    cfg.fp8_amax_history_len = 1024
    cfg.fp8_amax_compute_algo = "max"
    cfg.fp8_param_gather = True
    return cfg


def fp16_with_fp8_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using FP16 with FP8.

    Note: FP8 recipes are experimental and have not been tested for training convergence.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for FP16 with FP8 mixed precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_recipe = "delayed"
    cfg.fp8_margin = 0
    cfg.fp8_amax_history_len = 1024
    cfg.fp8_amax_compute_algo = "max"
    cfg.fp8_param_gather = True
    return cfg


def bf16_with_mxfp8_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using BF16 with MXFP8.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for BF16 with MXFP8 mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_recipe = "mxfp8"
    cfg.fp8_param_gather = True
    return cfg


def fp16_with_mxfp8_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using FP16 with MXFP8.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for FP16 with MXFP8 mixed precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_recipe = "mxfp8"
    cfg.fp8_param_gather = True
    return cfg


def bf16_with_fp8_current_scaling_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using BF16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses BF16 in the first and last Transformer layers. The user
    can choose to disable the BF16 layers or apply BF16 to more Transformer layers.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for BF16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_recipe = "tensorwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 1
    cfg.num_layers_at_end_in_bf16 = 1
    cfg.fp8_param_gather = True
    return cfg


def nemotron_h_bf16_with_fp8_current_scaling_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using BF16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses BF16 in the first and last Transformer layers. The user
    can choose to disable the BF16 layers or apply BF16 to more Transformer layers.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for BF16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_recipe = "tensorwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 2
    cfg.num_layers_at_end_in_bf16 = 2
    cfg.fp8_param_gather = True
    return cfg


def fp16_with_fp8_current_scaling_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using FP16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses FP16 in the first and last Transformer layers. The user
    can choose to disable the FP16 layers or apply FP16 to more Transformer layers.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for FP16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_recipe = "tensorwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 1
    cfg.num_layers_at_end_in_bf16 = 1
    cfg.fp8_param_gather = True
    return cfg


def bf16_with_fp8_subchannel_scaling_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using BF16 with FP8
    NV Subchannel scaling. This recipe uses 128x128 blockwise quantization for weight and 1x128 blockwise
    quantization for activation.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for BF16 with FP8 subchannel scaling mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_recipe = "blockwise"
    cfg.fp8_param_gather = False
    return cfg


def fp16_with_fp8_subchannel_scaling_mixed() -> run.Config[MegatronMixedPrecision]:
    """Create a MegatronMixedPrecision plugin configuration for mixed precision training using FP16 with FP8
    NV Subchannel scaling. This recipe uses 128x128 blockwise quantization for weight and 1x128 blockwise
    quantization for activation.

    Returns:
        run.Config[MegatronMixedPrecision]: Configuration for FP16 with FP8 subchannel scaling mixed precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = 'hybrid'
    cfg.fp8_recipe = "blockwise"
    cfg.fp8_param_gather = False
    return cfg
