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
import torch

from nemo.collections.llm.recipes.precision.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
    fp16_mixed,
    fp16_with_fp8_current_scaling_mixed,
    fp16_with_fp8_mixed,
    fp16_with_fp8_subchannel_scaling_mixed,
    fp16_with_mxfp8_mixed,
)


def test_bf16_mixed_config():
    config = bf16_mixed()
    assert config.precision == "bf16-mixed"
    assert config.params_dtype == torch.bfloat16
    assert config.pipeline_dtype == torch.bfloat16
    assert config.autocast_enabled is False
    assert config.grad_reduce_in_fp32 is True


def test_fp16_mixed_config():
    config = fp16_mixed()
    assert config.precision == "16-mixed"
    assert config.params_dtype == torch.half
    assert config.pipeline_dtype == torch.half
    assert config.autocast_enabled is False
    assert config.grad_reduce_in_fp32 is False


def test_bf16_with_fp8_mixed_config():
    config = bf16_with_fp8_mixed()
    # Check base bf16 settings
    assert config.precision == "bf16-mixed"
    assert config.params_dtype == torch.bfloat16
    assert config.pipeline_dtype == torch.bfloat16

    # Check FP8 specific settings
    assert config.fp8 == "hybrid"
    assert config.fp8_recipe == "delayed"
    assert config.fp8_margin == 0
    assert config.fp8_amax_history_len == 1024
    assert config.fp8_amax_compute_algo == "max"
    assert config.fp8_param_gather is True


def test_fp16_with_fp8_mixed_config():
    config = fp16_with_fp8_mixed()
    # Check base fp16 settings
    assert config.precision == "16-mixed"
    assert config.params_dtype == torch.half
    assert config.pipeline_dtype == torch.half

    # Check FP8 specific settings
    assert config.fp8 == "hybrid"
    assert config.fp8_recipe == "delayed"
    assert config.fp8_margin == 0
    assert config.fp8_amax_history_len == 1024
    assert config.fp8_amax_compute_algo == "max"
    assert config.fp8_param_gather is True


def test_bf16_with_mxfp8_mixed_config():
    config = bf16_with_mxfp8_mixed()
    assert config.precision == "bf16-mixed"
    assert config.params_dtype == torch.bfloat16
    assert config.pipeline_dtype == torch.bfloat16

    # Check FP8 specific settings
    assert config.fp8 == "hybrid"
    assert config.fp8_recipe == "mxfp8"
    assert config.fp8_param_gather is True
    assert config.reuse_grad_buf_for_mxfp8_param_ag is True


def test_fp16_with_mxfp8_mixed_config():
    config = fp16_with_mxfp8_mixed()
    assert config.precision == "16-mixed"
    assert config.params_dtype == torch.half
    assert config.pipeline_dtype == torch.half

    # Check FP8 specific settings
    assert config.fp8 == "hybrid"
    assert config.fp8_recipe == "mxfp8"
    assert config.fp8_param_gather is True
    assert config.reuse_grad_buf_for_mxfp8_param_ag is True


def test_bf16_with_fp8_current_scaling_mixed_config():
    config = bf16_with_fp8_current_scaling_mixed()
    assert config.precision == "bf16-mixed"
    assert config.params_dtype == torch.bfloat16
    assert config.pipeline_dtype == torch.bfloat16

    # Check FP8 specific settings
    assert config.fp8 == "hybrid"
    assert config.fp8_recipe == "tensorwise"
    assert config.fp8_param_gather is True
    assert config.first_last_layers_bf16 is True
    assert config.num_layers_at_start_in_bf16 == 1
    assert config.num_layers_at_end_in_bf16 == 1


def test_fp16_with_fp8_current_scaling_mixed_config():
    config = fp16_with_fp8_current_scaling_mixed()
    assert config.precision == "16-mixed"
    assert config.params_dtype == torch.half
    assert config.pipeline_dtype == torch.half
    # Check FP8 specific settings
    assert config.fp8 == "hybrid"
    assert config.fp8_recipe == "tensorwise"
    assert config.fp8_param_gather is True
    assert config.first_last_layers_bf16 is True
    assert config.num_layers_at_start_in_bf16 == 1
    assert config.num_layers_at_end_in_bf16 == 1


def test_bf16_with_fp8_subchannel_scaling_mixed_config():
    config = bf16_with_fp8_subchannel_scaling_mixed()
    assert config.precision == "bf16-mixed"
    assert config.params_dtype == torch.bfloat16
    assert config.pipeline_dtype == torch.bfloat16
    # Check FP8 specific settings
    assert config.fp8 == "hybrid"
    assert config.fp8_recipe == "blockwise"
    assert config.fp8_param_gather is False


def test_fp16_with_fp8_subchannel_scaling_mixed_config():
    config = fp16_with_fp8_subchannel_scaling_mixed()
    assert config.precision == "16-mixed"
    assert config.params_dtype == torch.half
    assert config.pipeline_dtype == torch.half

    # Check FP8 specific settings
    assert config.fp8 == "hybrid"
    assert config.fp8_recipe == "blockwise"
    assert config.fp8_param_gather is False
