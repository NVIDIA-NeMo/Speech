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

from typing import Any, Dict

import modelopt.torch.quantization as mtq


def get_quant_cfg_choices() -> Dict[str, Dict[str, Any]]:
    """
    Retrieve a dictionary of modelopt quantization configuration choices.

    This function checks for the availability of specific quantization configurations defined in
    the modelopt.torch.quantization (mtq) module and returns a dictionary mapping short names to
    their corresponding configurations. The function is intended to work for different modelopt
    library versions that come with variable configuration choices.

    Returns:
        dict: A dictionary where keys are short names (e.g., "fp8") and values are the
            corresponding modelopt quantization configuration objects.
    """
    QUANT_CFG_NAMES = [
        ("int8", "INT8_DEFAULT_CFG"),
        ("int8_sq", "INT8_SMOOTHQUANT_CFG"),
        ("fp8", "FP8_DEFAULT_CFG"),
        ("block_fp8", "FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG"),
        ("int4_awq", "INT4_AWQ_CFG"),
        ("w4a8_awq", "W4A8_AWQ_BETA_CFG"),
        ("int4", "INT4_BLOCKWISE_WEIGHT_ONLY_CFG"),
        ("nvfp4", "NVFP4_DEFAULT_CFG"),
    ]

    QUANT_CFG_CHOICES = {}

    for short_name, full_name in QUANT_CFG_NAMES:
        if config := getattr(mtq, full_name, None):
            QUANT_CFG_CHOICES[short_name] = config

    return QUANT_CFG_CHOICES
