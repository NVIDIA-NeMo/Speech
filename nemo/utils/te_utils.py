# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import functools
import importlib.metadata
from typing import Tuple

import packaging
import torch
from nemo.utils.import_utils import safe_import_from

# Check if Transformer Engine has quantized tensor classes
Float8Tensor, HAVE_TE_FLOAT8TENSOR = safe_import_from("transformer_engine.pytorch.float8_tensor", "Float8Tensor")
MXFP8Tensor, HAVE_TE_MXFP8TENSOR = safe_import_from("transformer_engine.pytorch.mxfp8_tensor", "MXFP8Tensor")


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor"""
    return HAVE_TE_FLOAT8TENSOR and isinstance(tensor, Float8Tensor)


def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)


@functools.lru_cache(maxsize=None)
def te_version() -> Tuple[int, ...]:
    """Transformer Engine version"""
    return packaging.version.Version(importlib.metadata.version("transformer-engine")).release
