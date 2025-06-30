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

from nemo.collections.vlm.qwen2vl.model.base import (
    Qwen2VLConfig,
    Qwen2VLModel,
    Qwen2VLVisionConfig,
    Qwen25VLVisionConfig,
)
from nemo.collections.vlm.qwen2vl.model.qwen2vl import (
    Qwen2VLConfig2B,
    Qwen2VLConfig7B,
    Qwen2VLConfig72B,
    Qwen25VLConfig3B,
    Qwen25VLConfig7B,
    Qwen25VLConfig32B,
    Qwen25VLConfig72B,
)

__all__ = [
    "Qwen2VLVisionConfig",
    "Qwen2VLConfig",
    "Qwen2VLConfig2B",
    "Qwen2VLConfig7B",
    "Qwen2VLConfig72B",
    "Qwen2VLModel",
    "Qwen25VLVisionConfig",
    "Qwen25VLConfig3B",
    "Qwen25VLConfig7B",
    "Qwen25VLConfig32B",
    "Qwen25VLConfig72B",
]
