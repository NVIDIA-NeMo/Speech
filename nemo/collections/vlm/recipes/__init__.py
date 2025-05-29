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


from nemo.collections.vlm.recipes import (
    clip_b32,
    gemma3vl_4b,
    gemma3vl_12b,
    gemma3vl_27b,
    llava15_7b,
    llava15_13b,
    llava_next_7b,
    mllama_11b,
    mllama_90b,
)

__all__ = [
    "llava15_7b",
    "llava15_13b",
    "mllama_11b",
    "mllama_90b",
    "llava_next_7b",
    "clip_b32",
    "gemma3vl_4b",
    "gemma3vl_12b",
    "gemma3vl_27b",
]
