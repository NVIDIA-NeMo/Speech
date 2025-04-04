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

from nemo.collections.vlm.vision.base import (
    BaseCLIPViTModel,
    CLIPViTConfig,
    HFCLIPVisionConfig,
    MultimodalProjectorConfig,
)
from nemo.collections.vlm.vision.clip_vit import CLIPViTL_14_336_Config, CLIPViTModel
from nemo.collections.vlm.vision.intern_vit import (
    InternViT_6B_448px_Config,
    InternViT_300M_448px_Config,
    InternViTModel,
)
from nemo.collections.vlm.vision.siglip_vit import SigLIPViT400M_14_384_Config, SigLIPViTModel

__all__ = [
    "MultimodalProjectorConfig",
    "HFCLIPVisionConfig",
    "CLIPViTConfig",
    "BaseCLIPViTModel",
    "CLIPViTL_14_336_Config",
    "SigLIPViTModel",
    "SigLIPViT400M_14_384_Config",
    "InternViTModel",
    "InternViT_300M_448px_Config",
    "InternViT_6B_448px_Config",
    "CLIPViTModel",
]
