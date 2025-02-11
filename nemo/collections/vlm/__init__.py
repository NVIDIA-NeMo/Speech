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

# CLIP
from nemo.collections.vlm.clip.data import ClipMockDataModule
from nemo.collections.vlm.clip.model import CLIPConfigB32, CLIPConfigL14, CLIPModel

# HF
from nemo.collections.vlm.hf.data.hf_dataset import HFDatasetDataModule
from nemo.collections.vlm.hf.model.hf_auto_model_for_image_text_to_text import HFAutoModelForImageTextToText

# LLAVA_NEXT
from nemo.collections.vlm.llava_next.data import LlavaNextMockDataModule, LlavaNextTaskEncoder
from nemo.collections.vlm.llava_next.model.base import LlavaNextConfig
from nemo.collections.vlm.llava_next.model.llava_next import LlavaNextConfig7B, LlavaNextConfig13B, LlavaNextModel

# MLLAMA
from nemo.collections.vlm.mllama.data import MLlamaMockDataModule, MLlamaPreloadedDataModule
from nemo.collections.vlm.mllama.model.base import (
    CrossAttentionTextConfig,
    CrossAttentionVisionConfig,
    MLlamaModel,
    MLlamaModelConfig,
)
from nemo.collections.vlm.mllama.model.mllama import (
    MLlamaConfig11B,
    MLlamaConfig11BInstruct,
    MLlamaConfig90B,
    MLlamaConfig90BInstruct,
)

# NEVA
from nemo.collections.vlm.neva.data import (
    DataConfig,
    ImageDataConfig,
    ImageToken,
    MultiModalToken,
    NevaMockDataModule,
    NevaPreloadedDataModule,
    VideoDataConfig,
    VideoToken,
)
from nemo.collections.vlm.neva.model.base import NevaConfig, NevaModel
from nemo.collections.vlm.neva.model.llava import Llava15Config7B, Llava15Config13B, LlavaConfig, LlavaModel

# PEFT
from nemo.collections.vlm.peft import LoRA

# RECIPES
from nemo.collections.vlm.recipes import *

# VISION
from nemo.collections.vlm.vision import (
    CLIPViTConfig,
    CLIPViTL_14_336_Config,
    HFCLIPVisionConfig,
    InternViT_6B_448px_Config,
    InternViT_300M_448px_Config,
    InternViTModel,
    MultimodalProjectorConfig,
    SigLIPViT400M_14_384_Config,
    SigLIPViTModel,
)

__all__ = [
    "HFDatasetDataModule",
    "HFAutoModelForImageTextToText",
    "NevaMockDataModule",
    "NevaPreloadedDataModule",
    "MLlamaMockDataModule",
    "MLlamaPreloadedDataModule",
    "DataConfig",
    "ImageDataConfig",
    "VideoDataConfig",
    "MultiModalToken",
    "ImageToken",
    "VideoToken",
    "CLIPViTConfig",
    "HFCLIPVisionConfig",
    "CLIPViTL_14_336_Config",
    "SigLIPViTModel",
    "SigLIPViT400M_14_384_Config",
    "MultimodalProjectorConfig",
    "NevaConfig",
    "NevaModel",
    "LlavaConfig",
    "Llava15Config7B",
    "Llava15Config13B",
    "LlavaModel",
    "LlavaNextTaskEncoder",
    "MLlamaModel",
    "MLlamaModelConfig",
    "CrossAttentionTextConfig",
    "CrossAttentionVisionConfig",
    "MLlamaConfig11B",
    "MLlamaConfig11BInstruct",
    "MLlamaConfig90B",
    "MLlamaConfig90BInstruct",
    "mllama_11b",
    "mllama_90b",
    "llava_next_7b",
    "LlavaNextConfig",
    "LlavaNextConfig7B",
    "LlavaNextConfig13B",
    "LlavaNextModel",
    "LlavaNextMockDataModule",
    "InternViTModel",
    "InternViT_300M_448px_Config",
    "InternViT_6B_448px_Config",
    "CLIPModel",
    "LoRA",
    "CLIPConfigL14",
    "CLIPConfigB32",
    "ClipMockDataModule",
]
