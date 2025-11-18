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

# CLIP
from nemo.collections.vlm.clip.data import ClipMockDataModule
from nemo.collections.vlm.clip.model import CLIPConfigB32, CLIPConfigL14, CLIPModel

# Gemma3
from nemo.collections.vlm.gemma3vl.model.base import Gemma3VLConfig, Gemma3VLModel
from nemo.collections.vlm.gemma3vl.model.gemma3vl import Gemma3VLConfig4B, Gemma3VLConfig12B, Gemma3VLConfig27B
from nemo.collections.vlm.gemma3vl.model.vision import Gemma3VLMultimodalProjectorConfig, Gemma3VLVisionConfig

# LLAMA4
from nemo.collections.vlm.llama4.data import Llama4MockDataModule
from nemo.collections.vlm.llama4.model.base import Llama4OmniConfig, Llama4OmniModel
from nemo.collections.vlm.llama4.model.llama4_omni import Llama4MaverickExperts128Config, Llama4ScoutExperts16Config
from nemo.collections.vlm.llama4.model.vision import Llama4VisionConfig, Llama4ViTModel

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
from nemo.collections.vlm.qwen2vl.data import Qwen2VLDataConfig, Qwen2VLMockDataModule, Qwen2VLPreloadedDataModule
from nemo.collections.vlm.qwen2vl.model.base import (
    Qwen2VLConfig,
    Qwen2VLModel,
    Qwen2VLVisionConfig,
    Qwen25VLVisionConfig,
)
from nemo.collections.vlm.qwen2vl.model.qwen2vl import (
    Qwen2VLConfig2B,
    Qwen2VLConfig7B,
    Qwen25VLConfig3B,
    Qwen25VLConfig7B,
    Qwen25VLConfig32B,
    Qwen25VLConfig72B,
)

# RECIPES
from nemo.collections.vlm.recipes import *

# VISION
from nemo.collections.vlm.vision import (
    BaseCLIPViTModel,
    CLIPViTConfig,
    CLIPViTL_14_336_Config,
    CLIPViTModel,
    HFCLIPVisionConfig,
    InternViT_6B_448px_Config,
    InternViT_300M_448px_Config,
    InternViTModel,
    MultimodalProjectorConfig,
    SigLIPViT400M_14_384_Config,
    SigLIPViTModel,
)

__all__ = [
    "CLIPViTModel",
    "BaseCLIPViTModel",
    "NevaMockDataModule",
    "NevaPreloadedDataModule",
    "MLlamaMockDataModule",
    "MLlamaPreloadedDataModule",
    "Qwen2VLMockDataModule",
    "Qwen2VLPreloadedDataModule",
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
    "Qwen2VLConfig",
    "Qwen2VLConfig2B",
    "Qwen2VLConfig7B",
    "Qwen2VLVisionConfig",
    "Qwen2VLModel",
    "Qwen25VLConfig3B",
    "Qwen25VLConfig7B",
    "Qwen25VLConfig32B",
    "Qwen25VLConfig72B",
    "Qwen25VLVisionConfig",
    "Qwen2VLDataConfig",
    "Gemma3VLConfig",
    "Gemma3VLConfig4B",
    "Gemma3VLConfig12B",
    "Gemma3VLConfig27B",
    "Gemma3VLVisionConfig",
    "Gemma3VLMultimodalProjectorConfig",
    "Gemma3VLModel",
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
    "Llama4MockDataModule",
    "Llama4OmniConfig",
    "Llama4OmniModel",
    "Llama4VisionConfig",
    "Llama4ViTModel",
    "Llama4ScoutExperts16Config",
    "Llama4MaverickExperts128Config",
]

try:
    from nemo.collections.vlm.api import ptq  # noqa: F401

    __all__.append("ptq")
except ImportError as error:
    from nemo.utils import logging

    logging.warning(f"Failed to import nemo.collections.vlm.api: {error}")
