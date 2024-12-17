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

from nemo.collections.vlm.hf.data.hf_dataset import HFDatasetDataModule
from nemo.collections.vlm.hf.model.hf_auto_model_for_image_text_to_text import HFAutoModelForImageTextToText
from nemo.collections.vlm.llava_next.data import LlavaNextMockDataModule, LlavaNextTaskEncoder
from nemo.collections.vlm.llava_next.model.base import LlavaNextConfig
from nemo.collections.vlm.llava_next.model.llava_next import LlavaNextConfig7B, LlavaNextConfig13B, LlavaNextModel
from nemo.collections.vlm.mllama.data import MLlamaLazyDataModule, MLlamaMockDataModule
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
from nemo.collections.vlm.neva.data import (
    DataConfig,
    ImageDataConfig,
    ImageToken,
    MultiModalToken,
    NevaLazyDataModule,
    NevaMockDataModule,
    VideoDataConfig,
    VideoToken,
)
from nemo.collections.vlm.neva.model.base import (
    CLIPViTConfig,
    HFCLIPVisionConfig,
    MultimodalProjectorConfig,
    NevaConfig,
    NevaModel,
)
from nemo.collections.vlm.neva.model.llava import Llava15Config7B, Llava15Config13B, LlavaConfig, LlavaModel
from nemo.collections.vlm.neva.model.vit_config import CLIPViTL_14_336_Config, SigLIPViT400M_14_384_Config
from nemo.collections.vlm.peft import LoRA
from nemo.collections.vlm.recipes import *

__all__ = [
    "HFDatasetDataModule",
    "HFAutoModelForImageTextToText",
    "NevaMockDataModule",
    "NevaLazyDataModule",
    "MLlamaMockDataModule",
    "MLlamaLazyDataModule",
    "DataConfig",
    "ImageDataConfig",
    "VideoDataConfig",
    "MultiModalToken",
    "ImageToken",
    "VideoToken",
    "CLIPViTConfig",
    "HFCLIPVisionConfig",
    "CLIPViTL_14_336_Config",
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
    "LlavaNextConfig7B",
    "LlavaNextConfig13B",
    "LlavaNextModel",
    "LlavaNextMockDataModule",
    "LlavaNextTaskEncoder",
]
