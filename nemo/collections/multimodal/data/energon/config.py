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

from dataclasses import dataclass, field
from typing import List
import torch
from nemo.collections.multimodal.data.energon.conversation import BaseConversationTemplateConfig


@dataclass
class MultiModalToken:
    token_str: str
    token_id: int
    media_type: str


@dataclass
class ImageToken(MultiModalToken):
    token_str: str = "<image>"
    token_id: int = -200
    media_type: str = "image"


@dataclass
class ImageTextSample:
    '''Sample type for template formatted raw image text sample'''

    __key__: str = ''
    images: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))


@dataclass
class ImageTextRawBatch:
    """Sample type for image text raw batch"""

    __keys__: List[str] = field(default_factory=list)
    #: Input images (N, C, H, W)
    images: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    #: Context string
    tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))


class LLaVATemplateConfig(BaseConversationTemplateConfig):
    """LLava specific template configuration which extends the base config"""

    pass


@dataclass
class MultiModalSampleConfig:
    image_token: ImageToken = ImageToken()
    ignore_place_holder: int = -100
    conversation_template_config: LLaVATemplateConfig = LLaVATemplateConfig()
    image_following_text: bool = True
