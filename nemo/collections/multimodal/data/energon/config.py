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

# pylint: disable=C0115,C0116
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
from megatron.core.packed_seq_params import PackedSeqParams

from nemo.collections.multimodal.data.energon.conversation import BaseConversationTemplateConfig, LLaVATemplateConfig


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
class AudioToken(MultiModalToken):
    token_str: str = "<audio>"
    token_id: int = -300
    media_type: str = "audio"


@dataclass
class VideoToken(MultiModalToken):
    token_str: str = "<video>"
    token_id: int = -400
    media_type: str = "video"


@dataclass
class ImageTextSample:
    """Sample type for template formatted raw image text sample"""

    __key__: str = ''
    images: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    num_image_tiles: Optional[List[int]] = None


@dataclass
class PackedImageTextSample(ImageTextSample):
    """Sample type for packed image text sample"""

    __restore_key__: Tuple[Union[str, int, tuple], ...] = ()
    position_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    packed_seq_params: PackedSeqParams = field(default_factory=lambda: PackedSeqParams())


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
    num_image_tiles: Optional[List[int]] = None


@dataclass
class PackedImageTextRawBatch(ImageTextRawBatch):
    """Sample type for image text raw batch"""

    position_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    packed_seq_params: PackedSeqParams = field(default_factory=lambda: PackedSeqParams())


@dataclass
class MultiModalSampleConfig:
    image_token: ImageToken = field(default_factory=ImageToken)
    ignore_place_holder: int = -100
    conversation_template_config: BaseConversationTemplateConfig = field(default_factory=LLaVATemplateConfig)
    image_following_text: bool = True
