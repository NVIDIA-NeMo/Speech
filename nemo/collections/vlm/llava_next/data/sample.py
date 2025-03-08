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
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from megatron.core.packed_seq_params import PackedSeqParams

from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch, ImageTextSample


@dataclass
class LlavaNextTextSample(ImageTextSample):
    '''
    Sample type for LLaVA-Next, extending ImageTextSample to support tiled image data.

    This class adds additional attributes for handling high-resolution images processed as tiles,
    along with metadata about the tiled images.

    Attributes:
        num_media_tiles (int): The number of tiles used to represent the high-resolution image.
        image_sizes (torch.Tensor): A tensor representing the sizes of the tiled images.
        attention_mask (Optional[torch.Tensor]): An optional attention mask for the sample,
        used to determine which tokens or tiles are attended to during processing. Defaults to None.
    '''

    num_media_tiles: int = 0
    image_sizes: torch.tensor = field(default_factory=lambda: torch.tensor([]))
    attention_mask: Optional[torch.tensor] = None


@dataclass
class PackedLlavaNextTextSample(LlavaNextTextSample):
    """Sample type for packed image text sample"""

    __restore_key__: tuple[Union[str, int, tuple], ...] = ()
    position_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    packed_seq_params: PackedSeqParams = field(default_factory=lambda: PackedSeqParams())


@dataclass
class LlavaNextTextRawBatch(ImageTextRawBatch):
    """
    Batch type for raw LLaVA-Next samples, supporting tiled image data.

    This class aggregates multiple `LlavaNextTextSample` instances into a batch for processing.
    It includes attributes for managing tiled images and associated metadata for each sample in the batch.

    Attributes:
        num_media_tiles (List[int]): A list containing the number of tiles for each image in the batch.
        image_sizes (torch.Tensor): A tensor containing the sizes of all tiled images in the batch.
        attention_mask (Optional[torch.Tensor]): Attention mask. Defaults to None.
    """

    num_media_tiles: List[int] = field(default_factory=list)
    image_sizes: torch.tensor = field(default_factory=lambda: torch.tensor([]))
    attention_mask: Optional[torch.tensor] = None


@dataclass
class PackedLlavaNextTextRawBatch(LlavaNextTextRawBatch):
    """Sample type for image text raw batch"""

    position_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    packed_seq_params: PackedSeqParams = field(default_factory=lambda: PackedSeqParams())
