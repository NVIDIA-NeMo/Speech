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

from dataclasses import dataclass
from typing import Optional

from .multimodal_tokens import ImageToken, MultiModalToken, VideoToken


@dataclass
class DataConfig:
    media_type: str  # currently supported: image or video
    media_token: MultiModalToken
    conv_template: str = "v1"  # check `nemo/collections/multimodal/data/neva/conversation.py`
    reset_position_ids: bool = False  # Option to reset the position IDs in the dataset at an interval
    reset_attention_mask: bool = False  # Option to reset the attention mask from the dataset
    eod_mask_loss: bool = False  # Option to enable the EOD mask loss


@dataclass
class ImageDataConfig(DataConfig):
    media_type: str = "image"
    media_token: MultiModalToken = ImageToken
    image_folder: Optional[str] = None
    image_process_mode: str = 'pad'


@dataclass
class VideoDataConfig(DataConfig):
    media_type: str = "video"
    media_token: MultiModalToken = VideoToken
    splice_single_frame: Optional[str] = None
    # 'first', 'middle', 'last' will represent video as first / middle / last frame only, all other frames discarded.
    num_frames: int = 8  # Selects the number of frames to use from the video
    sep_token_between_frames: bool = False  # TODO: Allow usage of separator tokens between frames
    video_folder: Optional[str] = None
