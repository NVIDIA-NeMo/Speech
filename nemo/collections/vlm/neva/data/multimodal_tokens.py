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
from typing import Callable, Optional


@dataclass
class MultiModalToken:
    """
    Base class for multimodal tokens representing different media types.
    """

    token_str: str
    token_index: int
    media_type: str
    use_start_end: bool
    encoder_fn: Optional[Callable] = None


@dataclass
class ImageToken(MultiModalToken):
    token_str: str = "<image>"
    token_index: int = -200
    media_type: str = "image"
    use_start_end: bool = False


@dataclass
class VideoToken(MultiModalToken):
    token_str: str = "<video>"
    token_index: int = -300
    media_type: str = "video"
    use_start_end: bool = False


# Constants for token indexing and special token mapping
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = ImageToken.token_index
VIDEO_TOKEN_INDEX = VideoToken.token_index
SPECIAL_TOKEN_MAP = [(ImageToken.token_str, ImageToken.token_index), (VideoToken.token_str, VideoToken.token_index)]
