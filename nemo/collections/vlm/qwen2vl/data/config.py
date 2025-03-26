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


@dataclass
class Qwen2VLDataConfig:
    # pylint: disable=C0115,C0116
    conv_template: str = "qwen2vl"  # check `nemo/collections/vlm/qwen2vl/data/conversation.py`
    reset_position_ids: bool = False  # Option to reset the position IDs in the dataset at an interval
    reset_attention_mask: bool = False  # Option to reset the attention mask from the dataset
    eod_mask_loss: bool = False  # Option to enable the EOD mask loss
    image_folder: Optional[str] = None
    image_process_mode: str = 'pad'
    video_folder: Optional[str] = None
