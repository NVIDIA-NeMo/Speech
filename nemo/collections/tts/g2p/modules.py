# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.tts.g2p.models.en_us_arpabet import EnglishG2p
from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p as IPAG2P

# TODO @xueyang: This file is kept for backward-compatibility purposes since all older NGC models (<= r1.16.0) used this
#  import path. We will remove this file soon; `IPAG2P` will be also renamed as `IpaG2p`. Please start using new import
#  path and the new `IpaG2p` name from r1.16.0.
from nemo.collections.tts.g2p.models.zh_cn_pinyin import ChineseG2p
