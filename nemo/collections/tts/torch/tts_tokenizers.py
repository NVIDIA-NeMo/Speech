# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# TODO @xueyang: deprecate this file since no other places import modules from here anymore. However,
#  all checkpoints uploaded in ngc used this path. So it requires to update all ngc checkpoints path as well.
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
    BaseCharsTokenizer,
    BaseTokenizer,
    EnglishCharsTokenizer,
    EnglishPhonemesTokenizer,
    GermanCharsTokenizer,
    GermanPhonemesTokenizer,
    IPATokenizer,
)
