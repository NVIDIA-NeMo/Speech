# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

# The MIT License (MIT)
# Copyright (c) 2016 The-Orizon
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

# The detokenize function is based on : https://github.com/The-Orizon/nlputils/blob/master/detokenizer.py

import re
from typing import List

import jieba
import opencc
from pangu import spacing


class ChineseProcessor:
    """
    Tokenizer, Detokenizer and Normalizer utilities for Chinese.
    """

    def __init__(self):
        self.normalizer = opencc.OpenCC('t2s.json')

    def normalize(self, text: str) -> str:
        return self.normalizer.convert(text)

    def detokenize(self, text: List[str]) -> str:
        RE_WS_IN_FW = re.compile(
            r'([\u2018\u2019\u201c\u201d\u2e80-\u312f\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
            r'\uff00-\uffef])\s+(?=[\u2018\u2019\u201c\u201d\u2e80-\u312f\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fff'
            r'\uf900-\ufaff\uff00-\uffef])'
        )

        detokenize = lambda s: spacing(RE_WS_IN_FW.sub(r'\1', s)).strip()
        return detokenize(' '.join(text))

    def tokenize(self, text: str) -> str:
        text = jieba.cut(text)
        return ' '.join(text)
