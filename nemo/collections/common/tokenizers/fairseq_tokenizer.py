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

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

""" Code from
https://github.com/NVIDIA/DeepLearningExamples/blob/
master/PyTorch/Translation/Transformer/fairseq/tokenizer.py
"""

import re
import sys
import unicodedata
from collections import defaultdict

__all__ = ['get_unicode_categories', 'tokenize_en']


def get_unicode_categories():
    cats = defaultdict(list)
    for c in map(chr, range(sys.maxunicode + 1)):
        cats[unicodedata.category(c)].append(c)
    return cats


NUMERICS = ''.join(get_unicode_categories()['No'])


def tokenize_en(line):
    """Tokenize and normalize an English text line."""
    line = line.strip()
    line = ' ' + line + ' '
    # remove ASCII junk
    line = re.sub(r'\s+', ' ', line)
    line = re.sub(r'[\x00-\x1F]', '', line)
    # fix whitespaces
    line = re.sub(r'\ +', ' ', line)
    line = re.sub('^ ', '', line)
    line = re.sub(' $', '', line)
    # separate other special characters
    line = re.sub(r'([^\s\.\'\`\,\-\w]|[_' + NUMERICS + '])', r' \g<1> ', line)
    line = re.sub(r'(\w)\-(?=\w)', r'\g<1> @-@ ', line)

    # multidots stay together
    line = re.sub(r'\.([\.]+)', r' DOTMULTI\g<1>', line)
    while re.search(r'DOTMULTI\.', line):
        line = re.sub(r'DOTMULTI\.([^\.])', r'DOTDOTMULTI \g<1>', line)
        line = re.sub(r'DOTMULTI\.', r'DOTDOTMULTI', line)

    # separate out "," except if within numbers (5,300)
    line = re.sub(r'([\D])[,]', r'\g<1> , ', line)
    line = re.sub(r'[,]([\D])', r' , \g<1>', line)

    # separate "," after a number if it's the end of sentence
    line = re.sub(r'(\d)[,]$', r'\g<1> ,', line)

    # split contractions right
    line = re.sub(r'([\W\d])[\']([\W\d])', r'\g<1> \' \g<2>', line)
    line = re.sub(r'(\W)[\']([\w\D])', r'\g<1> \' \g<2>', line)
    line = re.sub(r'([\w\D])[\']([\W\d])', r'\g<1> \' \g<2>', line)
    line = re.sub(r'([\w\D])[\']([\w\D])', r'\g<1> \'\g<2>', line)
    # special case for "1990's"
    line = re.sub(r'([\W\d])[\']([s])', r'\g<1> \'\g<2>', line)

    # apply nonbreaking prefixes
    words = line.split()
    line = ''
    for i in range(len(words)):
        word = words[i]
        match = re.search(r'^(\S+)\.$', word)
        if match:
            pre = match.group(1)
            if i == len(words) - 1:
                """split last words independently as they are unlikely
                to be non-breaking prefixes"""
                word = pre + ' .'
            else:
                word = pre + ' .'

        word += ' '
        line += word

    # clean up extraneous spaces
    line = re.sub(' +', ' ', line)
    line = re.sub('^ ', '', line)
    line = re.sub(' $', '', line)

    # .' at end of sentence is missed
    line = re.sub(r'\.\' ?$', ' . \' ', line)

    # restore multi-dots
    while re.search('DOTDOTMULTI', line):
        line = re.sub('DOTDOTMULTI', 'DOTMULTI.', line)

    line = re.sub('DOTMULTI', '.', line)

    # escape special characters
    line = re.sub(r'\&', r'&amp;', line)
    line = re.sub(r'\|', r'&#124;', line)
    line = re.sub(r'\<', r'&lt;', line)
    line = re.sub(r'\>', r'&gt;', line)
    line = re.sub(r'\'', r'&apos;', line)
    line = re.sub(r'\"', r'&quot;', line)
    line = re.sub(r'\[', r'&#91;', line)
    line = re.sub(r'\]', r'&#93;', line)

    # ensure final line breaks
    # if line[-1] is not '\n':
    #     line += '\n'

    return line
