# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from utils.data_prep import get_y_and_boundary_info_for_utt

from nemo.collections.asr.models import ASRModel

EN_TEXT = "hi world | hey"

EN_QN_EXPECTED_TOKEN_INFO = [
    {'text': '<b>', 's_start': 0, 's_end': 0},
    {'text': 'h', 's_start': 1, 's_end': 1},
    {'text': '<b>', 's_start': 2, 's_end': 2},
    {'text': 'i', 's_start': 3, 's_end': 3},
    {'text': '<b>', 's_start': 4, 's_end': 4},
    {'text': '<space>', 's_start': 5, 's_end': 5},
    {'text': '<b>', 's_start': 6, 's_end': 6},
    {'text': 'w', 's_start': 7, 's_end': 7},
    {'text': '<b>', 's_start': 8, 's_end': 8},
    {'text': 'o', 's_start': 9, 's_end': 9},
    {'text': '<b>', 's_start': 10, 's_end': 10},
    {'text': 'r', 's_start': 11, 's_end': 11},
    {'text': '<b>', 's_start': 12, 's_end': 12},
    {'text': 'l', 's_start': 13, 's_end': 13},
    {'text': '<b>', 's_start': 14, 's_end': 14},
    {'text': 'd', 's_start': 15, 's_end': 15},
    {'text': '<b>', 's_start': 16, 's_end': 16},
    {'text': '<space>', 's_start': 17, 's_end': 17},
    {'text': '<b>', 's_start': 18, 's_end': 18},
    {'text': 'h', 's_start': 19, 's_end': 19},
    {'text': '<b>', 's_start': 20, 's_end': 20},
    {'text': 'e', 's_start': 21, 's_end': 21},
    {'text': '<b>', 's_start': 22, 's_end': 22},
    {'text': 'y', 's_start': 23, 's_end': 23},
    {'text': '<b>', 's_start': 24, 's_end': 24},
]

EN_QN_EXPECTED_WORD_INFO = [
    {'text': 'hi', 's_start': 1, 's_end': 3},
    {'text': 'world', 's_start': 7, 's_end': 15},
    {'text': 'hey', 's_start': 19, 's_end': 23},
]

EN_QN_EXPECTED_SEGMENT_INFO = [
    {'text': 'hi world', 's_start': 1, 's_end': 15},
    {'text': 'hey', 's_start': 19, 's_end': 23},
]

EN_CN_EXPECTED_TOKEN_INFO = [
    {'text': '<b>', 's_start': 0, 's_end': 0},
    {'text': '▁hi', 's_start': 1, 's_end': 1},
    {'text': '<b>', 's_start': 2, 's_end': 2},
    {'text': '▁world', 's_start': 3, 's_end': 3},
    {'text': '<b>', 's_start': 4, 's_end': 4},
    {'text': '▁he', 's_start': 5, 's_end': 5},
    {'text': '<b>', 's_start': 6, 's_end': 6},
    {'text': 'y', 's_start': 7, 's_end': 7},
    {'text': '<b>', 's_start': 8, 's_end': 8},
]

EN_CN_EXPECTED_WORD_INFO = [
    {'text': 'hi', 's_start': 1, 's_end': 1},
    {'text': 'world', 's_start': 3, 's_end': 3},
    {'text': 'hey', 's_start': 5, 's_end': 7},
]

EN_CN_EXPECTED_SEGMENT_INFO = [
    {'text': 'hi world', 's_start': 1, 's_end': 3},
    {'text': 'hey', 's_start': 5, 's_end': 7},
]


ZH_TEXT = "人工 智能|技术"

ZH_EXPECTED_TOKEN_INFO = [
    {'text': '<b>', 's_start': 0, 's_end': 0},
    {'text': '人', 's_start': 1, 's_end': 1},
    {'text': '<b>', 's_start': 2, 's_end': 2},
    {'text': '工', 's_start': 3, 's_end': 3},
    {'text': '<b>', 's_start': 4, 's_end': 4},
    {'text': '<space>', 's_start': 5, 's_end': 5},
    {'text': '<b>', 's_start': 6, 's_end': 6},
    {'text': '智', 's_start': 7, 's_end': 7},
    {'text': '<b>', 's_start': 8, 's_end': 8},
    {'text': '能', 's_start': 9, 's_end': 9},
    {'text': '<b>', 's_start': 10, 's_end': 10},
    {'text': '<space>', 's_start': 11, 's_end': 11},
    {'text': '<b>', 's_start': 12, 's_end': 12},
    {'text': '技', 's_start': 13, 's_end': 13},
    {'text': '<b>', 's_start': 14, 's_end': 14},
    {'text': '术', 's_start': 15, 's_end': 15},
    {'text': '<b>', 's_start': 16, 's_end': 16},
]

ZH_EXPECTED_WORD_INFO = [
    {'text': '人工', 's_start': 1, 's_end': 3},
    {'text': '智能', 's_start': 7, 's_end': 9},
    {'text': '技术', 's_start': 13, 's_end': 15},
]

ZH_EXPECTED_SEGMENT_INFO = [
    {'text': '人工 智能', 's_start': 1, 's_end': 9},
    {'text': '技术', 's_start': 13, 's_end': 15},
]


@pytest.mark.parametrize(
    "text,model_pretrained_name,separator,expected_token_info",
    [
        (EN_TEXT, "stt_en_quartznet15x5", "|", EN_QN_EXPECTED_TOKEN_INFO),
        (EN_TEXT, "stt_en_citrinet_256_gamma_0_25", "|", EN_CN_EXPECTED_TOKEN_INFO),
        (ZH_TEXT, "stt_zh_citrinet_512", "|", ZH_EXPECTED_TOKEN_INFO),
    ],
)
def test_token_info(text, model_pretrained_name, separator, expected_token_info):
    model = ASRModel.from_pretrained(model_pretrained_name)
    _, token_info, *_ = get_y_and_boundary_info_for_utt(text, model, separator)
    assert token_info == expected_token_info


@pytest.mark.parametrize(
    "text,model_pretrained_name,separator,expected_word_info",
    [
        (EN_TEXT, "stt_en_quartznet15x5", "|", EN_QN_EXPECTED_WORD_INFO),
        (EN_TEXT, "stt_en_citrinet_256_gamma_0_25", "|", EN_CN_EXPECTED_WORD_INFO),
        (ZH_TEXT, "stt_zh_citrinet_512", "|", ZH_EXPECTED_WORD_INFO),
    ],
)
def test_word_info(text, model_pretrained_name, separator, expected_word_info):
    model = ASRModel.from_pretrained(model_pretrained_name)
    _, _, word_info, _ = get_y_and_boundary_info_for_utt(text, model, separator)
    assert word_info == expected_word_info


@pytest.mark.parametrize(
    "text,model_pretrained_name,separator,expected_segment_info",
    [
        (EN_TEXT, "stt_en_quartznet15x5", "|", EN_QN_EXPECTED_SEGMENT_INFO),
        (EN_TEXT, "stt_en_citrinet_256_gamma_0_25", "|", EN_CN_EXPECTED_SEGMENT_INFO),
        (ZH_TEXT, "stt_zh_citrinet_512", "|", ZH_EXPECTED_SEGMENT_INFO),
    ],
)
def test_segment_info(text, model_pretrained_name, separator, expected_segment_info):
    model = ASRModel.from_pretrained(model_pretrained_name)
    *_, segment_info = get_y_and_boundary_info_for_utt(text, model, separator)
    assert segment_info == expected_segment_info
