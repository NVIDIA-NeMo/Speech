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
from utils.make_output_files import add_t_start_end_to_boundary_info

ALIGNMENT = [
    1,
    1,
    3,
    3,
    4,
    5,
    7,
    7,
    9,
    10,
    11,
    12,
    13,
    15,
    17,
    17,
    19,
    21,
    23,
    23,
]

INPUT_TOKEN_INFO = [
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

EXPECTED_OUTPUT_TOKEN_INFO = [
    {'text': 'h', 's_start': 1, 's_end': 1, 't_start': 0, 't_end': 1},
    {'text': 'i', 's_start': 3, 's_end': 3, 't_start': 2, 't_end': 3},
    {'text': '<b>', 's_start': 4, 's_end': 4, 't_start': 4, 't_end': 4},
    {'text': '<space>', 's_start': 5, 's_end': 5, 't_start': 5, 't_end': 5},
    {'text': 'w', 's_start': 7, 's_end': 7, 't_start': 6, 't_end': 7},
    {'text': 'o', 's_start': 9, 's_end': 9, 't_start': 8, 't_end': 8},
    {'text': '<b>', 's_start': 10, 's_end': 10, 't_start': 9, 't_end': 9},
    {'text': 'r', 's_start': 11, 's_end': 11, 't_start': 10, 't_end': 10},
    {'text': '<b>', 's_start': 12, 's_end': 12, 't_start': 11, 't_end': 11},
    {'text': 'l', 's_start': 13, 's_end': 13, 't_start': 12, 't_end': 12},
    {'text': 'd', 's_start': 15, 's_end': 15, 't_start': 13, 't_end': 13},
    {'text': '<space>', 's_start': 17, 's_end': 17, 't_start': 14, 't_end': 15},
    {'text': 'h', 's_start': 19, 's_end': 19, 't_start': 16, 't_end': 16},
    {'text': 'e', 's_start': 21, 's_end': 21, 't_start': 17, 't_end': 17},
    {'text': 'y', 's_start': 23, 's_end': 23, 't_start': 18, 't_end': 19},
]


INPUT_WORD_INFO = [
    {'text': 'hi', 's_start': 1, 's_end': 3},
    {'text': 'world', 's_start': 7, 's_end': 15},
    {'text': 'hey', 's_start': 19, 's_end': 23},
]

EXPECTED_OUTPUT_WORD_INFO = [
    {'text': 'hi', 's_start': 1, 's_end': 3, 't_start': 0, 't_end': 3},
    {'text': 'world', 's_start': 7, 's_end': 15, 't_start': 6, 't_end': 13},
    {'text': 'hey', 's_start': 19, 's_end': 23, 't_start': 16, 't_end': 19},
]

INPUT_SEGMENT_INFO = [
    {'text': 'hi world', 's_start': 1, 's_end': 15},
    {'text': 'hey', 's_start': 19, 's_end': 23},
]

EXPECTED_OUTPUT_SEGMENT_INFO = [
    {'text': 'hi world', 's_start': 1, 's_end': 15, 't_start': 0, 't_end': 13},
    {'text': 'hey', 's_start': 19, 's_end': 23, 't_start': 16, 't_end': 19},
]


@pytest.mark.parametrize(
    "input_boundary_info_utt,alignment_utt,expected_output_boundary_info_utt",
    [
        (INPUT_TOKEN_INFO, ALIGNMENT, EXPECTED_OUTPUT_TOKEN_INFO),
        (INPUT_WORD_INFO, ALIGNMENT, EXPECTED_OUTPUT_WORD_INFO),
        (INPUT_SEGMENT_INFO, ALIGNMENT, EXPECTED_OUTPUT_SEGMENT_INFO),
    ],
)
def test_add_t_start_end_to_boundary_info(input_boundary_info_utt, alignment_utt, expected_output_boundary_info_utt):
    output_boundary_info_utt = add_t_start_end_to_boundary_info(input_boundary_info_utt, alignment_utt)
    assert output_boundary_info_utt == expected_output_boundary_info_utt
