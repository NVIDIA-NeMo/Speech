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

import copy

import pytest
from nemo.collections.asr.parts.utils.aligner_utils import Segment, Token, Utterance, Word, add_t_start_end_to_utt_obj

OUTPUT_TIMESTEP_DURATION = 0.04

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

EXPECTED_OUTPUT_UTTERANCE = Utterance(
    text='hi world | hey',
    token_ids_with_blanks=[
        28,
        8,
        28,
        9,
        28,
        0,
        28,
        23,
        28,
        15,
        28,
        18,
        28,
        12,
        28,
        4,
        28,
        0,
        28,
        8,
        28,
        5,
        28,
        25,
        28,
    ],
    segments_and_tokens=[
        Token(text='<b>', text_cased='<b>', s_start=0, s_end=0, t_start=-1, t_end=-1),
        Segment(
            text="hi world",
            s_start=1,
            s_end=15,
            t_start=0 * OUTPUT_TIMESTEP_DURATION,
            t_end=14 * OUTPUT_TIMESTEP_DURATION,
            words_and_tokens=[
                Word(
                    text="hi",
                    s_start=1,
                    s_end=3,
                    t_start=0 * OUTPUT_TIMESTEP_DURATION,
                    t_end=4 * OUTPUT_TIMESTEP_DURATION,
                    tokens=[
                        Token(
                            text='h',
                            text_cased='h',
                            s_start=1,
                            s_end=1,
                            t_start=0 * OUTPUT_TIMESTEP_DURATION,
                            t_end=2 * OUTPUT_TIMESTEP_DURATION,
                        ),
                        Token(text='<b>', text_cased='<b>', s_start=2, s_end=2, t_start=-1, t_end=-1),
                        Token(
                            text='i',
                            text_cased='i',
                            s_start=3,
                            s_end=3,
                            t_start=2 * OUTPUT_TIMESTEP_DURATION,
                            t_end=4 * OUTPUT_TIMESTEP_DURATION,
                        ),
                    ],
                ),
                Token(
                    text='<b>',
                    text_cased='<b>',
                    s_start=4,
                    s_end=4,
                    t_start=4 * OUTPUT_TIMESTEP_DURATION,
                    t_end=5 * OUTPUT_TIMESTEP_DURATION,
                ),
                Token(
                    text='<space>',
                    text_cased='<space>',
                    s_start=5,
                    s_end=5,
                    t_start=5 * OUTPUT_TIMESTEP_DURATION,
                    t_end=6 * OUTPUT_TIMESTEP_DURATION,
                ),
                Token(text='<b>', text_cased='<b>', s_start=6, s_end=6, t_start=-1, t_end=-1),
                Word(
                    text="world",
                    s_start=7,
                    s_end=15,
                    t_start=6 * OUTPUT_TIMESTEP_DURATION,
                    t_end=14 * OUTPUT_TIMESTEP_DURATION,
                    tokens=[
                        Token(
                            text='w',
                            text_cased='w',
                            s_start=7,
                            s_end=7,
                            t_start=6 * OUTPUT_TIMESTEP_DURATION,
                            t_end=8 * OUTPUT_TIMESTEP_DURATION,
                        ),
                        Token(text='<b>', text_cased='<b>', s_start=8, s_end=8, t_start=-1, t_end=-1),
                        Token(
                            text='o',
                            text_cased='o',
                            s_start=9,
                            s_end=9,
                            t_start=8 * OUTPUT_TIMESTEP_DURATION,
                            t_end=9 * OUTPUT_TIMESTEP_DURATION,
                        ),
                        Token(
                            text='<b>',
                            text_cased='<b>',
                            s_start=10,
                            s_end=10,
                            t_start=9 * OUTPUT_TIMESTEP_DURATION,
                            t_end=10 * OUTPUT_TIMESTEP_DURATION,
                        ),
                        Token(
                            text='r',
                            text_cased='r',
                            s_start=11,
                            s_end=11,
                            t_start=10 * OUTPUT_TIMESTEP_DURATION,
                            t_end=11 * OUTPUT_TIMESTEP_DURATION,
                        ),
                        Token(
                            text='<b>',
                            text_cased='<b>',
                            s_start=12,
                            s_end=12,
                            t_start=11 * OUTPUT_TIMESTEP_DURATION,
                            t_end=12 * OUTPUT_TIMESTEP_DURATION,
                        ),
                        Token(
                            text='l',
                            text_cased='l',
                            s_start=13,
                            s_end=13,
                            t_start=12 * OUTPUT_TIMESTEP_DURATION,
                            t_end=13 * OUTPUT_TIMESTEP_DURATION,
                        ),
                        Token(text='<b>', text_cased='<b>', s_start=14, s_end=14, t_start=-1, t_end=-1),
                        Token(
                            text='d',
                            text_cased='d',
                            s_start=15,
                            s_end=15,
                            t_start=13 * OUTPUT_TIMESTEP_DURATION,
                            t_end=14 * OUTPUT_TIMESTEP_DURATION,
                        ),
                    ],
                ),
            ],
        ),
        Token(text='<b>', text_cased='<b>', s_start=16, s_end=16, t_start=-1, t_end=-1),
        Token(
            text='<space>',
            text_cased='<space>',
            s_start=17,
            s_end=17,
            t_start=14 * OUTPUT_TIMESTEP_DURATION,
            t_end=16 * OUTPUT_TIMESTEP_DURATION,
        ),
        Token(text='<b>', text_cased='<b>', s_start=18, s_end=18, t_start=-1, t_end=-1),
        Segment(
            text="hey",
            s_start=19,
            s_end=23,
            t_start=16 * OUTPUT_TIMESTEP_DURATION,
            t_end=20 * OUTPUT_TIMESTEP_DURATION,
            words_and_tokens=[
                Word(
                    text="hey",
                    s_start=19,
                    s_end=23,
                    t_start=16 * OUTPUT_TIMESTEP_DURATION,
                    t_end=20 * OUTPUT_TIMESTEP_DURATION,
                    tokens=[
                        Token(
                            text='h',
                            text_cased='h',
                            s_start=19,
                            s_end=19,
                            t_start=16 * OUTPUT_TIMESTEP_DURATION,
                            t_end=17 * OUTPUT_TIMESTEP_DURATION,
                        ),
                        Token(text='<b>', text_cased='<b>', s_start=20, s_end=20, t_start=-1, t_end=-1),
                        Token(
                            text='e',
                            text_cased='e',
                            s_start=21,
                            s_end=21,
                            t_start=17 * OUTPUT_TIMESTEP_DURATION,
                            t_end=18 * OUTPUT_TIMESTEP_DURATION,
                        ),
                        Token(text='<b>', text_cased='<b>', s_start=22, s_end=22, t_start=-1, t_end=-1),
                        Token(
                            text='y',
                            text_cased='y',
                            s_start=23,
                            s_end=23,
                            t_start=18 * OUTPUT_TIMESTEP_DURATION,
                            t_end=20 * OUTPUT_TIMESTEP_DURATION,
                        ),
                    ],
                )
            ],
        ),
        Token(text='<b>', text_cased='<b>', s_start=24, s_end=24, t_start=-1, t_end=-1),
    ],
)


@pytest.mark.parametrize(
    "alignment,expected_output_utterance, output_timestep_duration",
    [
        (ALIGNMENT, EXPECTED_OUTPUT_UTTERANCE, OUTPUT_TIMESTEP_DURATION),
    ],
)
def test_add_t_start_end_to_utt_obj(alignment, expected_output_utterance, output_timestep_duration):
    input_utterance = copy.deepcopy(expected_output_utterance)

    # set all t_start and t_end to None in input_utterance
    for segment_or_token in input_utterance.segments_and_tokens:
        if type(segment_or_token) is Segment:
            segment = segment_or_token
            segment.t_start = None
            segment.t_end = None

            for word_or_token in segment.words_and_tokens:
                if type(word_or_token) is Word:
                    word = word_or_token
                    word.t_start = None
                    word.t_end = None

                    for token in word.tokens:
                        token.t_start = None
                        token.t_end = None
                else:
                    token = word_or_token
                    token.t_start = None
                    token.t_end = None

        else:
            token = segment_or_token
            token.t_start = None
            token.t_end = None

    output_utterance = add_t_start_end_to_utt_obj(input_utterance, alignment, output_timestep_duration)
    assert output_utterance == expected_output_utterance
