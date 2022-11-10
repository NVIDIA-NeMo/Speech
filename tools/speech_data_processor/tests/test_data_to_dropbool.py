# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from sdp.processors.modify_manifest.data_to_dropbool import (
    DropASRErrorBeginningEnd,
    DropHighCER,
    DropHighLowCharrate,
    DropHighLowDuration,
    DropHighLowWordrate,
    DropHighWER,
    DropIfRegexInAttribute,
    DropIfSubstringInAttribute,
    DropIfSubstringInInsertion,
    DropIfTextIsEmpty,
    DropLowWordMatchRate,
    DropNonAlphabet,
)

test_params_list = []

test_params_list.extend(
    [
        (
            DropHighLowCharrate,
            {"high_charrate_threshold": 9.9, "low_charrate_threshold": 0},
            {"text": "0123456789", "duration": 1},
            True,
        ),
        (
            DropHighLowCharrate,
            {"high_charrate_threshold": 99, "low_charrate_threshold": 10.1},
            {"text": "0123456789", "duration": 1},
            True,
        ),
        (
            DropHighLowCharrate,
            {"high_charrate_threshold": 10.1, "low_charrate_threshold": 9.9},
            {"text": "0123456789", "duration": 1},
            False,
        ),
    ]
)

test_params_list.extend(
    [
        (
            DropHighLowWordrate,
            {"high_wordrate_threshold": 3.9, "low_wordrate_threshold": 0},
            {"text": "11 22 33 44", "duration": 1},
            True,
        ),
        (
            DropHighLowWordrate,
            {"high_wordrate_threshold": 99, "low_wordrate_threshold": 4.1},
            {"text": "11 22 33 44", "duration": 1},
            True,
        ),
        (
            DropHighLowWordrate,
            {"high_wordrate_threshold": 4.1, "low_wordrate_threshold": 3.9},
            {"text": "11 22 33 44", "duration": 1},
            False,
        ),
    ]
)

test_params_list.extend(
    [
        (DropHighLowDuration, {"high_duration_threshold": 3.9, "low_duration_threshold": 0}, {"duration": 4}, True,),
        (DropHighLowDuration, {"high_duration_threshold": 99, "low_duration_threshold": 4.1}, {"duration": 4}, True,),
        (
            DropHighLowDuration,
            {"high_duration_threshold": 4.1, "low_duration_threshold": 3.9},
            {"duration": 4},
            False,
        ),
    ]
)


test_params_list.extend(
    [
        (DropNonAlphabet, {"alphabet": " abc"}, {"text": "ab ba cab dac"}, True,),
        (DropNonAlphabet, {"alphabet": " abcd"}, {"text": "ab ba cab dac"}, False,),
    ]
)


test_params_list.extend(
    [
        (
            DropASRErrorBeginningEnd,
            {"beginning_error_char_threshold": 0, "end_error_char_threshold": 2},
            {"text": "2", "pred_text": "1 2 3"},
            True,
        ),
        (
            DropASRErrorBeginningEnd,
            {"beginning_error_char_threshold": 2, "end_error_char_threshold": 0},
            {"text": "2", "pred_text": "1 2 3"},
            True,
        ),
        (
            DropASRErrorBeginningEnd,
            {"beginning_error_char_threshold": 2, "end_error_char_threshold": 2},
            {"text": "2", "pred_text": "1 2 3"},
            False,
        ),
        (
            DropASRErrorBeginningEnd,
            {"beginning_error_char_threshold": 0, "end_error_char_threshold": 2},
            {"text": "sentence with some text here", "pred_text": "sentence with some text her"},
            False,
        ),
        (
            DropASRErrorBeginningEnd,
            {"beginning_error_char_threshold": 0, "end_error_char_threshold": 2},
            {
                "text": "sentence with some text here but actually more text was spoken",
                "pred_text": "sentence with some text her",
            },
            True,
        ),
    ]
)

test_params_list.extend(
    [
        (DropHighCER, {"cer_threshold": 9.9}, {"text": "0123456789", "pred_text": "012345678"}, True,),
        (DropHighCER, {"cer_threshold": 10.1}, {"text": "0123456789", "pred_text": "012345678"}, False,),
    ]
)

test_params_list.extend(
    [
        (DropHighWER, {"wer_threshold": 0}, {"text": "11  22", "pred_text": "11 22"}, False,),
        (DropHighWER, {"wer_threshold": 50.1}, {"text": "11 22", "pred_text": "11 22 33"}, False,),
        (DropHighWER, {"wer_threshold": 49.9}, {"text": "11 22", "pred_text": "11 22 33"}, True,),
    ]
)

test_params_list.extend(
    [
        (
            DropLowWordMatchRate,
            {"wmr_threshold": 50.1},
            {"text": "hello world i'm nemo", "pred_text": "hello world"},
            True,
        ),
        (
            DropLowWordMatchRate,
            {"wmr_threshold": 49.9},
            {"text": "hello world i'm nemo", "pred_text": "hello world"},
            False,
        ),
    ]
)

test_params_list.extend(
    [
        (
            DropIfSubstringInAttribute,
            {"attribute_to_substring": {"filepath": ["002"]}},
            {"text": "hello world", "filepath": "path/to/file/002.wav"},
            True,
        ),
        (
            DropIfSubstringInAttribute,
            {"attribute_to_substring": {"filepath": ["002"]}},
            {"text": "hello world", "filepath": "path/to/file/001.wav"},
            False,
        ),
    ]
)

test_params_list.extend(
    [
        (
            DropIfRegexInAttribute,
            {"attribute_to_regex": {"text": ["(\\D ){5,20}"]}},
            {"text": "h e l l o world"},
            True,
        ),
        (DropIfRegexInAttribute, {"attribute_to_regex": {"text": ["(\\D ){5,20}"]}}, {"text": "hello world"}, False,),
    ]
)

test_params_list.extend(
    [
        (
            DropIfSubstringInInsertion,
            {"substrings_in_insertion": ["might "]},
            {"text": "we miss certain words", "pred_text": "we might miss certain words"},
            True,
        ),
        (
            DropIfSubstringInInsertion,
            {"substrings_in_insertion": ["might "]},
            {"text": "we may certain words", "pred_text": "we might miss certain words"},
            False,
        ),
    ]
)

test_params_list.extend(
    [
        (DropIfTextIsEmpty, {}, {"text": "", "pred_text": "uuuu"}, True,),
        (DropIfTextIsEmpty, {}, {"text": "uhuh", "pred_text": "uuuu"}, False,),
    ]
)


@pytest.mark.parametrize("test_class,class_kwargs,test_input,expected_output", test_params_list, ids=str)
def test_data_to_data(test_class, class_kwargs, test_input, expected_output):
    processor = test_class(**class_kwargs, output_manifest_file=None)

    output = processor.process_dataset_entry(test_input)
    if output:
        output = output[0].data

    if expected_output:
        assert output is None
    else:
        assert output == test_input
