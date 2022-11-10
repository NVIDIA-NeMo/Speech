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
from sdp.processors.modify_manifest.data_to_data import (
    InsIfASRInsertion,
    SubIfASRSubstitution,
    SubMakeLowercase,
    SubRegex,
    SubSubstringToSpace,
    SubSubstringToSubstring,
)

test_params_list = []

test_params_list.extend(
    [
        (SubSubstringToSpace, {"substrings": [","]}, {"text": "hello, nemo"}, {"text": "hello nemo"}),
        (SubSubstringToSpace, {"substrings": ["-"]}, {"text": "ice-cream"}, {"text": "ice cream"}),
    ]
)

test_params_list.extend(
    [
        (
            SubSubstringToSubstring,
            {"substring_pairs": {"nemon": "nemo"}},
            {"text": "using nemon"},
            {"text": "using nemo"},
        ),
    ]
)

test_params_list.extend(
    [
        (
            InsIfASRInsertion,
            {"insert_words": [" nemo", "nemo ", " nemo "]},
            {"text": "i love the toolkit", "pred_text": "i love the nemo toolkit"},
            {"text": "i love the nemo toolkit", "pred_text": "i love the nemo toolkit"},
        ),
        (
            InsIfASRInsertion,
            {"insert_words": [" nemo", "nemo ", " nemo "]},
            {"text": "i love the toolkit", "pred_text": "i love the new nemo toolkit"},
            {"text": "i love the toolkit", "pred_text": "i love the new nemo toolkit"},
        ),
    ]
)

test_params_list.extend(
    [
        (
            SubIfASRSubstitution,
            {"sub_words": {"nmo ": "nemo "}},
            {"text": "i love the nmo toolkit", "pred_text": "i love the nemo toolkit"},
            {"text": "i love the nemo toolkit", "pred_text": "i love the nemo toolkit"},
        ),
    ]
)

test_params_list.extend(
    [
        (
            SubIfASRSubstitution,
            {"sub_words": {"nmo ": "nemo "}},
            {"text": "i love the nmo toolkit", "pred_text": "i love the nemo toolkit"},
            {"text": "i love the nemo toolkit", "pred_text": "i love the nemo toolkit"},
        ),
    ]
)

test_params_list.extend(
    [(SubMakeLowercase, {}, {"text": "Hello Привет 123"}, {"text": "hello привет 123"},),]
)

test_params_list.extend(
    [(SubRegex, {"regex_to_sub": {"\s<.*>\s": " "}}, {"text": "hello <cough> world"}, {"text": "hello world"},),]
)


@pytest.mark.parametrize("test_class,class_kwargs,test_input,expected_output", test_params_list, ids=str)
def test_data_to_data(test_class, class_kwargs, test_input, expected_output):
    processor = test_class(**class_kwargs, output_manifest_file=None)

    output = processor.process_dataset_entry(test_input)[0].data

    assert output == expected_output
