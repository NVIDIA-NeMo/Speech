# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import re

import pytest
import torch

from nemo.collections.asr.inference.utils.pipeline_utils import (
    check_existance_of_required_attributes,
    drop_trailing_features,
    get_leading_punctuation_regex_pattern,
    get_repeated_punctuation_regex_pattern,
)


class TestPipelineUtils:

    @pytest.mark.unit
    def test_drop_trailing_features(self):
        x = torch.randn(10, 10, 20)
        expected_feature_buffer_len = 15
        x_dropped = drop_trailing_features(x, expected_feature_buffer_len)
        assert x_dropped.shape == (10, 10, 15)
        assert x_dropped.allclose(x[:, :, :15])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text, expected_text",
        [
            ("", ""),
            (" ", " "),
            ("simple text", "simple text"),
            ("just a 2nd . Yeah, I hope", "just a 2nd. Yeah, I hope"),
            ("Hello , world ! How are you ?", "Hello, world! How are you?"),
            ("The quick, brown fox jumps ? over the lazy ! dog.", "The quick, brown fox jumps? over the lazy! dog."),
        ],
    )
    def test_remove_leading_punctuation_spaces(self, text, expected_text):
        puncts = {"!", "?", ".", ","}
        pattern = get_leading_punctuation_regex_pattern(puncts)
        assert re.sub(pattern, r'\1', text) == expected_text

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text, expected_text",
        [
            ("", ""),
            (" ", " "),
            ("simple text", "simple text"),
            ("Hello, world!! How are you???", "Hello, world! How are you?"),
            ("The quick,, brown fox jumps? over the lazy! dog..", "The quick, brown fox jumps? over the lazy! dog."),
        ],
    )
    def test_remove_repeated_punctuation(self, text, expected_text):
        puncts = {"!", "?", ".", ","}
        pattern = get_repeated_punctuation_regex_pattern(puncts)
        assert re.sub(pattern, r'\1', text) == expected_text

    @pytest.mark.unit
    def test_check_existance_of_required_attributes(self):
        class TestClass:
            pass

        with pytest.raises(ValueError):
            check_existance_of_required_attributes(TestClass, ['test_attr'])
