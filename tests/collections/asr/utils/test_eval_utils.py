# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.utils.eval_utils import clean_label, convert_num_to_words


class TestConvertNumToWords:
    @pytest.mark.unit
    def test_standalone_zero_is_not_dropped(self):
        # Regression: `while num` is skipped for 0 (falsy), so a standalone "0" used to be
        # silently dropped, corrupting the text (and, in WER eval, the reference word count).
        assert convert_num_to_words("0").split() == ["zero"]
        assert convert_num_to_words("the answer is 0").split() == ["the", "answer", "is", "zero"]
        assert convert_num_to_words("5 0 3").split() == ["five", "zero", "three"]

    @pytest.mark.unit
    def test_nonzero_digits_unchanged(self):
        assert convert_num_to_words("2").split() == ["two"]
        assert convert_num_to_words("10").split() == ["one", "zero"]
        assert convert_num_to_words("2 apples").split() == ["two", "apples"]
        assert convert_num_to_words("no digits here").split() == ["no", "digits", "here"]

    @pytest.mark.unit
    def test_clean_label_keeps_zero(self):
        # clean_label converts numbers by default (num_to_words=True); the standalone "0" must
        # survive groundtruth cleaning used by cal_write_wer(clean_groundtruth_text=True).
        assert clean_label("The answer is 0") == "the answer is zero"
