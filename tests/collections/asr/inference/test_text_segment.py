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


import pytest

from nemo.collections.asr.inference.utils.text_segment import (
    TextSegment,
    Word,
    join_segments,
    normalize_segments_inplace,
)


class TestTextSegment:

    @pytest.mark.unit
    @pytest.mark.parametrize("text, expected_text", [("Hello!", "hello"), ("HeLLo!", "hello")])
    def test_normalize_text_inplace(self, text, expected_text):
        for cls in [Word, TextSegment]:
            text_segment = cls(text, 0, 1, 0.5)
            text_segment.normalize_text_inplace(punct_marks='!', sep=' ')
            assert text_segment.text == expected_text

    @pytest.mark.unit
    @pytest.mark.parametrize("text, expected_text", [("Hello!", "hello"), ("HeLLo!", "hello")])
    def test_with_normalized_text(self, text, expected_text):
        for cls in [Word, TextSegment]:
            text_segment = cls(text, 0, 1, 0.5)
            text_segment_copy = text_segment.with_normalized_text(punct_marks='!', sep=' ')
            assert text_segment_copy.text == expected_text
            assert text_segment.text == text

    @pytest.mark.unit
    def test_join_segments(self):
        for cls in [Word, TextSegment]:
            segments = [
                [cls('hello', 0, 1, 0.5), cls('world', 1, 2, 0.5)],
                [cls('how', 2, 3, 0.5), cls('are', 3, 4, 0.5), cls('you', 4, 5, 0.5)],
            ]
            transcriptions = join_segments(segments, sep=' ')
            assert transcriptions == ['hello world', 'how are you']

    @pytest.mark.unit
    def test_normalize_segments_inplace(self):
        for cls in [Word, TextSegment]:
            segments = [cls('Hello!', 0, 1, 0.5), cls('world?', 1, 2, 0.5)]
            normalize_segments_inplace(segments, punct_marks=set("!?"), sep=' ')
            assert segments[0].text == 'hello'
            assert segments[1].text == 'world'

    @pytest.mark.unit
    @pytest.mark.parametrize("text, expected_text", [("hello", "Hello"), ("World!", "World!")])
    def test_capitalize(self, text, expected_text):
        for cls in [Word, TextSegment]:
            text_segment = cls(text, 0, 1, 0.5)
            text_segment.capitalize()
            assert text_segment.text == expected_text
