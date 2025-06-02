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
from functools import cached_property
from typing import Any

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


class BaseTimestampsTest:
    """
    Base class for testing timestamps in decoders (CTC and RNNT).
    This class defines common test methods that can be inherited by both
    test_ctc_decoding.py and test_rnnt_decoding.py.
    """

    @cached_property
    def bpe_tokenizer(self):
        model = ASRModel.restore_from("/home/TestData/asr/stt_en_conformer_transducer_small.nemo", map_location="cpu")
        return model.tokenizer

    @property
    def char_offsets_chars(self):
        char_offsets = [
            {"char": "e", "start_offset": 0, "end_offset": 1},
            {"char": " ", "start_offset": 2, "end_offset": 2},
            {"char": "e", "start_offset": 3, "end_offset": 4},
            {"char": " ", "start_offset": 5, "end_offset": 5},
            {"char": ".", "start_offset": 6, "end_offset": 7},
            {"char": " ", "start_offset": 8, "end_offset": 9},
            {"char": "e", "start_offset": 10, "end_offset": 11},
            {"char": " ", "start_offset": 12, "end_offset": 13},
            {"char": "?", "start_offset": 14, "end_offset": 15},
            {"char": " ", "start_offset": 16, "end_offset": 17},
        ]
        return char_offsets

    @property
    def word_offsets_chars_expected_output(self):
        return [
            {'word': 'e', 'start_offset': 0, 'end_offset': 1},
            {'word': 'e.', 'start_offset': 3, 'end_offset': 7},
            {'word': 'e?', 'start_offset': 10, 'end_offset': 15},
        ]

    @property
    def word_offsets_chars_expected_output_other_delimiter(self):
        return [
            {'word': 'e e ', 'start_offset': 0, 'end_offset': 5},
            {'word': ' e? ', 'start_offset': 8, 'end_offset': 17},
        ]

    @property
    def segment_offsets_expected_output(self):
        return [
            {'segment': 'e e.', 'start_offset': 0, 'end_offset': 7},
            {'segment': 'e?', 'start_offset': 10, 'end_offset': 15},
        ]

    @property
    def segment_offsets_expected_output_gap(self):
        return [
            {'segment': 'e e. e?', 'start_offset': 0, 'end_offset': 15},
        ]

    @property
    def char_offsets_wpe(self):
        char_offsets = [
            {"char": 105, "start_offset": 0, "end_offset": 1},
            {"char": 126, "start_offset": 2, "end_offset": 2},
            {"char": 117, "start_offset": 3, "end_offset": 4},
            {"char": 68, "start_offset": 5, "end_offset": 6},
            {"char": 57, "start_offset": 6, "end_offset": 7},
            {"char": 122, "start_offset": 8, "end_offset": 9},
        ]

        return char_offsets

    @property
    def word_offsets_wpe_expected_output(self):
        return [
            {'word': 'nineteenth', 'start_offset': 0, 'end_offset': 2},
            {'word': 're', 'start_offset': 3, 'end_offset': 4},
            {'word': 'seventy', 'start_offset': 5, 'end_offset': 7},
            {'word': 'eighty', 'start_offset': 8, 'end_offset': 9},
        ]

    @property
    def word_offsets_wpe_expected_output_other_delimiter(self):
        return [
            {'word': 'nineteenth', 'start_offset': 0, 'end_offset': 2},
            {'word': 'seventy eighty', 'start_offset': 5, 'end_offset': 9},
        ]

    @property
    def char_offsets_bpe(self):
        char_offsets = [
            {"char": 1014, "start_offset": 0, "end_offset": 2},
            {"char": 1009, "start_offset": 2, "end_offset": 4},
            {"char": 6, "start_offset": 5, "end_offset": 5},
            {"char": 145, "start_offset": 5, "end_offset": 6},
            {"char": 349, "start_offset": 6, "end_offset": 7},
            {"char": 622, "start_offset": 8, "end_offset": 9},
        ]

        return char_offsets

    @property
    def word_offsets_bpe_expected_output(self):
        return [
            {'word': "discuss", 'start_offset': 0, 'end_offset': 2},
            {'word': "absolute'", 'start_offset': 2, 'end_offset': 5},
            {'word': "really", 'start_offset': 5, 'end_offset': 6},
            {'word': "friendship", 'start_offset': 6, 'end_offset': 9},
        ]

    @property
    def word_offsets_bpe_expected_output_other_delimiter(self):
        return [
            {'word': "discuss absolute'", 'start_offset': 0, 'end_offset': 5},
            {'word': "friendship", 'start_offset': 6, 'end_offset': 9},
        ]

    @staticmethod
    def check_char_timestamps(hyp: Hypothesis, decoding: Any):
        """Test character-level timestamps for both CTC and RNNT"""
        assert hyp.timestamp is not None
        assert isinstance(hyp.timestamp, dict)
        assert 'timestep' in hyp.timestamp
        assert 'char' in hyp.timestamp
        assert 'word' in hyp.timestamp
        assert 'segment' in hyp.timestamp

        hypothesis_text = re.sub(r'\s+', ' ', hyp.text.strip())

        words = hyp.text.split(decoding.word_seperator)
        words = list(filter(lambda x: x != '', words))
        assert len(hyp.timestamp['word']) == len(words)

        words_from_timestamps = [ts['word'] for ts in hyp.timestamp['word']]
        assert hypothesis_text == decoding.word_seperator.join(words_from_timestamps)

        segments = []
        segment = []

        for word in words:
            segment.append(word)
            if word[-1] in decoding.segment_seperators:
                segments.append(' '.join(segment))
                segment = []

        if segment:
            segments.append(' '.join(segment))

        assert len(hyp.timestamp['segment']) == len(segments)

        segments_from_timestamps = [ts['segment'] for ts in hyp.timestamp['segment']]
        assert hypothesis_text == decoding.word_seperator.join(segments_from_timestamps)

    @staticmethod
    def check_subword_timestamps(hyp: Hypothesis, decoding: Any):
        """Test subword-level timestamps for both CTC and RNNT"""
        assert hyp.timestamp is not None
        assert isinstance(hyp.timestamp, dict)
        assert 'timestep' in hyp.timestamp
        assert 'char' in hyp.timestamp
        assert 'word' in hyp.timestamp
        assert 'segment' in hyp.timestamp

        chars = list(hyp.text)
        chars = list(filter(lambda x: x not in ['', ' ', '#'], chars))
        all_chars = [list(decoding.tokenizer.tokens_to_text(data['char'])) for data in hyp.timestamp['char']]
        all_chars = [char for subword in all_chars for char in subword]
        all_chars = list(filter(lambda x: x not in ['', ' ', '#'], all_chars))
        assert len(chars) == len(all_chars)

        hypothesis_text = re.sub(r'\s+', ' ', hyp.text.strip())

        words_from_timestamps = [ts['word'] for ts in hyp.timestamp['word']]
        assert hypothesis_text == decoding.word_seperator.join(words_from_timestamps)

        segments_count = sum([hyp.text.count(seperator) for seperator in decoding.segment_seperators])
        if hyp.text[-1] not in decoding.segment_seperators:
            segments_count += 1

        if hyp.text in decoding.segment_seperators:
            segments_count = 0

        assert len(hyp.timestamp['segment']) == segments_count

        segments_from_timestamps = [ts['segment'] for ts in hyp.timestamp['segment']]
        assert hypothesis_text == decoding.word_seperator.join(segments_from_timestamps)

    def test_word_offsets_chars(self):
        word_offsets = self.decoding_char.get_words_offsets(
            char_offsets=self.char_offsets_chars,
            encoded_char_offsets=None,
            word_delimiter_char=" ",
            supported_punctuation={'.', '!', '?'},
        )

        assert word_offsets == self.word_offsets_chars_expected_output

    def test_word_offsets_char_other_delimiter(self):
        word_offsets = self.decoding_char.get_words_offsets(
            char_offsets=self.char_offsets_chars,
            encoded_char_offsets=None,
            word_delimiter_char=".",
            supported_punctuation={'.', '!', '?'},
        )

        assert word_offsets == self.word_offsets_chars_expected_output_other_delimiter

    def test_word_offsets_subword_wpe(self):
        word_offsets = self.decoding_subword_wpe.get_words_offsets(
            char_offsets=None,
            encoded_char_offsets=self.char_offsets_wpe,
            word_delimiter_char=" ",
            supported_punctuation={'.', '!', '?'},
        )

        assert word_offsets == self.word_offsets_wpe_expected_output

    def test_word_offsets_subword_wpe_other_delimiter(self):
        word_offsets = self.decoding_subword_wpe.get_words_offsets(
            char_offsets=None,
            encoded_char_offsets=self.char_offsets_wpe,
            word_delimiter_char="re",
            supported_punctuation={'.', '!', '?'},
        )

        assert word_offsets == self.word_offsets_wpe_expected_output_other_delimiter

    def test_word_offsets_subword_bpe(self):
        word_offsets = self.decoding_subword_bpe.get_words_offsets(
            char_offsets=None,
            encoded_char_offsets=self.char_offsets_bpe,
            word_delimiter_char=" ",
            supported_punctuation={'.', '!', '?'},
        )

        assert word_offsets == self.word_offsets_bpe_expected_output

    def test_word_offsets_subword_bpe_other_delimiter(self):
        word_offsets = self.decoding_subword_bpe.get_words_offsets(
            char_offsets=None,
            encoded_char_offsets=self.char_offsets_bpe,
            word_delimiter_char="really",
            supported_punctuation={'.', '!', '?'},
        )

        assert word_offsets == self.word_offsets_bpe_expected_output_other_delimiter

    def test_segment_offsets_delimiter(self):
        segment_offsets = self.decoding_char._get_segment_offsets(
            offsets=self.word_offsets_chars_expected_output,
            segment_delimiter_tokens=['.', '!', '?'],
            supported_punctuation={'.', '!', '?'},
        )

        assert segment_offsets == self.segment_offsets_expected_output

    def test_segment_offsets_gap(self):
        segment_offsets = self.decoding_char._get_segment_offsets(
            offsets=self.word_offsets_chars_expected_output,
            segment_delimiter_tokens=[],
            supported_punctuation={},
            segment_gap_threshold=10,
        )

        assert segment_offsets == self.segment_offsets_expected_output_gap
