# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import re

import pytest
import torch

from nemo.collections.asr.inference.streaming.text.text_processing import INCOMPLETE_LANG_TAG
from nemo.collections.asr.inference.utils.pipeline_utils import (
    check_existance_of_required_attributes,
    drop_trailing_features,
    filter_token_sequences,
    filter_token_triples,
    filter_tokens_from_greedy_output,
    get_leading_punctuation_regex_pattern,
    get_repeated_punctuation_regex_pattern,
)

BLANK_ID = 1024
# "<mt-MT>" as the model spells it out: "▁", "<", "m", "t", "-", "M", "T", ">"
MT_TAG = (2, 1844, 86, 34, 1846, 199, 189, 1845)
MT_TAG_NO_UNDERSCORE = MT_TAG[1:]


def make_greedy_output(tokens, timesteps=None, confidences=None):
    """Build a greedy-decoder chunk output like the one RNNTGreedyDecoder returns."""
    timesteps = list(range(len(tokens))) if timesteps is None else list(timesteps)
    confidences = [0.0] * len(tokens) if confidences is None else list(confidences)
    return {
        "tokens": list(tokens),
        "timesteps": timesteps,
        "confidences": confidences,
        "last_token": tokens[-1] if tokens else None,
        "last_token_idx": timesteps[-1] if timesteps else None,
    }


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


class TestFilterTokenTriples:
    """Removal of single-token language tags, keeping the aligned lists in step."""

    @pytest.mark.unit
    def test_removes_tokens_and_keeps_alignment(self):
        tokens, timesteps, confidences = filter_token_triples([5, 900, 7], [0, 1, 3], [0.5, 0.9, 0.7], {900})
        assert tokens == [5, 7]
        assert timesteps == [0, 3]
        assert confidences == [0.5, 0.7]

    @pytest.mark.unit
    def test_all_tokens_removed_returns_empty_lists(self):
        assert filter_token_triples([900, 901], [0, 1], [0.1, 0.2], {900, 901}) == ([], [], [])

    @pytest.mark.unit
    def test_empty_input_and_nothing_to_remove(self):
        assert filter_token_triples([], [], [], {900}) == ([], [], [])
        assert filter_token_triples([5, 7], [0, 1], [0.1, 0.2], set()) == ([5, 7], [0, 1], [0.1, 0.2])


class TestFilterTokenSequences:
    """Removal of language tags that are spelled out from several sub-word tokens."""

    @pytest.mark.unit
    def test_removes_sequence_and_keeps_alignment(self):
        tokens = [5, 7, *MT_TAG]
        timesteps = list(range(len(tokens)))
        confidences = [i / 10 for i in range(len(tokens))]
        assert filter_token_sequences(tokens, timesteps, confidences, (MT_TAG,)) == ([5, 7], [0, 1], [0.0, 0.1])

    @pytest.mark.unit
    def test_removes_sequence_in_the_middle(self):
        tokens = [5, *MT_TAG_NO_UNDERSCORE, 7]
        n = len(tokens)
        result_tokens, result_timesteps, _ = filter_token_sequences(
            tokens, list(range(n)), [0.0] * n, (MT_TAG_NO_UNDERSCORE,)
        )
        assert result_tokens == [5, 7]
        assert result_timesteps == [0, n - 1]

    @pytest.mark.unit
    def test_matching_follows_the_given_order_when_sequences_overlap(self):
        """A shorter sequence that prefixes a longer one wins if it is tried first, so callers must
        pass the sequences longest first (as ``_build_language_token_sequences`` does)."""
        tokens = [5, 1844, 86, 34, 7]
        short, long = (1844, 86), (1844, 86, 34)
        n = len(tokens)
        longest_first = filter_token_sequences(tokens, list(range(n)), [0.0] * n, (long, short))
        shortest_first = filter_token_sequences(tokens, list(range(n)), [0.0] * n, (short, long))
        assert longest_first[0] == [5, 7]
        assert shortest_first[0] == [5, 34, 7]

    @pytest.mark.unit
    def test_partial_sequence_at_end_is_not_matched(self):
        """A truncated tag is not a complete sequence and must be left untouched."""
        tokens = [5, *MT_TAG[:-1]]  # missing the closing ">"
        n = len(tokens)
        assert filter_token_sequences(tokens, list(range(n)), [0.0] * n, (MT_TAG,))[0] == tokens

    @pytest.mark.unit
    def test_no_match_returns_inputs_unchanged(self):
        """The all(keep) fast path returns the original objects rather than copies."""
        tokens, timesteps, confidences = [5, 7], [0, 1], [0.1, 0.2]
        result = filter_token_sequences(tokens, timesteps, confidences, (MT_TAG,))
        assert result[0] is tokens and result[1] is timesteps and result[2] is confidences

    @pytest.mark.unit
    def test_empty_sequences_or_tokens(self):
        tokens, timesteps, confidences = [5, 7], [0, 1], [0.1, 0.2]
        assert filter_token_sequences(tokens, timesteps, confidences, ())[0] is tokens
        assert filter_token_sequences([], [], [], (MT_TAG,))[0] == []


class TestFilterTokensFromGreedyOutput:
    """Combined filtering of a greedy chunk output plus its EOU label buffer."""

    @pytest.mark.unit
    def test_single_token_tag_is_blanked_in_labels(self):
        output = make_greedy_output([5, 900, 7], [0, 1, 3], [0.5, 0.9, 0.7])
        filtered, labels = filter_tokens_from_greedy_output(output, [5, 900, BLANK_ID, 7], {900}, BLANK_ID)
        assert filtered["tokens"] == [5, 7]
        assert filtered["timesteps"] == [0, 3]
        assert filtered["confidences"] == [0.5, 0.7]
        assert labels == [5, BLANK_ID, BLANK_ID, 7]

    @pytest.mark.unit
    def test_last_token_is_refreshed_when_tag_is_last(self):
        output = make_greedy_output([5, 7, 900], [0, 2, 3])
        filtered, _ = filter_tokens_from_greedy_output(output, [5, BLANK_ID, 7, 900], {900}, BLANK_ID)
        assert filtered["last_token"] == 7
        assert filtered["last_token_idx"] == 2

    @pytest.mark.unit
    def test_all_tokens_removed_clears_last_token(self):
        output = make_greedy_output([900, 901], [0, 1])
        filtered, labels = filter_tokens_from_greedy_output(output, [900, 901], {900, 901}, BLANK_ID)
        assert filtered["tokens"] == []
        assert filtered["last_token"] is None
        assert filtered["last_token_idx"] is None
        assert labels == [BLANK_ID, BLANK_ID]

    @pytest.mark.unit
    def test_multi_token_sequence_is_dropped_but_labels_are_untouched(self):
        """Sub-word pieces of a spelled-out tag also occur in normal text, so blanking them by
        membership would corrupt the transcript; only the output is filtered."""
        output = make_greedy_output([5, *MT_TAG])
        labels = [5, *MT_TAG]
        filtered, result_labels = filter_tokens_from_greedy_output(output, labels, set(), BLANK_ID, (MT_TAG,))
        assert filtered["tokens"] == [5]
        assert filtered["last_token"] == 5
        assert result_labels == labels

    @pytest.mark.unit
    def test_both_filters_apply_together(self):
        output = make_greedy_output([5, 900, *MT_TAG])
        filtered, labels = filter_tokens_from_greedy_output(output, [5, 900], {900}, BLANK_ID, (MT_TAG,))
        assert filtered["tokens"] == [5]
        assert labels == [5, BLANK_ID]

    @pytest.mark.unit
    def test_empty_output_returns_inputs_unchanged(self):
        output = make_greedy_output([])
        labels = [BLANK_ID]
        filtered, result_labels = filter_tokens_from_greedy_output(output, labels, {900}, BLANK_ID, (MT_TAG,))
        assert filtered is output and result_labels is labels

    @pytest.mark.unit
    def test_nothing_to_remove_returns_inputs_unchanged(self):
        output = make_greedy_output([5, 7])
        labels = [5, 7]
        filtered, result_labels = filter_tokens_from_greedy_output(output, labels, set(), BLANK_ID, ())
        assert filtered is output and result_labels is labels

    @pytest.mark.unit
    def test_input_output_dict_is_not_mutated(self):
        output = make_greedy_output([5, 900], [0, 1])
        original = dict(output)
        filter_tokens_from_greedy_output(output, [5, 900], {900}, BLANK_ID)
        assert output == original


class TestIncompleteLangTagPattern:
    """The regex that removes unterminated language tags from decoded text."""

    @pytest.mark.unit
    @pytest.mark.parametrize("fragment", ["<mt-MT", "<mt-M", "<sl-", "<\u0441\u043d-S-", "<"])
    def test_matches_truncated_fragments(self, fragment):
        assert INCOMPLETE_LANG_TAG.sub("", f"hello world. {fragment}") == "hello world."

    @pytest.mark.unit
    @pytest.mark.parametrize("text", ["a complete <en-US> tag", "hello world.", "the value 5 < 6 is true"])
    def test_does_not_touch_complete_tags_or_plain_text(self, text):
        """Complete tags are handled at token level, where timestamps are dropped alongside."""
        assert INCOMPLETE_LANG_TAG.sub("", text) == text

    @pytest.mark.unit
    def test_only_matches_at_end_of_string(self):
        assert INCOMPLETE_LANG_TAG.sub("", "before <mt-MT after") == "before <mt-MT after"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "word, is_fragment",
        [("<mt-MT", True), ("<sl-", True), ("<", True), ("<en-US>", False), ("hello", False)],
    )
    def test_fullmatch_identifies_a_word_that_is_only_a_fragment(self, word, is_fragment):
        """The word-granularity path drops the whole word; the segment path substrings it out."""
        assert bool(INCOMPLETE_LANG_TAG.fullmatch(word)) is is_fragment
