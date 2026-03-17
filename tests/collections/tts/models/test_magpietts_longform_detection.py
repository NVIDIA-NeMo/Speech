# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Unit tests for language-aware threshold detection (when to split text for inference).

Uses LanguageThresholds.exceeds_threshold from tts_dataset_utils, which drives
unified inference chunking (short text = single chunk, long text = sentence chunks).
"""

import pytest

from nemo.collections.tts.parts.utils.tts_dataset_utils import LanguageThresholds


@pytest.fixture
def language_thresholds():
    """Return default LanguageThresholds instance."""
    return LanguageThresholds()


class TestNeedsLongformInference:
    """Test cases for exceeds_threshold (language-aware split decision)."""

    # --- English tests (threshold: 45 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_below_threshold(self, language_thresholds):
        """English text with < 45 words should not trigger longform."""
        text = "Hello world. This is a short sentence."  # 7 words
        assert language_thresholds.exceeds_threshold(text, "en") is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_at_threshold(self, language_thresholds):
        """English text with exactly 45 words should trigger longform."""
        text = " ".join(["word"] * 45)
        assert language_thresholds.exceeds_threshold(text, "en") is True

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_above_threshold(self, language_thresholds):
        """English text with > 45 words should trigger longform."""
        text = " ".join(["word"] * 50)
        assert language_thresholds.exceeds_threshold(text, "en") is True

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_boundary_44_words(self, language_thresholds):
        """English text with 44 words (one below threshold) should not trigger longform."""
        text = " ".join(["word"] * 44)
        assert language_thresholds.exceeds_threshold(text, "en") is False

    # --- Spanish tests (threshold: 73 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_spanish_below_threshold(self, language_thresholds):
        """Spanish text with < 73 words should not trigger longform."""
        text = " ".join(["palabra"] * 72)
        assert language_thresholds.exceeds_threshold(text, "es") is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_spanish_at_threshold(self, language_thresholds):
        """Spanish text with >= 73 words should trigger longform."""
        text = " ".join(["palabra"] * 73)
        assert language_thresholds.exceeds_threshold(text, "es") is True

    # --- French tests (threshold: 69 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_french_at_threshold(self, language_thresholds):
        """French text with >= 69 words should trigger longform."""
        text = " ".join(["mot"] * 69)
        assert language_thresholds.exceeds_threshold(text, "fr") is True

    # --- German tests (threshold: 50 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_german_at_threshold(self, language_thresholds):
        """German text with >= 50 words should trigger longform."""
        text = " ".join(["wort"] * 50)
        assert language_thresholds.exceeds_threshold(text, "de") is True

    # --- Italian tests (threshold: 53 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_italian_at_threshold(self, language_thresholds):
        """Italian text with >= 53 words should trigger longform."""
        text = " ".join(["parola"] * 53)
        assert language_thresholds.exceeds_threshold(text, "it") is True

    # --- Vietnamese tests (threshold: 50 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_vietnamese_at_threshold(self, language_thresholds):
        """Vietnamese text with >= 50 words should trigger longform."""
        text = " ".join(["từ"] * 50)
        assert language_thresholds.exceeds_threshold(text, "vi") is True

    # --- Mandarin tests (threshold: 100 characters) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mandarin_below_threshold(self, language_thresholds):
        """Mandarin text below character threshold should not trigger split."""
        text = "你" * 99  # 99 characters
        assert language_thresholds.exceeds_threshold(text, "zh") is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mandarin_at_threshold(self, language_thresholds):
        """Mandarin text at threshold (100 chars) should trigger split."""
        text = "你" * 100
        assert language_thresholds.exceeds_threshold(text, "zh") is True

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mandarin_above_threshold(self, language_thresholds):
        """Mandarin text above threshold should trigger split."""
        text = "你" * 150
        assert language_thresholds.exceeds_threshold(text, "zh") is True

    # --- Edge cases ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_empty_text(self, language_thresholds):
        """Empty text should not trigger longform."""
        assert language_thresholds.exceeds_threshold("", "en") is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_whitespace_only(self, language_thresholds):
        """Whitespace-only text should not trigger longform."""
        assert language_thresholds.exceeds_threshold("   \t\n  ", "en") is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_single_long_word(self, language_thresholds):
        """Single very long word should count as 1 word."""
        text = "supercalifragilisticexpialidocious"  # 1 word
        assert language_thresholds.exceeds_threshold(text, "en") is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_text_with_punctuation(self, language_thresholds):
        """Words with punctuation should be counted correctly."""
        text = "word. " * 45  # 45 "word." tokens
        assert language_thresholds.exceeds_threshold(text, "en") is True

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_text_with_multiple_spaces(self, language_thresholds):
        """Multiple spaces between words should not affect word count."""
        text = "one  two   three    four     five      six       seven        eight         nine          ten"
        assert language_thresholds.exceeds_threshold(text, "en") is False  # 10 words < 45

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_realistic_english_long_text(self, language_thresholds):
        """Test with realistic long English text that should trigger longform."""
        text = """
        The quick brown fox jumps over the lazy dog. This sentence contains every
        letter of the alphabet. Sphinx of black quartz, judge my vow. Pack my box
        with five dozen liquor jugs. How vexingly quick daft zebras jump. The five
        boxing wizards jump quickly. Jackdaws love my big sphinx of quartz. The job
        requires extra pluck and zeal from every young wage earner. A wizard's job
        is to vex chumps quickly in fog.
        """
        assert language_thresholds.exceeds_threshold(text, "en") is True

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_realistic_english_short_text(self, language_thresholds):
        """Test with realistic short English text that should not trigger longform."""
        text = "Hello, how are you today? I hope you're having a great day."
        assert language_thresholds.exceeds_threshold(text, "en") is False
