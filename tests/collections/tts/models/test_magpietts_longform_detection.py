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
Unit tests for MagpieTTSModel longform inference detection.
"""

from unittest.mock import MagicMock

import pytest

from nemo.collections.tts.models.magpietts import MagpieTTSModel

# Get the method once at module level
_needs_longform_inference = MagpieTTSModel._needs_longform_inference


class TestNeedsLongformInference:
    """Test cases for MagpieTTSModel._needs_longform_inference method."""

    @pytest.fixture
    def mock_model(self):
        """Return the _needs_longform_inference method for testing."""
        return _needs_longform_inference

    # --- English tests (threshold: 45 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_below_threshold(self, mock_model):
        """English text with < 45 words should not trigger longform."""
        text = "Hello world. This is a short sentence."  # 7 words
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_at_threshold(self, mock_model):
        """English text with exactly 45 words should trigger longform."""
        # Generate exactly 45 words
        text = " ".join(["word"] * 45)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is True

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_above_threshold(self, mock_model):
        """English text with > 45 words should trigger longform."""
        text = " ".join(["word"] * 50)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is True

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_boundary_44_words(self, mock_model):
        """English text with 44 words (one below threshold) should not trigger longform."""
        text = " ".join(["word"] * 44)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is False

    # --- Spanish tests (threshold: 73 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_spanish_below_threshold(self, mock_model):
        """Spanish text with < 73 words should not trigger longform."""
        text = " ".join(["palabra"] * 72)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "es")
        assert result is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_spanish_at_threshold(self, mock_model):
        """Spanish text with >= 73 words should trigger longform with warning."""
        text = " ".join(["palabra"] * 73)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "es")
        assert result is True

    # --- French tests (threshold: 69 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_french_at_threshold(self, mock_model):
        """French text with >= 69 words should trigger longform."""
        text = " ".join(["mot"] * 69)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "fr")
        assert result is True

    # --- German tests (threshold: 50 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_german_at_threshold(self, mock_model):
        """German text with >= 50 words should trigger longform."""
        text = " ".join(["wort"] * 50)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "de")
        assert result is True

    # --- Italian tests (threshold: 53 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_italian_at_threshold(self, mock_model):
        """Italian text with >= 53 words should trigger longform."""
        text = " ".join(["parola"] * 53)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "it")
        assert result is True

    # --- Vietnamese tests (threshold: 50 words) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_vietnamese_at_threshold(self, mock_model):
        """Vietnamese text with >= 50 words should trigger longform."""
        text = " ".join(["từ"] * 50)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "vi")
        assert result is True

    # --- Mandarin tests (threshold: 100 characters, but always returns False) ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mandarin_below_threshold(self, mock_model):
        """Mandarin text below character threshold should not trigger longform."""
        text = "你" * 99  # 99 characters
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "zh")
        assert result is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mandarin_at_threshold_returns_false(self, mock_model):
        """Mandarin text at/above threshold should still return False (not supported)."""
        text = "你" * 100  # 100 characters - at threshold
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "zh")
        # Mandarin longform is not supported, should return False with warning
        assert result is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mandarin_above_threshold_returns_false(self, mock_model):
        """Mandarin text above threshold should still return False (not supported)."""
        text = "你" * 150  # 150 characters - above threshold
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "zh")
        # Mandarin longform is not supported, should return False
        assert result is False

    # --- Edge cases ---

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_empty_text(self, mock_model):
        """Empty text should not trigger longform."""
        text = ""
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_whitespace_only(self, mock_model):
        """Whitespace-only text should not trigger longform."""
        text = "   \t\n  "
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_single_long_word(self, mock_model):
        """Single very long word should count as 1 word."""
        text = "supercalifragilisticexpialidocious"  # 1 word
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_text_with_punctuation(self, mock_model):
        """Words with punctuation should be counted correctly."""
        # 45 "word." entries - split() will treat "word." as one word
        text = "word. " * 45
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is True

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_text_with_multiple_spaces(self, mock_model):
        """Multiple spaces between words should not affect word count."""
        # 10 words with multiple spaces
        text = "one  two   three    four     five      six       seven        eight         nine          ten"
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is False  # 10 words < 45 threshold

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_realistic_english_long_text(self, mock_model):
        """Test with realistic long English text that should trigger longform."""
        text = """
        The quick brown fox jumps over the lazy dog. This sentence contains every 
        letter of the alphabet. Sphinx of black quartz, judge my vow. Pack my box 
        with five dozen liquor jugs. How vexingly quick daft zebras jump. The five 
        boxing wizards jump quickly. Jackdaws love my big sphinx of quartz. The job
        requires extra pluck and zeal from every young wage earner. A wizard's job
        is to vex chumps quickly in fog.
        """
        # Count: ~75 words (above threshold)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is True

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_realistic_english_short_text(self, mock_model):
        """Test with realistic short English text that should not trigger longform."""
        text = "Hello, how are you today? I hope you're having a great day."
        # Count: ~12 words (below threshold)
        mock_self = MagicMock()
        result = mock_model(mock_self, text, "en")
        assert result is False
