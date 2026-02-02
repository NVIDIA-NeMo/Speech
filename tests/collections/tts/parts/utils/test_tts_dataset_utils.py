# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from pathlib import Path

import librosa
import numpy as np
import pytest
import torch

from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    _get_sentence_separators_for_language,
    chunk_and_tokenize_text_by_sentence,
    filter_dataset_by_duration,
    get_abs_rel_paths,
    get_audio_filepaths,
    get_word_count,
    load_audio,
    normalize_volume,
    split_by_sentence,
    stack_tensors,
)


class TestTTSDatasetUtils:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_abs_rel_paths_input_abs(self):
        input_path = Path("/home/data/audio/test")
        base_path = Path("/home/data")

        abs_path, rel_path = get_abs_rel_paths(input_path=input_path, base_path=base_path)

        assert abs_path == input_path
        assert rel_path == Path("audio/test")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_abs_rel_paths_input_rel(self):
        input_path = Path("audio/test")
        base_path = Path("/home/data")

        abs_path, rel_path = get_abs_rel_paths(input_path=input_path, base_path=base_path)

        assert abs_path == Path("/home/data/audio/test")
        assert rel_path == input_path

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_audio_paths(self):
        audio_dir = Path("/home/audio")
        audio_rel_path = Path("examples/example.wav")
        manifest_entry = {"audio_filepath": str(audio_rel_path)}

        abs_path, rel_path = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)

        assert abs_path == Path("/home/audio/examples/example.wav")
        assert rel_path == audio_rel_path

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_load_audio(self, test_data_dir):
        sample_rate = 22050
        test_data_dir = Path(test_data_dir)
        audio_filepath_rel = Path("tts/mini_ljspeech/wavs/LJ003-0182.wav")
        audio_filepath = test_data_dir / audio_filepath_rel
        manifest_entry = {"audio_filepath": str(audio_filepath_rel)}

        expected_audio, _ = librosa.load(path=audio_filepath, sr=sample_rate)
        audio, _, _ = load_audio(manifest_entry=manifest_entry, audio_dir=test_data_dir, sample_rate=sample_rate)

        np.testing.assert_array_almost_equal(audio, expected_audio)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_load_audio_with_offset(self, test_data_dir):
        sample_rate = 22050
        offset = 1.0
        duration = 2.0
        test_data_dir = Path(test_data_dir)
        audio_filepath_rel = Path("tts/mini_ljspeech/wavs/LJ003-0182.wav")
        audio_filepath = test_data_dir / audio_filepath_rel
        manifest_entry = {"audio_filepath": str(audio_filepath_rel), "offset": offset, "duration": duration}

        expected_audio, _ = librosa.load(path=audio_filepath, offset=offset, duration=duration, sr=sample_rate)
        audio, _, _ = load_audio(manifest_entry=manifest_entry, audio_dir=test_data_dir, sample_rate=sample_rate)

        np.testing.assert_array_almost_equal(audio, expected_audio)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume(self):
        input_audio = np.array([0.0, 0.1, 0.3, 0.5])
        expected_output = np.array([0.0, 0.18, 0.54, 0.9])

        output_audio = normalize_volume(audio=input_audio, volume_level=0.9)

        np.testing.assert_array_almost_equal(output_audio, expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_negative_peak(self):
        input_audio = np.array([0.0, 0.1, -0.3, -1.0, 0.5])
        expected_output = np.array([0.0, 0.05, -0.15, -0.5, 0.25])

        output_audio = normalize_volume(audio=input_audio, volume_level=0.5)

        np.testing.assert_array_almost_equal(output_audio, expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_zero(self):
        input_audio = np.array([0.0, 0.1, 0.3, 0.5])
        expected_output = np.array([0.0, 0.0, 0.0, 0.0])

        output_audio = normalize_volume(audio=input_audio, volume_level=0.0)

        np.testing.assert_array_almost_equal(output_audio, expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_max(self):
        input_audio = np.array([0.0, 0.1, 0.3, 0.5])
        expected_output = np.array([0.0, 0.2, 0.6, 1.0])

        output_audio = normalize_volume(audio=input_audio, volume_level=1.0)

        np.testing.assert_array_almost_equal(output_audio, expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_zeros(self):
        input_audio = np.array([0.0, 0.0, 0.0])

        output_audio = normalize_volume(audio=input_audio, volume_level=0.5)

        np.testing.assert_array_almost_equal(output_audio, input_audio)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_empty(self):
        input_audio = np.array([])

        output_audio = normalize_volume(audio=input_audio, volume_level=1.0)

        np.testing.assert_array_almost_equal(output_audio, input_audio)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_out_of_range(self):
        input_audio = np.array([0.0, 0.1, 0.3, 0.5])
        with pytest.raises(ValueError, match="Volume must be in range"):
            normalize_volume(audio=input_audio, volume_level=2.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_stack_tensors(self):
        tensors = [torch.ones([2]), torch.ones([4]), torch.ones([3])]
        max_lens = [6]
        expected_output = torch.tensor(
            [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.float32
        )

        stacked_tensor = stack_tensors(tensors=tensors, max_lens=max_lens)

        torch.testing.assert_close(stacked_tensor, expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_stack_tensors_3d(self):
        tensors = [torch.ones([2, 2]), torch.ones([1, 3])]
        max_lens = [4, 2]
        expected_output = torch.tensor(
            [[[1, 1, 0, 0], [1, 1, 0, 0]], [[1, 1, 1, 0], [0, 0, 0, 0]]], dtype=torch.float32
        )

        stacked_tensor = stack_tensors(tensors=tensors, max_lens=max_lens)

        torch.testing.assert_close(stacked_tensor, expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_filter_dataset_by_duration(self):
        min_duration = 1.0
        max_duration = 10.0
        entries = [
            {"duration": 0.5},
            {"duration": 10.0},
            {"duration": 20.0},
            {"duration": 0.1},
            {"duration": 100.0},
            {"duration": 5.0},
        ]

        filtered_entries, total_hours, filtered_hours = filter_dataset_by_duration(
            entries=entries, min_duration=min_duration, max_duration=max_duration
        )

        assert len(filtered_entries) == 2
        assert filtered_entries[0]["duration"] == 10.0
        assert filtered_entries[1]["duration"] == 5.0
        assert total_hours == (135.6 / 3600.0)
        assert filtered_hours == (15.0 / 3600.0)


class TestSentenceSplitting:
    """Tests for sentence splitting and tokenization functions."""

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_sentence_separators_english(self):
        """Test that English uses default Western punctuation."""
        separators = _get_sentence_separators_for_language("en")
        assert '.' in separators
        assert '?' in separators
        assert '!' in separators

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_sentence_separators_japanese(self):
        """Test that Japanese includes Japanese punctuation."""
        separators = _get_sentence_separators_for_language("ja")
        assert '。' in separators  # Japanese period
        assert '？' in separators  # Japanese question mark
        assert '！' in separators  # Japanese exclamation mark
        # Also includes Western for mixed text
        assert '.' in separators
        assert '?' in separators

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_sentence_separators_hindi(self):
        """Test that Hindi includes Devanagari Danda."""
        separators = _get_sentence_separators_for_language("hi")
        assert '।' in separators  # Devanagari Danda (primary Hindi sentence ender)
        assert '॥' in separators  # Double Danda
        # Also includes Western punctuation
        assert '?' in separators
        assert '!' in separators

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_sentence_separators_chinese(self):
        """Test that Chinese includes CJK punctuation."""
        separators = _get_sentence_separators_for_language("zh")
        assert '。' in separators  # Chinese period
        assert '？' in separators
        assert '！' in separators

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_sentence_separators_unknown_language(self):
        """Test that unknown languages fall back to Western punctuation defaults."""
        separators = _get_sentence_separators_for_language("xyz")
        assert separators == ['.', '?', '!']

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_english(self):
        """Test English sentence splitting."""
        text = "Hello world. How are you? I am fine!"
        sentences = split_by_sentence(text)

        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I am fine!"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_title_abbreviations(self):
        """Test that title abbreviations (Dr., Mr., etc.) don't cause splits."""
        text = "Dr. Smith is here. He arrived early."
        sentences = split_by_sentence(text)

        # "Dr." is a title - should not cause split
        assert len(sentences) == 2
        assert sentences[0] == "Dr. Smith is here."
        assert sentences[1] == "He arrived early."

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_multiple_titles(self):
        """Test multiple title abbreviations in one sentence."""
        text = "Mr. and Mrs. Johnson met Prof. Lee yesterday."
        sentences = split_by_sentence(text)

        # All titles should be preserved
        assert len(sentences) == 1
        assert "Mr. and Mrs. Johnson met Prof. Lee" in sentences[0]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_time_abbreviations(self):
        """Test that time abbreviations (a.m., p.m.) can end sentences."""
        # "a.m." followed by new sentence should split
        text = "The meeting is at 9 a.m. Please arrive early."
        sentences = split_by_sentence(text)
        assert len(sentences) == 2
        assert sentences[0] == "The meeting is at 9 a.m."
        assert sentences[1] == "Please arrive early."

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_japanese(self):
        """Test Japanese sentence splitting with Japanese punctuation."""
        text = "こんにちは。 お元気ですか？ 私は元気です！"
        sentences = split_by_sentence(text, language="ja")

        assert len(sentences) == 3
        assert sentences[0] == "こんにちは。"
        assert sentences[1] == "お元気ですか？"
        assert sentences[2] == "私は元気です！"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_hindi(self):
        """Test Hindi sentence splitting with Devanagari Danda."""
        text = "नमस्ते। आप कैसे हैं? मैं ठीक हूँ।"
        sentences = split_by_sentence(text, language="hi")

        assert len(sentences) == 3
        assert "नमस्ते" in sentences[0]
        assert "आप कैसे हैं" in sentences[1]
        assert "मैं ठीक हूँ" in sentences[2]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_japanese_no_spaces(self):
        """Test Japanese sentence splitting WITHOUT spaces between sentences."""
        # This is the common case in Japanese - no spaces after punctuation
        text = "こんにちは。元気ですか？私は元気です！"
        sentences = split_by_sentence(text, language="ja")

        assert len(sentences) == 3
        assert sentences[0] == "こんにちは。"
        assert sentences[1] == "元気ですか？"
        assert sentences[2] == "私は元気です！"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_chinese_no_spaces(self):
        """Test Chinese sentence splitting WITHOUT spaces between sentences."""
        text = "你好。你好吗？我很好！"
        sentences = split_by_sentence(text, language="zh")

        assert len(sentences) == 3
        assert sentences[0] == "你好。"
        assert sentences[1] == "你好吗？"
        assert sentences[2] == "我很好！"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_hindi_no_spaces(self):
        """Test Hindi sentence splitting WITHOUT spaces after Danda.

        Note: Danda (।) splits regardless of following whitespace.
        Western punctuation (? !) still requires whitespace in Hindi text.
        """
        # Danda followed by no space - should still split
        text = "नमस्ते।आप कैसे हैं।मैं ठीक हूँ।"
        sentences = split_by_sentence(text, language="hi")

        assert len(sentences) == 3
        assert "नमस्ते" in sentences[0]
        assert "आप कैसे हैं" in sentences[1]
        assert "मैं ठीक हूँ" in sentences[2]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_english_with_newline(self):
        """Test English sentence splitting with newline after punctuation."""
        text = "Hello world.\nHow are you?"
        sentences = split_by_sentence(text)

        assert len(sentences) == 2
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_english_end_of_string(self):
        """Test that sentence at end of string (no following char) is handled."""
        text = "Hello world."
        sentences = split_by_sentence(text)

        assert len(sentences) == 1
        assert sentences[0] == "Hello world."

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_empty(self):
        """Test that empty text returns empty list."""
        sentences = split_by_sentence("")
        assert sentences == []

        sentences = split_by_sentence("   ")
        assert sentences == []

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_by_sentence_single_sentence(self):
        """Test single sentence without ending punctuation."""
        text = "Hello world"
        sentences = split_by_sentence(text)

        assert len(sentences) == 1
        assert sentences[0] == "Hello world"


class TestChunkAndTokenizeTextBySentence:
    """Tests for chunk_and_tokenize_text_by_sentence function."""

    class MockTokenizer:
        """Mock tokenizer for testing."""

        def encode(self, text: str, tokenizer_name: str):
            """Simple mock that returns character codes."""
            return [ord(c) for c in text]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_chunk_and_tokenize_english(self):
        """Test tokenization with English text."""
        text = "Hello. World."
        tokenizer = self.MockTokenizer()
        eos_id = 0

        tokens, lens, texts = chunk_and_tokenize_text_by_sentence(
            text=text,
            tokenizer_name="test",
            text_tokenizer=tokenizer,
            eos_token_id=eos_id,
            language="en",
        )

        assert len(tokens) == 2
        assert len(lens) == 2
        assert len(texts) == 2
        assert texts[0] == "Hello."
        assert texts[1] == "World."
        # Each token tensor should end with EOS
        assert tokens[0][-1].item() == eos_id
        assert tokens[1][-1].item() == eos_id

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_chunk_and_tokenize_japanese(self):
        """Test tokenization with Japanese text."""
        text = "こんにちは。 世界。"
        tokenizer = self.MockTokenizer()
        eos_id = 0

        tokens, lens, texts = chunk_and_tokenize_text_by_sentence(
            text=text,
            tokenizer_name="test",
            text_tokenizer=tokenizer,
            eos_token_id=eos_id,
            language="ja",
        )

        assert len(tokens) == 2
        assert len(texts) == 2
        assert "こんにちは" in texts[0]
        assert "世界" in texts[1]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_chunk_and_tokenize_hindi(self):
        """Test tokenization with Hindi text."""
        text = "नमस्ते। दुनिया।"
        tokenizer = self.MockTokenizer()
        eos_id = 0

        tokens, lens, texts = chunk_and_tokenize_text_by_sentence(
            text=text,
            tokenizer_name="test",
            text_tokenizer=tokenizer,
            eos_token_id=eos_id,
            language="hi",
        )

        assert len(tokens) == 2
        assert len(texts) == 2
        assert "नमस्ते" in texts[0]
        assert "दुनिया" in texts[1]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_chunk_and_tokenize_returns_tensors(self):
        """Test that returned tokens are torch tensors."""
        text = "Hello. World."
        tokenizer = self.MockTokenizer()
        eos_id = 0

        tokens, lens, texts = chunk_and_tokenize_text_by_sentence(
            text=text,
            tokenizer_name="test",
            text_tokenizer=tokenizer,
            eos_token_id=eos_id,
            language="en",
        )

        for token_tensor in tokens:
            assert isinstance(token_tensor, torch.Tensor)
            assert token_tensor.dtype == torch.int32

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_chunk_and_tokenize_lens_match_tensors(self):
        """Test that lengths match tensor sizes."""
        text = "Hello. World. Test."
        tokenizer = self.MockTokenizer()
        eos_id = 0

        tokens, lens, texts = chunk_and_tokenize_text_by_sentence(
            text=text,
            tokenizer_name="test",
            text_tokenizer=tokenizer,
            eos_token_id=eos_id,
            language="en",
        )

        for token_tensor, length in zip(tokens, lens):
            assert token_tensor.shape[0] == length


class TestGetWordCount:
    """Tests for get_word_count function with language-aware word counting."""

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_word_count_english(self):
        """Test English word count using whitespace splitting."""
        text = "Hello world how are you"
        assert get_word_count(text, "en") == 5

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_word_count_english_with_punctuation(self):
        """Test English word count with punctuation."""
        text = "Hello, world! How are you?"
        assert get_word_count(text, "en") == 5

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_word_count_chinese_characters(self):
        """Test Chinese character count (no word segmentation)."""
        text = "你好世界"
        assert get_word_count(text, "zh") == 4

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_word_count_chinese_with_spaces(self):
        """Test Chinese ignores spaces for character count."""
        text = "你好 世界"
        assert get_word_count(text, "zh") == 4

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_word_count_hindi(self):
        """Test Hindi word count using whitespace splitting."""
        text = "नमस्ते दुनिया कैसे हो"
        assert get_word_count(text, "hi") == 4

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_word_count_empty_string(self):
        """Test empty string returns 0."""
        assert get_word_count("", "en") == 0
        assert get_word_count("", "ja") == 0
        assert get_word_count("", "zh") == 0

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_word_count_whitespace_only(self):
        """Test whitespace-only string returns 0."""
        assert get_word_count("   ", "en") == 0
        assert get_word_count("\t\n", "en") == 0

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_word_count_japanese_with_pyopenjtalk(self):
        """Test Japanese word count using pyopenjtalk morphological analysis."""
        try:
            import pyopenjtalk  # noqa: F401

            # "こんにちは世界" should be segmented into morphemes
            text = "こんにちは世界"
            word_count = get_word_count(text, "ja")
            # pyopenjtalk typically segments this into ~2-3 morphemes
            assert word_count >= 2

            # Longer sentence with more words
            text2 = "今日はいい天気ですね"
            word_count2 = get_word_count(text2, "ja")
            # Should have multiple morphemes
            assert word_count2 >= 4

        except ImportError:
            pytest.skip("pyopenjtalk not installed")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_word_count_japanese_fallback(self):
        """Test Japanese fallback when pyopenjtalk analysis returns empty."""
        # This tests the fallback path - if pyopenjtalk returns no words,
        # it should fall back to character count
        text = "テスト"
        word_count = get_word_count(text, "ja")
        # Should return something > 0 regardless of pyopenjtalk availability
        assert word_count >= 1

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_word_count_unknown_language_fallback(self):
        """Test unknown language falls back to whitespace splitting."""
        text = "word1 word2 word3"
        assert get_word_count(text, "unknown") == 3
