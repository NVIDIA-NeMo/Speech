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
    filter_dataset_by_duration,
    get_abs_rel_paths,
    get_audio_filepaths,
    load_audio,
    normalize_volume,
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


class TestLanguageThresholds:
    """Test cases for LanguageThresholds dataclass."""

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_default_thresholds(self):
        """Test default language thresholds are set correctly."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import LanguageThresholds

        thresholds = LanguageThresholds()

        assert thresholds.thresholds["en"] == 45
        assert thresholds.thresholds["es"] == 73
        assert thresholds.thresholds["fr"] == 69
        assert thresholds.thresholds["de"] == 50
        assert thresholds.thresholds["zh"] == 100
        assert "zh" in thresholds.character_based

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_word_count_english(self):
        """Test word count for English (word-based)."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import LanguageThresholds

        thresholds = LanguageThresholds()
        text = "Hello world this is a test"

        count = thresholds.get_word_count(text, "en")

        assert count == 6

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_word_count_chinese(self):
        """Test character count for Chinese (character-based)."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import LanguageThresholds

        thresholds = LanguageThresholds()
        text = "你好世界"  # 4 characters

        count = thresholds.get_word_count(text, "zh")

        assert count == 4

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_exceeds_threshold_short_text(self):
        """Test that short text does not exceed threshold."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import LanguageThresholds

        thresholds = LanguageThresholds()
        short_text = "Hello world."  # 2 words, below 45

        assert not thresholds.exceeds_threshold(short_text, "en")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_exceeds_threshold_long_text(self):
        """Test that long text exceeds threshold."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import LanguageThresholds

        thresholds = LanguageThresholds()
        # Generate text with more than 45 words
        long_text = " ".join(["word"] * 50)

        assert thresholds.exceeds_threshold(long_text, "en")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_exceeds_threshold_boundary(self):
        """Test boundary case exactly at threshold."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import LanguageThresholds

        thresholds = LanguageThresholds()
        # Generate text with exactly 45 words (threshold)
        boundary_text = " ".join(["word"] * 45)

        # At threshold should be True (>= comparison)
        assert thresholds.exceeds_threshold(boundary_text, "en")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_exceeds_threshold_fallback_to_english(self):
        """Test fallback to English threshold for unknown language."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import LanguageThresholds

        thresholds = LanguageThresholds()
        # Unknown language should use English threshold (45)
        text = " ".join(["word"] * 50)

        assert thresholds.exceeds_threshold(text, "unknown_lang")


class TestChunkTextForInference:
    """Test cases for chunk_text_for_inference function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a simple mock tokenizer for testing."""

        class MockTokenizer:
            def encode(self, text, tokenizer_name):
                # Simple mock: return list of integers based on word count
                return list(range(len(text.split())))

        return MockTokenizer()

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_short_text_single_chunk(self, mock_tokenizer):
        """Test that short text returns as single chunk."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference

        short_text = "Hello world."
        eos_id = 999

        tokens, lens, texts = chunk_text_for_inference(
            text=short_text,
            language="en",
            tokenizer_name="english_phoneme",
            text_tokenizer=mock_tokenizer,
            eos_token_id=eos_id,
        )

        assert len(tokens) == 1  # Single chunk
        assert len(lens) == 1
        assert len(texts) == 1
        assert texts[0] == short_text
        assert tokens[0][-1].item() == eos_id  # EOS token appended

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_long_text_multiple_chunks(self, mock_tokenizer):
        """Test that long text is split into multiple sentence chunks."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference

        # Create long text with multiple sentences (> 45 words)
        long_text = "This is sentence one. " * 20 + "This is sentence two."
        eos_id = 999

        tokens, lens, texts = chunk_text_for_inference(
            text=long_text,
            language="en",
            tokenizer_name="english_phoneme",
            text_tokenizer=mock_tokenizer,
            eos_token_id=eos_id,
        )

        assert len(tokens) > 1  # Multiple chunks
        assert len(tokens) == len(lens) == len(texts)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_custom_language_threshold(self, mock_tokenizer):
        """Test with different language thresholds."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference

        # German has threshold of 50 words
        text = " ".join(["wort"] * 40)  # 40 words, below German threshold
        eos_id = 999

        tokens, lens, texts = chunk_text_for_inference(
            text=text,
            language="de",
            tokenizer_name="german_phoneme",
            text_tokenizer=mock_tokenizer,
            eos_token_id=eos_id,
        )

        # 40 words is below German threshold (50), so should be single chunk
        assert len(tokens) == 1

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_empty_text(self, mock_tokenizer):
        """Test handling of empty text."""
        from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference

        empty_text = ""
        eos_id = 999

        tokens, lens, texts = chunk_text_for_inference(
            text=empty_text,
            language="en",
            tokenizer_name="english_phoneme",
            text_tokenizer=mock_tokenizer,
            eos_token_id=eos_id,
        )

        # Empty text should still return something valid
        assert len(tokens) == 1
        assert tokens[0][-1].item() == eos_id
