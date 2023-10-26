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
