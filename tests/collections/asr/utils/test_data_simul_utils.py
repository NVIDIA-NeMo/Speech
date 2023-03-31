# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts.utils.data_simulation_utils import (
    DataAnnotator,
    SpeechSampler,
    add_silence_to_alignments,
    binary_search_alignments,
    get_cleaned_base_path,
    get_split_points_in_alignments,
    normalize_audio,
    read_noise_manifest,
)


@pytest.fixture()
def annotator():
    cfg = get_data_simulation_configs()
    return DataAnnotator(cfg)


@pytest.fixture()
def sampler():
    cfg = get_data_simulation_configs()
    sampler = SpeechSampler(cfg)
    # Must get session-wise randomized silence/overlap mean
    sampler.get_session_overlap_mean()
    sampler.get_session_silence_mean()
    return sampler


def get_data_simulation_configs():
    config_dict = {
        'data_simulator': {
            'manifest_filepath': '???',
            'sr': 16000,
            'random_seed': 42,
            'multiprocessing_chunksize': 10000,
            'session_config': {'num_speakers': 4, 'num_sessions': 60, 'session_length': 600},
            'session_params': {
                'max_audio_read_sec': 20,
                'sentence_length_params': [0.4, 0.05],
                'dominance_var': 0.11,
                'min_dominance': 0.05,
                'turn_prob': 0.875,
                'min_turn_prob': 0.5,
                'mean_silence': 0.15,
                'mean_silence_var': 0.01,
                'per_silence_var': 900,
                'per_silence_min': 0.0,
                'per_silence_max': -1,
                'mean_overlap': 0.1,
                'mean_overlap_var': 0.01,
                'per_overlap_var': 900,
                'per_overlap_min': 0.0,
                'per_overlap_max': -1,
                'start_window': True,
                'window_type': 'hamming',
                'window_size': 0.05,
                'start_buffer': 0.1,
                'split_buffer': 0.1,
                'release_buffer': 0.1,
                'normalize': True,
                'normalization_type': 'equal',
                'normalization_var': 0.1,
                'min_volume': 0.75,
                'max_volume': 1.25,
                'end_buffer': 0.5,
            },
            'outputs': {
                'output_dir': '???',
                'output_filename': 'multispeaker_session',
                'overwrite_output': True,
                'output_precision': 3,
            },
            'background_noise': {
                'add_bg': False,
                'background_manifest': None,
                'num_noise_files': 10,
                'snr': 60,
                'snr_min': None,
            },
            'segment_augmentor': {
                'add_seg_aug': False,
                'augmentor': {'gain': {'prob': 0.5, 'min_gain_dbfs': -10.0, 'max_gain_dbfs': 10.0},},
            },
            'session_augmentor': {
                'add_sess_aug': False,
                'augmentor': {'white_noise': {'prob': 1.0, 'min_level': -90, 'max_level': -46},},
            },
            'speaker_enforcement': {'enforce_num_speakers': True, 'enforce_time': [0.25, 0.75]},
            'segment_manifest': {'window': 0.5, 'shift': 0.25, 'step_count': 50, 'deci': 3},
        }
    }
    return DictConfig(config_dict)


def generate_words_and_alignments(sample_index):
    if sample_index == 0:
        words = ['', 'hello', 'world']
        alignments = [0.5, 1.0, 1.5]
    elif sample_index == 1:
        words = ["", "stephanos", "dedalos", ""]
        alignments = [0.51, 1.31, 2.04, 2.215]
    elif sample_index == 2:
        words = ['', 'hello', 'world', '', 'welcome', 'to', 'nemo', '']
        alignments = [0.5, 1.0, 1.5, 1.7, 1.8, 2.2, 2.7, 2.8]
    else:
        raise ValueError(f"sample_index {sample_index} not supported")
    speaker_id = 'speaker_0'
    return words, alignments, speaker_id


class TestDataSimulatorUtils:
    # TODO: add tests for all util functions
    @pytest.mark.parametrize("max_audio_read_sec", [2.5, 3.5, 4.5])
    @pytest.mark.parametrize("min_alignment_count", [2, 3, 4])
    def test_binary_search_alignments(self, max_audio_read_sec, min_alignment_count):
        inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        alignments = [0.5, 11.0, 11.5, 12.0, 13.0, 14.0, 14.5, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 30, 40.0]
        offset_max = binary_search_alignments(inds, max_audio_read_sec, min_alignment_count, alignments)
        assert max_audio_read_sec <= alignments[-1 * min_alignment_count] - alignments[inds[offset_max]]

    @pytest.mark.parametrize("sample_len", [100, 16000])
    @pytest.mark.parametrize("gain", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_normalize_audio(self, sample_len, gain):
        array_raw = np.random.randn(sample_len)
        array_input = torch.from_numpy(gain * array_raw / np.max(np.abs(array_raw)))
        norm_array = normalize_audio(array_input)
        assert torch.max(torch.abs(norm_array)) == 1.0
        assert torch.min(torch.abs(norm_array)) < 1.0

    @pytest.mark.parametrize("output_dir", [os.path.join(os.getcwd(), "test_dir")])
    def test_get_cleaned_base_path(self, output_dir):
        result_path = get_cleaned_base_path(output_dir, overwrite_output=True)
        assert os.path.exists(result_path) and not os.path.isfile(result_path)
        result_path = get_cleaned_base_path(output_dir, overwrite_output=False)
        assert os.path.exists(result_path) and not os.path.isfile(result_path)
        os.rmdir(result_path)
        assert not os.path.exists(result_path)

    @pytest.mark.parametrize(
        "words, alignments, answers",
        [
            (['', 'hello', 'world'], [0.5, 1.0, 1.5], [[0, 16000.0]]),
            (
                ['', 'hello', 'world', '', 'welcome', 'to', 'nemo', ''],
                [0.27, 1.0, 1.7, 2.7, 2.8, 3.2, 3.7, 3.9],
                [[0, (1.7 + 0.5) * 16000], [(2.7 - 0.5) * 16000, (3.9 - 0.27) * 16000]],
            ),
        ],
    )
    @pytest.mark.parametrize("sr", [16000])
    @pytest.mark.parametrize("split_buffer", [0.5])
    @pytest.mark.parametrize("new_start", [0.0])
    def test_get_split_points_in_alignments(self, words, alignments, sr, new_start, split_buffer, answers):
        sentence_audio_len = sr * (alignments[-1] - alignments[0])
        splits = get_split_points_in_alignments(words, alignments, split_buffer, sr, sentence_audio_len, new_start)
        assert len(splits) == len(answers)
        for k, interval in enumerate(splits):
            assert abs(answers[k][0] - interval[0]) < 1e-4
            assert abs(answers[k][1] - interval[1]) < 1e-4

    @pytest.mark.parametrize(
        "alignments, words", [(['hello', 'world'], [1.0, 1.5]), (['', 'hello', 'world'], [0.0, 1.0, 1.5])]
    )
    def test_add_silence_to_alignments(self, alignments, words):
        """
        Test add_silence_to_alignments function.
        """
        audio_manifest = {
            'audio_filepath': 'test.wav',
            'alignments': alignments,
            'words': words,
        }
        audio_manifest = add_silence_to_alignments(audio_manifest)
        if words[0] == '':
            assert audio_manifest['alignments'] == [0.0] + alignments
            assert audio_manifest['words'] == [''] + words
        else:
            assert audio_manifest['alignments'] == alignments
            assert audio_manifest['words'] == words


class TestDataAnnotator:
    def test_init(self, annotator):
        assert isinstance(annotator, DataAnnotator)

    def test_create_new_rttm_entry(self, annotator):
        words, alignments, speaker_id = generate_words_and_alignments(sample_index=0)
        start, end = alignments[0], alignments[-1]
        rttm_list = annotator.create_new_rttm_entry(
            words=words, alignments=alignments, start=start, end=end, speaker_id=speaker_id
        )
        assert rttm_list[0] == f"{start} {end} {speaker_id}"

    def test_create_new_json_entry(self, annotator):
        words, alignments, speaker_id = generate_words_and_alignments(sample_index=0)
        start, end = alignments[0], alignments[-1]
        test_wav_filename = '/path/to/test_wav_filename.wav'
        test_rttm_filename = '/path/to/test_rttm_filename.rttm'
        test_ctm_filename = '/path/to/test_ctm_filename.ctm'
        text = " ".join(words)

        one_line_json_dict = annotator.create_new_json_entry(
            text=text,
            wav_filename=test_wav_filename,
            start=start,
            length=end - start,
            speaker_id=speaker_id,
            rttm_filepath=test_rttm_filename,
            ctm_filepath=test_ctm_filename,
        )
        start = round(float(start), annotator._params.data_simulator.outputs.output_precision)
        length = round(float(end - start), annotator._params.data_simulator.outputs.output_precision)
        meta = {
            "audio_filepath": test_wav_filename,
            "offset": start,
            "duration": length,
            "label": speaker_id,
            "text": text,
            "num_speakers": annotator._params.data_simulator.session_config.num_speakers,
            "rttm_filepath": test_rttm_filename,
            "ctm_filepath": test_ctm_filename,
            "uem_filepath": None,
        }
        assert one_line_json_dict == meta

    def test_create_new_ctm_entry(self, annotator):
        words, alignments, speaker_id = generate_words_and_alignments(sample_index=0)
        start = alignments[0]
        session_name = 'test_session'
        ctm_list = annotator.create_new_ctm_entry(
            words=words, alignments=alignments, session_name=session_name, speaker_id=speaker_id, start=start
        )
        assert ctm_list[0] == (
            alignments[1],
            f"{session_name} {speaker_id} {alignments[1]} {alignments[1]-alignments[0]} {words[1]} 0\n",
        )
        assert ctm_list[1] == (
            alignments[2],
            f"{session_name} {speaker_id} {alignments[2]} {alignments[2]-alignments[1]} {words[2]} 0\n",
        )


class TestSpeechSampler:
    def test_init(self, sampler):
        assert isinstance(sampler, SpeechSampler)

    def test_init_overlap_params(self, sampler):
        sampler._init_overlap_params()
        assert sampler.per_silence_min_len is not None
        assert sampler.per_silence_max_len is not None
        assert type(sampler.per_silence_min_len) == int
        assert type(sampler.per_silence_max_len) == int

    def test_init_silence_params(self, sampler):
        sampler._init_overlap_params()
        assert sampler.per_overlap_min_len is not None
        assert sampler.per_overlap_max_len is not None
        assert type(sampler.per_overlap_min_len) == int
        assert type(sampler.per_overlap_max_len) == int

    @pytest.mark.parametrize("mean", [0.1, 0.2, 0.3])
    @pytest.mark.parametrize("var", [0.05, 0.07])
    def test_get_session_silence_mean_pass(self, sampler, mean, var):
        sampler.mean_silence = mean
        sampler.mean_silence_var = var
        sampled_silence_mean = sampler.get_session_silence_mean()
        assert 0 <= sampled_silence_mean <= 1

    @pytest.mark.parametrize("mean", [0.5])
    @pytest.mark.parametrize("var", [0.5, 0.6])
    def test_get_session_silence_mean_fail(self, sampler, mean, var):
        """
        This test should raise `ValueError` because `mean_silence_var` 
        should be less than `mean_silence * (1 - mean_silence)`.
        """
        sampler.mean_silence = mean
        sampler.mean_silence_var = var
        with pytest.raises(ValueError) as execinfo:
            sampler.get_session_silence_mean()
        assert "ValueError" in str(execinfo) and "mean_silence_var" in str(execinfo)

    @pytest.mark.parametrize("mean", [0.1, 0.2, 0.3])
    @pytest.mark.parametrize("var", [0.05, 0.07])
    def test_get_session_overlap_mean_pass(self, sampler, mean, var):
        sampler.mean_overlap = mean
        sampler.mean_overlap_var = var
        sampled_overlap_mean = sampler.get_session_overlap_mean()
        assert 0 <= sampled_overlap_mean <= 1

    @pytest.mark.parametrize("mean", [0.4, 0.5])
    @pytest.mark.parametrize("var", [0.3, 0.8])
    def test_get_session_overlap_mean_fail(self, sampler, mean, var):
        """
        This test should raise `ValueError` because `mean_overlap_var` 
        should be less than `mean_overlap * (1 - mean_overlap)`.
        """
        sampler.mean_overlap = mean
        sampler.mean_overlap_var = var
        sampler._params = DictConfig(sampler._params)
        with pytest.raises(ValueError) as execinfo:
            sampler.get_session_overlap_mean()
        assert "ValueError" in str(execinfo) and "mean_overlap_var" in str(execinfo)

    @pytest.mark.parametrize("non_silence_len_samples", [16000, 32000])
    @pytest.mark.parametrize("running_overlap_len_samples", [8000, 12000])
    def test_sample_from_overlap_model(self, sampler, non_silence_len_samples, running_overlap_len_samples):
        sampler.get_session_overlap_mean()
        sampler.running_overlap_len_samples = running_overlap_len_samples
        overlap_amount = sampler.sample_from_overlap_model(non_silence_len_samples=non_silence_len_samples)
        assert type(overlap_amount) == int
        assert 0 <= overlap_amount

    @pytest.mark.parametrize("running_len_samples", [8000, 16000])
    @pytest.mark.parametrize("running_overlap_len_samples", [8000, 12000])
    def test_sample_from_silence_model(self, sampler, running_len_samples, running_overlap_len_samples):
        sampler.get_session_silence_mean()
        self.running_overlap_len_samples = running_overlap_len_samples
        silence_amount = sampler.sample_from_silence_model(running_len_samples=running_len_samples)
        assert type(silence_amount) == int
        assert 0 <= silence_amount

    @pytest.mark.with_downloads()
    @pytest.mark.parametrize("num_noise_files", [1, 2, 4])
    def test_sample_noise_manifest(self, sampler, num_noise_files, test_data_dir):
        sampler.num_noise_files = num_noise_files
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/an4_val.json'))
        noise_manifest = read_noise_manifest(add_bg=True, background_manifest=manifest_path)
        sampled_noise_manifests = sampler.sample_noise_manifest(noise_manifest=noise_manifest)
        assert len(sampled_noise_manifests) == num_noise_files

    @pytest.mark.parametrize("running_speech_len_samples", [32000, 64000])
    @pytest.mark.parametrize("running_overlap_len_samples", [16000, 32000])
    @pytest.mark.parametrize("running_len_samples", [64000, 96000])
    @pytest.mark.parametrize("non_silence_len_samples", [16000, 32000])
    def test_silence_vs_overlap_selector(
        self,
        sampler,
        running_overlap_len_samples,
        running_speech_len_samples,
        running_len_samples,
        non_silence_len_samples,
    ):
        sampler.running_overlap_len_samples = running_overlap_len_samples
        sampler.running_speech_len_samples = running_speech_len_samples
        add_overlap = sampler.silence_vs_overlap_selector(
            running_len_samples=running_len_samples, non_silence_len_samples=non_silence_len_samples
        )
        assert type(add_overlap) == bool
