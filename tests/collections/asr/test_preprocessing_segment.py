# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import tempfile
from typing import List, Type, Union

import numpy as np
import pytest
import soundfile as sf

from nemo.collections.asr.parts.preprocessing.perturb import NoisePerturbation, SilencePerturbation
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.audio_utils import select_channels


class TestAudioSegment:

    sample_rate = 16000
    signal_duration_sec = 2
    max_diff_tol = 1e-9

    @property
    def num_samples(self):
        return self.sample_rate * self.signal_duration_sec

    @pytest.mark.unit
    @pytest.mark.parametrize("num_channels", [1, 4])
    @pytest.mark.parametrize("channel_selector", [None, 'average', 0, 1, [0, 1]])
    def test_init_single_channel(self, num_channels: int, channel_selector: Type[Union[str, int, List[int]]]):
        """Test the constructor directly.
        """
        if num_channels == 1:
            # samples is a one-dimensional vector for single-channel signal
            samples = np.random.rand(self.num_samples)
        else:
            samples = np.random.rand(self.num_samples, num_channels)

        if (isinstance(channel_selector, int) and channel_selector >= num_channels) or (
            isinstance(channel_selector, list) and max(channel_selector) >= num_channels
        ):
            # Expect a failure if looking for a different channel when input is 1D
            with pytest.raises(ValueError):
                # Construct UUT
                uut = AudioSegment(samples=samples, sample_rate=self.sample_rate, channel_selector=channel_selector)
        else:
            # Construct UUT
            uut = AudioSegment(samples=samples, sample_rate=self.sample_rate, channel_selector=channel_selector)

            # Create golden reference
            # Note: AudioSegment converts input samples to float32
            golden_samples = select_channels(samples.astype('float32'), channel_selector)
            expected_num_channels = 1 if golden_samples.ndim == 1 else golden_samples.shape[1]

            # Test UUT
            assert uut.num_channels == expected_num_channels
            assert uut.num_samples == self.num_samples
            assert uut.sample_rate == self.sample_rate
            assert uut.duration == self.signal_duration_sec
            max_diff = np.max(np.abs(uut.samples - golden_samples))
            assert max_diff < self.max_diff_tol

            # Test zero padding
            pad_length = 42
            uut.pad(pad_length, symmetric=False)
            # compare to golden references
            assert uut.num_samples == self.num_samples + pad_length
            assert np.all(uut.samples[-pad_length:] == 0.0)
            max_diff = np.max(np.abs(uut.samples[:-pad_length] - golden_samples))
            assert max_diff < self.max_diff_tol

            # Test subsegment
            start_time = 0.2 * self.signal_duration_sec
            end_time = 0.5 * self.signal_duration_sec
            uut.subsegment(start_time=start_time, end_time=end_time)
            # compare to golden references
            start_sample = int(round(start_time * self.sample_rate))
            end_sample = int(round(end_time * self.sample_rate))
            max_diff = np.max(np.abs(uut.samples - golden_samples[start_sample:end_sample]))
            assert max_diff < self.max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize("num_channels", [1, 4])
    @pytest.mark.parametrize("channel_selector", [None, 'average', 0])
    def test_from_file(self, num_channels, channel_selector):
        """Test loading a signal from a file.
        """
        with tempfile.TemporaryDirectory() as test_dir:
            # Prepare a wav file
            audio_file = os.path.join(test_dir, 'audio.wav')
            if num_channels == 1:
                # samples is a one-dimensional vector for single-channel signal
                samples = np.random.rand(self.num_samples)
            else:
                samples = np.random.rand(self.num_samples, num_channels)
            sf.write(audio_file, samples, self.sample_rate, 'float')

            # Create UUT
            uut = AudioSegment.from_file(audio_file, channel_selector=channel_selector)

            # Create golden reference
            # Note: AudioSegment converts input samples to float32
            golden_samples = select_channels(samples.astype('float32'), channel_selector)
            expected_num_channels = 1 if golden_samples.ndim == 1 else golden_samples.shape[1]

            # Test UUT
            assert uut.num_channels == expected_num_channels
            assert uut.num_samples == self.num_samples
            assert uut.sample_rate == self.sample_rate
            assert uut.duration == self.signal_duration_sec
            max_diff = np.max(np.abs(uut.samples - golden_samples))
            assert max_diff < self.max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize("data_channels", [1, 4])
    @pytest.mark.parametrize("noise_channels", [1, 4])
    def test_noise_perturb_channels(self, data_channels, noise_channels):
        """Test loading a signal from a file.
        """
        with tempfile.TemporaryDirectory() as test_dir:
            # Prepare a wav file
            audio_file = os.path.join(test_dir, 'audio.wav')
            if data_channels == 1:
                # samples is a one-dimensional vector for single-channel signal
                samples = np.random.rand(self.num_samples)
            else:
                samples = np.random.rand(self.num_samples, data_channels)
            sf.write(audio_file, samples, self.sample_rate, 'float')

            noise_file = os.path.join(test_dir, 'noise.wav')
            if noise_channels == 1:
                # samples is a one-dimensional vector for single-channel signal
                samples = np.random.rand(self.num_samples)
            else:
                samples = np.random.rand(self.num_samples, noise_channels)
            sf.write(noise_file, samples, self.sample_rate, 'float')

            manifest_file = os.path.join(test_dir, 'noise_manifest.json')
            with open(manifest_file, 'w') as fout:
                item = {'audio_filepath': os.path.abspath(noise_file), 'label': '-', 'duration': 0.1, 'offset': 0.0}
                fout.write(f'{json.dumps(item)}\n')

            perturber = NoisePerturbation(manifest_file)
            audio = AudioSegment.from_file(audio_file)
            noise = AudioSegment.from_file(noise_file)

            if data_channels == noise_channels:
                try:
                    _ = perturber.perturb_with_input_noise(audio, noise, ref_mic=0)
                except ValueError as e:
                    assert False, "perturb_with_input_noise failed with ref_mic=0"

                with pytest.raises(ValueError):
                    _ = perturber.perturb_with_input_noise(audio, noise, ref_mic=data_channels)

                try:
                    _ = perturber.perturb_with_foreground_noise(audio, noise, ref_mic=0)
                except ValueError as e:
                    assert False, "perturb_with_foreground_noise failed with ref_mic=0"

                with pytest.raises(ValueError):
                    _ = perturber.perturb_with_foreground_noise(audio, noise, ref_mic=data_channels)
            else:
                with pytest.raises(ValueError):
                    _ = perturber.perturb_with_input_noise(audio, noise)
                with pytest.raises(ValueError):
                    _ = perturber.perturb_with_foreground_noise(audio, noise)

    def test_silence_perturb(self):
        """Test loading a signal from a file and apply silence perturbation
        """
        with tempfile.TemporaryDirectory() as test_dir:
            # Prepare a wav file
            audio_file = os.path.join(test_dir, 'audio.wav')
            # samples is a one-dimensional vector for single-channel signal
            samples = np.random.rand(self.num_samples)
            sf.write(audio_file, samples, self.sample_rate, 'float')

            dur = 2
            perturber = SilencePerturbation(
                min_start_silence_secs=dur,
                max_start_silence_secs=dur,
                min_end_silence_secs=dur,
                max_end_silence_secs=dur,
            )

            audio = AudioSegment.from_file(audio_file)
            ori_audio_len = len(audio._samples)
            _ = perturber.perturb(audio)

            assert len(audio._samples) == ori_audio_len + 2 * dur * self.sample_rate
