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

import os
from typing import List, Type, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pytest

from nemo.collections.asr.parts.utils.audio_utils import SOUND_VELOCITY as sound_velocity
from nemo.collections.asr.parts.utils.audio_utils import (
    db2mag,
    estimated_coherence,
    generate_approximate_noise_field,
    get_segment_start,
    mag2db,
    pow2db,
    rms,
    select_channels,
    theoretical_coherence,
)


class TestSelectChannels:
    num_samples = 1000
    max_diff_tol = 1e-9

    @pytest.mark.unit
    @pytest.mark.parametrize("channel_selector", [None, 'average', 0, 1, [0, 1]])
    def test_single_channel_input(self, channel_selector: Type[Union[str, int, List[int]]]):
        """Cover the case with single-channel input signal.
        Channel selector should not do anything in this case.
        """
        golden_out = signal_in = np.random.rand(self.num_samples)

        if channel_selector not in [None, 0, 'average']:
            # Expect a failure if looking for a different channel when input is 1D
            with pytest.raises(ValueError):
                # UUT
                signal_out = select_channels(signal_in, channel_selector)
        else:
            # UUT
            signal_out = select_channels(signal_in, channel_selector)

            # Check difference
            max_diff = np.max(np.abs(signal_out - golden_out))
            assert max_diff < self.max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize("num_channels", [2, 4])
    @pytest.mark.parametrize("channel_selector", [None, 'average', 0, [1], [0, 1]])
    def test_multi_channel_input(self, num_channels: int, channel_selector: Type[Union[str, int, List[int]]]):
        """Cover the case with multi-channel input signal and single-
        or multi-channel output.
        """
        num_samples = 1000
        signal_in = np.random.rand(self.num_samples, num_channels)

        # calculate golden output
        if channel_selector is None:
            golden_out = signal_in
        elif channel_selector == 'average':
            golden_out = np.mean(signal_in, axis=1)
        else:
            golden_out = signal_in[:, channel_selector].squeeze()

        # UUT
        signal_out = select_channels(signal_in, channel_selector)

        # Check difference
        max_diff = np.max(np.abs(signal_out - golden_out))
        assert max_diff < self.max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize("num_channels", [1, 2])
    @pytest.mark.parametrize("channel_selector", [2, [1, 2]])
    def test_select_more_channels_than_available(
        self, num_channels: int, channel_selector: Type[Union[str, int, List[int]]]
    ):
        """This test is expecting the UUT to fail because we ask for more channels
        than available in the input signal.
        """
        num_samples = 1000
        signal_in = np.random.rand(self.num_samples, num_channels)

        # expect failure since we ask for more channels than available
        with pytest.raises(ValueError):
            # UUT
            signal_out = select_channels(signal_in, channel_selector)


class TestGenerateApproximateNoiseField:
    @pytest.mark.unit
    @pytest.mark.parametrize('num_mics', [5])
    @pytest.mark.parametrize('mic_spacing', [0.05])
    @pytest.mark.parametrize('fft_length', [512, 2048])
    @pytest.mark.parametrize('sample_rate', [8000, 16000])
    @pytest.mark.parametrize('field', ['spherical'])
    def test_theoretical_coherence_matrix(
        self, num_mics: int, mic_spacing: float, fft_length: int, sample_rate: float, field: str
    ):
        """Test calculation of a theoretical coherence matrix.
        """
        # test setup
        max_diff_tol = 1e-9

        # golden reference: spherical coherence
        num_subbands = fft_length // 2 + 1
        angular_freq = 2 * np.pi * sample_rate * np.arange(0, num_subbands) / fft_length
        golden_coherence = np.zeros((num_subbands, num_mics, num_mics))

        for p in range(num_mics):
            for q in range(num_mics):
                if p == q:
                    golden_coherence[:, p, q] = 1.0
                else:
                    if field == 'spherical':
                        dist_pq = abs(p - q) * mic_spacing
                        sinc_arg = angular_freq * dist_pq / sound_velocity
                        golden_coherence[:, p, q] = np.sinc(sinc_arg / np.pi)
                    else:
                        raise NotImplementedError(f'Field {field} not supported.')

        # assume linear arrray
        mic_positions = np.zeros((num_mics, 3))
        mic_positions[:, 0] = mic_spacing * np.arange(num_mics)

        # UUT
        uut_coherence = theoretical_coherence(
            mic_positions, sample_rate=sample_rate, fft_length=fft_length, field='spherical'
        )

        # Check difference
        max_diff = np.max(np.abs(uut_coherence - golden_coherence))
        assert max_diff < max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize('num_mics', [5])
    @pytest.mark.parametrize('mic_spacing', [0.10])
    @pytest.mark.parametrize('fft_length', [256, 512])
    @pytest.mark.parametrize('sample_rate', [8000, 16000])
    @pytest.mark.parametrize('field', ['spherical'])
    def test_generate_approximate_noise_field(
        self,
        num_mics: int,
        mic_spacing: float,
        fft_length: int,
        sample_rate: float,
        field: str,
        save_figures: bool = False,
    ):
        """Test approximate noise field with white noise as the input noise.
        """
        duration_in_sec = 20
        relative_mse_tol_dB = -30
        relative_mse_tol = 10 ** (relative_mse_tol_dB / 10)

        num_samples = sample_rate * duration_in_sec
        noise_signal = np.random.rand(num_samples, num_mics)
        # random channel-wise power scaling
        noise_signal *= np.random.randn(num_mics)

        # assume linear arrray
        mic_positions = np.zeros((num_mics, 3))
        mic_positions[:, 0] = mic_spacing * np.arange(num_mics)

        # UUT
        noise_field = generate_approximate_noise_field(mic_positions, noise_signal, sample_rate, fft_length=fft_length)

        # Compare the estimated coherence with the theoretical coherence
        analysis_fft_length = 256

        # reference
        golden_coherence = theoretical_coherence(
            mic_positions, sample_rate=sample_rate, fft_length=analysis_fft_length
        )

        # estimated
        N = librosa.stft(noise_field.transpose(), n_fft=analysis_fft_length)
        # (channel, subband, frame) -> (subband, frame, channel)
        N = N.transpose(1, 2, 0)
        uut_coherence = estimated_coherence(N)

        # Check difference
        relative_mse_real = np.mean((uut_coherence.real - golden_coherence) ** 2)
        assert relative_mse_real < relative_mse_tol
        relative_mse_imag = np.mean((uut_coherence.imag) ** 2)
        assert relative_mse_imag < relative_mse_tol

        if save_figures:
            # For debugging and visualization template
            figure_dir = os.path.expanduser('~/_coherence')
            if not os.path.exists(figure_dir):
                os.mkdir(figure_dir)

            freq = librosa.fft_frequencies(sr=sample_rate, n_fft=analysis_fft_length)
            freq = freq / 1e3  # kHz

            plt.figure(figsize=(7, 10))
            for n in range(1, num_mics):
                plt.subplot(num_mics - 1, 2, 2 * n - 1)
                plt.plot(freq, golden_coherence[:, 0, n].real, label='golden')
                plt.plot(freq, uut_coherence[:, 0, n].real, label='estimated')
                plt.title(f'Real(coherence), p=0, q={n}')
                plt.xlabel('f / kHz')
                plt.grid()
                plt.legend(loc='upper right')

                plt.subplot(num_mics - 1, 2, 2 * n)
                plt.plot(golden_coherence[:, 0, n].imag, label='golden')
                plt.plot(uut_coherence[:, 0, n].imag, label='estimated')
                plt.title(f'Imag(coherence), p=0, q={n}')
                plt.xlabel('f / kHz')
                plt.grid()
                plt.legend(loc='upper right')

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    figure_dir, f'num_mics_{num_mics}_sample_rate_{sample_rate}_fft_length_{fft_length}_{field}.png'
                )
            )
            plt.close()


class TestAudioUtilsElements:
    @pytest.mark.unit
    def test_rms(self):
        """Test RMS calculation
        """
        # setup
        A = np.random.rand()
        omega = 100
        n_points = 1000
        rms_threshold = 1e-4
        # prep data
        t = np.linspace(0, 2 * np.pi, n_points)
        x = A * np.cos(2 * np.pi * omega * t)
        # test
        x_rms = rms(x)
        golden_rms = A / np.sqrt(2)
        assert (
            np.abs(x_rms - golden_rms) < rms_threshold
        ), f'RMS not matching for A={A}, omega={omega}, n_point={n_points}'

    @pytest.mark.unit
    def test_db_conversion(self):
        """Test conversions to and from dB.
        """
        num_examples = 10
        abs_threshold = 1e-6

        mag = np.random.rand(num_examples)
        mag_db = mag2db(mag)

        assert all(np.abs(mag - 10 ** (mag_db / 20)) < abs_threshold)
        assert all(np.abs(db2mag(mag_db) - 10 ** (mag_db / 20)) < abs_threshold)
        assert all(np.abs(pow2db(mag ** 2) - mag_db) < abs_threshold)

    @pytest.mark.unit
    def test_get_segment_start(self):
        random_seed = 42
        num_examples = 50
        num_samples = 2000

        _rng = np.random.default_rng(seed=random_seed)

        for n in range(num_examples):
            # Generate signal
            signal = _rng.normal(size=num_samples)
            # Random start in the first half
            start = _rng.integers(low=0, high=num_samples // 2)
            # Random length
            end = _rng.integers(low=start, high=num_samples)
            # Selected segment
            segment = signal[start:end]

            # UUT
            estimated_start = get_segment_start(signal=signal, segment=segment)

            assert (
                estimated_start == start
            ), f'Example {n}: estimated start ({estimated_start}) not matching the actual start ({start})'
