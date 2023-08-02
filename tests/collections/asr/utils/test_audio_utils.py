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
from collections import namedtuple
from typing import List, Type, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy
import soundfile as sf
import torch

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.audio_utils import SOUND_VELOCITY as sound_velocity
from nemo.collections.asr.parts.utils.audio_utils import (
    calculate_sdr_numpy,
    convmtx_mc_numpy,
    db2mag,
    estimated_coherence,
    generate_approximate_noise_field,
    get_segment_start,
    mag2db,
    pow2db,
    rms,
    select_channels,
    theoretical_coherence,
    toeplitz,
)


class TestAudioSegment:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_channels, channel_selectors", [(1, [None, 'average', 0]), (3, [None, 'average', 0, 1, [0, 1]]),]
    )
    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22500])
    def test_audio_segment_from_file(self, tmpdir, num_channels, channel_selectors, sample_rate):
        """Test loading and audio signal from a file.
        """
        signal_len_sec = 4
        num_samples = signal_len_sec * sample_rate
        num_examples = 10
        rtol, atol = 1e-5, 1e-6

        for n in range(num_examples):
            # Create a test vector
            audio_file = os.path.join(tmpdir, f'test_audio_{n:02}.wav')
            samples = np.random.randn(num_samples, num_channels)
            sf.write(audio_file, samples, sample_rate, 'float')

            for channel_selector in channel_selectors:
                if channel_selector is None:
                    ref_samples = samples
                elif isinstance(channel_selector, int) or isinstance(channel_selector, list):
                    ref_samples = samples[:, channel_selector]
                elif channel_selector == 'average':
                    ref_samples = np.mean(samples, axis=1)
                else:
                    raise ValueError(f'Unexpected value of channel_selector {channel_selector}')

                # 1) Load complete audio
                # Reference
                ref_samples = ref_samples.squeeze()
                ref_channels = 1 if ref_samples.ndim == 1 else ref_samples.shape[1]

                # UUT
                audio_segment = AudioSegment.from_file(audio_file, channel_selector=channel_selector)

                # Test
                assert (
                    audio_segment.sample_rate == sample_rate
                ), f'channel_selector {channel_selector}, sample rate not matching: {audio_segment.sample_rate} != {sample_rate}'
                assert (
                    audio_segment.num_channels == ref_channels
                ), f'channel_selector {channel_selector}, num channels not matching: {audio_segment.num_channels} != {ref_channels}'
                assert audio_segment.num_samples == len(
                    ref_samples
                ), f'channel_selector {channel_selector}, num samples not matching: {audio_segment.num_samples} != {len(ref_samples)}'
                assert np.allclose(
                    audio_segment.samples, ref_samples, rtol=rtol, atol=atol
                ), f'channel_selector {channel_selector}, samples not matching'

                # 2) Load a random segment
                offset = 0.45 * np.random.rand() * signal_len_sec
                duration = 0.45 * np.random.rand() * signal_len_sec

                # Reference
                start = int(offset * sample_rate)
                end = start + int(duration * sample_rate)
                ref_samples = ref_samples[start:end, ...]

                # UUT
                audio_segment = AudioSegment.from_file(
                    audio_file, offset=offset, duration=duration, channel_selector=channel_selector
                )

                # Test
                assert (
                    audio_segment.sample_rate == sample_rate
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, sample rate not matching: {audio_segment.sample_rate} != {sample_rate}'
                assert (
                    audio_segment.num_channels == ref_channels
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, num channels not matching: {audio_segment.num_channels} != {ref_channels}'
                assert audio_segment.num_samples == len(
                    ref_samples
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, num samples not matching: {audio_segment.num_samples} != {len(ref_samples)}'
                assert np.allclose(
                    audio_segment.samples, ref_samples, rtol=rtol, atol=atol
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, samples not matching'

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_channels, channel_selectors", [(1, [None, 'average', 0]), (3, [None, 'average', 0, 1, [0, 1]]),]
    )
    @pytest.mark.parametrize("offset", [0, 1.5])
    @pytest.mark.parametrize("duration", [1, 2])
    def test_audio_segment_multichannel_with_list(self, tmpdir, num_channels, channel_selectors, offset, duration):
        """Test loading an audio signal from a list of single-channel files.
        """
        sample_rate = 16000
        signal_len_sec = 5
        num_samples = signal_len_sec * sample_rate
        rtol, atol = 1e-5, 1e-6

        # Random samples
        samples = np.random.rand(num_samples, num_channels)

        # Save audio
        audio_files = []
        for m in range(num_channels):
            a_file = os.path.join(tmpdir, f'ch_{m}.wav')
            sf.write(a_file, samples[:, m], sample_rate)
            audio_files.append(a_file)
        mc_file = os.path.join(tmpdir, f'mc.wav')
        sf.write(mc_file, samples, sample_rate)

        for channel_selector in channel_selectors:

            # UUT: loading audio from a list of files
            uut_segment = AudioSegment.from_file(
                audio_file=audio_files, offset=offset, duration=duration, channel_selector=channel_selector
            )

            # Reference: load from the original file
            ref_segment = AudioSegment.from_file(
                audio_file=mc_file, offset=offset, duration=duration, channel_selector=channel_selector
            )

            # Check
            assert (
                uut_segment.sample_rate == ref_segment.sample_rate
            ), f'channel_selector {channel_selector}: expecting {ref_segment.sample_rate}, but UUT segment has {uut_segment.sample_rate}'
            assert (
                uut_segment.num_samples == ref_segment.num_samples
            ), f'channel_selector {channel_selector}: expecting {ref_segment.num_samples}, but UUT segment has {uut_segment.num_samples}'
            assert np.allclose(
                uut_segment.samples, ref_segment.samples, rtol=rtol, atol=atol
            ), f'channel_selector {channel_selector}: samples not matching'

        # Try to get a channel that is out of range.
        with pytest.raises(RuntimeError, match="Channel cannot be selected"):
            AudioSegment.from_file(audio_file=audio_files, channel_selector=num_channels)

        if num_channels > 1:
            # Try to load a list of multichannel files
            # This is expected to fail since we only support loading a single-channel signal
            # from each file when audio_file is a list
            with pytest.raises(RuntimeError, match="Expecting a single-channel audio signal"):
                AudioSegment.from_file(audio_file=[mc_file, mc_file])

            with pytest.raises(RuntimeError, match="Expecting a single-channel audio signal"):
                AudioSegment.from_file(audio_file=[mc_file, mc_file], channel_selector=0)

    @pytest.mark.unit
    @pytest.mark.parametrize("target_sr", [8000, 16000])
    def test_audio_segment_trim_match(self, tmpdir, target_sr):
        """Test loading and audio signal from a file matches when using a path and a list
        for different target_sr, int_values and trim setups.
        """
        sample_rate = 24000
        signal_len_sec = 2
        num_samples = signal_len_sec * sample_rate
        num_examples = 10
        rtol, atol = 1e-5, 1e-6

        TrimSetup = namedtuple("TrimSetup", "ref top_db frame_length hop_length")
        trim_setups = []
        trim_setups.append(TrimSetup(np.max, 10, 2048, 1024))
        trim_setups.append(TrimSetup(1.0, 35, 2048, 1024))
        trim_setups.append(TrimSetup(0.8, 45, 2048, 1024))

        for n in range(num_examples):
            # Create a test vector
            audio_file = os.path.join(tmpdir, f'test_audio_{n:02}.wav')
            samples = np.random.randn(num_samples)
            # normalize
            samples = samples / np.max(samples)
            # apply random scaling and window to have some samples cut by trim
            samples = np.random.rand() * np.hanning(num_samples) * samples
            sf.write(audio_file, samples, sample_rate, 'float')

            for trim_setup in trim_setups:
                # UUT 1: load from a path
                audio_segment_1 = AudioSegment.from_file(
                    audio_file,
                    target_sr=target_sr,
                    trim=True,
                    trim_ref=trim_setup.ref,
                    trim_top_db=trim_setup.top_db,
                    trim_frame_length=trim_setup.frame_length,
                    trim_hop_length=trim_setup.hop_length,
                )

                # UUT 2: load from a list
                audio_segment_2 = AudioSegment.from_file(
                    [audio_file],
                    target_sr=target_sr,
                    trim=True,
                    trim_ref=trim_setup.ref,
                    trim_top_db=trim_setup.top_db,
                    trim_frame_length=trim_setup.frame_length,
                    trim_hop_length=trim_setup.hop_length,
                )

                # Test
                assert audio_segment_1 == audio_segment_2, f'trim setup {trim_setup}, loaded segments not matching'


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
        noise_field = generate_approximate_noise_field(
            mic_positions, noise_signal, sample_rate=sample_rate, field=field, fft_length=fft_length
        )

        # Compare the estimated coherence with the theoretical coherence

        # reference
        golden_coherence = theoretical_coherence(
            mic_positions, sample_rate=sample_rate, field=field, fft_length=fft_length
        )

        # estimated
        N = librosa.stft(noise_field.transpose(), n_fft=fft_length)
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

            freq = librosa.fft_frequencies(sr=sample_rate, n_fft=fft_length)
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

    @pytest.mark.unit
    def test_calculate_sdr_numpy(self):
        atol = 1e-6
        random_seed = 42
        num_examples = 50
        num_samples = 2000

        _rng = np.random.default_rng(seed=random_seed)

        for n in range(num_examples):
            # Generate signal
            target = _rng.normal(size=num_samples)
            # Adjust the estimate
            golden_sdr = _rng.integers(low=-10, high=10)
            estimate = target * (1 + 10 ** (-golden_sdr / 20))

            # UUT
            estimated_sdr = calculate_sdr_numpy(estimate=estimate, target=target, remove_mean=False)

            assert np.isclose(
                estimated_sdr, golden_sdr, atol=atol
            ), f'Example {n}: estimated ({estimated_sdr}) not matching the actual value ({golden_sdr})'

            # Add random mean and use remove_mean=True
            # SDR should not change
            target += _rng.uniform(low=-10, high=10)
            estimate += _rng.uniform(low=-10, high=10)

            # UUT
            estimated_sdr = calculate_sdr_numpy(estimate=estimate, target=target, remove_mean=True)

            assert np.isclose(
                estimated_sdr, golden_sdr, atol=atol
            ), f'Example {n}: estimated ({estimated_sdr}) not matching the actual value ({golden_sdr})'

    @pytest.mark.unit
    def test_calculate_sdr_numpy_scale_invariant(self):
        atol = 1e-6
        random_seed = 42
        num_examples = 50
        num_samples = 2000

        _rng = np.random.default_rng(seed=random_seed)

        for n in range(num_examples):
            # Generate signal
            target = _rng.normal(size=num_samples)
            # Adjust the estimate
            estimate = target + _rng.uniform(low=0.01, high=1) * _rng.normal(size=target.size)

            # scaled target
            target_scaled = target / (np.linalg.norm(target) + 1e-16)
            target_scaled = np.sum(estimate * target_scaled) * target_scaled

            golden_sdr = calculate_sdr_numpy(
                estimate=estimate, target=target_scaled, scale_invariant=False, remove_mean=False
            )

            # UUT
            estimated_sdr = calculate_sdr_numpy(
                estimate=estimate, target=target, scale_invariant=True, remove_mean=False
            )

            print(golden_sdr, estimated_sdr)

            assert np.isclose(
                estimated_sdr, golden_sdr, atol=atol
            ), f'Example {n}: estimated ({estimated_sdr}) not matching the actual value ({golden_sdr})'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 3])
    @pytest.mark.parametrize('filter_length', [10])
    @pytest.mark.parametrize('delay', [0, 5])
    def test_convmtx_mc(self, num_channels: int, filter_length: int, delay: int):
        """Test convmtx against convolve and sum.
        Multiplication of convmtx_mc of input with a vectorized multi-channel filter
        should match the sum of convolution of each input channel with the corresponding
        filter.
        """
        atol = 1e-6
        random_seed = 42
        num_examples = 10
        num_samples = 2000

        _rng = np.random.default_rng(seed=random_seed)

        for n in range(num_examples):
            x = _rng.normal(size=(num_samples, num_channels))
            f = _rng.normal(size=(filter_length, num_channels))

            CM = convmtx_mc_numpy(x=x, filter_length=filter_length, delay=delay)

            # Multiply convmtx_mc with the vectorized filter
            uut = CM @ f.transpose().reshape(-1, 1)
            uut = uut.squeeze(1)

            # Calculate reference as sum of convolutions
            golden_ref = 0
            for m in range(num_channels):
                x_m_delayed = np.hstack([np.zeros(delay), x[:, m]])
                golden_ref += np.convolve(x_m_delayed, f[:, m], mode='full')[: len(x)]

            assert np.allclose(uut, golden_ref, atol=atol), f'Example {n}: UUT not matching the reference.'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 3])
    @pytest.mark.parametrize('filter_length', [10])
    @pytest.mark.parametrize('num_samples', [10, 100])
    def test_toeplitz(self, num_channels: int, filter_length: int, num_samples: int):
        """Test construction of a Toeplitz matrix for a given signal.
        """
        atol = 1e-6
        random_seed = 42
        num_batches = 10
        batch_size = 8

        _rng = np.random.default_rng(seed=random_seed)

        for n in range(num_batches):
            x = _rng.normal(size=(batch_size, num_channels, num_samples))

            # Construct Toeplitz matrix
            Tx = toeplitz(x=torch.tensor(x))

            # Compare against the reference
            for b in range(batch_size):
                for m in range(num_channels):
                    T_ref = scipy.linalg.toeplitz(x[b, m, ...])

                    assert np.allclose(
                        Tx[b, m, ...].cpu().numpy(), T_ref, atol=atol
                    ), f'Example {n}: not matching the reference for (b={b}, m={m}), .'
