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

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy
import torch

from nemo.collections.audio.parts.utils.audio import SOUND_VELOCITY as sound_velocity
from nemo.collections.audio.parts.utils.audio import (
    calculate_sdr_numpy,
    convmtx_mc_numpy,
    covariance_matrix,
    db2mag,
    estimated_coherence,
    generate_approximate_noise_field,
    get_segment_start,
    mag2db,
    pow2db,
    rms,
    theoretical_coherence,
    toeplitz,
)


try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


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
        """Test calculation of a theoretical coherence matrix."""
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
        """Test approximate noise field with white noise as the input noise."""
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
        """Test RMS calculation"""
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
        """Test conversions to and from dB."""
        num_examples = 10
        abs_threshold = 1e-6

        mag = np.random.rand(num_examples)
        mag_db = mag2db(mag)

        assert all(np.abs(mag - 10 ** (mag_db / 20)) < abs_threshold)
        assert all(np.abs(db2mag(mag_db) - 10 ** (mag_db / 20)) < abs_threshold)
        assert all(np.abs(pow2db(mag**2) - mag_db) < abs_threshold)

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
        """Test construction of a Toeplitz matrix for a given signal."""
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


class TestCovarianceMatrix:
    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize('num_channels', [1, 3])
    @pytest.mark.parametrize('num_freq', [17, 33])
    @pytest.mark.parametrize('use_mask', [True, False])
    @pytest.mark.parametrize('normalize_mask', [True, False])
    @pytest.mark.parametrize('mask_type', ['real', 'complex', 'bool'])
    def test_calculate_covariance_matrix_vs_psd(
        self, num_channels: int, num_freq: int, use_mask: bool, normalize_mask: bool, mask_type: str
    ):
        """Test against reference calculation using torchaudio."""
        # Element-wise relative tolerance
        atol = 1e-5

        # Random generator
        random_seed = 42
        rng = torch.Generator()
        rng.manual_seed(random_seed)

        num_examples = 10
        batch_size, num_steps = 8, 100

        # Reference calculation of multichannel covariance matrix
        psd_ref = torchaudio.transforms.PSD(multi_mask=False, normalize=normalize_mask, eps=1e-8)

        for n in range(num_examples):

            input = torch.randn(batch_size, num_channels, num_freq, num_steps, dtype=torch.cfloat, generator=rng)

            if mask_type == 'real':
                mask = torch.rand(batch_size, num_freq, num_steps, dtype=torch.float, generator=rng)
            elif mask_type == 'complex':
                mask = torch.rand(batch_size, num_freq, num_steps, dtype=torch.cfloat, generator=rng)
            elif mask_type == 'bool':
                mask = torch.randint(0, 2, (batch_size, num_freq, num_steps), dtype=torch.bool, generator=rng)
            else:
                raise ValueError(f'Mask type {mask_type} not supported.')

            # UUT
            uut = covariance_matrix(x=input, mask=mask if use_mask else None, normalize_mask=normalize_mask)

            # Reference
            ref = psd_ref(specgram=input, mask=mask if use_mask else None)
            if not use_mask:
                # torchaudio is summing over time, divide by num_steps to have an average over time
                ref = ref / num_steps

            # Check if the UUT matches the reference
            assert torch.allclose(uut, ref, atol=atol), f'Example {n}: UUT not matching the reference.'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 3])
    @pytest.mark.parametrize('num_freq', [3, 10])
    @pytest.mark.parametrize('use_mask', [True, False])
    @pytest.mark.parametrize('normalize_mask', [True, False])
    @pytest.mark.parametrize('mask_type', ['real', 'complex', 'bool'])
    def test_calculate_covariance_matrix(
        self, num_channels: int, num_freq: int, use_mask: bool, normalize_mask: bool, mask_type: str
    ):
        """Test against simple reference calculation."""
        # Element-wise relative tolerance
        atol = 1e-5

        # Random generator
        random_seed = 42
        rng = torch.Generator()
        rng.manual_seed(random_seed)

        num_examples = 10
        batch_size, num_steps = 8, 10

        for n in range(num_examples):

            input = torch.randn(batch_size, num_channels, num_freq, num_steps, dtype=torch.cfloat, generator=rng)

            if mask_type == 'real':
                mask = torch.rand(batch_size, num_freq, num_steps, dtype=torch.float, generator=rng)
            elif mask_type == 'complex':
                mask = torch.rand(batch_size, num_freq, num_steps, dtype=torch.cfloat, generator=rng)
            elif mask_type == 'bool':
                mask = torch.randint(0, 2, (batch_size, num_freq, num_steps), dtype=torch.bool, generator=rng)
            else:
                raise ValueError(f'Mask type {mask_type} not supported.')

            # UUT
            uut = covariance_matrix(x=input, mask=mask if use_mask else None, normalize_mask=normalize_mask)

            # Reference calculation
            ref = torch.zeros(batch_size, num_freq, num_channels, num_channels, num_steps, dtype=torch.cfloat)

            # calculate x(t) x(t)^H for each time step
            for b in range(batch_size):
                for f in range(num_freq):
                    for t in range(num_steps):
                        ref[b, f, :, :, t] = torch.outer(input[b, :, f, t], input[b, :, f, t].conj())

            # aggregate over time
            if use_mask:
                # mask: weighted sum over time
                if normalize_mask:
                    # normalize the mask
                    mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-8)
                # apply the mask
                ref = ref * mask[..., None, None, :]

                # aggregate over time
                ref = ref.sum(dim=-1)
            else:
                # no mask: average over time
                ref = ref.mean(dim=-1)

            # Check if the UUT matches the reference
            assert torch.allclose(uut, ref, atol=atol), f'Example {n}: UUT not matching the reference.'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 3])
    @pytest.mark.parametrize('num_freq', [17, 33])
    def test_mismatch_dimensions(self, num_channels: int, num_freq: int):
        """Test that the covariance matrix is not calculated if the mask has a different number of dimensions than the input."""
        batch_size, num_steps = 8, 100

        # Typically-shaped inputs
        input = torch.randn(batch_size, num_channels, num_freq, num_steps, dtype=torch.cfloat)
        mask = torch.rand(batch_size, num_freq, num_steps, dtype=torch.float)

        # Input has only (freq, time) dimensions -- missing at least one channel dimension
        with pytest.raises(ValueError):
            covariance_matrix(x=input[0, 0, ...])

        # Mask has only (freq, time) dimensions -- missing batch dimension
        with pytest.raises(ValueError):
            covariance_matrix(x=input, mask=mask[0, ...])

        # Mask has wrong number of time steps
        with pytest.raises(ValueError):
            covariance_matrix(x=input, mask=mask[..., :-1])

        # Mask has wrong number of frequency bins
        with pytest.raises(ValueError):
            covariance_matrix(x=input, mask=mask[..., :-1, :])
