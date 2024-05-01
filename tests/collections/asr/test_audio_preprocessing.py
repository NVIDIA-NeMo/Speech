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

import importlib

import numpy as np
import pytest
import torch

from nemo.collections.asr.modules.audio_preprocessing import AudioToSpectrogram, SpectrogramToAudio

try:
    importlib.import_module('torchaudio')

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


class TestAudioSpectrogram:
    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize('fft_length', [64, 512])
    @pytest.mark.parametrize('num_channels', [1, 3])
    def test_audio_to_spec(self, fft_length: int, num_channels: int):
        """Test output length for audio to spectrogram.

        Create signals of arbitrary length and check output
        length is matching the actual transform length.
        """
        hop_lengths = [fft_length // 2, fft_length // 3, fft_length // 4]
        batch_size = 4
        num_examples = 20
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        for n in range(num_examples):

            # Generate time-domain examples with different length
            input_length = _rng.integers(low=fft_length, high=100 * fft_length, size=batch_size)  # in samples
            x = _rng.normal(size=(batch_size, num_channels, np.max(input_length)))
            x = torch.tensor(x)
            for b in range(batch_size):
                x[b, :, input_length[b] :] = 0

            for hop_length in hop_lengths:
                # Prepare transform
                audio2spec = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length)

                # Transform the whole batch
                batch_spec, batch_spec_len = audio2spec(input=x, input_length=torch.tensor(input_length))

                for b in range(batch_size):

                    # Transform just the current example
                    b_spec, b_spec_len = audio2spec(input=x[b : b + 1, :, : input_length[b]])
                    actual_len = b_spec.size(-1)

                    # Check lengths
                    assert (
                        actual_len == b_spec_len
                    ), f'Output length not matching for example ({n}, {b}) with length {input_length[n]} (hop_length={hop_length}): true {actual_len} vs calculated {b_spec_len}.'

                    assert (
                        actual_len == batch_spec_len[b]
                    ), f'Output length not matching for example ({n}, {b}) with length {input_length[n]} (hop_length={hop_length}): true {actual_len} vs calculated batch len {batch_spec_len[b]}.'

                    # Make sure transforming a batch is the same as transforming individual examples
                    assert torch.allclose(
                        batch_spec[b, ..., :actual_len], b_spec, atol=atol
                    ), f'Spectrograms not matching for example ({n}, {b}) with length {input_length[b]} (hop_length={hop_length})'

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize('fft_length', [64, 512])
    @pytest.mark.parametrize('num_channels', [1, 3])
    def test_spec_to_audio(self, fft_length: int, num_channels: int):
        """Test output length for spectrogram to audio.

        Create signals of arbitrary length and check output
        length is matching the actual transform length.
        """
        hop_lengths = [fft_length // 2, fft_length // 3, fft_length // 4]
        batch_size = 4
        num_examples = 20
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        for n in range(num_examples):

            # Generate spectrogram examples with different lengths
            input_length = _rng.integers(low=10, high=100, size=batch_size)  # in frames
            input_shape = (batch_size, num_channels, fft_length // 2 + 1, np.max(input_length))
            spec = _rng.normal(size=input_shape) + 1j * _rng.normal(size=input_shape)
            spec = torch.tensor(spec)
            spec[..., 0, :] = spec[..., 0, :].real
            spec[..., -1, :] = spec[..., -1, :].real
            for b in range(batch_size):
                spec[b, ..., input_length[b] :] = 0

            for hop_length in hop_lengths:
                # Prepare transform
                spec2audio = SpectrogramToAudio(fft_length=fft_length, hop_length=hop_length)

                # Transform the whole batch
                batch_x, batch_x_len = spec2audio(input=spec, input_length=torch.tensor(input_length))

                for b in range(batch_size):

                    # Transform just the current example
                    b_x, b_x_len = spec2audio(input=spec[b : b + 1, ..., : input_length[b]])

                    actual_len = b_x.size(-1)

                    # Check lengths
                    assert (
                        b_x_len == actual_len
                    ), f'Output length not matching for example ({n}, {b}) with {input_length[b]} frames (hop_length={hop_length}): true {actual_len} vs calculated {b_x_len}.'

                    assert (
                        batch_x_len[b] == actual_len
                    ), f'Output length not matching for example ({n}, {b}) with {input_length[b]} frames (hop_length={hop_length}): true {actual_len} vs calculated batch {batch_x_len[b]}.'

                    # Make sure transforming a batch is the same as transforming individual examples
                    if input_length[b] < spec.size(-1):
                        # Discard the last bit of the signal which differs due to number of frames in batch (with zero padded frames) vs individual (only valid frames).
                        # The reason for this difference is normalization with `window_sumsquare` of the inverse STFT. More specifically,
                        # batched and non-batched transform are using on a different number of frames.
                        tail_length = max(fft_length // 2 - hop_length, 0)
                    else:
                        tail_length = 0
                    valid_len = actual_len - tail_length
                    batch_x_valid = batch_x[b, :, :valid_len]
                    b_x_valid = b_x[..., :valid_len]
                    assert torch.allclose(
                        batch_x_valid, b_x_valid, atol=atol
                    ), f'Signals not matching for example ({n}, {b}) with length {input_length[b]} (hop_length={hop_length}): max abs diff {torch.max(torch.abs(batch_x_valid-b_x_valid))} at {torch.argmax(torch.abs(batch_x_valid-b_x_valid))}'

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize('fft_length', [128, 1024])
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('magnitude_power', [0.5, 1, 2])
    @pytest.mark.parametrize('scale', [0.1, 1.0])
    def test_audio_to_spectrogram_reconstruction(
        self, fft_length: int, num_channels: int, magnitude_power: float, scale: float
    ):
        """Test analysis and synthesis transform result in a perfect reconstruction.
        """
        batch_size = 4
        num_samples = fft_length * 50
        num_examples = 25
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        hop_lengths = [fft_length // 2, fft_length // 4]

        for hop_length in hop_lengths:
            audio2spec = AudioToSpectrogram(
                fft_length=fft_length, hop_length=hop_length, magnitude_power=magnitude_power, scale=scale
            )
            spec2audio = SpectrogramToAudio(
                fft_length=fft_length, hop_length=hop_length, magnitude_power=magnitude_power, scale=scale
            )

            for n in range(num_examples):
                x = _rng.normal(size=(batch_size, num_channels, num_samples))

                x_spec, x_spec_length = audio2spec(input=torch.Tensor(x))
                x_hat, x_hat_length = spec2audio(input=x_spec, input_length=x_spec_length)

                assert np.allclose(
                    x_hat.cpu().detach().numpy(), x, atol=atol
                ), f'Reconstructed not matching for example {n} (hop length {hop_length})'
