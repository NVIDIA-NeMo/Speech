# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import functools
import math

import librosa
import numpy as np
import pytest
import torch

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessorConfig
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
from nemo.collections.asr.parts.submodules.batchnorm import MaskedBatchNorm1d


class TestFilterbankFeatures:
    @pytest.mark.unit
    def test_seq_len(self):
        fb_module = FilterbankFeatures(exact_pad=False, pad_to=1)
        test_1 = torch.randn(1, 800)
        test_1_len = torch.tensor([800])
        fb_spec, fb_len = fb_module(test_1, test_1_len)
        assert fb_spec.shape[2] - 1 == fb_len[0], f"{fb_spec.shape} != {fb_len}"
        librosa_spec = librosa.stft(test_1.cpu().detach().numpy().squeeze(), n_fft=512, hop_length=160, win_length=320)

        assert librosa_spec.shape[1] == fb_spec.shape[2], f"{librosa_spec.shape} != {fb_spec.shape}"

    @pytest.mark.unit
    def test_random_stft_sizes(self):
        for _ in range(5):
            nfft = 2 ** np.random.randint(7, 12)
            window_size = np.random.randint(100, nfft)
            hop_size = np.random.randint(64, window_size)
            fb_module = FilterbankFeatures(
                exact_pad=False,
                pad_to=1,
                n_fft=nfft,
                n_window_size=window_size,
                n_window_stride=hop_size,
                normalize=False,
            )
            audio_length = np.random.randint(nfft, 2**16)
            test_1 = torch.randn(1, audio_length)
            test_1_len = torch.tensor([audio_length])
            fb_spec, fb_len = fb_module(test_1, test_1_len)
            assert (
                fb_spec.shape[2] - 1 == fb_len[0]
            ), f"{fb_spec.shape} != {fb_len}: {nfft}, {window_size}, {hop_size}, {audio_length}"

            librosa_spec = librosa.stft(
                test_1.cpu().detach().numpy().squeeze(), n_fft=nfft, hop_length=hop_size, win_length=window_size
            )

            assert (
                librosa_spec.shape[1] == fb_spec.shape[2]
            ), f"{librosa_spec.shape} != {fb_spec.shape}: {nfft}, {window_size}, {hop_size}, {audio_length}"

    @pytest.mark.unit
    def test_random_stft_sizes_exact_pad(self):
        for _ in range(5):
            nfft = 2 ** np.random.randint(7, 12)
            window_size = np.random.randint(100, nfft)
            hop_size = np.random.randint(64, window_size)
            if hop_size % 2 == 1:
                hop_size = hop_size - 1
            fb_module = FilterbankFeatures(
                exact_pad=True,
                pad_to=1,
                n_fft=nfft,
                n_window_size=window_size,
                n_window_stride=hop_size,
                normalize=False,
            )
            audio_length = np.random.randint(nfft, 2**16)
            test_1 = torch.randn(1, audio_length)
            test_1_len = torch.tensor([audio_length])
            fb_spec, fb_len = fb_module(test_1, test_1_len)
            assert (
                fb_spec.shape[2] - 1 == fb_len[0]
            ), f"{fb_spec.shape} != {fb_len}: {nfft}, {window_size}, {hop_size}, {audio_length}"

            test_2 = test_1.cpu().detach().numpy().squeeze()
            test_2 = np.pad(test_2, int((nfft - hop_size) // 2), mode="reflect")
            librosa_spec = librosa.stft(
                test_2,
                n_fft=nfft,
                hop_length=hop_size,
                win_length=window_size,
                center=False,
            )

            assert (
                fb_spec.shape[2] == librosa_spec.shape[1]
            ), f"{fb_spec.shape} != {librosa_spec.shape}: {nfft}, {window_size}, {hop_size}, {audio_length}"

            assert (
                fb_spec.shape[2] == audio_length // hop_size
            ), f"{fb_spec.shape}, {nfft}, {window_size}, {hop_size}, {audio_length}, {audio_length // hop_size}"


class TestFilterbankFeaturesStft:
    @pytest.mark.unit
    def test_real_valued_spectrum(self):
        torch.manual_seed(0)
        fb_module = FilterbankFeatures(n_fft=512, n_window_size=400, n_window_stride=160)
        x = torch.randn(2, 4000)
        spectrum = fb_module.stft(x)
        reference = torch.view_as_real(
            torch.stft(
                x,
                n_fft=512,
                hop_length=160,
                win_length=400,
                center=True,
                window=fb_module.window,
                return_complex=True,
                pad_mode="constant",
            )
        )
        assert spectrum.shape == reference.shape == (2, 257, 26, 2)
        torch.testing.assert_close(spectrum, reference)


# A 25 ms window and 10 ms hop in a 512-point FFT, without dither, pre-emphasis, padding or normalization.
PLAIN = dict(n_window_size=400, n_window_stride=160, n_fft=512, nfilt=80, dither=0.0, preemph=None, pad_to=0)


def _htk_filterbank(sample_rate, n_fft, n_mels, f_min, f_max):
    """Unnormalized triangular filters with centers spaced evenly on log(1 + f / 700)."""

    def mel(frequency):
        return np.log1p(frequency / 700.0)

    centers = 700.0 * np.expm1(np.linspace(mel(f_min), mel(f_max), n_mels + 2))
    frequencies = np.arange(n_fft // 2 + 1) * sample_rate / n_fft
    weights = np.zeros((n_mels, len(frequencies)))
    for index in range(n_mels):
        lower, center, upper = centers[index : index + 3]
        rising = (frequencies - lower) / (center - lower)
        falling = (upper - frequencies) / (upper - center)
        weights[index] = np.clip(np.minimum(rising, falling), 0.0, None)
    return weights


def _pack(signal, lengths):
    data = torch.cat([signal[index, :length] for index, length in enumerate(lengths.tolist())])
    cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    return data, lengths, cu_seqlens


class TestFilterbankFeaturesOptions:
    """Mel scale, window, framing, log base and batch-norm normalization options."""

    @pytest.mark.unit
    def test_defaults_unchanged(self):
        module = FilterbankFeatures()
        expected = librosa.filters.mel(sr=16000, n_fft=512, n_mels=64, fmin=0, fmax=8000.0, norm="slaney")
        assert torch.equal(module.fb[0], torch.tensor(expected, dtype=torch.float))
        assert torch.equal(module.window, torch.hann_window(320, periodic=False))
        assert module.norm is None and module.log_base is None and not module.snip_edges
        assert set(module.state_dict()) == {"window", "fb"}
        features, lengths = module(torch.randn(1, 16000), torch.tensor([16000]))
        # centered framing: T // hop reported, one frame more produced, padded to a multiple of 16
        assert lengths.tolist() == [100] and features.shape == (1, 64, 112)

    @pytest.mark.unit
    def test_htk_mel_scale(self):
        module = FilterbankFeatures(mel_scale="htk", mel_norm=None, **PLAIN)
        expected = _htk_filterbank(16000, 512, 80, 0.0, 8000.0)
        torch.testing.assert_close(module.fb[0].double(), torch.from_numpy(expected), atol=1e-6, rtol=0)
        librosa_htk = librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000.0, htk=True, norm=None)
        assert torch.equal(module.fb[0], torch.tensor(librosa_htk, dtype=torch.float))
        with pytest.raises(ValueError, match="mel_scale"):
            FilterbankFeatures(mel_scale="mel")

    @pytest.mark.unit
    def test_periodic_window(self):
        module = FilterbankFeatures(periodic_window=True, **PLAIN)
        assert torch.equal(module.window, torch.hann_window(400, periodic=True))
        assert module.window[0] == 0.0 and module.window[200] == 1.0

    @pytest.mark.unit
    def test_snip_edges_framing(self):
        torch.manual_seed(0)
        module = FilterbankFeatures(snip_edges=True, periodic_window=True, normalize=None, **PLAIN).eval()
        signal = torch.randn(3, 3000)
        lengths = torch.tensor([3000, 2300, 399])
        features, out_lengths = module(signal, lengths)
        # (T - 400) // 160 + 1 frames, all valid, and none for a signal shorter than one window
        assert out_lengths.tolist() == [17, 12, 0]
        assert features.shape == (3, 80, 17)
        # frame t covers samples [160 t, 160 t + 400):
        # the TensorFlow framing, with the window and the signal zero-padded to n_fft on the right
        window = torch.nn.functional.pad(torch.hann_window(400, periodic=True), (0, 112))
        padded = torch.nn.functional.pad(signal[:1], (0, 112))
        spectrum = torch.stft(
            padded, n_fft=512, hop_length=160, win_length=512, window=window, center=False, return_complex=True
        )
        reference = torch.log(torch.matmul(module.fb, spectrum.abs().pow(2)) + 2**-24)
        torch.testing.assert_close(features[:1], reference, atol=1e-4, rtol=1e-4)
        alone, alone_lengths = module(signal[1:2, :2300], lengths[1:2])
        assert alone_lengths.tolist() == [12] and alone.shape == (1, 80, 12)
        torch.testing.assert_close(alone[0], features[1, :, :12], atol=1e-5, rtol=1e-5)
        assert features[2].abs().sum() == 0
        with pytest.raises(ValueError, match="exact_pad and snip_edges"):
            FilterbankFeatures(snip_edges=True, exact_pad=True)

    @pytest.mark.unit
    def test_snip_edges_batch_shorter_than_one_window(self):
        module = FilterbankFeatures(snip_edges=True, normalize=None, **PLAIN).eval()
        signal = torch.randn(2, 399)
        lengths = torch.tensor([399, 150])
        features, out_lengths = module(signal, lengths)
        # no frame fits into either signal; the one frame the STFT needs is masked
        assert out_lengths.tolist() == [0, 0]
        assert features.shape[0] == 2 and features.shape[1] == 80 and features.shape[2] >= 1
        assert features.abs().sum() == 0
        packed = module.forward_packed(*_pack(signal, lengths))
        assert packed.lengths.tolist() == [0, 0] and packed.data.shape == (0, 80)

    @pytest.mark.unit
    def test_snip_edges_packed_matches_dense(self):
        torch.manual_seed(1)
        module = FilterbankFeatures(snip_edges=True, normalize=None, **PLAIN).eval()
        lengths = torch.tensor([3000, 2300, 1000])
        signal = torch.randn(3, 3000) * (torch.arange(3000)[None, :] < lengths[:, None])
        dense, dense_lengths = module(signal, lengths)
        packed = module.forward_packed(*_pack(signal, lengths))
        assert packed.lengths.tolist() == dense_lengths.tolist()
        assert packed.padded_length == dense.shape[2]
        start = 0
        for index, frames in enumerate(dense_lengths.tolist()):
            torch.testing.assert_close(
                packed.data[start : start + frames], dense[index, :, :frames].T, atol=1e-4, rtol=1e-4
            )
            start += frames

    @pytest.mark.unit
    def test_log_base(self):
        torch.manual_seed(2)
        natural = FilterbankFeatures(normalize=None, **PLAIN).eval()
        decimal = FilterbankFeatures(log_base=10.0, normalize=None, **PLAIN).eval()
        signal = torch.randn(2, 4000)
        lengths = torch.tensor([4000, 3000])
        expected, _ = natural(signal, lengths)
        actual, _ = decimal(signal, lengths)
        torch.testing.assert_close(actual, expected / math.log(10), atol=1e-5, rtol=1e-5)
        with pytest.raises(ValueError, match="log_base"):
            FilterbankFeatures(log_base=1.0)

    @pytest.mark.unit
    def test_batch_norm_in_filterbank_features(self):
        torch.manual_seed(4)
        # the block names what deviates from the module's defaults; the feature count comes from the parent
        block = functools.partial(MaskedBatchNorm1d, eps=1e-3, momentum=0.1, affine=False, unbiased_running_var=False)
        module = FilterbankFeatures(normalize=block, **PLAIN)
        raw = FilterbankFeatures(normalize=None, **PLAIN).eval()
        assert set(module.state_dict()) == {
            "window",
            "fb",
            "norm.running_mean",
            "norm.running_var",
            "norm.num_batches_tracked",
        }
        assert isinstance(module.norm, MaskedBatchNorm1d) and module.norm.num_features == 80
        assert module.norm.eps == 1e-3 and module.norm.momentum == 0.1
        assert FilterbankFeatures(normalize=MaskedBatchNorm1d, use_grads=True, **PLAIN).norm.eps == 1e-5
        lengths = torch.tensor([4000, 2500])
        signal = torch.randn(2, 4000) * (torch.arange(4000)[None, :] < lengths[:, None])
        unnormalized, frame_lengths = raw(signal, lengths)
        valid = torch.cat([unnormalized[index, :, :frames] for index, frames in enumerate(frame_lengths.tolist())], 1)
        module.train()
        trained, _ = module(signal, lengths)
        torch.testing.assert_close(module.norm.running_mean, 0.1 * valid.mean(1))
        torch.testing.assert_close(module.norm.running_var, 0.9 + 0.1 * valid.var(1, unbiased=False))
        normalized = (valid - valid.mean(1, keepdim=True)) / torch.sqrt(
            valid.var(1, unbiased=False, keepdim=True) + 1e-3
        )
        torch.testing.assert_close(trained[0, :, : frame_lengths[0]], normalized[:, : frame_lengths[0]])
        module.eval()
        dense, dense_lengths = module(signal, lengths)
        packed = module.forward_packed(*_pack(signal, lengths))
        assert packed.lengths.tolist() == dense_lengths.tolist()
        torch.testing.assert_close(
            packed.data[: dense_lengths[0]], dense[0, :, : dense_lengths[0]].T, atol=1e-4, rtol=1e-4
        )
        assert dense[1, :, dense_lengths[1] :].abs().sum() == 0

    @pytest.mark.unit
    def test_trainable_block_needs_grads(self):
        # a block with trainable parameters makes no sense in a featurizer that runs without gradients
        with pytest.raises(ValueError, match="trainable parameters"):
            FilterbankFeatures(normalize=MaskedBatchNorm1d, **PLAIN)
        with pytest.raises(ValueError, match="trainable parameters"):
            AudioToMelSpectrogramPreprocessor(
                normalize={
                    "_target_": "nemo.collections.asr.parts.submodules.batchnorm.MaskedBatchNorm1d",
                    "_partial_": True,
                }
            )
        # with gradients, the loss reaches the block's parameters and the waveform, dense and packed
        torch.manual_seed(6)
        lengths = torch.tensor([4000, 2500])
        signal = torch.randn(2, 4000) * (torch.arange(4000)[None, :] < lengths[:, None])
        signal.requires_grad_()
        module = FilterbankFeatures(normalize=MaskedBatchNorm1d, use_grads=True, **PLAIN).train()
        features, frame_lengths = module(signal, lengths)
        assert features.requires_grad
        weights = torch.randn_like(features)
        (features * weights).sum().backward()
        for gradient in (module.norm.weight.grad, module.norm.bias.grad, signal.grad):
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        # the bias gradient is the sum of the weights on the valid frames
        valid = torch.arange(features.shape[2])[None, :] < frame_lengths[:, None]
        torch.testing.assert_close(module.norm.bias.grad, (weights * valid[:, None, :]).sum((0, 2)))
        module.zero_grad()
        signal.grad = None
        packed = module.forward_packed(*_pack(signal, lengths))
        (packed.data * torch.randn_like(packed.data)).sum().backward()
        for gradient in (module.norm.weight.grad, module.norm.bias.grad, signal.grad):
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0

    @pytest.mark.unit
    def test_normalize_values(self):
        torch.manual_seed(5)
        signal = torch.randn(1, 3000)
        lengths = torch.tensor([3000])
        raw, _ = FilterbankFeatures(normalize=None, **PLAIN).eval()(signal, lengths)
        for off in ("NA", "None", False):
            module = FilterbankFeatures(normalize=off, **PLAIN).eval()
            assert module.norm is None
            torch.testing.assert_close(module(signal, lengths)[0], raw)
        with pytest.raises(ValueError, match="normalize='per_features'"):
            FilterbankFeatures(normalize="per_features", **PLAIN)
        with pytest.raises(ValueError, match="fixed_std"):
            FilterbankFeatures(normalize={"fixed_mean": [0.0] * 80}, **PLAIN)

    @pytest.mark.unit
    def test_preprocessor_options_and_config(self):
        preprocessor = AudioToMelSpectrogramPreprocessor(
            window_size=0.025,
            n_fft=512,
            features=80,
            mel_scale="htk",
            mel_norm=None,
            periodic_window=True,
            snip_edges=True,
            preemph=None,
            dither=0.0,
            log_zero_guard_type="clamp",
            log_zero_guard_value=1e-10,
            log_base=10.0,
            normalize={
                "_target_": "nemo.collections.asr.parts.submodules.batchnorm.MaskedBatchNorm1d",
                "_partial_": True,
                "eps": 1e-3,
                "affine": False,
            },
            pad_to=0,
        ).eval()
        featurizer = preprocessor.featurizer
        assert featurizer.snip_edges and featurizer.log_base == 10.0
        assert isinstance(featurizer.norm, MaskedBatchNorm1d) and featurizer.norm.eps == 1e-3
        assert torch.equal(featurizer.window, torch.hann_window(400, periodic=True))
        assert set(preprocessor.state_dict()) == {
            "featurizer.window",
            "featurizer.fb",
            "featurizer.norm.running_mean",
            "featurizer.norm.running_var",
            "featurizer.norm.num_batches_tracked",
        }
        signal = torch.randn(2, 16000)
        lengths = torch.tensor([16000, 8000])
        with torch.no_grad():
            features, frame_lengths = preprocessor(input_signal=signal, length=lengths)
        assert frame_lengths.tolist() == [98, 48] and features.shape == (2, 80, 98)
        config = AudioToMelSpectrogramPreprocessorConfig()
        assert (config.mel_scale, config.periodic_window, config.snip_edges) == ("slaney", False, False)
        assert config.log_base is None and config.normalize == "per_feature"
