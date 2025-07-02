# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import librosa
import numpy as np
import pytest
import torch

from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures


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
