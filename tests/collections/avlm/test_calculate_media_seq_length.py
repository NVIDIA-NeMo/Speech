# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from nemo.collections.avlm.data.energon import calculate_media_seq_length


def test_calculate_encoded_audio_seq_length_whisper():
    # Typical whisper config: 16kHz, 30s audio, 0.02s stride, downsampling 2
    model_type = "whisper"
    audio_length = 16000 * 30  # 30 seconds
    fixed_max_audio_length = 16000 * 30
    sample_rate = 16000
    window_stride = 0.02
    encoder_down_sampling = 2
    num_mel_bins = 80
    patch_size = 16
    time_stride = 10
    frequency_stride = 10
    max_spectrogram_length = 3000

    seq_len = calculate_media_seq_length.calculate_encoded_audio_seq_length(
        model_type=model_type,
        audio_length=audio_length,
        fixed_max_audio_length=fixed_max_audio_length,
        sample_rate=sample_rate,
        window_stride=window_stride,
        encoder_down_sampling=encoder_down_sampling,
        num_mel_bins=num_mel_bins,
        patch_size=patch_size,
        time_stride=time_stride,
        frequency_stride=frequency_stride,
        max_spectrogram_length=max_spectrogram_length,
    )
    assert seq_len > 0


def test_calculate_encoded_audio_seq_length_wavlm():
    model_type = "wavlm"
    audio_length = 16000 * 10  # 10 seconds
    seq_len = calculate_media_seq_length.calculate_encoded_audio_seq_length(
        model_type=model_type,
        audio_length=audio_length,
        fixed_max_audio_length=None,
        sample_rate=None,
        window_stride=None,
        encoder_down_sampling=None,
        num_mel_bins=None,
        patch_size=None,
        time_stride=None,
        frequency_stride=None,
        max_spectrogram_length=None,
    )
    assert seq_len > 0


def test_calculate_encoded_audio_seq_length_ast():
    model_type = "ast"
    audio_length = 1000
    seq_len = calculate_media_seq_length.calculate_encoded_audio_seq_length(
        model_type=model_type,
        audio_length=audio_length,
        fixed_max_audio_length=None,
        sample_rate=None,
        window_stride=None,
        encoder_down_sampling=None,
        num_mel_bins=128,
        patch_size=16,
        time_stride=10,
        frequency_stride=10,
        max_spectrogram_length=1000,
    )
    assert seq_len > 0


def test_calculate_encoded_audio_seq_length_invalid():
    with pytest.raises(ValueError):
        calculate_media_seq_length.calculate_encoded_audio_seq_length(
            model_type="unknown",
            audio_length=1000,
            fixed_max_audio_length=None,
            sample_rate=None,
            window_stride=None,
            encoder_down_sampling=None,
            num_mel_bins=None,
            patch_size=None,
            time_stride=None,
            frequency_stride=None,
            max_spectrogram_length=None,
        )


def test_calculate_encoded_image_seq_length_vit():
    seq_len = calculate_media_seq_length.calculate_encoded_image_seq_length(
        num_one_image_tiles=1,
        model_type="vit",
        img_width=224,
        img_height=224,
        patch_size=16,
        projection_downsample_factor=2,
    )
    assert seq_len > 0


def test_calculate_encoded_image_seq_length_invalid():
    with pytest.raises(ValueError):
        calculate_media_seq_length.calculate_encoded_image_seq_length(
            num_one_image_tiles=1,
            model_type="unknown",
            img_width=224,
            img_height=224,
            patch_size=16,
            projection_downsample_factor=None,
        )
