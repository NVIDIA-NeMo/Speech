# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.tts.parts.utils import reward_audio
from nemo.collections.tts.parts.utils.reward_audio import resample_audio, split_audio_batch

pytestmark = pytest.mark.unit


def test_split_audio_trims_and_detaches():
    audio = torch.arange(24.0).reshape(3, 8).requires_grad_()
    result = split_audio_batch(audio, torch.tensor([3, 8, 5]))
    assert [wav.numel() for wav in result] == [3, 8, 5]
    assert all(not wav.requires_grad for wav in result)
    assert all(wav.device == audio.device for wav in result)
    torch.testing.assert_close(result[0], audio[0, :3])


@pytest.mark.parametrize(
    'lengths', [torch.tensor([0, 2]), torch.tensor([9, 2]), torch.tensor([3.5, 2.0]), torch.tensor([2])]
)
def test_split_audio_rejects_invalid_lengths(lengths):
    with pytest.raises(ValueError):
        split_audio_batch(torch.zeros(2, 8), lengths)


@pytest.mark.parametrize('source_rate', [16000, 22050, 24000, 44100, 48000])
def test_resample_to_16khz_preserves_duration_and_frequency(source_rate):
    time = torch.arange(source_rate, dtype=torch.float32) / source_rate
    tone = torch.sin(2 * torch.pi * 440 * time)
    result = resample_audio([tone], source_rate, 16000)[0]
    assert result.numel() == 16000
    peak = torch.fft.rfft(result).abs().argmax().item()
    assert peak == 440


def test_resample_same_rate_does_not_copy():
    audio = torch.randn(100)
    result = resample_audio([audio], 16000, 16000)[0]
    assert result.data_ptr() == audio.data_ptr()


@pytest.mark.parametrize('rate', [0, -1])
def test_resample_rejects_invalid_rate(rate):
    with pytest.raises(ValueError):
        resample_audio([torch.ones(8)], rate, 16000)


@pytest.mark.parametrize('source_rate,target_rate', [(16000, 24000), (24000, 16000), (22050, 16000), (16000, 16000)])
def test_waveform_adapter_uses_existing_repo_resampler(source_rate, target_rate):
    assert reward_audio.resample is resample
    audio = [torch.arange(size, dtype=torch.float32).requires_grad_() for size in (7, 29, 101)]
    actual = resample_audio(audio, source_rate, target_rate)
    for waveform, result in zip(audio, actual):
        expected = resample(waveform.detach(), source_rate, target_rate)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert not result.requires_grad
        assert result.device == waveform.device
