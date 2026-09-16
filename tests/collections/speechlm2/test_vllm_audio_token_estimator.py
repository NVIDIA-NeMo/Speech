# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Drift check for the vLLM plugin's audio-token estimator.

The plugin hand-rolls
``NeMoSpeechLMProcessingInfo._estimate_audio_tokens`` in pure Python for
speed (~90x faster than the equivalent tensor-ops path). Accuracy
matters: the result drives how many ``<|audio|>`` placeholders the
prompt processor inserts AND the encoder's real output frame count at
forward time; a mismatch breaks placeholder-to-feature alignment.

This test locks the hand-rolled math to NeMo's ``calc_length`` on a
canonical set of audio lengths. If FastConformer's subsampling ever
changes upstream (different kernel/stride/repeat), this test fails and
forces an update to the estimator.
"""

from types import SimpleNamespace
import pytest
import torch

pytest.importorskip("vllm")

from nemo.collections.asr.parts.submodules.subsampling import calc_length
from nemo.collections.common.data.lhotse.audio_token_estimator import AudioTokenEstimator
from nemo.collections.speechlm2.vllm.salm.audio import (
    _DUMMY_AUDIO_MAX_DURATION_S,
    _MIN_CHUNK_SIZE_SAMPLES,
    _SAMPLING_RATE,
    NeMoSpeechLMProcessingInfo,
)


def _reference(audio_length_samples: int) -> int:
    """Reference impl using NeMo's calc_length for the conv chain."""
    n_fft = 512
    hop_length = 160
    stft_pad = n_fft // 2
    fbank_len = (audio_length_samples + 2 * stft_pad - n_fft) // hop_length
    out = calc_length(
        torch.tensor([fbank_len], dtype=torch.float),
        all_paddings=2,
        kernel_size=3,
        stride=2,
        ceil_mode=False,
        repeat_num=3,
    )
    return max(1, int(out.item()))


@pytest.mark.parametrize(
    "samples",
    [
        1_600,  # 0.1 s
        16_000,  # 1 s
        80_000,  # 5 s
        160_000,  # 10 s
        320_000,  # 20 s
        640_000,  # 40 s, the typical max
        12_345,  # arbitrary small
        54_321,  # arbitrary mid
        100_001,  # arbitrary (odd)
    ],
)
def test_estimator_matches_calc_length(samples: int) -> None:
    ours = NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples)
    ref = _reference(samples)
    assert ours == ref, (
        f"audio_token estimator diverged from NeMo calc_length for "
        f"samples={samples}: ours={ours}, ref={ref}. "
        f"Check if FastConformer's subsampling stack changed upstream."
    )


def test_estimator_min_one() -> None:
    """Even for very short audio the estimator must return at least 1."""
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(1) >= 1


def test_estimator_chunking_disabled_matches_single_pass() -> None:
    """``chunk_size_seconds=None`` must match the legacy single-pass estimate."""
    samples = 30 * 16_000
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(
        samples, chunk_size_seconds=None
    ) == NeMoSpeechLMProcessingInfo._estimate_audio_tokens_single_pass(samples)


def test_pe_encoder_ignores_generic_chunk_size_for_placeholder_estimation() -> None:
    config = SimpleNamespace(
        pe_encoder_path="/checkpoint/ParallelExpertEncoder.nemo",
        encoder_chunk_size_seconds=15.0,
    )
    info = SimpleNamespace(get_hf_config=lambda: config)

    assert (NeMoSpeechLMProcessingInfo._get_encoder_chunk_size_seconds(info)) is None

    # Regression length from corrected ASR-LB row 0: generic 15-second
    # chunking overcounts the full-audio PEE output by one placeholder.
    samples = 559_280
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples, None) == 437
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples, 15.0) == 438


def test_embedded_pe_encoder_ignores_generic_chunk_size_for_placeholder_estimation() -> None:
    config = SimpleNamespace(
        pe_encoder_path=None,
        pe_encoder_config={"target": "ParallelExpertEncoderPT"},
        encoder_chunk_size_seconds=15.0,
    )
    info = SimpleNamespace(get_hf_config=lambda: config)

    assert NeMoSpeechLMProcessingInfo._get_encoder_chunk_size_seconds(info) is None


def test_standard_encoder_keeps_generic_chunk_size() -> None:
    config = SimpleNamespace(pe_encoder_path=None, encoder_chunk_size_seconds=15.0)
    info = SimpleNamespace(get_hf_config=lambda: config)

    assert NeMoSpeechLMProcessingInfo._get_encoder_chunk_size_seconds(info) == 15.0


def test_estimator_short_audio_falls_back_to_single_pass() -> None:
    """Audio shorter than the chunk size collapses to a single forward."""
    samples = 5 * 16_000
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(
        samples, chunk_size_seconds=30.0
    ) == NeMoSpeechLMProcessingInfo._estimate_audio_tokens_single_pass(samples)


def test_estimator_chunked_sums_per_chunk_frames() -> None:
    """Long audio is split into chunks and per-chunk frame counts are summed,
    matching ``encode_audio_with_optional_chunking``'s concat behavior."""
    samples = 90 * 16_000
    chunk_size_seconds = 30.0
    chunk_samples = int(round(chunk_size_seconds * 16_000))
    expected = sum(
        NeMoSpeechLMProcessingInfo._estimate_audio_tokens_single_pass(min(chunk_samples, samples - i))
        for i in range(0, samples, chunk_samples)
    )
    assert (
        NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples, chunk_size_seconds=chunk_size_seconds) == expected
    )


def test_estimator_chunked_tail_folded_into_previous_chunk() -> None:
    """A tiny tail (< min chunk size) is folded into the previous chunk so
    the total token count matches the runtime helper instead of producing a
    spurious single-frame chunk that the audio preprocessor would reject."""
    chunk_size_seconds = 30.0
    chunk_samples = int(round(chunk_size_seconds * 16_000))
    samples = chunk_samples + 100  # 100 sample tail < min_chunk_size_samples (320)
    # Folded: one chunk of `samples` samples (no split).
    expected = NeMoSpeechLMProcessingInfo._estimate_audio_tokens_single_pass(samples)
    assert (
        NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples, chunk_size_seconds=chunk_size_seconds) == expected
    )


def test_estimator_clamps_tiny_chunk_size_to_min_samples() -> None:
    assert _MIN_CHUNK_SIZE_SAMPLES == 320

    chunk_size_seconds = 1 / _SAMPLING_RATE
    samples = 2 * _MIN_CHUNK_SIZE_SAMPLES + 100
    expected = NeMoSpeechLMProcessingInfo._estimate_audio_tokens_single_pass(
        _MIN_CHUNK_SIZE_SAMPLES
    ) + NeMoSpeechLMProcessingInfo._estimate_audio_tokens_single_pass(_MIN_CHUNK_SIZE_SAMPLES + 100)

    assert (
        NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples, chunk_size_seconds=chunk_size_seconds) == expected
    )


def test_estimator_negative_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="encoder_chunk_size_seconds"):
        NeMoSpeechLMProcessingInfo._estimate_audio_tokens(16_000, chunk_size_seconds=-1.0)


@pytest.mark.parametrize("chunk_size_seconds", [None, 30.0])
def test_samples_for_audio_tokens_returns_minimum_sample_count(chunk_size_seconds: float | None) -> None:
    target_tokens = 17

    samples = NeMoSpeechLMProcessingInfo._samples_for_audio_tokens(target_tokens, chunk_size_seconds)

    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples, chunk_size_seconds) >= target_tokens
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples - 1, chunk_size_seconds) < target_tokens


def test_samples_for_audio_tokens_rejects_unreachable_target() -> None:
    max_samples = int(_DUMMY_AUDIO_MAX_DURATION_S * _SAMPLING_RATE)
    max_tokens = NeMoSpeechLMProcessingInfo._estimate_audio_tokens(max_samples)

    with pytest.raises(ValueError, match="Cannot produce"):
        NeMoSpeechLMProcessingInfo._samples_for_audio_tokens(max_tokens + 1)


_FEATURE_STACKING_CONFIG = {
    "preprocessor": {"n_fft": 512, "hop_length": 160, "stft_pad_amount": 256},
    "subsampling": {"type": "feature_stacking", "factor": 8},
    "chunk_size_seconds": None,
}


@pytest.mark.parametrize("samples", [1, 1_600, 16_000, 123_457, 480_000])
def test_exported_feature_stacking_estimator_matches_training(samples: int) -> None:
    reference = AudioTokenEstimator.from_config(_FEATURE_STACKING_CONFIG, sample_rate=_SAMPLING_RATE)
    assert reference is not None
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(
        samples, estimator_config=_FEATURE_STACKING_CONFIG
    ) == reference.estimate_samples(samples)


def test_exported_feature_stacking_estimator_matches_training_with_chunking() -> None:
    config = {**_FEATURE_STACKING_CONFIG, "chunk_size_seconds": 30.0}
    reference = AudioTokenEstimator.from_config(config, sample_rate=_SAMPLING_RATE)
    assert reference is not None
    samples = 31 * _SAMPLING_RATE
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(
        samples, estimator_config=config
    ) == reference.estimate_samples(samples)


def test_feature_stacking_inverse_returns_minimum_sample_count() -> None:
    config = {**_FEATURE_STACKING_CONFIG, "chunk_size_seconds": 30.0}
    target_tokens = 388
    samples = NeMoSpeechLMProcessingInfo._samples_for_audio_tokens(target_tokens, estimator_config=config)
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples, estimator_config=config) >= target_tokens
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples - 1, estimator_config=config) < target_tokens
