# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import math

import numpy as np
import pytest
import torch
from lhotse.testing.dummies import dummy_recording

from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking, calc_length
from nemo.collections.common.data.lhotse.audio_token_estimator import AudioTokenEstimator
from nemo.collections.common.data.lhotse.sampling import MultimodalSamplingConstraint, TokenCountFilter
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, NeMoMultimodalConversation, TextTurn


def _config(chunk_size_seconds=15.0):
    return {
        "preprocessor": {
            "n_fft": 512,
            "hop_length": 160,
            "stft_pad_amount": 256,
        },
        "subsampling": {
            "type": "conv",
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "repeat": 3,
            "ceil_mode": False,
        },
        "chunk_size_seconds": chunk_size_seconds,
    }


def _feature_stacking_config(chunk_size_seconds=15.0):
    config = _config(chunk_size_seconds=chunk_size_seconds)
    config["subsampling"] = {"type": "feature_stacking", "factor": 8}
    return config


def _reference_single_pass(num_samples: int) -> int:
    feature_frames = (num_samples + 2 * 256 - 512) // 160
    output = calc_length(
        torch.tensor([feature_frames]),
        all_paddings=2,
        kernel_size=3,
        stride=2,
        ceil_mode=False,
        repeat_num=3,
    )
    return max(1, int(output.item()))


@pytest.mark.parametrize("num_samples", [1_600, 12_345, 16_000, 54_321, 100_001, 240_000])
def test_audio_token_estimator_matches_fastconformer_integer_lengths(num_samples):
    estimator = AudioTokenEstimator.from_config(_config(chunk_size_seconds=None), sample_rate=16000)
    assert estimator.estimate_samples(num_samples) == _reference_single_pass(num_samples)


@pytest.mark.parametrize("num_samples", [1_600, 12_345, 16_000, 54_321, 100_001, 239_999, 240_000])
def test_audio_token_estimator_matches_pee_feature_stacking_lengths(num_samples):
    estimator = AudioTokenEstimator.from_config(_feature_stacking_config(chunk_size_seconds=None), sample_rate=16000)
    feature_frames = (num_samples + 2 * 256 - 512) // 160
    subsampling = FeatureStacking(subsampling_factor=8, feat_in=4, feat_out=4)
    expected = max(1, int(subsampling.compute_num_out_frames(feature_frames)))
    assert estimator.estimate_samples(num_samples) == expected


@pytest.mark.parametrize(
    "num_samples",
    [1, 159, 160, 1_119, 1_120, 1_279, 1_280, 1_281, 239_999, 240_000, 240_001],
)
def test_current_canary_v2_and_pee_length_estimators_are_numerically_equivalent(
    num_samples,
):
    canary_v2 = AudioTokenEstimator.from_config(_config(chunk_size_seconds=None), sample_rate=16000)
    pee = AudioTokenEstimator.from_config(_feature_stacking_config(chunk_size_seconds=None), sample_rate=16000)
    assert canary_v2.estimate_samples(num_samples) == pee.estimate_samples(num_samples)


def test_audio_token_estimator_keeps_legacy_convolution_config_compatible():
    config = _config(chunk_size_seconds=None)
    config["subsampling"].pop("type")
    estimator = AudioTokenEstimator.from_config(config, sample_rate=16000)
    assert estimator.estimate_samples(16_000) == _reference_single_pass(16_000)


@pytest.mark.parametrize(
    ("subsampling", "message"),
    [
        ({"type": "feature_stacking"}, "missing: \\['factor'\\]"),
        ({"type": "feature_stacking", "factor": 0}, "Invalid"),
        ({"type": "unknown"}, "must be 'conv' or 'feature_stacking'"),
    ],
)
def test_audio_token_estimator_validates_subsampling_type(subsampling, message):
    config = _config(chunk_size_seconds=None)
    config["subsampling"] = subsampling
    with pytest.raises(ValueError, match=message):
        AudioTokenEstimator.from_config(config, sample_rate=16000)


def test_audio_token_estimator_accounts_for_per_chunk_rounding():
    estimator = AudioTokenEstimator.from_config(_config(chunk_size_seconds=15.0), sample_rate=16000)
    samples = 30 * 16000

    # Each 15-second chunk produces 188 frames. The historical duration-only
    # estimate rounds once for the whole 30 seconds and therefore reports 375.
    assert estimator.estimate_samples(samples) == 376
    assert math.ceil((samples / 16000) / 0.08) == 375


def test_audio_token_estimator_folds_tiny_tail_like_runtime_chunking():
    estimator = AudioTokenEstimator.from_config(_config(chunk_size_seconds=15.0), sample_rate=16000)
    chunk_samples = 15 * 16000
    samples = chunk_samples + 100
    assert estimator.estimate_samples(samples) == _reference_single_pass(samples)


def test_audio_token_estimator_clamps_chunk_size_to_preprocessor_minimum():
    estimator = AudioTokenEstimator.from_config(_config(chunk_size_seconds=1 / 16000), sample_rate=16000)
    # The preprocessor needs two feature frames, so runtime clamps one-sample
    # chunks to 320 samples before splitting. The 100-sample tail is folded.
    assert estimator.estimate_samples(740) == _reference_single_pass(320) + _reference_single_pass(420)


def test_exact_multimodal_constraint_matches_real_placeholder_replacement_length():
    estimator = AudioTokenEstimator.from_config(_config(), sample_rate=16000)
    cut = dummy_recording(0, duration=30.0).to_cut()
    conversation = NeMoMultimodalConversation(
        id="chunked-audio",
        turns=[AudioTurn(cut, "user", "<|audio|>"), TextTurn("ok", "assistant")],
        token_equivalent_duration=0.08,
    )
    conversation.context_ids = np.arange(7)
    conversation.answer_ids = np.arange(3)
    conversation.input_ids = np.arange(10)

    legacy = MultimodalSamplingConstraint(measure_total_length=True)
    exact = MultimodalSamplingConstraint(
        measure_total_length=True,
        batch_tokens=16384,
        use_packed_sequence_sampling=True,
        audio_token_estimator=estimator,
    )

    assert legacy.measure_length(conversation) == 384  # 10 - locator + 375 approximate frames
    assert exact.measure_length(conversation) == 385  # 10 - locator + 376 real frames
    assert exact.measure_length(cut) == 376


def test_exact_packed_audio_sampling_requires_estimator_metadata():
    cut = dummy_recording(0, duration=1.0).to_cut()
    conversation = NeMoMultimodalConversation(
        id="audio",
        turns=[AudioTurn(cut, "user", "<|audio|>"), TextTurn("ok", "assistant")],
        token_equivalent_duration=0.08,
    )
    conversation.context_ids = np.arange(2)
    conversation.answer_ids = np.arange(1)
    conversation.input_ids = np.arange(3)
    constraint = MultimodalSamplingConstraint(
        batch_tokens=16,
        use_packed_sequence_sampling=True,
        measure_total_length=True,
    )

    with pytest.raises(ValueError, match="requires audio_token_estimator"):
        constraint.measure_length(conversation)
    with pytest.raises(ValueError, match="requires audio_token_estimator"):
        constraint.measure_length(cut)


def test_exact_estimator_is_used_by_max_tokens_filter():
    estimator = AudioTokenEstimator.from_config(_config(), sample_rate=16000)
    cut = dummy_recording(0, duration=30.0).to_cut()
    conversation = NeMoMultimodalConversation(
        id="filter",
        turns=[AudioTurn(cut, "user", "<|audio|>"), TextTurn("ok", "assistant")],
        token_equivalent_duration=0.08,
    )
    conversation.context_ids = np.arange(7)
    conversation.answer_ids = np.arange(3)
    conversation.input_ids = np.arange(10)

    legacy = TokenCountFilter(None, 384, measure_total_length=True)
    exact = TokenCountFilter(None, 384, measure_total_length=True, audio_token_estimator=estimator)
    assert legacy(conversation)
    assert not exact(conversation)


def test_exact_estimator_rejects_unresampled_cut():
    estimator = AudioTokenEstimator.from_config(_config(), sample_rate=16000)
    cut = dummy_recording(0, duration=1.0, sampling_rate=8000).to_cut()
    with pytest.raises(ValueError, match="requires audio to be resampled first"):
        estimator.estimate_cut(cut)
