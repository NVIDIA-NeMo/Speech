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

import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.data.audio_to_eou_label_lhotse import LhotseSpeechToTextBpeEOUDataset

pytestmark = pytest.mark.unit

SAMPLE_RATE = 16000


def _dataset(pad_distribution: str, checkpointed_rng: bool = True) -> LhotseSpeechToTextBpeEOUDataset:
    cfg = OmegaConf.create(
        {
            "check_tokenizer": False,
            "sample_rate": SAMPLE_RATE,
            "random_padding": {
                "prob": 0.9,
                "max_total_duration": 6.0,
                "min_post_pad_duration": 0.5,
                "pad_distribution": pad_distribution,
            },
        }
    )
    dataset = LhotseSpeechToTextBpeEOUDataset(cfg=cfg, tokenizer=None)
    if checkpointed_rng:
        dataset.enable_checkpointed_rng()
    return dataset


def _pad(dataset: LhotseSpeechToTextBpeEOUDataset):
    audio = torch.ones(SAMPLE_RATE)
    eou_targets = torch.zeros(13, dtype=torch.long)
    eou_targets[-1] = 1
    padded_audio, padded_audio_len, padded_eou_targets = dataset._random_pad_audio(
        audio, torch.tensor(SAMPLE_RATE), eou_targets
    )
    return padded_audio, int(padded_audio_len), padded_eou_targets


@pytest.mark.parametrize("pad_distribution", ["uniform", "normal"])
def test_restored_rng_state_reproduces_random_padding(pad_distribution):
    dataset = _dataset(pad_distribution)
    dataset.rng.seed(0)  # a fixed sequence that pads to several different lengths
    state = dataset.state_dict()
    expected = [_pad(dataset) for _ in range(8)]

    # A resumed process does not share the global RNG state of the original one.
    np.random.seed(1234)
    restored = _dataset(pad_distribution)
    restored.load_state_dict(state)

    for padded_audio, padded_audio_len, padded_eou_targets in expected:
        actual_audio, actual_len, actual_eou_targets = _pad(restored)
        assert actual_len == padded_audio_len
        torch.testing.assert_close(actual_audio, padded_audio)
        torch.testing.assert_close(actual_eou_targets, padded_eou_targets)
    # The padding is drawn at random, so it differs between reads of the same audio.
    assert len({padded_audio_len for _, padded_audio_len, _ in expected}) > 1


@pytest.mark.parametrize("pad_distribution", ["uniform", "normal"])
def test_random_padding_uses_numpy_rng_unless_checkpointed_rng_is_enabled(pad_distribution):
    # Legacy behaviour: without use_stateful_dataloader the padding follows the global ``np.random``.
    dataset = _dataset(pad_distribution, checkpointed_rng=False)
    np.random.seed(0)
    first = [_pad(dataset)[1] for _ in range(8)]
    np.random.seed(0)
    second = [_pad(dataset)[1] for _ in range(8)]

    assert first == second
    assert len(set(first)) > 1


def _dataset_with_augmentor(checkpointed_rng: bool = True) -> LhotseSpeechToTextBpeEOUDataset:
    cfg = OmegaConf.create(
        {
            "check_tokenizer": False,
            "sample_rate": SAMPLE_RATE,
            "augmentor": {
                "gain": {"prob": 0.5, "min_gain_dbfs": -10, "max_gain_dbfs": 10},
                "white_noise": {"prob": 1.0, "min_level": -60, "max_level": -40},
            },
        }
    )
    dataset = LhotseSpeechToTextBpeEOUDataset(cfg=cfg, tokenizer=None)
    if checkpointed_rng:
        dataset.enable_checkpointed_rng()
    return dataset


def _augment(dataset: LhotseSpeechToTextBpeEOUDataset):
    audio = torch.sin(torch.arange(SAMPLE_RATE, dtype=torch.float32) / 10.0)
    augmented, augmented_len = dataset._maybe_augment_audio(audio, torch.tensor(SAMPLE_RATE))
    return augmented, int(augmented_len)


def test_restored_rng_state_reproduces_augmentor_perturbations():
    dataset = _dataset_with_augmentor()
    dataset.rng.seed(0)  # a fixed sequence in which the gain perturbation is applied to some reads only
    state = dataset.state_dict()
    expected = [_augment(dataset) for _ in range(4)]

    # A resumed process does not share the global RNG states of the original one.
    random.seed(1234)
    np.random.seed(1234)
    restored = _dataset_with_augmentor()
    restored.load_state_dict(state)

    for expected_audio, expected_len in expected:
        actual_audio, actual_len = _augment(restored)
        assert actual_len == expected_len
        torch.testing.assert_close(actual_audio, expected_audio)
    # The perturbations are random, so the output differs between reads of the same audio.
    assert len({audio.numpy().tobytes() for audio, _ in expected}) == len(expected)


def test_augmentor_uses_global_rngs_unless_checkpointed_rng_is_enabled():
    # Legacy behaviour: without use_stateful_dataloader the perturbations follow the global generators.
    dataset = _dataset_with_augmentor(checkpointed_rng=False)
    random.seed(0)
    np.random.seed(0)
    first = _augment(dataset)[0]
    random.seed(0)
    np.random.seed(0)
    second = _augment(dataset)[0]

    assert dataset.augmentor.rng is random
    torch.testing.assert_close(first, second)
