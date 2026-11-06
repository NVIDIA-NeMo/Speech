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

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.data.audio_to_eou_label_lhotse import LhotseSpeechToTextBpeEOUDataset

pytestmark = pytest.mark.unit

SAMPLE_RATE = 16000


def _dataset(pad_distribution: str) -> LhotseSpeechToTextBpeEOUDataset:
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
    return LhotseSpeechToTextBpeEOUDataset(cfg=cfg, tokenizer=None)


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
