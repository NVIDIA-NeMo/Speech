# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from types import SimpleNamespace

import pytest

from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset

pytestmark = pytest.mark.unit


def test_challenging_text_probability_linear_schedule():
    dataset = MagpieTTSLhotseMultiturnDataset.__new__(MagpieTTSLhotseMultiturnDataset)
    dataset.challenging_text_start_prob = 0.1
    dataset.challenging_text_end_prob = 0.5
    dataset.challenging_text_start_step = 100
    dataset.challenging_text_end_step = 500
    dataset._training_step = SimpleNamespace(value=0)

    assert dataset.get_challenging_text_replacement_prob(99) == 0.0
    assert dataset.get_challenging_text_replacement_prob(100) == pytest.approx(0.1)
    assert dataset.get_challenging_text_replacement_prob(300) == pytest.approx(0.3)
    assert dataset.get_challenging_text_replacement_prob(500) == pytest.approx(0.5)
    assert dataset.get_challenging_text_replacement_prob(700) == pytest.approx(0.5)

    dataset.set_training_step(250)
    assert dataset.get_challenging_text_replacement_prob() == pytest.approx(0.25)
