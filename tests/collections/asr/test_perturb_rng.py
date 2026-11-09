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

from nemo.collections.asr.parts.preprocessing.perturb import (
    GainPerturbation,
    Perturbation,
    ShiftPerturbation,
    SilencePerturbation,
    WhiteNoisePerturbation,
    process_augmentations,
)
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

pytestmark = pytest.mark.unit

SAMPLE_RATE = 16000
AUGMENTOR_CFG = {
    "gain": {"prob": 0.5, "min_gain_dbfs": -10, "max_gain_dbfs": 10},
    "shift": {"prob": 0.5, "min_shift_ms": -50.0, "max_shift_ms": 50.0},
    "silence": {"prob": 0.5, "min_start_silence_secs": 0.0, "max_start_silence_secs": 0.2},
    "white_noise": {"prob": 1.0, "min_level": -60, "max_level": -40},
}


def _segment():
    samples = np.sin(np.arange(SAMPLE_RATE, dtype=np.float32) / 10.0).astype(np.float32)
    return AudioSegment(samples=samples, sample_rate=SAMPLE_RATE)


def _augment(augmentor, times: int = 6):
    outputs = []
    for _ in range(times):
        segment = _segment()
        augmentor.perturb(segment)
        outputs.append(segment.samples.copy())
    return outputs


def _assert_same(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


def test_injected_rng_drives_all_augmentation_draws():
    augmentor = process_augmentations(AUGMENTOR_CFG)
    augmentor.rng = random.Random(0)
    random.seed(1)
    np.random.seed(1)
    expected = _augment(augmentor)

    other = process_augmentations(AUGMENTOR_CFG)
    other.rng = random.Random(0)
    random.seed(2)  # the global generators play no part once a generator is injected
    np.random.seed(2)

    _assert_same(_augment(other), expected)
    # The draws vary between calls (different lengths from silence/shift, different noise).
    assert len({out.tobytes() for out in expected}) == len(expected)


def test_restoring_injected_rng_state_reproduces_augmentation():
    augmentor = process_augmentations(AUGMENTOR_CFG)
    rng = random.Random(0)
    augmentor.rng = rng
    _augment(augmentor, times=2)
    state = rng.getstate()
    expected = _augment(augmentor)

    restored = random.Random()
    restored.setstate(state)
    augmentor.rng = restored

    _assert_same(_augment(augmentor), expected)


def test_injected_rng_reaches_every_perturbation():
    augmentor = process_augmentations(AUGMENTOR_CFG)
    rng = random.Random(0)
    augmentor.rng = rng

    assert augmentor.rng is rng
    assert all(p.rng is rng for _, p in augmentor._pipeline)


def test_injected_rng_reaches_nested_perturbations():
    class Composite(Perturbation):
        def __init__(self):
            self._by_sample_rate = {SAMPLE_RATE: GainPerturbation()}
            self._chain = [ShiftPerturbation(), SilencePerturbation()]
            self._single = WhiteNoisePerturbation()

        def perturb(self, data):
            pass

    composite = Composite()
    rng = random.Random(0)
    composite.rng = rng

    nested = [composite._by_sample_rate[SAMPLE_RATE], *composite._chain, composite._single]
    assert all(p.rng is rng for p in nested)


def test_default_rng_is_the_global_random_module():
    # Backward compatibility: without an injected generator, perturbations use the global RNG as before.
    assert GainPerturbation().rng is random
    assert process_augmentations(AUGMENTOR_CFG).rng is random
