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

import json
import random
import tarfile

import numpy as np
import pytest
import soundfile as sf

from nemo.collections.asr.parts.preprocessing import perturb
from nemo.collections.asr.parts.preprocessing.perturb import (
    ImpulsePerturbation,
    NoisePerturbation,
    NoisePerturbationWithNormalization,
)
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index

pytestmark = pytest.mark.unit

SAMPLE_RATE = 16000
NUM_FILES = 8


@pytest.fixture(params=[True, False], ids=["with_idx", "without_idx"])
def tarred_audio(tmp_path, request):
    """Eight distinct 1.2 s audio files in two NeMo-style tars, with or without .idx files.

    NoisePerturbationWithNormalization requires noise of at least 1 s.
    """
    manifest_path = tmp_path / "manifest.json"
    tar_paths = [tmp_path / f"audio_{i}.tar" for i in range(2)]
    tars = [tarfile.open(path, "w") for path in tar_paths]
    with manifest_path.open("w") as manifest:
        for i in range(NUM_FILES):
            t = np.arange(int(1.2 * SAMPLE_RATE)) / SAMPLE_RATE
            samples = 0.1 * np.sin(2 * np.pi * (200 + 150 * i) * t) * np.exp(-t * (i + 1))
            wav_path = tmp_path / f"audio_{i}.wav"
            sf.write(wav_path, samples.astype(np.float32), SAMPLE_RATE)
            tars[i % 2].add(wav_path, arcname=wav_path.name)
            manifest.write(json.dumps({"audio_filepath": wav_path.name, "duration": 1.2, "text": ""}) + "\n")
    for tar in tars:
        tar.close()
    if request.param:
        for path in tar_paths:
            create_tar_index(str(path), str(path) + ".idx")
    return str(manifest_path), [str(path) for path in tar_paths]


def _make(kind, manifest_path, tar_paths):
    if kind == "noise":
        return NoisePerturbation(manifest_path, min_snr_db=10, max_snr_db=10, audio_tar_filepaths=tar_paths)
    if kind == "impulse":
        return ImpulsePerturbation(manifest_path, audio_tar_filepaths=tar_paths, shuffle_n=4)
    return NoisePerturbationWithNormalization(
        manifest_path, min_snr_db=10, max_snr_db=10, audio_tar_filepaths=[tar_paths], shuffle_n=4
    )


def _apply(perturbation, times):
    outputs = []
    for _ in range(times):
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        segment = AudioSegment(samples=(0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sample_rate=SAMPLE_RATE)
        perturbation.perturb(segment)
        outputs.append(segment.samples.copy())
    return outputs


@pytest.mark.parametrize("kind", ["noise", "impulse", "noise_with_normalization"])
def test_tarred_audio_choice_is_restored_from_injected_rng(tarred_audio, kind):
    manifest_path, tar_paths = tarred_audio
    perturbation = _make(kind, manifest_path, tar_paths)
    rng = random.Random(0)
    perturbation.rng = rng
    _apply(perturbation, 2)
    state = rng.getstate()
    expected = _apply(perturbation, 4)

    # A resumed run builds a fresh perturbation (with a fresh tar iterator) and restores the generator state.
    resumed = _make(kind, manifest_path, tar_paths)
    restored_rng = random.Random()
    restored_rng.setstate(state)
    resumed.rng = restored_rng
    actual = _apply(resumed, 4)

    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    # Different noise/RIR files are picked over the calls.
    assert len({out.tobytes() for out in expected}) > 1


@pytest.mark.parametrize("kind", ["noise", "impulse", "noise_with_normalization"])
def test_tarred_audio_without_injected_rng_keeps_sequential_reading(tarred_audio, kind, monkeypatch):
    manifest_path, tar_paths = tarred_audio
    perturbation = _make(kind, manifest_path, tar_paths)

    def fail(*args, **kwargs):
        raise AssertionError("random access must only be used with an injected generator")

    monkeypatch.setattr(perturb.TarredAudioRandomAccess, "load", fail)

    assert len(_apply(perturbation, 3)) == 3
