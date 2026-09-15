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
import soundfile as sf

from nemo.collections.tts.metrics.prosody import compute_prosody_distances

_SAMPLE_RATE = 16000
_TEXT = "hello world"


def _write_sine(path, duration_sec: float, frequency_hz: float = 220.0, amplitude: float = 0.2) -> None:
    sample_count = int(round(_SAMPLE_RATE * duration_sec))
    time = np.arange(sample_count, dtype=np.float32) / _SAMPLE_RATE
    audio = amplitude * np.sin(2.0 * np.pi * frequency_hz * time)
    sf.write(path, audio.astype(np.float32), _SAMPLE_RATE)


@pytest.mark.unit
def test_prosody_distances_are_zero_for_identical_audio(tmp_path):
    gt_path = tmp_path / "gt.wav"
    pred_path = tmp_path / "pred.wav"
    _write_sine(gt_path, duration_sec=1.0)
    _write_sine(pred_path, duration_sec=1.0)

    metrics = compute_prosody_distances(
        gt_audio_path=str(gt_path),
        pred_audio_path=str(pred_path),
        text=_TEXT,
    )

    assert metrics.pitch_distance == pytest.approx(0.0, abs=1.0e-6)
    assert metrics.intensity_distance == pytest.approx(0.0, abs=1.0e-6)
    assert metrics.speech_rate_distance == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.unit
def test_speech_rate_distance_tracks_duration_difference(tmp_path):
    gt_path = tmp_path / "gt.wav"
    pred_path = tmp_path / "pred.wav"
    _write_sine(gt_path, duration_sec=1.0)
    _write_sine(pred_path, duration_sec=2.0)

    metrics = compute_prosody_distances(
        gt_audio_path=str(gt_path),
        pred_audio_path=str(pred_path),
        text=_TEXT,
    )

    assert np.isfinite(metrics.pitch_distance)
    assert np.isfinite(metrics.intensity_distance)
    assert metrics.speech_rate_distance == pytest.approx(5.0, abs=1.0e-6)
