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

"""
Tests for the per-file metrics that MagpieTTS evaluation saves (evaluate_generated_audio.py).
"""

import math

import pytest

from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import evaluate

EVALUATE_MODULE = "nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio"


@pytest.mark.unit
def test_evaluate_keeps_pred_gt_ssim_in_the_filewise_metrics(monkeypatch):
    """The comparison report reads the per-file ground-truth SSIM, so it must survive the saved-metrics filter."""
    nan = float("nan")
    # One row with ground-truth audio and one text-context row without it (the evaluator writes NaN per row).
    rows = [
        _filewise_row(pred_gt_ssim=0.81, pred_context_ssim=0.42),
        _filewise_row(pred_gt_ssim=nan, pred_context_ssim=nan, pred_audio_filepath="pred_1.wav"),
    ]
    monkeypatch.setattr(f"{EVALUATE_MODULE}.evaluate_dir", lambda **kwargs: rows)
    monkeypatch.setattr(f"{EVALUATE_MODULE}.compute_global_metrics", lambda **kwargs: {})

    _, filewise = evaluate(
        manifest_path="manifest.json",
        audio_dir=None,
        generated_audio_dir="generated",
        with_fcd=False,
        with_utmosv2=False,
    )

    assert [row["pred_gt_ssim"] for row in filewise[:1]] == [0.81]
    assert math.isnan(filewise[1]["pred_gt_ssim"])
    assert [row["pred_context_ssim"] for row in filewise[:1]] == [0.42]
    # The filter is still applied: non-saved keys do not leak into the file.
    assert "detailed_cer" not in filewise[0]


def _filewise_row(**overrides):
    """Return a per-file metric row with saved and non-saved keys, as evaluate_dir() produces it."""
    row = {
        "gt_text": "you want to ski",
        "pred_text": "you want to ski",
        "detailed_cer": (0.0, 15, 0.0, 0.0, 0.0),
        "cer": 0.0,
        "wer": 0.0,
        "pred_gt_ssim": 0.5,
        "pred_context_ssim": 0.5,
        "gt_context_ssim": 0.5,
        "gt_audio_filepath": "gt_0.wav",
        "pred_audio_filepath": "pred_0.wav",
        "context_audio_filepath": "context_0.wav",
        "utmosv2": 3.0,
        "total_gen_audio_seconds": 1.0,
        "predicted_codes_path": None,
    }
    row.update(overrides)
    return row
