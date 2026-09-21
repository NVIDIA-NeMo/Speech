# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Tests for MagpieTTS inference.
"""

import csv
import json
import os

import pytest

from examples.tts.magpietts_inference import main as magpietts_inference_main
from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import (
    FILEWISE_METRICS_TO_SAVE,
    _get_record_texts,
)
from nemo.collections.tts.modules.magpietts_inference.utils import (
    _group_multiturn_filewise_metrics_by_sample,
    _write_grouped_multiturn_filewise_metrics_csv,
)


class TestMagpieTTSInferenceCLI:
    """Tests for MagpieTTS inference command-line interface options."""

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.parametrize(
        "disable_flag,metric_key",
        [
            # Test both the --disable_fcd and --disable_utmosv2 flags
            ("--disable_fcd", "frechet_codec_distance"),
            ("--disable_utmosv2", "utmosv2_avg"),
        ],
        # Test names
        ids=["disable_fcd", "disable_utmosv2"],
    )
    def test_disable_metric_produces_nan(self, tmp_path, disable_flag, metric_key):
        """
        Test that disabling a metric via CLI flag:
        1. Does not cause the script to crash
        2. Produces NaN for the corresponding metric
        """

        # Test data paths in CI environment
        codec_model_path = "/home/TestData/tts/AudioCodec_21Hz_no_eliz_without_wavlm_disc.nemo"
        hparams_file = (
            "/home/TestData/tts/2506_ZeroShot/lrhm_short_yt_prioralways_alignement_0.002_priorscale_0.1.yaml"
        )
        checkpoint_file = "/home/TestData/tts/2506_ZeroShot/dpo-T5TTS--val_loss=0.4513-epoch=3.ckpt"
        datasets_json_path = "examples/tts/evalset_config.json"

        # Build command-line arguments
        args = [
            "--codecmodel_path", codec_model_path,
            "--datasets_json_path", datasets_json_path,
            "--datasets", "an4_val_tiny_ci",
            "--out_dir", str(tmp_path),
            "--batch_size", "4",
            "--num_repeats", "1",
            "--temperature", "0.6",
            "--hparams_files", hparams_file,
            "--checkpoint_files", checkpoint_file,
            "--legacy_codebooks",
            "--legacy_text_conditioning",
            "--apply_attention_prior",
            "--run_evaluation",
            disable_flag,
        ]  # fmt: skip

        # Run the main function directly with arguments
        magpietts_inference_main(args)

        # Look for the metrics file
        metrics_file = os.path.join(tmp_path, "all_experiment_metrics_with_ci.csv")
        assert os.path.exists(metrics_file), f"Metrics file not found at {metrics_file}"

        # Load and verify the metrics
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "No data rows found in metrics CSV"
        metrics = rows[0]  # Get the first data row

        metric_value = metrics.get(metric_key)
        assert metric_value is not None, f"{metric_key} key not found in metrics"
        assert "nan" in metric_value.lower(), f"{metric_key} should be NaN but got: {metric_value}"


@pytest.mark.unit
def test_evaluation_uses_normalized_text_for_metrics():
    record = {
        "text": "July 15th",
        "normalized_text": "july fifteenth",
        "original_text": "legacy original text",
    }

    tts_text_input, dataloader_normalized_text, metric_reference_text = _get_record_texts(record)

    assert tts_text_input == "July 15th"
    assert dataloader_normalized_text == "july fifteenth"
    assert metric_reference_text == "july fifteenth"
    assert "tts_text_input" in FILEWISE_METRICS_TO_SAVE
    assert "dataloader_normalized_text" in FILEWISE_METRICS_TO_SAVE


@pytest.mark.unit
def test_grouped_multiturn_exports_input_and_normalized_text(tmp_path):
    grouped_rows = _group_multiturn_filewise_metrics_by_sample(
        [
            {
                "source_sample_idx": 0,
                "turn_id": 0,
                "tts_text_input": "July 15th",
                "dataloader_normalized_text": "july fifteenth",
                "gt_text": "july fifteenth",
                "pred_text": "july fifteenth",
            }
        ]
    )

    assert grouped_rows[0]["tts_text_input"] == ["July 15th"]
    assert grouped_rows[0]["dataloader_normalized_text"] == ["july fifteenth"]

    csv_path = tmp_path / "metrics.csv"
    _write_grouped_multiturn_filewise_metrics_csv(str(csv_path), grouped_rows)
    with csv_path.open(encoding="utf-8") as csv_file:
        csv_row = next(csv.DictReader(csv_file))

    assert json.loads(csv_row["tts_text_input"]) == ["July 15th"]
    assert json.loads(csv_row["dataloader_normalized_text"]) == ["july fifteenth"]
