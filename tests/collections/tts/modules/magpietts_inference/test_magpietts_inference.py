# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import os

import pytest

from examples.tts.magpietts_inference import main as magpietts_inference_main


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
