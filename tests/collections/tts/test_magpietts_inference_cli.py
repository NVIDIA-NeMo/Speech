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
Tests for MagpieTTS inference CLI options.
"""

import csv
import os
from glob import glob

import pytest

from examples.tts.magpietts_inference import main as magpietts_inference_main


class TestMagpieTTSInferenceCLI:
    """Tests for MagpieTTS inference command-line interface options."""

    # Test data paths - these should match CI environment
    CODEC_MODEL_PATH = "/home/TestData/tts/AudioCodec_21Hz_no_eliz_without_wavlm_disc.nemo"
    HPARAMS_FILE = "/home/TestData/tts/2506_ZeroShot/lrhm_short_yt_prioralways_alignement_0.002_priorscale_0.1.yaml"
    CHECKPOINT_FILE = "/home/TestData/tts/2506_ZeroShot/dpo-T5TTS--val_loss=0.4513-epoch=3.ckpt"
    EVALSET_CONFIG = "examples/tts/evalset_config.json"

    @pytest.mark.run_only_on('GPU')
    def test_disable_fcd_produces_nan_metric(self, tmp_path):
        """
        Test that the --disable_fcd option:
        1. Does not cause the script to crash
        2. Produces NaN for the frechet_codec_distance metric
        """
        # Build command-line arguments
        args = [
            "--codecmodel_path",
            self.CODEC_MODEL_PATH,
            "--datasets_json_path",
            self.EVALSET_CONFIG,
            "--datasets",
            "an4_val_tiny_ci",
            "--out_dir",
            str(tmp_path),
            "--batch_size",
            "4",
            "--num_repeats",
            "2",  # multiple repeats tests that NaNs don't crash the confidence interval calculation
            "--temperature",
            "0.6",
            "--hparams_files",
            self.HPARAMS_FILE,
            "--checkpoint_files",
            self.CHECKPOINT_FILE,
            "--legacy_codebooks",
            "--legacy_text_conditioning",
            "--apply_attention_prior",
            "--run_evaluation",
            "--disable_fcd",
        ]

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

        fcd_value = metrics.get("frechet_codec_distance")
        assert fcd_value is not None, "frechet_codec_distance key not found in metrics"
        assert "nan" in fcd_value.lower(), f"frechet_codec_distance should be NaN but got: {fcd_value}"

    @pytest.mark.run_only_on('GPU')
    def test_disable_utmosv2_produces_nan_metric(self, tmp_path):
        """
        Test that the --disable_utmosv2 option:
        1. Does not cause the script to crash
        2. Produces NaN for the utmosv2 metric
        """
        # Build command-line arguments
        args = [
            "--codecmodel_path",
            self.CODEC_MODEL_PATH,
            "--datasets_json_path",
            self.EVALSET_CONFIG,
            "--datasets",
            "an4_val_tiny_ci",
            "--out_dir",
            str(tmp_path),
            "--batch_size",
            "4",
            "--num_repeats",
            "1",  # single repeat to keep test short
            "--temperature",
            "0.6",
            "--hparams_files",
            self.HPARAMS_FILE,
            "--checkpoint_files",
            self.CHECKPOINT_FILE,
            "--legacy_codebooks",
            "--legacy_text_conditioning",
            "--apply_attention_prior",
            "--run_evaluation",
            "--disable_utmosv2",
        ]

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

        utmosv2_value = metrics.get("utmosv2_avg")
        assert utmosv2_value is not None, "utmosv2 key not found in metrics"
        assert "nan" in utmosv2_value.lower(), f"utmosv2 should be NaN but got: {utmosv2_value}"
