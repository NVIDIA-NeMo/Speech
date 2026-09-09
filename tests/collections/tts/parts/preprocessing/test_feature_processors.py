# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import contextlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from nemo.collections.tts.parts.preprocessing.feature_processors import (
    FeatureScaler,
    LogCompression,
    MeanVarianceNormalization,
    MeanVarianceSpeakerNormalization,
)


class TestTTSFeatureProcessors:
    @contextlib.contextmanager
    def _write_test_dict(self, test_dict, filename):
        temp_dir = tempfile.TemporaryDirectory()
        try:
            test_dir = Path(temp_dir.name)
            test_dict_filepath = test_dir / filename
            with open(test_dict_filepath, 'w', encoding="utf-8") as stats_f:
                json.dump(test_dict, stats_f, indent=4)

            yield test_dict_filepath
        finally:
            temp_dir.cleanup()

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_feature_scalar(self):
        field = "test_feat"
        input_tensor = torch.tensor([-2.5, 0.0, 1.0], dtype=torch.float32)
        expected_tensor = torch.tensor([0.0, 2.0, 2.8], dtype=torch.float32)
        processor = FeatureScaler(field, add_value=2.5, div_value=1.25)

        training_example = {field: input_tensor}
        processor.process(training_example)
        output_tensor = training_example[field]

        torch.testing.assert_close(output_tensor, expected_tensor)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_log_compression(self):
        field = "test_feat"

        input_tensor = torch.tensor([-0.5, 0.0, 2.0], dtype=torch.float32)
        expected_tensor = torch.tensor([np.log(0.5), 0.0, np.log(3.0)], dtype=torch.float32)
        processor = LogCompression(field)

        training_example = {field: input_tensor}
        processor.process(training_example)
        output_tensor = training_example[field]

        torch.testing.assert_close(output_tensor, expected_tensor)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_log_compression_clamp(self):
        field = "test_feat"

        input_tensor = torch.tensor([0.1, 1.0, 2.0], dtype=torch.float32)
        expected_tensor = torch.tensor([np.log(0.5), 0.0, np.log(2.0)], dtype=torch.float32)
        processor = LogCompression(field, log_zero_guard_type="clamp", log_zero_guard_value=0.5)

        training_example = {field: input_tensor}
        processor.process(training_example)
        output_tensor = training_example[field]

        torch.testing.assert_close(output_tensor, expected_tensor)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mean_variance_normalization(self):
        field = "test_feat"
        filename = "stats.json"
        stat_dict = {"default": {"test_feat_mean": 1.5, "test_feat_std": 0.5}}

        input_tensor = torch.tensor([0.0, 1.5, 2.0], dtype=torch.float32)
        expected_tensor = torch.tensor([-3.0, 0.0, 1.0], dtype=torch.float32)
        training_example = {field: input_tensor}

        with self._write_test_dict(stat_dict, filename=filename) as stat_dict_filepath:
            processor = MeanVarianceNormalization(field, stats_path=stat_dict_filepath, mask_field=None)
            processor.process(training_example)

        output_tensor = training_example[field]
        torch.testing.assert_close(output_tensor, expected_tensor)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mean_variance_normalization_masked(self):
        field = "test_feat"
        mask_field = "mask"
        filename = "stats.json"
        stat_dict = {"default": {"test_feat_mean": 1.0, "test_feat_std": 0.5}}

        input_tensor = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        input_mask = torch.tensor([True, False, False, True], dtype=torch.bool)
        expected_tensor = torch.tensor([2.0, 0.0, 0.0, 8.0], dtype=torch.float32)
        training_example = {field: input_tensor, mask_field: input_mask}

        with self._write_test_dict(stat_dict, filename=filename) as stat_dict_filepath:
            processor = MeanVarianceNormalization(field, stats_path=stat_dict_filepath, mask_field=mask_field)
            processor.process(training_example)

        output_tensor = training_example[field]
        torch.testing.assert_close(output_tensor, expected_tensor)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mean_variance_speaker_normalization(self):
        field = "pitch"
        filename = "stats.json"
        stat_dict = {
            "default": {"pitch_mean": 1.5, "pitch_std": 0.5},
            "speaker1": {"pitch_mean": 0.5, "pitch_std": 1.0},
            "speaker2": {"pitch_mean": 0.0, "pitch_std": 2.0},
        }

        input_tensor = torch.tensor([0.0, 1.0], dtype=torch.float32)

        training_example1 = {field: input_tensor, "speaker": "speaker1"}
        training_example2 = {field: input_tensor, "speaker": "speaker2"}
        training_example3 = {field: input_tensor, "speaker": "unknown"}
        expected_tensor1 = torch.tensor([-0.5, 0.5], dtype=torch.float32)
        expected_tensor2 = torch.tensor([0.0, 0.5], dtype=torch.float32)
        expected_tensor3 = torch.tensor([-3.0, -1.0], dtype=torch.float32)

        with self._write_test_dict(stat_dict, filename=filename) as stat_dict_filepath:
            processor = MeanVarianceSpeakerNormalization(
                field, stats_path=stat_dict_filepath, mask_field=None, fallback_to_default=True
            )
            processor.process(training_example1)
            processor.process(training_example2)
            processor.process(training_example3)

        output_tensor1 = training_example1[field]
        output_tensor2 = training_example2[field]
        output_tensor3 = training_example3[field]
        torch.testing.assert_close(output_tensor1, expected_tensor1)
        torch.testing.assert_close(output_tensor2, expected_tensor2)
        torch.testing.assert_close(output_tensor3, expected_tensor3)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mean_variance_speaker_normalization_masked(self):
        field = "test_feat"
        mask_field = "test_mask"
        filename = "stats.json"
        stat_dict = {"steve": {"test_feat_mean": -1.0, "test_feat_std": 2.0}}

        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        input_mask = torch.tensor([False, True, False, True], dtype=torch.bool)
        expected_tensor = torch.tensor([0.0, 1.5, 0.0, 2.5], dtype=torch.float32)

        training_example = {field: input_tensor, "speaker": "steve", mask_field: input_mask}

        with self._write_test_dict(stat_dict, filename=filename) as stat_dict_filepath:
            processor = MeanVarianceSpeakerNormalization(field, stats_path=stat_dict_filepath, mask_field=mask_field)
            processor.process(training_example)

        output_tensor = training_example[field]
        torch.testing.assert_close(output_tensor, expected_tensor)
