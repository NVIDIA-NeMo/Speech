# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch


class FeatureProcessor(ABC):
    @abstractmethod
    def process(self, training_example: dict) -> None:
        """
        Process the input training example dictionary, modifying necessary fields in place.

        Args:
            training_example: training example dictionary.
        """


class FeatureScaler(FeatureProcessor):
    def __init__(self, field: str, add_value: float = 0.0, div_value: float = 1.0):
        """
        Scales a field by constant factors. For example, for mean-variance normalization.

        Specifically: input[field] = (input[field] + add_value) / div_value

        Args:
            field: Field to scale
            add_value: Constant float value to add to feature.
            div_value: Constant float value to divide feature by.
        """
        self.field = field
        self.add_value = add_value
        self.div_value = div_value

    def process(self, training_example: dict) -> None:
        feature = training_example[self.field]
        feature = (feature + self.add_value) / self.div_value
        training_example[self.field] = feature


class LogCompression(FeatureProcessor):
    def __init__(self, field: str, log_zero_guard_type: str = "add", log_zero_guard_value: float = 1.0):
        """
        Apply log compression to a field.

        By default: input[field] = log(1.0 + input[field])
        For clamp mode: input[field] = log(max(log_zero_guard_value, input[field]))

        Args:
            field: Field to apply log compression to.
            log_zero_guard_type: Method to avoid logarithm approaching -inf, either "add" or "clamp".
            log_zero_guard_value: Value to add or clamp input with.
        """

        self.field = field

        if log_zero_guard_type == "add":
            self.guard_fn = self._add_guard
        elif log_zero_guard_type == "clamp":
            self.guard_fn = self._clamp_guard
        else:
            raise ValueError(f"Unsupported log zero guard type: '{log_zero_guard_type}'")

        self.guard_type = log_zero_guard_type
        self.guard_value = log_zero_guard_value

    def _add_guard(self, feature: torch.Tensor):
        return feature + self.guard_value

    def _clamp_guard(self, feature: torch.Tensor):
        return torch.clamp(feature, min=self.guard_value)

    def process(self, training_example: dict) -> None:
        feature = training_example[self.field]

        feature = self.guard_fn(feature)
        feature = torch.log(feature)

        training_example[self.field] = feature


class MeanVarianceNormalization(FeatureProcessor):
    def __init__(self, field: str, stats_path: Path, mask_field: Optional[str] = "voiced_mask"):
        """
        Apply mean and variance to the input field. Statistics are provided in JSON format, and can be
        computed using scripts.dataset_processing.tts.compute_feature_stats.py

        Specifically: input[field] = (input[field] + mean) / standard_deviation

        Stats file format example for field 'pitch':

        {
            "default": {
                "pitch_mean": 100.0,
                "pitch_std": 50.0,
            }
        }

        Args:
            field: Field to apply normalization to.
            stats_path: JSON file with feature mean and variance.
            mask_field: Optional, field in example dictionary with boolean array indicating which values to
                mask to 0. Defaults to 'voiced_mask', expected to be computed by pyin pitch estimator.
        """

        self.field = field
        self.mask_field = mask_field

        with open(stats_path, 'r', encoding="utf-8") as stats_f:
            stats_dict = json.load(stats_f)
            self.mean = stats_dict["default"][f"{self.field}_mean"]
            self.std = stats_dict["default"][f"{self.field}_std"]

    def process(self, training_example: dict) -> None:
        feature = training_example[self.field]

        feature = (feature - self.mean) / self.std
        if self.mask_field:
            voiced_mask = training_example[self.mask_field]
            feature[~voiced_mask] = 0.0

        training_example[self.field] = feature


class MeanVarianceSpeakerNormalization(FeatureProcessor):
    def __init__(
        self,
        field: str,
        stats_path: Path,
        speaker_field: str = "speaker",
        mask_field: Optional[str] = "voiced_mask",
        fallback_to_default: bool = False,
    ):
        """
        Apply speaker level mean and variance to the input field. Statistics are provided in JSON format, and can be
        computed using scripts.dataset_processing.tts.compute_feature_stats.py

        Specifically: input[field] = (input[field] + speaker_mean) / speaker_standard_deviation

        Stats file format example for field 'pitch':

        {
            "default": {
                "pitch_mean": 100.0,
                "pitch_std": 50.0,
            },
            "speaker1": {
                "pitch_mean": 110.0,
                "pitch_std": 45.0,
            },
            "speaker2": {
                "pitch_mean": 105.0,
                "pitch_std": 30.0,
            },
            ...
        }

        Args:
            field: Field to apply normalization to.
            stats_path: JSON file with feature mean and variance.
            speaker_field: field containing speaker ID string.
            mask_field: Optional, field in example dictionary with boolean array indicating which values to
                mask to 0. Defaults to 'voiced_mask', expected to be computed by pyin pitch estimator.
            fallback_to_default: Whether to use 'default' feature statistics when speaker is not found in
                the statistics dictionary.
        """

        self.field = field
        self.key_mean = f"{self.field}_mean"
        self.key_std = f"{self.field}_std"
        self.speaker_field = speaker_field
        self.mask_field = mask_field
        self.fallback_to_default = fallback_to_default

        with open(stats_path, 'r', encoding="utf-8") as stats_f:
            self.stats_dict = json.load(stats_f)

    def process(self, training_example: dict) -> None:
        feature = training_example[self.field]

        speaker = training_example[self.speaker_field]
        if speaker in self.stats_dict:
            stats = self.stats_dict[speaker]
        elif self.fallback_to_default:
            stats = self.stats_dict["default"]
        else:
            raise ValueError(f"Statistics not found for speaker: {speaker}")

        feature_mean = stats[self.key_mean]
        feature_std = stats[self.key_std]

        feature = (feature - feature_mean) / feature_std

        if self.mask_field:
            mask = training_example[self.mask_field]
            feature[~mask] = 0.0

        training_example[self.field] = feature
