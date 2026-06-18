#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import math
from dataclasses import dataclass
from numbers import Number
from typing import Literal

from lhotse import compute_num_samples
from omegaconf import OmegaConf

from nemo.core.neural_types import LabelsType, NeuralType


def is_2d_bucketing(buckets) -> bool:
    return all(
        isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, Number) for v in item)
        for item in buckets
    )


@dataclass
class SequenceLengthResolver:
    cfg: object
    ratio: float
    salm_audio_token_ratio: float
    module_name: str | None = None
    model: object | None = None
    schema: dict | None = None

    def resolve_many(self, buckets) -> list[tuple[int, int]]:
        return [self.resolve_one(bucket) for bucket in buckets]

    def resolve_one(self, bucket) -> tuple[int, int]:
        if self._uses_audio_locator_expansion():
            return self._audio_locator_lens(bucket)

        if is_2d_bucketing([bucket]):
            input_len, output_len = bucket
            return int(input_len), int(output_len)

        input_len = bucket
        output_len = int(math.ceil(self.ratio * input_len))
        if self.schema is None:
            return compute_num_samples(input_len, sampling_rate=16000), output_len

        sampling_rate = self._sampling_rate()
        match self._modalities():
            case ("audio", "audio"):
                return (
                    compute_num_samples(input_len, sampling_rate=sampling_rate),
                    compute_num_samples(output_len, sampling_rate=sampling_rate),
                )
            case ("audio", "text"):
                return compute_num_samples(input_len, sampling_rate=sampling_rate), output_len
            case ("text", "audio"):
                return int(input_len), compute_num_samples(output_len, sampling_rate=sampling_rate)
            case ("text", "text"):
                return int(input_len), output_len
            case unexpected:
                raise RuntimeError(f"Unexpected modality combination: {unexpected}")

    def _matches_model_name(self, *suffixes: str) -> bool:
        return self.module_name is not None and any(self.module_name.endswith(suffix) for suffix in suffixes)

    def _matches_model_class_name(self, *names: str) -> bool:
        return self.model is not None and type(self.model).__name__ in names

    def _uses_audio_locator_expansion(self) -> bool:
        return self._matches_model_name(
            "SALMAutomodel", "SALM", "SALMWithAsrDecoder"
        ) or self._matches_model_class_name("SALMAutomodel", "SALM", "SALMWithAsrDecoder")

    def _modalities(self) -> tuple[str, str]:
        if self.schema is None:
            return "audio", "text"

        def _modality(direction: Literal["input", "output"]) -> str:
            for item in self.schema["inputs"]:
                nt = item["type"]
                if nt == "dummy":
                    continue
                if (
                    isinstance(nt, NeuralType)
                    and isinstance(nt.elements_type, LabelsType)
                    and item["seq_length"] == direction
                ):
                    return "text"
            return "audio"

        return _modality("input"), _modality("output")

    def _sampling_rate(self) -> int:
        return int(getattr(self.model, "sample_rate", 16000))

    def _audio_locator_lens(self, bucket) -> tuple[int, int]:
        sampling_rate = OmegaConf.select(self.cfg, "data.train_ds.sample_rate", default=16000)
        token_equivalent_duration = OmegaConf.select(self.cfg, "data.train_ds.token_equivalent_duration", default=0.08)
        audio_tokens = max(1, int(math.ceil(self.salm_audio_token_ratio * bucket)))
        text_tokens = max(2, int(math.ceil((1.0 - self.salm_audio_token_ratio) * bucket)))
        audio_len = int(math.ceil(audio_tokens * token_equivalent_duration * sampling_rate))
        return audio_len, text_tokens
