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

import pytest
from omegaconf import OmegaConf

from nemo.collections.asr.inference.factory.cache_aware_pipeline_builder import CacheAwarePipelineBuilder


class _FakeASRModel:
    def __init__(self):
        self.cuda_graph_settings = []

    def set_streaming_cuda_graphs(self, enabled: bool) -> None:
        self.cuda_graph_settings.append(enabled)


class _FakePipeline:
    def __init__(self):
        self.asr_model = _FakeASRModel()


@pytest.mark.unit
@pytest.mark.parametrize(("configured_value", "expected"), [(True, True), (False, False), (None, False)])
def test_cache_aware_pipeline_configures_encoder_cuda_graphs(monkeypatch, configured_value, expected):
    """The official cache-aware pipeline must pass its config flag to the encoder wrapper."""
    cfg_data = {"asr_decoding_type": "rnnt", "asr": {}}
    if configured_value is not None:
        cfg_data["asr"]["use_cuda_graphs"] = configured_value
    cfg = OmegaConf.create(cfg_data)
    pipeline = _FakePipeline()
    monkeypatch.setattr(
        CacheAwarePipelineBuilder,
        "build_cache_aware_rnnt_pipeline",
        classmethod(lambda cls, cfg: pipeline),
    )

    result = CacheAwarePipelineBuilder.build(cfg)

    assert result is pipeline
    assert pipeline.asr_model.cuda_graph_settings == [expected]
