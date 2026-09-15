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
"""Tests for the standalone EasyMagpie LM pipeline topology."""
from __future__ import annotations

import pytest

pytest.importorskip("vllm_omni")

from easymagpie_vllm_omni.pipeline import EASYMAGPIE_LM_PIPELINE, EASYMAGPIE_PIPELINE  # noqa: E402


def test_lm_pipeline_is_single_stage():
    assert EASYMAGPIE_LM_PIPELINE.model_type == "easymagpie_lm"
    assert len(EASYMAGPIE_LM_PIPELINE.stages) == 1
    stage = EASYMAGPIE_LM_PIPELINE.stages[0]
    assert stage.stage_id == 0
    assert stage.model_stage == "easymagpie"
    assert stage.final_output is True
    assert stage.final_output_type == "audio"
    assert stage.engine_output_type == "audio"
    assert stage.custom_process_next_stage_input_func is None
    assert stage.async_chunk_process_next_stage_input_func is None


def test_two_stage_pipeline_unchanged():
    assert EASYMAGPIE_PIPELINE.model_type == "easymagpie"
    assert len(EASYMAGPIE_PIPELINE.stages) == 2
    assert EASYMAGPIE_PIPELINE.stages[1].final_output is True
    assert EASYMAGPIE_PIPELINE.stages[1].final_output_type == "audio"
