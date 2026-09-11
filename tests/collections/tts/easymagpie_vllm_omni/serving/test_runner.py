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
"""Tests for the vLLM-Omni 0.24 streaming runner compatibility layer."""
from __future__ import annotations

import json
import torch
import yaml

from conftest import EASYMAGPIE_ROOT
from easymagpie_vllm_omni.runner import merge_streaming_additional_information

WORKER_CLS = "easymagpie_vllm_omni.runner.EasyMagpieGPUARWorker"


def test_streaming_update_preserves_model_state_and_replaces_latest_chunk():
    cached = {
        "decode_offset": 7,
        "text_tokens": [10, 20],
        "text_token": [20],
        "meta": {"num_processed_tokens": 3},
    }

    merged = merge_streaming_additional_information(cached, {"text_token": [30]})

    assert merged["decode_offset"] == 7
    assert merged["text_tokens"] == [10, 20]
    assert merged["text_token"] == [30]
    assert merged["meta"]["num_processed_tokens"] == 0
    assert merged["meta"]["resumable"] is True


def test_streaming_update_accumulates_declared_tensor_keys():
    cached = {"hidden_states": {"output": torch.tensor([[1.0]])}}
    incoming = {"hidden_states": {"output": torch.tensor([[2.0]])}}

    merged = merge_streaming_additional_information(
        cached,
        incoming,
        accumulated_keys={("hidden_states", "output")},
    )

    torch.testing.assert_close(merged["hidden_states"]["output"], torch.tensor([[1.0], [2.0]]))


def test_deploy_configs_select_compatibility_worker_for_lm():
    for filename in ("easymagpie_lm.yaml", "easymagpie.yaml"):
        deploy = yaml.safe_load((EASYMAGPIE_ROOT / "deploy" / filename).read_text())
        lm_stage = next(stage for stage in deploy["stages"] if stage["stage_id"] == 0)
        assert lm_stage["engine_extras"]["worker_cls"] == WORKER_CLS


def test_deploy_configs_initialize_tokenizer_for_multimodal_profiling():
    for filename in ("easymagpie_lm.yaml", "easymagpie.yaml"):
        deploy = yaml.safe_load((EASYMAGPIE_ROOT / "deploy" / filename).read_text())
        lm_stage = next(stage for stage in deploy["stages"] if stage["stage_id"] == 0)
        assert lm_stage["skip_tokenizer_init"] is False


def test_offline_demo_reuses_one_engine_with_self_contained_requests():
    notebook_path = EASYMAGPIE_ROOT.parents[1] / "tutorials" / "tts" / "easymagpie_vllm_omni" / "offline_demo.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")

    assert source.count("AsyncOmni(") == 1
    assert code.count("omni.generate(") == 3
    assert source.count("omni.shutdown()") == 1
    assert code.count("yield StreamingInput(") == 2
    assert "await turn_finished[0].wait()" in code
    assert "await turn_finished[1].wait()" in code
    assert "is_audio_segment_finished(stage_output)" in code
    assert "def finish_reason(" not in code
    assert code.count("\"reset_codec_on_segment\": True") == 2
    assert "arch.ensure_reference_audio_available()" in code
    assert "arch.require_user_audio_prefill()" in code
    assert "arch.codec_input_sample_rate" in code
    assert "read_mono" not in code
    assert "resample" not in code
    assert "MULTITURN_OUT_WAVS = [\"out_multiturn_turn1.wav\", \"out_multiturn_turn2.wav\"]" in source
    assert "SPEAKER_ID = \"eng\"" in source
    assert "an4_clstk/fash/cen5-fash-b.wav" in source
    assert "an4_clstk/ffmm/cen5-ffmm-b.wav" in source
    assert "an4_clstk/fash/an251-fash-b.wav" in source
    assert "an4_clstk/fash/an253-fash-b.wav" in source
    assert "ASSISTANT_TURN_1_TEXT" in source
    assert "ASSISTANT_TURN_2_TEXT" in source
    assert "class MultiturnInput" not in source
