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
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from conftest import EASYMAGPIE_ROOT
from easymagpie_vllm_omni.cfg_pairing import CFGPair
from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner, merge_streaming_additional_information
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

WORKER_CLS = "easymagpie_vllm_omni.runner.EasyMagpieGPUARWorker"


@pytest.mark.parametrize("use_cfg", [False, True])
def test_cfg_runner_builds_layout_after_parent_preprocess_reorders_rows(
    monkeypatch: pytest.MonkeyPatch,
    use_cfg: bool,
) -> None:
    """Verify CFG runner builds layout after parent preprocess reorders rows.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
        use_cfg: Whether the request enables classifier-free guidance.
    """
    pair = CFGPair.create("speech")
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = SimpleNamespace()
    runner.requests = {
        rid: SimpleNamespace(additional_information_cpu={**pair.metadata(rid), "use_cfg": use_cfg})
        for rid in pair.physical_ids
    }
    ids = list(pair.physical_ids)
    runner.input_batch = SimpleNamespace(req_ids=ids, num_computed_tokens_cpu=[5, 5])
    step = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(ids, 1))

    def prepare(self, *args: Any) -> str:
        """Simulate base-runner input preparation and request reordering.

        Args:
            *args: Additional positional arguments forwarded to the base API or accepted by the test double.

        Returns:
            The base-runner result after the simulated request reordering.
        """
        ids.reverse()
        return "prepared"

    monkeypatch.setattr(GPUARModelRunner, "_preprocess", prepare)
    assert runner._preprocess(step, 2) == "prepared"
    assert runner.model.cfg_pairing_layout.conditional_rows == ((1,),)
    assert runner.model.cfg_pairing_layout.unconditional_rows == ((0,),)
    assert (runner.model.cfg_guidance_layout is not None) is use_cfg


def test_ordinary_runner_batch_clears_previous_pair_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ordinary runner batch clears previous pair layout.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
    """
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = SimpleNamespace(cfg_pairing_layout="old")
    runner.input_batch = SimpleNamespace(req_ids=["ordinary"])
    monkeypatch.setattr(GPUARModelRunner, "_preprocess", lambda self, *args: "prepared")
    assert runner._preprocess(None, 1) == "prepared"
    assert runner.model.cfg_pairing_layout is None


def test_cfg_runner_does_not_build_private_codec_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CFG runner does not build private codec payload.

    Args:
        monkeypatch: Fixture for replacing dependencies during the test.
    """
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    pair = CFGPair.create("speech")
    calls = []

    def downstream(self, rid: str) -> bool:
        """Record whether downstream payload processing reaches the base runner.

        Args:
            rid: Request identity.

        Returns:
            Whether the requested condition holds.
        """
        calls.append(rid)
        return True

    monkeypatch.setattr(GPUARModelRunner, "_request_needs_downstream_stage_payload", downstream)
    assert runner._request_needs_downstream_stage_payload(pair.unconditional_id) is False
    assert runner._request_needs_downstream_stage_payload(pair.conditional_id) is True
    assert calls == [pair.conditional_id]


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


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({}, True),
        ({"use_cfg": True, "cfg_scale": 2.5}, True),
        ({"use_cfg": False, "cfg_scale": 9.0}, True),
        ({"temperature": 0.0}, False),
        ({"temperature": 0.8}, False),
        ({"top_k": 40}, False),
        ({"use_cfg": True, "cfg_scale": 1.0}, False),
    ],
)
def test_shared_sampling_settings_choose_fallback(
    settings: dict[str, Any],
    expected: bool,
) -> None:
    """Only default acoustic settings qualify for the captured shared path."""
    from easymagpie_vllm_omni.cfg import AcousticSamplingSettings

    assert AcousticSamplingSettings.from_info(settings).uses_fixed_sampling is expected


def test_sampling_fallback_disables_graphs_and_restores_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never replay a default-settings graph for a non-default batch."""
    import easymagpie_vllm_omni.runner as runner_module
    from vllm.config import CUDAGraphMode

    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = SimpleNamespace(use_fixed_acoustic_sampling=False)
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    monkeypatch.setattr(runner_module, "get_forward_context", lambda: context)
    modes = []

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Record the graph policy observed by the parent forward."""
        modes.append(context.cudagraph_runtime_mode)
        return "fallback"

    monkeypatch.setattr(GPUARModelRunner, "_model_forward", forward)
    assert runner._model_forward() == "fallback"
    assert context.cudagraph_runtime_mode == CUDAGraphMode.FULL
    runner.model.use_fixed_acoustic_sampling = True
    runner._model_forward()
    assert modes == [CUDAGraphMode.NONE, CUDAGraphMode.FULL]


def test_runner_resets_sampling_policy_before_each_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom batch must not force later default batches onto the fallback."""
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = SimpleNamespace(use_fixed_acoustic_sampling=False)
    runner.input_batch = SimpleNamespace(req_ids=["ordinary"])
    flags = []

    def prepare(
        self,
        *args: Any,
    ) -> str:
        """Record the reset before simulating a custom request's preprocessing."""
        flags.append(self.model.use_fixed_acoustic_sampling)
        self.model.use_fixed_acoustic_sampling = False
        return "prepared"

    monkeypatch.setattr(GPUARModelRunner, "_preprocess", prepare)
    for _ in range(2):
        assert runner._preprocess(None, 1) == "prepared"
    assert flags == [True, True]
