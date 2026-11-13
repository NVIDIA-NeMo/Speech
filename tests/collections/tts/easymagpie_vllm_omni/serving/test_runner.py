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
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import yaml

from conftest import EASYMAGPIE_ROOT
from easymagpie_vllm_omni import runner as runner_module
from easymagpie_vllm_omni.runner import EasyMagpieGPUARWorker, merge_streaming_additional_information

WORKER_CLS = "easymagpie_vllm_omni.runner.EasyMagpieGPUARWorker"


def _grpo_worker():
    class Policy(torch.nn.Module):
        hf_config = SimpleNamespace(grpo_rollout=True)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.fail = False
            self.calls = 0
            self._prefill_cache = {"context": torch.ones(1)}

        def load_weights(self, weights):
            self.calls += 1
            if self.fail:
                raise RuntimeError("weight load failed")
            values = dict(weights)
            with torch.no_grad():
                self.weight.copy_(values["decoder.weight"])
            return {"decoder.weight"}

    worker = EasyMagpieGPUARWorker.__new__(EasyMagpieGPUARWorker)
    worker.device = torch.device("cpu")
    worker.model_config = SimpleNamespace(dtype=torch.float32, quantization=None, enable_sleep_mode=False)
    worker.model_runner = SimpleNamespace(model=Policy(), requests={}, model_config=worker.model_config)
    return worker


def test_grpo_weight_reload_updates_policy_and_acknowledges_versions(tmp_path):
    worker = _grpo_worker()
    path = tmp_path / "policy.pt"
    torch.save({"decoder.weight": torch.ones(2)}, path)

    assert worker.reload_grpo_weights(str(path), 0) == {"version": 0, "loaded_weights": 1}
    torch.testing.assert_close(worker.model_runner.model.weight, torch.ones(2))
    assert worker.model_runner.model._prefill_cache == {}
    assert worker.reload_grpo_weights(str(path), 0) == {"version": 0, "loaded_weights": 1}
    assert worker.model_runner.model.calls == 1
    torch.save({"decoder.weight": torch.full((2,), 2.0)}, path)
    assert worker.reload_grpo_weights(str(path), 1)["version"] == 1
    torch.testing.assert_close(worker.model_runner.model.weight, torch.full((2,), 2.0))
    with pytest.raises(ValueError, match="stale GRPO version"):
        worker.reload_grpo_weights(str(path), 0)


def test_failed_grpo_reload_does_not_acknowledge_new_version(tmp_path):
    worker = _grpo_worker()
    path = tmp_path / "policy.pt"
    torch.save({"decoder.weight": torch.ones(2)}, path)
    worker.reload_grpo_weights(str(path), 0)
    worker.model_runner.model.fail = True

    with pytest.raises(RuntimeError, match="weight load failed"):
        worker.reload_grpo_weights(str(path), 1)

    assert worker._grpo_weight_version == 0
    worker.model_runner.model.fail = False
    assert worker.reload_grpo_weights(str(path), 1)["version"] == 1


def test_failed_grpo_weight_postprocessing_does_not_acknowledge_version(tmp_path, monkeypatch):
    worker = _grpo_worker()
    path = tmp_path / "policy.pt"
    torch.save({"decoder.weight": torch.ones(2)}, path)

    def fail_postprocessing(*args):
        raise RuntimeError("weight postprocessing failed")

    monkeypatch.setattr(runner_module, "process_weights_after_loading", fail_postprocessing)

    with pytest.raises(RuntimeError, match="weight postprocessing failed"):
        worker.reload_grpo_weights(str(path), 0)

    assert not hasattr(worker, "_grpo_weight_version")


@pytest.mark.parametrize("postprocessing_fails", [False, True])
def test_refit_and_postprocessing_use_the_sleep_managed_weight_pool(tmp_path, monkeypatch, postprocessing_fails):
    worker = _grpo_worker()
    worker.model_config.enable_sleep_mode = True
    path = tmp_path / "policy.pt"
    torch.save({"decoder.weight": torch.ones(2)}, path)
    events = []

    existing_pool = object()
    allocator = SimpleNamespace(allocator_and_pools={"weights": (existing_pool, object())}, current_tag="default")

    @contextmanager
    def weight_pool(pool, device):
        assert pool is existing_pool and device == worker.device
        assert allocator.current_tag == "weights"
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(runner_module.CuMemAllocator, "get_instance", staticmethod(lambda: allocator))
    monkeypatch.setattr(torch.cuda.memory, "use_mem_pool", weight_pool)
    load_weights = worker.model_runner.model.load_weights

    def load_in_pool(weights):
        assert events == ["enter"]
        events.append("load")
        return load_weights(weights)

    def postprocess_in_pool(*args):
        assert events == ["enter", "load"]
        events.append("postprocess")
        if postprocessing_fails:
            raise RuntimeError("postprocessing failed")

    monkeypatch.setattr(worker.model_runner.model, "load_weights", load_in_pool)
    monkeypatch.setattr(runner_module, "process_weights_after_loading", postprocess_in_pool)
    if postprocessing_fails:
        with pytest.raises(RuntimeError, match="postprocessing failed"):
            worker.reload_grpo_weights(str(path), 0)
        assert not hasattr(worker, "_grpo_weight_version")
    else:
        assert worker.reload_grpo_weights(str(path), 0)["version"] == 0

    assert events == ["enter", "load", "postprocess", "exit"]
    assert allocator.current_tag == "default"
    assert allocator.allocator_and_pools["weights"][0] is existing_pool


@pytest.mark.parametrize("requests", [{"active": object()}, None])
def test_grpo_reload_requires_an_idle_runner(requests):
    worker = _grpo_worker()
    worker.model_runner.requests = requests

    with pytest.raises(RuntimeError, match="idle|Finish all rollout"):
        worker.reload_grpo_weights("unused.pt", 0)


def test_grpo_reload_requires_explicit_rollout_artifact():
    worker = _grpo_worker()
    worker.model_runner.model.hf_config = SimpleNamespace()

    with pytest.raises(RuntimeError, match="--grpo-rollout"):
        worker.reload_grpo_weights("unused.pt", 0)


@pytest.mark.parametrize("version", [-1, True, 1.5, "1"])
def test_grpo_reload_rejects_invalid_versions(version):
    with pytest.raises(ValueError, match="non-negative integer"):
        _grpo_worker().reload_grpo_weights("unused.pt", version)


@pytest.mark.parametrize("weights", [{}, {"decoder.weight": "invalid"}, [torch.ones(2)]])
def test_grpo_reload_rejects_invalid_snapshot(tmp_path, weights):
    path = tmp_path / "invalid.pt"
    torch.save(weights, path)
    worker = _grpo_worker()

    with pytest.raises(ValueError, match="non-empty mapping"):
        worker.reload_grpo_weights(str(path), 0)

    assert not hasattr(worker, "_grpo_weight_version")


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
