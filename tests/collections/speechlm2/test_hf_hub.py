# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from nemo.collections.speechlm2.parts import hf_hub
from nemo.collections.speechlm2.parts.hf_hub import _inject_local_artifact_paths


def _cached_file_kwargs():
    return {
        "cache_dir": None,
        "force_download": False,
        "local_files_only": True,
        "token": None,
        "revision": None,
        "_raise_exceptions_for_gated_repo": False,
        "_raise_exceptions_for_missing_entries": False,
        "_raise_exceptions_for_connection_errors": False,
    }


def _write_local_export_artifacts(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "llm_backbone").mkdir()
    (tmp_path / "llm_backbone" / "config.json").write_text("{}")


def test_inject_local_artifact_paths_salm_config(tmp_path):
    _write_local_export_artifacts(tmp_path)
    cfg = {
        "pretrained_llm": "remote-llm",
        "pretrained_asr": "remote-asr",
    }

    _inject_local_artifact_paths(cfg, str(tmp_path), _cached_file_kwargs())

    assert cfg["pretrained_llm"] == str(tmp_path / "llm_backbone")
    assert cfg["pretrained_asr"] == "remote-asr"
    assert cfg["tokenizer_path"] == str(tmp_path)


def test_inject_local_artifact_paths_duplex_eartts_config(tmp_path):
    _write_local_export_artifacts(tmp_path)
    cfg = {
        "pretrained_lm_name": "remote-llm",
        "tts_config": {},
    }

    _inject_local_artifact_paths(cfg, str(tmp_path), _cached_file_kwargs())

    assert cfg["pretrained_lm_name"] == str(tmp_path / "llm_backbone")
    assert cfg["tokenizer_path"] == str(tmp_path)


def test_inject_local_artifact_paths_no_artifacts_keeps_old_config(tmp_path):
    cfg = {
        "pretrained_llm": "remote-llm",
        "pretrained_weights": True,
    }

    _inject_local_artifact_paths(cfg, str(tmp_path), _cached_file_kwargs())

    assert cfg == {
        "pretrained_llm": "remote-llm",
        "pretrained_weights": True,
    }
def test_distributed_loader_forwards_nested_automodel_controls(monkeypatch, tmp_path):
    config_path = tmp_path / hf_hub.CONFIG_NAME
    config_path.write_text(
        json.dumps(
            {
                "pretrained_weights": True,
                "use_nemo_automodel": True,
            }
        ),
        encoding="utf-8",
    )
    weight_path = tmp_path / hf_hub.SAFETENSORS_SINGLE_FILE
    weight_path.touch()

    def fake_cached_file(_model_id, filename, **_kwargs):
        if filename == hf_hub.CONFIG_NAME:
            return str(config_path)
        if filename == hf_hub.SAFETENSORS_SINGLE_FILE:
            return str(weight_path)
        raise AssertionError(filename)

    monkeypatch.setattr(hf_hub, "cached_file", fake_cached_file)
    monkeypatch.setattr(
        hf_hub,
        "_load_state_dict_with_dtensors",
        lambda model, weight_dir: setattr(model, "loaded_from", weight_dir),
    )

    class FakeSALM:
        def __init__(self, cfg):
            self.cfg = cfg
            self.configure_kwargs = None

        def configure_model(self, **kwargs):
            self.configure_kwargs = kwargs

    device_mesh = object()
    moe_mesh = object()
    distributed_config = object()
    moe_config = object()
    backend = object()

    model = hf_hub.HFHubMixin._from_pretrained.__func__(
        FakeSALM,
        model_id="checkpoint",
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=True,
        token=None,
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        distributed_config=distributed_config,
        moe_config=moe_config,
        activation_checkpointing=True,
        backend=backend,
        sdpa_method=["math"],
        use_liger_kernel=False,
        torch_dtype=torch.bfloat16,
        architectures=["NemotronHForCausalLM"],
    )

    assert model.configure_kwargs == {
        "device_mesh": device_mesh,
        "distributed_config": distributed_config,
        "moe_config": moe_config,
        "moe_mesh": moe_mesh,
        "activation_checkpointing_llm": True,
        "activation_checkpointing_perception": None,
        "llm_automodel_kwargs": {
            "backend": backend,
            "sdpa_method": ["math"],
            "use_liger_kernel": False,
        },
    }
    assert model.cfg["torch_dtype"] == "bfloat16"
    assert model.loaded_from == str(tmp_path)
