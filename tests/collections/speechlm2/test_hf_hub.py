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
from huggingface_hub import PyTorchModelHubMixin

import nemo.collections.speechlm2.parts.hf_hub as hf_hub
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin, _inject_local_artifact_paths


class _DummyHubModel(HFHubMixin):
    pass


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


def _capture_pretrained_config(tmp_path, monkeypatch, repo_trust_remote_code, **model_kwargs):
    config_path = tmp_path / "config.json"
    config_path.write_text(f"trust_remote_code: {str(repo_trust_remote_code).lower()}\n")

    def fake_cached_file(_model_id, filename, **_kwargs):
        return str(config_path) if filename == hf_hub.CONFIG_NAME else None

    captured = {}

    def fake_from_pretrained(_cls, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(hf_hub, "cached_file", fake_cached_file)
    monkeypatch.setattr(PyTorchModelHubMixin, "_from_pretrained", classmethod(fake_from_pretrained))

    _DummyHubModel._from_pretrained(
        model_id="untrusted/repository",
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=True,
        token=None,
        **model_kwargs,
    )
    return captured["cfg"]


@pytest.mark.parametrize(
    ("repo_trust_remote_code", "model_kwargs", "expected"),
    [
        pytest.param(True, {}, False, id="repository-cannot-opt-in"),
        pytest.param(True, {"trust_remote_code": False}, False, id="explicit-opt-out-wins"),
        pytest.param(False, {"trust_remote_code": True}, True, id="explicit-opt-in-wins"),
    ],
)
def test_from_pretrained_remote_code_requires_explicit_opt_in(
    tmp_path, monkeypatch, repo_trust_remote_code, model_kwargs, expected
):
    cfg = _capture_pretrained_config(tmp_path, monkeypatch, repo_trust_remote_code, **model_kwargs)

    assert cfg["trust_remote_code"] is expected


def test_save_pretrained_does_not_persist_remote_code_trust(tmp_path, monkeypatch):
    captured = {}

    def fake_save_pretrained(_self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(PyTorchModelHubMixin, "save_pretrained", fake_save_pretrained)
    model = object.__new__(_DummyHubModel)
    model.cfg = {"trust_remote_code": True}

    model.save_pretrained(tmp_path)

    assert "trust_remote_code" not in captured["config"]
    assert model.cfg["trust_remote_code"] is True


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
