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

import json

import pytest
import torch
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import save_file

import nemo.collections.speechlm2.parts.hf_hub as hf_hub
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin, _inject_local_artifact_paths, _load_sharded_safetensors


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


def test_from_pretrained_single_file_path_forwards_required_hub_kwargs(tmp_path, monkeypatch):
    """The single-file fallback relies on **model_kwargs to still carry proxies/resume_download
    through to PyTorchModelHubMixin._from_pretrained, which requires them (no default value).
    Regression test: a blanket model_kwargs.pop() of these before this branch would break it,
    but a permissive fake_from_pretrained (as in the other tests here) wouldn't catch that."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    def fake_cached_file(_model_id, filename, **_kwargs):
        return str(config_path) if filename == hf_hub.CONFIG_NAME else None

    def strict_from_pretrained(_cls, *, proxies, resume_download, **_kwargs):
        return object()

    monkeypatch.setattr(hf_hub, "cached_file", fake_cached_file)
    monkeypatch.setattr(PyTorchModelHubMixin, "_from_pretrained", classmethod(strict_from_pretrained))

    _DummyHubModel._from_pretrained(
        model_id="some/repo",
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=True,
        token=None,
        proxies=None,
        resume_download=None,
    )


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


class _TinyEncoderLLM(torch.nn.Module):
    """Stand-in for a SALM-shaped model with two submodules a checkpoint could be sharded by."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4, bias=False)
        self.llm = torch.nn.Linear(4, 4, bias=False)


def _write_sharded_checkpoint(tmp_path, state_dict_by_shard: dict[str, dict[str, torch.Tensor]]):
    """Write shard files + an index.json; shard filenames are used as given."""
    weight_map = {}
    for shard_filename, shard_state_dict in state_dict_by_shard.items():
        save_file(shard_state_dict, str(tmp_path / shard_filename))
        weight_map.update({name: shard_filename for name in shard_state_dict})
    index_path = tmp_path / hf_hub.SAFETENSORS_INDEX_FILE
    index_path.write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    return str(index_path)


def test_load_sharded_safetensors_loads_weights_split_by_submodule(tmp_path):
    expected_encoder_weight = torch.randn(4, 4)
    expected_llm_weight = torch.randn(4, 4)
    index_file = _write_sharded_checkpoint(
        tmp_path,
        {
            "encoder.safetensors": {"encoder.weight": expected_encoder_weight},
            "llm.safetensors": {"llm.weight": expected_llm_weight},
        },
    )

    model = _load_sharded_safetensors(
        cls=_TinyEncoderLLM,
        model_kwargs={},
        model_id=str(tmp_path),
        index_file=index_file,
        cached_file_kwargs=_cached_file_kwargs(),
        map_location="cpu",
        strict=True,
    )

    assert isinstance(model, _TinyEncoderLLM)
    torch.testing.assert_close(model.encoder.weight, expected_encoder_weight)
    torch.testing.assert_close(model.llm.weight, expected_llm_weight)


def test_load_sharded_safetensors_loads_weights_split_by_size(tmp_path):
    """Conventional model-XXXXX-of-XXXXX.safetensors naming works the same way."""
    expected_encoder_weight = torch.randn(4, 4)
    expected_llm_weight = torch.randn(4, 4)
    index_file = _write_sharded_checkpoint(
        tmp_path,
        {
            "model-00001-of-00002.safetensors": {"encoder.weight": expected_encoder_weight},
            "model-00002-of-00002.safetensors": {"llm.weight": expected_llm_weight},
        },
    )

    model = _load_sharded_safetensors(
        cls=_TinyEncoderLLM,
        model_kwargs={},
        model_id=str(tmp_path),
        index_file=index_file,
        cached_file_kwargs=_cached_file_kwargs(),
        map_location="cpu",
        strict=True,
    )

    torch.testing.assert_close(model.encoder.weight, expected_encoder_weight)
    torch.testing.assert_close(model.llm.weight, expected_llm_weight)


def test_load_sharded_safetensors_missing_shard_raises(tmp_path):
    index_path = tmp_path / hf_hub.SAFETENSORS_INDEX_FILE
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {"encoder.weight": "missing.safetensors", "llm.weight": "missing.safetensors"},
            }
        )
    )

    with pytest.raises(RuntimeError, match="missing.safetensors"):
        _load_sharded_safetensors(
            cls=_TinyEncoderLLM,
            model_kwargs={},
            model_id=str(tmp_path),
            index_file=str(index_path),
            cached_file_kwargs=_cached_file_kwargs(),
            map_location="cpu",
            strict=True,
        )


class _DummyShardedModel(HFHubMixin, torch.nn.Module):
    """Minimal SALM-shaped model (mixin + nn.Module) for a real cls(**model_kwargs) test."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.llm = torch.nn.Linear(2, 2, bias=False)


def test_from_pretrained_dispatches_to_sharded_loader(tmp_path, monkeypatch):
    """A sharded checkpoint (index file present) should use the sharded loader, not
    PyTorchModelHubMixin (single-file only)."""
    (tmp_path / "config.json").write_text("{}")
    expected_llm_weight = torch.randn(2, 2)
    _write_sharded_checkpoint(tmp_path, {"model-00001-of-00001.safetensors": {"llm.weight": expected_llm_weight}})

    monkeypatch.setattr(
        PyTorchModelHubMixin,
        "_from_pretrained",
        classmethod(
            lambda *a, **k: pytest.fail(
                "should not delegate to PyTorchModelHubMixin._from_pretrained for a sharded checkpoint"
            )
        ),
    )

    instance = _DummyShardedModel._from_pretrained(
        model_id=str(tmp_path),
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=True,
        token=None,
    )

    assert isinstance(instance, _DummyShardedModel)
    torch.testing.assert_close(instance.llm.weight, expected_llm_weight)
