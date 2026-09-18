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

from types import SimpleNamespace

import pytest
from huggingface_hub import PyTorchModelHubMixin

import nemo.collections.speechlm2.parts.hf_hub as hf_hub
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin, _inject_local_artifact_paths


def _automodel_parameter_names_available():
    try:
        from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
    except ImportError:
        return False
    return canonical_parameter_fqn is not None


def _automodel_storage_reader_available():
    try:
        from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
    except ImportError:
        return False
    return _HuggingFaceStorageReader is not None


_AUTOMODEL_PARAMETER_NAMES_AVAILABLE = _automodel_parameter_names_available()
_AUTOMODEL_STORAGE_READER_AVAILABLE = _automodel_storage_reader_available()
requires_automodel_parameter_names = pytest.mark.skipif(
    not _AUTOMODEL_PARAMETER_NAMES_AVAILABLE,
    reason="nemo_automodel parameter-name canonicalization is not installed",
)
requires_automodel_dtensor_loader = pytest.mark.skipif(
    not (_AUTOMODEL_PARAMETER_NAMES_AVAILABLE and _AUTOMODEL_STORAGE_READER_AVAILABLE),
    reason="required nemo_automodel distributed-checkpoint symbols are not installed",
)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def checkpoint_wrapper(torch):
    del torch
    module = pytest.importorskip("torch.distributed.algorithms._checkpoint.checkpoint_wrapper")
    return module.checkpoint_wrapper


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


def test_from_pretrained_distributed_forwards_strict(tmp_path, monkeypatch):
    config_path = tmp_path / hf_hub.CONFIG_NAME
    config_path.write_text("{}")
    captured = {}
    sentinel = object()

    def fake_cached_file(_model_id, filename, **_kwargs):
        return str(config_path) if filename == hf_hub.CONFIG_NAME else None

    def fake_distributed_from_pretrained(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(hf_hub, "cached_file", fake_cached_file)
    monkeypatch.setattr(hf_hub, "_distributed_from_pretrained", fake_distributed_from_pretrained)
    distributed_setup = SimpleNamespace(mesh_context=SimpleNamespace(device_mesh=object()))

    result = _DummyHubModel._from_pretrained(
        model_id="checkpoint",
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=True,
        token=None,
        strict=True,
        distributed_setup=distributed_setup,
    )

    assert result is sentinel
    assert captured["strict"] is True
    assert captured["distributed_setup"] is distributed_setup


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


@requires_automodel_parameter_names
def test_checkpoint_state_dict_maps_activation_checkpoint_wrappers(torch, checkpoint_wrapper):
    class WrappedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Linear(1, 1, bias=False)
            block.register_buffer("running_mean", torch.zeros(1))
            self.block = checkpoint_wrapper(block)
            self.register_buffer("runtime_only", torch.zeros(1), persistent=False)

    model = WrappedModel()

    assert list(dict(model.named_parameters())) == ["block._checkpoint_wrapped_module.weight"]
    assert list(model.state_dict()) == ["block.weight", "block.running_mean"]

    state_dict = hf_hub._checkpoint_state_dict(model, {"block.weight", "block.running_mean"}, strict=True)

    assert list(state_dict) == ["block.weight", "block.running_mean"]
    assert state_dict["block.weight"] is model.block._checkpoint_wrapped_module.weight
    assert state_dict["block.running_mean"] is model.block._checkpoint_wrapped_module.running_mean


@requires_automodel_parameter_names
def test_checkpoint_state_dict_rejects_partial_parameter_coverage(torch):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False))

    with pytest.raises(RuntimeError, match=r"matched only 50.0%.*missing model parameters.*1.weight"):
        hf_hub._checkpoint_state_dict(model, {"0.weight"})


@requires_automodel_parameter_names
def test_checkpoint_state_dict_accepts_saved_tied_parameter_alias(torch):
    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.z = torch.nn.Linear(1, 1, bias=False)
            self.a = torch.nn.Linear(1, 1, bias=False)
            self.a.weight = self.z.weight

    model = TiedModel()

    assert list(dict(model.named_parameters())) == ["z.weight"]
    assert [name for name, _ in model.named_parameters(remove_duplicate=False)] == ["z.weight", "a.weight"]

    state_dict = hf_hub._checkpoint_state_dict(model, {"a.weight"})

    assert list(state_dict) == ["a.weight"]
    assert state_dict["a.weight"] is model.z.weight


@requires_automodel_parameter_names
def test_checkpoint_state_dict_warns_about_small_parameter_gap(monkeypatch, torch):
    from nemo.utils import logging

    class ModelWithRuntimeParameter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.checkpointed = torch.nn.Parameter(torch.zeros(19))
            self.runtime_only = torch.nn.Parameter(torch.zeros(1))

    model = ModelWithRuntimeParameter()
    warnings = []
    monkeypatch.setattr(logging, "warning", warnings.append)

    state_dict = hf_hub._checkpoint_state_dict(model, {"checkpointed"})

    assert list(state_dict) == ["checkpointed"]
    assert len(warnings) == 1
    assert "matched only 95.0%" in warnings[0]
    assert "runtime_only" in warnings[0]


@requires_automodel_parameter_names
def test_checkpoint_state_dict_loads_one_persistent_buffer_alias(monkeypatch, torch):
    from nemo.utils import logging

    class ModelWithTiedBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.ones(1)
            self.register_buffer("z", shared)
            self.register_buffer("a", shared)

    model = ModelWithTiedBuffer()
    warnings = []
    monkeypatch.setattr(logging, "warning", warnings.append)

    assert list(dict(model.named_buffers())) == ["z"]
    assert [name for name, _ in model.named_buffers(remove_duplicate=False)] == ["z", "a"]

    state_dict = hf_hub._checkpoint_state_dict(model, {"z", "a"})

    assert list(state_dict) == ["z"]
    assert state_dict["z"] is model.z
    assert len(warnings) == 1
    assert "multiple aliases" in warnings[0]
    assert "Each shared tensor is loaded once" in warnings[0]

    warnings.clear()
    hf_hub._checkpoint_state_dict(model, {"z", "a"}, strict=True)
    assert len(warnings) == 1


@requires_automodel_parameter_names
def test_checkpoint_state_dict_accepts_distributed_export_tied_parameter_aliases(monkeypatch, torch):
    from nemo.utils import logging

    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Linear(1, 1, bias=False)
            self.lm_head = torch.nn.Linear(1, 1, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

    model = TiedModel()
    warnings = []
    monkeypatch.setattr(logging, "warning", warnings.append)

    state_dict = hf_hub._checkpoint_state_dict(
        model,
        {"embed_tokens.weight", "lm_head.weight"},
        strict=True,
    )

    assert list(state_dict) == ["embed_tokens.weight"]
    assert state_dict["embed_tokens.weight"] is model.embed_tokens.weight
    assert len(warnings) == 1
    assert "multiple aliases" in warnings[0]
    assert "lm_head.weight" in warnings[0]


@requires_automodel_parameter_names
def test_checkpoint_state_dict_rejects_canonical_name_collision(torch):
    class CollidingModel:
        def named_parameters(self, remove_duplicate=True):
            del remove_duplicate
            return iter(
                [
                    ("block.weight", torch.nn.Parameter(torch.zeros(1))),
                    ("block._checkpoint_wrapped_module.weight", torch.nn.Parameter(torch.ones(1))),
                ]
            )

        def named_modules(self, remove_duplicate=True):
            del remove_duplicate
            return iter(())

    with pytest.raises(RuntimeError, match="Multiple model tensors map to checkpoint key 'block.weight'"):
        hf_hub._checkpoint_state_dict(CollidingModel(), {"block.weight"})


@requires_automodel_parameter_names
def test_checkpoint_state_dict_rejects_unmatched_top_level_component(torch):
    class TwoComponentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.llm = torch.nn.Linear(97, 1, bias=False)
            self.perception = torch.nn.Linear(3, 1, bias=False)

    with pytest.raises(RuntimeError, match=r"97.0%.*top-level model components.*perception"):
        hf_hub._checkpoint_state_dict(TwoComponentModel(), {"llm.weight"})


@requires_automodel_parameter_names
def test_checkpoint_state_dict_rejects_partially_unmatched_top_level_component(torch):
    class PartiallyMatchedPerceptionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.llm = torch.nn.Linear(9700, 1, bias=False)
            self.perception = torch.nn.Module()
            self.perception.adapter = torch.nn.Linear(3, 1, bias=False)
            self.perception.encoder = torch.nn.Linear(300, 1, bias=False)

    checkpoint_keys = {"llm.weight", "perception.adapter.weight"}

    with pytest.raises(RuntimeError, match=r"97.0%.*top-level model components.*perception \(1.0%, 3/303\)"):
        hf_hub._checkpoint_state_dict(PartiallyMatchedPerceptionModel(), checkpoint_keys)


@requires_automodel_parameter_names
def test_checkpoint_state_dict_warns_about_tolerated_top_level_component_gap(monkeypatch, torch):
    from nemo.utils import logging

    model = torch.nn.Module()
    model.llm = torch.nn.Module()
    model.llm.checkpointed = torch.nn.Parameter(torch.zeros(95))
    model.llm.missing = torch.nn.Parameter(torch.zeros(5))
    warnings = []
    monkeypatch.setattr(logging, "warning", warnings.append)

    state_dict = hf_hub._checkpoint_state_dict(model, {"llm.checkpointed"})

    assert list(state_dict) == ["llm.checkpointed"]
    assert len(warnings) == 1
    assert "matched only 95.0%" in warnings[0]
    assert "below 90%" not in warnings[0]


@requires_automodel_parameter_names
def test_checkpoint_state_dict_root_wrapper_keeps_root_parameter_policy(monkeypatch, torch, checkpoint_wrapper):
    from nemo.utils import logging

    model = torch.nn.Module()
    model.checkpointed = torch.nn.Parameter(torch.zeros(95))
    model.missing = torch.nn.Parameter(torch.zeros(5))
    model = checkpoint_wrapper(model)
    warnings = []
    monkeypatch.setattr(logging, "warning", warnings.append)

    state_dict = hf_hub._checkpoint_state_dict(model, {"checkpointed"})

    assert list(state_dict) == ["checkpointed"]
    assert len(warnings) == 1
    assert "matched only 95.0%" in warnings[0]


@requires_automodel_parameter_names
def test_checkpoint_state_dict_strict_mode_rejects_small_gap(torch):
    model = torch.nn.Module()
    model.checkpointed = torch.nn.Parameter(torch.zeros(19))
    model.missing = torch.nn.Parameter(torch.zeros(1))

    with pytest.raises(RuntimeError, match=r"Strict checkpoint loading failed.*95.0%.*missing"):
        hf_hub._checkpoint_state_dict(model, {"checkpointed"}, strict=True)


@pytest.mark.parametrize(
    ("matched_numel", "missing_numel", "raises"),
    [
        pytest.param(9, 1, False, id="exactly-90-percent-warns"),
        pytest.param(89, 11, True, id="below-90-percent-raises"),
    ],
)
@requires_automodel_parameter_names
def test_checkpoint_state_dict_parameter_coverage_boundary(monkeypatch, matched_numel, missing_numel, raises, torch):
    from nemo.utils import logging

    model = torch.nn.Module()
    model.checkpointed = torch.nn.Parameter(torch.zeros(matched_numel))
    model.missing = torch.nn.Parameter(torch.zeros(missing_numel))
    warnings = []
    monkeypatch.setattr(logging, "warning", warnings.append)

    if raises:
        with pytest.raises(RuntimeError, match="matched only 89.0%"):
            hf_hub._checkpoint_state_dict(model, {"checkpointed"})
        assert not warnings
    else:
        hf_hub._checkpoint_state_dict(model, {"checkpointed"})
        assert len(warnings) == 1
        assert "matched only 90.0%" in warnings[0]


@requires_automodel_parameter_names
def test_checkpoint_state_dict_warns_about_missing_persistent_buffer_and_unused_key(monkeypatch, torch):
    from nemo.utils import logging

    model = torch.nn.Module()
    model.register_buffer("persistent", torch.ones(1))
    model.register_buffer("runtime_only", torch.zeros(1), persistent=False)
    warnings = []
    monkeypatch.setattr(logging, "warning", warnings.append)

    state_dict = hf_hub._checkpoint_state_dict(model, {"old_persistent"})

    assert state_dict == {}
    assert len(warnings) == 1
    assert "missing persistent model buffers" in warnings[0]
    assert "['persistent']" in warnings[0]
    assert "runtime_only" not in warnings[0]
    assert "unused checkpoint tensors" in warnings[0]
    assert "old_persistent" in warnings[0]


@requires_automodel_parameter_names
def test_checkpoint_state_dict_treats_non_persistent_checkpoint_buffer_as_unused(monkeypatch, torch):
    from nemo.utils import logging

    model = torch.nn.Module()
    model.register_buffer("runtime_only", torch.zeros(1), persistent=False)
    warnings = []
    monkeypatch.setattr(logging, "warning", warnings.append)

    state_dict = hf_hub._checkpoint_state_dict(model, {"runtime_only"})

    assert state_dict == {}
    assert len(warnings) == 1
    assert "unused checkpoint tensors" in warnings[0]
    assert "runtime_only" in warnings[0]

    warnings.clear()
    with pytest.raises(RuntimeError, match=r"Strict checkpoint loading failed.*runtime_only"):
        hf_hub._checkpoint_state_dict(model, {"runtime_only"}, strict=True)
    assert not warnings


@requires_automodel_dtensor_loader
def test_dtensor_loader_passes_metadata_keys_to_dcp(monkeypatch, tmp_path, torch, checkpoint_wrapper):
    import torch.distributed.checkpoint as dcp
    from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader

    class WrappedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Linear(1, 1, bias=False)
            block.register_buffer("running_mean", torch.zeros(1))
            self.block = checkpoint_wrapper(block)

    model = WrappedModel()
    metadata = SimpleNamespace(state_dict_metadata={"block.weight": object(), "block.running_mean": object()})
    loaded = {}

    monkeypatch.setattr(_HuggingFaceStorageReader, "read_metadata", lambda _self: metadata)
    monkeypatch.setattr(
        dcp,
        "load",
        lambda state_dict, storage_reader: loaded.update(state_dict=state_dict, storage_reader=storage_reader),
    )

    hf_hub._load_state_dict_with_dtensors(model, str(tmp_path))

    assert list(loaded["state_dict"]) == ["block.weight", "block.running_mean"]
    assert loaded["state_dict"]["block.weight"] is model.block._checkpoint_wrapped_module.weight
    assert loaded["state_dict"]["block.running_mean"] is model.block._checkpoint_wrapped_module.running_mean
    assert isinstance(loaded["storage_reader"], _HuggingFaceStorageReader)


@requires_automodel_parameter_names
def test_checkpoint_state_dict_warns_when_ignoring_extra_state_in_strict_mode(monkeypatch, torch):
    from nemo.utils import logging

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(1))
    warnings = []
    monkeypatch.setattr(logging, "warning", warnings.append)

    state_dict = hf_hub._checkpoint_state_dict(
        model,
        {"weight", "transformer.layer._extra_state"},
        strict=True,
    )

    assert list(state_dict) == ["weight"]
    assert len(warnings) == 1
    assert "Ignoring checkpoint extra-state tensors" in warnings[0]
    assert "transformer.layer._extra_state" in warnings[0]


@requires_automodel_parameter_names
def test_checkpoint_state_dict_warns_about_extra_state_before_strict_failure(monkeypatch, torch):
    from nemo.utils import logging

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(1))
    model.register_buffer("persistent", torch.zeros(1))
    warnings = []
    monkeypatch.setattr(logging, "warning", warnings.append)

    with pytest.raises(RuntimeError, match=r"Strict checkpoint loading failed.*missing persistent model buffers"):
        hf_hub._checkpoint_state_dict(model, {"weight", "transformer.layer._extra_state"}, strict=True)

    assert len(warnings) == 1
    assert "Ignoring checkpoint extra-state tensors" in warnings[0]


@requires_automodel_dtensor_loader
def test_dtensor_loader_forwards_strict_to_checkpoint_validation(monkeypatch, tmp_path, torch):
    from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(1))
    metadata = SimpleNamespace(state_dict_metadata={"weight": object(), "unexpected": object()})
    monkeypatch.setattr(_HuggingFaceStorageReader, "read_metadata", lambda _self: metadata)

    with pytest.raises(RuntimeError, match=r"Strict checkpoint loading failed.*unused checkpoint.*unexpected"):
        hf_hub._load_state_dict_with_dtensors(model, str(tmp_path), strict=True)


def test_distributed_loader_forwards_strict(monkeypatch, tmp_path):
    weight_file = tmp_path / hf_hub.SAFETENSORS_SINGLE_FILE
    weight_file.touch()
    loaded = {}

    class FakeModel:
        def __init__(self, cfg):
            self.cfg = cfg

        def configure_model(self, distributed_setup):
            self.distributed_setup = distributed_setup

    monkeypatch.setattr(hf_hub, "cached_file", lambda *_args, **_kwargs: str(weight_file))
    monkeypatch.setattr(
        hf_hub,
        "_load_state_dict_with_dtensors",
        lambda model, weight_dir, strict=False: loaded.update(model=model, weight_dir=weight_dir, strict=strict),
    )

    distributed_setup = object()
    model = hf_hub._distributed_from_pretrained(
        cls=FakeModel,
        model_id="checkpoint",
        model_kwargs={"cfg": {}},
        torch_dtype=None,
        strict=True,
        distributed_setup=distributed_setup,
        cached_file_kwargs={},
    )

    assert loaded == {"model": model, "weight_dir": str(tmp_path), "strict": True}
    assert model.distributed_setup is distributed_setup
