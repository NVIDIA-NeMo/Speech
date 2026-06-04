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

import sys
import types
from contextlib import contextmanager
from importlib import import_module

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.speechlm2.parts import fp8


def install_fake_module(monkeypatch, name, module):
    parts = name.split(".")
    for idx in range(1, len(parts) + 1):
        full_name = ".".join(parts[:idx])
        if idx == len(parts):
            current_module = module
        else:
            current_module = sys.modules.get(full_name)
            if current_module is None:
                current_module = types.ModuleType(full_name)
                current_module.__path__ = []
        monkeypatch.setitem(sys.modules, full_name, current_module)

    for idx in range(1, len(parts)):
        parent_name = ".".join(parts[:idx])
        child_name = parts[idx]
        child_full_name = ".".join(parts[: idx + 1])
        monkeypatch.setattr(sys.modules[parent_name], child_name, sys.modules[child_full_name], raising=False)


def test_install_fake_module_preserves_real_package_paths():
    nemo_automodel = pytest.importorskip("nemo_automodel")
    original_path = list(nemo_automodel.__path__)

    with pytest.MonkeyPatch.context() as monkeypatch:
        fake_te_patches = types.ModuleType("nemo_automodel.shared.te_patches")
        fake_te_patches.apply_te_patches = lambda: None
        install_fake_module(monkeypatch, "nemo_automodel.shared.te_patches", fake_te_patches)

        assert list(nemo_automodel.__path__) == original_path
        import_module("nemo_automodel.components.distributed.config")

    assert list(nemo_automodel.__path__) == original_path
    import_module("nemo_automodel.components.distributed.config")


@contextmanager
def recording_context(events, name):
    events.append(f"{name}:enter")
    try:
        yield
    finally:
        events.append(f"{name}:exit")


def test_fp8_config_detection_and_validation():
    assert fp8.has_torchao_fp8(DictConfig({"fp8": {"enabled": True}}))
    assert not fp8.has_torchao_fp8(DictConfig({"fp8": {"enabled": False}}))
    assert fp8.has_te_fp8(DictConfig({"te_fp8": {"recipe": "block"}}))
    assert not fp8.has_te_fp8(DictConfig({"te_fp8": None}))

    with pytest.raises(ValueError, match="only one FP8 mode"):
        fp8.validate_fp8_config(
            DictConfig({"fp8": {"enabled": True}, "automodel_backend": {"te_fp8": {"recipe": "block"}}})
        )


def test_maybe_apply_te_patches_only_when_te_fp8_is_configured(monkeypatch):
    calls = []
    te_patches_module = types.ModuleType("nemo_automodel.shared.te_patches")
    te_patches_module.apply_te_patches = lambda: calls.append("patched")
    install_fake_module(monkeypatch, "nemo_automodel.shared.te_patches", te_patches_module)

    fp8.maybe_apply_te_patches(DictConfig({}))
    assert calls == []

    fp8.maybe_apply_te_patches(DictConfig({"te_fp8": {"recipe": "block"}}))
    assert calls == ["patched"]


def test_make_fp8_config_builds_automodel_fp8_config(monkeypatch):
    sentinel = object()
    seen = {}

    fp8_module = types.ModuleType("nemo_automodel.components.quantization.fp8")

    def fake_build_fp8_config(cfg):
        seen["cfg"] = cfg
        return sentinel

    fp8_module.build_fp8_config = fake_build_fp8_config
    install_fake_module(monkeypatch, "nemo_automodel.components.quantization.fp8", fp8_module)

    cfg = DictConfig(
        {
            "fp8": {
                "enabled": True,
                "recipe_name": "tensorwise",
                "filter_fqns": ["lm_head"],
            }
        }
    )

    assert fp8.make_fp8_config(cfg) is sentinel
    assert seen["cfg"] == {
        "enabled": True,
        "recipe_name": "tensorwise",
        "filter_fqns": ["lm_head"],
    }
    assert fp8.make_fp8_config(DictConfig({})) is None


def test_te_fp8_context_builds_te_config_and_strips_target(monkeypatch):
    events = []
    seen = []

    common_utils_module = types.ModuleType("nemo_automodel.components.models.common.utils")

    class FakeTEFp8Config:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def maybe_te_autocast(self):
            return recording_context(events, "te_fp8")

    common_utils_module.TEFp8Config = FakeTEFp8Config
    install_fake_module(monkeypatch, "nemo_automodel.components.models.common.utils", common_utils_module)

    automodel_backend_config = DictConfig(
        {
            "te_fp8": {
                "_target_": "nemo_automodel.components.models.common.utils.TEFp8Config",
                "recipe": "block",
            }
        }
    )

    with fp8.te_fp8_context(automodel_backend_config):
        events.append("body")

    assert seen == [{"recipe": "block"}]
    assert events == ["te_fp8:enter", "body", "te_fp8:exit"]

    with fp8.te_fp8_context(DictConfig({})):
        events.append("no_te")
    assert events[-1] == "no_te"


def test_maybe_pad_bshd_inputs_for_te_fp8_noops_without_te_fp8():
    input_embeds = torch.ones(1, 5, 16)
    attention_mask = torch.ones(1, 5, dtype=torch.bool)

    padded, padded_mask, llm_kwargs, original_seq_len = fp8.maybe_pad_bshd_inputs_for_te_fp8(
        None,
        input_embeds,
        attention_mask,
    )

    assert padded is input_embeds
    assert padded_mask is attention_mask
    assert llm_kwargs == {}
    assert original_seq_len == 5


def test_maybe_pad_bshd_inputs_for_te_fp8_pads_sequence_tensors():
    input_embeds = torch.ones(2, 5, 16)
    attention_mask = torch.ones(2, 5, dtype=torch.bool)
    position_ids = torch.arange(5).expand(2, -1)

    padded, padded_mask, llm_kwargs, original_seq_len = fp8.maybe_pad_bshd_inputs_for_te_fp8(
        DictConfig({"recipe": "block"}),
        input_embeds,
        attention_mask,
        {"position_ids": position_ids},
    )

    assert original_seq_len == 5
    assert padded.shape == (2, 8, 16)
    assert padded_mask.shape == (2, 8)
    assert llm_kwargs["position_ids"].shape == (2, 8)
    assert torch.equal(padded[:, :5], input_embeds)
    assert (padded[:, 5:] == 0).all()
    assert padded_mask.all()
    assert (llm_kwargs["position_ids"][:, 5:] == 0).all()


def test_te_fp8_hidden_size_validation():
    te_fp8_config = DictConfig({"recipe": "block"})

    with pytest.raises(ValueError, match="hidden size"):
        fp8.maybe_pad_bshd_inputs_for_te_fp8(te_fp8_config, torch.ones(1, 5, 15), None)


def test_maybe_pad_thd_padded_lengths_for_te_fp8_preserves_cp_alignment():
    te_fp8_config = DictConfig({"recipe": "block"})

    padded_lens = fp8.maybe_pad_thd_padded_lengths_for_te_fp8(te_fp8_config, [8, 4], cp_size=2, tp_size=1)

    assert padded_lens == [8, 8]
    assert sum(padded_lens) % (8 * 2) == 0
    assert all(length % 4 == 0 for length in padded_lens)
    assert (sum(padded_lens) // 2) % 8 == 0


def test_maybe_pad_thd_padded_lengths_for_te_fp8_accounts_for_cp_and_tp():
    te_fp8_config = DictConfig({"recipe": "block"})

    padded_lens = fp8.maybe_pad_thd_padded_lengths_for_te_fp8(te_fp8_config, [8, 4], cp_size=2, tp_size=3)

    assert padded_lens == [8, 40]
    assert sum(padded_lens) % (8 * 2 * 3) == 0
    assert all(length % 4 == 0 for length in padded_lens)


def test_maybe_precompute_float8_dynamic_scale_for_fsdp_guards(monkeypatch):
    calls = []
    torchao_float8_module = types.ModuleType("torchao.float8")
    torchao_float8_module.precompute_float8_dynamic_scale_for_fsdp = lambda llm: calls.append(llm)
    install_fake_module(monkeypatch, "torchao.float8", torchao_float8_module)

    class MeshDim:
        def __init__(self, size):
            self.value = size

        def size(self):
            return self.value

    class DeviceMesh:
        mesh_dim_names = ("dp_shard",)

        def __init__(self, dp_shard_size):
            self.dp_shard_size = dp_shard_size

        def __getitem__(self, name):
            assert name == "dp_shard"
            return MeshDim(self.dp_shard_size)

    cfg = DictConfig(
        {
            "fp8": {
                "enabled": True,
                "precompute_float8_dynamic_scale_for_fsdp": True,
            }
        }
    )
    llm = object()

    fp8.maybe_precompute_float8_dynamic_scale_for_fsdp(DictConfig({}), llm, DeviceMesh(2), True)
    fp8.maybe_precompute_float8_dynamic_scale_for_fsdp(cfg, llm, DeviceMesh(2), False)
    fp8.maybe_precompute_float8_dynamic_scale_for_fsdp(cfg, llm, DeviceMesh(1), True)
    assert calls == []

    fp8.maybe_precompute_float8_dynamic_scale_for_fsdp(cfg, llm, DeviceMesh(2), True)
    assert calls == [llm]
