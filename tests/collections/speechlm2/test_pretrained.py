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
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig
from safetensors.torch import save_file

from nemo.collections.speechlm2.parts import pretrained


def test_setup_speech_encoder_hydrates_missing_config_without_weights():
    model = SimpleNamespace(
        cfg=DictConfig(
            {
                "pretrained_asr": "fake-asr",
                "perception": {
                    "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
                    "output_dim": 1,
                    "modality_adapter": {"output_dim": 1},
                },
            }
        ),
        llm=SimpleNamespace(config=SimpleNamespace(hidden_size=8)),
    )
    asr_cfg = DictConfig(
        {
            "preprocessor": {"_target_": "fake.Preprocessor"},
            "encoder": {"d_model": 4, "n_layers": 2},
        }
    )

    with (
        patch.object(pretrained, "load_pretrained_nemo_config", return_value=asr_cfg) as load_config,
        patch.object(pretrained, "AudioPerceptionModule") as perception,
    ):
        pretrained.setup_speech_encoder(model, pretrained_weights=False)

    load_config.assert_called_once_with(pretrained.ASRModel, "fake-asr")
    perception.assert_called_once()
    assert model.cfg.perception.preprocessor._target_ == "fake.Preprocessor"
    assert model.cfg.perception.encoder.n_layers == 2
    assert model.cfg.perception.output_dim == 8
    assert model.cfg.perception.modality_adapter.output_dim == 8


def _mock_automodel_loader(config):
    automodel = SimpleNamespace(from_config=MagicMock(return_value=object()), from_pretrained=MagicMock())
    return (
        automodel,
        patch.object(pretrained.AutoConfig, "from_pretrained", return_value=config),
        patch.dict("sys.modules", {"nemo_automodel": SimpleNamespace(NeMoAutoModelForCausalLM=automodel)}),
        patch("nemo.collections.speechlm2.parts.automodel_compat.remove_automodel_backend_for_hf_fallback"),
    )


def test_load_pretrained_automodel_llm_builds_missing_mtp_before_loading_weights():
    config = SimpleNamespace(num_nextn_predict_layers=0, name_or_path="base-checkpoint")
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(pretrained, "_resolve_automodel_checkpoint_path", return_value="base-checkpoint"),
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        result = pretrained.load_pretrained_automodel_llm(
            "base-checkpoint",
            pretrained_weights=True,
            dtype=torch.bfloat16,
            mtp_config_overrides={
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "*",
                "mtp_layers_block_type": None,
            },
        )

    assert result is automodel.from_config.return_value
    assert config.num_nextn_predict_layers == 1
    assert config.mtp_hybrid_override_pattern == "*"
    assert config.mtp_layers_block_type is None
    automodel.from_config.assert_called_once_with(
        config,
        torch_dtype=torch.bfloat16,
        load_base_model=False,
        trust_remote_code=False,
    )
    base_load.assert_called_once_with(result, "base-checkpoint", {})
    automodel.from_pretrained.assert_not_called()


def test_load_pretrained_automodel_llm_preserves_native_mtp_config_by_default():
    config = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
        name_or_path="native-mtp-checkpoint",
    )
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(pretrained, "_resolve_automodel_checkpoint_path", return_value="native-mtp-checkpoint"),
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        pretrained.load_pretrained_automodel_llm(
            "native-mtp-checkpoint",
            mtp_config_overrides={
                "num_nextn_predict_layers": 2,
                "mtp_hybrid_override_pattern": "**",
            },
        )

    assert config.num_nextn_predict_layers == 1
    assert config.mtp_hybrid_override_pattern == "*E"
    automodel.from_pretrained.assert_called_once_with(
        "native-mtp-checkpoint",
        torch_dtype=torch.float32,
        trust_remote_code=False,
    )
    automodel.from_config.assert_not_called()
    base_load.assert_not_called()


def test_load_pretrained_automodel_llm_can_replace_native_mtp_config():
    config = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
        name_or_path="native-mtp-checkpoint",
    )
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(pretrained, "_resolve_automodel_checkpoint_path", return_value="native-mtp-checkpoint"),
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        result = pretrained.load_pretrained_automodel_llm(
            "native-mtp-checkpoint",
            mtp_config_overrides={
                "num_nextn_predict_layers": 2,
                "mtp_hybrid_override_pattern": "**",
                "mtp_layers_block_type": None,
            },
            replace_mtp_config=True,
        )

    assert config.num_nextn_predict_layers == 2
    assert config.mtp_hybrid_override_pattern == "**"
    assert config.mtp_layers_block_type is None
    automodel.from_config.assert_called_once_with(
        config,
        torch_dtype=torch.float32,
        load_base_model=False,
        trust_remote_code=False,
    )
    base_load.assert_called_once_with(result, "native-mtp-checkpoint", {})


_REPEATED_MTP_OVERRIDES = {
    "num_nextn_predict_layers": 1,
    "mtp_hybrid_override_pattern": "*",
}


@pytest.mark.parametrize(
    "mtp_config_overrides",
    [
        pytest.param(None, id="native-config-only"),
        pytest.param(_REPEATED_MTP_OVERRIDES, id="fallback-config-present"),
    ],
)
def test_load_pretrained_automodel_llm_accepts_one_depth_native_head_as_repeated(mtp_config_overrides):
    config = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
        name_or_path="one-depth-mtp-checkpoint",
    )
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)
    loader_kwargs = {
        "num_nextn_predict_layers": 4,
        "mtp_use_repeated_layer": True,
    }
    if mtp_config_overrides is not None:
        loader_kwargs["mtp_config_overrides"] = mtp_config_overrides

    with (
        config_patch as config_loader,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="one-depth-mtp-checkpoint",
        ) as resolve_checkpoint,
    ):
        pretrained.load_pretrained_automodel_llm("one-depth-mtp-checkpoint", **loader_kwargs)

    resolve_checkpoint.assert_called_once_with("one-depth-mtp-checkpoint", {})
    config_loader.assert_called_once_with(
        "one-depth-mtp-checkpoint",
        trust_remote_code=False,
        local_files_only=True,
    )
    automodel.from_pretrained.assert_called_once_with(
        "one-depth-mtp-checkpoint",
        torch_dtype=torch.float32,
        trust_remote_code=False,
        num_nextn_predict_layers=4,
        mtp_use_repeated_layer=True,
    )
    automodel.from_config.assert_not_called()


@pytest.mark.parametrize(
    "mtp_config_overrides",
    [
        pytest.param(None, id="native-config-only"),
        pytest.param(_REPEATED_MTP_OVERRIDES, id="fallback-config-present"),
    ],
)
def test_load_pretrained_automodel_llm_rejects_multi_depth_native_head_as_repeated(mtp_config_overrides):
    config = SimpleNamespace(
        num_nextn_predict_layers=4,
        mtp_hybrid_override_pattern="*E",
        name_or_path="independent-mtp-checkpoint",
    )
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)
    loader_kwargs = {
        "num_nextn_predict_layers": 4,
        "mtp_use_repeated_layer": True,
    }
    if mtp_config_overrides is not None:
        loader_kwargs["mtp_config_overrides"] = mtp_config_overrides

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="independent-mtp-checkpoint",
        ),
        pytest.raises(ValueError, match="one physical MTP depth"),
    ):
        pretrained.load_pretrained_automodel_llm("independent-mtp-checkpoint", **loader_kwargs)

    automodel.from_pretrained.assert_not_called()
    automodel.from_config.assert_not_called()


def test_load_pretrained_automodel_llm_builds_repeated_head_for_checkpoint_without_mtp():
    config = SimpleNamespace(num_nextn_predict_layers=0, name_or_path="base-checkpoint")
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(pretrained, "_resolve_automodel_checkpoint_path", return_value="base-checkpoint"),
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        result = pretrained.load_pretrained_automodel_llm(
            "base-checkpoint",
            mtp_config_overrides=_REPEATED_MTP_OVERRIDES,
            num_nextn_predict_layers=4,
            mtp_use_repeated_layer=True,
        )

    assert config.num_nextn_predict_layers == 1
    automodel.from_config.assert_called_once_with(
        config,
        torch_dtype=torch.float32,
        load_base_model=False,
        trust_remote_code=False,
        num_nextn_predict_layers=4,
        mtp_use_repeated_layer=True,
    )
    base_load.assert_called_once_with(
        result,
        "base-checkpoint",
        {"num_nextn_predict_layers": 4, "mtp_use_repeated_layer": True},
    )
    automodel.from_pretrained.assert_not_called()


@pytest.mark.parametrize(
    ("checkpoint_depth", "mtp_config_overrides"),
    [
        pytest.param(0, _REPEATED_MTP_OVERRIDES, id="fresh-head"),
        pytest.param(1, None, id="native-head"),
    ],
)
def test_load_pretrained_automodel_llm_builds_repeated_model_without_checkpoint_weights(
    checkpoint_depth, mtp_config_overrides
):
    config = SimpleNamespace(num_nextn_predict_layers=checkpoint_depth, name_or_path="config-only-checkpoint")
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)
    loader_kwargs = {
        "num_nextn_predict_layers": 4,
        "mtp_use_repeated_layer": True,
    }
    if mtp_config_overrides is not None:
        loader_kwargs["mtp_config_overrides"] = mtp_config_overrides

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="config-only-checkpoint",
        ) as resolve_checkpoint,
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        result = pretrained.load_pretrained_automodel_llm(
            "config-only-checkpoint",
            pretrained_weights=False,
            **loader_kwargs,
        )

    assert result is automodel.from_config.return_value
    assert config.num_nextn_predict_layers == 1
    resolve_checkpoint.assert_called_once_with("config-only-checkpoint", {}, include_weights=False)
    automodel.from_config.assert_called_once_with(
        config,
        torch_dtype=torch.float32,
        load_base_model=False,
        trust_remote_code=False,
        num_nextn_predict_layers=4,
        mtp_use_repeated_layer=True,
    )
    automodel.from_pretrained.assert_not_called()
    base_load.assert_not_called()


def test_load_pretrained_automodel_llm_rejects_repeated_mode_without_head_definition():
    config = SimpleNamespace(num_nextn_predict_layers=0, name_or_path="base-checkpoint")
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(pretrained, "_resolve_automodel_checkpoint_path", return_value="base-checkpoint"),
        pytest.raises(ValueError, match="requires either a checkpoint with a native MTP head"),
    ):
        pretrained.load_pretrained_automodel_llm(
            "base-checkpoint",
            num_nextn_predict_layers=4,
            mtp_use_repeated_layer=True,
        )

    automodel.from_pretrained.assert_not_called()
    automodel.from_config.assert_not_called()


def test_load_pretrained_automodel_llm_rejects_replace_without_config_overrides():
    with pytest.raises(ValueError, match="requires mtp_config_overrides"):
        pretrained.load_pretrained_automodel_llm(
            "native-mtp-checkpoint",
            replace_mtp_config=True,
        )


def test_load_pretrained_automodel_llm_forwards_hf_resolution_kwargs():
    config = SimpleNamespace(num_nextn_predict_layers=0, name_or_path="private-checkpoint")
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch as config_loader,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="/cache/exact-snapshot/subdir",
            create=True,
        ) as resolve_checkpoint,
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        result = pretrained.load_pretrained_automodel_llm(
            "private-checkpoint",
            trust_remote_code=True,
            mtp_config_overrides={"num_nextn_predict_layers": 1, "mtp_hybrid_override_pattern": "*"},
            token="secret-token",
            revision="exact-revision",
            cache_dir="/cache",
            local_files_only=True,
            subfolder="subdir",
        )

    config_loader.assert_called_once_with(
        "/cache/exact-snapshot/subdir",
        trust_remote_code=True,
        local_files_only=True,
    )
    resolve_checkpoint.assert_called_once_with(
        "private-checkpoint",
        {
            "token": "secret-token",
            "revision": "exact-revision",
            "cache_dir": "/cache",
            "local_files_only": True,
            "subfolder": "subdir",
        },
    )
    automodel.from_config.assert_called_once_with(
        config,
        torch_dtype=torch.float32,
        load_base_model=False,
        trust_remote_code=True,
        cache_dir="/cache",
    )
    base_load.assert_called_once_with(result, "/cache/exact-snapshot/subdir", {"cache_dir": "/cache"})


@pytest.mark.parametrize(
    ("include_weights", "expected_extra_kwargs"),
    [(True, {}), (False, {"allow_patterns": ["*.json", "*.py"]})],
)
def test_resolve_automodel_checkpoint_path_uses_exact_snapshot(tmp_path, include_weights, expected_extra_kwargs):
    snapshot_root = tmp_path / "snapshot"
    expected_path = snapshot_root / "weights"
    expected_path.mkdir(parents=True)

    with patch("huggingface_hub.snapshot_download", return_value=str(snapshot_root)) as snapshot_download:
        result = pretrained._resolve_automodel_checkpoint_path(
            "private-checkpoint",
            {
                "cache_dir": str(tmp_path / "cache"),
                "local_files_only": True,
                "revision": "exact-revision",
                "subfolder": "weights",
                "token": "secret-token",
            },
            include_weights=include_weights,
        )

    assert result == str(expected_path)
    snapshot_download.assert_called_once_with(
        repo_id="private-checkpoint",
        cache_dir=str(tmp_path / "cache"),
        local_files_only=True,
        revision="exact-revision",
        token="secret-token",
        **expected_extra_kwargs,
    )


def test_automodel_mtp_depth_supports_non_nemotron_config_fields():
    assert (
        pretrained._automodel_config_mtp_depth(
            SimpleNamespace(num_nextn_predict_layers=2, mtp_hybrid_override_pattern=None, mtp_layers_block_type=None)
        )
        == 2
    )
    assert (
        pretrained._automodel_config_mtp_depth(SimpleNamespace(num_nextn_predict_layers=None, mtp_num_hidden_layers=2))
        == 2
    )


def test_automodel_base_load_skips_checkpoint_mtp_for_fresh_head(tmp_path):
    model = SimpleNamespace(config=SimpleNamespace(model_type="nemotron_h"), backbone=object())

    with (
        patch("nemo_automodel.components.checkpoint.checkpointing.Checkpointer") as checkpointer_cls,
        patch.object(torch.cuda, "is_available", return_value=False),
    ):
        pretrained._load_automodel_base_checkpoint_without_mtp(
            model,
            str(tmp_path),
            {},
        )

    checkpoint_config = checkpointer_cls.call_args.args[0]
    assert checkpoint_config.skip_task_head_prefixes_for_base_model == ["mtp."]
    checkpointer_cls.return_value.load_base_model.assert_called_once_with(
        model,
        torch.device("cpu"),
        None,
        str(tmp_path),
        load_base_model=True,
    )
    checkpointer_cls.return_value.load_model.assert_not_called()


@pytest.mark.parametrize("checkpoint_format", ["bin", "safetensors"])
def test_automodel_base_load_keeps_fresh_mtp_on_direct_fast_paths(tmp_path, checkpoint_format):
    class IdentityStateDictAdapter:
        def __init__(self):
            self.loaded_keys = None

        def from_hf(self, state_dict, **_kwargs):
            self.loaded_keys = set(state_dict)
            return state_dict

        def to_hf(self, state_dict, **_kwargs):
            return state_dict

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(2, 2, bias=False)
            self.mtp = torch.nn.Linear(2, 2, bias=False)
            self.config = SimpleNamespace(model_type="tiny_test", tie_word_embeddings=False)
            self.state_dict_adapter = IdentityStateDictAdapter()

    model = TinyModel()
    fresh_mtp = model.mtp.weight.detach().clone()
    checkpoint_state = {
        "base.weight": torch.full_like(model.base.weight, 3.0),
        "mtp.weight": torch.full_like(model.mtp.weight, 9.0),
    }
    if checkpoint_format == "bin":
        torch.save(checkpoint_state, tmp_path / "pytorch_model.bin")
    else:
        # Exercise Automodel's single-device custom-model safetensors branch.
        TinyModel.__module__ = "nemo_automodel.components.models.test"
        save_file(checkpoint_state, tmp_path / "model.safetensors")

    pretrained._load_automodel_base_checkpoint_without_mtp(model, str(tmp_path), {})

    torch.testing.assert_close(model.base.weight, torch.full_like(model.base.weight, 3.0))
    torch.testing.assert_close(model.mtp.weight, fresh_mtp)
    assert model.state_dict_adapter.loaded_keys == {"base.weight"}


def test_exclude_mtp_checkpoint_state_restores_hook_and_adapter_after_error():
    class IdentityStateDictAdapter:
        def from_hf(self, state_dict, **_kwargs):
            return state_dict

    def fail_checkpoint_load():
        raise RuntimeError("checkpoint load failed")

    model = torch.nn.Linear(2, 2)
    model.state_dict_adapter = IdentityStateDictAdapter()

    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        with pretrained._exclude_mtp_checkpoint_state(model):
            assert model._load_state_dict_pre_hooks
            assert "from_hf" in model.state_dict_adapter.__dict__
            fail_checkpoint_load()

    assert not model._load_state_dict_pre_hooks
    assert "from_hf" not in model.state_dict_adapter.__dict__
    assert model.state_dict_adapter.from_hf.__func__ is IdentityStateDictAdapter.from_hf
