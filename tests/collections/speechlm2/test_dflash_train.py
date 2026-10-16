# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

pytest.importorskip("nemo_automodel")
pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "dflash_train_for_test", _REPO_ROOT / "examples/speechlm2/dflash_train.py"
)
_DFLASH_TRAIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DFLASH_TRAIN)


def test_dflash_config_inherits_salm_without_enabling_draft_training_for_salm():
    with initialize_config_dir(config_dir=str(_REPO_ROOT / "examples/speechlm2/conf"), version_base=None):
        base = compose(config_name="salm_automodel")
        draft = compose(config_name="dflash", overrides=["dflash.block_size=16"])

    assert "dflash" not in base
    assert "enabled" not in draft.dflash
    for section in ("model", "data", "trainer"):
        assert OmegaConf.to_container(draft[section]) == OmegaConf.to_container(base[section])
    assert draft.exp_manager.explicit_log_dir != base.exp_manager.explicit_log_dir
    assert draft.exp_manager.explicit_log_dir == "dflash_results/"
    assert draft.exp_manager.name == "dflash"
    assert draft.dflash.variant == "dflash2"
    assert draft.dflash.block_size == 16
    assert draft.dflash.label_source == "target_argmax"
    assert draft.model.use_nemo_automodel


def test_dflash_recipe_allows_reference_label_override():
    with initialize_config_dir(config_dir=str(_REPO_ROOT / "examples/speechlm2/conf"), version_base=None):
        draft = compose(config_name="dflash", overrides=["dflash.label_source=ground_truth"])

    assert draft.dflash.label_source == "ground_truth"


@pytest.mark.parametrize("validate_only", [False, True])
def test_dflash_entrypoint_wraps_target_and_routes_training(monkeypatch, tmp_path, validate_only):
    import nemo.collections.speechlm2
    from nemo.collections.speechlm2.parts import dflash

    target = SimpleNamespace(tokenizer=object())
    target_factory = Mock(return_value=target)
    draft_factory = Mock(return_value=object())
    trainer = Mock(callbacks=[])
    trainer.init_module.return_value = nullcontext()
    dataset_factory = Mock(return_value=object())
    datamodule_factory = Mock(return_value=object())
    monkeypatch.setattr(nemo.collections.speechlm2, "SALMAutomodel", target_factory)
    monkeypatch.setattr(dflash, "SALMDFlashModule", draft_factory)
    monkeypatch.setattr(_DFLASH_TRAIN, "Trainer", Mock(return_value=trainer))
    monkeypatch.setattr(_DFLASH_TRAIN, "SALMDataset", dataset_factory)
    monkeypatch.setattr(_DFLASH_TRAIN, "DataModule", datamodule_factory)
    monkeypatch.setattr(_DFLASH_TRAIN, "exp_manager", lambda *args: tmp_path)
    monkeypatch.setattr(_DFLASH_TRAIN, "resolve_trainer_cfg", lambda cfg: {})
    monkeypatch.setattr(_DFLASH_TRAIN.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_DFLASH_TRAIN.torch.distributed, "is_initialized", lambda: False)
    cfg = OmegaConf.create(
        {
            "model": {"use_nemo_automodel": True, "packed_sequences": True},
            "dflash": {"variant": "dflash2", "block_size": 16, "label_source": "target_argmax"},
            "data": {"train_ds": {"seed": 7, "batch_tokens": 1024}, "multispeaker_cfg": {"num_speakers": 2}},
            "trainer": {},
            "run_validate_only": validate_only,
        }
    )

    _DFLASH_TRAIN.train.__wrapped__(cfg)

    target_factory.assert_called_once_with(OmegaConf.to_container(cfg.model))
    draft_factory.assert_called_once_with(target, OmegaConf.to_container(cfg))
    dataset_factory.assert_called_once_with(
        tokenizer=target.tokenizer, multispeaker_cfg=cfg.data.multispeaker_cfg, batch_tokens=1024
    )
    datamodule_factory.assert_called_once_with(
        cfg.data, tokenizer=target.tokenizer, dataset=dataset_factory.return_value
    )
    selected, unused = (trainer.validate, trainer.fit) if validate_only else (trainer.fit, trainer.validate)
    selected.assert_called_once_with(draft_factory.return_value, datamodule_factory.return_value)
    unused.assert_not_called()
    assert isinstance(trainer.callbacks[0], _DFLASH_TRAIN.TrainingStatsCallback)
    assert OmegaConf.load(tmp_path / "exp_config.yaml") == cfg


def test_dflash_entrypoint_rejects_non_automodel_target():
    with pytest.raises(ValueError, match="use_nemo_automodel=true"):
        _DFLASH_TRAIN.train.__wrapped__(OmegaConf.create({"model": {"use_nemo_automodel": False}}))


def test_dflash_timeout_prefers_strategy_then_environment(monkeypatch):
    monkeypatch.setenv("TORCH_NCCL_TIMEOUT_S", "600")
    cfg = OmegaConf.create({"trainer": {"strategy": {"timeout_minutes": 60}}})
    assert _DFLASH_TRAIN._process_group_timeout(cfg) == timedelta(minutes=60)
    assert _DFLASH_TRAIN._process_group_timeout(OmegaConf.create({})) == timedelta(seconds=600)
