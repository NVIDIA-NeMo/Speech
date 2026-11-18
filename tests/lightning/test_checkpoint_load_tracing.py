# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading through Lightning strategies emits paired lifecycle spans."""

import inspect
import pickle
from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nemo.lightning.callback_group import CallbackGroup
from nemo.lightning.one_logger_callback import OneLoggerNeMoCallback
from nemo.utils.exp_manager import exp_manager
from tests.lightning.test_one_logger_callback import _enabled_callback
from tests.lightning.test_one_logger_windows import _TrainingModel


class _CheckpointStrategy:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def load_checkpoint(self, path, *, weights_only=None):
        self.calls.append((path, weights_only))
        if self.fail:
            raise ValueError('checkpoint load failed')
        return {'path': path}


@pytest.fixture(autouse=True)
def reset_callbacks(monkeypatch):
    monkeypatch.setattr(OneLoggerNeMoCallback, '_instance', None)
    monkeypatch.setattr(CallbackGroup, '_instance', None)


def _group(monkeypatch):
    callback, provider = _enabled_callback()
    group = CallbackGroup()
    monkeypatch.setattr(CallbackGroup, '_instance', group)
    return group, provider


def _assert_load_span(provider):
    starts = [call for call in provider.recorder.start.call_args_list if call.args[0] == 'nemo_speech.checkpoint_load']
    assert len(starts) == 1
    stopped = [call.args[0] for call in provider.recorder.stop.call_args_list]
    assert sum(span.name == 'nemo_speech.checkpoint_load' for span in stopped) == 1


@pytest.mark.parametrize('weights_only', [None, False, True])
def test_strategy_load_is_paired_and_preserves_arguments(monkeypatch, weights_only):
    group, provider = _group(monkeypatch)
    strategy = _CheckpointStrategy()
    trainer = SimpleNamespace(callbacks=[], strategy=strategy)
    signature = inspect.signature(strategy.load_checkpoint)
    group.attach_to_trainer(trainer)
    group.attach_to_trainer(trainer)
    assert inspect.signature(strategy.load_checkpoint) == signature

    result = strategy.load_checkpoint('checkpoint.ckpt', weights_only=weights_only)

    assert result == {'path': 'checkpoint.ckpt'}
    assert strategy.calls == [('checkpoint.ckpt', weights_only)]
    _assert_load_span(provider)


def test_strategy_load_span_closes_without_swallowing_failure(monkeypatch):
    group, provider = _group(monkeypatch)
    trainer = SimpleNamespace(callbacks=[], strategy=_CheckpointStrategy(fail=True))
    group.attach_to_trainer(trainer)

    with pytest.raises(ValueError, match='checkpoint load failed'):
        trainer.strategy.load_checkpoint('broken.ckpt')

    _assert_load_span(provider)


def test_traced_strategy_survives_spawn_serialization(monkeypatch):
    group, provider = _group(monkeypatch)
    trainer = SimpleNamespace(callbacks=[], strategy=_CheckpointStrategy())
    group.attach_to_trainer(trainer)

    restored = pickle.loads(pickle.dumps(trainer.strategy))

    assert restored.load_checkpoint('checkpoint.ckpt') == {'path': 'checkpoint.ckpt'}
    assert restored.calls == [('checkpoint.ckpt', None)]
    _assert_load_span(provider)


def test_disabled_integration_leaves_strategy_method_unchanged(monkeypatch):
    monkeypatch.setenv('NEMO_ONE_LOGGER_ENABLED', 'false')
    group = CallbackGroup.get_instance()
    strategy = _CheckpointStrategy()

    group.attach_to_trainer(SimpleNamespace(callbacks=[], strategy=strategy))

    assert strategy.load_checkpoint.__func__ is _CheckpointStrategy.load_checkpoint


def test_real_trainer_resume_emits_one_load_span(tmp_path, monkeypatch):
    group, provider = _group(monkeypatch)
    data = DataLoader(
        TensorDataset(torch.zeros(4, 160), torch.full((4,), 160), torch.zeros(4, 2), torch.full((4,), 2)),
        batch_size=1,
    )

    def trainer(name, steps):
        trainer = pl.Trainer(
            accelerator='cpu',
            devices=1,
            max_steps=steps,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        exp_manager(
            trainer,
            {
                'explicit_log_dir': str(tmp_path / name),
                'create_tensorboard_logger': False,
                'create_checkpoint_callback': False,
            },
        )
        return trainer

    original = trainer('original', 2)
    original.fit(_TrainingModel(), train_dataloaders=data)
    checkpoint = tmp_path / 'checkpoint.ckpt'
    original.save_checkpoint(checkpoint)
    provider.recorder.reset_mock()

    resumed = trainer('resumed', 4)
    resumed.fit(_TrainingModel(), train_dataloaders=data, ckpt_path=checkpoint)

    assert resumed.global_step == 4
    _assert_load_span(provider)
