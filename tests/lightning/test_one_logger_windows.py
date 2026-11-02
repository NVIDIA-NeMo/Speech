# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Throughput windows across checkpoint and accumulation boundaries."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nemo.collections.asr.one_logger import ASRThroughputPolicy
from nemo.lightning.callback_group import CallbackGroup
from nemo.lightning.one_logger_callback import OneLoggerNeMoCallback
from nemo.utils.exp_manager import exp_manager
from tests.lightning.test_one_logger_callback import _enabled_callback


@pytest.fixture(autouse=True)
def reset_callbacks(monkeypatch):
    monkeypatch.setattr(OneLoggerNeMoCallback, '_instance', None)
    monkeypatch.setattr(CallbackGroup, '_instance', None)


def test_checkpoint_windows_include_completed_batches(tmp_path, monkeypatch):
    callback, provider = _enabled_callback()
    group = CallbackGroup()
    monkeypatch.setattr(CallbackGroup, '_instance', group)
    with (
        patch(
            'nemo.lightning.one_logger_callback.Event.create', side_effect=lambda name, attrs: (name, attrs.to_json())
        ),
        patch('nemo.lightning.one_logger_callback._get_throughput_interval', return_value=1000),
    ):
        trainer = pl.Trainer(
            accelerator='cpu',
            devices=1,
            max_steps=4,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        exp_manager(
            trainer,
            {
                'explicit_log_dir': str(tmp_path / 'run'),
                'create_tensorboard_logger': False,
                'checkpoint_callback_params': {
                    'monitor': 'train_loss',
                    'save_top_k': -1,
                    'save_last': False,
                    'every_n_epochs': 0,
                    'every_n_train_steps': 2,
                    'save_nemo_on_train_end': False,
                },
            },
        )
        trainer.fit(
            _TrainingModel(),
            train_dataloaders=DataLoader(
                TensorDataset(torch.zeros(4, 160), torch.full((4,), 160), torch.zeros(4, 2), torch.full((4,), 2)),
                batch_size=1,
            ),
        )
    windows = _events(provider)
    assert len(windows) == 2
    assert [window['examples'] for window in windows] == [2, 2]
    assert all(window['window_optimizer_steps'] == 2 and window['mean_batch_size'] == 1 for window in windows)
    assert callback._throughput_sums == {} and callback._throughput_batches == 0
    assert trainer.callbacks.index(callback) < trainer.callbacks.index(trainer.checkpoint_callback)


def test_lifecycle_flush_during_batch_does_not_leave_untimed_measurements():
    callback, provider = _enabled_callback()
    trainer, model, batch = _loop(accumulation=1)
    with patch(
        'nemo.lightning.one_logger_callback.Event.create', side_effect=lambda name, attrs: (name, attrs.to_json())
    ):
        callback.on_train_start(trainer, model)
        _run_batch(callback, trainer, model, batch, 0, step=True)
        callback.on_train_batch_start(trainer, model, batch, 1)
        callback.on_before_optimizer_step(trainer, model, None)
        trainer.global_step += 1
        callback.on_save_checkpoint_start(trainer.global_step)
        callback.on_train_batch_end(trainer, model, None, batch, 1)
        callback.on_save_checkpoint_end()
        assert callback._throughput_batches == 0 and callback._throughput_sums == {}
        _run_batch(callback, trainer, model, batch, 2, step=True)
        callback._report_throughput(trainer=trainer)
    windows = _events(provider)
    assert len(windows) == 2
    assert all(window['examples'] == 1 and window['mean_batch_size'] == 1 for window in windows)


def test_validation_partial_start_omits_means_until_a_fully_aligned_window():
    callback, provider = _enabled_callback()
    trainer, model, batch = _loop(accumulation=2)
    with patch(
        'nemo.lightning.one_logger_callback.Event.create', side_effect=lambda name, attrs: (name, attrs.to_json())
    ):
        callback.on_train_start(trainer, model)
        for index in range(3):
            _run_batch(callback, trainer, model, batch, index, step=index == 1)
        callback.on_validation_start(trainer, model)
        callback.on_validation_end(trainer, model)
        _run_batch(callback, trainer, model, batch, 3, step=True)
        callback.on_validation_start(trainer, model)
        callback.on_validation_end(trainer, model)
        for index in (4, 5):
            _run_batch(callback, trainer, model, batch, index, step=index == 5)
        callback._report_throughput(trainer=trainer)
    first, partial, aligned = _events(provider)
    assert first['examples'] == 3 and 'mean_batch_size' not in first
    assert partial['examples'] == 1 and partial['target_text_tokens'] == 2
    assert 'mean_batch_size' not in partial and 'target_text_tokens_per_step' not in partial
    assert partial['target_text_tokens_per_second'] > 0
    assert aligned['mean_batch_size'] == 2 and aligned['target_text_tokens_per_step'] == 4


def test_empty_measurement_does_not_align_a_partial_start():
    callback, provider = _enabled_callback()
    trainer, model, batch = _loop(accumulation=2)
    model.one_logger_throughput_policy = _SkippingPolicy
    model.skip = True
    with patch(
        'nemo.lightning.one_logger_callback.Event.create', side_effect=lambda name, attrs: (name, attrs.to_json())
    ):
        callback.on_train_start(trainer, model)
        _run_batch(callback, trainer, model, batch, 0, step=False)
        model.skip = False
        _run_batch(callback, trainer, model, batch, 1, step=True)
        callback._report_throughput(trainer=trainer)
    window = _events(provider)[0]
    assert window['examples'] == 1 and 'mean_batch_size' not in window


def test_pending_cuda_window_tracks_skipped_accumulation_boundaries():
    callback, provider = _enabled_callback()
    trainer, model, batch = _loop(accumulation=2)
    start, end = MagicMock(), MagicMock()
    start.elapsed_time.return_value = 1000.0
    end.query.return_value = False
    with (
        patch(
            'nemo.lightning.one_logger_callback.Event.create', side_effect=lambda name, attrs: (name, attrs.to_json())
        ),
        patch('nemo.lightning.one_logger_callback.torch.cuda.device'),
        patch('nemo.lightning.one_logger_callback.torch.cuda.Event', return_value=end),
    ):
        callback.on_train_start(trainer, model)
        _run_batch(callback, trainer, model, batch, 0, step=False)
        _run_batch(callback, trainer, model, batch, 1, step=True)
        callback._throughput_cuda_start = start
        callback._throughput_cuda_device = torch.device('cuda', 0)
        callback._report_throughput(trainer=trainer)
        _run_batch(callback, trainer, model, batch, 2, step=False)
        assert callback._throughput_batches == 0
        end.query.return_value = True
        _run_batch(callback, trainer, model, batch, 3, step=True)
        callback._report_throughput(trainer=trainer)
    aligned, partial = _events(provider)
    assert aligned['mean_batch_size'] == 2
    assert partial['examples'] == 1 and 'mean_batch_size' not in partial
    assert 'target_text_tokens_per_step' not in partial
    end.synchronize.assert_not_called()


def test_telemetry_precedes_other_callbacks_without_reordering_them():
    callback, _ = _enabled_callback()
    group = CallbackGroup()
    first, second = pl.Callback(), pl.Callback()
    trainer = SimpleNamespace(callbacks=[first, second])
    group.attach_to_trainer(trainer)
    group.attach_to_trainer(trainer)
    assert trainer.callbacks == [callback, first, second]


class _TrainingModel(pl.LightningModule):
    one_logger_throughput_policy = ASRThroughputPolicy

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.preprocessor = SimpleNamespace(_sample_rate=16000)

    def training_step(self, batch, batch_idx):
        loss = self.weight.square()
        self.log('train_loss', loss, batch_size=1)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _SkippingPolicy(ASRThroughputPolicy):
    def measure(self, model, batch):
        return {} if model.skip else super().measure(model, batch)


def _loop(accumulation):
    trainer = SimpleNamespace(global_step=0, log_every_n_steps=1000, accumulate_grad_batches=accumulation)
    model = SimpleNamespace(
        one_logger_throughput_policy=ASRThroughputPolicy,
        preprocessor=SimpleNamespace(_sample_rate=16000),
        device=torch.device('cpu'),
    )
    batch = (torch.zeros(1, 160), torch.tensor([160]), torch.zeros(1, 2), torch.tensor([2]))
    return trainer, model, batch


def _run_batch(callback, trainer, model, batch, index, step):
    callback.on_train_batch_start(trainer, model, batch, index)
    if step:
        callback.on_before_optimizer_step(trainer, model, None)
        trainer.global_step += 1
    callback.on_train_batch_end(trainer, model, None, batch, index)


def _events(provider):
    return [
        entry.args[1][1]
        for entry in provider.recorder.event.call_args_list
        if entry.args[1][0] == 'nemo_speech.throughput'
    ]
