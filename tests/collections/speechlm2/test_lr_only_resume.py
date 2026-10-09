# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate historical LR-only continuation without discarding optimizer state."""
import copy
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nemo.collections.speechlm2.parts.lr_only_resume import ScaleRestoredLearningRate
from nemo.core.classes.common import _is_target_allowed, safe_instantiate
from nemo.core.optim.lr_scheduler import CosineAnnealing


def _restored_trainer():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    scheduler = CosineAnnealing(optimizer, max_steps=100, min_lr=1e-5)
    for _ in range(50):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    trainer = SimpleNamespace(
        global_step=50,
        ckpt_path='/checkpoints/step=50.ckpt',
        optimizers=[optimizer],
        lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
        is_global_zero=False,
    )
    return trainer, optimizer, scheduler


def test_scale_restored_lr_preserves_moments_and_schedule_position_once():
    trainer, optimizer, scheduler = _restored_trainer()
    before_state = copy.deepcopy(optimizer.state_dict()['state'])
    old_lr = optimizer.param_groups[0]['lr']
    old_last_lr = scheduler._last_lr[:]
    old_epoch = scheduler.last_epoch
    callback = ScaleRestoredLearningRate(0.75, 50, 1e-3, 1e-5)
    callback.on_train_start(trainer, None)

    torch.testing.assert_close(optimizer.state_dict()['state'], before_state)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(old_lr * 0.75)
    assert optimizer.param_groups[0]['initial_lr'] == pytest.approx(0.00075)
    assert scheduler.base_lrs == pytest.approx([0.00075])
    assert scheduler.min_lr == pytest.approx(0.0000075)
    assert scheduler._last_lr == pytest.approx([lr * 0.75 for lr in old_last_lr])
    assert scheduler.last_epoch == old_epoch

    restored = ScaleRestoredLearningRate(0.75, 50, 1e-3, 1e-5)
    restored.load_state_dict(callback.state_dict())
    trainer.global_step = 51
    restored.on_train_start(trainer, None)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(old_lr * 0.75)
    assert restored.state_key == callback.state_key


@pytest.mark.parametrize('wrong', ['step', 'checkpoint', 'base_lr', 'minimum_lr'])
def test_scale_restored_lr_rejects_wrong_source_before_mutating(wrong):
    trainer, optimizer, scheduler = _restored_trainer()
    if wrong == 'step':
        trainer.global_step = 49
    elif wrong == 'checkpoint':
        trainer.ckpt_path = '/checkpoints/step=50-last.ckpt'
    elif wrong == 'base_lr':
        scheduler.base_lrs = [2e-3]
    else:
        scheduler.min_lr = 2e-5
    old_lr = optimizer.param_groups[0]['lr']
    callback = ScaleRestoredLearningRate(0.75, 50, 1e-3, 1e-5)
    with pytest.raises(RuntimeError):
        callback.on_train_start(trainer, None)
    assert optimizer.param_groups[0]['lr'] == old_lr
    assert callback.state_dict() == {'applied': False}


def test_historical_callback_is_safely_instantiable():
    target = 'nemo.collections.speechlm2.parts.lr_only_resume.ScaleRestoredLearningRate'
    assert _is_target_allowed(target)
    callback = safe_instantiate(
        DictConfig(
            {
                '_target_': target,
                'factor': 0.75,
                'source_step': 50,
                'original_base_lr': 1e-3,
                'original_min_lr': 1e-5,
            }
        )
    )
    assert isinstance(callback, ScaleRestoredLearningRate)


@pytest.mark.unit
def test_lr_only_full_checkpoint_resume_preserves_state_and_future_schedule(tmp_path):
    source_step, intermediate_step, final_step = 4, 7, 10
    factor = 0.75
    source, _, _ = _fit_tiny_continuation(tmp_path, 'source', source_step)
    source_path = tmp_path / f'step={source_step}.ckpt'
    source.save_checkpoint(source_path)
    source_state = torch.load(source_path, map_location='cpu', weights_only=False)
    assert source_state['global_step'] == source_step
    assert source_state['lr_schedulers'][0]['last_epoch'] == source_step

    first_callback = ScaleRestoredLearningRate(factor, source_step, 1e-3, 1e-5)
    continued, _, first_probe = _fit_tiny_continuation(
        tmp_path,
        'first-resume',
        intermediate_step,
        checkpoint=source_path,
        callback=first_callback,
    )
    assert continued.optimizers[0] is not source.optimizers[0]
    assert continued.lr_scheduler_configs[0].scheduler is not source.lr_scheduler_configs[0].scheduler
    _assert_exact_restored_state(first_probe, source_state)
    assert first_callback.applied
    # Before the first update, only LR metadata changes; optimizer moments are exact.
    torch.testing.assert_close(
        first_probe.batches[0]['optimizer']['state'], source_state['optimizer_states'][0]['state'], rtol=0, atol=0
    )
    assert first_probe.batches[0]['lr'] == pytest.approx(
        source_state['optimizer_states'][0]['param_groups'][0]['lr'] * factor
    )

    continued_path = tmp_path / f'step={intermediate_step}.ckpt'
    continued.save_checkpoint(continued_path)
    continued_state = torch.load(continued_path, map_location='cpu', weights_only=False)
    assert continued_state['callbacks'][first_callback.state_key] == {'applied': True}
    assert continued_state['lr_schedulers'][0]['last_epoch'] == intermediate_step

    second_callback = ScaleRestoredLearningRate(factor, source_step, 1e-3, 1e-5)
    resumed, resumed_model, second_probe = _fit_tiny_continuation(
        tmp_path,
        'second-resume',
        final_step,
        checkpoint=continued_path,
        callback=second_callback,
    )
    assert second_callback is not first_callback
    assert resumed.optimizers[0] is not continued.optimizers[0]
    assert resumed.lr_scheduler_configs[0].scheduler is not continued.lr_scheduler_configs[0].scheduler
    _assert_exact_restored_state(second_probe, continued_state)
    assert second_callback.applied
    assert second_probe.batches[0]['lr'] == continued_state['optimizer_states'][0]['param_groups'][0]['lr']
    torch.testing.assert_close(
        second_probe.batches[0]['optimizer'], continued_state['optimizer_states'][0], rtol=0, atol=0
    )
    assert second_probe.batches[0]['scheduler'] == continued_state['lr_schedulers'][0]

    # A continuously trained factor-scaled baseline supplies the future LR oracle.
    # Constant gradients ensure optimizer moments are independent of LR/model value.
    baseline, _, baseline_probe = _fit_tiny_continuation(tmp_path, 'scaled-baseline', final_step, initial_scale=factor)
    combined = first_probe.batches + second_probe.batches
    expected = baseline_probe.batches[source_step:final_step]
    assert [batch['step'] for batch in combined] == list(range(source_step, final_step))
    assert [batch['lr'] for batch in combined] == pytest.approx([batch['lr'] for batch in expected])
    assert [batch['scheduler']['last_epoch'] for batch in combined] == [
        batch['scheduler']['last_epoch'] for batch in expected
    ]
    torch.testing.assert_close(
        resumed.optimizers[0].state_dict()['state'], baseline.optimizers[0].state_dict()['state']
    )
    assert (
        resumed.lr_scheduler_configs[0].scheduler.state_dict()
        == baseline.lr_scheduler_configs[0].scheduler.state_dict()
    )
    assert resumed.global_step == final_step

    # An uninterrupted continuation from the original source reaches the same final model.
    reference_callback = ScaleRestoredLearningRate(factor, source_step, 1e-3, 1e-5)
    _, reference_model, _ = _fit_tiny_continuation(
        tmp_path,
        'continuation-reference',
        final_step,
        checkpoint=source_path,
        callback=reference_callback,
    )
    torch.testing.assert_close(resumed_model.state_dict(), reference_model.state_dict(), rtol=0, atol=0)


class _TinyContinuationModel(LightningModule):
    def __init__(self, initial_scale=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.initial_scale = initial_scale

    def training_step(self, batch, batch_idx):
        return (self.weight * batch).sum()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([self.weight], lr=1e-3 * self.initial_scale)
        scheduler = CosineAnnealing(optimizer, max_steps=16, warmup_steps=2, min_lr=1e-5 * self.initial_scale)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


class _ContinuationProbe(Callback):
    def __init__(self):
        self.restored = None
        self.batches = []

    def _snapshot(self, trainer, pl_module):
        return {
            'step': trainer.global_step,
            'lr': trainer.optimizers[0].param_groups[0]['lr'],
            'model': copy.deepcopy(pl_module.state_dict()),
            'optimizer': copy.deepcopy(trainer.optimizers[0].state_dict()),
            'scheduler': copy.deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict()),
        }

    def on_train_start(self, trainer, pl_module):
        # This probe precedes the scaling callback and sees the full restored state.
        self.restored = self._snapshot(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batches.append(self._snapshot(trainer, pl_module))


def _fit_tiny_continuation(tmp_path, name, max_steps, checkpoint=None, callback=None, initial_scale=1.0):
    probe = _ContinuationProbe()
    trainer = Trainer(
        accelerator='cpu',
        devices=1,
        max_steps=max_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path / name,
        callbacks=[probe] + ([callback] if callback is not None else []),
    )
    model = _TinyContinuationModel(initial_scale=initial_scale)
    trainer.fit(model, train_dataloaders=DataLoader([torch.ones(2)] * 20, batch_size=1), ckpt_path=checkpoint)
    return trainer, model, probe


def _assert_exact_restored_state(probe, checkpoint):
    assert probe.restored['step'] == checkpoint['global_step']
    torch.testing.assert_close(probe.restored['model'], checkpoint['state_dict'], rtol=0, atol=0)
    torch.testing.assert_close(probe.restored['optimizer'], checkpoint['optimizer_states'][0], rtol=0, atol=0)
    assert probe.restored['scheduler'] == checkpoint['lr_schedulers'][0]


@pytest.fixture(autouse=True)
def _cpu_default_device():
    """Keep CPU checkpoint and Adam allocations independent of other tests' defaults.

    Some SpeechLM test modules select CUDA at import time. Adam's step counter
    otherwise inherits that default even though Lightning uses a CPU trainer.
    The context restores the enclosing default device after each test.
    """
    with torch.device('cpu'):
        yield
