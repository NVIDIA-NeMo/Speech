# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import os
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loops.progress import _BatchProgress
from lightning.pytorch.utilities.exceptions import _TunerExitException
from omegaconf import OmegaConf

from nemo.core import ModelPT
from nemo.utils import logging
from nemo.utils.exp_manager import (
    CallbackParams,
    ExpManagerConfig,
    StatelessTimer,
    _flush_in_flight_batch_progress,
    exp_manager,
)


class OnesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.ones(2)

    def __len__(self):
        return self.__dataset_len


class ExampleModel(ModelPT):
    def __init__(self, *args, **kwargs):
        cfg = OmegaConf.structured({})
        super().__init__(cfg, trainer=kwargs.get('trainer', None))
        # dummy parameter in order to allow DDP to execute
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)

    def train_dataloader(self):
        dataset = OnesDataset(10000)
        return torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4)

    def val_dataloader(self):
        dataset = OnesDataset(10)
        return torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4)

    def predict_dataloader(self):
        dataset = OnesDataset(10)
        return torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4)

    def forward(self, batch):
        return (self.l1(batch) - batch.mean(dim=1)).mean()

    def validation_step(self, batch, batch_idx):
        loss = (self.l1(batch) - batch.mean(dim=1)).mean()
        self.validation_step_outputs.append(loss)
        return loss

    def training_step(self, batch, batch_idx):
        return (self.l1(batch) - batch.mean(dim=1)).mean()

    def list_available_models(self):
        pass

    def setup_training_data(self):
        pass

    def setup_validation_data(self):
        pass

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        self.log("val_loss", torch.stack(self.validation_step_outputs).mean(), sync_dist=True)
        self.validation_step_outputs.clear()  # free memory


class TestStatelessTimer:
    def setup_model(self):
        # Stateless timer for 3 seconds.
        # Max steps shouldn't matter for it should stop in 3 seconds based on the timer.
        # Val check interval makes sure a checkpoint is written and can be restored from.
        callback_params = CallbackParams()
        callback_params.monitor = "val_loss"
        callback_params.save_top_k = 1
        trainer = Trainer(
            devices=1,
            val_check_interval=5,
            max_steps=10000,
            accelerator='gpu',
            strategy='ddp',
            logger=False,
            enable_checkpointing=False,
        )
        exp_manager_cfg = ExpManagerConfig(
            explicit_log_dir='./ptl_stateless_timer_check/',
            use_datetime_version=False,
            version="",
            resume_ignore_no_checkpoint=True,
            create_checkpoint_callback=True,
            checkpoint_callback_params=callback_params,
            resume_if_exists=True,
            max_time_per_run="00:00:00:03",
        )
        exp_manager(trainer, cfg=OmegaConf.structured(exp_manager_cfg))
        model = ExampleModel(trainer=trainer)
        trainer.fit(model)
        return trainer

    def cleanup(self):
        if os.path.exists('./ptl_stateless_timer_check'):
            shutil.rmtree('./ptl_stateless_timer_check', ignore_errors=True)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_stateless_timer(self):
        self.cleanup()
        trainer = self.setup_model()
        global_step_1 = trainer.global_step
        trainer = self.setup_model()
        global_step_2 = trainer.global_step
        trainer = self.setup_model()
        global_step_3 = trainer.global_step
        logging.info(f"Global steps : {global_step_1}, {global_step_2}, {global_step_3}")
        assert global_step_3 > global_step_2 > global_step_1
        self.cleanup()


class TestStatelessTimerSlurmStartTime:
    @pytest.mark.unit
    def test_counts_time_since_slurm_job_start(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "1234")
        monkeypatch.setenv("SLURM_JOB_START_TIME", "100")
        clock = {"wall": 1000.0, "monotonic": 50.0}
        monkeypatch.setattr("nemo.utils.exp_manager.time.time", lambda: clock["wall"])
        monkeypatch.setattr("nemo.utils.exp_manager.time.monotonic", lambda: clock["monotonic"])

        timer = StatelessTimer(duration="00:00:20:00", max_time_from_slurm=True)
        timer.on_train_start(None, None)
        clock["monotonic"] = 55.0

        assert timer.time_elapsed() == pytest.approx(905.0)

    @pytest.mark.unit
    def test_refreshes_slurm_elapsed_time_before_fit(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_START_TIME", "100")
        clock = {"wall": 1000.0, "monotonic": 50.0}
        monkeypatch.setattr("nemo.utils.exp_manager.time.time", lambda: clock["wall"])
        monkeypatch.setattr("nemo.utils.exp_manager.time.monotonic", lambda: clock["monotonic"])
        timer = StatelessTimer(duration="00:00:20:00", max_time_from_slurm=True)
        trainer = MagicMock()
        trainer.should_stop = False
        trainer.strategy.broadcast.side_effect = lambda value: value

        clock["wall"] = 1100.0
        timer.on_fit_start(trainer)

        assert timer.time_elapsed() == pytest.approx(1000.0)

    @pytest.mark.unit
    def test_expired_slurm_budget_saves_before_training(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_START_TIME", "100")
        monkeypatch.setattr("nemo.utils.exp_manager.time.time", lambda: 1000.0)
        checkpoint_callback = MagicMock()
        checkpoint_callback._monitor_candidates.return_value = {}
        trainer = SimpleNamespace(
            strategy=SimpleNamespace(broadcast=lambda value: value),
            should_stop=False,
            checkpoint_callback=checkpoint_callback,
            global_step=0,
            current_epoch=0,
        )
        timer = StatelessTimer(duration="00:00:10:00", max_time_from_slurm=True)

        with pytest.raises(_TunerExitException):
            timer.on_fit_start(trainer)

        checkpoint_callback._save_last_checkpoint.assert_called_once_with(trainer, {})

    @pytest.mark.unit
    def test_slurm_timing_is_opt_in(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_START_TIME", "not-a-timestamp")
        clock = {"monotonic": 50.0}
        monkeypatch.setattr("nemo.utils.exp_manager.time.monotonic", lambda: clock["monotonic"])

        timer = StatelessTimer(duration="00:00:20:00")
        timer.on_train_start(None, None)
        clock["monotonic"] = 55.0

        assert timer.time_elapsed() == pytest.approx(5.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("value", [None, "not-a-timestamp", "0", "-1"])
    def test_rejects_missing_or_invalid_slurm_start_time(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("SLURM_JOB_START_TIME", raising=False)
        else:
            monkeypatch.setenv("SLURM_JOB_START_TIME", value)

        with pytest.raises(ValueError, match="SLURM_JOB_START_TIME"):
            StatelessTimer(duration="00:00:20:00", max_time_from_slurm=True)

    @pytest.mark.unit
    def test_exp_manager_enables_slurm_timing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLURM_JOB_START_TIME", "100")
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
        cfg = ExpManagerConfig(
            explicit_log_dir=str(tmp_path),
            create_tensorboard_logger=False,
            create_checkpoint_callback=False,
            log_step_timing=False,
            disable_validation_on_resume=False,
            max_time_per_run="00:00:20:00",
            max_time_per_run_from_slurm=True,
        )

        exp_manager(trainer, cfg=OmegaConf.structured(cfg))

        timer = next(callback for callback in trainer.callbacks if isinstance(callback, StatelessTimer))
        assert timer._slurm_job_start_time == 100.0


def _make_trainer_with_batch_progress(batch_progress: _BatchProgress) -> MagicMock:
    trainer = MagicMock()
    trainer.fit_loop.epoch_loop.batch_progress = batch_progress
    return trainer


class TestFlushInFlightBatchProgress:
    """Regression tests for the mid-batch save off-by-one on wall-time/preemption resume.

    PTL's TrainingEpochLoop.advance() calls on_train_batch_end hooks BEFORE
    batch_progress.current.increment_completed(), but the batch's optim step has already
    advanced global_step (optim_progress.step.total.completed). If a timer/preemption hook
    saves in that window, the checkpoint captures batch_progress lagging one behind
    optim_progress. On resume, PTL's reset_on_restart rewinds batch_progress, PTL replays
    the in-flight batch, and its optim step runs a second time — leaking one extra
    global_step per resume. _flush_in_flight_batch_progress closes that gap.
    """

    @pytest.mark.unit
    def test_flushes_in_flight_batch(self):
        batch_progress = _BatchProgress()
        # Simulate the mid-batch state: batch K just had its optim step run, but
        # increment_completed() has not yet fired (it fires AFTER on_train_batch_end
        # hooks in PTL's TrainingEpochLoop.advance).
        batch_progress.total.ready = 500
        batch_progress.total.processed = 500
        batch_progress.total.completed = 499
        batch_progress.current.ready = 500
        batch_progress.current.processed = 500
        batch_progress.current.completed = 499

        trainer = _make_trainer_with_batch_progress(batch_progress)
        _flush_in_flight_batch_progress(trainer)

        assert batch_progress.current.completed == 500
        assert batch_progress.total.completed == 500
        # Ready/processed unchanged — we're only closing the completed gap.
        assert batch_progress.current.ready == 500
        assert batch_progress.total.ready == 500

    @pytest.mark.unit
    def test_noop_when_state_already_consistent(self):
        batch_progress = _BatchProgress()
        batch_progress.total.ready = 500
        batch_progress.total.processed = 500
        batch_progress.total.completed = 500
        batch_progress.current.ready = 500
        batch_progress.current.processed = 500
        batch_progress.current.completed = 500

        trainer = _make_trainer_with_batch_progress(batch_progress)
        _flush_in_flight_batch_progress(trainer)

        # Nothing over-incremented.
        assert batch_progress.current.completed == 500
        assert batch_progress.total.completed == 500

    @pytest.mark.unit
    def test_tolerates_missing_fit_loop(self):
        # on_fit_start may fire before fit_loop attributes are materialized; helper must
        # not explode in that case.
        trainer = SimpleNamespace()
        _flush_in_flight_batch_progress(trainer)  # must not raise


class TestStatelessTimerResumeStateConsistency:
    """Verify StatelessTimer._check_time_remaining flushes batch_progress before saving.

    Without this flush, the saved -last.ckpt drifts global_step by +1 per resume (see
    _flush_in_flight_batch_progress docstring). This test drives _check_time_remaining
    directly with a mid-batch batch_progress and a mock checkpoint_callback, and asserts
    that the state handed to _save_last_checkpoint is already self-consistent.
    """

    @pytest.mark.unit
    def test_check_time_remaining_flushes_before_save(self, monkeypatch):
        # Simulate a mid-batch state: batch 500 in flight, optim step ran (global_step
        # effectively at 8500 on the optim side), but batch_progress.current.completed
        # is still 499.
        batch_progress = _BatchProgress()
        batch_progress.total.ready = 8500
        batch_progress.total.processed = 8500
        batch_progress.total.completed = 8499
        batch_progress.current.ready = 500
        batch_progress.current.processed = 500
        batch_progress.current.completed = 499

        seen = {}

        def _capture_save_last(trainer, monitor_candidates):
            bp = trainer.fit_loop.epoch_loop.batch_progress
            seen['current_completed'] = bp.current.completed
            seen['current_ready'] = bp.current.ready
            seen['total_completed'] = bp.total.completed

        checkpoint_callback = MagicMock()
        checkpoint_callback._monitor_candidates.return_value = {}
        checkpoint_callback._save_last_checkpoint.side_effect = _capture_save_last

        trainer = MagicMock()
        trainer.should_stop = True  # pretend the wall-time budget is exhausted
        trainer.fit_loop.epoch_loop.batch_progress = batch_progress
        trainer.checkpoint_callback = checkpoint_callback

        timer = StatelessTimer(duration="00:00:00:01")
        # Bypass the real Timer._check_time_remaining (which would re-set should_stop
        # via a broadcast and rely on a started clock); we only want to verify the
        # branch that runs when should_stop is already True.
        monkeypatch.setattr(
            "lightning.pytorch.callbacks.timer.Timer._check_time_remaining",
            lambda self, trainer: None,
        )

        with pytest.raises(_TunerExitException):
            timer._check_time_remaining(trainer)

        # The checkpoint save saw a self-consistent state — no +1 drift on resume.
        assert seen['current_completed'] == 500
        assert seen['current_ready'] == 500
        assert seen['total_completed'] == 8500
