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
"""Regression tests for ``training_step(dataloader_iter)`` resumability."""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Type

import lightning.pytorch as pl
import pytest
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.exceptions import _TunerExitException
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo.core.utils.lightning_utils import read_batch
from nemo.utils.exp_manager import StatelessTimer, configure_no_restart_validation_training_loop


class _RangeDataset(torch.utils.data.Dataset):
    """Small deterministic dataset whose sample id is also its stream position."""

    def __init__(self, size: int = 1000) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x = torch.tensor([float(index % 7)], dtype=torch.float32)
        y = torch.tensor([float((index % 7) * 0.5)], dtype=torch.float32)
        return {
            "sample_id": torch.tensor(index, dtype=torch.long),
            "x": x,
            "y": y,
        }


class _BaseParityModel(pl.LightningModule):
    def __init__(self, seen: list[dict[str, int]], sleep_sec: float = 0.0) -> None:
        super().__init__()
        self.seen = seen
        self.sleep_sec = sleep_sec
        self.proj = torch.nn.Linear(1, 1)
        torch.nn.init.constant_(self.proj.weight, 0.25)
        torch.nn.init.constant_(self.proj.bias, 0.0)

    def train_dataloader(self):
        return StatefulDataLoader(_RangeDataset(), batch_size=1, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(_RangeDataset(size=2), batch_size=1, num_workers=0)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("val_loss", loss)
        return loss

    def _step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.sleep_sec:
            time.sleep(self.sleep_sec)
        self.seen.append(
            {
                "sample_id": int(batch["sample_id"].item()),
                "epoch": int(self.current_epoch),
                "batch_idx": int(batch_idx),
                "global_step": int(self.global_step),
            }
        )
        return self._loss(batch)

    def _loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return F.mse_loss(self.proj(batch["x"].float()), batch["y"].float())


class _BatchStepModel(_BaseParityModel):
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)


class _DataloaderIterStepModel(_BaseParityModel):
    def training_step(self, dataloader_iter):
        batch, batch_idx = read_batch(dataloader_iter, self)
        return self._step(batch, batch_idx)


class _StopBeforeNextBatchTimer(StatelessTimer):
    def _check_time_remaining(self, trainer: pl.Trainer) -> None:
        raise _TunerExitException()


class _CountingIterator:
    consumed = False

    def __next__(self):
        self.consumed = True
        raise AssertionError("read_batch consumed a sample after the timer requested stop")


class _PreemptedCallback:
    preemption_enabled = True
    interrupted = True


def _make_checkpoint_callback(root: Path) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=str(root / "checkpoints"),
        filename="{step}",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=1,
    )


def _make_trainer(root: Path, max_steps: int, callbacks: list | None = None) -> pl.Trainer:
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=str(root),
        callbacks=callbacks or [],
        max_steps=max_steps,
        max_epochs=10,
        limit_train_batches=5,
        val_check_interval=5,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=bool(callbacks),
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    configure_no_restart_validation_training_loop(trainer)
    return trainer


def _fit(
    root: Path,
    model_cls: Type[_BaseParityModel],
    max_steps: int,
    ckpt_path: str | None = None,
    callbacks: list | None = None,
    sleep_sec: float = 0.0,
) -> tuple[list[dict[str, int]], pl.Trainer, _BaseParityModel]:
    seen: list[dict[str, int]] = []
    model = model_cls(seen=seen, sleep_sec=sleep_sec)
    trainer = _make_trainer(root, max_steps=max_steps, callbacks=callbacks)
    trainer.fit(model, ckpt_path=ckpt_path)
    return seen, trainer, model


def _fit_two_phase(root: Path, model_cls: Type[_BaseParityModel]) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    first_callback = _make_checkpoint_callback(root / "first")
    first, _, _ = _fit(root / "first", model_cls, max_steps=5, callbacks=[first_callback])
    assert first_callback.last_model_path

    second_callback = _make_checkpoint_callback(root / "second")
    second, _, _ = _fit(
        root / "second",
        model_cls,
        max_steps=10,
        ckpt_path=first_callback.last_model_path,
        callbacks=[second_callback],
    )
    return first, second


def _project(records: list[dict[str, int]], key: str) -> list[int]:
    return [record[key] for record in records]


@pytest.mark.unit
def test_uninterrupted_dataloader_iter_matches_batch_step(tmp_path):
    batch_seen, _, _ = _fit(tmp_path / "batch", _BatchStepModel, max_steps=10)
    iter_seen, _, _ = _fit(tmp_path / "iter", _DataloaderIterStepModel, max_steps=10)

    expected = list(range(5)) + list(range(5))
    assert _project(iter_seen, "sample_id") == _project(batch_seen, "sample_id") == expected
    assert _project(iter_seen, "global_step") == _project(batch_seen, "global_step")
    assert _project(iter_seen, "epoch") == _project(batch_seen, "epoch")


@pytest.mark.unit
def test_interrupted_resume_dataloader_iter_matches_batch_step(tmp_path):
    batch_first, batch_second = _fit_two_phase(tmp_path / "batch", _BatchStepModel)
    iter_first, iter_second = _fit_two_phase(tmp_path / "iter", _DataloaderIterStepModel)

    assert _project(iter_first, "sample_id") == _project(batch_first, "sample_id") == list(range(5))
    assert _project(iter_second, "sample_id") == _project(batch_second, "sample_id") == list(range(5, 10))
    assert _project(iter_second, "global_step") == _project(batch_second, "global_step") == list(range(5, 10))
    assert _project(iter_second, "epoch") == _project(batch_second, "epoch")


@pytest.mark.unit
def test_resume_boundary_does_not_replay_old_epoch_batch(tmp_path):
    first_callback = _make_checkpoint_callback(tmp_path / "first")
    first, _, _ = _fit(tmp_path / "first", _DataloaderIterStepModel, max_steps=5, callbacks=[first_callback])
    assert _project(first, "sample_id") == list(range(5))

    second_callback = _make_checkpoint_callback(tmp_path / "second")
    second, _, _ = _fit(
        tmp_path / "second",
        _DataloaderIterStepModel,
        max_steps=10,
        ckpt_path=first_callback.last_model_path,
        callbacks=[second_callback],
    )

    assert _project(second, "sample_id") == list(range(5, 10))
    assert second[0] == {
        "sample_id": 5,
        "epoch": 1,
        "batch_idx": 0,
        "global_step": 5,
    }


@pytest.mark.unit
def test_read_batch_checks_timer_before_consuming_next_sample():
    iterator = _CountingIterator()
    model = SimpleNamespace(trainer=SimpleNamespace(callbacks=[_StopBeforeNextBatchTimer(timedelta(seconds=1))]))

    with pytest.raises(_TunerExitException):
        read_batch(iterator, model)

    assert not iterator.consumed


@pytest.mark.unit
def test_read_batch_checks_preemption_before_consuming_next_sample():
    iterator = _CountingIterator()
    trainer = SimpleNamespace(callbacks=[_PreemptedCallback()], checkpoint_callback=None)
    model = SimpleNamespace(trainer=trainer)

    with pytest.raises(_TunerExitException):
        read_batch(iterator, model)

    assert not iterator.consumed


@pytest.mark.unit
def test_read_batch_checks_lightning_sigterm_before_consuming_next_sample():
    iterator = _CountingIterator()
    trainer = SimpleNamespace(callbacks=[], checkpoint_callback=None, received_sigterm=True)
    model = SimpleNamespace(trainer=trainer)

    with pytest.raises(_TunerExitException):
        read_batch(iterator, model)

    assert not iterator.consumed


@pytest.mark.unit
def test_timer_checkpoint_resume_has_consistent_progress_and_no_sample_drift(tmp_path):
    checkpoint_callback = _make_checkpoint_callback(tmp_path / "timer")
    callbacks = [checkpoint_callback, StatelessTimer(duration=timedelta(seconds=0.15))]
    first, _, _ = _fit(
        tmp_path / "timer",
        _DataloaderIterStepModel,
        max_steps=50,
        callbacks=callbacks,
        sleep_sec=0.05,
    )

    assert 0 < len(first) < 50
    assert checkpoint_callback.last_model_path
    ckpt = torch.load(checkpoint_callback.last_model_path, map_location="cpu", weights_only=False)
    batch_progress = ckpt["loops"]["fit_loop"]["epoch_loop.batch_progress"]
    saved_step = int(ckpt["global_step"])

    assert saved_step == len(first)
    assert batch_progress["total"]["completed"] == saved_step
    assert batch_progress["total"]["ready"] == saved_step
    assert batch_progress["current"]["completed"] == batch_progress["current"]["ready"]

    resumed_callback = _make_checkpoint_callback(tmp_path / "timer-resume")
    resumed, _, _ = _fit(
        tmp_path / "timer-resume",
        _DataloaderIterStepModel,
        max_steps=saved_step + 3,
        ckpt_path=checkpoint_callback.last_model_path,
        callbacks=[resumed_callback],
    )

    assert _project(first, "sample_id") == [idx % 5 for idx in range(saved_step)]
    assert _project(resumed, "sample_id") == [(saved_step + idx) % 5 for idx in range(3)]
