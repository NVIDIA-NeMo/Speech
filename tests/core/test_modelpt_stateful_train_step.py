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
"""ModelPT's no-prefetch ``training_step(dataloader_iter)`` adapter for stateful train dataloaders.

The tiny model trains on an unknown-length (iterable) dataset of integer sample ids, so Lightning's default fetcher
would read one batch ahead. Training steps are assigned as instance attributes, the way ``ModelPT`` itself wraps
``training_step``.
"""

import inspect
from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo.core.classes.modelPT import ModelPT, _has_stateful_dataloader

NUM_SAMPLES = 20
BATCH_SIZE = 2
SPLIT = 3
TOTAL = 6


class _Ids(torch.utils.data.IterableDataset):
    def __init__(self, n: int = NUM_SAMPLES):
        self.n = n

    def __iter__(self):
        return iter(range(self.n))


def _stateful(n: int = NUM_SAMPLES, batch_size=BATCH_SIZE) -> StatefulDataLoader:
    return StatefulDataLoader(_Ids(n), batch_size=batch_size)


def _plain(n: int = NUM_SAMPLES) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(_Ids(n), batch_size=BATCH_SIZE)


class _TinyModel(ModelPT):
    """Learns one scalar; ``calls`` records ``(global_step, sample ids)`` for each ``training_step`` call."""

    def __init__(self, train_dl=None, step: str = 'batch_idx', optimizer: str = 'sgd'):
        super().__init__(OmegaConf.create({}))
        self.w = torch.nn.Parameter(torch.tensor([3.0]))
        self._train_dl = train_dl
        self.optimizer = optimizer
        self.calls = []
        self.batch_idxs = []
        self.training_step = {
            'batch_idx': self._step_with_idx,
            'batch_nb': self._step_with_nb,
            'batch_only': self._step_batch_only,
            'manual': self._step_manual,
        }[step]
        if step == 'manual':
            self.automatic_optimization = False

    def _loss(self, batch):
        ids = _ids(batch)
        self.calls.append((self.trainer.global_step, ids))
        loss = ((self.w - torch.tensor(ids, dtype=torch.float).mean()) ** 2).sum()
        self.log('loss', loss, on_step=True, on_epoch=True)
        return loss

    def _step_with_idx(self, batch, batch_idx):
        self.batch_idxs.append(batch_idx)
        return self._loss(batch)

    def _step_with_nb(self, batch, batch_nb):
        self.batch_idxs.append(batch_nb)
        return self._loss(batch)

    def _step_batch_only(self, batch):
        return self._loss(batch)

    def _step_manual(self, batch, batch_idx):
        for opt in self.optimizers():
            opt.zero_grad()
            self.manual_backward(self._loss(batch))
            opt.step()

    def configure_optimizers(self):
        if self.optimizer == 'lbfgs':
            return torch.optim.LBFGS(self.parameters(), lr=0.1, max_iter=3)
        if self.optimizer == 'two':
            return [torch.optim.SGD(self.parameters(), lr=0.1), torch.optim.SGD(self.parameters(), lr=0.1)]
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def list_available_models(self):
        return []

    def setup_training_data(self, train_data_config):
        pass

    def setup_validation_data(self, val_data_config):
        pass


@pytest.mark.unit
def test_resume_continues_at_next_batch(tmp_path):
    full, resumed = _fit_and_resume(tmp_path, lambda: _TinyModel(_stateful()))
    assert resumed == full[SPLIT:]


@pytest.mark.unit
@pytest.mark.parametrize("step", ["batch_only", "batch_nb"])
def test_batch_hooks_without_batch_idx_name(tmp_path, step):
    full, resumed = _fit_and_resume(tmp_path, lambda: _TinyModel(_stateful(), step=step))
    assert resumed == full[SPLIT:]
    if step == "batch_nb":
        model = _fit(_TinyModel(_stateful(), step=step), tmp_path / "idx", max_steps=3)
        assert model.batch_idxs == [0, 1, 2]


@pytest.mark.unit
def test_lbfgs_closure_evaluations_reuse_the_batch():
    model = _fit(_TinyModel(_stateful(), optimizer='lbfgs'), None, max_steps=3)
    by_step = {}
    for step, ids in model.calls:
        by_step.setdefault(step, []).append(ids)
    assert any(len(calls) > 1 for calls in by_step.values())  # LBFGS evaluated the closure several times
    assert {step: set(map(tuple, calls)) for step, calls in by_step.items()} == {
        0: {(0, 1)},
        1: {(2, 3)},
        2: {(4, 5)},
    }


@pytest.mark.unit
def test_gradient_accumulation_reads_one_batch_per_micro_batch():
    model = _fit(_TinyModel(_stateful()), None, max_steps=2, accumulate_grad_batches=2)
    assert model.calls == [(0, [0, 1]), (0, [2, 3]), (1, [4, 5]), (1, [6, 7])]


@pytest.mark.unit
def test_manual_optimization_with_two_updates_reads_one_batch(tmp_path):
    model = _fit(_TinyModel(_stateful(), step='manual', optimizer='two'), None, max_steps=4)
    assert [ids for _, ids in model.calls] == [[0, 1], [0, 1], [2, 3], [2, 3]]


@pytest.mark.unit
def test_follows_explicit_non_stateful_train_dataloaders():
    model = _TinyModel(_stateful())  # the model's own loader is stateful, but this fit uses another one
    _fit(model, None, max_steps=2, train_dataloaders=_plain())
    assert list(inspect.signature(model.training_step).parameters) == ["batch", "batch_idx"]


@pytest.mark.unit
@pytest.mark.parametrize("source", ["train_dataloaders", "datamodule", "combined_loader"])
def test_follows_external_stateful_train_dataloaders(tmp_path, source):
    def fit_kwargs():
        if source == "train_dataloaders":
            return dict(train_dataloaders=_stateful())
        if source == "combined_loader":
            return dict(train_dataloaders=CombinedLoader([_stateful()], mode="max_size_cycle"))
        return dict(datamodule=_DataModule())

    full, resumed = _fit_and_resume(tmp_path, lambda: _TinyModel(), fit_kwargs=fit_kwargs)
    assert resumed == full[SPLIT:]


@pytest.mark.unit
def test_adapter_removed_when_a_later_fit_is_not_stateful():
    model = _fit(_TinyModel(_stateful()), None, max_steps=1)
    assert list(inspect.signature(model.training_step).parameters) == ["dataloader_iter"]
    _fit(model, None, max_steps=1, train_dataloaders=_plain())
    assert list(inspect.signature(model.training_step).parameters) == ["batch", "batch_idx"]


@pytest.mark.unit
def test_has_stateful_dataloader():
    stateful, plain = _stateful(), _plain()
    for loaders in (stateful, [stateful], {"a": stateful}, CombinedLoader([stateful], mode="max_size_cycle")):
        assert _has_stateful_dataloader(loaders)
    for loaders in (None, plain, [plain], {"a": plain}, CombinedLoader([plain], mode="max_size_cycle")):
        assert not _has_stateful_dataloader(loaders)


@pytest.mark.unit
def test_log_keeps_explicit_positional_batch_size():
    model = _TinyModel(_stateful())
    names = list(inspect.signature(pl.LightningModule.log).parameters)[1:]
    defaults = {k: p.default for k, p in inspect.signature(pl.LightningModule.log).parameters.items()}
    positional = names[: names.index('batch_size') + 1]
    seen = []

    def step(batch, batch_idx):
        loss = model._loss(batch)
        args = [defaults[n] for n in positional]
        args[0], args[1], args[-1] = 'pos', loss, 7
        args[positional.index('on_epoch')] = True
        model.log(*args)
        model.log('kw', loss, on_epoch=True, batch_size=7)
        seen.append(model.trainer._results['training_step.pos'].cumulated_batch_size.item())
        return loss

    model.training_step = step
    _fit(model, None, max_steps=2)
    assert list(inspect.signature(model.training_step).parameters) == ["dataloader_iter"]
    assert seen == [7, 14]


@pytest.mark.unit
def test_log_epoch_mean_is_weighted_by_batch_size():
    model = _TinyModel(_stateful(n=3))  # batches of 2 and 1 samples

    def step(batch, batch_idx):
        model.log('v', 1.0 if batch_idx == 0 else 4.0, on_step=False, on_epoch=True)
        return model._loss(batch)

    model.training_step = step
    _fit(model, None, max_epochs=1)
    assert model.trainer.callback_metrics['v'].item() == pytest.approx(2.0)  # (2 * 1 + 1 * 4) / 3


@pytest.mark.unit
def test_log_step_only_does_not_need_a_batch_size():
    # Batches are plain Python ints, so no batch size can be inferred; step-only logging doesn't need one.
    model = _TinyModel(_stateful(batch_size=None))

    def step(batch, batch_idx):
        loss = ((model.w - float(batch)) ** 2).sum()
        model.log('step_loss', loss, on_step=True, on_epoch=False)
        return loss

    model.training_step = step
    _fit(model, None, max_steps=2)
    assert list(inspect.signature(model.training_step).parameters) == ["dataloader_iter"]


@pytest.mark.unit
def test_log_epoch_mean_before_the_batch_is_read_requires_batch_size():
    model = _TinyModel(_stateful())
    model.on_train_batch_start = lambda batch, batch_idx: model.log('early', 1.0, on_epoch=True)
    with pytest.raises(MisconfigurationException, match="Pass batch_size= explicitly"):
        _fit(model, None, max_steps=2)


class _DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return _stateful()


def _ids(batch) -> list:
    if isinstance(batch, (list, tuple)):  # CombinedLoader of a list yields a list of batches
        (batch,) = batch
    return batch.tolist()


def _fit(model: _TinyModel, ckpt_dir, ckpt_path=None, **kwargs) -> _TinyModel:
    trainer_kwargs = {k: kwargs.pop(k) for k in ('max_steps', 'max_epochs', 'accumulate_grad_batches') if k in kwargs}
    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_checkpointing=ckpt_dir is not None,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=(
            [ModelCheckpoint(dirpath=ckpt_dir, every_n_train_steps=SPLIT, save_top_k=-1)] if ckpt_dir else None
        ),
        **trainer_kwargs,
    )
    trainer.fit(model, ckpt_path=ckpt_path, **kwargs)
    return model


def _fit_and_resume(tmp_path: Path, make_model, fit_kwargs=dict) -> tuple:
    model_a = _fit(make_model(), tmp_path / "a", max_steps=TOTAL, **fit_kwargs())
    assert len(model_a.calls) == TOTAL
    ckpt = next((tmp_path / "a").glob(f"*step={SPLIT}.ckpt"))
    model_b = _fit(make_model(), tmp_path / "b", ckpt_path=str(ckpt), max_steps=TOTAL, **fit_kwargs())
    return [ids for _, ids in model_a.calls], [ids for _, ids in model_b.calls]
