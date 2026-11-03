# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resume and callback compatibility with the supported Lightning release."""

import copy
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import lightning.pytorch as pl
import pytest
import torch
from lightning.fabric.plugins import TorchCheckpointIO
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.trainer.connectors.checkpoint_connector import _CheckpointConnector
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from nemo.collections.common.callbacks import EMA
from nemo.collections.speechlm2.parts.parallel import AutomodelParallelStrategy
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO, DistributedCheckpointIO
from nemo.utils.callbacks.nemo_model_checkpoint import NeMoModelCheckpoint
from nemo.utils.exp_manager import NeMoCheckpointConnector, exp_manager


@pytest.mark.unit
@pytest.mark.parametrize('async_save', [False, True])
@pytest.mark.parametrize('at_epoch_end', [False, True])
def test_last_checkpoint_advances_when_metric_plateaus(tmp_path, async_save, at_epoch_end):
    checkpoint = NeMoModelCheckpoint(
        dirpath=tmp_path,
        filename='{step}',
        monitor='val_loss',
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=at_epoch_end,
        async_save=async_save,
        save_nemo_on_train_end=False,
    )
    deferred = _DeferredCheckpointIO()
    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator='cpu',
        devices=1,
        max_epochs=2,
        callbacks=[checkpoint],
        plugins=[deferred] if async_save else None,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(_PlateauModel())
    if async_save:
        # Scheduling updates Lightning bookkeeping; completion must not rewind it.
        assert checkpoint._last_global_step_saved == trainer.global_step
        deferred.flush()
        assert checkpoint._last_global_step_saved == trainer.global_step
    saved = torch.load(checkpoint.last_model_path, weights_only=True)
    assert saved['global_step'] == trainer.global_step == 8
    assert torch.load(checkpoint.best_model_path, weights_only=True)['global_step'] == 4


@pytest.mark.unit
@pytest.mark.parametrize('weights_only', [None, False, True])
def test_automodel_checkpoint_forwards_load_policy(tmp_path, weights_only):
    strategy = AutomodelParallelStrategy()
    strategy.broadcast = lambda path: path
    with patch.object(ModelParallelStrategy, 'load_checkpoint', return_value={'global_step': 7}) as load:
        assert strategy.load_checkpoint(tmp_path / 'model.ckpt', weights_only=weights_only) == {'global_step': 7}
    load.assert_called_once_with(tmp_path / 'model.ckpt', weights_only=weights_only)


@pytest.mark.unit
def test_resume_connector_forwards_load_policy():
    trainer = SimpleNamespace(ckpt_path=None)
    connector = NeMoCheckpointConnector(trainer)
    with patch.object(_CheckpointConnector, 'resume_start') as resume:
        connector.resume_start(weights_only=True)
    resume.assert_called_once_with(None, weights_only=True)


@pytest.mark.unit
def test_distributed_checkpoint_rejects_unsupported_restricted_load():
    # The optional Megatron backend has no weights_only parameter. Never silently
    # discard an explicit request for restricted unpickling.
    checkpoint_io = object.__new__(DistributedCheckpointIO)
    with pytest.raises(ValueError, match='weights_only=True'):
        checkpoint_io.load_checkpoint('model', sharded_state_dict={}, weights_only=True)


@pytest.mark.unit
@pytest.mark.parametrize('shared_memory', [False, True])
@pytest.mark.parametrize('weights_only', [False, True])
def test_s3_checkpoint_load_policy_and_map_location(tmp_path, monkeypatch, shared_memory, weights_only):
    pytest.importorskip('boto3')
    from nemo.utils.callbacks import s3_checkpoint_io as module

    path = tmp_path / 'model.ckpt'
    torch.save({'tensor': torch.ones(2)}, path)
    monkeypatch.setattr(module, 'SHARED_MEM_DIR', str(tmp_path if shared_memory else tmp_path / 'missing'))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(module.S3Utils, 'download_s3_file_to_stream', lambda **kw: io.BytesIO(path.read_bytes()))
    monkeypatch.setattr(
        module.S3Utils, 'download_s3_file_to_path', lambda **kw: Path(kw['file_path']).write_bytes(path.read_bytes())
    )
    checkpoint_io = module.S3CheckpointIO('s3://bucket')
    map_location = Mock(side_effect=lambda storage, loc: storage)
    with patch.object(module.torch, 'load', wraps=torch.load) as load:
        loaded = checkpoint_io.load_checkpoint(
            's3://bucket/model.ckpt', map_location=map_location, weights_only=weights_only
        )
    assert torch.equal(loaded['tensor'], torch.ones(2))
    assert load.call_args.kwargs['weights_only'] is weights_only
    assert load.call_args.kwargs['map_location'] is map_location
    map_location.assert_called()


@pytest.mark.unit
@pytest.mark.parametrize('stage', [TrainerFn.VALIDATING, TrainerFn.TESTING, TrainerFn.PREDICTING])
def test_ema_evaluation_without_sidecar(tmp_path, stage):
    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator='cpu',
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[NeMoModelCheckpoint(dirpath=tmp_path)],
    )
    trainer.state.fn = stage
    trainer.ckpt_path = str(tmp_path / 'model.ckpt')
    checkpoint = {'optimizer_states': ['unchanged']}
    EMA(decay=0.9).on_load_checkpoint(trainer, pl.LightningModule(), checkpoint)
    assert checkpoint['optimizer_states'] == ['unchanged']


@pytest.mark.unit
def test_neptune_rejected_before_creating_experiment(tmp_path):
    trainer = pl.Trainer(default_root_dir=tmp_path, accelerator='cpu', logger=False, enable_checkpointing=False)
    directory = tmp_path / 'experiment'
    with pytest.raises(ValueError, match='create_neptune_logger.*Lightning'):
        exp_manager(trainer, {'explicit_log_dir': str(directory), 'create_neptune_logger': True})
    assert not directory.exists()


class _PlateauModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.log('val_loss', 1.0)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(torch.ones(4), batch_size=1)

    def val_dataloader(self):
        return DataLoader(torch.ones(1), batch_size=1)


class _DeferredCheckpointIO(AsyncFinalizableCheckpointIO):
    def __init__(self):
        _WrappingCheckpointIO.__init__(self, TorchCheckpointIO())
        self.pending = []

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        self.pending.append((copy.deepcopy(checkpoint), path, storage_options['finalize_fn']))

    def flush(self):
        for checkpoint, path, finalize in self.pending:
            self.checkpoint_io.save_checkpoint(checkpoint, path)
            finalize()
        self.pending.clear()

    def teardown(self):
        pass


@pytest.mark.unit
def test_automodel_resume_connector_restores_metadata(tmp_path):
    strategy = AutomodelParallelStrategy()
    strategy.broadcast = lambda path: path
    connector = _CheckpointConnector(SimpleNamespace(strategy=strategy))
    state = {'global_step': 7, 'pytorch-lightning_version': pl.__version__}
    with patch.object(ModelParallelStrategy, 'load_checkpoint', return_value=state) as load:
        connector.resume_start(tmp_path / 'model.ckpt', weights_only=True)
    assert connector._loaded_checkpoint['global_step'] == 7
    load.assert_called_once_with(tmp_path / 'model.ckpt', weights_only=True)


@pytest.mark.unit
@pytest.mark.parametrize('sidecar', ['valid', 'missing', 'unsupported'])
def test_ema_fit_sidecar_policy(tmp_path, sidecar):
    import pickle
    from lightning.pytorch.utilities.exceptions import MisconfigurationException

    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator='cpu',
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[NeMoModelCheckpoint(dirpath=tmp_path)],
    )
    trainer.state.fn = TrainerFn.FITTING
    trainer.ckpt_path = str(tmp_path / 'model.ckpt')
    if sidecar != 'missing':
        value = torch.ones(2) if sidecar == 'valid' else _UnsupportedValue()
        torch.save({'optimizer_states': [value]}, tmp_path / 'model-EMA.ckpt')
    checkpoint = {'optimizer_states': ['unchanged']}
    if sidecar == 'valid':
        EMA(decay=0.9).on_load_checkpoint(trainer, pl.LightningModule(), checkpoint)
        torch.testing.assert_close(checkpoint['optimizer_states'][0], torch.ones(2))
    else:
        error = MisconfigurationException if sidecar == 'missing' else pickle.UnpicklingError
        with pytest.raises(error):
            EMA(decay=0.9).on_load_checkpoint(trainer, pl.LightningModule(), checkpoint)
        assert checkpoint['optimizer_states'] == ['unchanged']


class _UnsupportedValue:
    pass
