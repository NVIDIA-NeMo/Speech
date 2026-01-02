# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import types
from pathlib import Path
from typing import Any, Dict

import lightning.pytorch as pl
import pytest
import torch
from lightning.fabric.plugins import TorchCheckpointIO
from lightning.pytorch.demos.boring_classes import BoringModel

from nemo.collections.common.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils.callbacks.dist_ckpt_io import (
    AsyncFinalizableCheckpointIO,
    AsyncFinalizerCallback,
    DistributedCheckpointIO,
)

try:
    from megatron.core.dist_checkpointing import ShardedTensor

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


class ExampleModel(BoringModel):
    def on_validation_epoch_end(self) -> None:
        self.log("val_loss", torch.tensor(1.0))


class ExampleMCoreModel(ExampleModel):
    def sharded_state_dict(self):
        return {
            'a': ShardedTensor.from_rank_offsets('a', self.layer.weight, replica_id=torch.distributed.get_rank()),
            'const': 3,
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['sharded_state_dict'] = self.sharded_state_dict()


class MockDistributedCheckpointIO(DistributedCheckpointIO):
    def __init__(self, save_ckpt_format):
        super().__init__(save_ckpt_format)
        self.save_checkpoint_called_args = None

    def save_checkpoint(self, *args, **kwargs) -> None:
        self.save_checkpoint_called_args = args, kwargs


class MockTorchCheckpointIO(TorchCheckpointIO):
    def __init__(self):
        super().__init__()
        self.save_checkpoint_called_args = None

    def save_checkpoint(self, *args, **kwargs) -> None:
        self.save_checkpoint_called_args = args, kwargs


def _get_last_checkpoint_dir(root_dir: Path, model: pl.LightningModule, suffix: str = '') -> Path:
    steps = len(model.train_dataloader().dataset) * model.trainer.max_epochs // torch.distributed.get_world_size()
    return root_dir / 'checkpoints' / f'epoch={model.trainer.max_epochs - 1}-step={steps}{suffix}'


def _get_nlp_strategy_without_optimizer_state():
    strategy = NLPDDPStrategy()
    # this ensures optimizer sharded state creation is skipped
    strategy.optimizer_sharded_state_dict = types.MethodType(
        lambda self, unsharded_optim_state: unsharded_optim_state, strategy
    )
    return strategy


class TestDistCkptIO:
    @pytest.mark.run_only_on('GPU')
    def test_dist_ckpt_io_called_for_mcore_models(self, tmp_path):
        strategy = _get_nlp_strategy_without_optimizer_state()
        checkpoint_io = MockDistributedCheckpointIO('xxx')

        test_trainer = pl.Trainer(
            enable_checkpointing=True,
            logger=False,
            max_epochs=2,
            strategy=strategy,
            plugins=[checkpoint_io],
            default_root_dir=tmp_path,
        )
        model = ExampleMCoreModel()
        test_trainer.fit(model)

        assert isinstance(test_trainer.strategy.checkpoint_io, MockDistributedCheckpointIO)
        assert checkpoint_io.save_checkpoint_called_args is not None
        (state_dict, path), _ = checkpoint_io.save_checkpoint_called_args
        # Ckpt path doesn't contain the .ckpt suffix
        assert path.name == _get_last_checkpoint_dir(tmp_path, model).name

    @pytest.mark.run_only_on('GPU')
    def test_dist_ckpt_path_not_executed_for_non_core_models(self, tmp_path):
        strategy = NLPDDPStrategy()
        checkpoint_io = MockTorchCheckpointIO()

        test_trainer = pl.Trainer(
            enable_checkpointing=True,
            logger=False,
            max_epochs=2,
            strategy=strategy,
            plugins=[checkpoint_io],
            default_root_dir=tmp_path,
        )
        model = ExampleModel()
        test_trainer.fit(model)

        assert isinstance(test_trainer.strategy.checkpoint_io, MockTorchCheckpointIO)
        if test_trainer.is_global_zero:
            assert checkpoint_io.save_checkpoint_called_args is not None
            (state_dict, path), _ = checkpoint_io.save_checkpoint_called_args
            # Ckpt path *does* contain the .ckpt suffix
            assert os.path.basename(path) == _get_last_checkpoint_dir(tmp_path, model, suffix='.ckpt').name
        else:
            assert checkpoint_io.save_checkpoint_called_args is None


class TestAsyncSave:
    @pytest.mark.run_only_on('GPU')
    def test_async_save_produces_same_checkpoints_as_sync(self, tmp_path):
        strategy = _get_nlp_strategy_without_optimizer_state()
        sync_checkpoint_io = DistributedCheckpointIO('torch_dist')
        async_checkpoint_io = AsyncFinalizableCheckpointIO(DistributedCheckpointIO('torch_dist', async_save=True))

        model = ExampleMCoreModel()

        # dummy_trainer just to initialize NCCL
        dummy_trainer = pl.Trainer(
            enable_checkpointing=False,
            logger=False,
            max_epochs=1,
            strategy=_get_nlp_strategy_without_optimizer_state(),
            plugins=[sync_checkpoint_io],
        )
        dummy_trainer.fit(model)
        tmp_path = strategy.broadcast(tmp_path)

        sync_ckpt_dir = tmp_path / 'sync_checkpoints'
        async_ckpt_dir = tmp_path / 'async_checkpoints'

        sync_test_trainer = pl.Trainer(
            enable_checkpointing=True,
            logger=False,
            max_epochs=1,
            strategy=_get_nlp_strategy_without_optimizer_state(),
            plugins=[sync_checkpoint_io],
            default_root_dir=sync_ckpt_dir,
        )
        sync_test_trainer.fit(model)

        async_test_trainer = pl.Trainer(
            enable_checkpointing=True,
            logger=False,
            max_epochs=1,
            strategy=_get_nlp_strategy_without_optimizer_state(),
            plugins=[async_checkpoint_io],
            callbacks=AsyncFinalizerCallback(),
            default_root_dir=async_ckpt_dir,
        )
        async_test_trainer.fit(model)

        # Load and compare checkpoints
        checkpoint = {'sharded_state_dict': model.sharded_state_dict()}
        sync_state_dict = sync_checkpoint_io.load_checkpoint(
            _get_last_checkpoint_dir(sync_ckpt_dir, model), sharded_state_dict=checkpoint
        )
        async_state_dict = async_checkpoint_io.load_checkpoint(
            _get_last_checkpoint_dir(async_ckpt_dir, model), sharded_state_dict=checkpoint
        )

        assert sync_state_dict['sharded_state_dict']['const'] == async_state_dict['sharded_state_dict']['const']
        assert torch.all(sync_state_dict['sharded_state_dict']['a'] == async_state_dict['sharded_state_dict']['a'])
