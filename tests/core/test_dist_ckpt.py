import os
import types
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from lightning_fabric.plugins import TorchCheckpointIO
from pytorch_lightning.demos.boring_classes import BoringModel

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils.callbacks.dist_ckpt_io import DistributedCheckpointIO


class ExampleModel(BoringModel):
    def on_validation_epoch_end(self) -> None:
        self.log("val_loss", torch.tensor(1.0))


class ExampleMCoreModel(ExampleModel):
    def sharded_state_dict(self):
        return {'a': 3}


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
    return root_dir / 'checkpoints' / f'epoch=1-step={steps}{suffix}'


class TestDistCkptIO:
    @pytest.mark.run_only_on('GPU')
    def test_dist_ckpt_io_called_for_mcore_models(self, tmp_path):
        strategy = NLPDDPStrategy()
        # skip optimizer sharded state creation:
        strategy.optimizer_sharded_state_dict = types.MethodType(
            lambda self, unsharded_optim_state: unsharded_optim_state, strategy
        )
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
        assert path.name == _get_last_checkpoint_dir(tmp_path, model).name, len(test_trainer.strategy.parallel_devices)

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
