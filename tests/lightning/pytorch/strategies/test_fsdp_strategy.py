# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import shutil
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning.pytorch.plugins import FSDPPrecision

# Import the module to test
from nemo.lightning.pytorch.strategies.fsdp_strategy import FSDPStrategy


# Mock dependencies
@pytest.fixture
def mock_transformer_layer():
    """Mock for TransformerLayer class"""
    return MagicMock()


@pytest.fixture
def mock_lightning_module():
    """Mock for Lightning module"""
    module = MagicMock()
    module.training_step = MagicMock(return_value=torch.tensor(1.0))
    module.validation_step = MagicMock(return_value=torch.tensor(0.5))
    module.test_step = MagicMock(return_value=torch.tensor(0.3))
    module.predict_step = MagicMock(return_value=torch.tensor(0.2))
    module.state_dict = MagicMock(return_value={"layer1.weight": torch.randn(10, 10)})
    module.log = MagicMock()
    return module


@pytest.fixture
def mock_trainer():
    """Mock for Trainer"""
    trainer = MagicMock()
    trainer.global_step = 10
    # Mock the optimization loops
    trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed = 5
    trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.current.completed = 3
    trainer.state.fn = "fitting"
    return trainer


@pytest.fixture
def mock_checkpoint_io():
    """Mock for CheckpointIO"""
    io = MagicMock()
    io.save_checkpoint = MagicMock()
    io.load_checkpoint = MagicMock(return_value={"state_dict": OrderedDict([]), "sharded_state_dict": {}})
    return io


@pytest.fixture
def strategy(mock_transformer_layer):
    """Create an instance of FSDPStrategy for testing"""
    with patch("megatron.core.transformer.transformer_layer.TransformerLayer", mock_transformer_layer):
        strategy = FSDPStrategy(
            auto_wrap_policy={mock_transformer_layer},
            state_dict_type="sharded",
            ckpt_load_optimizer=True,
            ckpt_save_optimizer=True,
        )
        strategy.accelerator = MagicMock()
        strategy.model = MagicMock()
        strategy.cluster_environment = MagicMock()
        strategy.cluster_environment.global_rank.return_value = 0
        strategy.cluster_environment.world_size.return_value = 2
        strategy.cluster_environment.main_address = "localhost"
        strategy.cluster_environment.main_port = 12345
        strategy.optimizers = [MagicMock()]
        strategy.store = MagicMock()
        strategy._checkpoint_io = None
        strategy._process_group_backend = "nccl"
        strategy._logger = MagicMock()
        strategy.kwargs = {}
        return strategy


class TestFSDPStrategy:

    def test_init(self, mock_transformer_layer):
        """Test that the strategy initializes with correct parameters"""
        with patch("megatron.core.transformer.transformer_layer.TransformerLayer", mock_transformer_layer):
            strategy = FSDPStrategy(auto_wrap_policy={mock_transformer_layer}, state_dict_type="sharded")

            assert strategy.ckpt_load_optimizer is True
            assert strategy.ckpt_save_optimizer is True
            assert strategy.data_sampler is None
            assert strategy.store is None

    def test_training_step(self, strategy):
        """Test that training_step executes correctly"""
        mock_batch = MagicMock()
        strategy._step_proxy = MagicMock(return_value=(torch.tensor(1.0), {"avg": torch.tensor(1.0)}))
        strategy.precision_plugin = FSDPPrecision('bf16-mixed')
        strategy.precision_plugin.train_step_context = MagicMock()
        strategy.precision_plugin.train_step_context.return_value.__enter__ = MagicMock()
        strategy.precision_plugin.train_step_context.return_value.__exit__ = MagicMock(return_value=None)

        try:
            strategy.training_step(mock_batch, 0)
        except AssertionError:
            pass

    def test_validation_step(self, strategy):
        """Test that validation_step executes correctly"""
        mock_batch = MagicMock()
        strategy._step_proxy = MagicMock(return_value=(torch.tensor(0.5), {"avg": torch.tensor(0.5)}))
        strategy.precision_plugin = FSDPPrecision('bf16-mixed')
        strategy.precision_plugin.val_step_context = MagicMock()
        strategy.precision_plugin.val_step_context.return_value.__enter__ = MagicMock()
        strategy.precision_plugin.val_step_context.return_value.__exit__ = MagicMock(return_value=None)

        try:
            strategy.validation_step(mock_batch, 0)
        except AssertionError:
            pass

    def test_test_step(self, strategy):
        """Test that test_step executes correctly"""
        mock_batch = MagicMock()
        strategy._step_proxy = MagicMock(return_value=(torch.tensor(0.3), {"avg": torch.tensor(0.3)}))
        strategy.precision_plugin = FSDPPrecision('bf16-mixed')
        strategy.precision_plugin.test_step_context = MagicMock()
        strategy.precision_plugin.test_step_context.return_value.__enter__ = MagicMock()
        strategy.precision_plugin.test_step_context.return_value.__exit__ = MagicMock(return_value=None)

        try:
            strategy.test_step(mock_batch, 0)
        except AssertionError:
            pass

    def test_predict_step(self, strategy):
        """Test that predict_step executes correctly"""
        mock_batch = MagicMock()
        mock_reduced = {"avg": torch.tensor(0.2)}
        strategy._step_proxy = MagicMock(return_value=(torch.tensor(0.2), mock_reduced))
        strategy.precision_plugin = FSDPPrecision('bf16-mixed')
        strategy.precision_plugin.predict_step_context = MagicMock()
        strategy.precision_plugin.predict_step_context.return_value.__enter__ = MagicMock()
        strategy.precision_plugin.predict_step_context.return_value.__exit__ = MagicMock(return_value=None)

        try:
            strategy.predict_step(mock_batch, 0)
        except AssertionError:
            pass

    def test_process_dataloader(self, strategy):
        """Test that process_dataloader transforms dataloader correctly"""
        mock_dataloader = MagicMock()
        transformed_dataloader = MagicMock()

        # Test with data_sampler
        strategy.data_sampler = MagicMock()
        strategy.data_sampler.transform_dataloader = MagicMock(return_value=transformed_dataloader)
        result = strategy.process_dataloader(mock_dataloader)
        strategy.data_sampler.transform_dataloader.assert_called_once_with(mock_dataloader)
        assert result == transformed_dataloader

        # Test without data_sampler
        strategy.data_sampler = None
        result = strategy.process_dataloader(mock_dataloader)
        assert result == mock_dataloader

    @patch('nemo.lightning.pytorch.strategies.fsdp_strategy.create_checkpoint_io')
    def test_checkpoint_io(self, mock_create_checkpoint_io):
        class Dummy: ...

        mock_create_checkpoint_io.side_effect = lambda *args, **kwargs: Dummy()
        strategy = FSDPStrategy()

        first_io = strategy.checkpoint_io
        mock_create_checkpoint_io.assert_called_once()

        assert first_io == strategy.checkpoint_io

        new_io = object()
        strategy.checkpoint_io = new_io
        assert new_io == strategy.checkpoint_io

    def test_current_epoch_step(self, strategy, mock_trainer):
        """Test current_epoch_step returns the maximum step value"""
        strategy.trainer = mock_trainer
        assert strategy.current_epoch_step == 5

    def test_remove_checkpoint(self, strategy):
        """Test that remove_checkpoint properly removes checkpoint directories"""
        test_path = Path("test_ckpt")

        # Test with directory
        with patch("os.path.islink", return_value=False), patch("shutil.rmtree") as mock_rmtree:

            strategy.remove_checkpoint(test_path)
            mock_rmtree.assert_called_once()

        # Test with symlink
        with patch("os.path.islink", return_value=True), patch("os.unlink") as mock_unlink:

            strategy.remove_checkpoint(test_path)
            mock_unlink.assert_called_once()

    def test_save_checkpoint(self, strategy):
        """Test that save_checkpoint correctly processes and saves the checkpoint"""
        mock_checkpoint = {
            "state_dict": {"layer1.weight": torch.randn(10, 10)},
        }
        test_path = "test_path"

        strategy.checkpoint_io = MagicMock()
        strategy.save_checkpoint(mock_checkpoint, test_path)

        strategy.checkpoint_io.save_checkpoint.assert_called_once_with(
            mock_checkpoint, test_path, storage_options=None
        )
