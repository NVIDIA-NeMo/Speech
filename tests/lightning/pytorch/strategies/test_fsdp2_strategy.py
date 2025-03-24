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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Import the module to test
from nemo.lightning.pytorch.strategies.fsdp2_strategy import FSDP2Strategy


# Mock dependencies
@pytest.fixture
def mock_mixed_precision_policy():
    mock_policy = MagicMock()
    return mock_policy


@pytest.fixture
def mock_device_mesh():
    mesh = MagicMock()
    return mesh


@pytest.fixture
def mock_lightning_module():
    module = MagicMock()
    module.training_step = MagicMock(return_value=torch.tensor(1.0))
    module.state_dict = MagicMock(return_value={"layer1.weight": torch.randn(10, 10)})
    return module


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.global_step = 10
    # Mock the optimization loops
    trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed = 5
    trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.current.completed = 3
    trainer.state.fn = "fitting"
    return trainer


@pytest.fixture
def strategy(mock_mixed_precision_policy):
    strategy = FSDP2Strategy(
        data_parallel_size=2,
        tensor_parallel_size=1,
    )
    strategy.accelerator = MagicMock()
    strategy._lightning_module = MagicMock()
    strategy.model = MagicMock()
    strategy.num_nodes = 1
    strategy.optimizers = [MagicMock()]
    strategy.store = MagicMock()
    return strategy


class TestFSDP2Strategy:

    def test_init(self):
        """Test that the strategy initializes with default parameters"""
        strategy = FSDP2Strategy(data_parallel_size=2, tensor_parallel_size=1)

        assert strategy._data_parallel_size == 2
        assert strategy._tensor_parallel_size == 1
        assert strategy.checkpoint is None
        assert strategy.parallelize_fn is not None
        assert strategy.mp_policy is not None

    def test_lightning_restore_optimizer(self, strategy):
        """Test that optimizer state restoration is enabled"""
        assert strategy.lightning_restore_optimizer is True

    def test_load_optimizer_state_dict(self, strategy):
        """Test that optimizer state dict is stored for later restoration"""
        mock_checkpoint = {"optimizer_states": [{"param_groups": []}]}
        strategy.load_optimizer_state_dict(mock_checkpoint)
        assert strategy.checkpoint == mock_checkpoint

    def test_setup(self, strategy, mock_trainer):
        """Test that setup configures the strategy correctly"""
        strategy.parallelize = MagicMock()

        strategy.setup(mock_trainer)

        strategy.accelerator.setup.assert_called_once_with(mock_trainer)
        strategy.parallelize.assert_called_once()
        assert strategy.trainer == mock_trainer

    def test_parallelize(self, strategy):
        """Test that parallelize applies FSDP to the model"""
        mock_parallelize_fn = MagicMock()
        strategy.parallelize_fn = mock_parallelize_fn
        strategy._device_mesh = MagicMock()

        strategy.parallelize()

        mock_parallelize_fn.assert_called_once()
        assert strategy.parallelize_fn is None  # Should be cleared after use

        # Test calling parallelize again
        with patch("nemo.utils.logging.warning") as mock_warning:
            strategy.parallelize()
            mock_warning.assert_called_once()

    @patch('nemo.lightning.pytorch.strategies.fsdp2_strategy.create_checkpoint_io')
    def test_checkpoint_io(self, mock_create_checkpoint_io):
        class Dummy: ...

        mock_create_checkpoint_io.side_effect = lambda *args, **kwargs: Dummy()
        strategy = FSDP2Strategy()

        first_io = strategy.checkpoint_io
        mock_create_checkpoint_io.assert_called_once()

        assert first_io == strategy.checkpoint_io

        new_io = object()
        strategy.checkpoint_io = new_io
        assert new_io == strategy.checkpoint_io

    def test_remove_checkpoint(self, strategy):
        """Test that remove_checkpoint properly removes checkpoint directories"""
        test_path = Path("test_ckpt")

        # Test removing directory
        with (
            patch("os.path.islink", return_value=False),
            patch("os.path.exists", return_value=True),
            patch("shutil.rmtree") as mock_rmtree,
        ):

            strategy.remove_checkpoint(test_path)

            mock_rmtree.assert_called_once()

        # Test removing symlink
        with patch("os.path.islink", return_value=True), patch("os.unlink") as mock_unlink:

            strategy.remove_checkpoint(test_path)

            mock_unlink.assert_called_once()

    def test_save_checkpoint(self, strategy):
        """Test that save_checkpoint correctly delegates to checkpoint_io"""
        mock_checkpoint = {"state": "test"}
        test_path = "test_path"

        strategy.checkpoint_io = MagicMock()
        strategy.save_checkpoint(mock_checkpoint, test_path)

        strategy.checkpoint_io.save_checkpoint.assert_called_once_with(
            mock_checkpoint, test_path, storage_options=None
        )

    def test_load_checkpoint(self, strategy):
        """Test that load_checkpoint correctly delegates to checkpoint_io"""
        test_path = "test_path"
        mock_result = {"state": "test"}

        strategy.checkpoint_io = MagicMock()
        strategy.checkpoint_io.load_checkpoint.return_value = mock_result

        result = strategy.load_checkpoint(test_path)

        strategy.checkpoint_io.load_checkpoint.assert_called_once_with(test_path)
        assert result == mock_result

    def test_validation_step(self, strategy):
        """Test that validation_step executes correctly"""
        mock_batch = MagicMock()
        strategy._step_proxy = MagicMock(return_value=(torch.tensor(0.5), {"avg": torch.tensor(0.5)}))
        strategy.precision_plugin = MagicMock()
        strategy.precision_plugin.val_step_context = MagicMock()
        strategy.precision_plugin.val_step_context.return_value.__enter__ = MagicMock()
        strategy.precision_plugin.val_step_context.return_value.__exit__ = MagicMock(return_value=None)

        result = strategy.validation_step(mock_batch, 0)

        strategy._step_proxy.assert_called_once_with("validation", mock_batch, 0)
        strategy.lightning_module.log.assert_called_once()
        assert result == torch.tensor(0.5)

    def test_test_step(self, strategy):
        """Test that test_step executes correctly"""
        mock_batch = MagicMock()
        strategy._step_proxy = MagicMock(return_value=(torch.tensor(0.3), {"avg": torch.tensor(0.3)}))
        strategy.precision_plugin = MagicMock()
        strategy.precision_plugin.test_step_context = MagicMock()
        strategy.precision_plugin.test_step_context.return_value.__enter__ = MagicMock()
        strategy.precision_plugin.test_step_context.return_value.__exit__ = MagicMock(return_value=None)

        result = strategy.test_step(mock_batch, 0)

        strategy._step_proxy.assert_called_once_with("test", mock_batch, 0)
        strategy.lightning_module.log.assert_called_once()
        assert result == torch.tensor(0.3)

    def test_predict_step(self, strategy):
        """Test that predict_step executes correctly"""
        mock_batch = MagicMock()
        mock_reduced = {"avg": torch.tensor(0.2)}
        strategy._step_proxy = MagicMock(return_value=(torch.tensor(0.2), mock_reduced))
        strategy.precision_plugin = MagicMock()
        strategy.precision_plugin.predict_step_context = MagicMock()
        strategy.precision_plugin.predict_step_context.return_value.__enter__ = MagicMock()
        strategy.precision_plugin.predict_step_context.return_value.__exit__ = MagicMock(return_value=None)

        result = strategy.predict_step(mock_batch, 0)

        strategy._step_proxy.assert_called_once_with("predict", mock_batch, 0)
        assert result == mock_reduced
