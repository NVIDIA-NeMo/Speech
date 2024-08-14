from unittest.mock import MagicMock, patch

import pytest
import torch
from nemo.lightning.pytorch.callbacks.nsys import NsysCallback


class TestNsysCallback:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        self.cuda_mock = patch('torch.cuda')
        self.cudart_mock = patch('torch.cuda.cudart')
        self.emit_nvtx_mock = patch('torch.autograd.profiler.emit_nvtx')
        self.get_rank_mock = patch('nemo.lightning.pytorch.callbacks.nsys.get_rank')

        self.cuda_mock.start()
        self.cudart_mock.start()
        self.emit_nvtx_mock.start()
        self.get_rank_mock.start()

        # Mock CUDA availability
        torch.cuda.is_available = MagicMock(return_value=True)
        torch.cuda.current_device = MagicMock(return_value=0)

        yield

        self.cuda_mock.stop()
        self.cudart_mock.stop()
        self.emit_nvtx_mock.stop()
        self.get_rank_mock.stop()

    @pytest.fixture
    def mock_trainer(self):
        trainer = MagicMock()
        trainer.strategy.root_device.type = 'cuda'
        return trainer

    @pytest.fixture
    def mock_pl_module(self):
        return MagicMock()

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        callback = NsysCallback(start_step=10, end_step=20, ranks=[0, 1], gen_shape=True)
        assert callback._nsys_profile_start_step == 10
        assert callback._nsys_profile_end_step == 20
        assert callback._nsys_profile_ranks == [0, 1]
        assert callback._nsys_profile_gen_shape == True

    def test_init_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(AssertionError):
            NsysCallback(start_step='10', end_step=20)

        with pytest.raises(AssertionError):
            NsysCallback(start_step=10, end_step='20')

        with pytest.raises(AssertionError):
            NsysCallback(start_step=20, end_step=10)

    @patch('nemo.lightning.pytorch.callbacks.nsys.get_rank')
    @patch('torch.cuda.cudart')
    @patch('torch.autograd.profiler.emit_nvtx')
    def test_on_train_batch_start_profiling(
        self, mock_emit_nvtx, mock_cudart, mock_get_rank, mock_trainer, mock_pl_module
    ):
        """Test on_train_batch_start when profiling should start."""
        mock_get_rank.return_value = 0
        callback = NsysCallback(start_step=10, end_step=20, ranks=[0], gen_shape=True)

        mock_trainer.strategy.current_epoch_step = 10
        callback.on_train_batch_start(mock_trainer, mock_pl_module, None, 10)

        mock_cudart().cudaProfilerStart.assert_called_once()
        mock_emit_nvtx.assert_called_once_with(record_shapes=True)

    @patch('nemo.lightning.pytorch.callbacks.nsys.get_rank')
    @patch('torch.cuda.cudart')
    def test_on_train_batch_start_no_profiling(self, mock_cudart, mock_get_rank, mock_trainer, mock_pl_module):
        """Test on_train_batch_start when profiling should not start."""
        mock_get_rank.return_value = 0
        callback = NsysCallback(start_step=10, end_step=20, ranks=[0])

        mock_trainer.strategy.current_epoch_step = 9
        callback.on_train_batch_start(mock_trainer, mock_pl_module, None, 9)

        mock_cudart().cudaProfilerStart.assert_not_called()

    @patch('nemo.lightning.pytorch.callbacks.nsys.get_rank')
    @patch('torch.cuda.cudart')
    @patch('torch.autograd.profiler.emit_nvtx')
    def test_on_train_batch_end_profiling(
        self, mock_emit_nvtx, mock_cudart, mock_get_rank, mock_trainer, mock_pl_module
    ):
        """Test on_train_batch_end when profiling should end."""
        mock_get_rank.return_value = 0
        callback = NsysCallback(start_step=10, end_step=20, ranks=[0])

        mock_trainer.strategy.current_epoch_step = 20
        callback.on_train_batch_end(mock_trainer, mock_pl_module, None, None, 20)

        mock_cudart().cudaProfilerStop.assert_called_once()

    @patch('nemo.lightning.pytorch.callbacks.nsys.get_rank')
    @patch('torch.cuda.cudart')
    @patch('torch.autograd.profiler.emit_nvtx')
    def test_on_train_batch_end_no_profiling(
        self, mock_emit_nvtx, mock_cudart, mock_get_rank, mock_trainer, mock_pl_module
    ):
        """Test on_train_batch_end when profiling should not end."""
        mock_get_rank.return_value = 0
        callback = NsysCallback(start_step=10, end_step=20, ranks=[0])

        callback.on_train_batch_end(mock_trainer, mock_pl_module, None, None, 19)

        mock_cudart().cudaProfilerStop.assert_not_called()

    def test_non_cuda_device(self, mock_trainer, mock_pl_module):
        """Test behavior when the device is not CUDA."""
        mock_trainer.strategy.root_device.type = 'cpu'
        callback = NsysCallback(start_step=10, end_step=20, ranks=[0])

        callback.on_train_batch_start(mock_trainer, mock_pl_module, None, 10)
        callback.on_train_batch_end(mock_trainer, mock_pl_module, None, None, 20)

        # No exceptions should be raised, and no profiling calls should be made

    @patch('nemo.lightning.pytorch.callbacks.nsys.get_rank')
    def test_rank_not_in_profile_ranks(self, mock_get_rank, mock_trainer, mock_pl_module):
        """Test behavior when the current rank is not in the profile ranks."""
        mock_get_rank.return_value = 1
        callback = NsysCallback(start_step=10, end_step=20, ranks=[0])
        callback = NsysCallback(start_step=10, end_step=20, ranks=[0])

        callback.on_train_batch_start(mock_trainer, mock_pl_module, None, 10)
        callback.on_train_batch_end(mock_trainer, mock_pl_module, None, None, 20)

        # No profiling calls should be made

    @pytest.mark.parametrize(
        "start_step,end_step,batch_idx,expected_call",
        [
            (10, 20, 9, False),
            (10, 20, 10, True),
            (10, 20, 15, False),
            (10, 20, 20, False),
            (10, 20, 21, False),
        ],
    )
    @patch('nemo.lightning.pytorch.callbacks.nsys.get_rank')
    @patch('torch.cuda.cudart')
    @patch('torch.autograd.profiler.emit_nvtx')
    def test_profiling_range(
        self,
        mock_emit_nvtx,
        mock_cudart,
        mock_get_rank,
        start_step,
        end_step,
        batch_idx,
        expected_call,
        mock_trainer,
        mock_pl_module,
    ):
        """Test profiling behavior across different batch indices."""
        mock_get_rank.return_value = 0
        callback = NsysCallback(start_step=start_step, end_step=end_step, ranks=[0])

        mock_trainer.strategy.current_epoch_step = batch_idx
        callback.on_train_batch_start(mock_trainer, mock_pl_module, None, batch_idx)

        if expected_call:
            mock_cudart().cudaProfilerStart.assert_called_once()
            mock_emit_nvtx.assert_called_once()
        else:
            mock_cudart().cudaProfilerStart.assert_not_called()
            mock_emit_nvtx.assert_not_called()

    @patch('nemo.lightning.pytorch.callbacks.nsys.get_rank')
    @patch('torch.cuda.cudart')
    def test_single_profile_range(self, mock_cudart, mock_get_rank, mock_trainer, mock_pl_module):
        """Test behavior with a single profile range."""
        mock_get_rank.return_value = 0
        callback = NsysCallback(start_step=10, end_step=40, ranks=[0])

        # Ensure the device type is 'cuda'
        mock_trainer.strategy.root_device.type = 'cuda'

        # Start of range
        mock_trainer.strategy.current_epoch_step = 10
        callback.on_train_batch_start(mock_trainer, mock_pl_module, None, 10)
        assert mock_cudart().cudaProfilerStart.call_count == 1, "cudaProfilerStart was not called"

        # Middle of range
        mock_trainer.strategy.current_epoch_step = 25
        callback.on_train_batch_start(mock_trainer, mock_pl_module, None, 25)
        assert mock_cudart().cudaProfilerStart.call_count == 1, "cudaProfilerStart was called again"

        # End of range
        mock_trainer.strategy.current_epoch_step = 40
        callback.on_train_batch_end(mock_trainer, mock_pl_module, None, None, 40)
        assert mock_cudart().cudaProfilerStop.call_count == 1, "cudaProfilerStop was not called"
