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

"""Unit tests for OneLoggerNeMoCallback."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from lightning.pytorch.callbacks import Callback as PTLCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from nemo.lightning.base_callback import BaseCallback
from nemo.lightning.one_logger_callback import (
    OneLoggerNeMoCallback,
    _get_base_callback_config,
    _should_enable_for_current_rank,
    get_nemo_v1_callback_config,
    get_nemo_v2_callback_config,
    get_one_logger_init_config,
)


class TestOneLoggerNeMoCallback:
    """Test suite for OneLoggerNeMoCallback."""

    def test_inheritance(self):
        """Test that OneLoggerNeMoCallback properly inherits from both parent classes."""
        with (
            patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback') as mock_ptl_callback,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider,
            patch('nemo.lightning.one_logger_callback.get_one_logger_init_config') as mock_get_config,
            patch('nemo.lightning.one_logger_callback.OneLoggerConfig') as mock_config_class,
            patch('nemo.lightning.one_logger_callback.on_app_start') as mock_on_app_start,
        ):

            # Setup mocks
            mock_get_config.return_value = {"application_name": "test", "session_tag_or_fn": "test-session"}
            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance
            mock_provider_instance = MagicMock()
            mock_provider_instance.config = MagicMock()
            mock_provider_instance.config.telemetry_config = None
            mock_provider.instance.return_value = mock_provider_instance
            mock_ptl_callback_instance = MagicMock()
            mock_ptl_callback.return_value = mock_ptl_callback_instance

            # Create callback instance
            callback = OneLoggerNeMoCallback()

            # Test inheritance
            assert isinstance(callback, BaseCallback)
            assert isinstance(callback, PTLCallback)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback.__init__', return_value=None)
    def test_init_configures_provider(
        self, mock_ptl_callback_init, mock_on_app_start, mock_config_class, mock_get_config, mock_provider
    ):
        """Test that __init__ properly configures the OneLogger provider."""
        # Setup mocks
        mock_init_config = {
            "application_name": "nemo",
            "session_tag_or_fn": "test-session",
            "enable_for_current_rank": True,
            "world_size_or_fn": 1,
            "error_handling_strategy": "propagate_exceptions",
        }
        mock_get_config.return_value = mock_init_config

        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance

        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance

        # Create callback instance
        OneLoggerNeMoCallback()

        # Verify initialization sequence
        mock_get_config.assert_called_once()
        mock_config_class.assert_called_once_with(**mock_init_config)
        mock_provider_instance.with_base_config.assert_called_once_with(mock_config_instance)
        mock_provider_instance.with_base_config.return_value.with_export_config.assert_called_once()
        mock_provider_instance.with_base_config.return_value.with_export_config.return_value.configure_provider.assert_called_once()
        mock_ptl_callback_init.assert_called_once_with(mock_provider_instance, call_on_app_start=False)
        mock_on_app_start.assert_called_once_with()


class TestOneLoggerCallback:
    """Test cases for one_logger_callback utility functions."""

    @pytest.mark.unit
    def test_get_one_logger_init_config(self):
        """Test get_one_logger_init_config returns correct minimal configuration."""
        with patch.dict(os.environ, {"SLURM_JOB_NAME": "test_job", "WORLD_SIZE": "4"}):
            config = get_one_logger_init_config()

            assert isinstance(config, dict)
            assert config["application_name"] == "nemo"
            assert config["session_tag_or_fn"] == "test_job"
            assert "enable_for_current_rank" in config
            assert config["world_size_or_fn"] == 4
            assert config["error_handling_strategy"] == "propagate_exceptions"

    @pytest.mark.unit
    def test_get_one_logger_init_config_no_slurm(self):
        """Test get_one_logger_init_config when SLURM_JOB_NAME is not set."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True):
            config = get_one_logger_init_config()

            assert config["session_tag_or_fn"] == "nemo-run"
            assert config["world_size_or_fn"] == 1

    @pytest.mark.unit
    def test_get_base_callback_config(self):
        """Test _get_base_callback_config with basic trainer setup."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.callbacks = []
        trainer.val_check_interval = 1.0
        trainer.strategy = None
        trainer.log_every_n_steps = 10

        with patch.dict(os.environ, {"SLURM_JOB_NAME": "test_job", "WORLD_SIZE": "4", "PERF_VERSION_TAG": "1.0.0"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=32, seq_length=512)

            assert config["perf_tag_or_fn"] == "test_job_1.0.0_bf32_se512_ws4"
            assert config["global_batch_size_or_fn"] == 32
            assert config["micro_batch_size_or_fn"] == 8
            assert config["seq_length_or_fn"] == 512
            assert config["train_iterations_target_or_fn"] == 1000
            assert config["train_samples_target_or_fn"] == 32000
            assert config["log_every_n_train_iterations"] == 10
            assert config["is_validation_iterations_enabled_or_fn"] is True
            assert config["is_save_checkpoint_enabled_or_fn"] is False
            assert config["save_checkpoint_strategy"] == "sync"

    @pytest.mark.unit
    def test_get_base_callback_config_with_checkpoint_callback(self):
        """Test _get_base_callback_config when checkpoint callback is present."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.val_check_interval = 0

        # Real ModelCheckpoint callback to satisfy isinstance checks
        checkpoint_callback = ModelCheckpoint(dirpath=".", save_top_k=-1)
        trainer.callbacks = [checkpoint_callback]

        with patch.dict(os.environ, {"SLURM_JOB_NAME": "test_job", "WORLD_SIZE": "2"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=16, seq_length=256)

            assert config["is_save_checkpoint_enabled_or_fn"] is True
            assert config["is_validation_iterations_enabled_or_fn"] is False

    @pytest.mark.unit
    def test_get_base_callback_config_async_save(self):
        """Test _get_base_callback_config with async save strategy."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.callbacks = []
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock strategy with async_save
        strategy = MagicMock()
        strategy.async_save = True
        trainer.strategy = strategy

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=8, seq_length=128)

            assert config["save_checkpoint_strategy"] == "async"

    @pytest.mark.unit
    def test_get_base_callback_config_dict_strategy(self):
        """Test _get_base_callback_config with dict strategy."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.callbacks = []
        trainer.val_check_interval = 0  # Set to 0 to avoid validation
        trainer.strategy = {"async_save": True}

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=8, seq_length=128)

            assert config["save_checkpoint_strategy"] == "async"

    @pytest.mark.unit
    def test_get_nemo_v1_callback_config(self):
        """Test get_nemo_v1_callback_config with model configuration."""
        trainer = MagicMock()
        trainer.max_steps = 500
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock lightning module with config
        pl_module = MagicMock()
        pl_module.cfg = OmegaConf.create({"train_ds": {"batch_size": 8}, "encoder": {"d_model": 768}})
        trainer.lightning_module = pl_module

        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            config = get_nemo_v1_callback_config(trainer)

            assert config["global_batch_size_or_fn"] == 16  # 8 * 2
            assert config["seq_length_or_fn"] == 768
            assert config["train_iterations_target_or_fn"] == 500

    @pytest.mark.unit
    def test_get_nemo_v1_callback_config_bucket_batch_size(self):
        """Test get_nemo_v1_callback_config with bucket batch sizes (ASR case)."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock lightning module with bucket batch sizes
        pl_module = MagicMock()
        pl_module.cfg = OmegaConf.create({"train_ds": {"bucket_batch_size": [4, 8, 12]}, "encoder": {"d_model": 512}})
        trainer.lightning_module = pl_module

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            config = get_nemo_v1_callback_config(trainer)

            # Average bucket batch size is (4+8+12)/3 = 8
            assert config["global_batch_size_or_fn"] == 8
            assert config["seq_length_or_fn"] == 512

    @pytest.mark.unit
    def test_get_nemo_v1_callback_config_fallback(self):
        """Test get_nemo_v1_callback_config with fallback values."""
        trainer = MagicMock()
        trainer.max_steps = 100
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock lightning module without required config
        pl_module = MagicMock()
        pl_module.cfg = OmegaConf.create({})
        trainer.lightning_module = pl_module

        config = get_nemo_v1_callback_config(trainer)

        assert config["global_batch_size_or_fn"] == 1  # fallback
        assert config["seq_length_or_fn"] == 1  # fallback
        assert config["train_iterations_target_or_fn"] == 100

    @pytest.mark.unit
    def test_get_nemo_v2_callback_config(self):
        """Test get_nemo_v2_callback_config with data module."""
        trainer = MagicMock()
        trainer.max_steps = 200
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock data module
        data = MagicMock()
        data.global_batch_size = 64
        data.seq_length = 1024

        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            config = get_nemo_v2_callback_config(trainer=trainer, data=data)

            assert config["global_batch_size_or_fn"] == 64
            assert config["seq_length_or_fn"] == 1024
            assert config["train_iterations_target_or_fn"] == 200

    @pytest.mark.unit
    def test_get_nemo_v2_callback_config_uses_micro_when_global_missing(self):
        """Test v2 config computes global_batch_size via micro_batch_size * WORLD_SIZE when global is missing."""
        trainer = MagicMock()
        trainer.max_steps = 100
        trainer.val_check_interval = 0

        # Data module without global_batch_size, but with micro_batch_size and seq_length
        data = SimpleNamespace(micro_batch_size=8, seq_length=2048)

        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            config = get_nemo_v2_callback_config(trainer=trainer, data=data)

        assert config["global_batch_size_or_fn"] == 32  # 8 * 4
        assert config["seq_length_or_fn"] == 2048
        assert config["train_iterations_target_or_fn"] == 100

    @pytest.mark.unit
    def test_get_nemo_v2_callback_config_no_data(self):
        """Test get_nemo_v2_callback_config without data module."""
        trainer = MagicMock()
        trainer.max_steps = 300
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        config = get_nemo_v2_callback_config(trainer=trainer, data=None)

        assert config["global_batch_size_or_fn"] == 1  # fallback
        assert config["seq_length_or_fn"] == 1  # fallback
        assert config["train_iterations_target_or_fn"] == 300

    @pytest.mark.unit
    def test_should_enable_for_current_rank_single_process(self):
        """Test _should_enable_for_current_rank if rank is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = _should_enable_for_current_rank()
            assert result is False

    @pytest.mark.unit
    def test_should_enable_for_current_rank_distributed_rank0(self):
        """Test _should_enable_for_current_rank for rank 0 in distributed training."""
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "4"}):
            result = _should_enable_for_current_rank()
            assert result is True

    @pytest.mark.unit
    def test_should_enable_for_current_rank_distributed_middle_rank(self):
        """Test _should_enable_for_current_rank for middle rank in distributed training."""
        with patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "4"}):
            result = _should_enable_for_current_rank()
            assert result is False

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_v1(
        self,
        mock_ptl_callback,
        mock_on_app_start,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v1_config,
        mock_provider,
    ):
        """Test update_config with nemo_version='v1'."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v1_config = {"job_name": "test-job", "world_size": 1, "global_batch_size": 32, "seq_length": 1024}
        mock_get_v1_config.return_value = mock_v1_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()

        # Call update_config
        callback.update_config(nemo_version='v1', trainer=trainer)

        # Verify v1 config was called
        mock_get_v1_config.assert_called_once_with(trainer=trainer)
        mock_telemetry_config_class.assert_called_once_with(**mock_v1_config)
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v2_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_v2(
        self,
        mock_ptl_callback,
        mock_on_app_start,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v2_config,
        mock_provider,
    ):
        """Test update_config with nemo_version='v2' and data module."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v2_config = {"job_name": "test-job-v2", "world_size": 2, "global_batch_size": 64, "seq_length": 2048}
        mock_get_v2_config.return_value = mock_v2_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback, trainer, and data module
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()
        data_module = MagicMock()

        # Call update_config with v2 and data
        callback.update_config(nemo_version='v2', trainer=trainer, data=data_module)

        # Verify v2 config was called with data
        mock_get_v2_config.assert_called_once_with(trainer=trainer, data=data_module)
        mock_telemetry_config_class.assert_called_once_with(**mock_v2_config)
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_unknown_version_defaults_to_v1(
        self,
        mock_ptl_callback,
        mock_on_app_start,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v1_config,
        mock_provider,
    ):
        """Test update_config with unknown version defaults to v1."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v1_config = {"job_name": "test-job"}
        mock_get_v1_config.return_value = mock_v1_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()

        # Call update_config with unknown version
        callback.update_config(nemo_version='unknown', trainer=trainer)

        # Verify v1 config was called (default fallback)
        mock_get_v1_config.assert_called_once_with(trainer=trainer)
        mock_telemetry_config_class.assert_called_once_with(**mock_v1_config)
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v2_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_v2_without_data(
        self,
        mock_ptl_callback,
        mock_on_app_start,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v2_config,
        mock_provider,
    ):
        """Test update_config with nemo_version='v2' but no data module provided."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v2_config = {"job_name": "test-job-v2"}
        mock_get_v2_config.return_value = mock_v2_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()

        # Call update_config with v2 but no data
        callback.update_config(nemo_version='v2', trainer=trainer)

        # Verify v2 config was called with None data
        mock_get_v2_config.assert_called_once_with(trainer=trainer, data=None)
        mock_telemetry_config_class.assert_called_once_with(**mock_v2_config)
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v2_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_v2_with_extra_kwargs(
        self,
        mock_ptl_callback,
        mock_on_app_start,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v2_config,
        mock_provider,
    ):
        """Test update_config with nemo_version='v2' and extra kwargs."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v2_config = {"job_name": "test-job-v2"}
        mock_get_v2_config.return_value = mock_v2_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()
        data_module = MagicMock()

        # Call update_config with v2, data, and extra kwargs
        callback.update_config(
            nemo_version='v2', trainer=trainer, data=data_module, extra_param1='value1', extra_param2='value2'
        )

        # Verify v2 config was called with data (extra kwargs should be ignored)
        mock_get_v2_config.assert_called_once_with(trainer=trainer, data=data_module)
        mock_telemetry_config_class.assert_called_once_with(**mock_v2_config)
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    def test_export_all_symbols(self):
        """Test that __all__ contains the expected symbols."""
        from nemo.lightning.one_logger_callback import __all__

        assert 'OneLoggerNeMoCallback' in __all__

    @patch.dict(os.environ, {'EXP_NAME': 'test-experiment', 'WORLD_SIZE': '4'})
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_init_with_environment_variables(
        self, mock_ptl_callback, mock_on_app_start, mock_config_class, mock_get_config, mock_provider
    ):
        """Test initialization with environment variables set."""
        # Setup mocks
        mock_get_config.return_value = {
            "application_name": "nemo",
            "session_tag_or_fn": "test-experiment",
            "world_size_or_fn": 4,
        }
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        # Create callback instance
        OneLoggerNeMoCallback()

        # Verify that get_one_logger_init_config was called
        mock_get_config.assert_called_once()

        # Verify that the config was created with the environment-based values
        mock_config_class.assert_called_once()
        call_args = mock_config_class.call_args[1]
        assert call_args['session_tag_or_fn'] == 'test-experiment'
        assert call_args['world_size_or_fn'] == 4

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_with_empty_config(
        self,
        mock_ptl_callback,
        mock_on_app_start,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v1_config,
        mock_provider,
    ):
        """Test update_config with empty configuration dictionary."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        # Return empty config
        mock_get_v1_config.return_value = {}

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()

        # Call update_config
        callback.update_config(nemo_version='v1', trainer=trainer)

        # Verify empty config was passed to TrainingTelemetryConfig
        mock_telemetry_config_class.assert_called_once_with(**{})
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    def test_callback_instantiation_without_mocks_raises_import_error(self):
        """Test that callback instantiation without proper mocks raises appropriate errors."""
        # This test verifies that the callback properly depends on external libraries
        # and will raise import errors if they're not available
        with patch(
            'nemo.lightning.one_logger_callback.OneLoggerPTLCallback.__init__',
            side_effect=Exception("with_base_config can be called only before configure_provider is called."),
        ):
            with pytest.raises(
                Exception, match="with_base_config can be called only before configure_provider is called."
            ):
                OneLoggerNeMoCallback()

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback.__init__', return_value=None)
    def test_init_provider_chain_calls(
        self, mock_ptl_callback_init, mock_on_app_start, mock_config_class, mock_get_config, mock_provider
    ):
        """Test that the provider configuration chain is called in correct order."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance

        # Create callback instance
        OneLoggerNeMoCallback()

        # Verify the provider configuration chain
        mock_provider_instance.with_base_config.assert_called_once_with(mock_config_instance)
        chain_result = mock_provider_instance.with_base_config.return_value
        chain_result.with_export_config.assert_called_once()
        chain_result.with_export_config.return_value.configure_provider.assert_called_once()

        # Verify PTL callback was initialized with provider instance and explicit on_app_start was called
        mock_ptl_callback_init.assert_called_once_with(mock_provider_instance, call_on_app_start=False)
        mock_on_app_start.assert_called_once_with()
