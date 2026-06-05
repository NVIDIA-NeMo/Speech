# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Unit tests for MagpieTTSModel dataloader setup methods.

Tests for setup_multiple_validation_data (lhotse — datasets is a list):
1. Single lhotse dataset entry
2. Multiple lhotse dataset entries with default/custom names

Tests for setup_multiple_validation_data (non-lhotse — datasets is a dict):
3. Single dataset_meta in datasets dict
4. Multiple dataset_meta entries in datasets dict
5. Error cases: missing or empty 'datasets' key

Tests for setup_training_data (non-lhotse — datasets is a dict):
6. Single dataset_meta with num_workers=0 (persistent_workers=False, inline tokenizer setup)
7. Multiple dataset_meta with num_workers>0 (persistent_workers=True, deferred tokenizer setup)

Tests for setup_training_data (lhotse):
8. Single lhotse input_cfg: sample_rate injection, get_lhotse_dataloader dispatch
9. Multiple lhotse input_cfg entries: weighted multi-source config passes through
"""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from nemo.collections.tts.models.magpietts import MagpieTTSModel


class TestSetupMultipleValidationData:
    """Test cases for MagpieTTSModel.setup_multiple_validation_data method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock MagpieTTSModel instance with required methods mocked."""
        model = MagicMock(spec=MagpieTTSModel)
        model._update_dataset_config = MagicMock()
        model._setup_test_dataloader = MagicMock(return_value=MagicMock())
        return model

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_single_dataset_merges_shared_config(self, mock_model):
        """Single dataset entry merges shared config, strips 'name', and stores dataloader as a list."""
        config = OmegaConf.create(
            {
                'use_lhotse': True,
                'batch_duration': 100,  # Shared config
                'num_workers': 2,  # Shared config
                'datasets': [
                    {
                        'name': 'custom_single_val',
                        'batch_duration': 50,  # Override shared
                        'input_cfg': [{'type': 'lhotse_shar', 'shar_path': '/path/to/data'}],
                    }
                ],
            }
        )

        MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        # Single dataset goes through the same unified loop as multi-dataset
        mock_model._setup_test_dataloader.assert_called_once()
        passed_config = mock_model._setup_test_dataloader.call_args[0][0]
        assert passed_config.batch_duration == 50  # Dataset override wins
        assert passed_config.num_workers == 2  # Shared config preserved
        assert 'name' not in passed_config  # 'name' stripped before dataloader setup
        assert isinstance(mock_model._validation_dl, list)
        assert len(mock_model._validation_dl) == 1
        assert mock_model._validation_names == ['custom_single_val']

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_multiple_datasets_with_default_and_custom_names(self, mock_model):
        """Multiple dataset entries create separate dataloaders and assign default names when unspecified."""
        config = OmegaConf.create(
            {
                'use_lhotse': True,
                'batch_duration': 100,
                'datasets': [
                    {'input_cfg': [{'type': 'lhotse_shar', 'shar_path': '/path/to/data0'}]},  # No name
                    {'name': 'custom_name', 'input_cfg': [{'type': 'lhotse_shar', 'shar_path': '/path/to/data1'}]},
                ],
            }
        )

        MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        # Should call _setup_test_dataloader twice (once per dataset)
        assert mock_model._setup_test_dataloader.call_count == 2
        assert isinstance(mock_model._validation_dl, list)
        assert len(mock_model._validation_dl) == 2
        # First dataset gets default name, second uses explicit name
        assert mock_model._validation_names == ['val_set_0', 'custom_name']

    # ==================== Non-Lhotse (NeMo Manifest) Tests ====================

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_non_lhotse_datasets_dict_creates_single_dataloader(self, mock_model):
        """Non-lhotse: datasets as dict creates a single dataloader, name derived from dataset_meta key."""
        config = OmegaConf.create(
            {
                'datasets': {
                    '_target_': 'nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDataset',
                    'dataset_meta': {'an4': {'manifest_path': '/data/val.json', 'audio_dir': '/'}},
                    'min_duration': 0.2,
                    'max_duration': 20.0,
                },
                'dataloader_params': {'batch_size': 4, 'num_workers': 0, 'pin_memory': True},
            }
        )

        MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        mock_model._setup_test_dataloader.assert_called_once()
        passed_config = mock_model._setup_test_dataloader.call_args[0][0]
        assert passed_config.datasets._target_ == 'nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDataset'
        assert passed_config.datasets.dataset_meta.an4.manifest_path == '/data/val.json'
        assert passed_config.dataloader_params.batch_size == 4
        assert mock_model._validation_names == ['an4']
        assert len(mock_model._validation_dl) == 1

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_non_lhotse_datasets_dict_with_multiple_dataset_meta(self, mock_model):
        """Non-lhotse: datasets dict with multiple dataset_meta entries, name joined with '+'."""
        config = OmegaConf.create(
            {
                'datasets': {
                    '_target_': 'nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDataset',
                    'dataset_meta': {
                        'en': {'manifest_path': '/data/en_val.json', 'audio_dir': '/'},
                        'es': {'manifest_path': '/data/es_val.json', 'audio_dir': '/'},
                    },
                    'min_duration': 0.2,
                    'max_duration': 20.0,
                },
                'dataloader_params': {'batch_size': 8, 'num_workers': 2, 'pin_memory': True},
            }
        )

        MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        mock_model._setup_test_dataloader.assert_called_once()
        passed_config = mock_model._setup_test_dataloader.call_args[0][0]
        assert passed_config.datasets.dataset_meta.en.manifest_path == '/data/en_val.json'
        assert passed_config.datasets.dataset_meta.es.manifest_path == '/data/es_val.json'
        assert passed_config.dataloader_params.batch_size == 8
        assert mock_model._validation_names == ['en+es']
        assert len(mock_model._validation_dl) == 1

    # ==================== Error Case Tests ====================

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_missing_datasets_key_raises_value_error(self, mock_model):
        """Config without 'datasets' key raises ValueError."""
        config = OmegaConf.create({'use_lhotse': True, 'batch_duration': 100})

        with pytest.raises(ValueError) as exc_info:
            MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        assert "datasets" in str(exc_info.value).lower()

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_empty_datasets_list_raises_value_error(self, mock_model):
        """Empty 'datasets' list raises ValueError."""
        config = OmegaConf.create({'use_lhotse': True, 'datasets': []})

        with pytest.raises(ValueError) as exc_info:
            MagpieTTSModel.setup_multiple_validation_data(mock_model, config)

        assert "non-empty list" in str(exc_info.value).lower()


class TestSetupTrainingData:
    """Test cases for MagpieTTSModel.setup_training_data method (lhotse and non-lhotse paths)."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock MagpieTTSModel with get_dataset and get_lhotse_dataloader mocked."""
        model = MagicMock(spec=MagpieTTSModel)
        mock_dataset = MagicMock()
        mock_dataset.get_sampler.return_value = MagicMock()
        mock_dataset.collate_fn = MagicMock()
        model.get_dataset.return_value = mock_dataset
        model.get_lhotse_dataloader.return_value = MagicMock()
        model.sample_rate = 22050
        model.trainer.world_size = 1
        return model

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @patch('torch.utils.data.DataLoader', return_value=MagicMock())
    @patch('nemo.collections.tts.models.magpietts.setup_tokenizers', return_value=MagicMock())
    def test_single_non_lhotse_train_dataset_num_workers_zero(
        self, mock_setup_tokenizers, mock_dataloader_cls, mock_model
    ):
        """Single dataset_meta, num_workers=0: tokenizer set up inline, persistent_workers=False."""
        config = OmegaConf.create(
            {
                'datasets': {
                    '_target_': 'nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDataset',
                    'dataset_meta': {'an4': {'manifest_path': '/data/train.json', 'audio_dir': '/'}},
                    'min_duration': 0.2,
                    'max_duration': 20.0,
                },
                'dataloader_params': {'batch_size': 4, 'num_workers': 0, 'pin_memory': True, 'drop_last': True},
            }
        )

        MagpieTTSModel.setup_training_data(mock_model, config)

        mock_model.get_dataset.assert_called_once_with(config, dataset_type='train')
        mock_dataset = mock_model.get_dataset.return_value
        mock_dataset.get_sampler.assert_called_once_with(4, world_size=1)

        mock_setup_tokenizers.assert_called_once()
        assert mock_dataset.text_tokenizer == mock_setup_tokenizers.return_value

        mock_dataloader_cls.assert_called_once()
        dl_call_kwargs = mock_dataloader_cls.call_args
        assert dl_call_kwargs.kwargs['persistent_workers'] is False
        assert dl_call_kwargs.kwargs['batch_size'] == 4
        assert dl_call_kwargs.kwargs['num_workers'] == 0

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @patch('torch.utils.data.DataLoader', return_value=MagicMock())
    @patch('nemo.collections.tts.models.magpietts.setup_tokenizers', return_value=MagicMock())
    def test_multiple_non_lhotse_train_datasets_num_workers_positive(
        self, mock_setup_tokenizers, mock_dataloader_cls, mock_model
    ):
        """Multiple dataset_meta entries, num_workers>0: tokenizer deferred to worker_init_fn, persistent_workers=True."""
        config = OmegaConf.create(
            {
                'datasets': {
                    '_target_': 'nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDataset',
                    'dataset_meta': {
                        'en': {'manifest_path': '/data/en_train.json', 'audio_dir': '/'},
                        'es': {'manifest_path': '/data/es_train.json', 'audio_dir': '/'},
                    },
                    'weighted_sampling_steps_per_epoch': 1000,
                    'min_duration': 0.2,
                    'max_duration': 20.0,
                },
                'dataloader_params': {'batch_size': 16, 'num_workers': 4, 'pin_memory': True, 'drop_last': True},
            }
        )

        MagpieTTSModel.setup_training_data(mock_model, config)

        mock_model.get_dataset.assert_called_once_with(config, dataset_type='train')
        mock_dataset = mock_model.get_dataset.return_value
        mock_dataset.get_sampler.assert_called_once_with(16, world_size=1)

        mock_setup_tokenizers.assert_not_called()

        mock_dataloader_cls.assert_called_once()
        dl_call_kwargs = mock_dataloader_cls.call_args
        assert dl_call_kwargs.kwargs['persistent_workers'] is True
        assert dl_call_kwargs.kwargs['batch_size'] == 16
        assert dl_call_kwargs.kwargs['num_workers'] == 4

    # ==================== Lhotse Tests ====================

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_single_lhotse_train_dataset(self, mock_model):
        """Single lhotse input_cfg: sample_rate injected into config and get_lhotse_dataloader called."""
        config = OmegaConf.create(
            {
                'use_lhotse': True,
                'volume_norm': True,
                'min_duration': 0.2,
                'batch_duration': 100,
                'num_workers': 4,
                'input_cfg': [
                    {
                        'type': 'lhotse_shar',
                        'shar_path': '/path/to/data',
                        'weight': 1.0,
                        'tags': {'tokenizer_names': ['english_phoneme']},
                    }
                ],
            }
        )

        MagpieTTSModel.setup_training_data(mock_model, config)

        mock_model.get_lhotse_dataloader.assert_called_once()
        passed_config = mock_model.get_lhotse_dataloader.call_args.args[0]
        assert passed_config.sample_rate == 22050
        assert passed_config.use_lhotse is True
        assert len(passed_config.input_cfg) == 1
        assert passed_config.input_cfg[0].shar_path == '/path/to/data'
        assert mock_model.get_lhotse_dataloader.call_args.kwargs['mode'] == 'train'
        mock_model.get_dataset.assert_not_called()

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_multiple_lhotse_train_datasets(self, mock_model):
        """Multiple lhotse input_cfg entries: all data sources passed through with sample_rate injected."""
        config = OmegaConf.create(
            {
                'use_lhotse': True,
                'volume_norm': True,
                'min_duration': 0.2,
                'batch_duration': 200,
                'num_workers': 6,
                'input_cfg': [
                    {
                        'type': 'lhotse_shar',
                        'shar_path': '/path/to/en_data',
                        'weight': 0.7,
                        'tags': {'tokenizer_names': ['english_phoneme']},
                    },
                    {
                        'type': 'lhotse_shar',
                        'shar_path': '/path/to/es_data',
                        'weight': 0.3,
                        'tags': {'tokenizer_names': ['spanish_phoneme']},
                    },
                ],
            }
        )

        MagpieTTSModel.setup_training_data(mock_model, config)

        mock_model.get_lhotse_dataloader.assert_called_once()
        passed_config = mock_model.get_lhotse_dataloader.call_args.args[0]
        assert passed_config.sample_rate == 22050
        assert len(passed_config.input_cfg) == 2
        assert passed_config.input_cfg[0].shar_path == '/path/to/en_data'
        assert passed_config.input_cfg[0].weight == 0.7
        assert passed_config.input_cfg[1].shar_path == '/path/to/es_data'
        assert passed_config.input_cfg[1].weight == 0.3
        assert mock_model.get_lhotse_dataloader.call_args.kwargs['mode'] == 'train'
        mock_model.get_dataset.assert_not_called()
