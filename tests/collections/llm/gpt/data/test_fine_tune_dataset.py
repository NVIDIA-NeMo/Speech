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

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs


class TestFineTuningDataModule:
    @pytest.fixture
    def dataset_root(self, tmp_path):
        # Create temporary dataset files
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "training.jsonl").touch()
        (root / "validation.jsonl").touch()
        (root / "test.jsonl").touch()
        return root

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "test/tokenizer"
        return tokenizer

    @pytest.fixture
    def basic_datamodule(self, dataset_root, mock_tokenizer):
        return FineTuningDataModule(
            dataset_root=dataset_root, tokenizer=mock_tokenizer, micro_batch_size=4, global_batch_size=8
        )

    def test_init_default_values(self, dataset_root, mock_tokenizer):
        dm = FineTuningDataModule(dataset_root=dataset_root, tokenizer=mock_tokenizer)
        assert dm.seq_length == 2048
        assert dm.micro_batch_size == 4
        assert dm.global_batch_size == 8
        assert dm.seed == 1234
        assert dm.num_workers == 8
        assert dm.pin_memory is True
        assert dm.persistent_workers is False

    def test_validate_batch_size_for_packed_sequence(self, dataset_root, mock_tokenizer):
        # Should raise error when micro_batch_size > 1 with packed sequence
        packed_specs = PackedSequenceSpecs(packed_sequence_size=512)
        with pytest.raises(ValueError, match="Micro batch size should be 1"):
            FineTuningDataModule(
                dataset_root=dataset_root,
                tokenizer=mock_tokenizer,
                micro_batch_size=4,
                packed_sequence_specs=packed_specs,
            )

        # Should not raise error when micro_batch_size = 1
        dm = FineTuningDataModule(
            dataset_root=dataset_root, tokenizer=mock_tokenizer, micro_batch_size=1, packed_sequence_specs=packed_specs
        )
        assert dm.packed_sequence_size == 512

    def test_dataset_paths(self, basic_datamodule):
        assert basic_datamodule.train_path.name == "training.jsonl"
        assert basic_datamodule.validation_path.name == "validation.jsonl"
        assert basic_datamodule.test_path.name == "test.jsonl"

    @patch('nemo.collections.llm.gpt.data.fine_tuning.create_sft_dataset')
    def test_create_dataset(self, mock_create_dataset, basic_datamodule):
        basic_datamodule._create_dataset(basic_datamodule.train_path)
        mock_create_dataset.assert_called_once()

    @patch('nemo.collections.llm.gpt.data.fine_tuning.create_sft_dataset')
    @patch('nemo.collections.llm.gpt.data.fine_tuning.WrappedDataLoader')
    def test_train_dataloader(self, mock_dataloader, mock_create_dataset, basic_datamodule):
        # Mock trainer for setup
        basic_datamodule.trainer = MagicMock()
        basic_datamodule.trainer.max_steps = 100
        basic_datamodule.trainer.global_step = 0

        # Setup the datamodule
        basic_datamodule.setup('fit')

        # Test train_dataloader
        basic_datamodule.train_dataloader()
        mock_create_dataset.assert_called_once()
        mock_dataloader.assert_called_once()

    def test_state_dict_and_load_state_dict(self, basic_datamodule):
        mock_update_num_microbatches = MagicMock()

        with patch.dict(
            'sys.modules',
            {
                'megatron': MagicMock(),
                'megatron.core': MagicMock(),
                'megatron.core.num_microbatches_calculator': MagicMock(
                    update_num_microbatches=mock_update_num_microbatches
                ),
            },
        ):
            # Mock trainer
            basic_datamodule.trainer = MagicMock()
            basic_datamodule.trainer.global_step = 10
            basic_datamodule.init_global_step = 0

            # Setup
            basic_datamodule.setup('fit')

            # Test state_dict
            state = basic_datamodule.state_dict()
            assert 'consumed_samples' in state

            # Test load_state_dict
            basic_datamodule.load_state_dict(state)

            # Verify the state was loaded correctly
            assert basic_datamodule.data_sampler.init_consumed_samples == state['consumed_samples']
            assert basic_datamodule.data_sampler.prev_consumed_samples == state['consumed_samples']

            # Verify update_num_microbatches was called correctly
            mock_update_num_microbatches.assert_called_once_with(
                consumed_samples=state['consumed_samples'], consistency_check=False
            )

    @pytest.fixture
    def packed_files(self, tmp_path):
        # Create temporary packed sequence files
        pack_dir = tmp_path / "packed"
        pack_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy numpy file for packed training data
        train_path = pack_dir / "custom_train.npy"
        val_path = pack_dir / "custom_val.npy"

        # Create dummy data and save it
        dummy_data = np.zeros((100, 512), dtype=np.int64)
        np.save(str(train_path), dummy_data)  # Convert Path to string for np.save
        np.save(str(val_path), dummy_data)

        # Create dummy metadata file
        metadata_path = pack_dir / "custom_metadata.jsonl"
        with open(metadata_path, 'w') as f:
            f.write('{"total_sequences": 100}\n')

        return {
            'train': train_path.absolute(),  # Use absolute paths
            'val': val_path.absolute(),
            'metadata': metadata_path.absolute(),
        }

    def test_packed_sequence_paths(self, dataset_root, mock_tokenizer, packed_files):
        # Verify the files exist before creating the module
        assert packed_files['train'].exists(), f"Train file does not exist: {packed_files['train']}"
        assert packed_files['val'].exists(), f"Val file does not exist: {packed_files['val']}"
        assert packed_files['metadata'].exists(), f"Metadata file does not exist: {packed_files['metadata']}"

        packed_specs = PackedSequenceSpecs(
            packed_sequence_size=512,
            packed_train_data_path=str(packed_files['train']),  # Convert Path to string
            packed_val_data_path=str(packed_files['val']),
            packed_metadata_path=str(packed_files['metadata']),
        )

        dm = FineTuningDataModule(
            dataset_root=dataset_root, tokenizer=mock_tokenizer, micro_batch_size=1, packed_sequence_specs=packed_specs
        )

        # Compare paths using resolve() to handle any path differences
        assert dm.train_path_packed == packed_files['train'].resolve()
        assert dm.validation_path_packed == packed_files['val'].resolve()

    def test_tokenizer_initialization(self, dataset_root):
        from nemo.collections.common.tokenizers import TokenizerSpec

        mock_tokenizer = MagicMock(spec=TokenizerSpec)
        dm = FineTuningDataModule(dataset_root=dataset_root, tokenizer=mock_tokenizer)
        assert dm.tokenizer == mock_tokenizer

    def test_setup_with_max_train_samples(self, basic_datamodule):
        basic_datamodule.trainer = MagicMock()
        basic_datamodule.trainer.max_steps = 100
        basic_datamodule.setup('fit')
        expected_samples = int(math.ceil(basic_datamodule.global_batch_size * 100 * 1.005))
        assert basic_datamodule.max_train_samples == expected_samples

    def test_consumed_samples_calculation(self, basic_datamodule):
        basic_datamodule.trainer = MagicMock()
        basic_datamodule.trainer.global_step = 10
        basic_datamodule.init_global_step = 5
        basic_datamodule.setup('fit')
        state = basic_datamodule.state_dict()
        assert 'consumed_samples' in state
        # Should compute consumed samples for 5 steps (10 - 5)
        expected_samples = basic_datamodule.data_sampler.compute_consumed_samples(5)
        assert state['consumed_samples'] == expected_samples

    @pytest.mark.parametrize("is_test", [True, False])
    def test_create_dataset_with_different_modes(self, basic_datamodule, is_test):
        with patch('nemo.collections.llm.gpt.data.fine_tuning.create_sft_dataset') as mock_create:
            basic_datamodule._create_dataset(basic_datamodule.train_path, is_test=is_test)
            mock_create.assert_called_once()
            _, kwargs = mock_create.call_args
            assert kwargs['is_test'] == is_test

    def test_default_pack_path(self, dataset_root, mock_tokenizer):
        dm = FineTuningDataModule(dataset_root=dataset_root, tokenizer=mock_tokenizer)

        # Mock tokenizer model name extraction
        with patch.object(dm, '_extract_tokenizer_model_name', return_value='test-model'):
            default_path = dm.default_pack_path
            expected_path = Path(dataset_root) / "packed" / "test-model"
            assert default_path == expected_path
            assert default_path.exists()

    def test_pack_metadata_paths(self, dataset_root, mock_tokenizer):
        custom_metadata_path = Path("custom_metadata.jsonl")
        specs = PackedSequenceSpecs(packed_sequence_size=512, packed_metadata_path=custom_metadata_path)
        dm = FineTuningDataModule(
            dataset_root=dataset_root,
            tokenizer=mock_tokenizer,
            packed_sequence_specs=specs,
            micro_batch_size=1,
        )
        assert dm.pack_metadata == custom_metadata_path

        # Test without packed sequence
        dm_no_pack = FineTuningDataModule(dataset_root=dataset_root, tokenizer=mock_tokenizer)
        with pytest.raises(ValueError, match="pack_metadata invalid"):
            _ = dm_no_pack.pack_metadata

    def test_pad_cu_seqlens_property(self, dataset_root, mock_tokenizer):
        dm = FineTuningDataModule(dataset_root=dataset_root, tokenizer=mock_tokenizer)
        assert dm.pad_cu_seqlens is False
