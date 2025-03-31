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
import tempfile

import pytest
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from nemo.collections.llm.bert.data.pre_training import BERTPreTrainingDataModule


# Helper function to create temporary dataset files
def create_temp_dataset_files(temp_dir, prefix="dataset"):
    path = os.path.join(temp_dir, prefix)
    # Create .bin and .idx files
    with open(f"{path}.bin", "w") as f:
        f.write("dummy data")
    with open(f"{path}.idx", "wb") as f:
        f.write(b"MMIDIDX\x00\x00")
    return path


class TestBERTPreTrainingDataModule:
    @pytest.fixture
    def temp_dataset_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def basic_datamodule(self, temp_dataset_dir):
        dataset_path = create_temp_dataset_files(temp_dataset_dir)
        return BERTPreTrainingDataModule(
            paths=dataset_path,
            seq_length=128,
            micro_batch_size=4,
            global_batch_size=8,
        )

    def test_initialization(self, basic_datamodule):
        assert basic_datamodule.seq_length == 128
        assert basic_datamodule.micro_batch_size == 4
        assert basic_datamodule.global_batch_size == 8
        assert basic_datamodule.num_workers == 8  # default value
        assert basic_datamodule.split == "900,50,50"  # default value
        assert basic_datamodule.reset_position_ids == False
        assert basic_datamodule.reset_attention_mask == False
        assert basic_datamodule.eod_mask_loss == False

    def test_initialization_with_weighted_paths(self, temp_dataset_dir):
        path1 = create_temp_dataset_files(temp_dataset_dir, "dataset1")
        path2 = create_temp_dataset_files(temp_dataset_dir, "dataset2")

        datamodule = BERTPreTrainingDataModule(
            paths=["30", path1, "70", path2],
            seq_length=128,
            micro_batch_size=4,
            global_batch_size=8,
        )
        assert "blend" in datamodule.build_kwargs

    def test_initialization_with_dict_paths(self, temp_dataset_dir):
        path1 = create_temp_dataset_files(temp_dataset_dir, "dataset1")
        path2 = create_temp_dataset_files(temp_dataset_dir, "dataset2")

        datamodule = BERTPreTrainingDataModule(
            paths={"train": [path1], "validation": [path2], "test": [path1]},
            seq_length=128,
            micro_batch_size=4,
            global_batch_size=8,
        )
        assert "blend_per_split" in datamodule.build_kwargs

    def test_bert_dataset_config(self, basic_datamodule):
        config = basic_datamodule.bert_dataset_config
        assert config.sequence_length == 128
        assert config.random_seed == 1234  # default value
        assert config.classification_head == True
        assert config.masking_probability == 0.15
        assert config.short_sequence_probability == 0.10
        assert config.masking_max_ngram == 3
        assert config.masking_do_full_word == True
        assert config.masking_do_permutation == False

    def test_data_sampler_initialization(self, basic_datamodule):
        assert basic_datamodule.data_sampler is not None
        assert basic_datamodule.data_sampler.seq_len == 128
        assert basic_datamodule.data_sampler.micro_batch_size == 4
        assert basic_datamodule.data_sampler.global_batch_size == 8

    def test_rampup_batch_size(self, temp_dataset_dir):
        dataset_path = create_temp_dataset_files(temp_dataset_dir)
        rampup_config = [4, 2, 1000]  # start_size, increment, samples
        datamodule = BERTPreTrainingDataModule(
            paths=dataset_path,
            seq_length=128,
            micro_batch_size=4,
            global_batch_size=8,
            rampup_batch_size=rampup_config,
        )
        assert datamodule.data_sampler.rampup_batch_size == rampup_config

    def test_state_dict_and_load_state_dict(self, basic_datamodule, mocker):
        # Mock trainer
        mock_trainer = mocker.MagicMock()
        mock_trainer.global_step = 100
        basic_datamodule.trainer = mock_trainer
        basic_datamodule.init_global_step = 50

        # Mock update_num_microbatches function
        mock_update = mocker.patch('megatron.core.num_microbatches_calculator.update_num_microbatches')

        # Mock data_sampler methods
        basic_datamodule.data_sampler.compute_consumed_samples = mocker.MagicMock(return_value=1000)

        # Test state_dict
        state = basic_datamodule.state_dict()
        assert 'consumed_samples' in state
        assert state['consumed_samples'] == 1000

        # Test load_state_dict
        basic_datamodule.load_state_dict({'consumed_samples': 2000})

        # Verify data_sampler values were updated
        assert basic_datamodule.data_sampler.init_consumed_samples == 2000
        assert basic_datamodule.data_sampler.prev_consumed_samples == 2000
        assert basic_datamodule.data_sampler.if_first_step == 1

        # Verify update_num_microbatches was called correctly
        mock_update.assert_called_once_with(
            consumed_samples=2000,
            consistency_check=False,
        )

    def test_reconfigure_limit_batches(self, basic_datamodule, mocker):
        # Mock trainer and datasets
        mock_trainer = mocker.MagicMock()
        mock_trainer.limit_train_batches = 100
        mock_trainer.limit_val_batches = 0.1
        mock_trainer.num_sanity_val_steps = 2
        basic_datamodule.trainer = mock_trainer

        # Mock datasets with non-zero length
        mock_train_ds = mocker.MagicMock()
        mock_train_ds.__len__ = mocker.MagicMock(return_value=1000)
        mock_val_ds = mocker.MagicMock()
        mock_val_ds.__len__ = mocker.MagicMock(return_value=100)

        basic_datamodule._train_ds = mock_train_ds
        basic_datamodule._validation_ds = mock_val_ds

        # Mock get_num_microbatches to return a non-zero value
        mocker.patch('megatron.core.num_microbatches_calculator.get_num_microbatches', return_value=2)

        # Test reconfiguration
        basic_datamodule.reconfigure_limit_batches()

        # Verify the trainer's attributes were updated correctly
        assert basic_datamodule.trainer.limit_train_batches == 200  # 100 * 2 (num_microbatches)

        # Test with float limit_val_batches that would result in less than 1 batch
        mock_trainer.limit_val_batches = 0.001  # Very small value
        with pytest.raises(MisconfigurationException):
            basic_datamodule._reconfigure_limit_batches(0.001, mock_val_ds, 'val')
