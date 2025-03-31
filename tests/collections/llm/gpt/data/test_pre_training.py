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

from nemo.collections.llm.gpt.data.pre_training import (
    PreTrainingDataModule,
    is_number_tryexcept,
    is_zipped_list,
    validate_dataset_asset_accessibility,
)


# Helper function to create temporary dataset files
def create_temp_dataset_files(temp_dir, prefix="dataset"):
    path = os.path.join(temp_dir, prefix)
    # Create .bin and .idx files
    with open(f"{path}.bin", "w") as f:
        f.write("dummy data")
    with open(f"{path}.idx", "wb") as f:
        f.write(b"MMIDIDX\x00\x00")
    return path


class TestPreTrainingHelperFunctions:
    def test_is_number_tryexcept(self):
        assert is_number_tryexcept("123") == True
        assert is_number_tryexcept("12.3") == True
        assert is_number_tryexcept("-123") == True
        assert is_number_tryexcept("abc") == False
        assert is_number_tryexcept(None) == False
        assert is_number_tryexcept("") == False

    def test_is_zipped_list(self):
        assert is_zipped_list(["30", "path1", "70", "path2"]) == True
        assert is_zipped_list(["path1", "path2"]) == False
        assert is_zipped_list([]) == False

    def test_validate_dataset_asset_accessibility(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test valid single path
            valid_path = create_temp_dataset_files(temp_dir)
            validate_dataset_asset_accessibility(valid_path)

            # Test valid list of paths
            paths = [valid_path, create_temp_dataset_files(temp_dir, "dataset2")]
            validate_dataset_asset_accessibility(paths)

            # Test valid weighted paths
            weighted_paths = ["30", valid_path, "70", paths[1]]
            validate_dataset_asset_accessibility(weighted_paths)

            # Test valid dictionary paths
            dict_paths = {"train": [valid_path], "validation": [paths[1]], "test": [valid_path]}
            validate_dataset_asset_accessibility(dict_paths)

            # Test invalid path
            with pytest.raises(FileNotFoundError):
                validate_dataset_asset_accessibility("nonexistent_path")

            # Test invalid type
            with pytest.raises(ValueError):
                validate_dataset_asset_accessibility(123)


class TestPreTrainingDataModule:
    @pytest.fixture
    def temp_dataset_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def basic_datamodule(self, temp_dataset_dir):
        dataset_path = create_temp_dataset_files(temp_dataset_dir)
        return PreTrainingDataModule(
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

    def test_initialization_with_weighted_paths(self, temp_dataset_dir):
        path1 = create_temp_dataset_files(temp_dataset_dir, "dataset1")
        path2 = create_temp_dataset_files(temp_dataset_dir, "dataset2")

        datamodule = PreTrainingDataModule(
            paths=["30", path1, "70", path2],
            seq_length=128,
            micro_batch_size=4,
            global_batch_size=8,
        )
        assert "blend" in datamodule.build_kwargs

    def test_initialization_with_dict_paths(self, temp_dataset_dir):
        path1 = create_temp_dataset_files(temp_dataset_dir, "dataset1")
        path2 = create_temp_dataset_files(temp_dataset_dir, "dataset2")

        datamodule = PreTrainingDataModule(
            paths={"train": [path1], "validation": [path2], "test": [path1]},
            seq_length=128,
            micro_batch_size=4,
            global_batch_size=8,
        )
        assert "blend_per_split" in datamodule.build_kwargs

    def test_gpt_dataset_config(self, basic_datamodule):
        config = basic_datamodule.gpt_dataset_config
        assert config.sequence_length == 128
        assert config.random_seed == 1234  # default value
        assert config.reset_position_ids == False  # default value
        assert config.eod_mask_loss == False  # default value

    @pytest.mark.skip(reason="Requires full training environment setup")
    def test_build(self, basic_datamodule):
        # This test would require setting up a full training environment
        # Including trainer and other dependencies
        pass

    def test_data_sampler_initialization(self, basic_datamodule):
        assert basic_datamodule.data_sampler is not None
        assert basic_datamodule.data_sampler.seq_len == 128
        assert basic_datamodule.data_sampler.micro_batch_size == 4
        assert basic_datamodule.data_sampler.global_batch_size == 8

    def test_rampup_batch_size(self, temp_dataset_dir):
        dataset_path = create_temp_dataset_files(temp_dataset_dir)
        rampup_config = [4, 2, 1000]  # start_size, increment, samples
        datamodule = PreTrainingDataModule(
            paths=dataset_path,
            seq_length=128,
            micro_batch_size=4,
            global_batch_size=8,
            rampup_batch_size=rampup_config,
        )
        assert datamodule.data_sampler.rampup_batch_size == rampup_config

    def test_build_with_custom_samples(self, basic_datamodule):
        with pytest.raises(AssertionError, match="num_val_samples must be greater than"):
            basic_datamodule.num_train_samples = 10000
            basic_datamodule.num_val_samples = 1000
            basic_datamodule.num_test_samples = 1000
            basic_datamodule.build(
                trainer_max_steps=100,
                trainer_val_check_interval=10,
                trainer_limit_val_batches=1.0,
                trainer_limit_test_batches=1.0,
            )

    def test_create_dataloader(self, basic_datamodule, mocker):
        # Mock dataset
        mock_dataset = mocker.MagicMock()
        mock_dataset.collate_fn = None

        # Mock trainer
        mock_trainer = mocker.MagicMock()
        mock_trainer.global_step = 0
        basic_datamodule.trainer = mock_trainer

        dataloader = basic_datamodule._create_dataloader(dataset=mock_dataset, mode="train")
        assert dataloader is not None
        assert dataloader.mode == "train"

    @pytest.mark.parametrize(
        "paths,expected_error",
        [
            (None, ValueError),
            (123, ValueError),
            ("nonexistent_path.bin", FileNotFoundError),
            (["/path1", "/path2"], FileNotFoundError),
        ],
    )
    def test_validate_dataset_asset_accessibility_errors(self, paths, expected_error):
        with pytest.raises(expected_error):
            validate_dataset_asset_accessibility(paths)

    def test_build_pretraining_datamodule(self, basic_datamodule, mocker):
        # Mock torch.distributed
        mock_dist = mocker.patch('torch.distributed')
        mock_dist.is_initialized.return_value = False

        # Mock BlendedMegatronDatasetBuilder
        mock_builder = mocker.patch(
            'megatron.core.datasets.blended_megatron_dataset_builder.BlendedMegatronDatasetBuilder'
        )
        mock_builder_instance = mocker.MagicMock()
        mock_builder.return_value = mock_builder_instance
        # Set up the build method to return mock datasets
        mock_train_ds = mocker.MagicMock()
        mock_valid_ds = mocker.MagicMock()
        mock_test_ds = mocker.MagicMock()
        mock_builder_instance.build.return_value = (mock_train_ds, mock_valid_ds, mock_test_ds)

        # Test the build_pretraining_datamodule function
        from nemo.collections.llm.gpt.data.pre_training import build_pretraining_datamodule

        build_pretraining_datamodule(
            datamodule=basic_datamodule,
            trainer_max_steps=100,
            trainer_val_check_interval=10,
            trainer_limit_val_batches=1,
            trainer_limit_test_batches=1,
        )

        # Verify dist.init_process_group was called correctly
        mock_dist.init_process_group.assert_called_once_with(world_size=1, rank=0)

        # Verify BlendedMegatronDatasetBuilder was called with correct arguments
        mock_builder.assert_called_once()
        _, call_args, _ = mock_builder.mock_calls[0]
        assert call_args[0] == basic_datamodule.dataset_cls  # First arg should be dataset class
        assert isinstance(call_args[1], list)  # Second arg should be list of num samples
        assert len(call_args[1]) == 3  # Should have train, valid, test samples
        assert mock_builder_instance.build.called  # Verify build() was called
