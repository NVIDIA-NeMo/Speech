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
from unittest.mock import MagicMock, patch

import pytest

from nemo.collections.llm.bert.data.fine_tuning import FineTuningDataModule


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
        assert dm.dataset_kwargs == {}

    def test_dataset_paths(self, basic_datamodule):
        assert basic_datamodule.train_path.name == "training.jsonl"
        assert basic_datamodule.validation_path.name == "validation.jsonl"
        assert basic_datamodule.test_path.name == "test.jsonl"

    @patch('nemo.collections.llm.bert.data.core.create_sft_dataset')
    def test_create_dataset(self, mock_create_dataset, basic_datamodule):
        basic_datamodule._create_dataset(basic_datamodule.train_path)
        mock_create_dataset.assert_called_once()

    @patch('nemo.collections.llm.bert.data.core.create_sft_dataset')
    @patch('nemo.lightning.data.WrappedDataLoader')
    def test_train_dataloader(self, mock_dataloader, mock_create_dataset, basic_datamodule):
        # Mock trainer for setup
        basic_datamodule.trainer = MagicMock()
        basic_datamodule.trainer.max_steps = 100
        basic_datamodule.trainer.global_step = 0

        # Setup the datamodule
        basic_datamodule.setup('fit')

        # Test train_dataloader
        basic_datamodule.train_dataloader()

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

    def test_setup_with_max_train_samples(self, basic_datamodule):
        basic_datamodule.trainer = MagicMock()
        basic_datamodule.trainer.max_steps = 100
        basic_datamodule.setup('fit')
        expected_samples = int(math.ceil(basic_datamodule.global_batch_size * 100 * 1.005))
        assert basic_datamodule.max_train_samples == expected_samples

    def test_consumed_samples_calculation(self, basic_datamodule):
        basic_datamodule.trainer = MagicMock()
        basic_datamodule.trainer.global_step = 10
        basic_datamodule.data_sampler = MagicMock()
        basic_datamodule.data_sampler.init_global_step = 5
        basic_datamodule.setup('fit')
        state = basic_datamodule.state_dict()
        assert 'consumed_samples' in state

    def test_tokenizer_initialization(self, dataset_root):
        from nemo.collections.common.tokenizers import TokenizerSpec

        mock_tokenizer = MagicMock(spec=TokenizerSpec)
        dm = FineTuningDataModule(dataset_root=dataset_root, tokenizer=mock_tokenizer)
        assert dm.tokenizer == mock_tokenizer

    @patch('nemo.collections.llm.bert.data.core.create_sft_dataset')
    def test_val_dataloader(self, mock_create_dataset, basic_datamodule):
        basic_datamodule.trainer = MagicMock()
        basic_datamodule.trainer.max_steps = 100
        basic_datamodule.setup('fit')
        basic_datamodule.val_dataloader()
        mock_create_dataset.assert_called_once()

    @patch('nemo.collections.llm.bert.data.core.create_sft_dataset')
    def test_test_dataloader(self, mock_create_dataset, basic_datamodule):
        basic_datamodule.trainer = MagicMock()
        basic_datamodule.trainer.max_steps = 100
        basic_datamodule.setup('fit')
        basic_datamodule.test_dataloader()
        mock_create_dataset.assert_called_once()

    def test_extract_tokenizer_model_name(self, basic_datamodule):
        from nemo.collections.common.tokenizers import AutoTokenizer

        # Test case 1: AutoTokenizer with context/nemo_tokenizer path
        mock_auto_tokenizer = MagicMock(spec=AutoTokenizer)
        # Create the nested tokenizer attribute
        mock_auto_tokenizer.configure_mock(
            **{'tokenizer': MagicMock(name_or_path="NEMO_HOME/huggingface/bert-base/context/nemo_tokenizer")}
        )
        basic_datamodule.tokenizer = mock_auto_tokenizer
        assert basic_datamodule._extract_tokenizer_model_name() == "huggingface--bert-base"

        # Test case 2: AutoTokenizer with nemo_tokenizer path
        mock_auto_tokenizer.tokenizer.name_or_path = "NEMO_HOME/roberta/base/nemo_tokenizer"
        basic_datamodule.tokenizer = mock_auto_tokenizer
        assert basic_datamodule._extract_tokenizer_model_name() == "roberta--base"

        # Test case 3: Regular HuggingFace model path
        mock_auto_tokenizer.tokenizer.name_or_path = "bert-base-uncased"
        basic_datamodule.tokenizer = mock_auto_tokenizer
        assert basic_datamodule._extract_tokenizer_model_name() == "bert-base-uncased"

        # Test case 4: HuggingFace organization/model path
        mock_auto_tokenizer.tokenizer.name_or_path = "google/bert-base-uncased"
        basic_datamodule.tokenizer = mock_auto_tokenizer
        assert basic_datamodule._extract_tokenizer_model_name() == "google--bert-base-uncased"

        # Test case 5: Non-AutoTokenizer
        custom_tokenizer = MagicMock()
        basic_datamodule.tokenizer = custom_tokenizer
        result = basic_datamodule._extract_tokenizer_model_name()
        assert result.startswith("unknown_tokenizer_")
        assert str(hash(custom_tokenizer)) in result

        # Helper method for creating mock tokenizer with specific path
        def create_mock_tokenizer(path):
            mock = MagicMock(spec=AutoTokenizer)
            mock.configure_mock(**{'tokenizer': MagicMock(name_or_path=path)})
            return mock

        # Test case 6: Deep nested path with context/nemo_tokenizer
        basic_datamodule.tokenizer = create_mock_tokenizer("NEMO_HOME/org/team/model/version/context/nemo_tokenizer")
        assert basic_datamodule._extract_tokenizer_model_name() == "model--version"

        # Test case 7: Path with special characters
        basic_datamodule.tokenizer = create_mock_tokenizer("NEMO_HOME/org-name/model-name/context/nemo_tokenizer")
        assert basic_datamodule._extract_tokenizer_model_name() == "org-name--model-name"

        # Test case 8: Empty or None path
        basic_datamodule.tokenizer = create_mock_tokenizer("")
        assert basic_datamodule._extract_tokenizer_model_name() == ""

        # Test case 9: Path with no organization
        basic_datamodule.tokenizer = create_mock_tokenizer("bert-base")
        assert basic_datamodule._extract_tokenizer_model_name() == "bert-base"

        # Test case 10: Multiple slashes in path
        basic_datamodule.tokenizer = create_mock_tokenizer("org/team/model/v1")
        assert basic_datamodule._extract_tokenizer_model_name() == "org--team--model--v1"
