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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def trainer():
    return MagicMock()


@patch(
    'nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset.GPTSFTDataset.__init__', return_value=None
)
def test_finetuning_module(mock_gpt_sft_dataset, trainer) -> None:
    from nemo.collections.llm.gpt.data import FineTuningDataModule

    dataset_root = 'random_root'
    datamodule = FineTuningDataModule(
        dataset_root,
        seq_length=2048,
        micro_batch_size=4,
        global_batch_size=8,
        seed=1234,
    )
    datamodule.trainer = trainer
    datamodule.setup(stage='train')

    datamodule.train_dataloader()
    mock_gpt_sft_dataset.assert_called_once()


@patch(
    'nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset.GPTSFTDataset.__init__', return_value=None
)
def test_dolly_module(mock_gpt_sft_dataset, trainer) -> None:
    from nemo.collections.llm.gpt.data import DollyDataModule

    datamodule = DollyDataModule(
        seq_length=2048,
        micro_batch_size=4,
        global_batch_size=8,
        seed=1234,
    )
    datamodule.trainer = trainer
    datamodule.setup(stage='train')

    datamodule.train_dataloader()
    mock_gpt_sft_dataset.assert_called_once()


@patch(
    'nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset.GPTSFTDataset.__init__', return_value=None
)
def test_squad_module(mock_gpt_sft_dataset, trainer) -> None:
    from nemo.collections.llm.gpt.data import SquadDataModule

    datamodule = SquadDataModule(
        seq_length=2048,
        micro_batch_size=4,
        global_batch_size=8,
        seed=1234,
    )
    datamodule.trainer = trainer
    datamodule.setup(stage='train')

    datamodule.train_dataloader()
    mock_gpt_sft_dataset.assert_called_once()


# TODO @chcui fix test for pretrain data module
# @patch('megatron.core.datasets.blended_megatron_dataset_builder.BlendedMegatronDatasetBuilder')
# @patch('nemo.lightning.pytorch.trainer.Trainer')
# def test_pretraining_module(mock_pretraining_dataset_builder, mock_trainer) -> None:
#     from nemo.collections.llm.gpt.data import PreTrainingDataModule
#
#     datamodule = PreTrainingDataModule(
#         path=Path('random_path'),
#         seq_length=2048,
#         micro_batch_size=4,
#         global_batch_size=8,
#         seed=1234,
#     )
#     mock_trainer.max_steps = 100
#     mock_trainer.val_check_interval = 5
#     mock_trainer.limit_val_batches = 10
#     mock_trainer.limit_test_batches = 10
#     datamodule.trainer = mock_trainer
#
#     datamodule.setup()
#     datamodule.train_dataloader()
#     mock_pretraining_dataset_builder.assert_called_once()
