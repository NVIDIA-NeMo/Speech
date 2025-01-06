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

import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm.bert.data.core import create_sft_dataset
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec


class FineTuningDataModule(pl.LightningDataModule):
    """Base class for fine-tuning an Bert.

    This class provides a foundation for building custom data modules for fine-tuning Nemo NLP models. It inherits from
    `pl.LightningDataModule` from the PyTorch Lightning library and handles data loading, preprocessing, and batch
    creation for training, validation, and testing.

    Args:
        dataset_root (Union[str, Path]): The root directory containing the training, validation, and test data.
        seq_length (int, optional): The maximum sequence length for the input and output text. Defaults to 2048.
        tokenizer (Optional[TokenizerSpec], optional): The tokenizer to use for preprocessing the text.
            If not provided, a Megatron GPT2 BPE tokenizer will be used.
        micro_batch_size (int, optional): The micro batch size for training. Defaults to 4.
        global_batch_size (int, optional): The global batch size for training. Defaults to 8.
        rampup_batch_size (Optional[List[int]], optional): A list of batch sizes for ramping up during training.
            Defaults to None.
        seed (int, optional): The random seed for data shuffling. Defaults to 1234.
        memmap_workers (int, optional): The number of worker processes for loading data using TextMemMapDataset.
            Defaults to 1.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 8.
        pin_memory (bool, optional): Whether to pin memory during data loading for faster GPU training.
            Defaults to True.
        persistent_workers (bool, optional): Whether to keep data loading workers persistent across epochs.
            Defaults to False.
        dataset_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments to pass into the GPTSFTDataset class
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.seed = seed
        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.memmap_workers = memmap_workers
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.rampup_batch_size = rampup_batch_size
        self.data_sampler = None
        self.max_train_samples = None
        self.dataset_kwargs = dataset_kwargs or {}

    def setup(self, stage: str):
        """Called by pytorch lightning in datamodule setup"""

        # data_sampler is used in `setup_data_sampler` in MegatronStrategy.setup
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
            dataloader_type="batch",
        )

        # Follows the calculation in nemo.collections.nlp.data.language_modeling.megatron.
        # base_dataset_utils.get_datasets_weights_and_num_samples
        self.max_train_samples = int(math.ceil(self.global_batch_size * self.trainer.max_steps * 1.005))

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        consumed_samples = self.data_sampler.compute_consumed_samples(
            self.trainer.global_step - self.data_sampler.init_global_step
        )
        return {"consumed_samples": consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule stat

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """
        try:
            from megatron.core.num_microbatches_calculator import update_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import update_num_microbatches
        consumed_samples = state_dict["consumed_samples"]
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples

        update_num_microbatches(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        self.data_sampler.if_first_step = 1

    def train_dataloader(self) -> DataLoader:
        # pylint: disable=C0115,C0116
        return self._create_dataloader(
            self._create_dataset(
                self.train_path,
                max_num_samples=self.max_train_samples,
                **self.dataset_kwargs,
            ),
            mode="train",
        )

    def val_dataloader(self) -> DataLoader:
        # pylint: disable=C0115,C0116
        return self._create_dataloader(
            self._create_dataset(
                self.train_path,
                max_num_samples=self.max_train_samples,
                **self.dataset_kwargs,
            ),
            mode="train",
        )

    def test_dataloader(self) -> DataLoader:
        # pylint: disable=C0115,C0116
        return self._create_dataloader(
            self._create_dataset(
                self.train_path,
                max_num_samples=self.max_train_samples,
                **self.dataset_kwargs,
            ),
            mode="train",
        )

    @lru_cache
    def _create_dataset(self, path, **kwargs):
        return create_sft_dataset(
            path,
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
            memmap_workers=self.memmap_workers,
            seed=self.seed,
            **kwargs,
        )

    def _create_dataloader(self, dataset, mode, **kwargs) -> DataLoader:
        return WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )

    @property
    def train_path(self) -> Path:
        """Path to training dataset file"""
        return self.dataset_root / "training.jsonl"

    @property
    def validation_path(self) -> Path:
        """Path to validation dataset file"""
        return self.dataset_root / "validation.jsonl"

    @property
    def test_path(self) -> Path:
        """Path to test dataset file"""
        return self.dataset_root / "test.jsonl"

    def _extract_tokenizer_model_name(self) -> str:
        """Automatically get the model name from model path."""
        if isinstance(self.tokenizer, AutoTokenizer):
            name = self.tokenizer.tokenizer.name_or_path
            if name.endswith("context/nemo_tokenizer"):
                # NEMO_HOME/hf_org/hf_model/context/nemo_tokenizer => hf_org--hf_model
                tokenizer_model_name = '--'.join(name.split("/")[-4:-2])
            elif name.endswith("nemo_tokenizer"):
                # NEMO_HOME/hf_org/hf_model/nemo_tokenizer => hf_org--hf_model
                tokenizer_model_name = '--'.join(name.split("/")[-3:-1])
            else:
                # hf_org/hf_model => hf_org--hf_model
                tokenizer_model_name = name.replace("/", "--")
        else:
            tokenizer_model_name = f"unknown_tokenizer_{hash(self.tokenizer)}"
        return tokenizer_model_name
