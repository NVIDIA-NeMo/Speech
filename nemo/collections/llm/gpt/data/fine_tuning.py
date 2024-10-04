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
from typing import TYPE_CHECKING, List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from nemo.collections.llm.gpt.data.core import create_sft_dataset
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec


class FineTuningDataModule(pl.LightningDataModule):
    """Base class for fine-tuning an LLM.

    This class provides a foundation for building custom data modules for fine-tuning Nemo NLP models. It inherits from
    `pl.LightningDataModule` from the PyTorch Lightning library and handles data loading, preprocessing, and batch creation
    for training, validation, and testing.

    Args:
        dataset_root (Union[str, Path]): The root directory containing the training, validation, and test data.
        seq_length (int, optional): The maximum sequence length for the input and output text. Defaults to 2048.
        tokenizer (Optional[TokenizerSpec], optional): The tokenizer to use for preprocessing the text. Defaults to None.
            If not provided, a Megatron GPT2 BPE tokenizer will be used.
        micro_batch_size (int, optional): The micro batch size for training. Defaults to 4.
        global_batch_size (int, optional): The global batch size for training. Defaults to 8.
        rampup_batch_size (Optional[List[int]], optional): A list of batch sizes for ramping up during training. Defaults to None.
        seed (int, optional): The random seed for data shuffling. Defaults to 1234.
        memmap_workers (int, optional): The number of worker processes for loading data using TextMemMapDataset. Defaults to 1.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 8.
        pin_memory (bool, optional): Whether to pin memory during data loading for faster GPU training. Defaults to True.
        persistent_workers (bool, optional): Whether to keep data loading workers persistent across epochs. Defaults to False.
        max_train_steps (int, optional): Maximum number of steps to train. Used to calculate samples mapping for the mmap dataset
        pad_to_max_length (bool, optional): Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        packed_sequence_size (int, optional): If a positive integer, this arg enables training with sequence packing and specifies the pack size
            If less than or equal to 0, sequence packing is disabled. Defaults to -1.
            Note: This arg is distinct from `seq_length` because `seq_length` specifies the maximum length of the original sequence
            (i.e. the length to truncate long sequences in the input data).
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
        pad_to_max_length: bool = False,
        packed_sequence_size: int = -1,
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
        self.pad_to_max_length = pad_to_max_length
        self.packed_sequence_size = packed_sequence_size
        self._adjust_batch_sizes_for_packed_sequence()

    def _adjust_batch_sizes_for_packed_sequence(self):
        if self.packed_sequence_size > 0 and self.micro_batch_size > 1:
            logging.warning(
                "Micro batch size should be 1 when training with packed sequence, but your micro batch size "
                f"is {self.micro_batch_size}. Your config will be automatically updated to the following: "
                f"MBS will be set to 1 (from {self.micro_batch_size}), "
                f"GBS will be set to {self.global_batch_size // self.micro_batch_size} (from {self.global_batch_size}), "
                f"packed sequence length will be set to {self.packed_sequence_size*self.micro_batch_size} (from {self.packed_sequence_size}). "
                f"For details please visit https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/sequence_packing.html"
            )
            self.global_batch_size //= self.micro_batch_size
            self.packed_sequence_size *= self.micro_batch_size
            self.micro_batch_size = 1

    def prepare_data(self) -> None:
        if self.packed_sequence_size > 0 and not self.train_path_packed.is_file():
            from nemo.collections.llm.gpt.data.packed_sequence import prepare_packed_sequence_data

            prepare_packed_sequence_data(
                input_path=self.train_path,
                output_path=self.train_path_packed,
                packed_sequence_size=self.packed_sequence_size,
                tokenizer=self.tokenizer,
                max_seq_length=self.seq_length,
                seed=self.seed,
            )

    def setup(self, stage: str):
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

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self._create_dataset(
                self.train_path if self.packed_sequence_size <= 0 else self.train_path_packed,
                max_num_samples=self.max_train_samples,
                pad_to_max_length=self.pad_to_max_length,
            )
        )

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self._create_dataset(
                self.validation_path,
                is_test=True,
                pad_to_max_length=self.pad_to_max_length,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self._create_dataset(
                self.test_path,
                tokens_to_generate=32,
                is_test=True,
                pad_to_max_length=self.pad_to_max_length,
            )
        )

    @lru_cache
    def _create_dataset(self, path, is_test=False, **kwargs):
        return create_sft_dataset(
            path,
            tokenizer=self.tokenizer,
            seq_length=(self.seq_length if is_test or self.packed_sequence_size <= 0 else self.packed_sequence_size),
            memmap_workers=self.memmap_workers,
            seed=self.seed,
            is_test=is_test,
            **kwargs,
        )

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )

    @property
    def train_path(self) -> Path:
        return self.dataset_root / "training.jsonl"

    @property
    def train_path_packed(self) -> Path:
        if self.packed_sequence_size > 0:
            return self.dataset_root / f"training_packed{self.packed_sequence_size}.npy"
        else:
            raise ValueError("`train_path_packed` invalid since packed sequence size is not specified.")

    @property
    def validation_path(self) -> Path:
        return self.dataset_root / "validation.jsonl"

    @property
    def test_path(self) -> Path:
        return self.dataset_root / "test.jsonl"
