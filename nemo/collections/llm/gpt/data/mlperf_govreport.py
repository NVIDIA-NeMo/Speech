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

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from datasets import DatasetDict, load_dataset

from nemo.collections.llm.gpt.data.core import get_dataset_root
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs


class MLPerfGovReportDataModule(FineTuningDataModule, IOMixin):
    """
    A data module for fine-tuning on the govreport dataset as preprocessed for MLPerf;
    see https://huggingface.co/datasets/regisss/scrolls_gov_report_preprocessed_mlperf_2

    Inherits from `FineTuningDataModule` and handles data download, splitting, and
    saving in a format ready for training.

    Args:
        force_redownload (bool, optional): Whether to force re-download the dataset even
            if it exists locally. Defaults to False.
        delete_raw (bool, optional): Whether to delete the raw downloaded dataset after
            preprocessing. Defaults to True.
        See FineTuningDataModule for the other args
    """

    def __init__(
        self,
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        force_redownload: bool = False,
        delete_raw: bool = True,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        packed_sequence_specs: Optional["PackedSequenceSpecs"] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.force_redownload = force_redownload
        self.delete_raw = delete_raw

        super().__init__(
            dataset_root=get_dataset_root("govreport"),
            seq_length=seq_length,
            tokenizer=tokenizer,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            seed=seed,
            memmap_workers=memmap_workers,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            packed_sequence_specs=packed_sequence_specs,
            dataset_kwargs=dataset_kwargs,
        )

        if self.packed_sequence_size != self.seq_length:
            raise ValueError(
                f"{self.__class__.__name__} requires `packed_sequence_specs.packed_sequence_size` to be nonzero "
                f"and equal to `seq_length`.  Instead got packed_sequence_size = {self.packed_sequence_size} "
                f"and seq_length = {self.seq_length}"
            )

    def prepare_data(self) -> None:
        # if train file is specified, no need to do anything
        if not self.train_path.exists() or self.force_redownload:
            dset = self._download_data()
            self._preprocess_and_split_data(dset)
        super().prepare_data()

    def _download_data(self):
        logging.info(f"Downloading {self.__class__.__name__}...")
        return load_dataset(
            "regisss/scrolls_gov_report_preprocessed_mlperf_2",
            cache_dir=str(self.dataset_root),
            download_mode="force_redownload" if self.force_redownload else None,
        )

    def _preprocess_and_split_data(
        self, dset: DatasetDict, split_val_from_train: bool = True, val_proportion: float = 0.05
    ):
        """Preprocesses and splits the downloaded dataset into training, validation, and test sets.

        Args:
            dset (DatasetDict): The downloaded dataset object.
            split_val_from_train (bool, optional): Whether to split the validation set from the training set.
                If False, the validation set is split from the test set. Defaults to True.
            val_proportion (float, optional): The proportion of the training or test set to be used for
                the validation split.
                Defaults to 0.05.
        """
        logging.info(f"Preprocessing {self.__class__.__name__} to npy format and splitting...")
        save_splits = {}
        train_set = dset.get('train')
        val_set = dset.get('validation')

        if split_val_from_train:
            split_dataset = train_set.train_test_split(test_size=val_proportion, seed=self.seed)
            save_splits['training'] = split_dataset['train']
            save_splits['validation'] = split_dataset['test']
            save_splits['test'] = val_set
        else:
            split_dataset = val_set.train_test_split(test_size=val_proportion, seed=self.seed)
            save_splits['training'] = train_set
            save_splits['validation'] = split_dataset['test']
            save_splits['test'] = split_dataset['train']

        for split_name, dataset in save_splits.items():
            output_file = self.dataset_root / f"{split_name}.npy"
            processed_data = [
                {
                    "input_ids": example["input_ids"],
                    "loss_mask": [int(x != -100) for x in example["labels"]],
                    "seq_start_id": [0],
                }
                for example in dataset
            ]
            np.save(output_file, processed_data)

            logging.info(f"{split_name} split saved to {output_file}")

        if self.delete_raw:
            for p in self.dataset_root.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                elif '.npy' not in str(p.name):
                    p.unlink()

    @property
    def train_path(self) -> Path:
        """Path to training dataset file"""
        return self.dataset_root / "training.npy"

    @property
    def validation_path(self) -> Path:
        """Path to validation dataset file"""
        return self.dataset_root / "validation.npy"

    @property
    def test_path(self) -> Path:
        """Path to test dataset file"""
        return self.dataset_root / "test.npy"

    @property
    def default_pack_path(self) -> Path:
        return None

    @property
    def pack_metadata(self) -> Path:
        return None

    @property
    def train_path_packed(self) -> Path:
        """Path to training dataset file for packed sequence. The file path contains a reference to the
        tokenizer/model name since packed sequence dataset consists of tokenized indices."""
        return self.train_path

    @property
    def validation_path_packed(self) -> Path:
        """Path to validation dataset file for packed sequence. The file path contains a reference to the
        tokenizer/model name since packed sequence dataset consists of tokenized indices."""
        return self.validation_path
