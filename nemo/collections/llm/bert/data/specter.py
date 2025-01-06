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
import json
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from datasets import DatasetDict, load_dataset

from nemo.collections.llm.bert.data.core import get_dataset_root
from nemo.collections.llm.bert.data.fine_tuning import FineTuningDataModule
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec


class SpecterDataModule(FineTuningDataModule, IOMixin):
    """A data module for fine-tuning on the Specter dataset.

    This class inherits from the `FineTuningDataModule` class and is specifically designed for fine-tuning models
    on the SPECTER Datasets. It handles data download, preprocessing, splitting, and preparing the data
    in a format suitable for training, validation, and testing.

    Args:
        force_redownload (bool, optional): Whether to force re-download the dataset even if it exists locally.
                                           Defaults to False.
        delete_raw (bool, optional): Whether to delete the raw downloaded dataset after preprocessing.
                                     Defaults to True.
        See FineTuningDataModule for the other args
    """

    def __init__(
        self,
        seq_length: int = 512,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        force_redownload: bool = False,
        delete_raw: bool = True,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.force_redownload = force_redownload
        self.delete_raw = delete_raw

        super().__init__(
            dataset_root=get_dataset_root("specter"),
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
            dataset_kwargs=dataset_kwargs,
        )

    def prepare_data(self) -> None:
        """Prepare dataset for fine-tuning."""
        # if train file is specified, no need to do anything
        if not self.train_path.exists() or self.force_redownload:
            dset = self._download_data()
            self._preprocess_and_split_data(dset)
        super().prepare_data()

    def _download_data(self):
        logging.info(f"Downloading {self.__class__.__name__}...")
        return load_dataset(
            "sentence-transformers/specter",
            "triplet",
            cache_dir=str(self.dataset_root),
            download_mode="force_redownload" if self.force_redownload else None,
        )

    def _preprocess_and_split_data(self, dset: DatasetDict, train_ratio: float = 0.80, val_ratio: float = 0.15):
        """Preprocesses and splits the downloaded dataset into training, validation, and test sets.

        Args:
            dset (DatasetDict): The downloaded dataset object.
            split_val_from_train (bool, optional): Whether to split the validation set from the training set.
                If False, the validation set is split from the test set. Defaults to True.
            val_proportion (float, optional): The proportion of the training or test set to be used
                for the validation split. Defaults to 0.05.
        """
        logging.info(f"Preprocessing {self.__class__.__name__} to jsonl format and splitting...")

        test_ratio = 1 - train_ratio - val_ratio
        save_splits = {}
        dataset = dset.get('train')
        split_dataset = dataset.train_test_split(test_size=val_ratio + test_ratio, seed=self.seed)
        split_dataset2 = split_dataset['test'].train_test_split(
            test_size=test_ratio / (val_ratio + test_ratio), seed=self.seed
        )
        save_splits['training'] = split_dataset['train']
        save_splits['validation'] = split_dataset2['train']
        save_splits['test'] = split_dataset2['test']

        for split_name, dataset in save_splits.items():
            output_file = self.dataset_root / f"{split_name}.jsonl"
            with output_file.open("w", encoding="utf-8") as f:
                for o in dataset:
                    f.write(
                        json.dumps({"query": o["anchor"], "pos_doc": o["positive"], "neg_doc": [o["negative"]]}) + "\n"
                    )

            logging.info(f"{split_name} split saved to {output_file}")

        if self.delete_raw:
            for p in self.dataset_root.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                elif '.jsonl' not in str(p.name):
                    p.unlink()

    def reconfigure_limit_batches(self):
        """No need to reconfigure trainer.limit_val_batches for finetuning"""
        return
