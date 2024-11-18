# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import lightning.pytorch as pl
from omegaconf.omegaconf import DictConfig
from torch.utils.data import DataLoader


# TODO(ahmadki): multi-GPU needs more work, we currently don't shard data
# across GPUs, which is OK for trainnig, but needs fixing for validation and testing.
class AggregatorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: DictConfig = None,
        train_batch_size: int = 1,
        train_shuffle: bool = False,
        val_dataset: DictConfig = None,
        val_batch_size: int = 1,
        val_shuffle: bool = False,
        test_dataset: DictConfig = None,
        test_batch_size: int = 1,
        test_shuffle: bool = False,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.val_dataset = val_dataset
        self.val_batch_size = val_batch_size
        self.val_shuffle = val_shuffle
        self.test_dataset = test_dataset
        self.test_batch_size = test_batch_size
        self.test_shuffle = test_shuffle

    # TODO(ahmadki): lazy init
    # def setup(self, stage=None) -> None:
    #     if stage in [None, "fit"]:
    #         self.train_dataset = instantiate(self.train_dataset)
    #     if stage in [None, "fit", "validate"]:
    #         self.val_dataset = instantiate(self.val_dataset)
    #     if stage in [None, "test", "predict"]:
    #         self.test_dataset = instantiate(self.test_dataset)

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
            num_workers=4,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.val_dataset.collate_fn,
            shuffle=self.val_shuffle,
            pin_memory=True,
            num_workers=0,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            collate_fn=self.test_dataset.collate_fn,
            shuffle=self.test_shuffle,
            pin_memory=True,
            num_workers=0,
        )
        return loader
