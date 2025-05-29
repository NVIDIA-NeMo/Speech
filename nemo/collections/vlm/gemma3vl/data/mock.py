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

from typing import Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from transformers import Gemma3ImageProcessor

from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging

IMAGE_TOKEN_INDEX = 262144
IMAGE_SIZE = 896
IMAGE_TOKENS = 256


class Gemma3VLMockDataModule(pl.LightningDataModule):
    """
    A mock data module for Gemma3VL training, validation, and testing.
    """

    def __init__(
        self,
        tokenizer: Optional = None,
        seq_length: int = 2048,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        if tokenizer is None:
            logging.warning("Tokenizer is not provided! Fall back to `'google/gemma-3-4b-it'`.")
            from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

            self.tokenizer = tokenizer or AutoTokenizer("google/gemma-3-4b-it")

        self.image_processor = Gemma3ImageProcessor(
            size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
            do_pan_and_scan=False,
            pan_and_scan_min_crop_size=256,
            pan_and_scan_max_num_crops=4,
            pan_and_scan_min_ratio_to_activate=1.2,
        )
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        # pylint: disable=C0115,C0116
        self._train_ds = _Qwen2VLMockDataset(
            self.tokenizer, self.image_processor, "train", self.num_train_samples, self.seq_length
        )
        self._validation_ds = _Qwen2VLMockDataset(
            self.tokenizer, self.image_processor, "valid", self.num_val_samples, self.seq_length
        )
        self._test_ds = _Qwen2VLMockDataset(
            self.tokenizer, self.image_processor, "test", self.num_test_samples, self.seq_length
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        # pylint: disable=C0115,C0116
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )


def prepare_image_inputs(num_channels: np.uint8 = 3, width=1024, height=1024):
    """This function prepares a list of PIL images"""
    image_inputs = [np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8)]
    image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
    return image_inputs


class _Qwen2VLMockDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        image_processor,
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.name = name
        self.seq_length = seq_length

        self.vocab_size = tokenizer.vocab_size - 10

        self.image_processor = image_processor

        self.length = num_samples
        self.seed = seed

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

    def __len__(self) -> int:
        return self.length

    def _get_text(self, idx: int) -> np.ndarray:
        np_gen = np.random.default_rng(seed=self.seed + idx)
        return np_gen.integers(low=0, high=self.vocab_size, size=[self.seq_length], dtype=np.int64)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        # 1) processor image input
        image_inputs = prepare_image_inputs(3, IMAGE_SIZE, IMAGE_SIZE)
        prcocess_out = self.image_processor(image_inputs, return_tensors="pt")
        pixel_values = prcocess_out.pixel_values.to(torch.bfloat16)

        # 2) prepare input token ids
        np_gen = np.random.default_rng(seed=self.seed + idx)
        tokens = torch.from_numpy(
            np_gen.integers(low=0, high=self.vocab_size, size=[self.seq_length + 1], dtype=np.int64)
        )

        # 3) fill IMAGE_TOKEN_INDEX to input token ids
        img_start_idx = 20  # pick a rnd value, where img token begins
        tokens[img_start_idx : img_start_idx + IMAGE_TOKENS] = IMAGE_TOKEN_INDEX  # ImageToken token index
        input_ids = tokens[:-1]
        position_ids = torch.arange(self.seq_length, dtype=torch.int64)

        # 4) prepare labels
        labels = tokens.clone()
        generation_prompt_size = 5  # "ASSISTANT:" like
        prompt_end_idx = img_start_idx + IMAGE_TOKENS + generation_prompt_size
        labels[:prompt_end_idx] = -100
        labels = labels[1:]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "loss_mask": self.loss_mask,
            "labels": labels,
        }

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        collated_batch = data.dataloader.default_collate(batch)
        collated_batch["attention_mask"] = None
        return collated_batch

    def collate_fn(self, batch):
        """Method that user pass as functor to DataLoader.

        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        # Usage:
        dataloader = torch.utils.data.DataLoader(
                ....,
                collate_fn=dataset.collate_fn,
                ....
        )

        Returns
        -------
            Collated batch, with or without types.
        """
        return self._collate_fn(batch)
