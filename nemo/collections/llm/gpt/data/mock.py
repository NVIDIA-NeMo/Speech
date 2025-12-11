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

from typing import TYPE_CHECKING, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils.import_utils import safe_import

_, HAVE_TE = safe_import("transformer_engine")

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class MockDataModule(pl.LightningDataModule):
    """PyTorch Lightning-compatible data module for testing pre-training and fine-tuning workloads.
    MockDataModule will generate random token indices to simulate a dataset.
    Args:
        seq_length (int): Sequence length.
        tokenizer (Optional["TokenizerSpec"]): An instance of a TokenizerSpec object.
        micro_batch_size (int): Batch size per GPU.
        global_batch_size (int): Global batch size.
        rampup_batch_size (Optional[List[int]]): Rampup batch size, should be in format of
            [start_global_batch_size, batch_size_increment, ramup_samples].
        num_workers (int): See ``torch.utils.data.DataLoader`` documentation.
        pin_memory (bool): See ``torch.utils.data.DataLoader`` documentation.
        persistent_workers (bool): See ``torch.utils.data.DataLoader`` documentation.
        num_train_samples (Optional[int]): The number of samples to use for training, defaults to total
            train steps times global batch size.
        num_val_samples (Optional[int]): The number of samples to use for validation, defaults to total
            validation steps times global batch size.
        num_test_samples (Optional[int]): The number of samples to use for testing, defaults to total
            test steps times global batch size.
    """

    def __init__(
        self,
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        create_attention_mask: bool = False,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
    ):
        from nemo.lightning.callback_group import CallbackGroup

        CallbackGroup.get_instance().on_dataloader_init_start()

        super().__init__()
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.create_attention_mask = create_attention_mask or not HAVE_TE

        if tokenizer is None:
            from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer

            self.tokenizer = get_nmt_tokenizer(
                "megatron", "GPT2BPETokenizer", vocab_file=vocab_file, merges_file=merges_file
            )
        else:
            self.tokenizer = tokenizer

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

        CallbackGroup.get_instance().on_dataloader_init_end()

    def setup(self, stage: str = "") -> None:
        """
        Setup the data module.
        """
        self._train_ds = _MockGPTDataset(
            self.tokenizer, "train", self.num_train_samples, self.seq_length, self.create_attention_mask
        )
        self._validation_ds = _MockGPTDataset(
            self.tokenizer, "valid", self.num_val_samples, self.seq_length, self.create_attention_mask
        )
        self._test_ds = _MockGPTDataset(
            self.tokenizer, "test", self.num_test_samples, self.seq_length, self.create_attention_mask
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Get the train dataloader.
        """
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Get the validation dataloader.
        """
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Get the test dataloader.
        """
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )

    def reconfigure_limit_batches(self):
        """
        Reconfigure trainer.limit_train_batches and trainer.limit_val_batches in terms of num of microbatches.
        """
        from nemo.collections.llm.gpt.data.utils import _reconfigure_limit_batches

        # Override limit_train_batches in terms of num of microbatches
        self.trainer.limit_train_batches = _reconfigure_limit_batches(self.trainer.limit_train_batches, self._train_ds)
        # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting
        #   in between a step
        self.trainer.limit_val_batches = _reconfigure_limit_batches(
            self.trainer.limit_val_batches, self._validation_ds
        )

        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches

        except (ImportError, ModuleNotFoundError):
            from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        # Override num sanity steps to be a multiple of num of microbatches
        self.trainer.num_sanity_val_steps *= get_num_microbatches()


class _MockGPTDataset(Dataset):
    def __init__(
        self,
        tokenizer: "TokenizerSpec",
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
        create_attention_mask: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.seq_length = seq_length
        self.vocab_size = tokenizer.vocab_size
        self.length = num_samples
        self.seed = seed
        self.create_attention_mask = create_attention_mask

        if create_attention_mask:
            self.attention_mask = torch.tril(torch.ones((self.seq_length, self.seq_length), device='cpu')).unsqueeze(0)
            self.attention_mask = self.attention_mask < 0.5

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

    def __len__(self) -> int:
        return self.length

    def _get_text(self, idx: int) -> np.ndarray:
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        return np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Generate data of the expected size and datatype (based on GPTDataset).
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        tokens = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length + 1], dtype=np.int64))

        batch = {
            "tokens": tokens[:-1],
            "labels": tokens[1:],
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
        }

        if self.create_attention_mask:
            batch["attention_mask"] = self.attention_mask

        return batch

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return data.dataloader.default_collate(batch)

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
