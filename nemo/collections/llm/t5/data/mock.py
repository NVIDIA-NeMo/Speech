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
    def __init__(
        self,
        seq_length: int = 512,
        seq_length_dec: int = 128,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        create_attention_mask: bool = False,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.seq_length_dec = seq_length_dec
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.create_attention_mask = create_attention_mask or not HAVE_TE

        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        self.tokenizer = tokenizer or get_nmt_tokenizer("megatron", "BertWordPieceCase")
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        self._train_ds = _MockT5Dataset(
            self.tokenizer, "train", self.num_train_samples, self.seq_length, self.seq_length_dec
        )
        self._validation_ds = _MockT5Dataset(
            self.tokenizer, "valid", self.num_val_samples, self.seq_length, self.seq_length_dec
        )
        self._test_ds = _MockT5Dataset(
            self.tokenizer, "test", self.num_test_samples, self.seq_length, self.seq_length_dec
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
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


class _MockT5Dataset(Dataset):
    def __init__(
        self,
        tokenizer: "TokenizerSpec",
        name: str,
        num_samples: int,
        seq_length: int,
        seq_length_dec: int,
        seed: int = 42,
        create_attention_mask: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.seq_length = seq_length
        self.seq_length_dec = seq_length_dec
        self.vocab_size = tokenizer.vocab_size
        self.length = num_samples
        self.seed = seed
        self.create_attention_mask = create_attention_mask

        # update for T5 now use FlashFused attention (b11s)
        self.mask_encoder = torch.ones(self.seq_length, device='cpu')
        self.mask_decoder = torch.ones(self.seq_length_dec, device='cpu')
        self.mask_encoder = self.mask_encoder < 0.5
        self.mask_decoder = self.mask_decoder < 0.5
        self.loss_mask = torch.ones(self.seq_length_dec, dtype=torch.float)

    def __len__(self) -> int:
        return self.length

    def _get_text(self, idx: int) -> np.ndarray:
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        return np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Generate data of the expected size and datatype (based on GPTDataset).
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        encoder_input = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64))
        decoder_input = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length_dec], dtype=np.int64))
        labels = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length_dec], dtype=np.int64))

        batch = {
            "text_enc": encoder_input,
            "text_dec": decoder_input,
            "labels": labels,
            "loss_mask": self.loss_mask,
            "truncated": 0,
            "enc_mask": self.mask_encoder,
            "dec_mask": self.mask_decoder,
        }

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
