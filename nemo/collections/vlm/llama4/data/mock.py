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
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging


class MockDataModule(pl.LightningDataModule):
    """A mock LightningDataModule for generating synthetic data for Llama4 models.

    This module creates dummy datasets (train, validation, test) using `MockLlama4Dataset`
    for testing or development purposes without requiring actual data.

    Args:
        seq_length (int): The sequence length for text tokens. Defaults to 2048.
        decoder_seq_length (Optional[int]): The sequence length for the decoder (if applicable). Defaults to None.
        tokenizer (Optional): Tokenizer object.
        image_processor (Optional): Image processor object.
        micro_batch_size (int): Micro batch size per GPU. Defaults to 4.
        global_batch_size (int): Global batch size across all GPUs. Defaults to 8.
        rampup_batch_size (Optional[List[int]]): Ramp-up schedule for batch size. Defaults to None.
        num_train_samples (int): Number of synthetic samples for the training set. Defaults to 10,000,000.
        num_val_samples (int): Number of synthetic samples for the validation set. Defaults to 10,000,000.
        num_test_samples (int): Number of synthetic samples for the test set. Defaults to 10,000,000.
        num_workers (int): Number of worker processes for data loading. Defaults to 8.
        pin_memory (bool): Whether to pin memory for faster data transfer to GPU. Defaults to True.
        persistent_workers (bool): Whether to keep worker processes alive between epochs. Defaults to False.
        packed_sequence (bool): Whether to use packed sequences for efficiency. Defaults to False.
    """

    def __init__(
        self,
        seq_length: int = 2048,
        decoder_seq_length: Optional[int] = None,
        tokenizer: Optional = None,
        image_processor: Optional = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000_000,
        num_val_samples: int = 10_000_000,
        num_test_samples: int = 10_000_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        packed_sequence: bool = False,
    ):
        """A mock LightningDataModule for generating synthetic data for Llama4 models.

        This module creates dummy datasets (train, validation, test) using `MockLlama4Dataset`
        for testing or development purposes without requiring actual data.

        Args:
            seq_length (int): The sequence length for text tokens. Defaults to 2048.
            decoder_seq_length (Optional[int]): The sequence length for the decoder (if applicable). Defaults to None.
            tokenizer (Optional): Tokenizer object.
            image_processor (Optional): Image processor object.
            micro_batch_size (int): Micro batch size per GPU. Defaults to 4.
            global_batch_size (int): Global batch size across all GPUs. Defaults to 8.
            rampup_batch_size (Optional[List[int]]): Ramp-up schedule for batch size. Defaults to None.
            num_train_samples (int): Number of synthetic samples for the training set. Defaults to 10,000,000.
            num_val_samples (int): Number of synthetic samples for the validation set. Defaults to 10,000,000.
            num_test_samples (int): Number of synthetic samples for the test set. Defaults to 10,000,000.
            num_workers (int): Number of worker processes for data loading. Defaults to 8.
            pin_memory (bool): Whether to pin memory for faster data transfer to GPU. Defaults to True.
            persistent_workers (bool): Whether to keep worker processes alive between epochs. Defaults to False.
            packed_sequence (bool): Whether to use packed sequences for efficiency. Defaults to False.
        """
        super().__init__()
        self.seq_length = seq_length
        self.decoder_seq_len = decoder_seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.packed_sequence = packed_sequence

        if tokenizer is None or image_processor is None:
            logging.warning(
                "Processor or tokenizer are not provided! Fall back to `'meta-llama/Llama-4-Scout-17B-16E-Instruct'`."
            )
            from transformers import AutoProcessor
            from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

            processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
            self.tokenizer = tokenizer or AutoTokenizer("meta-llama/Llama-4-Scout-17B-16E-Instruct")
            self.image_processor = image_processor or processor.image_processor
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_len,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        """Sets up the mock datasets for the specified stage.

        Initializes `MockLlama4Dataset` instances for train, validation, and test splits.
        Adjusts sequence length if packed sequences are used.

        Args:
            stage (str): The stage for which to set up data ('fit', 'validate', 'test', or '').
                         Defaults to "".
        """
        seq_length = self.seq_length
        if self.packed_sequence and self.micro_batch_size > 1:
            seq_length = seq_length // self.micro_batch_size
            logging.warning(
                f"Packed sequence is used with mock dataset. Sequence length for each "
                f"sample is update to `seq_length // self.micro_batch_size = {seq_length}`!"
            )
        self._train_ds = MockLlama4Dataset(
            self.tokenizer,
            self.image_processor,
            "train",
            self.num_train_samples,
            seq_length,
            packed_sequence=self.packed_sequence,
        )
        self._validation_ds = MockLlama4Dataset(
            self.tokenizer,
            self.image_processor,
            "valid",
            self.num_val_samples,
            seq_length,
            packed_sequence=self.packed_sequence,
        )
        self._test_ds = MockLlama4Dataset(
            self.tokenizer,
            self.image_processor,
            "test",
            self.num_test_samples,
            seq_length,
            packed_sequence=self.packed_sequence,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Returns the DataLoader for the training set."""
        if not hasattr(self, "_train_ds"):
            self.setup('fit')  # Ensure dataset is created
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Returns the DataLoader for the validation set."""
        if not hasattr(self, "_validation_ds"):
            self.setup('validate')  # Ensure dataset is created
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Returns the DataLoader for the test set."""
        if not hasattr(self, "_test_ds"):
            self.setup('test')  # Ensure dataset is created
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        """Creates a DataLoader for the given dataset.

        Args:
            dataset (Dataset): The dataset to wrap in a DataLoader.
            **kwargs: Additional arguments passed to the DataLoader constructor.

        Returns:
            DataLoader: The configured DataLoader instance.
        """
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )


class MockLlama4Dataset(Dataset):
    """A mock Dataset implementation for generating synthetic Llama4 data.

    Produces batches containing dummy image tensors and random token sequences,
    mimicking the structure expected by Llama4 models.

    Args:
        tokenizer: Tokenizer object to determine vocabulary size.
        image_processor: Image processor object to determine image dimensions.
        name (str): Name of the dataset split (e.g., "train", "valid", "test").
        num_samples (int): Total number of synthetic samples in this dataset.
        seq_length (int): Sequence length for the generated token sequences.
        seed (int): Random seed for data generation reproducibility. Defaults to 42.
        packed_sequence (bool): Whether the data should be formatted for packed sequences.
                                Defaults to False.
        pixel_shuffle_ratio (float): Ratio used for calculating the image sequence length
                                     after potential pixel shuffling. Defaults to 0.5.
        num_image_embeddings_per_tile (int): Number of embeddings produced per image tile
                                             by the vision encoder (before pixel shuffle).
                                             Defaults to 576.
        num_tiles_per_image (int): Number of tiles the image is split into. Defaults to 1.
    """

    def __init__(
        self,
        tokenizer,
        image_processor,
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
        packed_sequence: bool = False,
        pixel_shuffle_ratio: float = 0.5,
        num_image_embeddings_per_tile=576,
        num_tiles_per_image=1,
    ) -> None:
        """A mock Dataset implementation for generating synthetic Llama4 data.

        Produces batches containing dummy image tensors and random token sequences,
        mimicking the structure expected by Llama4 models.

        Args:
            tokenizer: Tokenizer object to determine vocabulary size.
            image_processor: Image processor object to determine image dimensions.
            name (str): Name of the dataset split (e.g., "train", "valid", "test").
            num_samples (int): Total number of synthetic samples in this dataset.
            seq_length (int): Sequence length for the generated token sequences.
            seed (int): Random seed for data generation reproducibility. Defaults to 42.
            packed_sequence (bool): Whether the data should be formatted for packed sequences.
                                    Defaults to False.
            pixel_shuffle_ratio (float): Ratio used for calculating the image sequence length
                                         after potential pixel shuffling. Defaults to 0.5.
            num_image_embeddings_per_tile (int): Number of embeddings produced per image tile
                                                 by the vision encoder (before pixel shuffle).
                                                 Defaults to 576.
            num_tiles_per_image (int): Number of tiles the image is split into. Defaults to 1.
        """
        super().__init__()
        self.name = name
        self.seq_length = seq_length

        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self.vocab_size = 200000
        size = image_processor.size
        self.image_height, self.image_width = size["height"], size["width"]

        self.length = num_samples
        self.seed = seed
        self.packed_sequence = packed_sequence
        self.num_image_embeddings_per_tile = num_image_embeddings_per_tile
        self.num_tiles_per_image = num_tiles_per_image
        self.pixel_shuffle_ratio = pixel_shuffle_ratio
        self._img_seq_len = int(
            num_image_embeddings_per_tile * num_tiles_per_image * pixel_shuffle_ratio * pixel_shuffle_ratio
        )

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.length

    def _get_text(self, idx: int) -> np.ndarray:
        """Generates a random sequence of token IDs (unused in current __getitem__).

        Args:
            idx (int): Index of the sample, used for seeding the random generator.

        Returns:
            np.ndarray: An array of random token IDs.
        """
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        return np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generates a single synthetic data sample.

        Creates random tensors for 'media' (images), 'tokens', and 'labels'.
        The 'tokens' sequence includes placeholder IDs where image features
        would normally be inserted.

        Args:
            idx (int): Index of the sample to generate. Used for seeding.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "media": A dummy image tensor [num_tiles, 3, H, W].
                - "tokens": Input token sequence [seq_length].
                - "labels": Target token sequence (shifted tokens) [seq_length].
                - "loss_mask": Mask indicating which tokens contribute to loss [seq_length].
                - "position_ids": Positional IDs for the sequence [seq_length].
        """
        # Generate data of the expected size and datatype (based on GPTDataset).
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        # Generate tokens + 1 for creating labels by shifting
        tokens = torch.from_numpy(
            np_gen.integers(
                # Using a large upper bound for random token IDs
                self.vocab_size,
                size=[self.seq_length + 1],
                dtype=np.int64,
            )
        )
        # Insert placeholder token ID where image embeddings would go
        # Assuming a fixed placeholder ID 200092 for <|patch|>
        tokens[2 : 2 + self._img_seq_len] = 200092  # <|patch|> token index TODO: Use actual token ID if available
        labels = tokens.clone()
        images = torch.from_numpy(
            np_gen.random(
                # num_concurrent_media, num_tiles, nch, h, w
                size=[self.num_tiles_per_image, 3, self.image_height, self.image_width],
                dtype=np.float32,
            )
        ).bfloat16()
        tokens = tokens[:-1]
        labels = labels[1:]
        return {
            "media": images,
            "tokens": tokens,
            "labels": labels,
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
        }

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
        """Collates a batch of samples from the dataset.

        Uses the default PyTorch collate function and then performs specific adjustments:
        - Sets 'attention_mask' to None.
        - Reshapes 'media' tensor.
        - If `packed_sequence` is True, it prepares the batch for packed sequence format
          by calculating cumulative sequence lengths (`cu_seqlens`) and reshaping
          relevant tensors ('tokens', 'labels', 'loss_mask', 'position_ids').

        Args:
            batch (List[Dict[str, torch.Tensor]]): A list of individual samples (dictionaries)
                                                  from `__getitem__`.

        Returns:
            Dict[str, Optional[torch.Tensor]]: The collated batch, ready for model input.
                                               Includes 'packed_seq_params' if packing is enabled.
        """
        collated_batch = data.dataloader.default_collate(batch)
        # Mock dataset doesn't typically need a specific attention mask
        collated_batch["attention_mask"] = None
        # Reshape media: [B, num_tiles, C, H, W] -> [B * num_tiles, C, H, W]
        collated_batch["media"] = collated_batch["media"].reshape(-1, *collated_batch["media"].shape[2:])
        if self.packed_sequence:
            from megatron.core.packed_seq_params import PackedSeqParams

            tokens = collated_batch["tokens"]
            batch_size = tokens.shape[0]
            valid_seqlen = self.seq_length
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * (valid_seqlen), step=(valid_seqlen), dtype=torch.int32, device=tokens.device
            )
            cu_seqlens_padded = torch.arange(
                0, (batch_size + 1) * (valid_seqlen), step=(valid_seqlen), dtype=torch.int32, device=tokens.device
            )
            qkv_format = 'thd'
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv_padded=cu_seqlens_padded,
                max_seqlen_q=valid_seqlen,
                max_seqlen_kv=valid_seqlen,
                qkv_format=qkv_format,
            )
            collated_batch["packed_seq_params"] = packed_seq_params

            for key in ["tokens", "labels", "loss_mask", "position_ids"]:
                collated_batch[key] = collated_batch[key].reshape(1, -1)

        return collated_batch

    def collate_fn(self, batch):
        """Method passed to the DataLoader's `collate_fn` argument.

        Simply calls the internal `_collate_fn` implementation. This structure allows for
        potential future additions like neural type checking within this wrapper method.

        Args:
            batch: A list of samples fetched from the dataset.

        Returns:
            The collated batch dictionary.
        """
        return self._collate_fn(batch)
