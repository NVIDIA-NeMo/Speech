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

from typing import Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from nemo.collections.multimodal.data.energon.config import AudioToken, ImageToken
from nemo.lightning.pytorch.plugins import MegatronDataSampler


class MockDataModule(pl.LightningDataModule):
    """
    A mock data module for AVLM training, validation, and testing.

    Provides datasets and data loaders for training, validation, and testing phases.
    Includes data sampling and preprocessing for multimodal tasks.
    """

    def __init__(
        self,
        seq_length: int = 8192,
        decoder_seq_length: Optional[int] = None,
        tokenizer=None,
        image_processor=None,
        audio_processor=None,
        image_embedding_tokens: int = 576,
        audio_embedding_tokens: int = 1500,
        micro_batch_size: int = 2,
        global_batch_size: int = 2,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000_000,
        num_val_samples: int = 10_000_000,
        num_test_samples: int = 10_000_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        """
        Initializes the mock data module with data sampling and preprocessing configurations.

        Args:
            seq_length (int): Maximum sequence length for tokens.
            decoder_seq_length (Optional[int]): Sequence length for the decoder.
            tokenizer: Tokenizer for text processing.
            image_processor: Processor for image preprocessing.
            audio_processor: Processor for audio preprocessing.
            image_embedding_tokens: Number of image embedding tokens for one image.
            audio_embedding_tokens: Number of audio embedding tokens for one audio.
            micro_batch_size (int): Batch size per GPU.
            global_batch_size (int): Total batch size across GPUs.
            rampup_batch_size (Optional[List[int]]): Batch size ramp-up schedule.
            num_train_samples (int): Number of training samples.
            num_val_samples (int): Number of validation samples.
            num_test_samples (int): Number of testing samples.
            num_workers (int): Number of workers for data loading.
            pin_memory (bool): Whether to pin memory for data loaders.
            persistent_workers (bool): Whether to keep workers alive after the first iteration.
        """
        super().__init__()
        self.seq_length = seq_length
        self.decoder_seq_len = decoder_seq_length
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.tokenizer = tokenizer
        model_name = "llava-hf/llava-1.5-7b-hf"
        self.image_processor = AutoProcessor.from_pretrained(model_name).image_processor
        self.audio_processor = audio_processor
        self.image_embedding_tokens = image_embedding_tokens
        self.audio_embedding_tokens = audio_embedding_tokens
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_len,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        """
        Sets up the training, validation, and testing datasets.

        Args:
            stage (str): Stage of the setup ('train', 'valid', 'test').
        """
        self._train_ds = _MockAVLMDataset(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            audio_processor=self.audio_processor,
            image_embedding_tokens=self.image_embedding_tokens,
            audio_embedding_tokens=self.audio_embedding_tokens,
            name="train",
            num_samples=self.num_train_samples,
            seq_length=self.seq_length,
        )
        self._validation_ds = _MockAVLMDataset(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            audio_processor=self.audio_processor,
            image_embedding_tokens=self.image_embedding_tokens,
            audio_embedding_tokens=self.audio_embedding_tokens,
            name="valid",
            num_samples=self.num_val_samples,
            seq_length=self.seq_length,
        )
        self._test_ds = _MockAVLMDataset(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            audio_processor=self.audio_processor,
            image_embedding_tokens=self.image_embedding_tokens,
            audio_embedding_tokens=self.audio_embedding_tokens,
            name="test",
            num_samples=self.num_test_samples,
            seq_length=self.seq_length,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Creates the training data loader.

        Returns:
            TRAIN_DATALOADERS: Training data loader.
        """
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Creates the validation data loader.

        Returns:
            EVAL_DATALOADERS: Validation data loader.
        """
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Creates the testing data loader.

        Returns:
            TEST_DATALOADERS: Testing data loader.
        """
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        """
        Creates a generic data loader for the given dataset.

        Args:
            dataset: The dataset for which the data loader is created.
            **kwargs: Additional arguments for the DataLoader.

        Returns:
            DataLoader: The created data loader.
        """
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )


class _MockAVLMDataset(Dataset):
    """
    A mock dataset for AVLM, generating synthetic multimodal data.

    Attributes:
        tokenizer: Tokenizer for text inputs.
        image_processor: Processor for image inputs.
        audio_processor: Processor for audio inputs.
        name (str): Name of the dataset ('train', 'valid', 'test').
        num_samples (int): Number of samples in the dataset.
        seq_length (int): Sequence length for text tokens.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        tokenizer,
        image_processor,
        audio_processor,
        image_embedding_tokens,
        audio_embedding_tokens,
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
    ) -> None:
        """
        Initializes the mock dataset with synthetic multimodal data.

        Args:
            tokenizer: Tokenizer for text inputs.
            image_processor: Processor for image inputs.
            audio_processor: Processor for audio inputs.
            name (str): Dataset name ('train', 'valid', 'test').
            num_samples (int): Total number of samples in the dataset.
            seq_length (int): Sequence length for text tokens.
            seed (int): Random seed for data generation.
        """
        super().__init__()
        self.name = name
        self.seq_length = seq_length

        self.vocab_size = tokenizer.vocab_size

        crop_size = image_processor.crop_size
        self.image_height, self.image_width = crop_size["height"], crop_size["width"]

        self.length = num_samples
        self.seed = seed

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.image_embedding_tokens = image_embedding_tokens
        self.audio_embedding_tokens = audio_embedding_tokens

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.length

    def _get_text(self, idx: int) -> np.ndarray:
        """
        Generates synthetic text data.

        Args:
            idx (int): Index of the sample.

        Returns:
            np.ndarray: Synthetic text token IDs.
        """
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        return np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Generates a synthetic multimodal sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing synthetic tokens, images, and metadata.
        """
        # Generate data of the expected size and datatype (based on GPTDataset).
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        num_images = 2
        num_audios = 2
        tokens = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length + 1], dtype=np.int64))
        images_tokens_index = [[5, 5 + self.image_embedding_tokens], [1000, 1000 + self.image_embedding_tokens]]
        tokens[images_tokens_index[0][0] : images_tokens_index[0][1]] = ImageToken.token_id  # ImageToken token index
        tokens[images_tokens_index[1][0] : images_tokens_index[1][1]] = ImageToken.token_id  # ImageToken token index
        audios_tokens_index = [[2000, 2000 + self.audio_embedding_tokens], [4000, 4000 + self.audio_embedding_tokens]]
        tokens[audios_tokens_index[0][0] : audios_tokens_index[0][1]] = AudioToken.token_id  # AudioToken token index
        tokens[audios_tokens_index[1][0] : audios_tokens_index[1][1]] = AudioToken.token_id  # AudioToken token index
        labels = tokens.clone()
        tokens = tokens[:-1]
        labels = labels[1:]

        #  attention_mask
        attention_mask = torch.ones(len(tokens), dtype=torch.long)

        # mock images
        images = num_images * [
            torch.from_numpy(np_gen.random(size=[3, self.image_height, self.image_width], dtype=np.float32))
        ]
        processed_images = []
        num_image_tiles = []
        for image in images:
            processed_image = self.image_processor.preprocess(image, return_tensors='pt', do_rescale=False)[
                'pixel_values'
            ][0]
            processed_images.append(processed_image)
            num_image_tiles.append(processed_image.shape[0])
        # processed_images = torch.concat(processed_images, dim=0)
        processed_images = torch.stack(processed_images)
        num_image_tiles = torch.tensor(num_image_tiles, dtype=torch.long)
        image_sizes = torch.tensor(num_images * [[self.image_height, self.image_width]], dtype=torch.long)

        # mock audios
        audio_max_length = int(29.9999 * 16000)
        audios = torch.from_numpy(np_gen.uniform(-1.0, 1.0, size=[num_audios, audio_max_length]))
        # audios = torch.zeros([num_audios, audio_max_length])
        audio_lengths = torch.from_numpy(np_gen.integers(16000, audio_max_length, size=[num_audios]))

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": self.loss_mask,
            "attention_mask": attention_mask,
            "position_ids": self.position_ids,  # -> remove this
            "images": processed_images,
            "image_sizes": image_sizes,
            "num_image_tiles": num_image_tiles,
            "audios": audios,
            "audio_lengths": audio_lengths,
        }

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        collated_batch = data.dataloader.default_collate(batch)
        collated_batch['images'] = collated_batch['images'].contiguous().view(-1, *collated_batch['images'].shape[2:])
        collated_batch['num_image_tiles'] = collated_batch['num_image_tiles'].flatten()
        collated_batch['image_sizes'] = (
            collated_batch['image_sizes'].contiguous().view(-1, *collated_batch['image_sizes'].shape[2:])
        )
        collated_batch['audios'] = collated_batch['audios'].contiguous().view(-1, *collated_batch['audios'].shape[2:])
        collated_batch['audio_lengths'] = collated_batch['audio_lengths'].flatten()
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
