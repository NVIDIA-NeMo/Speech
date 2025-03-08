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
from transformers import AutoProcessor, LlavaNextConfig

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.vlm.llava_next.model.utils import get_number_of_features
from nemo.collections.vlm.neva.data.multimodal_tokens import IMAGE_TOKEN_INDEX
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging


class MockDataModule(pl.LightningDataModule):
    """
    A mock data module for LLaVA-Next training, validation, and testing.

    Provides datasets and data loaders for training, validation, and testing phases.
    Includes data sampling and preprocessing for multimodal tasks.
    """

    def __init__(
        self,
        seq_length: int = 2048,
        decoder_seq_length: Optional[int] = None,
        tokenizer=None,
        image_processor=None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
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
        model_name = ''
        processor = None
        if tokenizer is None or image_processor is None:
            logging.warning(
                "Processor or tokenizer are not provided! Fall back to `llava-hf/llava-v1.6-vicuna-7b-hf`."
            )

            model_name = "llava-hf/llava-v1.6-vicuna-7b-hf"

            processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = tokenizer or AutoTokenizer(model_name)
        self.image_processor = image_processor or processor.image_processor
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
        self._train_ds = _MockLlavaNextDataset(
            self.tokenizer, self.image_processor, "train", self.num_train_samples, self.seq_length
        )
        self._validation_ds = _MockLlavaNextDataset(
            self.tokenizer, self.image_processor, "valid", self.num_val_samples, self.seq_length
        )
        self._test_ds = _MockLlavaNextDataset(
            self.tokenizer, self.image_processor, "test", self.num_test_samples, self.seq_length
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


class _MockLlavaNextDataset(Dataset):
    """
    A mock dataset for LLaVA-Next, generating synthetic multimodal data.

    Attributes:
        tokenizer: Tokenizer for text inputs.
        image_processor: Processor for image inputs.
        name (str): Name of the dataset ('train', 'valid', 'test').
        num_samples (int): Number of samples in the dataset.
        seq_length (int): Sequence length for text tokens.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        tokenizer,
        image_processor,
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

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.hf_config = LlavaNextConfig()

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
        tokens = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length + 1], dtype=np.int64))
        num_image_tokens = get_number_of_features(
            self.image_height,
            self.image_width,
            self.image_height,
            self.image_width,
            self.hf_config.image_grid_pinpoints,
            self.hf_config.vision_config.patch_size,
        )
        tokens = torch.concatenate([tokens[:2], torch.tensor([IMAGE_TOKEN_INDEX] * num_image_tokens), tokens[2:]], 0)
        labels = tokens.clone()
        images = torch.from_numpy(np_gen.random(size=[3, self.image_height, self.image_width], dtype=np.float32))
        tokens = tokens[:-1]
        labels = labels[1:]

        seq_length = len(tokens)
        loss_mask = torch.ones(seq_length, dtype=torch.float)
        position_ids = torch.arange(seq_length, dtype=torch.int64)

        #  attention_mask, image_sizes, num_media_tiles required for llava-next. Neva model will ignore these
        attention_mask = torch.ones(len(tokens), dtype=torch.long)
        image_sizes = torch.tensor([[self.image_height, self.image_width]], dtype=torch.long)
        image_array = self.image_processor.preprocess(images, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        num_media_tiles = image_array.shape[0]
        return {
            "media": image_array,
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "image_sizes": image_sizes,
            "num_media_tiles": num_media_tiles,
            "attention_mask": attention_mask,
        }

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        collated_batch = data.dataloader.default_collate(batch)

        collated_batch['media'] = collated_batch['media'].contiguous().view(-1, *collated_batch['media'].shape[2:])
        collated_batch['image_sizes'] = (
            collated_batch['image_sizes'].contiguous().view(-1, *collated_batch['image_sizes'].shape[2:])
        )
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
