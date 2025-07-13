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

from typing import List, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from nemo.lightning.pytorch.plugins import MegatronDataSampler


class MockDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for creating mock datasets for training, validation, and testing.

    Args:
        image_h (int): Height of the images in the dataset. Default is 1024.
        image_w (int): Width of the images in the dataset. Default is 1024.
        micro_batch_size (int): Micro batch size for the data sampler. Default is 4.
        global_batch_size (int): Global batch size for the data sampler. Default is 8.
        rampup_batch_size (Optional[List[int]]): Ramp-up batch size for the data sampler. Default is None.
        num_train_samples (int): Number of training samples. Default is 10,000.
        num_val_samples (int): Number of validation samples. Default is 10,000.
        num_test_samples (int): Number of testing samples. Default is 10,000.
        num_workers (int): Number of worker threads for data loading. Default is 8.
        pin_memory (bool): Whether to use pinned memory for data loading. Default is True.
        persistent_workers (bool): Whether to use persistent workers for data loading. Default is False.
        image_precached (bool): Whether the images are pre-cached. Default is False.
        text_precached (bool): Whether the text data is pre-cached. Default is False.
    """

    def __init__(
        self,
        image_h: int = 1024,
        image_w: int = 1024,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        image_precached=False,
        text_precached=False,
    ):

        super().__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.image_precached = image_precached
        self.text_precached = text_precached
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.tokenizer = None
        self.seq_length = 10

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        """
        Sets up datasets for training, validation, and testing.

        Args:
            stage (str): The stage of the process (e.g., 'fit', 'test'). Default is an empty string.
        """
        self._train_ds = _MockT2IDataset(
            image_H=1024,
            image_W=1024,
            length=self.num_train_samples,
            image_precached=self.image_precached,
            text_precached=self.text_precached,
        )
        self._validation_ds = _MockT2IDataset(
            image_H=1024,
            image_W=1024,
            length=self.num_val_samples,
            image_precached=self.image_precached,
            text_precached=self.text_precached,
        )
        self._test_ds = _MockT2IDataset(
            image_H=1024,
            image_W=1024,
            length=self.num_test_samples,
            image_precached=self.image_precached,
            text_precached=self.text_precached,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Returns the training DataLoader.

        Returns:
            TRAIN_DATALOADERS: DataLoader for the training dataset.
        """
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns the validation DataLoader.

        Returns:
            EVAL_DATALOADERS: DataLoader for the validation dataset.
        """
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns the testing DataLoader.

        Returns:
            EVAL_DATALOADERS: DataLoader for the testing dataset.
        """
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        """
        Creates a DataLoader for the given dataset.

        Args:
            dataset: The dataset to load.
            **kwargs: Additional arguments for the DataLoader.

        Returns:
            DataLoader: Configured DataLoader for the dataset.
        """
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            **kwargs,
        )


class _MockT2IDataset(Dataset):
    """
    A mock dataset class for text-to-image tasks, simulating data samples for training and testing.

    This dataset generates synthetic data for both image and text inputs, with options to use
    pre-cached latent representations or raw data. The class is designed for use in testing and
    prototyping machine learning models.

    Attributes:
        image_H (int): Height of the generated images.
        image_W (int): Width of the generated images.
        length (int): Total number of samples in the dataset.
        image_key (str): Key for accessing image data in the output dictionary.
        txt_key (str): Key for accessing text data in the output dictionary.
        hint_key (str): Key for accessing hint data in the output dictionary.
        image_precached (bool): Whether to use pre-cached latent representations for images.
        text_precached (bool): Whether to use pre-cached embeddings for text.
        prompt_seq_len (int): Sequence length for text prompts.
        pooled_prompt_dim (int): Dimensionality of pooled text embeddings.
        context_dim (int): Dimensionality of the text embedding context.
        vae_scale_factor (int): Scaling factor for the VAE latent representation.
        vae_channels (int): Number of channels in the VAE latent representation.
        latent_shape (tuple): Shape of the latent representation for images (if pre-cached).
        prompt_embeds_shape (tuple): Shape of the text prompt embeddings (if pre-cached).
        pooped_prompt_embeds_shape (tuple): Shape of pooled text embeddings (if pre-cached).
        text_ids_shape (tuple): Shape of the text token IDs (if pre-cached).

    Methods:
        __getitem__(index):
            Retrieves a single sample from the dataset based on the specified index.
        __len__():
            Returns the total number of samples in the dataset.
    """

    def __init__(
        self,
        image_H,
        image_W,
        length=100000,
        image_key='images',
        txt_key='txt',
        hint_key='hint',
        image_precached=False,
        text_precached=False,
        prompt_seq_len=256,
        pooled_prompt_dim=768,
        context_dim=4096,
        vae_scale_factor=8,
        vae_channels=16,
    ):
        super().__init__()
        self.length = length
        self.H = image_H
        self.W = image_W
        self.image_key = image_key
        self.txt_key = txt_key
        self.hint_key = hint_key
        self.image_precached = image_precached
        self.text_precached = text_precached
        if self.image_precached:
            self.latent_shape = (vae_channels, int(image_H // vae_scale_factor), int(image_W // vae_scale_factor))
        if self.text_precached:
            self.prompt_embeds_shape = (prompt_seq_len, context_dim)
            self.pooped_prompt_embeds_shape = (pooled_prompt_dim,)
            self.text_ids_shape = (prompt_seq_len, 3)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset.

        The sample can include raw image and text data or pre-cached latent representations,
        depending on the configuration.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the generated data sample. The keys and values
                  depend on whether `image_precached` and `text_precached` are set.
                  Possible keys include:
                    - 'latents': Pre-cached latent representation of the image.
                    - 'control_latents': Pre-cached control latent representation.
                    - 'images': Raw image tensor.
                    - 'hint': Hint tensor for the image.
                    - 'prompt_embeds': Pre-cached text prompt embeddings.
                    - 'pooled_prompt_embeds': Pooled text prompt embeddings.
                    - 'text_ids': Text token IDs.
                    - 'txt': Text input string (if text is not pre-cached).
        """
        item = {}
        if self.image_precached:
            item['latents'] = torch.randn(self.latent_shape)
            item['control_latents'] = torch.randn(self.latent_shape)
        else:
            item[self.image_key] = torch.randn(3, self.H, self.W)
            item[self.hint_key] = torch.randn(3, self.H, self.W)

        if self.text_precached:
            item['prompt_embeds'] = torch.randn(self.prompt_embeds_shape)
            item['pooled_prompt_embeds'] = torch.randn(self.pooped_prompt_embeds_shape)
            item['text_ids'] = torch.randn(self.text_ids_shape)
        else:
            item[self.txt_key] = "This is a sample caption input"

        return item

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples (`length` attribute).
        """
        return self.length
