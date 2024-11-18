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

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from nemo.collections.diffusion.models.model import DiTConfig
from nemo.lightning.pytorch.plugins import MegatronDataSampler

from .diffusion_taskencoder import pos_id_3d


class PosEmb3D:
    """Generates and provides 3D positional embeddings for video data."""

    def __init__(self, *, max_t=96, max_h=960, max_w=960):
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.generate_pos_id()

    def generate_pos_id(self):
        """Generates the positional ID grid based on max_t, max_h, and max_w."""
        self.grid = torch.stack(
            torch.meshgrid(
                torch.arange(self.max_t, device='cpu'),
                torch.arange(self.max_h, device='cpu'),
                torch.arange(self.max_w, device='cpu'),
            ),
            dim=-1,
        )

    def get_pos_id_3d(self, *, t, h, w):
        """Retrieves a subset of the positional IDs for the specified dimensions.

        Parameters:
            t (int): Number of time frames.
            h (int): Height dimension.
            w (int): Width dimension.

        Returns:
            torch.Tensor: The positional IDs tensor with shape (t, h, w, 3).
        """
        if t > self.max_t or h > self.max_h or w > self.max_w:
            self.max_t = max(self.max_t, t)
            self.max_h = max(self.max_h, h)
            self.max_w = max(self.max_w, w)
            self.generate_pos_id()
        return self.grid[:t, :h, :w]


class DiTVideoLatentFakeDataset(torch.utils.data.Dataset):
    """A fake dataset for generating synthetic video latent data."""

    def __init__(
        self,
        n_frames,
        max_h,
        max_w,
        patch_size,
        in_channels,
        crossattn_emb_size,
        max_text_seqlen=512,
        seq_length=8192,
    ):
        self.max_t = n_frames
        self.max_height = max_h
        self.max_width = max_w
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.text_dim = crossattn_emb_size
        self.text_seqlen = max_text_seqlen
        self.seq_length = seq_length

    def __len__(self):
        """Returns the total number of samples."""
        return 100000000

    def __getitem__(self, idx):
        """Generates a single sample of data.

        Parameters:
            idx (int): Index of the data sample.

        Returns:
            dict: A dictionary containing video latent data and related information.
        """
        t = self.max_t
        h = self.max_height
        w = self.max_width
        p = self.patch_size
        c = self.in_channels

        video_latent = torch.ones(self.seq_length, c * p**2, dtype=torch.bfloat16) * 0.5
        text_embedding = torch.randn(self.text_seqlen, self.text_dim, dtype=torch.bfloat16)
        pos_emb = pos_id_3d.get_pos_id_3d(t=t, h=h // p, w=w // p).reshape(-1, 3)

        return {
            'video': video_latent,
            't5_text_embeddings': text_embedding,
            'seq_len_q': torch.tensor([video_latent.shape[0]], dtype=torch.int32).squeeze(),
            'seq_len_kv': torch.tensor([self.text_seqlen], dtype=torch.int32).squeeze(),
            'pos_ids': torch.zeros((self.seq_length, 3), dtype=torch.int32),
            'loss_mask': torch.ones(video_latent.shape[0], dtype=torch.bfloat16),
        }

    def _collate_fn(self, batch):
        """A default implementation of a collation function.

        Users should override this method to define custom data loaders.
        """
        return torch.utils.data.dataloader.default_collate(batch)

    def collate_fn(self, batch):
        """Method that user passes as a functor to DataLoader.

        The method optionally performs neural type checking and adds types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        Usage:
            dataloader = torch.utils.data.DataLoader(
                    ....,
                    collate_fn=dataset.collate_fn,
                    ....
            )

        Returns:
            Collated batch, with or without types.
        """
        return self._collate_fn(batch)


class VideoLatentFakeDataModule(pl.LightningDataModule):
    """A LightningDataModule for generating fake video latent data for training."""

    def __init__(
        self,
        model_config: DiTConfig,
        seq_length: int = 2048,
        micro_batch_size: int = 1,
        global_batch_size: int = 8,
        num_workers: int = 1,
        pin_memory: bool = True,
        task_encoder=None,
        use_train_split_for_val: bool = False,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.model_config = model_config

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        """Sets up the dataset for training and validation.

        Parameters:
            stage (str): Optional stage argument (unused).
        """
        self._train_ds = DiTVideoLatentFakeDataset(
            n_frames=self.model_config.max_frames,
            max_h=self.model_config.max_img_h,
            max_w=self.model_config.max_img_w,
            patch_size=self.model_config.patch_spatial,
            in_channels=self.model_config.in_channels,
            crossattn_emb_size=self.model_config.crossattn_emb_size,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Returns the training DataLoader."""
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Returns the validation DataLoader."""
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        """Creates a DataLoader for the given dataset.

        Parameters:
            dataset (Dataset): The dataset to load.
            **kwargs: Additional arguments for DataLoader.

        Returns:
            DataLoader: The DataLoader instance.
        """
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )
