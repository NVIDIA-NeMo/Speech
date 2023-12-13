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
import pathlib
from typing import Callable, List, Optional, Tuple

import torch
from omegaconf.dictconfig import DictConfig
from PIL import Image

from nemo.collections.multimodal.data.clip.augmentations.augmentations import image_transform


class DirectoryBasedDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for loading images from a directory structure.
    This class inherits from torch.utils.data.Dataset.
    """

    def __init__(self, path: str, transform: Optional[Callable] = None):
        super(DirectoryBasedDataset, self).__init__()

        self._transform = transform
        self._samples = self._get_files(path, "nsfw", 1) + self._get_files(path, "safe", 0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if index >= len(self):
            raise IndexError(f"Index {index} ot of bound {len(self)}")

        sample_path, category = self._samples[index]

        image = Image.open(sample_path)

        if self._transform is not None:
            image = self._transform(image)

        return image, category

    def __len__(self) -> int:
        return len(self._samples)

    def _get_files(self, path: str, subdir: str, category: int) -> List[Tuple[str, int]]:
        globpath = pathlib.Path(path) / subdir
        return [(x, category) for x in globpath.glob("*.*")]


def build_dataset(model_cfg: DictConfig, consumed_samples: int, is_train: bool):
    """
    Builds and returns a DirectoryBasedDataset instance.
    """
    img_fn = image_transform(
        (model_cfg.vision.img_h, model_cfg.vision.img_w),
        is_train=False,
        mean=model_cfg.vision.image_mean,
        std=model_cfg.vision.image_std,
        resize_longest_max=True,
    )

    if is_train:
        path = model_cfg.data.train.dataset_path
    else:
        path = model_cfg.data.validation.dataset_path

    return DirectoryBasedDataset(path, transform=img_fn)
