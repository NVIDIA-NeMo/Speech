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
import os
from pathlib import Path

import torch
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from torchvision import transforms

    TORCHVISION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TORCHVISION_AVAILABLE = False


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.

    :param instance_data_root: required, a directory with images files of the object
    :param instance_prompt: captions with special token associated with instance images
    :param with_prior_preservation: whether to regularize the model finetuning with the original inference output from the backbone
    :param reg_data_root: a directory to save inference images from the backbone
    :param  reg_prompt: prompt used to generate regularization images
    :param size: resizing images for training data pipeline
    :param center_crop: whether performing center cropping on input images
    :param load_cache_latents: when set to True, images will be converted to cached latents which will be directly loaded for training
    :param vae: vae instance to encode imamges from pixel space to latent space
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        with_prior_preservation=False,
        reg_data_root=None,
        reg_prompt=None,
        size=512,
        center_crop=True,
        repeat=10000,
        load_cache_latents=False,
        cached_instance_data_root=None,
        cached_reg_data_root=None,
        vae=None,
        text_encoder=None,
    ):
        self.size = size
        self.center_crop = center_crop

        assert instance_data_root or cached_instance_data_root, "must provide instance images to start training."
        self.instance_data_root = Path(instance_data_root)
        self.cached_instance_data_root = cached_instance_data_root
        self.cached_reg_data_root = cached_reg_data_root

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images * repeat
        self.load_cache_latents = load_cache_latents
        self.with_prior_preservation = with_prior_preservation

        if reg_data_root is not None:
            self.reg_data_root = Path(reg_data_root)
            self.reg_images_path = list(self.reg_data_root.iterdir())
            self.num_reg_images = len(self.reg_images_path)
            self.reg_prompt = reg_prompt
        else:
            self.reg_data_root = None

        assert TORCHVISION_AVAILABLE, "Torchvision imports failed but they are required."
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        if self.load_cache_latents:
            if (self.cached_instance_data_root is None) or (
                self.with_prior_preservation and self.cached_reg_data_root is None
            ):
                self.cache_latents(vae, text_encoder)

                self.cached_instance_data_root = f'{self.instance_data_root}_cached'
                self.cached_reg_data_root = f'{self.reg_data_root}_cached'
                self.instance_images_path = list(Path(self.cached_instance_data_root).iterdir())
                self.num_instance_images = len(self.instance_images_path)

                if self.with_prior_preservation:
                    self.reg_images_path = list(Path(self.cached_reg_data_root).iterdir())
                    self.num_reg_images = len(self.reg_images_path)

            if self.cached_instance_data_root:
                self.instance_images_path = list(Path(self.cached_instance_data_root).iterdir())
                self.num_instance_images = len(self.instance_images_path)
            if self.with_prior_preservation and self.cached_reg_data_root:
                self.reg_images_path = list(Path(self.cached_reg_data_root).iterdir())
                self.num_reg_images = len(self.reg_images_path)

    def __len__(self):
        return self._length

    def get_image(self, path):
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.image_transforms(image)
        return image

    def __getitem__(self, index):
        example = {}
        if self.load_cache_latents:
            example["instance_images"] = torch.load(self.instance_images_path[index % self.num_instance_images])
        else:
            example["instance_images"] = self.get_image(self.instance_images_path[index % self.num_instance_images])
        example["instance_prompt"] = self.instance_prompt

        if self.reg_data_root:
            if self.load_cache_latents:
                example["reg_images"] = torch.load(self.reg_images_path[index % self.num_reg_images])
            else:
                example["reg_images"] = self.get_image(self.reg_images_path[index % self.num_reg_images])
            example["reg_prompt"] = self.reg_prompt

        return example

    @rank_zero_only
    def cache_latents(self, vae, text_encoder):
        os.makedirs(f'{self.instance_data_root}_cached', exist_ok=True)
        self.cached_instance_data_root = f'{self.instance_data_root}_cached'
        self.cached_reg_data_root = f'{self.reg_data_root}_cached'
        if self.instance_data_root and (len(os.listdir(self.cached_instance_data_root)) < self.num_instance_images):
            for i in tqdm(range(self.num_instance_images)):
                x = torch.Tensor(self.get_image(self.instance_images_path[i % self.num_instance_images]))
                x = torch.unsqueeze(x, dim=0)
                params = vae.encode(x).parameters.squeeze(dim=0)
                torch.save(params, f'{self.instance_data_root}_cached/instance_image_cache_{i}.pt')

        if self.with_prior_preservation:
            os.makedirs(f'{self.reg_data_root}_cached', exist_ok=True)
            if self.reg_data_root and (len(os.listdir(self.cached_reg_data_root)) < self.num_reg_images):
                for i in tqdm(range(self.num_reg_images)):
                    x = torch.Tensor(self.get_image(self.reg_images_path[i % self.num_reg_images]))
                    x = torch.unsqueeze(x, dim=0)
                    params = vae.encode(x).parameters.squeeze(dim=0)
                    torch.save(params, f'{self.reg_data_root}_cached/reg_image_cache_{i}.pt')
