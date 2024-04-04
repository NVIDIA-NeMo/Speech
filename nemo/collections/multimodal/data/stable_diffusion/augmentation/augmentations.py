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
try:
    import torchvision.transforms as transforms

    TORCHVISION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TORCHVISION_AVAILABLE = False
import numpy as np
import torch


def construct_clip_augmentations(n_px=224):
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    assert TORCHVISION_AVAILABLE, "Torchvision imports failed but they are required."
    return transforms.Compose(
        [
            transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def construct_image_augmentations(augmentation_dict, normalize=True):
    train_img_transform = []
    for aug in augmentation_dict:
        if aug == 'resize_smallest_side':
            img_size = int(augmentation_dict[aug])
            train_img_transform.append(
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            )

        elif aug == 'center_crop_h_w':
            img_w, img_h = augmentation_dict[aug].split(',')
            img_w = int(img_w)
            img_h = int(img_h)
            train_img_transform.append(transforms.CenterCrop((img_w, img_h)))

        elif aug == 'random_crop_h_w':
            img_w, img_h = augmentation_dict[aug].split(',')
            img_w = int(img_w)
            img_h = int(img_h)
            train_img_transform.append(transforms.RandomCrop((img_w, img_h)))

        elif aug == 'horizontal_flip':
            enabled = augmentation_dict[aug]
            if enabled:
                train_img_transform.append(transforms.RandomHorizontalFlip(p=0.5))
        else:
            raise ValueError('Augmentation not supported')

    # Always need to convert data to tensor
    train_img_transform.append(transforms.ToTensor())
    if normalize:
        train_img_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    train_img_transform = transforms.Compose(train_img_transform)
    return train_img_transform


def identical_transform(x):
    return x
