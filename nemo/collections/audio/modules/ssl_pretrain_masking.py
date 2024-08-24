# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import random

import einops
import torch

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import LengthsType, NeuralType, SpectrogramType

__all__ = ['SSLPretrainWithMaskedPatch']


class SSLPretrainWithMaskedPatch(NeuralModule):
    """
    Zeroes out fixed size time patches of the spectrogram.
    All samples in batch are guaranteed to have the same amount of masked time steps.
    Note that this may be problematic when we do pretraining on a unbalanced dataset.

    For example, say a batch contains two spectrograms of length 87 and 276.
    With mask_fraction=0.7 and patch_size=10, we'll obrain mask_patches=7.
    Each of the two data will then have 7 patches of 10-frame mask.

    Args:
        patch_size (int): up to how many time steps does one patch consist of.
            Defaults to 10.
        mask_fraction (float): how much fraction in each sample to be masked (number of patches is rounded up).
            Range from 0.0 to 1.0. Defaults to 0.7.
    """

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            "input_spec": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {"augmented_spec": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType())}

    def __init__(
        self,
        patch_size: int = 10,
        mask_fraction: float = 0.7,
    ):
        super().__init__()
        self.patch_size = patch_size
        if mask_fraction > 1.0 or mask_fraction < 0.0:
            raise ValueError('mask_patches cannot be negative')
        else:
            self.mask_fraction = mask_fraction

    @typecheck()
    def forward(self, input_spec, length):
        """
        Apply Patched masking on the input_spec.


        During the training stage, the mask is generated randomly, with
        approximately `self.mask_fraction` of the time frames being masked out.

        In the validation stage, the masking pattern is fixed to ensure
        consistent evaluation of checkpoints and to prevent overfitting. Note
        that the same masking pattern is applied to all data, regardless of
        their lengths. On average, approximately `self.mask_fraction` of the
        time frames will be masked out.

        """
        augmented_spec = input_spec

        min_len = torch.min(length)
        if self.training:
            len_fraction = int(min_len * self.mask_fraction)
            mask_patches = len_fraction // self.patch_size + int(len_fraction % self.patch_size != 0)

            if min_len < self.patch_size * mask_patches:
                mask_patches = min_len // self.patch_size

            for idx, cur_len in enumerate(length.tolist()):
                patches = range(cur_len // self.patch_size)
                masked_patches = random.sample(patches, mask_patches)
                for mp in masked_patches:
                    augmented_spec[idx, :, :, mp * self.patch_size : (mp + 1) * self.patch_size] = 0.0
        else:
            chunk_length = self.patch_size // self.mask_fraction
            mask = torch.arange(augmented_spec.size(-1), device=augmented_spec.device)
            mask = (mask % chunk_length) >= self.patch_size
            mask = einops.rearrange(mask, 'T -> 1 1 1 T').float()
            augmented_spec = augmented_spec * mask

        return augmented_spec
