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


import math
from typing import Optional, Union

import torch
import torch.nn as nn

from nemo.core.classes import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType


class RandomBlockMasking(NeuralModule):
    """
    Performs random block masking on sequence of features.
    Args:
        mask_prob (float): percentage of sequence to mask
        block_size (int): size of each block to mask
        mask_value (Optional[float]): value to use for masking, if None, use random values
        feat_in (Optional[int]): size of input features, required if mask_value is None
        freeze (bool): if True, mask embedding is not trainable
        allow_overlap (bool): if True, masked blocks can overlap
    """

    def __init__(
        self,
        feat_in: int,
        mask_prob: float = 0.5,
        block_size: int = 48,
        mask_value: Optional[float] = None,
        freeze: bool = True,
        allow_overlap: bool = False,
        max_mask_ratio: float = 0.8,
    ):
        super().__init__()
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.allow_overlap = allow_overlap
        self.max_mask_ratio = max_mask_ratio

        if mask_value is None:
            self.mask_embedding = nn.Parameter(torch.FloatTensor(feat_in))
            nn.init.normal_(self.mask_embedding, mean=0.0, std=0.1)
        else:
            self.mask_embedding = nn.Parameter(torch.ones(feat_in) * mask_value, requires_grad=False)
        if freeze:
            self.freeze()

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            "input_feats": NeuralType(("B", "D", "T"), AcousticEncodedRepresentation()),
            "input_lengths": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {
            "maksed_feats": NeuralType(("B", "D", "T"), AcousticEncodedRepresentation()),
            "masks": NeuralType(("B", "D", "T"), AcousticEncodedRepresentation()),
        }

    def forward(self, input_feats: torch.Tensor, input_lengths: torch.Tensor):
        """
        Args:
            input_feats (Tensor): input sequence features, shape=(batch, features, time)
            input_length (Tensor): length of each sequence in the batch, shape=(batch)
        Returns:
            masked_feats (Tensor): masked features, shape=(batch, features, time)
            masks (Tensor): the generated masks, shape=(batch, features, time)
        """
        if self.allow_overlap:
            return self.forward_with_overlap(input_feats, input_lengths)
        else:
            return self.forward_without_overlap(input_feats, input_lengths)

    def forward_without_overlap(self, input_feats, input_lengths):
        batch_size, _, max_time = input_feats.shape

        num_patches = torch.ceil(input_lengths * self.mask_prob / self.block_size).long()
        block_sizes = torch.full_like(input_lengths, self.block_size)
        needs_shrink = (num_patches + 1) * block_sizes > input_lengths
        block_sizes = torch.where(
            needs_shrink,
            input_lengths // (num_patches + 1),
            block_sizes,
        ).clamp_min(1)

        num_slots = (input_lengths // block_sizes - 1).clamp_min(0)
        num_patches = torch.minimum(num_patches, num_slots)

        slots = torch.arange(max_time, device=input_feats.device)
        valid_slots = slots.unsqueeze(0) < num_slots.unsqueeze(1)
        scores = torch.rand(batch_size, max_time, device=input_feats.device)
        scores.masked_fill_(~valid_slots, float("-inf"))

        max_patches = math.ceil(max_time * self.mask_prob / self.block_size)
        selected_slots = scores.topk(max_patches, dim=1).indices
        selected = torch.arange(max_patches, device=input_feats.device).unsqueeze(0) < num_patches.unsqueeze(1)

        offsets = (torch.rand(batch_size, device=input_feats.device) * block_sizes).long()
        starts = selected_slots * block_sizes.unsqueeze(1) + offsets.unsqueeze(1)
        ends = starts + block_sizes.unsqueeze(1)
        starts = torch.where(selected, starts, 0)
        ends = torch.where(selected, ends, 0)
        deltas = selected.int()

        coverage_diff = torch.zeros(
            batch_size,
            max_time + 1,
            dtype=torch.int32,
            device=input_feats.device,
        )
        coverage_diff.scatter_add_(1, starts, deltas)
        coverage_diff.scatter_add_(1, ends, -deltas)
        time_mask = coverage_diff[:, :max_time].cumsum(dim=1) > 0
        time_mask = time_mask.unsqueeze(1)

        masked_feats = torch.where(
            time_mask,
            self.mask_embedding.view(1, -1, 1),
            input_feats,
        )
        masks = time_mask.to(input_feats.dtype).expand_as(input_feats)
        return masked_feats, masks

    def forward_with_overlap(self, input_feats, input_lengths):
        batch_size, _, max_time = input_feats.shape
        positions = torch.arange(max_time, device=input_feats.device)

        valid_starts = positions + self.block_size <= input_lengths.unsqueeze(1)
        start_mask = (torch.rand(batch_size, max_time, device=input_feats.device) < self.mask_prob) & valid_starts
        time_mask = torch.nn.functional.max_pool1d(
            torch.nn.functional.pad(start_mask.unsqueeze(1).float(), (self.block_size - 1, 0)),
            kernel_size=self.block_size,
            stride=1,
        ).bool()

        masked_feats = torch.where(
            time_mask,
            self.mask_embedding.view(1, -1, 1),
            input_feats,
        )
        masks = time_mask.to(input_feats.dtype).expand_as(input_feats)
        return masked_feats, masks


class ConvFeatureMaksingWrapper(NeuralModule):
    """
    A wrapper module that applies masking to the features after subsampling layer of ConformerEncoder.
    """

    def __init__(self, pre_encode_module: nn.Module, masking_module: Union[nn.Module, NeuralModule]) -> None:
        """
        Args:
            pre_encode_module: the pre_encode module of the ConformerEncoder instance
            masking_module: the module that performs masking on the extracted features
        """
        super().__init__()
        self.pre_encode = pre_encode_module
        self.masking = masking_module
        self.curr_mask = None
        self.curr_feat = None
        self.apply_mask = False

    def forward(self, x, lengths):
        """
        Same interface as ConformerEncoder.pre_encode
        """
        feats, lengths = self.pre_encode(x=x, lengths=lengths)
        self.curr_feat = feats.detach()
        if self.apply_mask:
            feats = feats.transpose(1, 2)
            masked_feats, self.curr_mask = self.masking(input_feats=feats, input_lengths=lengths)
            masked_feats = masked_feats.transpose(1, 2).detach()
        else:
            masked_feats = feats
            self.curr_mask = torch.zeros_like(feats)
        return masked_feats, lengths

    def set_masking_enabled(self, apply_mask: bool):
        self.apply_mask = apply_mask

    def get_current_mask(self):
        return self.curr_mask

    def get_current_feat(self):
        return self.curr_feat
