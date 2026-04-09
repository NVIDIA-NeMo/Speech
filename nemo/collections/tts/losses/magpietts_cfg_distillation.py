# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""
Losses used in CFG distillation of the MagpieTTS model.
"""

from typing import Generator, Optional

import torch
from torch import Tensor, nn

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LogitsType, LossType, MaskType, NeuralType

__all__ = [
    "KLDivergenceLoss",
    "CodesCrossEntropyLoss",
    "NRMSELogitsLoss",
]


def _iter_slices(
    num_codebooks: int,
    num_tokens_per_codebook: int,
    frame_stacking_factor: int,
    mask: Tensor,
) -> Generator[tuple[int, int, int, int, Tensor, Tensor], None, None]:
    for fs_index in range(frame_stacking_factor):
        slice_mask = mask[:, fs_index::frame_stacking_factor].float()
        slice_len = slice_mask.sum(dim=-1).clamp_min(1)
        offset = num_codebooks * fs_index * num_tokens_per_codebook

        for codebook in range(num_codebooks):
            start = offset + codebook * num_tokens_per_codebook
            end = start + num_tokens_per_codebook

            yield fs_index, codebook, start, end, slice_mask, slice_len


class KLDivergenceLoss(Loss):
    """The Kullback-Leibler divergence loss."""

    @property
    def input_types(self) -> dict[str, NeuralType]:
        """Define definitions of module input ports.

        Returns:
            dict[str, NeuralType]: A dictionary describing expected input tensors.
        """
        return {
            "student_logits": NeuralType(("B", "T", "D"), LogitsType()),
            "teacher_logits": NeuralType(("B", "T", "D"), LogitsType()),
            "mask": NeuralType(("B", "T"), MaskType()),
            "sample_weights": NeuralType(tuple("B"), MaskType(), optional=True),
        }

    @property
    def output_types(self) -> dict[str, NeuralType]:
        """Define definitions of module output ports.

        Returns:
            dict[str, NeuralType]: A dictionary describing expected output tensors.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        num_codebooks: int,
        num_tokens_per_codebook: int,
        frame_stacking_factor: int,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_tokens_per_codebook = num_tokens_per_codebook
        self.frame_stacking_factor = frame_stacking_factor
        self.criterion = nn.KLDivLoss(reduction="none", log_target=False)

    @typecheck()
    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        mask: Tensor,
        sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the Kullback-Leibler divergence loss between student and teacher logits.

        Args:
            student_logits (Tensor): Student logits of shape `(B, T', D)`, where `B` is batch size,
                `T'` is the frame-stacked sequence length, and `D` is the concatenated logit dimension
                across all codebooks and frame-stacking positions.
            teacher_logits (Tensor): Teacher logits of shape `(B, T', D)`.
            mask (Tensor): Binary mask of shape `(B, T)` over the unstacked time dimension. For each
                frame-stacking position, the corresponding stacked-time mask is obtained by slicing.
            sample_weights (Optional[Tensor]): Optional per-sample weighting factors of shape `(B,)`.
                If provided, these weights scale the per-sample loss contribution before averaging.
                If `None`, all samples contribute equally.

        Returns:
            Tensor: Scalar tensor representing the averaged masked KL divergence loss.
        """
        loss = 0.0
        student_log_probs = student_logits.log_softmax(dim=-1)
        teacher_probs = teacher_logits.softmax(dim=-1)

        for _, _, start, end, slice_mask, slice_len in _iter_slices(
            self.num_codebooks,
            self.num_tokens_per_codebook,
            self.frame_stacking_factor,
            mask,
        ):
            teacher_probs_slice = teacher_probs[:, :, start:end]
            student_log_probs_slice = student_log_probs[:, :, start:end]
            slice_loss = self.criterion(input=student_log_probs_slice, target=teacher_probs_slice)
            slice_loss = slice_loss.sum(dim=-1)
            slice_loss = (slice_loss * slice_mask).sum(dim=-1) / slice_len
            loss = loss + slice_loss

        loss = loss / (self.num_codebooks * self.frame_stacking_factor)

        if sample_weights is not None:
            loss = loss * sample_weights

        return loss.mean()


class CodesCrossEntropyLoss(Loss):
    """Cross-entropy loss that supports time masks."""

    @property
    def input_types(self) -> dict[str, NeuralType]:
        """Define definitions of module input ports.

        Returns:
            dict[str, NeuralType]: A dictionary describing expected input tensors.
        """
        return {
            "predicted_logits": NeuralType(("B", "T", "D"), LogitsType()),
            "target_codes": NeuralType(("B", "C", "T"), LabelsType()),
            "mask": NeuralType(("B", "T"), MaskType()),
            "sample_weights": NeuralType(tuple("B"), MaskType(), optional=True),
        }

    @property
    def output_types(self) -> dict[str, NeuralType]:
        """Define definitions of module output ports.

        Returns:
            dict[str, NeuralType]: A dictionary describing expected output tensors.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        num_codebooks: int,
        num_tokens_per_codebook: int,
        frame_stacking_factor: int,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_tokens_per_codebook = num_tokens_per_codebook
        self.frame_stacking_factor = frame_stacking_factor
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    @typecheck()
    def forward(
        self,
        predicted_logits: Tensor,
        target_codes: Tensor,
        mask: Tensor,
        sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute cross-entropy loss for discretized code sequences with frame stacking and time masking.

        Args:
            predicted_logits (Tensor): Predicted logits of shape `(B, T', D)`, where `B` is batch size,
                `T'` is the frame-stacked sequence length, and `D` is the concatenated logit dimension
                across all codebooks and frame-stacking positions.
            target_codes (Tensor): Target code indices of shape `(B, C, T)`, where `C` is the number
                of codebooks and `T` is the unstacked time dimension.
            mask (Tensor): Binary mask of shape `(B, T)` over the unstacked time dimension.
            sample_weights (Optional[Tensor]): Optional per-sample weighting factors of shape `(B,)`.
                If provided, these weights scale the per-sample loss contribution before averaging.
                If `None`, all samples contribute equally.

        Returns:
            Tensor: Scalar tensor representing the averaged masked cross-entropy loss.
        """
        loss = 0.0

        for fs_index, codebook, start, end, slice_mask, slice_len in _iter_slices(
            self.num_codebooks,
            self.num_tokens_per_codebook,
            self.frame_stacking_factor,
            mask,
        ):
            target_slice = target_codes[:, codebook, fs_index :: self.frame_stacking_factor]
            logits_slice = predicted_logits[:, :, start:end].permute(0, 2, 1)
            slice_loss = self.criterion(input=logits_slice, target=target_slice)
            slice_loss = (slice_loss * slice_mask).sum(dim=-1) / slice_len
            loss = loss + slice_loss

        loss = loss / (self.num_codebooks * self.frame_stacking_factor)

        if sample_weights is not None:
            loss = loss * sample_weights

        return loss.mean()


class NRMSELogitsLoss(Loss):
    """Normalized Root Mean Square Error (NRMSE) loss applied to raw logits."""

    @property
    def input_types(self) -> dict[str, NeuralType]:
        """Define definitions of module input ports.

        Returns:
            dict[str, NeuralType]: A dictionary describing expected input tensors.
        """
        return {
            "student_logits": NeuralType(("B", "T", "D"), LogitsType()),
            "teacher_logits": NeuralType(("B", "T", "D"), LogitsType()),
            "mask": NeuralType(("B", "T"), MaskType()),
            "sample_weights": NeuralType(tuple("B"), MaskType(), optional=True),
        }

    @property
    def output_types(self) -> dict[str, NeuralType]:
        """Define definitions of module output ports.

        Returns:
            dict[str, NeuralType]: A dictionary describing expected output tensors.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        num_codebooks: int,
        num_tokens_per_codebook: int,
        frame_stacking_factor: int,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_tokens_per_codebook = num_tokens_per_codebook
        self.frame_stacking_factor = frame_stacking_factor
        self.eps = 1e-8
        self.criterion = nn.MSELoss(reduction="none")

    @typecheck()
    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        mask: Tensor,
        sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the normalized RMSE loss between student and teacher logits.

        Args:
            student_logits (Tensor): Student logits of shape `(B, T', D)`, where `B` is batch size,
                `T'` is the frame-stacked sequence length, and `D` is the concatenated logit dimension
                across all codebooks and frame-stacking positions.
            teacher_logits (Tensor): Teacher logits of shape `(B, T', D)`.
            mask (Tensor): Binary mask of shape `(B, T)` over the unstacked time dimension.
            sample_weights (Optional[Tensor]): Optional per-sample weighting factors of shape `(B,)`.
                If provided, these weights scale the per-sample loss contribution before averaging.
                If `None`, all samples contribute equally.

        Returns:
            Tensor: Scalar tensor representing the averaged masked normalized RMSE loss.
        """
        inf_mask = torch.isinf(teacher_logits) | torch.isinf(student_logits)
        teacher_logits = teacher_logits.masked_fill(inf_mask, 0.0)
        student_logits = student_logits.masked_fill(inf_mask, 0.0)
        loss = 0.0

        for _, _, start, end, slice_mask, slice_len in _iter_slices(
            self.num_codebooks,
            self.num_tokens_per_codebook,
            self.frame_stacking_factor,
            mask,
        ):
            student_logits_slice = student_logits[:, :, start:end]
            teacher_logits_slice = teacher_logits[:, :, start:end]
            slice_loss = self.criterion(input=student_logits_slice, target=teacher_logits_slice)
            slice_loss = torch.sqrt(slice_loss.mean(dim=-1))
            norm = teacher_logits_slice.std(dim=-1).clamp_min(self.eps)
            slice_loss = slice_loss / norm
            slice_loss = (slice_loss * slice_mask).sum(dim=-1) / slice_len
            loss = loss + slice_loss

        loss = loss / (self.num_codebooks * self.frame_stacking_factor)

        if sample_weights is not None:
            loss = loss * sample_weights

        return loss.mean()
