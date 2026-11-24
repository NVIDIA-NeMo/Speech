# ! /usr/bin/python
# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import torch
from torch import Tensor, nn

from nemo.collections.asr.parts.packed_sequence import PackedEncoderActivations, unpack_encoder_output
from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.triton.ctc_loss import PACKED_CTC_LOSS_DTYPES, packed_ctc_loss

__all__ = ['CTCLoss']


class CTCLoss(nn.CTCLoss, Serialization, Typing):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss."""
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
            "target_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, zero_infinity=False, reduction='mean_batch'):
        self._blank = num_classes
        # Don't forget to properly call base constructor
        if reduction not in ['none', 'mean', 'sum', 'mean_batch', 'mean_volume']:
            raise ValueError('`reduction` must be one of [mean, sum, mean_batch, mean_volume]')

        self.config_reduction = reduction
        if reduction == 'mean_batch' or reduction == 'mean_volume':
            ctc_reduction = 'none'
            self._apply_reduction = True
        elif reduction in ['sum', 'mean', 'none']:
            ctc_reduction = reduction
            self._apply_reduction = False
        super().__init__(blank=self._blank, reduction=ctc_reduction, zero_infinity=zero_infinity)

    def reduce(self, losses, target_lengths):
        if self.config_reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.config_reduction == 'mean_volume':
            losses = losses.sum() / target_lengths.sum()  # same as above but longer samples weigh more

        return losses

    def forward_packed(self, log_probs: PackedEncoderActivations, targets: Tensor, target_lengths: Tensor) -> Tensor:
        """Compute the CTC loss from token-flat log probabilities with explicit sequence boundaries.

        On CUDA, for float32, bfloat16 and float16 log probabilities,
        the loss runs as a kernel over the token-flat log probabilities
        (:func:`nemo.collections.asr.parts.triton.ctc_loss.packed_ctc_loss`),
        so neither the padded ``(B, T, D)`` log probabilities nor their gradient are formed.
        Elsewhere, and for float64 (the kernel computes in float32),
        the padded log probabilities are materialized for the padded kernel of ``forward``.
        Both give the loss and the gradient of ``forward``;
        ``reduction`` and ``zero_infinity`` apply unchanged,
        and ``targets`` and ``target_lengths`` may live on the CPU, as for ``forward``.

        Args:
            log_probs: concatenated per-utterance log probabilities, shape ``(sum(T_i), D)`` including the blank.
            targets: padded target ids, shape ``(B, U)``.
            target_lengths: number of target ids per utterance, shape ``(B,)``.

        Returns:
            The loss as ``forward`` returns it for the configured reduction.
        """
        if not isinstance(log_probs, PackedEncoderActivations):
            raise TypeError(
                f"forward_packed expects PackedEncoderActivations as log_probs, got {type(log_probs).__name__}; "
                "padded log probabilities go to forward."
            )
        if log_probs.data.is_cuda and TRITON_AVAILABLE and log_probs.data.dtype in PACKED_CTC_LOSS_DTYPES:
            # the kernel reads the targets and their lengths on the device of the log probabilities,
            # and the reductions below divide by the lengths, so both move there
            targets = targets.to(device=log_probs.data.device, dtype=torch.long)
            target_lengths = target_lengths.to(device=log_probs.data.device, dtype=torch.long)
            losses = packed_ctc_loss(
                log_probs.data,
                log_probs.cu_seqlens,
                log_probs.lengths,
                targets,
                target_lengths,
                blank=self._blank,
                zero_infinity=self.zero_infinity,
                bounded=log_probs.bounded,
            )
            # a bounded batch has unused utterance slots of length zero: the batch means count the utterances in use
            batch_size = (log_probs.lengths > 0).sum().clamp(min=1) if log_probs.bounded else None
            if self.reduction == 'sum':
                loss = losses.sum()
            elif self.reduction == 'mean':
                # the padded kernel's mean: each loss over its target length, then the batch mean
                normalized = losses / target_lengths.clamp(min=1)
                loss = normalized.mean() if batch_size is None else normalized.sum() / batch_size
            elif self.reduction == 'none':
                loss = losses
            else:
                raise ValueError(f"unexpected reduction {self.reduction!r}")
            if self._apply_reduction:
                if batch_size is not None and self.config_reduction == 'mean_batch':
                    loss = loss.sum() / batch_size
                else:
                    loss = self.reduce(loss, target_lengths)
            return loss
        padded, input_lengths = unpack_encoder_output(log_probs), log_probs.lengths
        if log_probs.bounded and (self.reduction != 'none' or self._apply_reduction):
            # this fallback reads the host anyway: the unused utterance slots leave the batch, so the batch means
            # count the utterances in use as the kernel path does (reduction 'none' keeps one loss per slot)
            keep = input_lengths > 0
            padded, input_lengths = padded[keep], input_lengths[keep]
            targets, target_lengths = targets[keep.to(targets.device)], target_lengths[keep.to(target_lengths.device)]
        return self.forward(
            log_probs=padded, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths
        )

    @typecheck()
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # override forward implementation
        # custom logic, if necessary
        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()
        targets = targets.long()
        # here we transpose because we expect [B, T, D] while PyTorch assumes [T, B, D]
        log_probs = log_probs.transpose(1, 0)
        loss = super().forward(
            log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths
        )
        if self._apply_reduction:
            loss = self.reduce(loss, target_lengths)
        return loss
