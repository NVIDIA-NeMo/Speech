# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn.functional as F
from torch import nn

from nemo.core import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType, SpectrogramType

__all__ = ["MLMLoss"]


class MLMLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for Contrastive."""
        return {
            "spec_masks": NeuralType(("B", "D", "T"), SpectrogramType(), optional=True),
            "decoder_outputs": NeuralType(("B", "T", "D"), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "masks": NeuralType(("B", "D", "T"), SpectrogramType(), optional=True),
        }

    @property
    def output_types(self):
        """Output types definitions for Contrastive.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @property
    def needs_labels(self):
        return True

    def __init__(
        self,
        combine_time_steps: int = 1,
        mask_threshold: float = 0.8,
    ):
        super().__init__()
        self.nll_loss = nn.NLLLoss()
        self.combine_time_steps = combine_time_steps
        self.mask_threshold = mask_threshold

    @typecheck()
    def forward(
        self, decoder_outputs, targets, decoder_lengths=None, target_lengths=None, spec_masks=None, masks=None
    ):
        """
        Args:
            decoder_outputs: (B, T, D)
            targets: (B, T)
            decoder_lengths: (B,)
            target_lengths: (B,)
            spec_masks: (B, D, T)
            masks: (B, D, T)

        Returns:
            loss: (1,)
        """

        if masks is None:
            masks = spec_masks

        if masks is None:
            masks = torch.ones_like(decoder_outputs, dtype=torch.bool)
        else:
            # B,D,T -> B,T,D
            masks = masks.transpose(1, 2)

        masks = masks.reshape(masks.shape[0], masks.shape[1] // self.combine_time_steps, -1)
        masks = masks.mean(-1) > self.mask_threshold  # (B, T)

        # Truncate the length of decoder_outputs, masks, and targets to the minimum length of the three.
        loss_length = min(decoder_outputs.shape[1], masks.shape[1], targets.shape[1])

        decoder_outputs = decoder_outputs[:, :loss_length]
        masks = masks[:, :loss_length]
        targets = targets[:, :loss_length]

        if decoder_lengths is not None:
            decoder_lengths = torch.clamp(decoder_lengths, max=loss_length)
        if target_lengths is not None:
            target_lengths = torch.clamp(target_lengths, max=loss_length)

        out_masked_only = decoder_outputs[masks]
        targets_masked_only = targets[masks]

        loss = self.nll_loss(out_masked_only, targets_masked_only)
        loss = torch.mean(loss)

        return loss


class MultiMLMLoss(Loss):
    """
    Masked language model loss for multiple decoders, where cross-entropy loss is applied separately on each decoder.
    This loss can be used with `nemo.collections.asr.modules.ssl_modules.MultiSoftmaxDecoder` to train a model with multiple targets per frame.
    Reference: https://arxiv.org/abs/2202.01855
    """

    @property
    def input_types(self):
        if self.squeeze_single and self.num_decoders == 1:
            decoder_outputs = NeuralType(("B", "T", "C"), LogprobsType())
            targets = NeuralType(('B', 'T'), LabelsType())
        else:
            decoder_outputs = NeuralType(("B", "T", "C", "H"), LogprobsType())
            targets = NeuralType(("B", "T", "H"), LabelsType())
        return {
            "masks": NeuralType(("B", "D", "T"), SpectrogramType()),
            "decoder_outputs": decoder_outputs,
            "targets": targets,
            "decoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        combine_time_steps: int = 1,
        mask_threshold: float = 0.8,
        num_decoders: int = 1,
        squeeze_single: bool = False,
    ):
        super().__init__()
        self.num_decoders = num_decoders
        self.squeeze_single = squeeze_single
        self.mlm_loss = MLMLoss(combine_time_steps, mask_threshold)

    @typecheck()
    def forward(self, masks, decoder_outputs, targets, decoder_lengths=None, target_lengths=None):
        if self.squeeze_single and self.num_decoders == 1:
            return self.mlm_loss(
                spec_masks=masks,
                decoder_outputs=decoder_outputs,
                targets=targets,
                decoder_lengths=decoder_lengths,
                target_lengths=target_lengths,
            )
        loss = 0.0
        for i in range(self.num_decoders):
            loss += self.mlm_loss(
                spec_masks=masks,
                decoder_outputs=decoder_outputs[:, :, :, i],
                targets=targets[:, :, i],
                decoder_lengths=decoder_lengths,
                target_lengths=target_lengths,
            )
        return loss / self.num_decoders
