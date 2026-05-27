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

from nemo.collections.asr.losses import CTCLoss
from nemo.core import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LossType, NeuralType, SpectrogramType, VoidType

__all__ = ["CTCLossForSSL"]


class CTCLossForSSL(Loss):
    @property
    def input_types(self):
        """Input types definitions for Contrastive."""
        return {
            "spec_masks": NeuralType(("B", "D", "T"), SpectrogramType()),
            "decoder_outputs": NeuralType(("B", "T", "D"), VoidType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
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

    def __init__(self, num_classes, zero_infinity=True, reduction='mean_batch'):
        super().__init__()
        self.loss = CTCLoss(num_classes=num_classes, reduction=reduction, zero_infinity=zero_infinity)

    @typecheck()
    def forward(self, spec_masks, decoder_outputs, targets, decoder_lengths=None, target_lengths=None):
        loss = self.loss(
            log_probs=decoder_outputs, targets=targets, input_lengths=decoder_lengths, target_lengths=target_lengths
        )

        return loss
