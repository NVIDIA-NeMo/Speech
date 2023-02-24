# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LengthsType, LogitsType, LossType, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType


class Tacotron2Loss(Loss):
    """A Loss module that computes loss for Tacotron2"""

    @property
    def input_types(self):
        return {
            "spec_pred_dec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spec_pred_postnet": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "gate_pred": NeuralType(('B', 'T'), LogitsType()),
            "spec_target": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spec_target_len": NeuralType(('B'), LengthsType()),
            "pad_value": NeuralType(),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
            "gate_target": NeuralType(('B', 'T'), LogitsType()),  # Used for evaluation
        }

    @typecheck()
    def forward(self, *, spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, pad_value):
        # Make the gate target
        max_len = spec_target.shape[2]
        gate_target = torch.zeros(spec_target_len.shape[0], max_len)
        gate_target = gate_target.type_as(gate_pred)
        for i, length in enumerate(spec_target_len):
            gate_target[i, length.data - 1 :] = 1

        spec_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        max_len = spec_target.shape[2]

        if max_len < spec_pred_dec.shape[2]:
            # Predicted len is larger than reference
            # Need to slice
            spec_pred_dec = spec_pred_dec.narrow(2, 0, max_len)
            spec_pred_postnet = spec_pred_postnet.narrow(2, 0, max_len)
            gate_pred = gate_pred.narrow(1, 0, max_len).contiguous()
        elif max_len > spec_pred_dec.shape[2]:
            # Need to do padding
            pad_amount = max_len - spec_pred_dec.shape[2]
            spec_pred_dec = torch.nn.functional.pad(spec_pred_dec, (0, pad_amount), value=pad_value)
            spec_pred_postnet = torch.nn.functional.pad(spec_pred_postnet, (0, pad_amount), value=pad_value)
            gate_pred = torch.nn.functional.pad(gate_pred, (0, pad_amount), value=1e3)

        mask = ~get_mask_from_lengths(spec_target_len, spec_pred_dec)
        mask = mask.expand(spec_target.shape[1], mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        spec_pred_dec.data.masked_fill_(mask, pad_value)
        spec_pred_postnet.data.masked_fill_(mask, pad_value)
        gate_pred.data.masked_fill_(mask[:, 0, :], 1e3)

        gate_pred = gate_pred.view(-1, 1)
        rnn_mel_loss = torch.nn.functional.mse_loss(spec_pred_dec, spec_target)
        postnet_mel_loss = torch.nn.functional.mse_loss(spec_pred_postnet, spec_target)
        gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(gate_pred, gate_target)
        return rnn_mel_loss + postnet_mel_loss + gate_loss, gate_target
