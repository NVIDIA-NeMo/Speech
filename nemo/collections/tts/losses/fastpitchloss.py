# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
#
# BSD 3-Clause License
#
# Copyright (c) 2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F

from nemo.collections.tts.modules.transformer import mask_from_lens
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import (
    LengthsType,
    LossType,
    MelSpectrogramType,
    RegressionValuesType,
    TokenDurationType,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType


class DurationLoss(Loss):
    def __init__(self, loss_scale=0.1):
        super().__init__()
        self.loss_scale = loss_scale

    @property
    def input_types(self):
        return {
            "log_durs_predicted": NeuralType(('B', 'T'), TokenLogDurationType()),
            "durs_tgt": NeuralType(('B', 'T'), TokenDurationType()),
            "len": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, log_durs_predicted, durs_tgt, len):
        dur_mask = mask_from_lens(len, max_len=durs_tgt.size(1))
        log_durs_tgt = torch.log(durs_tgt.float() + 1)
        loss_fn = F.mse_loss
        dur_loss = loss_fn(log_durs_predicted, log_durs_tgt, reduction='none')
        dur_loss = (dur_loss * dur_mask).sum() / dur_mask.sum()
        dur_loss *= self.loss_scale

        return dur_loss


class PitchLoss(Loss):
    def __init__(self, loss_scale=0.1):
        super().__init__()
        self.loss_scale = loss_scale

    @property
    def input_types(self):
        return {
            "pitch_predicted": NeuralType(('B', 'T'), RegressionValuesType()),
            "pitch_tgt": NeuralType(('B', 'T'), RegressionValuesType()),
            "len": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, pitch_predicted, pitch_tgt, len):
        dur_mask = mask_from_lens(len, max_len=pitch_tgt.size(1))
        ldiff = pitch_tgt.size(1) - pitch_predicted.size(1)
        pitch_predicted = F.pad(pitch_predicted, (0, ldiff, 0, 0), value=0.0)
        pitch_loss = F.mse_loss(pitch_tgt, pitch_predicted, reduction='none')
        pitch_loss = (pitch_loss * dur_mask).sum() / dur_mask.sum()
        pitch_loss *= self.loss_scale

        return pitch_loss


class EnergyLoss(Loss):
    def __init__(self, loss_scale=0.1):
        super().__init__()
        self.loss_scale = loss_scale

    @property
    def input_types(self):
        return {
            "energy_predicted": NeuralType(('B', 'T'), RegressionValuesType()),
            "energy_tgt": NeuralType(('B', 'T'), RegressionValuesType()),
            "length": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, energy_predicted, energy_tgt, length):
        if energy_tgt is None:
            return 0.0
        dur_mask = mask_from_lens(length, max_len=energy_tgt.size(1))
        energy_loss = F.mse_loss(energy_tgt, energy_predicted, reduction='none')
        energy_loss = (energy_loss * dur_mask).sum() / dur_mask.sum()
        energy_loss *= self.loss_scale

        return energy_loss


class MelLoss(Loss):
    @property
    def input_types(self):
        return {
            "spect_predicted": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spect_tgt": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, spect_predicted, spect_tgt):
        spect_tgt.requires_grad = False
        spect_tgt = spect_tgt.transpose(1, 2)  # (B, T, H)
        spect_predicted = spect_predicted.transpose(1, 2)  # (B, T, H)

        ldiff = spect_tgt.size(1) - spect_predicted.size(1)
        spect_predicted = F.pad(spect_predicted, (0, 0, 0, ldiff, 0, 0), value=0.0)
        mel_mask = spect_tgt.ne(0).float()
        loss_fn = F.mse_loss
        mel_loss = loss_fn(spect_predicted, spect_tgt, reduction='none')
        mel_loss = (mel_loss * mel_mask).sum() / mel_mask.sum()

        return mel_loss
