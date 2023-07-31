# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange

from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import (
    AudioSignal,
    LengthsType,
    LossType,
    NeuralType,
    PredictionsType,
    RegressionValuesType,
    VoidType,
)


class MaskedLoss(Loss):
    def __init__(self, loss_fn, loss_scale: float = 1.0):
        super(MaskedLoss, self).__init__()
        self.loss_scale = loss_scale
        self.loss_fn = loss_fn

    @property
    def input_types(self):
        return {
            "target": NeuralType(('B', 'D', 'T'), RegressionValuesType()),
            "predicted": NeuralType(('B', 'D', 'T'), PredictionsType()),
            "target_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, predicted, target, target_len):
        assert target.shape[2] == predicted.shape[2]

        # [B, D, T]
        loss = self.loss_fn(input=predicted, target=target)
        # [B, T]
        loss = torch.mean(loss, dim=1)
        # [B]
        loss = torch.sum(loss, dim=1) / torch.clamp(target_len, min=1.0)

        # [1]
        loss = torch.mean(loss)
        loss = self.loss_scale * loss

        return loss


class MaskedMAELoss(MaskedLoss):
    def __init__(self, loss_scale: float = 1.0):
        loss_fn = torch.nn.L1Loss(reduction='none')
        super(MaskedMAELoss, self).__init__(loss_fn=loss_fn, loss_scale=loss_scale)


class MaskedMSELoss(MaskedLoss):
    def __init__(self, loss_scale: float = 1.0):
        loss_fn = torch.nn.MSELoss(reduction='none')
        super(MaskedMSELoss, self).__init__(loss_fn=loss_fn, loss_scale=loss_scale)


class TimeDomainLoss(Loss):
    def __init__(self):
        super(TimeDomainLoss, self).__init__()
        self.loss_fn = MaskedMAELoss()

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": [NeuralType(elements_type=LossType())],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        audio_real = rearrange(audio_real, "B T -> B 1 T")
        audio_gen = rearrange(audio_gen, "B T -> B 1 T")
        loss = self.loss_fn(target=audio_real, predicted=audio_gen, target_len=audio_len)
        return loss


class MultiResolutionMelLoss(Loss):
    def __init__(self, sample_rate: int, mel_dim: int, resolutions: List[List], l1_scale: float = 1.0):
        super(MultiResolutionMelLoss, self).__init__()

        self.l1_loss_fn = MaskedMAELoss(loss_scale=l1_scale)
        self.l2_loss_fn = MaskedMSELoss()

        self.mel_features = torch.nn.ModuleList()
        for n_fft, hop_len, win_len in resolutions:
            mel_feature = FilterbankFeatures(
                sample_rate=sample_rate,
                nfilt=mel_dim,
                n_window_size=win_len,
                n_window_stride=hop_len,
                n_fft=n_fft,
                pad_to=1,
                mag_power=1.0,
                log_zero_guard_type="add",
                log_zero_guard_value=1.0,
                mel_norm=None,
                normalize=None,
                preemph=None,
                dither=0.0,
                use_grads=True,
            )
            self.mel_features.append(mel_feature)

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": [NeuralType(elements_type=LossType())],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        loss = 0.0
        for mel_feature in self.mel_features:
            mel_real, mel_real_len = mel_feature(x=audio_real, seq_len=audio_len)
            mel_gen, _ = mel_feature(x=audio_gen, seq_len=audio_len)
            loss += self.l1_loss_fn(predicted=mel_gen, target=mel_real, target_len=mel_real_len)
            loss += self.l2_loss_fn(predicted=mel_gen, target=mel_real, target_len=mel_real_len)

        loss /= len(self.mel_features)

        return loss


class RelativeFeatureMatchingLoss(Loss):
    def __init__(self, div_guard=1e-3):
        super(RelativeFeatureMatchingLoss, self).__init__()
        self.div_guard = div_guard

    @property
    def input_types(self):
        return {
            "fmaps_real": [[NeuralType(elements_type=VoidType())]],
            "fmaps_gen": [[NeuralType(elements_type=VoidType())]],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, fmaps_real, fmaps_gen):
        loss = 0.0
        for fmap_real, fmap_gen in zip(fmaps_real, fmaps_gen):
            # [B, ..., time]
            for feat_real, feat_gen in zip(fmap_real, fmap_gen):
                # [B, ...]
                feat_mean = torch.mean(torch.abs(feat_real), dim=-1)
                diff = torch.mean(torch.abs(feat_real - feat_gen), dim=-1)
                feat_loss = diff / (feat_mean + self.div_guard)
                # [1]
                feat_loss = torch.mean(feat_loss) / len(fmap_real)
                loss += feat_loss

        loss /= len(fmaps_real)

        return loss


class GeneratorHingedLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_scores_gen": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, disc_scores_gen):
        loss = 0.0
        for disc_score_gen in disc_scores_gen:
            loss += torch.mean(F.relu(1 - disc_score_gen))

        loss /= len(disc_scores_gen)

        return loss


class GeneratorSquaredLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_scores_gen": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, disc_scores_gen):
        loss = 0.0
        for disc_score_gen in disc_scores_gen:
            loss += torch.mean((1 - disc_score_gen) ** 2)

        loss /= len(disc_scores_gen)

        return loss


class DiscriminatorHingedLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_scores_real": [NeuralType(('B', 'C', 'T'), VoidType())],
            "disc_scores_gen": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, disc_scores_real, disc_scores_gen):
        loss = 0.0
        for disc_score_real, disc_score_gen in zip(disc_scores_real, disc_scores_gen):
            loss_real = torch.mean(F.relu(1 - disc_score_real))
            loss_gen = torch.mean(F.relu(1 + disc_score_gen))
            loss += (loss_real + loss_gen) / 2

        loss /= len(disc_scores_real)

        return loss


class DiscriminatorSquaredLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_scores_real": [NeuralType(('B', 'C', 'T'), VoidType())],
            "disc_scores_gen": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, disc_scores_real, disc_scores_gen):
        loss = 0.0
        for disc_score_real, disc_score_gen in zip(disc_scores_real, disc_scores_gen):
            loss_real = torch.mean((1 - disc_score_real) ** 2)
            loss_gen = torch.mean(disc_score_gen ** 2)
            loss += (loss_real + loss_gen) / 2

        loss /= len(disc_scores_real)

        return loss
