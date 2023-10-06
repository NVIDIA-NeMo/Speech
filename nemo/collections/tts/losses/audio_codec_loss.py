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
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths, mask_sequence_tensor
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
            "predicted": NeuralType(('B', 'D', 'T'), PredictionsType()),
            "target": NeuralType(('B', 'D', 'T'), RegressionValuesType()),
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
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        audio_real = rearrange(audio_real, "B T -> B 1 T")
        audio_gen = rearrange(audio_gen, "B T -> B 1 T")
        loss = self.loss_fn(target=audio_real, predicted=audio_gen, target_len=audio_len)
        return loss


class MultiResolutionMelLoss(Loss):
    """
    Multi-resolution log mel spectrogram loss.

    Args:
        sample_rate: Sample rate of audio.
        resolutions: List of resolutions, each being 3 integers ordered [num_fft, hop_length, window_length]
        mel_dims: Dimension of mel spectrogram to compute for each resolution. Should be same length as 'resolutions'.
        log_guard: Value to add to mel spectrogram to avoid taking log of 0.
    """

    def __init__(self, sample_rate: int, resolutions: List[List], mel_dims: List[int], log_guard: float = 1.0):
        super(MultiResolutionMelLoss, self).__init__()
        assert len(resolutions) == len(mel_dims)

        self.l1_loss_fn = MaskedMAELoss()
        self.l2_loss_fn = MaskedMSELoss()

        self.mel_features = torch.nn.ModuleList()
        for mel_dim, (n_fft, hop_len, win_len) in zip(mel_dims, resolutions):
            mel_feature = FilterbankFeatures(
                sample_rate=sample_rate,
                nfilt=mel_dim,
                n_window_size=win_len,
                n_window_stride=hop_len,
                n_fft=n_fft,
                pad_to=1,
                mag_power=1.0,
                log_zero_guard_type="add",
                log_zero_guard_value=log_guard,
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
            "l1_loss": NeuralType(elements_type=LossType()),
            "l2_loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        l1_loss = 0.0
        l2_loss = 0.0
        for mel_feature in self.mel_features:
            mel_real, mel_real_len = mel_feature(x=audio_real, seq_len=audio_len)
            mel_gen, _ = mel_feature(x=audio_gen, seq_len=audio_len)
            l1_loss += self.l1_loss_fn(predicted=mel_gen, target=mel_real, target_len=mel_real_len)
            l2_loss += self.l2_loss_fn(predicted=mel_gen, target=mel_real, target_len=mel_real_len)

        l1_loss /= len(self.mel_features)
        l2_loss /= len(self.mel_features)

        return l1_loss, l2_loss


class STFTLoss(Loss):
    """
    Log magnitude STFT loss.

    Args:
        resolution: Resolution of spectrogram, a list of 3 numbers ordered [num_fft, hop_length, window_length]
        log_guard: Value to add to magnitude spectrogram to avoid taking log of 0.
        sqrt_guard: Value to add to when computing absolute value of STFT to avoid NaN loss.
    """

    def __init__(self, resolution: List[int], log_guard: float = 1.0, sqrt_guard: float = 1e-5):
        super(STFTLoss, self).__init__()
        self.loss_fn = MaskedMAELoss()
        self.n_fft, self.hop_length, self.win_length = resolution
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.log_guard = log_guard
        self.sqrt_guard = sqrt_guard

    def _compute_spectrogram(self, audio, spec_len):
        # [B, n_fft, T_spec]
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        # [B, n_fft, T_spec, 2]
        spec = torch.view_as_real(spec)
        # [B, n_fft, T_spec]
        spec_mag = torch.sqrt(spec.pow(2).sum(-1) + self.sqrt_guard)
        spec_log = torch.log(spec_mag + self.log_guard)
        spec_log = mask_sequence_tensor(spec_log, spec_len)
        return spec_log

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        spec_len = (audio_len // self.hop_length) + 1
        spec_real = self._compute_spectrogram(audio=audio_real, spec_len=spec_len)
        spec_gen = self._compute_spectrogram(audio=audio_gen, spec_len=spec_len)
        loss = self.loss_fn(predicted=spec_gen, target=spec_real, target_len=spec_len)
        return loss


class MultiResolutionSTFTLoss(Loss):
    """
    Multi-resolution log magnitude STFT loss.

    Args:
        resolutions: List of resolutions, each being 3 integers ordered [num_fft, hop_length, window_length]
        log_guard: Value to add to magnitude spectrogram to avoid taking log of 0.
        sqrt_guard: Value to add to when computing absolute value of STFT to avoid NaN loss.
    """

    def __init__(self, resolutions: List[List], log_guard: float = 1.0, sqrt_guard: float = 1e-5):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.loss_fns = torch.nn.ModuleList(
            [STFTLoss(resolution=resolution, log_guard=log_guard, sqrt_guard=sqrt_guard) for resolution in resolutions]
        )

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        loss = 0.0
        for loss_fn in self.loss_fns:
            loss += loss_fn(audio_real=audio_real, audio_gen=audio_gen, audio_len=audio_len)
        loss /= len(self.loss_fns)
        return loss


class SISDRLoss(Loss):
    """
    SI-SDR loss based off of torchmetrics.functional.audio.sdr.scale_invariant_signal_distortion_ratio
    with added support for masking.
    """

    def __init__(self, epsilon: float = 1e-8):
        super(SISDRLoss, self).__init__()
        self.epsilon = epsilon

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        mask = get_mask_from_lengths(x=audio_real, lengths=audio_len)
        audio_len = rearrange(audio_len, 'B -> B 1')

        # Shift audio to have zero-mean
        # [B, 1]
        target_mean = torch.sum(audio_real, dim=-1, keepdim=True) / audio_len
        pred_mean = torch.sum(audio_gen, dim=-1, keepdim=True) / audio_len

        # [B, T]
        target = audio_real - target_mean
        target = target * mask
        pred = audio_gen - pred_mean
        pred = pred * mask

        # [B, 1]
        ref_pred = torch.sum(pred * target, dim=-1, keepdim=True)
        ref_target = torch.sum(target ** 2, dim=-1, keepdim=True)
        alpha = (ref_pred + self.epsilon) / (ref_target + self.epsilon)

        # [B, T]
        target_scaled = alpha * target
        distortion = target_scaled - pred

        # [B]
        target_scaled_power = torch.sum(target_scaled ** 2, dim=-1)
        distortion_power = torch.sum(distortion ** 2, dim=-1)

        ratio = (target_scaled_power + self.epsilon) / (distortion_power + self.epsilon)
        si_sdr = 10 * torch.log10(ratio)

        # [1]
        loss = -torch.mean(si_sdr)
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
