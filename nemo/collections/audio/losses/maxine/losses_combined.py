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

import hashlib
from pathlib import Path
from typing import Optional

import torch

from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.audio.parts.utils.transforms import MelSpectrogram, resample
from nemo.core import Loss, Typing, typecheck
from nemo.core.neural_types import LengthsType, LossType, NeuralType, VoidType
from nemo.utils import logging
from nemo.utils.cloud import maybe_download_from_cloud
from nemo.utils.data_utils import resolve_cache_dir

from .sisnr_loss import sisnr_loss

# ASR model used for loss
# Note: Currently only this model is supported
STT_EN_CONFORMER_CTC_SMALL_v1_6_0 = 'https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small/versions/1.6.0/files/stt_en_conformer_ctc_small.nemo'


def restore_asr_model_from_cloud(location: str, refresh_cache: bool = False) -> ASRModel:
    """Restore an ASR model from the cloud.

    Args:
        location (str): The URL of the model in the cloud.
        refresh_cache (bool): Whether to force re-download of the model.

    Returns:
        nemo_asr.models.ASRModel: The restored model.
    """
    logging.debug('Restoring model from cloud location: %s', location)

    # Split into filename and base URL
    filename = location.split('/')[-1]
    url = location.replace(filename, '')

    # Get local cache dir
    cache_dir = Path.joinpath(resolve_cache_dir(), f'{filename[:-5]}')

    # If location in the cloud changes, this will force re-download
    cache_subfolder = hashlib.md5(location.encode('utf-8')).hexdigest()

    nemo_model_file_in_cache = maybe_download_from_cloud(
        url=url, filename=filename, cache_dir=cache_dir, subfolder=cache_subfolder, refresh_cache=refresh_cache
    )

    logging.debug('Model file in cache: %s', nemo_model_file_in_cache)

    # Restore model from local cache
    model = ASRModel.restore_from(nemo_model_file_in_cache)

    return model


class CombinedLoss(Loss, Typing):
    """
    Combination of three losses (signal quality/spectral+cepstral features/acoustic error)
    See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1083798
    """

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        num_mels: int,
        fft_length: int,
        sisnr_loss_weight: float,
        spectral_loss_weight: float,
        asr_loss_weight: float,
        use_asr_loss: bool = True,
        use_mel_spec: bool = True,
        conformer_model=STT_EN_CONFORMER_CTC_SMALL_v1_6_0,
        epsilon=float(5.9604644775390625e-8),
    ):

        super().__init__()
        self.sample_rate = sample_rate

        window = torch.hann_window(fft_length)
        self.register_buffer("window", window)
        epsilon = torch.tensor(epsilon)
        self.register_buffer("epsilon", epsilon, persistent=False)
        self.source_lengths = None
        self.source_value = sample_rate * sample_rate
        if use_asr_loss:
            self.asr_model = restore_asr_model_from_cloud(conformer_model)
            self.asr_model.eval()
            self.asr_model.freeze()

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=fft_length,
            hop_length=hop_length,
            n_mels=num_mels,
            center=False,
        )
        self.mae_loss = torch.nn.L1Loss()
        self.use_mel_spec = use_mel_spec
        self.use_asr_loss = use_asr_loss

        self.sisnr_loss_weight = sisnr_loss_weight
        self.spectral_loss_weight = spectral_loss_weight
        self.asr_loss_weight = asr_loss_weight

    def spectral_loss(self, predicted_audio: torch.Tensor, primary_audio: torch.Tensor) -> torch.Tensor:
        loss = 0
        if self.use_mel_spec:
            primary_mel_spec = self.mel_transform(primary_audio)
            predicted_mel_spec = self.mel_transform(predicted_audio)
            melLoss = self.mae_loss(predicted_mel_spec, primary_mel_spec)
            loss += melLoss

            log_pred = 2 * torch.log10(torch.clamp(predicted_mel_spec, min=self.epsilon))
            log_prim = 2 * torch.log10(torch.clamp(primary_mel_spec, min=self.epsilon))
            logMelLoss = self.mae_loss(log_pred, log_prim)
            loss += logMelLoss
        return loss

    def asr_loss(self, predicted_audio: torch.Tensor, primary_audio: torch.Tensor) -> torch.Tensor:
        primary_audio_ = torch.squeeze(primary_audio, dim=1).to(next(self.parameters()).device)
        predicted_audio_ = torch.squeeze(predicted_audio, dim=1).to(next(self.parameters()).device)

        input_len = torch.full([primary_audio_.size(dim=0)], primary_audio_.size(dim=-1)).to(
            next(self.parameters()).device
        )

        asr_sample_rate = self.asr_model.cfg.sample_rate
        if self.sample_rate != asr_sample_rate:
            # Resample to 16kHz
            primary_audio_ = resample(primary_audio_, self.sample_rate, asr_sample_rate)
            predicted_audio_ = resample(predicted_audio_, self.sample_rate, asr_sample_rate)

        primary_log, _, _ = self.asr_model(input_signal=primary_audio_, input_signal_length=input_len)
        predicted_log, _, _ = self.asr_model(input_signal=predicted_audio_, input_signal_length=input_len)

        primary_prob = 10**primary_log
        predicted_prob = 10**predicted_log

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(predicted_prob, primary_prob)
        return loss

    @property
    def input_types(self):
        """Input types definitions for CombinedLoss."""
        signal_shape = ('B', 'C', 'T')
        return {
            "estimate": NeuralType(signal_shape, VoidType()),
            "target": NeuralType(signal_shape, VoidType()),
            "input_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        """Output types definitions for CombinedLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(
        self, estimate: torch.Tensor, target: torch.Tensor, input_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.source_lengths is None:
            batch = estimate.shape[0]
            self.source_lengths = torch.full((batch,), self.source_value).to(device)
        # Clip at min_len
        min_len = int(torch.min(torch.tensor([estimate.size(-1), target.size(-1)])))
        if input_length is None:
            input_length = torch.full((estimate.shape[0],), estimate.shape[1]).to(device)
        source_lengths_l = torch.where(input_length > min_len, min_len, input_length)
        primary_audio = estimate[..., :min_len]
        predicted_audio = target[..., :min_len]

        loss_total = torch.tensor([0.0]).to(device)

        # SiSNR loss
        loss_total += self.sisnr_loss_weight * sisnr_loss(primary_audio, predicted_audio, source_lengths_l)

        # Spectral Loss
        loss = self.spectral_loss(predicted_audio, primary_audio)
        loss_total += self.spectral_loss_weight * loss

        # ASR loss
        if self.use_asr_loss:
            loss_total += self.asr_loss_weight * self.asr_loss(predicted_audio, primary_audio)

        return loss_total
