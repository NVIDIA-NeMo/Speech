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

"""
Implementation of Maxine BNR2 denoising network

Maxine Background Noise Removal (BNR) 2.0 is an audio background noise removal
model from NVIDIA. This is the second generation of BNR from
Maxine Audio Effects SDK. BNR 2.0 removes unwanted noises from audio improving
speech intelligibility and also improving the speech recognition accuracy of
various ASR systems under noisy environments.

BNR 2.0 uses the SEASR architecture described in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10837982
"""

from typing import Dict, Optional

import einops
import lightning.pytorch as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from nemo.collections.audio.models.audio_to_audio import AudioToAudioModel
from nemo.collections.audio.parts.utils.maxine import apply_weight_norm_lstm, remove_weight_norm_lstm
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, NeuralType

SUPPORTED_SAMPLE_RATE = 16000
SUPPORTED_INPUT_ALIGN_MS = 10
SUPPORTED_INPUT_ALIGN_SAMPLES = SUPPORTED_INPUT_ALIGN_MS * (SUPPORTED_SAMPLE_RATE // 1000)


class _Seasr(plt.LightningModule):
    """Internal implementation of the model class"""

    def __init__(
        self, sample_rate, hidden_nodes=128, streaming=False, kernel_size=320, f1=1024, f2=512, stride=160, dropout=0.5
    ):

        if sample_rate != SUPPORTED_SAMPLE_RATE:
            raise AssertionError("Currently only 16k is supported")

        super().__init__()
        self.f1 = f1
        self.f2 = f2
        self.f3 = hidden_nodes * 2
        self.gru_nodes = self.f1
        padding = 0 if streaming else 'same'

        self.conv1d = nn.Conv1d(1, self.f1, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn0 = nn.BatchNorm1d(self.f1, eps=0.001)
        self.feature_gru0 = nn.GRU(self.f1, self.f3, num_layers=1, batch_first=True)

        self.conv1d_out1 = nn.Conv1d(self.f1, self.f2, kernel_size=3, padding=padding)
        self.bn1 = nn.BatchNorm1d(self.f2, eps=0.001)
        self.feature_gru1 = nn.GRU(self.f2, self.f3, num_layers=1, batch_first=True)

        self.conv1d_out2 = nn.Conv1d(self.f2, self.f2, kernel_size=3, padding=padding)
        self.bn2 = nn.BatchNorm1d(self.f2, eps=0.001)
        self.feature_gru2 = nn.GRU(self.f2, self.f3, num_layers=1, batch_first=True)

        self.conv1d_out3 = nn.Conv1d(self.f2, self.f2, kernel_size=3, padding=padding)
        self.bn3 = nn.BatchNorm1d(self.f2, eps=0.001)

        self.denoise_gru = nn.GRU(3 * self.f3 + self.f2, self.gru_nodes, batch_first=True, dropout=dropout)
        self.denoise_gru_1 = nn.GRU(self.gru_nodes, self.gru_nodes, num_layers=1, batch_first=True, dropout=dropout)
        self.denoise_gru_2 = nn.GRU(self.gru_nodes, self.gru_nodes, num_layers=1, batch_first=True, dropout=dropout)
        self.denoise_gru_3 = nn.GRU(self.gru_nodes, self.gru_nodes, batch_first=True)

        self.denoise_mask = nn.Linear(self.gru_nodes, self.f1)
        self.mask_act = nn.Sigmoid()

        self.inv_conv = nn.ConvTranspose1d(self.f1, 1, kernel_size=kernel_size, stride=stride)
        self.inv_conv_activation = nn.Tanh()

    def forward(self, **kwargs):
        x0 = kwargs.get('x0')
        x0 = F.relu(self.conv1d(x0))
        xc0 = self.bn0(x0)

        xc1 = F.leaky_relu(self.conv1d_out1(xc0))
        xc1 = self.bn1(xc1)
        fg1, _ = self.feature_gru1(xc1.permute(0, 2, 1))

        xc2 = F.leaky_relu(self.conv1d_out2(xc1))
        xc2 = self.bn2(xc2)
        fg2, _ = self.feature_gru2(xc2.permute(0, 2, 1))

        xc3 = F.leaky_relu(self.conv1d_out3(xc2))
        xc3 = self.bn3(xc3)

        xc3 = xc3.permute(0, 2, 1)

        xc0 = xc0.permute(0, 2, 1)
        fg0, _ = self.feature_gru0(xc0)

        xi = torch.cat((fg0, fg1, fg2, xc3), 2)
        xi, _ = self.denoise_gru(xi)
        xi = xi + xc0
        xi1, _ = self.denoise_gru_1(xi)
        xi1 = xi1 + xi
        xi2, _ = self.denoise_gru_2(xi1)
        xi = xi1 + xi2

        xi, _ = self.denoise_gru_3(xi)

        mask = self.mask_act(self.denoise_mask(xi))
        mask = mask.permute(0, 2, 1)
        xi = x0 * mask

        xi = self.inv_conv(xi)
        xi = self.inv_conv_activation(xi)

        return xi

    def apply_weight_norm(self):
        """Apply weight normalization module from all layers."""

        def _apply_weight_norm(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.utils.weight_norm(m)
            elif isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
                apply_weight_norm_lstm(m)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all layers."""

        def _remove_weight_norm(m):
            try:
                if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                    torch.nn.utils.remove_weight_norm(m)
                elif isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
                    remove_weight_norm_lstm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class BNR2(AudioToAudioModel):
    """Implementation of the BNR 2 model"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)
        self.sample_rate = self._cfg.sample_rate

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        self.seasr = _Seasr(self.sample_rate)
        if (
            hasattr(self._cfg, "train")
            and hasattr(self._cfg.train, "enable_weight_norm")
            and self._cfg.train.enable_weight_norm
        ):
            self.seasr.apply_weight_norm()

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        return {
            "input_signal": NeuralType(
                ('B', 'C', 'T'), AudioSignal(freq=self.sample_rate)
            )  # multi-channel format, only channel dimension of 1 supported currently
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return {
            "output_signal": NeuralType(
                ('B', 'C', 'T'), AudioSignal(freq=self.sample_rate)
            )  # multi-channel format, channel dimension can be 1 for single-channel audio
        }

    @typecheck()
    def forward(self, input_signal):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T] or [B, T, C]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.

        Returns:
            Output signal `output` in the time domain and the length of the output signal `output_length`.
        """
        if input_signal.ndim == 3:
            if input_signal.shape[1] != 1:
                raise ValueError("This network currently only supports single channel audio signals.")
        elif input_signal.ndim != 2:
            raise ValueError(
                "Invalid shape for input signal (received {}, supported [B, 1, T] or [B, T])".format(
                    input_signal.shape
                )
            )

        if input_signal.shape[-1] % SUPPORTED_INPUT_ALIGN_SAMPLES != 0:
            raise ValueError("Input samples must be a multiple of {}".format(SUPPORTED_INPUT_ALIGN_SAMPLES))

        return self.seasr.forward(x0=input_signal)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch['target_signal']
        else:
            input_signal, input_length, target_signal, _ = batch

        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        predicted_audio = self.forward(input_signal=input_signal)

        loss = self.loss(target=target_signal, estimate=predicted_audio, input_length=input_length)

        self.log('train_loss', loss)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return loss

    def evaluation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):
        if isinstance(batch, dict):
            input_signal = batch['input_signal']
            input_length = batch['input_length']
            target_signal = batch['target_signal']
        else:
            input_signal, input_length, target_signal, _ = batch

        if input_signal.ndim == 2:
            input_signal = einops.rearrange(input_signal, 'B T -> B 1 T')
        if target_signal.ndim == 2:
            target_signal = einops.rearrange(target_signal, 'B T -> B 1 T')

        # Process input
        processed_signal = self(input_signal=input_signal)

        # Calculate the loss
        loss = self.loss(target=target_signal, estimate=processed_signal, input_length=input_length)

        # Update metrics
        if hasattr(self, 'metrics') and tag in self.metrics:
            # Update metrics for this (tag, dataloader_idx)
            for name, metric in self.metrics[tag][dataloader_idx].items():
                metric.update(preds=processed_signal, target=target_signal, input_length=input_length)

        # Log global step
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Return loss
        return {f'{tag}_loss': loss}

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """

        return None
