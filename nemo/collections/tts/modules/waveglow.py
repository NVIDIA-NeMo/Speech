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
from typing import Tuple

import torch

from nemo.collections.tts.modules.submodules import Invertible1x1Conv, WaveNet
from nemo.collections.tts.parts.utils.helpers import OperationMode, remove, split_view
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    IntType,
    MelSpectrogramType,
    NormalDistributionSamplesType,
    VoidType,
)
from nemo.core.neural_types.neural_type import NeuralType


class WaveGlowModule(NeuralModule, Exportable):
    def __init__(
        self,
        n_mel_channels: int,
        n_flows: int,
        n_group: int,
        n_early_every: int,
        n_early_size: int,
        n_wn_channels: int,
        n_wn_layers: int,
        wn_kernel_size: int,
    ):
        """
        WaveGlow module

        Args:
            n_mel_channels (int): Number of mel channels to output.
            n_flows (int): Number of flow layers
            n_group (int): Number of groups to respace the inputs
            n_early_every (int): Every n_early_every layers, n_early_size gets skip connected to the output
            n_early_size (int): The size of the chunk to be skip connected
            n_wn_channels (int): Number of channels for the non-invertible wavenet transformation
            n_wn_layers (int): Number of layers for the non-invertible wavenet transformation
            wn_kernel_size (int): Kernel size for the non-invertible wavenet transformation
        """
        super().__init__()

        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels, n_mel_channels, 1024, stride=256)
        self.n_mel_channels = n_mel_channels
        assert n_group % 2 == 0
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.wavenet = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        self.mode = OperationMode.infer

        n_half = n_group // 2

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.wavenet.append(
                WaveNet(
                    n_half,
                    n_mel_channels * n_group,
                    n_layers=n_wn_layers,
                    n_channels=n_wn_channels,
                    kernel_size=wn_kernel_size,
                )
            )
        self.n_remaining_channels = n_remaining_channels
        self.time_cutoff = self.upsample.stride[0] - self.upsample.kernel_size[0]

        # Pre-calculating the sizes of noise to use so it's not dynamic
        n_halves = []
        n_half = self.n_remaining_channels // 2
        for k in reversed(range(self.n_flows)):
            n_halves.append(n_half)
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half + int(self.n_early_size / 2)
        n_halves.reverse()
        self.n_halves = n_halves

        self.removed_weightnorm = False

    def _prepare_for_export(self, **kwargs):
        """
        Override this method to prepare module for export. This is in-place operation.
        Base version does common necessary module replacements (Apex etc)
        """
        self.remove_weightnorm()
        super()._prepare_for_export(**kwargs)

    @typecheck()
    def forward(self, spec, z=None, audio=None, run_inverse=True, sigma=1.0):
        """ TODO
        """
        if self.training and self.mode != OperationMode.training:
            raise ValueError(f"{self} has self.training set to True but self.OperationMode was not set to training")
        if not self.training and self.mode == OperationMode.training:
            raise ValueError(f"{self} has self.training set to False but self.OperationMode was set to training")

        audio_pred = torch.zeros((1, 1))
        if run_inverse:
            # norm_dist_to_audio is used to predict audio from spectrogram so only used in val or infer mode
            # Could also log train audio but currently not done
            audio_pred = self.norm_dist_to_audio(spec=spec, sigma=sigma, z=z)

        if audio is not None and self.mode != OperationMode.infer:
            # audio_to_normal_dist is used to calculate loss so only run this in train or val model
            z1, log_s_list, log_det_W_list = self.audio_to_normal_dist(spec=spec, audio=audio)
            return z1, log_s_list, log_det_W_list, audio_pred

        return audio_pred

    @property
    def input_types(self):
        if self.mode == OperationMode.infer:
            return {
                "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
                "z": NeuralType(('B', 'D', 'T'), MelSpectrogramType(), optional=True),
                "sigma": NeuralType(optional=True),
            }
        else:
            return {
                "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
                "z": NeuralType(('B', 'D', 'T'), MelSpectrogramType(), optional=True),
                "audio": NeuralType(('B', 'T'), AudioSignal(), optional=True),
                "run_inverse": NeuralType(elements_type=IntType(), optional=True),
                "sigma": NeuralType(optional=True),
            }

    @property
    def output_types(self):
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            return {
                "pred_normal_dist": NeuralType(('B', 'flowgroup', 'T'), NormalDistributionSamplesType()),
                "log_s_list": [NeuralType(('B', 'flowgroup', 'T'), VoidType())],  # TODO: Figure out a good typing
                "log_det_W_list": [NeuralType(elements_type=VoidType())],  # TODO: Figure out a good typing
                "audio_pred": NeuralType(('B', 'T'), AudioSignal()),
            }
        else:
            return {
                "audio": NeuralType(('B', 'T'), AudioSignal()),
            }

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        par = next(self.parameters())
        mel = torch.randn((max_batch, self.n_mel_channels, max_dim), device=par.device, dtype=par.dtype)
        z = torch.randn(
            (max_batch, self.n_mel_channels, max_dim * self.upsample.stride[0] // self.n_group),
            device=par.device,
            dtype=par.dtype,
        )
        return {"spec": mel, "z": z}

    def audio_to_normal_dist(self, *, spec: torch.Tensor, audio: torch.Tensor) -> Tuple[torch.Tensor, list, list]:
        #  Upsample spectrogram to size of audio
        spec = self.upsample(spec)
        assert spec.size(2) >= audio.size(1)
        if spec.size(2) > audio.size(1):
            spec = spec[:, :, : audio.size(1)]

        # logging.debug(f"spec: {spec.shape}. n_group: {self.n_group}")
        spec = split_view(spec, self.n_group, 2).permute(0, 2, 1, 3)
        spec = spec.contiguous().view(spec.size(0), spec.size(1), -1)
        spec = spec.permute(0, 2, 1)

        audio = split_view(audio, self.n_group, 1).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, : self.n_early_size, :])
                audio = audio[:, self.n_early_size :, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.wavenet[k]((audio_0, spec))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def norm_dist_to_audio(self, *, spec, z=None, sigma: float = 1.0):
        spec = self.upsample(spec)
        spec = spec.contiguous().view(spec.size(0), spec.size(1), -1)
        # trim conv artifacts. maybe pad spec to kernel multiple
        if self.time_cutoff != 0:
            spec = spec[:, :, : self.time_cutoff]

        spec = split_view(spec, self.n_group, 2).permute(0, 2, 1, 3)
        spec = spec.contiguous().view(spec.size(0), spec.size(1), -1)
        spec = spec.permute(0, 2, 1)

        z_size = torch.Size([spec.size(0), self.n_group, spec.size(2)])
        if z is None:
            z = sigma * torch.randn(z_size, device=spec.device).to(spec.dtype)

        audio, z = torch.split(z, [self.n_remaining_channels, z.size(1) - self.n_remaining_channels], 1)

        for k in reversed(range(self.n_flows)):
            n_half = self.n_halves[k]
            audio_0, audio_1 = torch.split(audio, [n_half, audio.size(1) - n_half], 1)

            output = self.wavenet[k]((audio_0, spec))

            b, s = torch.split(output, [n_half, output.size(1) - n_half], 1)

            audio_1 = audio_1 - b
            audio_1 = audio_1 / torch.exp(s)
            audio = torch.cat((audio_0, audio_1), 1)

            audio = self.convinv[k](audio, reverse=True)
            if k % self.n_early_every == 0 and k > 0:
                z1, z = torch.split(z, [self.n_early_size, z.size(1) - self.n_early_size], 1)
                audio = torch.cat((z1, audio), 1)
        return audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)

    def remove_weightnorm(self):
        if self.removed_weightnorm:
            return
        for wavenet in self.wavenet:
            wavenet.start = torch.nn.utils.remove_weight_norm(wavenet.start)
            wavenet.in_layers = remove(wavenet.in_layers)
            wavenet.cond_layer = torch.nn.utils.remove_weight_norm(wavenet.cond_layer)
            wavenet.res_skip_layers = remove(wavenet.res_skip_layers)
        self.removed_weightnorm = True
