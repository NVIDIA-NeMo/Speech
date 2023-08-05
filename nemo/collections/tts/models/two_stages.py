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

from dataclasses import dataclass
from typing import Any, Dict, List

import librosa
import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf

from nemo.collections.tts.models.base import MelToSpec, Vocoder
from nemo.collections.tts.parts.utils.helpers import OperationMode, griffin_lim
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types.elements import AudioSignal, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType


class MelPsuedoInverseModel(MelToSpec):
    def __init__(self, cfg: DictConfig):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        super().__init__(cfg=cfg)

        sampling_rate = self._cfg['sampling_rate']
        n_fft = self._cfg['n_fft']
        mel_fmin = self._cfg['mel_fmin']
        mel_fmax = self._cfg['mel_fmax']
        mel_freq = self._cfg['mel_freq']

        melinv = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, fmin=mel_fmin, fmax=mel_fmax, n_mels=mel_freq)
        self.mel_pseudo_inverse = torch.tensor(melinv, dtype=torch.float)

    def convert_mel_spectrogram_to_linear(self, mel):
        lin_spec = torch.tensordot(mel, self.mel_pseudo_inverse, dims=[[1], [0]])
        lin_spec = lin_spec.permute(0, 2, 1)
        return lin_spec

    def setup_training_data(self, cfg):
        pass

    def setup_validation_data(self, cfg):
        pass

    def cuda(self, *args, **kwargs):
        self.mel_pseudo_inverse = self.mel_pseudo_inverse.cuda(*args, **kwargs)
        return self


class GriffinLimModel(Vocoder):
    def __init__(self, cfg: DictConfig):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        super().__init__(cfg=cfg)

        self.n_iters = self._cfg['n_iters']
        self.n_fft = self._cfg['n_fft']
        self.l_hop = self._cfg['l_hop']

    def convert_spectrogram_to_audio(self, spec, Ts=None):
        batch_size = spec.shape[0]

        T_max = spec.shape[2]
        if Ts is None:
            Ts = [T_max] * batch_size

        max_size = (max(Ts) - 1) * self.l_hop
        audios = torch.zeros(batch_size, max_size)
        # Lazy GL implementation. Could be improved by moving to pytorch.
        for i in range(batch_size):
            audio = griffin_lim(spec[i, :, 0 : Ts[i]].cpu().numpy(), n_iters=self.n_iters, n_fft=self.n_fft)
            my_len = audio.shape[0]
            audios[i, 0:my_len] = torch.from_numpy(audio)

        return audios

    def setup_training_data(self, cfg):
        pass

    def setup_validation_data(self, cfg):
        pass

    def cuda(self, *args, **kwargs):
        return self


@dataclass
class TwoStagesConfig:
    mel2spec: Dict[Any, Any] = MISSING
    linvocoder: Dict[Any, Any] = MISSING


class TwoStagesModel(Vocoder):
    """Two Stages model used to convert mel spectrograms, to linear spectrograms, and then to audio"""

    def __init__(self, cfg: DictConfig):

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        super().__init__(cfg=cfg)

        schema = OmegaConf.structured(TwoStagesConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        if '_target_' in self._cfg.mel2spec:
            self.mel2spec = instantiate(self._cfg.mel2spec)
        else:
            self.mel2spec = None

        if '_target_' in self._cfg.linvocoder:
            self.linvocoder = instantiate(self._cfg.linvocoder)
        else:
            self.linvocoder = None

    def set_mel_to_spec_model(self, mel2spec: MelToSpec):
        self.mel2spec = mel2spec

    def set_linear_vocoder(self, linvocoder: Vocoder):
        self.linvocoder = linvocoder

    def cuda(self, *args, **kwargs):
        self.mel2spec.cuda(*args, **kwargs)
        self.linvocoder.cuda(*args, **kwargs)
        return super().cuda(*args, **kwargs)

    @property
    def input_types(self):
        return {
            "mel": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "wave": NeuralType(('B', 'T'), AudioSignal()),
        }

    def forward(self, *, mel):
        pass

    def convert_spectrogram_to_audio(self, spec: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        try:
            self.mel2spec.mode = OperationMode.infer
        except AttributeError:
            pass
        try:
            self.linvocoder.mode = OperationMode.infer
        except AttributeError:
            pass

        with torch.no_grad():
            exp_spec = torch.exp(spec)
            linear_spec = self.mel2spec.convert_mel_spectrogram_to_linear(exp_spec)
            audio = self.linvocoder.convert_spectrogram_to_audio(linear_spec, **kwargs)

        return audio

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self, outputs):
        pass

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        pass

    def setup_training_data(self, cfg):
        pass

    def setup_validation_data(self, cfg):
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        return list_of_models
