# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from typing import List

import torch

from nemo.collections.tts.helpers.helpers import OperationMode
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import AudioSignal
from nemo.core.neural_types.neural_type import NeuralType


class SpectrogramGenerator(ModelPT, ABC):
    """ Base class for all TTS models that turn text into a spectrogram """

    @abstractmethod
    def parse(self, str_input: str, **kwargs) -> 'torch.tensor':
        """
        A helper function that accepts raw python strings and turns them into a tensor. The tensor should have 2
        dimensions. The first is the batch, which should be of size 1. The second should represent time. The tensor
        should represent either tokenized or embedded text, depending on the model.

        Note that some models have `normalize` parameter in this function which will apply normalizer if it is available.
        """

    @abstractmethod
    def generate_spectrogram(self, tokens: 'torch.tensor', **kwargs) -> 'torch.tensor':
        """
        Accepts a batch of text or text_tokens and returns a batch of spectrograms

        Args:
            tokens: A torch tensor representing the text to be generated

        Returns:
            spectrograms
        """

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        for subclass in cls.__subclasses__():
            subclass_models = subclass.list_available_models()
            if subclass_models is not None and len(subclass_models) > 0:
                list_of_models.extend(subclass_models)
        return list_of_models


class Vocoder(ModelPT, ABC):
    """
    A base class for models that convert spectrograms to audios. Note that this class takes as input either linear
    or mel spectrograms.
    """

    @abstractmethod
    def convert_spectrogram_to_audio(self, spec: 'torch.tensor', **kwargs) -> 'torch.tensor':
        """
        Accepts a batch of spectrograms and returns a batch of audio.

        Args:
            spec:  ['B', 'n_freqs', 'T'], A torch tensor representing the spectrograms to be vocoded.

        Returns:
            audio
        """

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        for subclass in cls.__subclasses__():
            subclass_models = subclass.list_available_models()
            if subclass_models is not None and len(subclass_models) > 0:
                list_of_models.extend(subclass_models)
        return list_of_models


class GlowVocoder(Vocoder):
    """ Base class for all Vocoders that use a Glow or reversible Flow-based setup. All child class are expected
    to have a parameter called audio_to_melspec_precessor that is an instance of
    nemo.collections.asr.parts.FilterbankFeatures"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = OperationMode.infer
        self.stft = None
        self.istft = None
        self.n_mel = None
        self.bias_spect = None

    @property
    def mode(self):
        return self._mode

    @contextmanager
    def temp_mode(self, mode):
        old_mode = self.mode
        self.mode = mode
        try:
            yield
        finally:
            self.mode = old_mode

    @contextmanager
    def nemo_infer(self):  # Prepend with nemo to avoid any .infer() clashes with lightning or pytorch
        with ExitStack() as stack:
            stack.enter_context(self.temp_mode(OperationMode.infer))
            stack.enter_context(torch.no_grad())
            yield

    def check_children_attributes(self):
        if self.stft is None:
            try:
                n_fft = self.audio_to_melspec_precessor.n_fft
                hop_length = self.audio_to_melspec_precessor.hop_length
                win_length = self.audio_to_melspec_precessor.win_length
                window = self.audio_to_melspec_precessor.window.to(self.device)
            except AttributeError as e:
                raise AttributeError(
                    f"{self} could not find a valid audio_to_melspec_precessor. GlowVocoder requires child class "
                    "to have audio_to_melspec_precessor defined to obtain stft parameters. "
                    "audio_to_melspec_precessor requires n_fft, hop_length, win_length, window, and nfilt to be "
                    "defined."
                ) from e

            def yet_another_patch(audio, n_fft, hop_length, win_length, window):
                spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
                if spec.dtype in [torch.cfloat, torch.cdouble]:
                    spec = torch.view_as_real(spec)
                return torch.sqrt(spec.pow(2).sum(-1)), torch.atan2(spec[..., -1], spec[..., 0])

            self.stft = lambda x: yet_another_patch(
                x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
            )
            self.istft = lambda x, y: torch.istft(
                torch.complex(x * torch.cos(y), x * torch.sin(y)),
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
            )

        if self.n_mel is None:
            try:
                self.n_mel = self.audio_to_melspec_precessor.nfilt
            except AttributeError as e:
                raise AttributeError(
                    f"{self} could not find a valid audio_to_melspec_precessor. GlowVocoder requires child class to "
                    "have audio_to_melspec_precessor defined to obtain stft parameters. audio_to_melspec_precessor "
                    "requires nfilt to be defined."
                ) from e

    def update_bias_spect(self):
        self.check_children_attributes()  # Ensure stft parameters are defined

        with self.nemo_infer():
            spect = torch.zeros((1, self.n_mel, 88)).to(self.device)
            bias_audio = self.convert_spectrogram_to_audio(spec=spect, sigma=0.0, denoise=False)
            bias_spect, _ = self.stft(bias_audio)
            self.bias_spect = bias_spect[..., 0][..., None]

    @typecheck(
        input_types={"audio": NeuralType(('B', 'T'), AudioSignal()), "strength": NeuralType(optional=True)},
        output_types={"audio": NeuralType(('B', 'T'), AudioSignal())},
    )
    def denoise(self, audio: 'torch.tensor', strength: float = 0.01):
        self.check_children_attributes()  # Ensure self.n_mel and self.stft are defined

        if self.bias_spect is None:
            self.update_bias_spect()
        audio_spect, audio_angles = self.stft(audio)
        audio_spect_denoised = audio_spect - self.bias_spect.to(audio.device) * strength
        audio_spect_denoised = torch.clamp(audio_spect_denoised, 0.0)
        audio_denoised = self.istft(audio_spect_denoised, audio_angles)
        return audio_denoised


class MelToSpec(ModelPT, ABC):
    """
    A base class for models that convert mel spectrograms to linear (magnitude) spectrograms
    """

    @abstractmethod
    def convert_mel_spectrogram_to_linear(self, mel: 'torch.tensor', **kwargs) -> 'torch.tensor':
        """
        Accepts a batch of spectrograms and returns a batch of linear spectrograms

        Args:
            mel: A torch tensor representing the mel spectrograms ['B', 'mel_freqs', 'T']

        Returns:
            spec: A torch tensor representing the linear spectrograms ['B', 'n_freqs', 'T']
        """

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        for subclass in cls.__subclasses__():
            subclass_models = subclass.list_available_models()
            if subclass_models is not None and len(subclass_models) > 0:
                list_of_models.extend(subclass_models)
        return list_of_models
