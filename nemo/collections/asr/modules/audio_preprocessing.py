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

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
from packaging import version

from nemo.collections.asr.parts.numba.spec_augment import SpecAugmentNumba, spec_augment_launch_heuristics
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures, FilterbankFeaturesTA
from nemo.collections.asr.parts.submodules.spectr_augment import SpecAugment, SpecCutout
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types import (
    AudioSignal,
    LengthsType,
    MelSpectrogramType,
    MFCCSpectrogramType,
    NeuralType,
    SpectrogramType,
)
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__
from nemo.utils import logging

try:
    import torchaudio
    import torchaudio.functional
    import torchaudio.transforms

    TORCHAUDIO_VERSION = version.parse(torchaudio.__version__)
    TORCHAUDIO_VERSION_MIN = version.parse('0.5')

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False

__all__ = [
    'AudioToMelSpectrogramPreprocessor',
    'AudioToMFCCPreprocessor',
    'SpectrogramAugmentation',
    'MaskedPatchAugmentation',
    'CropOrPadSpectrogramAugmentation',
]


class AudioPreprocessor(NeuralModule, ABC):
    """
        An interface for Neural Modules that performs audio pre-processing,
        transforming the wav files to features.
    """

    def __init__(self, win_length, hop_length):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length

        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }

    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length):
        processed_signal, processed_length = self.get_features(input_signal, length)

        return processed_signal, processed_length

    @abstractmethod
    def get_features(self, input_signal, length):
        # Called by forward(). Subclasses should implement this.
        pass


class AudioToMelSpectrogramPreprocessor(AudioPreprocessor, Exportable):
    """Featurizer module that converts wavs to mel spectrograms.

        Args:
            sample_rate (int): Sample rate of the input audio data.
                Defaults to 16000
            window_size (float): Size of window for fft in seconds
                Defaults to 0.02
            window_stride (float): Stride of window for fft in seconds
                Defaults to 0.01
            n_window_size (int): Size of window for fft in samples
                Defaults to None. Use one of window_size or n_window_size.
            n_window_stride (int): Stride of window for fft in samples
                Defaults to None. Use one of window_stride or n_window_stride.
            window (str): Windowing function for fft. can be one of ['hann',
                'hamming', 'blackman', 'bartlett']
                Defaults to "hann"
            normalize (str): Can be one of ['per_feature', 'all_features']; all
                other options disable feature normalization. 'all_features'
                normalizes the entire spectrogram to be mean 0 with std 1.
                'pre_features' normalizes per channel / freq instead.
                Defaults to "per_feature"
            n_fft (int): Length of FT window. If None, it uses the smallest power
                of 2 that is larger than n_window_size.
                Defaults to None
            preemph (float): Amount of pre emphasis to add to audio. Can be
                disabled by passing None.
                Defaults to 0.97
            features (int): Number of mel spectrogram freq bins to output.
                Defaults to 64
            lowfreq (int): Lower bound on mel basis in Hz.
                Defaults to 0
            highfreq  (int): Lower bound on mel basis in Hz.
                Defaults to None
            log (bool): Log features.
                Defaults to True
            log_zero_guard_type(str): Need to avoid taking the log of zero. There
                are two options: "add" or "clamp".
                Defaults to "add".
            log_zero_guard_value(float, or str): Add or clamp requires the number
                to add with or clamp to. log_zero_guard_value can either be a float
                or "tiny" or "eps". torch.finfo is used if "tiny" or "eps" is
                passed.
                Defaults to 2**-24.
            dither (float): Amount of white-noise dithering.
                Defaults to 1e-5
            pad_to (int): Ensures that the output size of the time dimension is
                a multiple of pad_to.
                Defaults to 16
            frame_splicing (int): Defaults to 1
            exact_pad (bool): If True, sets stft center to False and adds padding, such that num_frames = audio_length
                // hop_length. Defaults to False.
            pad_value (float): The value that shorter mels are padded with.
                Defaults to 0
            mag_power (float): The power that the linear spectrogram is raised to
                prior to multiplication with mel basis.
                Defaults to 2 for a power spec
            rng : Random number generator
            nb_augmentation_prob (float) : Probability with which narrowband augmentation would be applied to
                samples in the batch.
                Defaults to 0.0
            nb_max_freq (int) : Frequency above which all frequencies will be masked for narrowband augmentation.
                Defaults to 4000
            use_torchaudio: Whether to use the `torchaudio` implementation.
            stft_exact_pad: Deprecated argument, kept for compatibility with older checkpoints.
            stft_conv: Deprecated argument, kept for compatibility with older checkpoints.
        """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(
                tuple('B'), LengthsType()
            ),  # Please note that length should be in samples not seconds.
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.

        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(MelSpectrogramSignalTag)
            2: AxisType(ProcessedTimeTag)
        processed_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        features=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=1e-5,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        use_torchaudio: bool = False,
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__(n_window_size, n_window_stride)

        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        # Given the long and similar argument list, point to the class and instantiate it by reference
        if not use_torchaudio:
            featurizer_class = FilterbankFeatures
        else:
            featurizer_class = FilterbankFeaturesTA
        self.featurizer = featurizer_class(
            sample_rate=self._sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            pad_value=pad_value,
            mag_power=mag_power,
            rng=rng,
            nb_augmentation_prob=nb_augmentation_prob,
            nb_max_freq=nb_max_freq,
            stft_exact_pad=stft_exact_pad,  # Deprecated arguments; kept for config compatibility
            stft_conv=stft_conv,  # Deprecated arguments; kept for config compatibility
        )

    def input_example(self, max_batch: int = 8, max_dim: int = 32000, min_length: int = 200):
        batch_size = torch.randint(low=1, high=max_batch, size=[1]).item()
        max_length = torch.randint(low=min_length, high=max_dim, size=[1]).item()
        signals = torch.rand(size=[batch_size, max_length]) * 2 - 1
        lengths = torch.randint(low=min_length, high=max_dim, size=[batch_size])
        lengths[0] = max_length
        return signals, lengths

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks


class AudioToMFCCPreprocessor(AudioPreprocessor):
    """Preprocessor that converts wavs to MFCCs.
    Uses torchaudio.transforms.MFCC.

    Args:
        sample_rate: The sample rate of the audio.
            Defaults to 16000.
        window_size: Size of window for fft in seconds. Used to calculate the
            win_length arg for mel spectrogram.
            Defaults to 0.02
        window_stride: Stride of window for fft in seconds. Used to caculate
            the hop_length arg for mel spect.
            Defaults to 0.01
        n_window_size: Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride: Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window: Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett', 'none', 'null'].
            Defaults to 'hann'
        n_fft: Length of FT window. If None, it uses the smallest power of 2
            that is larger than n_window_size.
            Defaults to None
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        n_mels: Number of mel filterbanks.
            Defaults to 64
        n_mfcc: Number of coefficients to retain
            Defaults to 64
        dct_type: Type of discrete cosine transform to use
        norm: Type of norm to use
        log: Whether to use log-mel spectrograms instead of db-scaled.
            Defaults to True.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), MFCCSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window='hann',
        n_fft=None,
        lowfreq=0.0,
        highfreq=None,
        n_mels=64,
        n_mfcc=64,
        dct_type=2,
        norm='ortho',
        log=True,
    ):
        self._sample_rate = sample_rate
        if not HAVE_TORCHAUDIO:
            logging.error('Could not import torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary for "
                "AudioToMFCCPreprocessor. We recommend you try "
                "building it from source for the PyTorch version you have."
            )
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        # Get win_length (n_window_size) and hop_length (n_window_stride)
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        super().__init__(n_window_size, n_window_stride)

        mel_kwargs = {}

        mel_kwargs['f_min'] = lowfreq
        mel_kwargs['f_max'] = highfreq
        mel_kwargs['n_mels'] = n_mels

        mel_kwargs['n_fft'] = n_fft or 2 ** math.ceil(math.log2(n_window_size))

        mel_kwargs['win_length'] = n_window_size
        mel_kwargs['hop_length'] = n_window_stride

        # Set window_fn. None defaults to torch.ones.
        window_fn = self.torch_windows.get(window, None)
        if window_fn is None:
            raise ValueError(
                f"Window argument for AudioProcessor is invalid: {window}."
                f"For no window function, use 'ones' or None."
            )
        mel_kwargs['window_fn'] = window_fn

        # Use torchaudio's implementation of MFCCs as featurizer
        self.featurizer = torchaudio.transforms.MFCC(
            sample_rate=self._sample_rate,
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            log_mels=log,
            melkwargs=mel_kwargs,
        )

    def get_features(self, input_signal, length):
        features = self.featurizer(input_signal)
        seq_len = torch.ceil(length.to(torch.float32) / self.hop_length).to(dtype=torch.long)
        return features, seq_len


class SpectrogramAugmentation(NeuralModule):
    """
    Performs time and freq cuts in one of two ways.
    SpecAugment zeroes out vertical and horizontal sections as described in
    SpecAugment (https://arxiv.org/abs/1904.08779). Arguments for use with
    SpecAugment are `freq_masks`, `time_masks`, `freq_width`, and `time_width`.
    SpecCutout zeroes out rectangulars as described in Cutout
    (https://arxiv.org/abs/1708.04552). Arguments for use with Cutout are
    `rect_masks`, `rect_freq`, and `rect_time`.

    Args:
        freq_masks (int): how many frequency segments should be cut.
            Defaults to 0.
        time_masks (int): how many time segments should be cut
            Defaults to 0.
        freq_width (int): maximum number of frequencies to be cut in one
            segment.
            Defaults to 10.
        time_width (int): maximum number of time steps to be cut in one
            segment
            Defaults to 10.
        rect_masks (int): how many rectangular masks should be cut
            Defaults to 0.
        rect_freq (int): maximum size of cut rectangles along the frequency
            dimension
            Defaults to 5.
        rect_time (int): maximum size of cut rectangles along the time
            dimension
            Defaults to 25.
    """

    @property
    def input_types(self):
        """Returns definitions of module input types
        """
        return {
            "input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types
        """
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(
        self,
        freq_masks=0,
        time_masks=0,
        freq_width=10,
        time_width=10,
        rect_masks=0,
        rect_time=5,
        rect_freq=20,
        rng=None,
        mask_value=0.0,
        use_numba_spec_augment: bool = True,
    ):
        super().__init__()

        if rect_masks > 0:
            self.spec_cutout = SpecCutout(rect_masks=rect_masks, rect_time=rect_time, rect_freq=rect_freq, rng=rng,)
            # self.spec_cutout.to(self._device)
        else:
            self.spec_cutout = lambda input_spec: input_spec
        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng,
                mask_value=mask_value,
            )
        else:
            self.spec_augment = lambda input_spec, length: input_spec

        # Check if numba is supported, and use a Numba kernel if it is
        if use_numba_spec_augment and numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__):
            logging.info('Numba CUDA SpecAugment kernel is being used')
            self.spec_augment_numba = SpecAugmentNumba(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng,
                mask_value=mask_value,
            )
        else:
            self.spec_augment_numba = None

    @typecheck()
    def forward(self, input_spec, length):
        augmented_spec = self.spec_cutout(input_spec=input_spec)

        # To run the Numba kernel, correct numba version is required as well as
        # tensor must be on GPU and length must be provided
        if self.spec_augment_numba is not None and spec_augment_launch_heuristics(augmented_spec, length):
            augmented_spec = self.spec_augment_numba(input_spec=augmented_spec, length=length)
        else:
            augmented_spec = self.spec_augment(input_spec=augmented_spec, length=length)
        return augmented_spec


class MaskedPatchAugmentation(NeuralModule):
    """
        Zeroes out fixed size time patches of the spectrogram.
        All samples in batch are guaranteed to have the same amount of masked time steps.
        Optionally also performs frequency masking in the same way as SpecAugment.
        Args:
            patch_size (int): up to how many time steps does one patch consist of.
                Defaults to 48.
            mask_patches (float): how many patches should be masked in each sample.
                if >= 1., interpreted as number of patches (after converting to int)
                if <1.,   interpreted as fraction of total tokens to be masked (number of patches is rounded up)
                Defaults to 10.
            freq_masks (int): how many frequency segments should be cut.
                Defaults to 0.
            freq_width (int): maximum number of frequencies to be cut in a segment.
                Defaults to 0.
    """

    @property
    def input_types(self):
        """Returns definitions of module input types
        """
        return {
            "input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types
        """
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(
        self, patch_size: int = 48, mask_patches: float = 10.0, freq_masks: int = 0, freq_width: int = 0,
    ):
        super().__init__()
        self.patch_size = patch_size
        if mask_patches >= 1:
            self.mask_patches = int(mask_patches)
        elif mask_patches >= 0:
            self._mask_fraction = mask_patches
            self.mask_patches = None
        else:
            raise ValueError('mask_patches cannot be negative')

        if freq_masks > 0:
            self.spec_augment = SpecAugment(freq_masks=freq_masks, time_masks=0, freq_width=freq_width, time_width=0,)
        else:
            self.spec_augment = None

    @typecheck()
    def forward(self, input_spec, length):
        augmented_spec = input_spec

        min_len = torch.min(length)

        if self.mask_patches is None:
            # masking specified as fraction
            len_fraction = int(min_len * self._mask_fraction)
            mask_patches = len_fraction // self.patch_size + int(len_fraction % self.patch_size != 0)
        else:
            mask_patches = self.mask_patches

        if min_len < self.patch_size * mask_patches:
            mask_patches = min_len // self.patch_size

        for idx in range(input_spec.shape[0]):
            cur_len = length[idx]
            patches = range(cur_len // self.patch_size - 1)
            masked_patches = random.sample(patches, mask_patches)

            for mp in masked_patches:
                augmented_spec[idx, :, mp * self.patch_size : (mp + 1) * self.patch_size] = 0.0

        if self.spec_augment is not None:
            augmented_spec = self.spec_augment(input_spec=augmented_spec, length=length)

        return augmented_spec


class CropOrPadSpectrogramAugmentation(NeuralModule):
    """
    Pad or Crop the incoming Spectrogram to a certain shape.

    Args:
        audio_length (int): the final number of timesteps that is required.
            The signal will be either padded or cropped temporally to this
            size.
    """

    def __init__(self, audio_length):
        super(CropOrPadSpectrogramAugmentation, self).__init__()
        self.audio_length = audio_length

    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length):
        image = input_signal
        num_images = image.shape[0]

        audio_length = self.audio_length
        image_len = image.shape[-1]

        # Crop long signal
        if image_len > audio_length:  # randomly slice
            cutout_images = []
            offset = torch.randint(low=0, high=image_len - audio_length + 1, size=[num_images])

            for idx, offset in enumerate(offset):
                cutout_images.append(image[idx : idx + 1, :, offset : offset + audio_length])

            image = torch.cat(cutout_images, dim=0)
            del cutout_images

        else:  # symmetrically pad short signal with zeros
            pad_left = (audio_length - image_len) // 2
            pad_right = (audio_length - image_len) // 2

            if (audio_length - image_len) % 2 == 1:
                pad_right += 1

            image = torch.nn.functional.pad(image, [pad_left, pad_right], mode="constant", value=0)

        # Replace dynamic length sequences with static number of timesteps
        length = (length * 0) + audio_length

        return image, length

    @property
    def input_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "input_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass


@dataclass
class AudioToMelSpectrogramPreprocessorConfig:
    _target_: str = "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor"
    sample_rate: int = 16000
    window_size: float = 0.02
    window_stride: float = 0.01
    n_window_size: Optional[int] = None
    n_window_stride: Optional[int] = None
    window: str = "hann"
    normalize: str = "per_feature"
    n_fft: Optional[int] = None
    preemph: float = 0.97
    features: int = 64
    lowfreq: int = 0
    highfreq: Optional[int] = None
    log: bool = True
    log_zero_guard_type: str = "add"
    log_zero_guard_value: float = 2 ** -24
    dither: float = 1e-5
    pad_to: int = 16
    frame_splicing: int = 1
    exact_pad: bool = False
    pad_value: int = 0
    mag_power: float = 2.0
    rng: Optional[str] = None
    nb_augmentation_prob: float = 0.0
    nb_max_freq: int = 4000
    use_torchaudio: bool = False
    stft_exact_pad: bool = False  # Deprecated argument, kept for compatibility with older checkpoints.
    stft_conv: bool = False  # Deprecated argument, kept for compatibility with older checkpoints.


@dataclass
class AudioToMFCCPreprocessorConfig:
    _target_: str = 'nemo.collections.asr.modules.AudioToMFCCPreprocessor'
    sample_rate: int = 16000
    window_size: float = 0.02
    window_stride: float = 0.01
    n_window_size: Optional[int] = None
    n_window_stride: Optional[int] = None
    window: str = 'hann'
    n_fft: Optional[int] = None
    lowfreq: Optional[float] = 0.0
    highfreq: Optional[float] = None
    n_mels: int = 64
    n_mfcc: int = 64
    dct_type: int = 2
    norm: str = 'ortho'
    log: bool = True


@dataclass
class SpectrogramAugmentationConfig:
    _target_: str = "nemo.collections.asr.modules.SpectrogramAugmentation"
    freq_masks: int = 0
    time_masks: int = 0
    freq_width: int = 0
    time_width: Optional[Any] = 0
    rect_masks: int = 0
    rect_time: int = 0
    rect_freq: int = 0
    mask_value: float = 0
    rng: Optional[Any] = None  # random.Random() type
    use_numba_spec_augment: bool = True


@dataclass
class CropOrPadSpectrogramAugmentationConfig:
    audio_length: int
    _target_: str = "nemo.collections.asr.modules.CropOrPadSpectrogramAugmentation"


@dataclass
class MaskedPatchAugmentationConfig:
    patch_size: int = 48
    mask_patches: float = 10.0
    freq_masks: int = 0
    freq_width: int = 0
    _target_: str = "nemo.collections.asr.modules.MaskedPatchAugmentation"
