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
#
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# This file contains code artifacts adapted from https://github.com/ryanleary/patter
import math
import random
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn

from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.utils import logging

try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


CONSTANT = 1e-5


def normalize_batch(x, seq_len, normalize_type):
    x_mean = None
    x_std = None
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            if x[i, :, : seq_len[i]].shape[1] == 1:
                raise ValueError(
                    "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                    "in torch.std() returning nan. Make sure your audio length has enough samples for a single "
                    "feature (ex. at least `hop_length` for Mel Spectrograms)."
                )
            x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
    elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
        x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
        return (
            (x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2)) / x_std.view(x.shape[0], x.shape[1]).unsqueeze(2),
            x_mean,
            x_std,
        )
    else:
        return x, x_mean, x_std


def clean_spectrogram_batch(spectrogram: torch.Tensor, spectrogram_len: torch.Tensor, fill_value=0.0) -> torch.Tensor:
    """
    Fill spectrogram values outside the length with `fill_value`

    Args:
        spectrogram: Tensor with shape [B, C, L] containing batched spectrograms
        spectrogram_len: Tensor with shape [B] containing the sequence length of each batch element
        fill_value: value to fill with, 0.0 by default

    Returns:
        cleaned spectrogram, tensor with shape equal to `spectrogram`
    """
    device = spectrogram.device
    batch_size, _, max_len = spectrogram.shape
    mask = torch.arange(max_len, device=device)[None, :] >= spectrogram_len[:, None]
    mask = mask.unsqueeze(1).expand_as(spectrogram)
    return spectrogram.masked_fill(mask, fill_value)


def splice_frames(x, frame_splicing):
    """ Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


@torch.jit.script_if_tracing
def make_seq_mask_like(
    lengths: torch.Tensor, like: torch.Tensor, time_dim: int = -1, valid_ones: bool = True
) -> torch.Tensor:
    """

    Args:
        lengths: Tensor with shape [B] containing the sequence length of each batch element
        like: The mask will contain the same number of dimensions as this Tensor, and will have the same max
            length in the time dimension of this Tensor.
        time_dim: Time dimension of the `shape_tensor` and the resulting mask. Zero-based.
        valid_ones: If True, valid tokens will contain value `1` and padding will be `0`. Else, invert.

    Returns:
        A :class:`torch.Tensor` containing 1's and 0's for valid and invalid tokens, respectively, if `valid_ones`, else
        vice-versa. Mask will have the same number of dimensions as `like`. Batch and time dimensions will match
        the `like`. All other dimensions will be singletons. E.g., if `like.shape == [3, 4, 5]` and
        `time_dim == -1', mask will have shape `[3, 1, 5]`.
    """
    # Mask with shape [B, T]
    mask = torch.arange(like.shape[time_dim], device=like.device).repeat(lengths.shape[0], 1).lt(lengths.view(-1, 1))
    # [B, T] -> [B, *, T] where * is any number of singleton dimensions to expand to like tensor
    for _ in range(like.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    # If needed, transpose time dim
    if time_dim != -1 and time_dim != mask.dim() - 1:
        mask = mask.transpose(-1, time_dim)
    # Maybe invert the padded vs. valid token values
    if not valid_ones:
        mask = ~mask
    return mask


class WaveformFeaturizer(object):
    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(
        self,
        file_path,
        offset=0,
        duration=0,
        trim=False,
        trim_ref=np.max,
        trim_top_db=60,
        trim_frame_length=2048,
        trim_hop_length=512,
        orig_sr=None,
        channel_selector=None,
        normalize_db=None,
    ):
        audio = AudioSegment.from_file(
            file_path,
            target_sr=self.sample_rate,
            int_values=self.int_values,
            offset=offset,
            duration=duration,
            trim=trim,
            trim_ref=trim_ref,
            trim_top_db=trim_top_db,
            trim_frame_length=trim_frame_length,
            trim_hop_length=trim_hop_length,
            orig_sr=orig_sr,
            channel_selector=channel_selector,
            normalize_db=normalize_db,
        )
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        sample_rate = input_config.get("sample_rate", 16000)
        int_values = input_config.get("int_values", False)

        return cls(sample_rate=sample_rate, int_values=int_values, augmentor=aa)


class FeaturizerFactory(object):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, input_cfg, perturbation_configs=None):
        return WaveformFeaturizer.from_config(input_cfg, perturbation_configs=perturbation_configs)


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=CONSTANT,
        pad_to=16,
        max_duration=16.7,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        if stft_conv or stft_exact_pad:
            logging.warning(
                "Using torch_stft is deprecated and has been removed. The values have been forcibly set to False "
                "for FilterbankFeatures and AudioToMelSpectrogramPreprocessor. Please set exact_pad to True "
                "as needed."
            )
        if exact_pad and n_window_stride % 2 == 1:
            raise NotImplementedError(
                f"{self} received exact_pad == True, but hop_size was odd. If audio_length % hop_size == 0. Then the "
                "returned spectrogram would not be of length audio_length // hop_size. Please use an even hop_size."
            )
        self.log_zero_guard_value = log_zero_guard_value
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        logging.info(f"PADDING: {pad_to}")

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None

        if exact_pad:
            logging.info("STFT using exact pad")
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)
        self.stft = lambda x: torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if exact_pad else True,
            window=self.window.to(dtype=torch.float),
            return_complex=True,
        )

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq, norm=mel_norm
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * n_fft)

        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type
        logging.debug(f"sr: {sample_rate}")
        logging.debug(f"n_fft: {self.n_fft}")
        logging.debug(f"win_length: {self.win_length}")
        logging.debug(f"hop_length: {self.hop_length}")
        logging.debug(f"n_mels: {nfilt}")
        logging.debug(f"fmin: {lowfreq}")
        logging.debug(f"fmax: {highfreq}")
        logging.debug(f"using grads: {use_grads}")
        logging.debug(f"nb_augmentation_prob: {nb_augmentation_prob}")

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length) + 1
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def forward(self, x, seq_len, linear_spec=False):
        seq_len = self.get_seq_len(seq_len)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "reflect"
            ).squeeze(1)

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        # disable autocast to get full range of stft values
        with torch.cuda.amp.autocast(enabled=False):
            x = self.stft(x)

        # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else CONSTANT
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin :, :] = 0.0

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # return plain spectrogram if required
        if linear_spec:
            return x, seq_len

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)
        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
        return x, seq_len


class FilterbankFeaturesTA(nn.Module):
    """
    Exportable, `torchaudio`-based implementation of Mel Spectrogram extraction.

    See `AudioToMelSpectrogramPreprocessor` for args.

    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_window_size: int = 320,
        n_window_stride: int = 160,
        normalize: Optional[str] = "per_feature",
        nfilt: int = 64,
        n_fft: Optional[int] = None,
        preemph: float = 0.97,
        lowfreq: float = 0,
        highfreq: Optional[float] = None,
        log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: Union[float, str] = 2 ** -24,
        dither: float = 1e-5,
        window: str = "hann",
        pad_to: int = 0,
        pad_value: float = 0.0,
        mel_norm="slaney",
        # Seems like no one uses these options anymore. Don't convolute the code by supporting thm.
        use_grads: bool = False,  # Deprecated arguments; kept for config compatibility
        max_duration: float = 16.7,  # Deprecated arguments; kept for config compatibility
        frame_splicing: int = 1,  # Deprecated arguments; kept for config compatibility
        exact_pad: bool = False,  # Deprecated arguments; kept for config compatibility
        nb_augmentation_prob: float = 0.0,  # Deprecated arguments; kept for config compatibility
        nb_max_freq: int = 4000,  # Deprecated arguments; kept for config compatibility
        mag_power: float = 2.0,  # Deprecated arguments; kept for config compatibility
        rng: Optional[random.Random] = None,  # Deprecated arguments; kept for config compatibility
        stft_exact_pad: bool = False,  # Deprecated arguments; kept for config compatibility
        stft_conv: bool = False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        if not HAVE_TORCHAUDIO:
            raise ValueError(f"Need to install torchaudio to instantiate a {self.__class__.__name__}")

        # Make sure log zero guard is supported, if given as a string
        supported_log_zero_guard_strings = {"eps", "tiny"}
        if isinstance(log_zero_guard_value, str) and log_zero_guard_value not in supported_log_zero_guard_strings:
            raise ValueError(
                f"Log zero guard value must either be a float or a member of {supported_log_zero_guard_strings}"
            )

        # Copied from `AudioPreprocessor` due to the ad-hoc structuring of the Mel Spec extractor class
        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }

        # Ensure we can look up the window function
        if window not in self.torch_windows:
            raise ValueError(f"Got window value '{window}' but expected a member of {self.torch_windows.keys()}")

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self._sample_rate = sample_rate
        self._normalize_strategy = normalize
        self._use_log = log
        self._preemphasis_value = preemph
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value: Union[str, float] = log_zero_guard_value
        self.dither = dither
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.n_fft = n_fft
        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=nfilt,
            window_fn=self.torch_windows[window],
            mel_scale="slaney",
            norm=mel_norm,
            n_fft=n_fft,
            f_max=highfreq,
            f_min=lowfreq,
            wkwargs={"periodic": False},
        )

    @property
    def filter_banks(self):
        """ Matches the analogous class """
        return self._mel_spec_extractor.mel_scale.fb

    def _resolve_log_zero_guard_value(self, dtype: torch.dtype) -> float:
        if isinstance(self.log_zero_guard_value, float):
            return self.log_zero_guard_value
        return getattr(torch.finfo(dtype), self.log_zero_guard_value)

    def _apply_dithering(self, signals: torch.Tensor) -> torch.Tensor:
        if self.training and self.dither > 0.0:
            noise = torch.randn_like(signals) * self.dither
            signals = signals + noise
        return signals

    def _apply_preemphasis(self, signals: torch.Tensor) -> torch.Tensor:
        if self._preemphasis_value is not None:
            padded = torch.nn.functional.pad(signals, (1, 0))
            signals = signals - self._preemphasis_value * padded[:, :-1]
        return signals

    def _compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        out_lengths = input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()
        return out_lengths

    def _apply_pad_to(self, features: torch.Tensor) -> torch.Tensor:
        # Only apply during training; else need to capture dynamic shape for exported models
        if not self.training or self.pad_to == 0 or features.shape[-1] % self.pad_to == 0:
            return features
        pad_length = self.pad_to - (features.shape[-1] % self.pad_to)
        return torch.nn.functional.pad(features, pad=(0, pad_length), value=self.pad_value)

    def _apply_log(self, features: torch.Tensor) -> torch.Tensor:
        if self._use_log:
            zero_guard = self._resolve_log_zero_guard_value(features.dtype)
            if self.log_zero_guard_type == "add":
                features = features + zero_guard
            elif self.log_zero_guard_type == "clamp":
                features = features.clamp(min=zero_guard)
            else:
                raise ValueError(f"Unsupported log zero guard type: '{self.log_zero_guard_type}'")
            features = features.log()
        return features

    def _extract_spectrograms(self, signals: torch.Tensor) -> torch.Tensor:
        # Complex FFT needs to be done in single precision
        with torch.cuda.amp.autocast(enabled=False):
            features = self._mel_spec_extractor(waveform=signals)
        return features

    def _apply_normalization(self, features: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # For consistency, this function always does a masked fill even if not normalizing.
        mask: torch.Tensor = make_seq_mask_like(lengths=lengths, like=features, time_dim=-1, valid_ones=False)
        features = features.masked_fill(mask, 0.0)
        # Maybe don't normalize
        if self._normalize_strategy is None:
            return features
        # Use the log zero guard for the sqrt zero guard
        guard_value = self._resolve_log_zero_guard_value(features.dtype)
        if self._normalize_strategy == "per_feature" or self._normalize_strategy == "all_features":
            # 'all_features' reduces over each sample; 'per_feature' reduces over each channel
            reduce_dim = 2
            if self._normalize_strategy == "all_features":
                reduce_dim = [1, 2]
            # [B, D, T] -> [B, D, 1] or [B, 1, 1]
            means = features.sum(dim=reduce_dim, keepdim=True).div(lengths.view(-1, 1, 1))
            stds = (
                features.sub(means)
                .masked_fill(mask, 0.0)
                .pow(2.0)
                .sum(dim=reduce_dim, keepdim=True)  # [B, D, T] -> [B, D, 1] or [B, 1, 1]
                .div(lengths.view(-1, 1, 1) - 1)  # assume biased estimator
                .clamp(min=guard_value)  # avoid sqrt(0)
                .sqrt()
            )
            features = (features - means) / (stds + eps)
        else:
            # Deprecating constant std/mean
            raise ValueError(f"Unsupported norm type: '{self._normalize_strategy}")
        features = features.masked_fill(mask, 0.0)
        return features

    def forward(self, input_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_lengths = self._compute_output_lengths(input_lengths=length)
        signals = self._apply_dithering(signals=input_signal)
        signals = self._apply_preemphasis(signals=signals)
        features = self._extract_spectrograms(signals=signals)
        features = self._apply_log(features=features)
        features = self._apply_normalization(features=features, lengths=feature_lengths)
        features = self._apply_pad_to(features=features)
        return features, feature_lengths
