# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional, Union

import torch
import torch.nn as nn

CONSTANT = 1e-5


def _hz_to_mel(frequencies: torch.Tensor) -> torch.Tensor:
    """Slaney mel conversion matching librosa with htk=False."""
    f_sp = 200.0 / 3
    mels = frequencies / f_sp

    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = frequencies >= min_log_hz
    if log_t.any():
        mels = torch.where(log_t, min_log_mel + torch.log(frequencies / min_log_hz) / logstep, mels)
    return mels


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    """Inverse Slaney mel conversion matching librosa with htk=False."""
    f_sp = 200.0 / 3
    freqs = f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    if log_t.any():
        freqs = torch.where(log_t, min_log_hz * torch.exp(logstep * (mels - min_log_mel)), freqs)
    return freqs


def _mel_frequencies(n_mels: int, fmin: float, fmax: float) -> torch.Tensor:
    min_mel = _hz_to_mel(torch.tensor(float(fmin), dtype=torch.float64))
    max_mel = _hz_to_mel(torch.tensor(float(fmax), dtype=torch.float64))
    mels = torch.linspace(min_mel, max_mel, n_mels, dtype=torch.float64)
    return _mel_to_hz(mels)


def _normalize_filterbank(filterbank: torch.Tensor, norm: Optional[Union[str, float]]) -> torch.Tensor:
    if norm is None:
        return filterbank

    if norm == "slaney":
        return filterbank

    if not isinstance(norm, (int, float)):
        raise ValueError(f"Unsupported mel_norm value: {norm!r}")

    norm = float(norm)
    magnitudes = filterbank.abs()
    if math.isinf(norm):
        lengths = magnitudes.max(dim=-1, keepdim=True).values
    elif norm == 0:
        lengths = (magnitudes > 0).sum(dim=-1, keepdim=True).to(filterbank.dtype)
    else:
        lengths = magnitudes.pow(norm).sum(dim=-1, keepdim=True).pow(1.0 / norm)

    tiny = torch.finfo(filterbank.dtype).tiny
    return torch.where(lengths > tiny, filterbank / lengths.clamp_min(tiny), filterbank)


def _create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    norm: Optional[Union[str, float]],
) -> torch.Tensor:
    """Create a mel filter bank equivalent to librosa.filters.mel(..., htk=False)."""
    fftfreqs = torch.linspace(0, float(sample_rate) / 2, n_fft // 2 + 1, dtype=torch.float64)
    mel_f = _mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)

    fdiff = mel_f[1:] - mel_f[:-1]
    ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)

    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    upper = ramps[2:] / fdiff[1:].unsqueeze(1)
    weights = torch.minimum(lower, upper).clamp_min(0)

    if norm == "slaney":
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm.unsqueeze(1)
    else:
        weights = _normalize_filterbank(weights, norm)

    return weights.to(dtype=torch.float32).unsqueeze(0)


def normalize_batch(x: torch.Tensor, seq_len: torch.Tensor, normalize_type):
    x_mean = None
    x_std = None
    if normalize_type == "per_feature":
        batch_size = x.shape[0]
        max_time = x.shape[2]

        if (
            torch.cuda.is_available()
            and not torch.cuda.is_current_stream_capturing()
            and torch.any(seq_len == 1).item()
        ):
            raise ValueError(
                "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                "in torch.std() returning nan. Make sure your audio length has enough samples for a single feature "
                "(ex. at least `hop_length` for Mel Spectrograms)."
            )
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        valid_mask = time_steps < seq_len.unsqueeze(1)
        x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
        x_mean_denominator = valid_mask.sum(axis=1)
        x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)

        x_std = torch.sqrt(
            torch.sum(torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
            / (x_mean_denominator.unsqueeze(1) - 1.0)
        )
        x_std = x_std.masked_fill(x_std.isnan(), 0.0)
        x_std += CONSTANT
        normalized = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
        normalized.masked_fill_(~valid_mask.unsqueeze(1), 0.0)
        return normalized, x_mean, x_std
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
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


def splice_frames(x: torch.Tensor, frame_splicing: int) -> torch.Tensor:
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


class AudioToMelSpectrogramPreprocessor(nn.Module):
    """Standalone PyTorch implementation of NeMo's log-mel ASR preprocessor.

    This class mirrors ``nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor``
    without importing NeMo ASR, Lightning, Hydra, librosa, or NeMo neural type
    dependencies. It uses only Python stdlib and PyTorch primitives.
    """

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
        log_zero_guard_value=2**-24,
        dither=1e-5,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        use_torchaudio: bool = False,
        stft_exact_pad=False,
        stft_conv=False,
    ):
        del use_torchaudio, stft_exact_pad, stft_conv
        super().__init__()

        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(f"{self} received both window_stride and n_window_stride. Only one should be specified.")
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or n_window_stride. Both must be positive ints."
            )
        if exact_pad and n_window_stride % 2 == 1:
            raise NotImplementedError(
                f"{self} received exact_pad == True, but hop_size was odd. If audio_length % hop_size == 0. Then the "
                "returned spectrogram would not be of length audio_length // hop_size. Please use an even hop_size."
            )
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the log_zero_guard_type parameter. It must be either "
                "'add' or 'clamp'."
            )

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None
        self.exact_pad = exact_pad
        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = features
        self.preemph = preemph
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.mag_power = mag_power
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob

        window_fns = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
        }
        window_fn = window_fns.get(window, None)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)

        highfreq = highfreq or sample_rate / 2
        self.register_buffer(
            "fb",
            _create_mel_filterbank(
                sample_rate=sample_rate,
                n_fft=self.n_fft,
                n_mels=features,
                fmin=lowfreq,
                fmax=highfreq,
                norm=mel_norm,
            ),
        )

        max_length = self.get_seq_len(torch.tensor(16.7 * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad

        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * self.n_fft)

        self.register_buffer("dtype_sentinel_tensor", torch.tensor((), dtype=torch.float32), persistent=False)

    @property
    def filter_banks(self) -> torch.Tensor:
        return self.fb

    def input_example(self, max_batch: int = 8, max_dim: int = 32000, min_length: int = 200):
        dev = self.filter_banks.device
        signals = torch.randn(size=[max_batch, max_dim], device=dev)
        lengths = torch.randint(low=min_length, high=max_dim, size=[max_batch], device=dev)
        lengths[0] = max_dim
        return signals, lengths

    def get_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
        return seq_len.to(dtype=torch.long)

    def log_zero_guard_value_fn(self, x: torch.Tensor):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the log_zero_guard_type parameter. It must be "
                    "either a number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        window = self.window.to(dtype=torch.float, device=x.device) if self.window is not None else None
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if self.exact_pad else True,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )

    @torch.no_grad()
    def get_features(self, input_signal: torch.Tensor, length: torch.Tensor, linear_spec: bool = False):
        x = input_signal
        seq_len_time = length
        seq_len_unfixed = self.get_seq_len(length)
        seq_len = torch.where(length == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
            ).squeeze(1)

        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)

        with torch.amp.autocast(x.device.type, enabled=False):
            x = self.stft(x)

        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1))

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin :, :] = 0.0

        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        if linear_spec:
            return x, seq_len

        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.matmul(self.fb.to(x.dtype), x)

        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask

        if self.pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, self.pad_to - pad_amt), value=self.pad_value)
        return x, seq_len

    @torch.no_grad()
    def forward(self, input_signal: torch.Tensor, length: torch.Tensor):
        processed_signal, processed_length = self.get_features(input_signal.to(torch.float32), length)
        processed_signal = processed_signal.to(self.dtype_sentinel_tensor.dtype)
        return processed_signal, processed_length
