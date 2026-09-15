# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import librosa
import numpy as np
import torch
import torch.nn as nn

from nemo.collections.asr.parts.packed_sequence import PackedEncoderActivations, _new_packed_encoder_activations
from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.utils import logging

CONSTANT = 1e-5


def normalize_batch(x, seq_len, normalize_type):
    x_mean = None
    x_std = None
    if normalize_type == "per_feature":
        batch_size = x.shape[0]
        max_time = x.shape[2]

        # When doing stream capture to a graph, item() is not allowed
        # becuase it calls cudaStreamSynchronize(). Therefore, we are
        # sacrificing some error checking when running with cuda graphs.
        if (
            torch.cuda.is_available()
            and not torch.cuda.is_current_stream_capturing()
            and torch.any(seq_len == 1).item()
        ):
            raise ValueError(
                "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                "in torch.std() returning nan. Make sure your audio length has enough samples for a single "
                "feature (ex. at least `hop_length` for Mel Spectrograms)."
            )
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        valid_mask = time_steps < seq_len.unsqueeze(1)
        x_mean_denominator = valid_mask.sum(axis=1)
        # Reference-centering keeps constant inputs exact across reduction backends
        # and matches the packed normalization path.
        reference = x[:, :, 0] if max_time else x.new_zeros((batch_size, x.shape[1]))
        reference = reference.masked_fill((x_mean_denominator == 0).unsqueeze(1), 0.0)
        centered = torch.where(valid_mask.unsqueeze(1), x - reference.unsqueeze(2), 0.0)
        x_mean = reference + centered.sum(axis=2) / x_mean_denominator.clamp_min(1).unsqueeze(1)

        # Subtract 1 in the denominator to correct for the bias.
        x_std = torch.sqrt(
            torch.sum(torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
            / (x_mean_denominator.unsqueeze(1) - 1.0)
        )
        x_std = x_std.masked_fill(x_std.isnan(), 0.0)  # edge case: only 1 frame in denominator
        # make sure x_std is not zero
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


def normalize_packed_batch(packed: PackedEncoderActivations, normalize_type) -> PackedEncoderActivations:
    """Normalize token-flat features independently within each sequence."""
    if not normalize_type or packed.total_tokens == 0:
        return packed
    sequence_ids = torch.repeat_interleave(torch.arange(packed.batch_size, device=packed.data.device), packed.lengths)
    data, padding_value = _normalize_packed_features_and_padding(
        packed.data,
        packed.lengths,
        sequence_ids,
        normalize_type,
        padding_value=packed.padding_value,
    )
    return _new_packed_encoder_activations(
        data,
        packed.lengths,
        packed.cu_seqlens,
        packed.max_seqlen,
        padding_value,
        padded_length=packed.padded_length,
    )


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
    """Stacks frames together across feature dim

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
        log_zero_guard_value=2**-24,
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

        self.sample_rate = sample_rate
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None
        self.exact_pad = exact_pad
        self.sample_rate = sample_rate

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
            self.forward_packed = torch.no_grad()(self.forward_packed)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * self.n_fft)

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

    def stft(self, x, *, center=None):
        if center is None:
            center = not self.exact_pad
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=center,
            window=self.window.to(dtype=torch.float, device=x.device),
            return_complex=True,
            pad_mode="constant",
        )

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
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def forward_packed(self, x, seq_len, cu_seqlens, linear_spec=False) -> PackedEncoderActivations:
        """Compute features from concatenated waveforms with one vectorized STFT.

        Each utterance is placed in a hop-aligned block with the same zero guard
        that the dense STFT applies at its boundaries. Only valid frames are
        gathered from the resulting single STFT, so both input and output remain
        sequence-packed and no ``B x T`` waveform or feature tensor is created.

        ``pad_to`` is intentionally ignored: it is a dense-layout optimization and
        packed output contains exactly ``sum(output_lengths)`` frames.
        """
        seq_len, cu_seqlens, host_seq_len = _validate_packed_waveforms(x, seq_len, cu_seqlens)
        feature_lengths = torch.where(seq_len == 0, 0, self.get_seq_len(seq_len))
        host_feature_lengths = torch.where(host_seq_len == 0, 0, self.get_seq_len(host_seq_len))
        padded_length = _dense_feature_width(host_feature_lengths, self.pad_to, self.max_length)
        max_seqlen = int(host_feature_lengths.max()) if host_feature_lengths.numel() else 0
        total_frames = int(host_feature_lengths.sum())
        if bool((host_feature_lengths < 0).any()):
            raise ValueError(
                "Packed waveform lengths are too short for this STFT configuration; "
                f"computed feature lengths {host_feature_lengths.tolist()}."
            )
        if seq_len.numel() == 0 or total_frames == 0:
            feature_dim = self.n_fft // 2 + 1 if linear_spec else self.nfilt * self.frame_splicing
            return _empty_packed_features(
                x,
                feature_lengths,
                feature_dim,
                padding_value=self.pad_value,
                padded_length=padded_length,
                max_seqlen=max_seqlen,
            )

        guard = self.stft_pad_amount if self.stft_pad_amount is not None else self.n_fft // 2
        block_lengths = _round_up(seq_len + 2 * guard, self.hop_length)
        block_offsets = torch.cat([seq_len.new_zeros(1), block_lengths.cumsum(0)])
        guarded_size = int(_round_up(host_seq_len + 2 * guard, self.hop_length).sum())
        guarded = x.new_zeros(guarded_size)

        guarded_positions = torch.arange(x.numel(), device=x.device)
        sample_sequence_ids = torch.bucketize(guarded_positions, cu_seqlens[1:], right=True)
        guarded_positions -= cu_seqlens[sample_sequence_ids]
        guarded_positions += block_offsets[sample_sequence_ids] + guard

        if self.stft_pad_amount is None:
            samples = _dither_and_preemphasize_packed(
                x, seq_len, cu_seqlens, self.preemph, self.dither if self.training else 0.0
            )
            guarded[guarded_positions] = samples
        else:
            guarded[guarded_positions] = x
            guarded = _dither_and_preemphasize_exact_pad_blocks(
                guarded,
                seq_len,
                block_lengths,
                block_offsets,
                self.preemph,
                self.dither if self.training else 0.0,
            )
        del guarded_positions, sample_sequence_ids

        with torch.amp.autocast(x.device.type, enabled=False):
            spectra = self.stft(guarded.unsqueeze(0), center=False)[0]

        frame_cu_seqlens = torch.cat([feature_lengths.new_zeros(1), feature_lengths.cumsum(0)])
        frame_indices = torch.arange(total_frames, device=x.device)
        frame_sequence_ids = torch.bucketize(frame_indices, frame_cu_seqlens[1:], right=True)
        local_frames = frame_indices - frame_cu_seqlens[frame_sequence_ids]
        global_frames = torch.div(block_offsets[frame_sequence_ids], self.hop_length, rounding_mode="floor")
        global_frames = global_frames + local_frames
        spectra = spectra.index_select(-1, global_frames).transpose(0, 1)

        guard_value = 0 if not self.use_grads else CONSTANT
        spectra = torch.sqrt(torch.view_as_real(spectra).pow(2).sum(-1) + guard_value)
        if self.training and self.nb_augmentation_prob > 0.0:
            narrowband = torch.tensor(
                self._rng.choices(
                    (True, False),
                    weights=(self.nb_augmentation_prob, 1.0 - self.nb_augmentation_prob),
                    k=feature_lengths.numel(),
                ),
                device=x.device,
            )
            keep = ~(narrowband[frame_sequence_ids].unsqueeze(1) & _high_frequency_mask(spectra, self._nb_max_fft_bin))
            spectra = spectra * keep
        if self.mag_power != 1.0:
            spectra = spectra.pow(self.mag_power)
        if linear_spec:
            return _make_packed_features(
                spectra,
                feature_lengths,
                padding_value=self.pad_value,
                padded_length=padded_length,
                max_seqlen=max_seqlen,
            )

        with torch.amp.autocast(x.device.type, enabled=False):
            features = torch.matmul(self.fb.to(spectra.dtype), spectra.transpose(0, 1).unsqueeze(0))[0].transpose(0, 1)
        if self.log:
            if self.log_zero_guard_type == "add":
                features = torch.log(features + self.log_zero_guard_value_fn(features))
            elif self.log_zero_guard_type == "clamp":
                features = torch.log(torch.clamp(features, min=self.log_zero_guard_value_fn(features)))
            else:
                raise ValueError("log_zero_guard_type was not understood")
        if self.frame_splicing > 1:
            features = features.repeat(1, self.frame_splicing)
        if self.normalize:
            features = _normalize_packed_features(features, feature_lengths, frame_sequence_ids, self.normalize)
        return _make_packed_features(
            features,
            feature_lengths,
            padding_value=self.pad_value,
            padded_length=padded_length,
            max_seqlen=max_seqlen,
        )

    def forward(self, x, seq_len, linear_spec=False):
        seq_len_time = seq_len
        seq_len_unfixed = self.get_seq_len(seq_len)
        # fix for seq_len = 0 for streaming; if size was 0, it is always padded to 1, and normalizer fails
        seq_len = torch.where(seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
            ).squeeze(1)

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)

        # disable autocast to get full range of stft values
        with torch.amp.autocast(x.device.type, enabled=False):
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

        # disable autocast, otherwise it might be automatically casted to fp16
        # on fp16 compatible GPUs and get NaN values for input value of 65520
        with torch.amp.autocast(x.device.type, enabled=False):
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
        mask = torch.arange(max_len, device=x.device)
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


def _validate_packed_waveforms(x, seq_len, cu_seqlens):
    if x.ndim != 1:
        raise ValueError(f"packed waveform data must be 1D, got shape {tuple(x.shape)}.")
    if seq_len.ndim != 1:
        raise ValueError(f"length must be 1D, got shape {tuple(seq_len.shape)}.")
    if seq_len.dtype == torch.bool or seq_len.is_floating_point() or seq_len.is_complex():
        raise TypeError(f"length must have an integer dtype, got {seq_len.dtype}.")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != seq_len.numel() + 1:
        raise ValueError(f"cu_seqlens must have shape ({seq_len.numel() + 1},), got {tuple(cu_seqlens.shape)}.")
    if cu_seqlens.dtype == torch.bool or cu_seqlens.is_floating_point() or cu_seqlens.is_complex():
        raise TypeError(f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}.")
    if x.device != seq_len.device or x.device != cu_seqlens.device:
        raise ValueError("packed waveform data, length, and cu_seqlens must be on the same device.")
    seq_len = seq_len.to(torch.int64)
    cu_seqlens = cu_seqlens.to(torch.int64)
    host_metadata = torch.cat((seq_len, cu_seqlens)).detach().cpu()
    host_seq_len = host_metadata[: seq_len.numel()]
    host_cu_seqlens = host_metadata[seq_len.numel() :]
    if host_cu_seqlens.numel() and int(host_cu_seqlens[0]) != 0:
        raise ValueError("cu_seqlens must start at zero.")
    if not torch.equal(host_cu_seqlens[1:] - host_cu_seqlens[:-1], host_seq_len):
        raise ValueError("Differences in cu_seqlens must equal length.")
    if bool((host_seq_len < 0).any()):
        raise ValueError("length must be non-negative.")
    if int(host_cu_seqlens[-1]) != x.shape[0]:
        raise ValueError(
            f"packed waveform data has {x.shape[0]} samples, but cu_seqlens ends at {host_cu_seqlens[-1]}."
        )
    return seq_len, cu_seqlens, host_seq_len


def _round_up(values, multiple):
    return torch.div(values + multiple - 1, multiple, rounding_mode="floor") * multiple


def _dense_feature_width(lengths, pad_to, max_length):
    width = int(lengths.max().item()) + 1 if lengths.numel() else 0
    if pad_to == "max":
        return int(max_length)
    if pad_to > 0 and width % pad_to:
        width += pad_to - width % pad_to
    return width


def _dither_and_preemphasize_packed(x, lengths, cu_seqlens, preemph, dither):
    samples = x + dither * torch.randn_like(x) if dither > 0 else x
    if preemph is None or samples.numel() == 0:
        return samples
    emphasized = torch.cat([samples[:1], samples[1:] - preemph * samples[:-1]])
    starts = cu_seqlens[:-1][lengths > 0]
    return emphasized.scatter(0, starts, samples.index_select(0, starts))


def _dither_and_preemphasize_exact_pad_blocks(guarded, lengths, block_lengths, block_offsets, preemph, dither):
    if dither > 0:
        guarded = guarded + dither * torch.randn_like(guarded)
    if preemph is not None and guarded.numel() > 0:
        emphasized = torch.cat([guarded[:1], guarded[1:] - preemph * guarded[:-1]])
        starts = block_offsets[:-1]
        guarded = emphasized.scatter(0, starts, guarded.index_select(0, starts))
        block_ids = torch.repeat_interleave(torch.arange(lengths.numel(), device=guarded.device), block_lengths)
        local_samples = torch.arange(guarded.numel(), device=guarded.device) - block_offsets[block_ids]
        guarded = guarded.masked_fill(local_samples >= lengths[block_ids], 0.0)
    return guarded


def _high_frequency_mask(spectra, first_masked_bin):
    bins = torch.arange(spectra.shape[1], device=spectra.device)
    return bins.unsqueeze(0) >= first_masked_bin


def _normalize_packed_features(features, lengths, sequence_ids, normalize_type):
    normalized, _ = _normalize_packed_features_and_padding(features, lengths, sequence_ids, normalize_type)
    return normalized


def _normalize_packed_features_and_padding(features, lengths, sequence_ids, normalize_type, *, padding_value=None):
    if normalize_type == "per_feature":
        input_dtype = features.dtype
        statistics_features = _packed_normalization_statistics_features(features)
        denominator = lengths.clamp_min(1).unsqueeze(1)
        reference = _packed_segment_reference(statistics_features, lengths)
        mean = reference + _packed_segment_sum(statistics_features - reference[sequence_ids], lengths) / denominator
        centered = statistics_features - mean[sequence_ids]
        variance = _packed_segment_sum(centered.square(), lengths) / (denominator - 1)
        std = torch.sqrt(variance).masked_fill(variance.isnan(), 0.0) + CONSTANT
        normalized = (centered / std[sequence_ids]).to(input_dtype)
        normalized_padding = features.new_zeros((lengths.numel(), features.shape[1]))
        return normalized, normalized_padding
    if normalize_type == "all_features":
        input_dtype = features.dtype
        statistics_features = _packed_normalization_statistics_features(features)
        denominator = lengths * features.shape[1]
        mean = _packed_segment_sum(statistics_features.sum(1), lengths) / denominator.clamp_min(1)
        centered = statistics_features - mean[sequence_ids].unsqueeze(1)
        variance = _packed_segment_sum(centered.square().sum(1), lengths) / (denominator.clamp_min(1) - 1)
        std = torch.sqrt(variance).masked_fill(variance.isnan(), 0.0) + CONSTANT
        normalized = (centered / std[sequence_ids].unsqueeze(1)).to(input_dtype)
        padding = _expand_packed_padding(padding_value, statistics_features, lengths)
        normalized_padding = (
            None if padding is None else ((padding - mean.unsqueeze(1)) / std.unsqueeze(1)).to(input_dtype)
        )
        return normalized, normalized_padding
    if "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        mean = torch.as_tensor(normalize_type["fixed_mean"], device=features.device, dtype=features.dtype)
        std = torch.as_tensor(normalize_type["fixed_std"], device=features.device, dtype=features.dtype)
        if mean.numel() == features.shape[1]:
            normalized = (features - mean) / std
            padding = _expand_packed_padding(padding_value, features, lengths)
            normalized_padding = None if padding is None else (padding - mean) / std
            return normalized, normalized_padding
        mean = mean.view(lengths.numel(), features.shape[1])
        std = std.view(lengths.numel(), features.shape[1])
        normalized = (features - mean[sequence_ids]) / std[sequence_ids]
        padding = _expand_packed_padding(padding_value, features, lengths)
        normalized_padding = None if padding is None else (padding - mean) / std
        return normalized, normalized_padding
    return features, padding_value


def _packed_segment_sum(values, lengths):
    # Public packed entry points validate lengths; avoid repeating their synchronizing checks here.
    return torch.segment_reduce(values, "sum", lengths=lengths, unsafe=True)


def _packed_segment_reference(values, lengths):
    """Return one value per segment, with zeros for empty segments."""
    starts = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)[:-1]]).long()
    references = values.index_select(0, starts.clamp_max(values.shape[0] - 1))
    return references.masked_fill((lengths == 0).unsqueeze(1), 0.0)


def _packed_normalization_statistics_features(features):
    """Accumulate packed normalization statistics safely for low-precision inputs."""
    if features.dtype in (torch.float16, torch.bfloat16):
        return features.float()
    return features


def _expand_packed_padding(padding_value, features, lengths):
    if padding_value is None:
        return None
    if isinstance(padding_value, torch.Tensor):
        return padding_value.to(features)
    return features.new_full((lengths.numel(), features.shape[1]), padding_value)


def _make_packed_features(features, lengths, *, padding_value=0.0, padded_length=None, max_seqlen=None):
    cu_seqlens = torch.cat([lengths.new_zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)])
    if max_seqlen is None:
        max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
    return _new_packed_encoder_activations(
        features, lengths.to(torch.int64), cu_seqlens, max_seqlen, padding_value, padded_length
    )


def _empty_packed_features(x, lengths, feature_dim, *, padding_value=0.0, padded_length=None, max_seqlen=None):
    return _make_packed_features(
        x.new_empty((0, feature_dim)),
        lengths,
        padding_value=padding_value,
        padded_length=padded_length,
        max_seqlen=max_seqlen,
    )
