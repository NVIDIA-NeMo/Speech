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

import functools
import math
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import librosa
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F


@contextmanager
def disable_tf32():
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = prev


# ==============================================================================
# Utility Functions
# ==============================================================================


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zeros out the parameters of a PyTorch module in-place.

    This is a utility function that iterates through all parameters of a given
    `nn.Module` and sets their values to zero. This is often used for specific
    initialization strategies, for example in diffusion models where some layers
    are initialized to zero.

    From: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L68

    Args:
        module (nn.Module): The PyTorch module to be zeroed.

    Returns:
        nn.Module: The same module with its parameters zeroed.
    """
    for p in module.parameters():
        # p.detach().zero_() performs the operation in-place without tracking it in autograd
        p.detach().zero_()
    return module


def sequence_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    """
    Creates a boolean mask from a 1D tensor of sequence lengths.

    This function is useful for masking out padding in sequences. Given a tensor
    of lengths, it produces a 2D boolean tensor where `mask[i, j]` is `True` if
    `j < lengths[i]` and `False` otherwise.

    Example:
        >>> lengths = torch.tensor([1, 3, 2])
        >>> sequence_mask(lengths)
        tensor([[ True, False, False],
                [ True,  True,  True],
                [ True,  True, False]])

    Args:
        lengths (Tensor): A 1D tensor of integer lengths. Shape: `[batch_size]`.
        max_length (int | None, optional): The maximum length of the mask. If None,
                                           it is inferred from the maximum value
                                           in `lengths`. Defaults to None.

    Returns:
        Tensor: The boolean mask. Shape: `[batch_size, max_length]`.
    """
    if max_length is None:
        # If max_length is not provided, use the longest sequence length in the batch.
        max_length = int(lengths.max().item())

    # Create a range tensor from 0 to max_length - 1
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)

    # Compare each length with the range tensor to create the mask.
    # `x.unsqueeze(0)` is `[1, max_length]`
    # `lengths.unsqueeze(1)` is `[batch_size, 1]`
    # Broadcasting takes care of the comparison.
    return x.unsqueeze(0) < lengths.unsqueeze(1)


# ==============================================================================
# Signal Processing Functions
# ==============================================================================


def spectrogram(
    wav: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_fn: Callable = torch.hann_window,
) -> Tensor:
    """
    Computes the Short-Time Fourier Transform (STFT) of a waveform with manual padding.

    This implementation manually applies zero padding before computing the STFT.
    This is done to center the analysis window at the beginning of the signal
    without using the `center=True` argument in `torch.stft`, giving more control.

    Args:
        wav (Tensor): The input audio waveform.
                      Shape: [batch_size?, time_steps], where batch_size? is an
                      optional batch dimension.
        n_fft (int): The size of the FFT.
        hop_length (int): The number of samples between adjacent STFT columns.
        win_length (int): The size of the window function.
        window_fn (function, optional): The window function to apply.
                                        Defaults to torch.hann_window.

    Returns:
        Tensor: The complex-valued spectrogram.
                Shape: [batch_size?, n_fft // 2 + 1, num_frames]
    """
    # Calculate the padding required on the left and right sides to center the frames.
    pad_size_l = (n_fft - hop_length) // 2
    pad_size_r = (n_fft - hop_length) - pad_size_l

    # Use a torch.autocast context to perform STFT in float32 for precision.
    with torch.autocast(device_type=wav.device.type, enabled=False):
        # Apply reflection padding to the waveform.
        wav = F.pad(wav.float(), (pad_size_l, pad_size_r))

        # Create the window tensor on the same device as the waveform.
        window = window_fn(win_length, dtype=torch.float, device=wav.device)

        # Compute the STFT.
        # `center=False` because we have already manually padded the signal.
        spec = torch.stft(
            wav,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    return spec


def spec_to_wav(
    spec: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_fn: Callable = torch.hann_window,
    constrain_value_range: bool = False,
) -> Tensor:
    """
    Converts a spectrogram back into a waveform using the overlap-add method.
    This function is an approximate inverse of the `spectrogram` function.

    Args:
        spec (Tensor): The input complex-valued spectrogram.
                       Shape: [batch_size?, dim, time_steps], where batch_size?
                       is an optional batch dimension.
        n_fft (int): The size of the FFT used to create the spectrogram.
        hop_length (int): The number of samples between frames in the original signal.
        win_length (int): The size of the window function used in the original signal.
        window_fn (function, optional): The window function used. Currently only
                                        `torch.hann_window` is supported.
        constrain_value_range (bool, optional): If True, constrains the IFFT values
                                                to be within the range of the window.
                                                This ensures that the output values
                                                remain within the range of -1.0 to 1.0.
                                                Defaults to False.

    Returns:
        Tensor: The reconstructed waveform.
                Shape: [batch_size?, time_steps]

    Raises:
        ValueError: If a window function other than `torch.hann_window` is provided.
    """
    with torch.autocast(device_type=spec.device.type, enabled=False):
        if window_fn != torch.hann_window:
            raise ValueError(f"`window_fn` should be 'torch.hann_window', but got '{window_fn}'.")

        # Calculate padding and number of frames
        pad = (win_length - hop_length) // 2
        T = spec.size(-1)
        window = window_fn(win_length, device=spec.device)

        # 1. Inverse FFT
        # Convert from frequency domain back to time domain for each frame.
        ifft = torch.fft.irfft(spec, n=n_fft, dim=-2, norm="backward")
        window_unsqz = window.unsqueeze(-1)

        # 2. Optionally constrain values
        if constrain_value_range:
            ifft = torch.where(
                ifft >= 0,
                torch.minimum(ifft, window_unsqz),
                torch.maximum(ifft, -window_unsqz),
            )

        # 3. Apply window to the IFFT result
        ifft = ifft * window_unsqz

        # 4. Overlap and Add
        # Use `torch.nn.functional.fold` to perform the overlap-add operation efficiently.
        # This reconstructs the continuous signal from the windowed frames.
        output_size = (T - 1) * hop_length + win_length
        wav = F.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, win_length),
            stride=(1, hop_length),
        )[..., 0, 0, pad:-pad]

        # 5. Calculate the window envelope for normalization
        # This is necessary to correct for the energy added by overlapping windows.
        window_sq = window.square().expand(T, -1).transpose(0, 1)
        window_envelope = F.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, win_length),
            stride=(1, hop_length),
        ).squeeze()[pad:-pad]

        # 6. Normalize the waveform
        # Divide by the window envelope to get the final reconstructed signal.
        assert (window_envelope > 1e-11).all(), "Window envelope has zero values, cannot normalize."
        wav = wav / window_envelope

        return wav


def spectrogram_mag(
    wav: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_fn: Callable = torch.hann_window,
    power: float = 1.0,
) -> Tensor:
    """
    Computes the magnitude spectrogram from an audio waveform.

    This function first calculates the complex-valued spectrogram using the
    Short-Time Fourier Transform (STFT), then computes the magnitude of the
    resulting complex numbers. An optional power can be applied to the
    magnitude spectrogram.

    Args:
        wav (Tensor): The input audio waveform.
                      Shape: [batch_size?, time_steps], where batch_size? is
                      an optional batch dimension.
        n_fft (int): The size of the Fast Fourier Transform (FFT) to use.
        hop_length (int): The number of audio samples between adjacent STFT columns.
        win_length (int): The size of the window function for each frame.
        window_fn (function, optional): The windowing function to apply to each
                                        frame. Defaults to torch.hann_window.
        power (float, optional): The exponent to apply to the magnitude spectrogram.
                                 A value of 2.0 yields a power spectrogram.
                                 Defaults to 1.0 (magnitude).

    Returns:
        Tensor: The resulting magnitude spectrogram.
                Shape: [batch_size?, n_fft // 2 + 1, num_frames]
    """
    # Calculate the complex spectrogram
    spec = spectrogram(wav, n_fft, hop_length, win_length, window_fn)

    # Compute the magnitude by taking the absolute value
    spec = spec.abs()

    # Apply power if it's not the default value of 1.0
    if power != 1.0:
        spec = spec.pow(power)

    return spec


@functools.cache
def get_fbanks(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    norm: str = "slaney",
    mel_scale: str = "slaney",
) -> Tensor:
    """
    Creates and caches Mel filterbanks.

    This function generates a set of triangular filters on the Mel scale.
    The `@functools.cache` decorator memoizes the result, so the filterbanks
    are only computed once for a given set of parameters, improving efficiency
    when the function is called multiple times with the same arguments.

    Note: This implementation only supports Mel filterbanks via librosa.

    Args:
        sample_rate (int): The sample rate of the audio.
        n_fft (int): The size of the FFT used to compute the spectrogram.
        n_mels (int): The number of Mel bands to generate.
        f_min (float): The lowest frequency (in Hz) for the filterbanks.
        f_max (float): The highest frequency (in Hz) for the filterbanks.
        norm (str, optional): The normalization method to use for the triangles.
                              'slaney' normalizes to unit area. None applies no norm.
                              Defaults to "slaney".
        mel_scale (str, optional): The Mel scale to use, "htk" or "slaney".
                                   Defaults to "slaney".

    Returns:
        Tensor: The Mel filterbank matrix.
                Shape: [n_mels, n_fft // 2 + 1]
    """
    # Generate Mel filterbanks using librosa's functional API
    fb = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        norm=norm,
        htk=(mel_scale == "htk"),
    )  # [n_mels, n_freqs]
    fb = torch.from_numpy(fb).float()
    return fb


def mel_spectrogram(
    wav: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    sample_rate: int,
    n_mels: int,
    f_min: float,
    f_max: float | None = None,
    window_fn: Callable = torch.hann_window,
    power: float = 1.0,
    log_scale: str | None = "natural",
) -> Tensor:
    """
    Computes a Mel-scaled spectrogram from an audio waveform.

    This function transforms a standard spectrogram into a Mel spectrogram by
    applying Mel-scaled filterbanks. It can optionally return the result on a
    logarithmic scale.

    Args:
        wav (Tensor): The input audio waveform.
                      Shape: [batch_size?, time_steps], where batch_size? is an
                      optional batch dimension.
        n_fft (int): The size of the FFT.
        hop_length (int): The number of samples between adjacent frames.
        win_length (int): The size of the window function.
        sample_rate (int): The sample rate of the audio.
        n_mels (int): The number of Mel bands to generate.
        f_min (float): The lowest frequency (in Hz) for the Mel scale.
        f_max (float | None, optional): The highest frequency (in Hz). If None,
                                        it defaults to sample_rate / 2 (Nyquist).
        window_fn (function, optional): The windowing function. Defaults to torch.hann_window.
        power (float, optional): The exponent for the magnitude spectrogram before
                                 Mel conversion. Defaults to 1.0.
        log_scale (str | None, optional): The type of logarithmic scaling to apply.
                                          Can be "natural" (for `log`), "log10", or `None`
                                          to return the linear-amplitude Mel spectrogram.
                                          Defaults to "natural".

    Returns:
        Tensor: The resulting Mel spectrogram.
                Shape: [batch_size?, n_mels, num_frames]

    Raises:
        ValueError: If an unsupported string is provided for `log_scale`.
    """
    # If f_max is not provided, use the Nyquist frequency.
    f_max = f_max or sample_rate / 2

    # 1. Compute the magnitude spectrogram.
    spec = spectrogram_mag(wav, n_fft, hop_length, win_length, window_fn=window_fn, power=power)

    # Use a torch.autocast context to ensure the following operations
    # are performed in float32 precision for numerical stability, especially
    # when the input `spec` might be in a lower precision format like float16.
    with torch.autocast(device_type=spec.device.type, enabled=False):
        # 2. Get the Mel filterbanks (cached for efficiency).
        fb = (
            get_fbanks(
                sample_rate,
                n_fft,
                n_mels,
                f_min,
                f_max,
            )
            .float()
            .to(device=spec.device)
        )  # Ensure filterbank is float32 and on the correct device.

        # 3. Apply the filterbanks to the spectrogram via matrix multiplication.
        # This maps the linear frequency scale to the Mel scale.
        # (n_mels, n_freqs) @ (..., n_freqs, time) -> (..., n_mels, time)
        mel = torch.matmul(fb, spec.float())

        # 4. Optionally, apply a logarithmic function.
        # A small value (epsilon) is added to prevent taking the log of zero.
        if log_scale == "natural":
            mel = torch.log(torch.clamp(mel, min=1e-6))
        elif log_scale == "log10":
            mel = torch.log10(torch.clamp(mel, min=1e-6))
        elif log_scale is not None:
            raise ValueError(f"Unsupported log_scale: '{log_scale}'. Choose from 'natural', 'log10', or None.")

    return mel


# ==============================================================================
# Basic Modules
# ==============================================================================


class CausalConv1dCache:
    """
    A cache for managing states in causal 1D convolutions.

    This class is used during autoregressive inference to store and update the
    tail of the input to a causal convolution, which is used as padding for the
    next time step. This avoids re-computing the entire sequence at each step.
    """

    def __init__(self) -> None:
        self.cache: dict[int | str, Tensor] = {}

    def __getitem__(self, layer_id: int | str) -> Tensor:
        """Retrieves the cached tensor for a given layer."""
        return self.cache[layer_id]

    def update(
        self,
        states: Tensor,
        layer_id: int | str,
        padding: int,
        padding_value: int = 0,
        flush: bool = False,
    ) -> Tensor:
        """
        Updates the cache for a specific layer and returns the padded input.

        Args:
            states (Tensor): The new input tensor for the current time step.
            layer_id (int | str): An identifier for the convolutional layer.
            padding (int): The amount of left padding required by the convolution.
            padding_value (int, optional): The value to use for initial padding. Defaults to 0.
            flush (bool, optional): If True, the cache for this layer is deleted
                                    after use. Defaults to False.

        Returns:
            Tensor: The input states concatenated with the cached padding.
        """
        device = states.device
        dtype = states.dtype
        b, c, t = states.size()

        if layer_id not in self.cache:
            # Initialize cache with zero padding if it's the first time step
            padding_tensor = torch.zeros((b, c, padding), dtype=dtype, device=device) + padding_value
        else:
            padding_tensor = self.cache[layer_id]
            assert padding_tensor.size(2) == padding

        # Concatenate the cached padding with the new states
        padded_states = torch.cat([padding_tensor, states], dim=2)
        # Update the cache with the tail of the new padded states
        self.cache[layer_id] = padded_states[:, :, -padding:]

        if flush:
            del self.cache[layer_id]

        return padded_states


class LayerNormNd(nn.Module):
    """
    A LayerNorm module that works for N-dimensional inputs.

    This implementation normalizes over the channel dimension (dim=1), which is
    a common setup for convolutional networks.

    Args:
        channels (int): The number of channels of the input tensor.
        eps (float, optional): A value added to the denominator for numerical
                               stability. Defaults to 1e-6.
        elementwise_affine (bool, optional): If True, this module has learnable
                                             affine parameters (weight and bias).
                                             Defaults to True.
        bias (bool, optional): If True, this module has a learnable bias.
                               Defaults to True.
    """

    def __init__(self, channels: int, eps=1e-6, elementwise_affine: bool = True, bias: bool = True):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones((channels,)), requires_grad=elementwise_affine)
        self.bias = nn.Parameter(torch.zeros((channels,)), requires_grad=elementwise_affine and bias)

    def forward(self, x: Tensor) -> Tensor:
        # Calculate mean and reciprocal standard deviation over the channel dimension
        mean = x.mean(1, keepdim=True)
        x_shift = x - mean
        # Using rsqrt for potentially better performance
        x_rstd = torch.rsqrt(x_shift.pow(2).mean(1, keepdim=True) + self.eps)

        # Reshape weight and bias to be broadcastable with the input tensor
        shape = [-1 if i == 1 else 1 for i in range(x.ndim)]

        # Apply normalization and affine transformation
        return (x_shift * x_rstd) * self.weight.view(shape) + self.bias.view(shape)


class ConvNeXt1d(nn.Module):
    """
    A 1D ConvNeXt block adapted for causal convolutions on audio signals.

    This block is a core component of modern convolutional architectures, featuring
    a depthwise convolution, layer normalization, and pointwise convolutions to
    expand and contract the channel dimension, similar to an inverted bottleneck.

    Implementation adapted from: https://github.com/charactr-platform/vocos

    Args:
        dim (int): Number of input and output channels.
        intermediate_dim (int): Dimensionality of the intermediate (expanded) layer.
        kernel_size (int): The kernel size for the causal depthwise convolution.
        identity_init (bool, optional): If True, the final pointwise convolution
                                        is initialized to zero, making the block
                                        an identity function at the start of training.
                                        Defaults to False.
        layer_idx (int, optional): An index for this layer, used for caching. Defaults to 0.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel_size: int,
        identity_init: bool = False,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.kernel_size = kernel_size

        # Depthwise convolution
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim)

        self.norm = LayerNormNd(dim)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, 1)  # Pointwise/1x1 conv for expansion
        self.act = nn.GELU()

        # Pointwise/1x1 conv for projection
        if identity_init:
            self.pwconv2 = zero_module(nn.Conv1d(intermediate_dim, dim, 1))
        else:
            self.pwconv2 = nn.Conv1d(intermediate_dim, dim, 1)

    def forward(self, x: Tensor, cache: CausalConv1dCache | None = None, flush: bool = False) -> Tensor:
        residual = x

        # Apply causal padding, either through a cache or manually
        if cache is not None:
            x = cache.update(x, self.layer_idx, self.kernel_size - 1, flush=flush)
        else:
            x = F.pad(x, [self.kernel_size - 1, 0])  # Left padding for causality

        # Main ConvNeXt path
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Add residual connection
        x = residual + x
        return x


class PreTrainedEMAVariance(nn.Module):
    """
    Exponential Moving Average of Variance
    """

    def __init__(self, initial_value: float = 1.0):
        super().__init__()
        self.variance = nn.Parameter(
            torch.tensor(initial_value),
            requires_grad=False,
        )

    def forward(self) -> Tensor:
        return self.variance


class PreTrainedProbabilisticVQ(nn.Module):
    def __init__(
        self,
        channels: int,
        num_mixtures: int,
        depth: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.num_mixtures = num_mixtures
        self.depth = depth

        self.mus_list = nn.ParameterList(
            [
                nn.Parameter(
                    F.normalize(torch.randn(num_mixtures, channels), p=2.0, dim=1) * ((depth - i) / depth),
                    requires_grad=False,
                )
                for i in range(depth)
            ]
        )
        self._variance_list = nn.ModuleList([PreTrainedEMAVariance() for _ in range(depth)])

    @property
    def log_std(self) -> Tensor:
        return torch.log(self._variance_list[-1]()) * 0.5

    def encode(self, z: Tensor, return_z_q: bool = False) -> list[Tensor] | tuple[list[Tensor], Tensor]:
        r = z
        ids_sel = []
        for i in range(self.depth):
            mus = self.mus_list[i]
            idx_sel = self._dist_sq(r, mus).argmin(-1)  # [b, ?, h], [v, h] -> [b, ?]
            r = r - F.embedding(idx_sel, mus)
            ids_sel.append(idx_sel)
        if return_z_q:
            return ids_sel, z - r
        return ids_sel

    def decode(self, ids_sel: list[Tensor]) -> Tensor:
        z = torch.zeros((*ids_sel[0].size(), self.channels), device=ids_sel[0].device)
        for i in range(len(ids_sel)):
            mus = self.mus_list[i]
            z = z + F.embedding(ids_sel[i], mus)
        return z  # [b, ?, h]

    def _dist_sq(self, z: Tensor, mus: Tensor) -> Tensor:
        """
        z: [b, ?, d?, h]
        mus: [d?, v, h]
        """
        return (
            z.pow(2).sum(-1, keepdim=True)  # [b, ?, d?, 1]
            + mus.pow(2).sum(-1)  # [d?, v]
            - 2 * (z.unsqueeze(-2) @ mus.transpose(-1, -2)).squeeze(-2)  # [b, ?, d?, h] , [d?, h, v] -> [b, ?, d?, v]
        )


class Wav2Latent(nn.Module):
    """
    An encoder model that transforms a raw waveform into a latent representation.

    This model first converts the waveform to a spectrogram, then processes it
    through a series of ConvNeXt blocks and downsampling convolutional layers
    to produce a compressed latent tensor.

    Args:
        latent_size (int): The number of channels in the final latent representation.
        n_fft (int): The FFT size for the initial spectrogram transformation.
        hop_length (int): The hop length for the STFT.
        base_hidden_size (int): The base number of channels for the hidden layers.
        channel_mult (tuple[int, ...]): A tuple of multipliers for the hidden
                                        size at each stage of downsampling.
        rates (tuple[int, ...]): A tuple of downsampling factors (strides) for
                                 the convolutional layers.
        num_blocks (int): The number of ConvNeXt blocks per stage.
        kernel_size (int): The kernel size for the ConvNeXt blocks.
        groups (int): The number of groups for the downsampling convolutions.
    """

    def __init__(
        self,
        latent_size: int = 1024,
        n_fft: int = 32,
        hop_length: int = 8,
        base_hidden_size: int = 384,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        rates: tuple[int, ...] = (8, 8, 8),
        num_blocks: int = 3,
        kernel_size: int = 7,
        groups: int = 1,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Initial projection from spectrogram to hidden size
        layers: list[nn.Module] = [nn.Conv1d(n_fft + 2, base_hidden_size * channel_mult[0], 1, bias=False)]

        # Downsampling stages
        for i in range(len(channel_mult)):
            ch_mult, rate = channel_mult[i], rates[i]
            hidden_size = base_hidden_size * ch_mult
            # Add ConvNeXt blocks for this stage
            for j in range(num_blocks):
                layers.append(
                    ConvNeXt1d(hidden_size, hidden_size * 4, kernel_size, True, layer_idx=i * num_blocks + j)
                )
            # Add downsampling convolution
            next_hidden_size = base_hidden_size * channel_mult[i + 1] if i < len(channel_mult) - 1 else latent_size
            layers.append(
                nn.Conv1d(hidden_size, next_hidden_size, kernel_size=rate, stride=rate, bias=False, groups=groups)
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, cache=None, flush: bool = False) -> Tensor:
        if cache is not None:
            raise NotImplementedError("Caching is not implemented for the encoder.")

        # Convert waveform to spectrogram (magnitude and phase)
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = spectrogram(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft)
            # Split complex spectrogram into real and imaginary, then treat as magnitude and phase
            mag, ph = torch.view_as_real(spec).chunk(2, dim=-1)
            x = torch.cat([mag, ph], 1).squeeze(-1)

        # Pass through the network
        for layer in self.layers:
            if isinstance(layer, ConvNeXt1d):
                x = layer(x, cache=cache, flush=flush)
            else:
                x = layer(x)

        # Transpose to [batch, time, channels] for compatibility with transformers
        x = x.transpose(-1, -2)
        return x


class Latent2Wav(nn.Module):
    """
    A decoder (vocoder) model that transforms a latent representation back into a raw waveform.

    This model processes a latent tensor through a series of ConvNeXt blocks and
    upsampling transposed convolutional layers to produce a spectrogram, which is
    then converted back to a waveform using an inverse STFT.

    Args:
        latent_size (int): The number of channels in the input latent representation.
        n_fft (int): The FFT size for the final spectrogram reconstruction.
        hop_length (int): The hop length for the ISTFT.
        base_hidden_size (int): The base number of channels for the hidden layers.
        channel_mult (tuple[int, ...]): A tuple of multipliers for the hidden
                                        size at each stage of upsampling.
        rates (tuple[int, ...]): A tuple of upsampling factors (strides) for
                                 the transposed convolutional layers.
        num_blocks (int): The number of ConvNeXt blocks per stage.
        kernel_size (int): The kernel size for the ConvNeXt blocks.
        groups (int): The number of groups for the upsampling convolutions.
    """

    def __init__(
        self,
        latent_size: int = 1024,
        n_fft: int = 32,
        hop_length: int = 8,
        base_hidden_size: int = 384,
        channel_mult: tuple[int, ...] = (4, 2, 1),
        rates: tuple[int, ...] = (8, 8, 8),
        num_blocks: int = 3,
        kernel_size: int = 7,
        groups=1,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spec_cache_idx = (len(channel_mult)) * num_blocks

        layers: list[nn.Module] = []

        # Upsampling stages
        for i in range(len(channel_mult)):
            ch_mult, rate = channel_mult[i], rates[i]
            hidden_size = base_hidden_size * ch_mult
            # Add upsampling transposed convolution
            in_size = base_hidden_size * channel_mult[i - 1] if i != 0 else latent_size
            layers.append(
                nn.ConvTranspose1d(in_size, hidden_size, kernel_size=rate, stride=rate, bias=False, groups=groups)
            )
            # Add ConvNeXt blocks for this stage
            for j in range(num_blocks):
                layers.append(
                    ConvNeXt1d(hidden_size, hidden_size * 4, kernel_size, True, layer_idx=i * num_blocks + j)
                )

        # Final projection to spectrogram dimensions (magnitude + phase)
        layers.append(nn.Conv1d(hidden_size, n_fft + 2, 1, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, cache=None, flush: bool = False, constrain_value_range: bool = True) -> Tensor:
        # Transpose input from [batch, time, channels] to [batch, channels, time]
        x = x.transpose(-1, -2)

        # Pass through the network
        for layer in self.layers:
            if isinstance(layer, ConvNeXt1d):
                x = layer(x, cache=cache, flush=flush)
            else:
                x = layer(x)

        # Convert network output to a complex spectrogram and then to a waveform
        with torch.autocast(device_type=x.device.type, enabled=False):
            max_mag = 100.0
            # Split output into magnitude and phase components
            mag, ph = x.float().chunk(2, dim=1)
            # Safeguard to prevent excessively large magnitudes
            mag = max_mag * torch.exp(-F.softplus(-mag + math.log(max_mag)))

            # Reconstruct the complex spectrogram from magnitude and phase
            # The DC and Nyquist components are real, so their phase is applied via cosine.
            mag_dc, mag_mid, mag_nyquist = mag.split([1, mag.size(1) - 2, 1], dim=1)
            ph_dc, ph_mid, ph_nyquist = torch.cos(ph).split([1, ph.size(1) - 2, 1], dim=1)
            ph_imag = torch.sin(ph[:, 1:-1, :])

            spec_real = mag_mid * ph_mid
            spec_imag = mag_mid * ph_imag

            spec = torch.cat([mag_dc * ph_dc, spec_real + 1j * spec_imag, mag_nyquist * ph_nyquist], 1)

            # Handle caching for autoregressive generation of the spectrogram
            if cache is not None:
                half_spec_padding = math.ceil(((self.n_fft - self.hop_length) // 2) / self.hop_length)
                spec = cache.update(spec, self.spec_cache_idx, padding=half_spec_padding * 2, flush=flush)
                if flush:
                    spec = F.pad(spec, [0, half_spec_padding])

            # Convert spectrogram to waveform
            x = spec_to_wav(
                spec, self.n_fft, self.hop_length, self.n_fft, constrain_value_range=constrain_value_range
            ).unsqueeze(1)

        if cache is not None:
            # Trim the output to remove the padded region from the start
            half_wav_padding = half_spec_padding * self.hop_length
            x = x[:, :, half_wav_padding:-half_wav_padding]

        return x


class RVQVAEModel(nn.Module):
    """
    Residual Vector-Quantized Variational Autoencoder (RVQ-VAE) model.

    This model learns a discrete representation of audio by encoding a waveform
    into a latent space and then quantizing the latents into discrete codes.
    It consists of an encoder, a quantizer, and a decoder.

    Args:
        config (DictConfig | dict[str, Any]): A configuration object with model hyperparameters.
    """

    config_class: type[DictConfig] = DictConfig

    def __init__(self, config: DictConfig | dict[str, Any]):
        super().__init__()
        self.config = config

        self.encoder = Wav2Latent(
            latent_size=self.config.latent_size,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            base_hidden_size=self.config.base_hidden_size,
            channel_mult=self.config.channel_mult,
            rates=self.config.rates,
            num_blocks=self.config.num_blocks,
            kernel_size=self.config.kernel_size,
            groups=self.config.groups,
        )

        # Layers for quantization
        self.prvq = PreTrainedProbabilisticVQ(
            channels=self.config.latent_size,
            num_mixtures=self.config.codebook_size,
            depth=self.config.num_quantizers,
        )

        self.decoder = Latent2Wav(
            latent_size=self.config.latent_size,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            base_hidden_size=self.config.base_hidden_size,
            channel_mult=tuple(reversed(self.config.channel_mult)),
            rates=tuple(reversed(self.config.rates)),
            num_blocks=self.config.num_blocks,
            kernel_size=self.config.kernel_size,
            groups=self.config.groups,
        )

        for p in self.parameters():
            p.requires_grad = False

    def ae_encode(self, x: Tensor, cache: CausalConv1dCache | None = None, flush: bool = False) -> Tensor:
        """
        Runs the encoder part of the autoencoder.

        Args:
            x (Tensor): Input waveform. Shape: `[batch, 1, time]`.
            cache (CausalConv1dCache | None): Not implemented for the encoder.
            flush (bool): Not implemented for the encoder.

        Returns:
            Tensor: The continuous latent representation. Shape: `[batch, time', channels]`.
        """
        assert x.size(1) == 1 and x.dim() == 3, "Input must be a batch of mono audio."
        assert x.size(2) % self.config.wav_to_token_ratio == 0, (
            f"Input audio length ({x.size(2)}) must be divisible by the model's "
            f"wav_to_token_ratio ({self.config.wav_to_token_ratio}). "
            f"Please pad the input to a compatible length."
        )

        if cache is not None:
            raise NotImplementedError("Caching is not supported for the encoder.")

        return self.encoder(x, cache=cache, flush=flush)

    def ae_decode(
        self,
        x: Tensor,
        constrain_value_range: bool = True,
        cache: CausalConv1dCache | None = None,
        flush: bool = False,
    ) -> Tensor:
        """
        Runs the decoder part of the autoencoder.

        Args:
            x (Tensor): The (de-quantized) latent representation. Shape: `[batch, time', channels]`.
            constrain_value_range (bool): If True, constrains the output of the ISTFT.
            cache (CausalConv1dCache | None): Cache for autoregressive generation.
            flush (bool): If True, flushes the cache.

        Returns:
            Tensor: The reconstructed waveform. Shape: `[batch, 1, time]`.
        """
        return self.decoder(x, constrain_value_range=constrain_value_range, cache=cache, flush=flush)

    def encode(self, x: Tensor, x_len: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encodes a waveform into discrete codes.

        Args:
            x (Tensor): Input waveform. Shape: `[batch, 1, time]`.
            x_len (Tensor): The original lengths of the waveforms in the batch.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - The discrete codes. Shape: `[batch, time', n_quantizers]`.
                - The lengths of the code sequences.
        """
        with disable_tf32():
            z_e = self.ae_encode(x)
            code_len = x_len // self.config.wav_to_token_ratio
            return self.quantize(z_e), code_len

    def decode(
        self,
        code: Tensor,
        code_len: Tensor | None = None,
        constrain_value_range: bool = True,
        cache: CausalConv1dCache | None = None,
        flush: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Decodes discrete codes back into a waveform.

        Args:
            code (Tensor): The discrete codes. Shape: `[batch, time', n_quantizers]`.
            code_len (Tensor | None): The lengths of the code sequences.
            constrain_value_range (bool): If True, constrains the output of the ISTFT.
            cache (CausalConv1dCache | None): Cache for autoregressive generation.
            flush (bool): If True, flushes the cache.

        Returns:
            tuple[Tensor, Tensor | None]: A tuple containing:
                - The reconstructed waveform. Shape: `[batch, 1, time]`.
                - The lengths of the reconstructed waveforms.
        """
        with disable_tf32():
            z_q = self.dequantize(code)
            x_hat = self.ae_decode(z_q, constrain_value_range=constrain_value_range, cache=cache, flush=flush)
            wav_len = code_len * self.config.wav_to_token_ratio if code_len is not None else None
            return x_hat, wav_len

    def quantize(self, z: Tensor) -> Tensor:
        """
        Quantizes a continuous latent tensor into discrete codes.

        Args:
            z (Tensor): The continuous latent tensor from the encoder.
                        Shape: `[batch, time, channels]`.

        Returns:
            Tensor: The quantized codes. Shape: `[batch, time, n_quantizers]`.
        """
        with disable_tf32():
            ids_sel = self.prvq.encode(z, return_z_q=False)
            return torch.stack(ids_sel, -1)

    def dequantize(self, code: Tensor) -> Tensor:
        """
        De-quantizes discrete codes back into a continuous latent tensor.

        Args:
            code (Tensor): The quantized codes. Shape: `[batch, time, n_quantizers]`.

        Returns:
            Tensor: The de-quantized continuous latent tensor.
                    Shape: `[batch, time, latent_size]`.
        """
        ids_sel = [x.squeeze(-1) for x in torch.split(code, 1, -1)]
        return self.prvq.decode(ids_sel)

    def forward(self, x: Tensor, constrain_value_range: bool = False) -> Tensor:
        """
        Performs a full autoencoding pass: encode, quantize, dequantize, and decode.

        Args:
            x (Tensor): The input waveform. Shape: `[batch, 1, time]`.
            constrain_value_range (bool): If True, constrains the output of the ISTFT.

        Returns:
            Tensor: The reconstructed waveform. Shape: `[batch, 1, time]`.
        """

        with torch.no_grad():
            z_e = self.ae_encode(x)
            code = self.quantize(z_e)
            z_d = self.dequantize(code)
            x_hat = self.ae_decode(z_d, constrain_value_range=constrain_value_range)
        return x_hat
