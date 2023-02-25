# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional

import numpy as np
import torch

from nemo.collections.asr.parts.preprocessing.features import make_seq_mask_like
from nemo.collections.asr.parts.utils.audio_utils import toeplitz
from nemo.core.classes import Loss, Typing, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, LossType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['SDRLoss']


def temporal_mean(
    input: torch.Tensor,
    input_length: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    keepdim: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Calculate mean along temporal dimension with optionally
    averaging only over valid samples (based on the input length).

    Args:
        input: Batch of signals, shape (B, C, T)
        input_length: Optional, length of each example in the batch, shape (B,)
        mask: Optional, temporal mask for each example in the batch, shape (B, T)
        keepdim: Whether to keep the temporal dimension
        eps: Regularization to avoid division by zero

    Returns:
        (B, C, 1) if keepdim=True, otherwise (B, C)
    """
    if input_length is not None:
        if mask is not None:
            raise RuntimeError(
                'Argument `input_length` is mutually exclusive with `mask`. Both cannot be used at the same time.'
            )
        # Construct a binary mask
        mask = make_seq_mask_like(lengths=input_length, like=input, time_dim=-1, valid_ones=True).squeeze(1)

    if mask is None:
        # No length information, assume all samples are valid
        mean = torch.mean(input, dim=-1, keepdim=keepdim)
    else:
        # Average using temporal mask
        mean = mask.unsqueeze(1) * input
        mean = torch.sum(mean, axis=-1, keepdim=keepdim)
        normalization = torch.sum(mask, axis=-1, keepdim=keepdim)
        mean = mean / (normalization.unsqueeze(1) + eps)

    return mean


def scale_invariant_target(
    estimate: torch.Tensor,
    target: torch.Tensor,
    input_length: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Calculate optimal scale-invariant target.
    Assumes time dimension is the last dimension in the array.

    Calculate scaled target obtained by solving

        min_scale || scale * target - estimate ||^2

    for each example in batch and each channel (b, c).

    Args:
        estimate: tensor, shape (B, C, T)
        target: tensor, shape (B, C, T)
        input_length: optional, length of valid samples, shape (B,)
        mask: optional, mask for input samples, shape (B, T)
        eps: regularization constant

    Returns:
        Scaled target, shape (B, C, T)
    """
    if input_length is not None:
        if mask is not None:
            raise RuntimeError(
                'Argument `input_length` is mutually exclusive with `mask`. Both cannot be used at the same time.'
            )

        # Construct a binary mask
        mask = make_seq_mask_like(lengths=input_length, like=estimate, time_dim=-1, valid_ones=True).squeeze(1)

    estimate_dot_target = temporal_mean(estimate * target, mask=mask, keepdim=True, eps=eps)
    target_pow = temporal_mean(torch.abs(target) ** 2, mask=mask, keepdim=True, eps=eps)
    scale = estimate_dot_target / (target_pow + eps)
    target_scaled = scale * target

    # Mask to keep only the valid samples
    if mask is not None:
        target_scaled = mask.unsqueeze(1) * target_scaled

    return target_scaled


def convolution_invariant_target(
    estimate: torch.Tensor,
    target: torch.Tensor,
    input_length: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    filter_length: int = 512,
    diag_reg: float = 1e-8,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Calculate optimal convolution-invariant target for a given estimate.
    Assumes time dimension is the last dimension in the array.

    Calculate target filtered with a linear f obtained by solving

        min_filter || conv(filter, target) - estimate ||^2

    for each example in batch and each channel (b, c).

    Args:
        estimate: tensor, shape (B, C, T)
        target: tensor, shape (B, C, T)
        input_length: optional, length of valid samples, shape (B,)
        mask: optional, mask for input samples, shape (B, T)
        filter_length: length of the (convolutional) filter for target
        diag_reg: relative diagonal regularization for the linear system
        eps: absolute regularization for the diagonal

    Returns:
        Filtered target, shape (B, C, T)

    Reference:
        C. Boeddeker et al., Convolutive Transfer Function Invariant SDR training criteria for Multi-Channel Reverberant Speech Separation, 2021
    """
    if input_length is not None:
        if mask is not None:
            raise RuntimeError(
                'Argument `input_length` is mutually exclusive with `mask`. Both cannot be used at the same time.'
            )

        if torch.min(input_length) < filter_length:
            logging.warning(
                'Current min input_length (%d) is smaller than filter_length (%d). This will result in a singular linear system.',
                torch.min(input_length),
                filter_length,
            )

        # Construct a binary mask
        mask = make_seq_mask_like(lengths=input_length, like=estimate, time_dim=-1, valid_ones=True).squeeze(1)

    # Apply a mask, if available
    if mask is not None:
        estimate = mask.unsqueeze(1) * estimate
        target = mask.unsqueeze(1) * target

    # Calculate filtered target
    input_shape = estimate.shape
    estimate = estimate.view(-1, input_shape[-1])
    target = target.view(-1, input_shape[-1])

    n_fft = 2 ** math.ceil(math.log2(2 * input_shape[-1] - 1))

    T = torch.fft.rfft(target, n=n_fft)
    E = torch.fft.rfft(estimate, n=n_fft)

    # Target autocorrelation
    tt_corr = torch.fft.irfft(torch.abs(T) ** 2, n=n_fft)
    # Target-estimate crosscorrelation
    te_corr = torch.fft.irfft(T.conj() * E, n=n_fft)

    # Use only filter_length
    tt_corr = tt_corr[..., :filter_length]
    te_corr = te_corr[..., :filter_length]

    # Diagonal regularization
    if diag_reg is not None:
        tt_corr[..., 0] += diag_reg * tt_corr[..., 0] + eps

    # Construct the Toeplitz system matrix
    TT = toeplitz(tt_corr)

    # Solve the linear system for the optimal filter
    filt = torch.linalg.solve(TT, te_corr)

    # Calculate filtered target
    T_filt = T * torch.fft.rfft(filt, n=n_fft)
    target_filt = torch.fft.irfft(T_filt, n=n_fft)

    # Reshape to the original format
    target_filt = target_filt[..., : input_shape[-1]].view(*input_shape)

    # Mask to keep only the valid samples
    if mask is not None:
        target_filt = mask.unsqueeze(1) * target_filt

    return target_filt


def calculate_sdr_batch(
    estimate: torch.Tensor,
    target: torch.Tensor,
    input_length: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    scale_invariant: bool = False,
    convolution_invariant: bool = False,
    convolution_filter_length: Optional[int] = 512,
    remove_mean: bool = True,
    sdr_max: Optional[float] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Calculate signal-to-distortion ratio per channel.

        SDR = 10 * log10( ||t||_2^2 / (||e-t||_2^2 + alpha * ||t||^2)

    where
        alpha = 10^(-sdr_max/10)

    Optionally, use scale- or convolution- invariant target signal.

    Args:
        estimate: estimated signal, shape (B, C, T)
        target: target signal, shape (B, C, T)
        input_length: Optional, length of valid samples, shape (B,)
        mask: Optional, temporal mask, shape (B, T)
        scale_invariant: Use scale invariant SDR
        convolution_invariant: Use convolution invariant SDR
        convolution_filter_length: Filter length for convolution invariant SDR
        remove_mean: If True, mean will be removed before calculating SDR
        eps: Small regularization constant

    Returns:
        SDR in dB for each channel, shape (B, C)
    """
    if scale_invariant and convolution_invariant:
        raise ValueError(f'Arguments scale_invariant and convolution_invariant cannot be used simultaneously.')

    assert (
        estimate.shape == target.shape
    ), f'Estimate shape ({estimate.shape}) not matching target shape ({target.shape})'

    if input_length is not None:
        if mask is not None:
            raise RuntimeError(
                'Argument `input_length` is mutually exclusive with `mask`. Both cannot be used at the same time.'
            )

        # Construct a binary mask
        mask = make_seq_mask_like(lengths=input_length, like=estimate, time_dim=-1, valid_ones=True).squeeze(1)

    if remove_mean:
        estimate = estimate - temporal_mean(estimate, mask=mask, keepdim=True, eps=eps)
        target = target - temporal_mean(target, mask=mask, keepdim=True, eps=eps)

    if scale_invariant or (convolution_invariant and convolution_filter_length == 1):
        target = scale_invariant_target(estimate=estimate, target=target, mask=mask, eps=eps)
    elif convolution_invariant:
        target = convolution_invariant_target(
            estimate=estimate, target=target, mask=mask, filter_length=convolution_filter_length, eps=eps,
        )

    distortion = estimate - target

    target_pow = temporal_mean(torch.abs(target) ** 2, mask=mask, eps=eps)
    distortion_pow = temporal_mean(torch.abs(distortion) ** 2, mask=mask, eps=eps)

    if sdr_max is not None:
        distortion_pow = distortion_pow + 10 ** (-sdr_max / 10) * target_pow

    sdr = target_pow / (distortion_pow + eps)
    sdr = 10 * torch.log10(sdr + eps)

    return sdr


class SDRLoss(Loss, Typing):
    """
    Computes signal-to-distortion ratio (SDR) loss with weighted average across channels.

    Args:
        weight: weight for SDR of each output channel, used for averaging the loss across channels. Defaults to `None` (averaging).
        reduction: batch reduction. Defaults to `mean` over the batch.
        scale_invariant: If `True`, use scale-invariant SDR. Defaults to `False`.
        remove_mean: Remove mean before calculating the loss. Defaults to `True`.
        sdr_max: Soft thresholding of the loss to SDR_max.
        eps: Small value for regularization.
    """

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        reduction: str = 'mean',
        scale_invariant: bool = False,
        convolution_invariant: bool = False,
        convolution_filter_length: Optional[int] = 512,
        remove_mean: bool = True,
        sdr_max: Optional[float] = None,
        eps: float = 1e-10,
    ):
        super().__init__()

        # SDR weight buffer
        if weight is not None:
            if any([w <= 0 for w in weight]):
                raise ValueError(f'Weight must be positive! Current value: {weight}')
            elif not np.isclose(sum(weight), 1, atol=1e-6):
                raise ValueError(f'Weight should add to one, current weight: {weight}')
            weight = torch.tensor(weight).reshape(1, -1)
            logging.info(f'Channel weight set to %s', weight)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]

        # Batch reduction
        self.reduction = reduction
        if reduction == 'mean':
            self.reduce = torch.mean
        else:
            raise ValueError(f'Unexpected reduction mode {reduction}.')

        # SDR calculation setup
        if scale_invariant and convolution_invariant:
            raise ValueError(
                f'{self.__class__.__name__}: arguments scale_invariant and convolution_invariant cannot be used simultaneously.'
            )
        self.scale_invariant = scale_invariant
        self.convolution_invariant = convolution_invariant
        self.convolution_filter_length = convolution_filter_length
        self.remove_mean = remove_mean
        self.sdr_max = sdr_max
        self.eps = eps

    @property
    def input_types(self):
        """Input types definitions for SDRLoss.
        """
        signal_shape = ('B', 'C', 'T')
        return {
            "estimate": NeuralType(signal_shape, AudioSignal()),
            "target": NeuralType(signal_shape, AudioSignal()),
            "input_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "mask": NeuralType(('B', 'T'), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        """Output types definitions for SDRLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        input_length: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """For input batch of multi-channel signals, calculate SDR between estimate and target for each channel,
        perform averaging across channels (weighting optional), and apply reduction across the batch.

        Args:
            estimate: Batch of signals, shape (B, T, C)
            target: Batch of signals, shape (B, T, C)
            input_length: Batch of lengths, shape (B,)
            mask: Batch of temporal masks, shape (B, T)

        Returns:
            Scalar loss.
        """

        sdr = calculate_sdr_batch(
            estimate=estimate,
            target=target,
            input_length=input_length,
            mask=mask,
            scale_invariant=self.scale_invariant,
            convolution_invariant=self.convolution_invariant,
            convolution_filter_length=self.convolution_filter_length,
            remove_mean=self.remove_mean,
            sdr_max=self.sdr_max,
            eps=self.eps,
        )

        # channel averaging
        if self.weight is None:
            sdr = torch.mean(sdr, dim=1)
        else:
            # weighting across channels
            sdr = sdr * self.weight
            sdr = torch.sum(sdr, dim=1)

        # reduction
        sdr = self.reduce(sdr)

        return -sdr
