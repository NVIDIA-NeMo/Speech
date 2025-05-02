# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from nemo.collections.asr.parts.preprocessing.features import make_seq_mask_like
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention
from nemo.collections.audio.parts.utils.audio import covariance_matrix
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import AudioSignal, FloatType, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging


class ChannelAugment(NeuralModule):
    """Randomly permute and selects a subset of channels.

    Args:
        permute_channels (bool): Apply a random permutation of channels.
        num_channels_min (int): Minimum number of channels to select.
        num_channels_max (int): Max number of channels to select.
        rng: Optional, random generator.
        seed: Optional, seed for the generator.
    """

    def __init__(
        self,
        permute_channels: bool = True,
        num_channels_min: int = 1,
        num_channels_max: Optional[int] = None,
        rng: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self._rng = random.Random(seed) if rng is None else rng
        self.permute_channels = permute_channels
        self.num_channels_min = num_channels_min
        self.num_channels_max = num_channels_max

        if num_channels_max is not None and num_channels_min > num_channels_max:
            raise ValueError(
                f'Min number of channels {num_channels_min} cannot be greater than max number of channels {num_channels_max}'
            )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tpermute_channels: %s', self.permute_channels)
        logging.debug('\tnum_channels_min: %s', self.num_channels_min)
        logging.debug('\tnum_channels_max: %s', self.num_channels_max)

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            'input': NeuralType(('B', 'C', 'T'), AudioSignal()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {
            'output': NeuralType(('B', 'C', 'T'), AudioSignal()),
        }

    @typecheck()
    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Expecting (B, C, T)
        assert input.ndim == 3, 'Expecting input with shape (B, C, T)'
        num_channels_in = input.size(1)

        if num_channels_in < self.num_channels_min:
            raise RuntimeError(
                f'Number of input channels ({num_channels_in}) is smaller than the min number of output channels ({self.num_channels_min})'
            )

        num_channels_max = num_channels_in if self.num_channels_max is None else self.num_channels_max
        num_channels_out = self._rng.randint(self.num_channels_min, num_channels_max)

        channels = list(range(num_channels_in))

        if self.permute_channels:
            self._rng.shuffle(channels)

        channels = channels[:num_channels_out]

        return input[:, channels, :]


class TransformAverageConcatenate(NeuralModule):
    """Apply transform-average-concatenate across channels.
    We're using a version from [2].

    Args:
        in_features: Number of input features
        out_features: Number of output features

    References:
        [1] Luo et al, End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation, 2019
        [2] Yoshioka et al, VarArray: Array-Geometry-Agnostic Continuous Speech Separation, 2022
    """

    def __init__(self, in_features: int, out_features: Optional[int] = None):
        super().__init__()

        if out_features is None:
            out_features = in_features

        # Parametrize with the total number of features (needs to be divisible by two due to stacking)
        if out_features % 2 != 0:
            raise ValueError(f'Number of output features should be divisible by two, currently set to {out_features}')

        self.transform_channel = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features // 2, bias=False), torch.nn.ReLU()
        )
        self.transform_average = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features // 2, bias=False), torch.nn.ReLU()
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_features:  %d', in_features)
        logging.debug('\tout_features: %d', out_features)

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            'input': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {
            'output': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: shape (B, M, in_features, T)

        Returns:
            Output tensor with shape shape (B, M, out_features, T)
        """
        B, M, F, T = input.shape

        # (B, M, F, T) -> (B, T, M, F)
        input = input.permute(0, 3, 1, 2)

        # transform and average across channels
        average = self.transform_average(input)
        average = torch.mean(average, dim=-2, keepdim=True)
        # view with the number of channels expanded to M
        average = average.expand(-1, -1, M, -1)

        # transform each channel
        transform = self.transform_channel(input)

        # concatenate along feature dimension
        output = torch.cat([transform, average], dim=-1)

        # Return to the original layout
        # (B, T, M, F) -> (B, M, F, T)
        output = output.permute(0, 2, 3, 1)

        return output


class TransformAttendConcatenate(NeuralModule):
    """Apply transform-attend-concatenate across channels.
    The output is a concatenation of transformed channel and MHA
    over channels.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        n_head: Number of heads for the MHA module
        dropout_rate: Dropout rate for the MHA module

    References:
        - Jukić et al, Flexible multichannel speech enhancement for noise-robust frontend, 2023
    """

    def __init__(self, in_features: int, out_features: Optional[int] = None, n_head: int = 4, dropout_rate: float = 0):
        super().__init__()

        if out_features is None:
            out_features = in_features

        # Parametrize with the total number of features (needs to be divisible by two due to stacking)
        if out_features % 2 != 0:
            raise ValueError(f'Number of output features should be divisible by two, currently set to {out_features}')

        self.transform_channel = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features // 2, bias=False), torch.nn.ReLU()
        )
        self.transform_attend = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features // 2, bias=False), torch.nn.ReLU()
        )
        self.attention = MultiHeadAttention(n_head=n_head, n_feat=out_features // 2, dropout_rate=dropout_rate)

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_features:  %d', in_features)
        logging.debug('\tout_features: %d', out_features)
        logging.debug('\tn_head:       %d', n_head)
        logging.debug('\tdropout_rate: %f', dropout_rate)

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            'input': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {
            'output': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: shape (B, M, in_features, T)

        Returns:
            Output tensor with shape shape (B, M, out_features, T)
        """
        B, M, F, T = input.shape

        # (B, M, F, T) -> (B, T, M, F)
        input = input.permute(0, 3, 1, 2)
        input = input.reshape(B * T, M, F)

        # transform each channel
        transform = self.transform_channel(input)

        # attend
        attend = self.transform_attend(input)
        # attention across channels
        attend = self.attention(query=attend, key=attend, value=attend, mask=None)

        # concatenate along feature dimension
        output = torch.cat([transform, attend], dim=-1)

        # return to the original layout
        output = output.view(B, T, M, -1)

        # (B, T, M, num_features) -> (B, M, num_features, T)
        output = output.permute(0, 2, 3, 1)

        return output


class ChannelAveragePool(NeuralModule):
    """Apply average pooling across channels."""

    def __init__(self):
        super().__init__()
        logging.debug('Initialized %s', self.__class__.__name__)

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            'input': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {
            'output': NeuralType(('B', 'D', 'T'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: shape (B, M, F, T)

        Returns:
            Output tensor with shape shape (B, F, T)
        """
        return torch.mean(input, dim=-3)


class ChannelAttentionPool(NeuralModule):
    """Use attention pooling to aggregate information across channels.
    First apply MHA across channels and then apply averaging.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        n_head: Number of heads for the MHA module
        dropout_rate: Dropout rate for the MHA module

    References:
        - Wang et al, Neural speech separation using sparially distributed microphones, 2020
        - Jukić et al, Flexible multichannel speech enhancement for noise-robust frontend, 2023
    """

    def __init__(self, in_features: int, n_head: int = 1, dropout_rate: float = 0):
        super().__init__()
        self.in_features = in_features
        self.attention = MultiHeadAttention(n_head=n_head, n_feat=in_features, dropout_rate=dropout_rate)

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_features:  %d', in_features)
        logging.debug('\tnum_heads:    %d', n_head)
        logging.debug('\tdropout_rate: %d', dropout_rate)

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            'input': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {
            'output': NeuralType(('B', 'D', 'T'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: shape (B, M, F, T)

        Returns:
            Output tensor with shape shape (B, F, T)
        """
        B, M, F, T = input.shape

        # (B, M, F, T) -> (B, T, M, F)
        input = input.permute(0, 3, 1, 2)
        input = input.reshape(B * T, M, F)

        # attention across channels
        output = self.attention(query=input, key=input, value=input, mask=None)

        # return to the original layout
        output = output.view(B, T, M, -1)

        # (B, T, M, num_features) -> (B, M, out_features, T)
        output = output.permute(0, 2, 3, 1)

        # average across channels
        output = torch.mean(output, axis=-3)

        return output


class ParametricMultichannelWienerFilter(NeuralModule):
    """Parametric multichannel Wiener filter, with an adjustable
    tradeoff between noise reduction and speech distortion.
    It supports automatic reference channel selection based
    on the estimated output SNR.

    Args:
        beta: Parameter of the parameteric filter, tradeoff between noise reduction
              and speech distortion (0: MVDR, 1: MWF).
        rank: Rank assumption for the speech covariance matrix.
        postfilter: Optional postfilter. If None, no postfilter is applied.
        ref_channel: Optional, reference channel. If None, it will be estimated automatically.
        ref_hard: If true, estimate a hard (one-hot) reference. If false, a soft reference.
        ref_hard_use_grad: If true, use straight-through gradient when using the hard reference
        ref_subband_weighting: If true, use subband weighting when estimating reference channel
        num_subbands: Optional, used to determine the parameter size for reference estimation
        diag_reg: Optional, diagonal regularization for the multichannel filter
        eps: Small regularization constant to avoid division by zero

    References:
        - Souden et al, On Optimal Frequency-Domain Multichannel Linear Filtering for Noise Reduction, 2010
    """

    def __init__(
        self,
        beta: float = 1.0,
        rank: str = 'one',
        postfilter: Optional[str] = None,
        ref_channel: Optional[int] = None,
        ref_hard: bool = True,
        ref_hard_use_grad: bool = True,
        ref_subband_weighting: bool = False,
        num_subbands: Optional[int] = None,
        diag_reg: Optional[float] = 1e-6,
        eps: float = 1e-8,
    ):
        super().__init__()

        # Parametric filter
        # 0=MVDR, 1=MWF
        self.beta = beta

        # Rank
        # Assumed rank for the signal covariance matrix (psd_s)
        self.rank = rank

        if self.rank == 'full' and self.beta == 0:
            raise ValueError(f'Rank {self.rank} is not compatible with beta {self.beta}.')

        # Postfilter, applied on the output of the multichannel filter
        if postfilter not in [None, 'ban']:
            raise ValueError(f'Postfilter {postfilter} is not supported.')
        self.postfilter = postfilter

        # Regularization
        if diag_reg is not None and diag_reg < 0:
            raise ValueError(f'Diagonal regularization {diag_reg} must be positive.')
        self.diag_reg = diag_reg

        if eps <= 0:
            raise ValueError(f'Epsilon {eps} must be positive.')
        self.eps = eps

        # Reference channel
        self.ref_channel = ref_channel
        if self.ref_channel == 'max_snr':
            self.ref_estimator = ReferenceChannelEstimatorSNR(
                hard=ref_hard,
                hard_use_grad=ref_hard_use_grad,
                subband_weighting=ref_subband_weighting,
                num_subbands=num_subbands,
                eps=eps,
            )
        else:
            self.ref_estimator = None
        # Flag to determine if the filter is MISO or MIMO
        self.is_mimo = self.ref_channel is None

        logging.debug('Initialized %s', self.__class__.__name__)
        logging.debug('\tbeta:        %f', self.beta)
        logging.debug('\trank:        %s', self.rank)
        logging.debug('\tpostfilter:  %s', self.postfilter)
        logging.debug('\tdiag_reg:    %g', self.diag_reg)
        logging.debug('\teps:         %g', self.eps)
        logging.debug('\tref_channel: %s', self.ref_channel)
        logging.debug('\tis_mimo:     %s', self.is_mimo)

    @staticmethod
    def trace(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Calculate trace of matrix slices over the last
        two dimensions in the input tensor.

        Args:
            x: tensor, shape (..., C, C)

        Returns:
            Trace for each (C, C) matrix. shape (...)
        """
        trace = torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)
        if keepdim:
            trace = trace.unsqueeze(-1).unsqueeze(-1)
        return trace

    def apply_diag_reg(self, psd: torch.Tensor) -> torch.Tensor:
        """Apply diagonal regularization on psd.

        Args:
            psd: tensor, shape (..., C, C)

        Returns:
            Tensor, same shape as input.
        """
        # Regularization: diag_reg * trace(psd) + eps
        diag_reg = self.diag_reg * self.trace(psd).real + self.eps

        # Apply regularization
        psd = psd + torch.diag_embed(diag_reg.unsqueeze(-1) * torch.ones(psd.shape[-1], device=psd.device))

        return psd

    def apply_filter(self, input: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
        """Apply the MIMO filter on the input.

        Args:
            input: batch with C input channels, shape (B, C, F, T)
            filter: batch of C-input, M-output filters, shape (B, F, C, M)

        Returns:
            M-channel filter output, shape (B, M, F, T)
        """
        if not filter.is_complex():
            raise TypeError(f'Expecting complex-valued filter, found {filter.dtype}')

        if not input.is_complex():
            raise TypeError(f'Expecting complex-valued input, found {input.dtype}')

        if filter.ndim != 4 or filter.size(-2) != input.size(-3) or filter.size(-3) != input.size(-2):
            raise ValueError(f'Filter shape {filter.shape}, not compatible with input shape {input.shape}')

        output = torch.einsum('bfcm,bcft->bmft', filter.conj(), input)

        return output

    def apply_ban(self, input: torch.Tensor, filter: torch.Tensor, psd_n: torch.Tensor) -> torch.Tensor:
        """Apply blind analytic normalization postfilter. Note that this normalization has been
        derived for the GEV beamformer in [1]. More specifically, the BAN postfilter aims to scale GEV
        to satisfy the distortionless constraint and the final analytical expression is derived using
        an assumption on the norm of the transfer function.
        However, this may still be useful in some instances.

        Args:
            input: batch with M output channels (B, M, F, T)
            filter: batch of C-input, M-output filters, shape (B, F, C, M)
            psd_n: batch of noise PSDs, shape (B, F, C, C)

        Returns:
            Filtere input, shape (B, M, F, T)

        References:
            - Warsitz and Haeb-Umbach, Blind Acoustic Beamforming Based on Generalized Eigenvalue Decomposition, 2007
        """
        # number of input channel, used to normalize the numerator
        num_inputs = filter.size(-2)
        numerator = torch.einsum('bfcm,bfci,bfij,bfjm->bmf', filter.conj(), psd_n, psd_n, filter)
        numerator = torch.sqrt(numerator.abs() / num_inputs)

        denominator = torch.einsum('bfcm,bfci,bfim->bmf', filter.conj(), psd_n, filter)
        denominator = denominator.abs()

        # Scalar filter per output channel, frequency and batch
        # shape (B, M, F)
        ban = numerator / (denominator + self.eps)

        input = ban[..., None] * input

        return input

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            'input': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            'mask_s': NeuralType(('B', 'D', 'T'), FloatType()),
            'mask_n': NeuralType(('B', 'D', 'T'), FloatType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {
            'output': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor, mask_s: torch.Tensor, mask_n: torch.Tensor) -> torch.Tensor:
        """Return processed signal.
        The output has either one channel (M=1) if a ref_channel is selected,
        or the same number of channels as the input (M=C) if ref_channel is None.

        Args:
            input: Input signal, complex tensor with shape (B, C, F, T)
            mask_s: Mask for the desired signal, shape (B, F, T)
            mask_n: Mask for the undesired noise, shape (B, F, T)

        Returns:
            Processed signal, shape (B, M, F, T)
        """
        iodtype = input.dtype

        with torch.amp.autocast(input.device.type, enabled=False):
            # Convert to double
            input = input.cdouble()
            mask_s = mask_s.double()
            mask_n = mask_n.double()

            # Calculate signal statistics
            psd_s = covariance_matrix(x=input, mask=mask_s)
            psd_n = covariance_matrix(x=input, mask=mask_n)

            if self.rank == 'one':
                # Calculate filter W using (18) in [1]
                # Diagonal regularization
                if self.diag_reg:
                    psd_n = self.apply_diag_reg(psd_n)

                # MIMO filter
                # (B, F, C, C)
                W = torch.linalg.solve(psd_n, psd_s)
                lam = self.trace(W, keepdim=True).real
                W = W / (self.beta + lam + self.eps)
            elif self.rank == 'full':
                # Calculate filter W using (15) in [1]
                psd_sn = psd_s + self.beta * psd_n

                if self.diag_reg:
                    psd_sn = self.apply_diag_reg(psd_sn)

                # MIMO filter
                # (B, F, C, C)
                W = torch.linalg.solve(psd_sn, psd_s)
            else:
                raise RuntimeError(f'Unexpected rank {self.rank}')

            if torch.jit.isinstance(self.ref_channel, int):
                # Fixed ref channel
                # (B, F, C, 1)
                W = W[..., self.ref_channel].unsqueeze(-1)
            elif self.ref_estimator is not None:
                # Estimate ref channel tensor (one-hot or soft across C)
                # (B, C)
                ref_channel_tensor = self.ref_estimator(W=W, psd_s=psd_s, psd_n=psd_n).to(W.dtype)
                # Weighting across channels
                # (B, F, C, 1)
                W = torch.sum(W * ref_channel_tensor[:, None, None, :], dim=-1, keepdim=True)

            output = self.apply_filter(input=input, filter=W)

            # Optional: postfilter
            if self.postfilter == 'ban':
                output = self.apply_ban(input=output, filter=W, psd_n=psd_n)

        return output.to(iodtype)


class ReferenceChannelEstimatorSNR(NeuralModule):
    """Estimate a reference channel by selecting the reference
    that maximizes the output SNR. It returns one-hot encoded
    vector or a soft reference.

    A straight-through estimator is used for gradient when using
    hard reference.

    Args:
        hard: If true, use hard estimate of ref channel.
            If false, use a soft estimate across channels.
        hard_use_grad: Use straight-through estimator for
            the gradient.
        subband_weighting: If true, use subband weighting when
            adding across subband SNRs. If false, use average
            across subbands.

    References:
        Boeddeker et al, Front-End Processing for the CHiME-5 Dinner Party Scenario, 2018
    """

    def __init__(
        self,
        hard: bool = True,
        hard_use_grad: bool = True,
        subband_weighting: bool = False,
        num_subbands: Optional[int] = None,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.hard = hard
        self.hard_use_grad = hard_use_grad
        self.subband_weighting = subband_weighting
        self.eps = eps

        if subband_weighting and num_subbands is None:
            raise ValueError(f'Number of subbands must be provided when using subband_weighting={subband_weighting}.')
        # Subband weighting
        self.weight_s = torch.nn.Parameter(torch.ones(num_subbands)) if subband_weighting else None
        self.weight_n = torch.nn.Parameter(torch.ones(num_subbands)) if subband_weighting else None

        logging.debug('Initialized %s', self.__class__.__name__)
        logging.debug('\thard:              %d', self.hard)
        logging.debug('\thard_use_grad:     %d', self.hard_use_grad)
        logging.debug('\tsubband_weighting: %d', self.subband_weighting)
        logging.debug('\tnum_subbands:      %s', num_subbands)
        logging.debug('\teps:               %e', self.eps)

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            'W': NeuralType(('B', 'D', 'C', 'C'), SpectrogramType()),
            'psd_s': NeuralType(('B', 'D', 'C', 'C'), SpectrogramType()),
            'psd_n': NeuralType(('B', 'D', 'C', 'C'), SpectrogramType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {
            'output': NeuralType(('B', 'C'), FloatType()),
        }

    @typecheck()
    def forward(self, W: torch.Tensor, psd_s: torch.Tensor, psd_n: torch.Tensor) -> torch.Tensor:
        """
        Args:
            W: Multichannel input multichannel output filter, shape (B, F, C, M), where
               C is the number of input channels and M is the number of output channels
            psd_s: Covariance for the signal, shape (B, F, C, C)
            psd_n: Covariance for the noise, shape (B, F, C, C)

        Returns:
            One-hot or soft reference channel, shape (B, M)
        """
        if self.subband_weighting:
            # (B, F, M)
            pow_s = torch.einsum('...jm,...jk,...km->...m', W.conj(), psd_s, W).abs()
            pow_n = torch.einsum('...jm,...jk,...km->...m', W.conj(), psd_n, W).abs()

            # Subband-weighting
            # (B, F, M) -> (B, M)
            pow_s = torch.sum(pow_s * self.weight_s.softmax(dim=0).unsqueeze(1), dim=-2)
            pow_n = torch.sum(pow_n * self.weight_n.softmax(dim=0).unsqueeze(1), dim=-2)
        else:
            # Sum across f as well
            # (B, F, C, M), (B, F, C, C), (B, F, C, M) -> (B, M)
            pow_s = torch.einsum('...fjm,...fjk,...fkm->...m', W.conj(), psd_s, W).abs()
            pow_n = torch.einsum('...fjm,...fjk,...fkm->...m', W.conj(), psd_n, W).abs()

        # Estimated SNR per channel (B, C)
        snr = pow_s / (pow_n + self.eps)
        snr = 10 * torch.log10(snr + self.eps)

        # Soft reference
        ref_soft = snr.softmax(dim=-1)

        if self.hard:
            _, idx = ref_soft.max(dim=-1, keepdim=True)
            ref_hard = torch.zeros_like(snr).scatter(-1, idx, 1.0)
            if self.hard_use_grad:
                # Straight-through for gradient
                # Propagate ref_soft gradient, as if thresholding is identity
                ref = ref_hard - ref_soft.detach() + ref_soft
            else:
                # No gradient
                ref = ref_hard
        else:
            ref = ref_soft

        return ref


class WPEFilter(NeuralModule):
    """A weighted prediction error filter.
    Given input signal, and expected power of the desired signal, this
    class estimates a multiple-input multiple-output prediction filter
    and returns the filtered signal. Currently, estimation of statistics
    and processing is performed in batch mode.

    Args:
        filter_length: Length of the prediction filter in frames, per channel
        prediction_delay: Prediction delay in frames
        diag_reg: Diagonal regularization for the correlation matrix Q, applied as diag_reg * trace(Q) + eps
        eps: Small positive constant for regularization

    References:
        - Yoshioka and Nakatani, Generalization of Multi-Channel Linear Prediction
            Methods for Blind MIMO Impulse Response Shortening, 2012
        - Jukić et al, Group sparsity for MIMO speech dereverberation, 2015
    """

    def __init__(self, filter_length: int, prediction_delay: int, diag_reg: Optional[float] = 1e-6, eps: float = 1e-8):
        super().__init__()
        self.filter_length = filter_length
        self.prediction_delay = prediction_delay
        self.diag_reg = diag_reg
        self.eps = eps

        logging.debug('Initialized %s', self.__class__.__name__)
        logging.debug('\tfilter_length:    %d', self.filter_length)
        logging.debug('\tprediction_delay: %d', self.prediction_delay)
        logging.debug('\tdiag_reg:         %g', self.diag_reg)
        logging.debug('\teps:              %g', self.eps)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "power": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, input: torch.Tensor, power: torch.Tensor, input_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Given input and the predicted power for the desired signal, estimate
        the WPE filter and return the processed signal.

        Args:
            input: Input signal, shape (B, C, F, N)
            power: Predicted power of the desired signal, shape (B, C, F, N)
            input_length: Optional, length of valid frames in `input`. Defaults to `None`

        Returns:
            Tuple of (processed_signal, output_length). Processed signal has the same
            shape as the input signal (B, C, F, N), and the output length is the same
            as the input length.
        """
        # Temporal weighting: average power over channels, output shape (B, F, N)
        weight = torch.mean(power, dim=1)
        # Use inverse power as the weight
        weight = 1 / (weight + self.eps)

        # Multi-channel convolution matrix for each subband
        tilde_input = self.convtensor(input, filter_length=self.filter_length, delay=self.prediction_delay)

        # Estimate correlation matrices
        Q, R = self.estimate_correlations(
            input=input, weight=weight, tilde_input=tilde_input, input_length=input_length
        )

        # Estimate prediction filter
        G = self.estimate_filter(Q=Q, R=R)

        # Apply prediction filter
        undesired_signal = self.apply_filter(filter=G, tilde_input=tilde_input)

        # Dereverberation
        desired_signal = input - undesired_signal

        if input_length is not None:
            # Mask padded frames
            length_mask: torch.Tensor = make_seq_mask_like(
                lengths=input_length, like=desired_signal, time_dim=-1, valid_ones=False
            )
            desired_signal = desired_signal.masked_fill(length_mask, 0.0)

        return desired_signal, input_length

    @classmethod
    def convtensor(
        cls, x: torch.Tensor, filter_length: int, delay: int = 0, n_steps: Optional[int] = None
    ) -> torch.Tensor:
        """Create a tensor equivalent of convmtx_mc for each example in the batch.
        The input signal tensor `x` has shape (B, C, F, N).
        Convtensor returns a view of the input signal `x`.

        Note: We avoid reshaping the output to collapse channels and filter taps into
        a single dimension, e.g., (B, F, N, -1). In this way, the output is a view of the input,
        while an additional reshape would result in a contiguous array and more memory use.

        Args:
            x: input tensor, shape (B, C, F, N)
            filter_length: length of the filter, determines the shape of the convolution tensor
            delay: delay to add to the input signal `x` before constructing the convolution tensor
            n_steps: Optional, number of time steps to keep in the out. Defaults to the number of
                    time steps in the input tensor.

        Returns:
            Return a convolutional tensor with shape (B, C, F, n_steps, filter_length)
        """
        if x.ndim != 4:
            raise RuntimeError(f'Expecting a 4-D input. Received input with shape {x.shape}')

        B, C, F, N = x.shape

        if n_steps is None:
            # Keep the same length as the input signal
            n_steps = N

        # Pad temporal dimension
        x = torch.nn.functional.pad(x, (filter_length - 1 + delay, 0))

        # Build Toeplitz-like matrix view by unfolding across time
        tilde_X = x.unfold(-1, filter_length, 1)

        # Trim to the set number of time steps
        tilde_X = tilde_X[:, :, :, :n_steps, :]

        return tilde_X

    @classmethod
    def permute_convtensor(cls, x: torch.Tensor) -> torch.Tensor:
        """Reshape and permute columns to convert the result of
        convtensor to be equal to convmtx_mc. This is used for verification
        purposes and it is not required to use the filter.

        Args:
            x: output of self.convtensor, shape (B, C, F, N, filter_length)

        Returns:
            Output has shape (B, F, N, C*filter_length) that corresponds to
            the layout of convmtx_mc.
        """
        B, C, F, N, filter_length = x.shape

        # .view will not work, so a copy will have to be created with .reshape
        # That will result in more memory use, since we don't use a view of the original
        # multi-channel signal
        x = x.permute(0, 2, 3, 1, 4)
        x = x.reshape(B, F, N, C * filter_length)

        permute = []
        for m in range(C):
            permute[m * filter_length : (m + 1) * filter_length] = m * filter_length + np.flip(
                np.arange(filter_length)
            )
        return x[..., permute]

    def estimate_correlations(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        tilde_input: torch.Tensor,
        input_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            input: Input signal, shape (B, C, F, N)
            weight: Time-frequency weight, shape (B, F, N)
            tilde_input: Multi-channel convolution tensor, shape (B, C, F, N, filter_length)
            input_length: Length of each input example, shape (B)

        Returns:
            Returns a tuple of correlation matrices for each batch.

            Let `X` denote the input signal in a single subband,
            `tilde{X}` the corresponding multi-channel correlation matrix,
            and `w` the vector of weights.

            The first output is Q = tilde{X}^H * diag(w) * tilde{X}, for each (b, f).
            The matrix Q has shape (C * filter_length, C * filter_length)
            The output is returned in a tensor with shape (B, F, C, filter_length, C, filter_length).

            The second output is R = tilde{X}^H * diag(w) * X, for each (b, f).
            The matrix R has shape (C * filter_length, C)
            The output is returned in a tensor with shape (B, F, C, filter_length, C). The last
            dimension corresponds to output channels.
        """
        if input_length is not None:
            # Take only valid samples into account
            length_mask: torch.Tensor = make_seq_mask_like(
                lengths=input_length, like=weight, time_dim=-1, valid_ones=False
            )
            weight = weight.masked_fill(length_mask, 0.0)

        # Calculate (1)
        # result: (B, F, C, filter_length, C, filter_length)
        Q = torch.einsum('bjfik,bmfin->bfjkmn', tilde_input.conj(), weight[:, None, :, :, None] * tilde_input)

        # Calculate (2)
        # result: (B, F, C, filter_length, C)
        R = torch.einsum('bjfik,bmfi->bfjkm', tilde_input.conj(), weight[:, None, :, :] * input)

        return Q, R

    def estimate_filter(self, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        r"""Estimate the MIMO prediction filter as G(b,f) = Q(b,f) \ R(b,f)
        for each subband in each example in the batch (b, f).

        Args:
            Q: shape (B, F, C, filter_length, C, filter_length)
            R: shape (B, F, C, filter_length, C)

        Returns:
            Complex-valued prediction filter, shape (B, C, F, C, filter_length)
        """
        B, F, C, filter_length, _, _ = Q.shape
        assert (
            filter_length == self.filter_length
        ), f'Shape of Q {Q.shape} is not matching filter length {self.filter_length}'

        # Reshape to analytical dimensions for each (b, f)
        Q = Q.reshape(B, F, C * self.filter_length, C * filter_length)
        R = R.reshape(B, F, C * self.filter_length, C)

        # Diagonal regularization
        if self.diag_reg:
            # Regularization: diag_reg * trace(Q) + eps
            diag_reg = self.diag_reg * torch.diagonal(Q, dim1=-2, dim2=-1).sum(-1).real + self.eps
            # Apply regularization on Q
            Q = Q + torch.diag_embed(diag_reg.unsqueeze(-1) * torch.ones(Q.shape[-1], device=Q.device))

        # Solve for the filter
        G = torch.linalg.solve(Q, R)

        # Reshape to desired representation: (B, F, input channels, filter_length, output channels)
        G = G.reshape(B, F, C, filter_length, C)
        # Move output channels to front: (B, output channels, F, input channels, filter_length)
        G = G.permute(0, 4, 1, 2, 3)

        return G

    def apply_filter(
        self, filter: torch.Tensor, input: Optional[torch.Tensor] = None, tilde_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply a prediction filter `filter` on the input `input` as

            output(b,f) = tilde{input(b,f)} * filter(b,f)

        If available, directly use the convolution matrix `tilde_input`.

        Args:
            input: Input signal, shape (B, C, F, N)
            tilde_input: Convolution matrix for the input signal, shape (B, C, F, N, filter_length)
            filter: Prediction filter, shape (B, C, F, C, filter_length)

        Returns:
            Multi-channel signal obtained by applying the prediction filter on
            the input signal, same shape as input (B, C, F, N)
        """
        if input is None and tilde_input is None:
            raise RuntimeError('Both inputs cannot be None simultaneously.')
        if input is not None and tilde_input is not None:
            raise RuntimeError('Both inputs cannot be provided simultaneously.')

        if tilde_input is None:
            tilde_input = self.convtensor(input, filter_length=self.filter_length, delay=self.prediction_delay)

        # For each (batch, output channel, f, time step), sum across (input channel, filter tap)
        output = torch.einsum('bjfik,bmfjk->bmfi', tilde_input, filter)

        return output
