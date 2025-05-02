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

from typing import Dict, Optional

import torch

from nemo.collections.audio.losses.audio import calculate_mean
from nemo.collections.audio.parts.utils.audio import wrap_to_pi
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging


class SpectrogramToMultichannelFeatures(NeuralModule):
    """Convert a complex-valued multi-channel spectrogram to
    multichannel features.

    Args:
        num_subbands: Expected number of subbands in the input signal
        num_input_channels: Optional, provides the number of channels
                            of the input signal. Used to infer the number
                            of output channels.
        mag_reduction: Reduction across channels. Default `None`, will calculate
                       magnitude of each channel.
        mag_power: Optional, apply power on the magnitude.
        use_ipd: Use inter-channel phase difference (IPD).
        mag_normalization: Normalization for magnitude features
        ipd_normalization: Normalization for IPD features
        eps: Small regularization constant.
    """

    def __init__(
        self,
        num_subbands: int,
        num_input_channels: Optional[int] = None,
        mag_reduction: Optional[str] = None,
        mag_power: Optional[float] = None,
        use_ipd: bool = False,
        mag_normalization: Optional[str] = None,
        ipd_normalization: Optional[str] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.mag_reduction = mag_reduction
        self.mag_power = mag_power
        self.use_ipd = use_ipd

        if mag_normalization not in [None, 'mean', 'mean_var']:
            raise NotImplementedError(f'Unknown magnitude normalization {mag_normalization}')
        self.mag_normalization = mag_normalization

        if ipd_normalization not in [None, 'mean', 'mean_var']:
            raise NotImplementedError(f'Unknown ipd normalization {ipd_normalization}')
        self.ipd_normalization = ipd_normalization

        if self.use_ipd:
            self._num_features = 2 * num_subbands
            self._num_channels = num_input_channels
        else:
            self._num_features = num_subbands
            self._num_channels = num_input_channels if self.mag_reduction is None else 1

        self.eps = eps

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tnum_subbands:      %d', num_subbands)
        logging.debug('\tmag_reduction:     %s', self.mag_reduction)
        logging.debug('\tmag_power:         %s', self.mag_power)
        logging.debug('\tuse_ipd:           %s', self.use_ipd)
        logging.debug('\tmag_normalization: %s', self.mag_normalization)
        logging.debug('\tipd_normalization: %s', self.ipd_normalization)
        logging.debug('\teps:               %f', self.eps)
        logging.debug('\t_num_features:     %s', self._num_features)
        logging.debug('\t_num_channels:     %s', self._num_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @property
    def num_features(self) -> int:
        """Configured number of features"""
        return self._num_features

    @property
    def num_channels(self) -> int:
        """Configured number of channels"""
        if self._num_channels is not None:
            return self._num_channels
        else:
            raise ValueError(
                'Num channels is not configured. To configure this, `num_input_channels` '
                'must be provided when constructing the object.'
            )

    @staticmethod
    def get_mean_time_channel(input: torch.Tensor, input_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate mean across time and channel dimensions.

        Args:
            input: tensor with shape (B, C, F, T)
            input_length: tensor with shape (B,)

        Returns:
            Mean of `input` calculated across time and channel dimension
            with shape (B, 1, F, 1)
        """
        assert input.ndim == 4, f'Expected input to have 4 dimensions, got {input.ndim}'

        if input_length is None:
            mean = torch.mean(input, dim=(-1, -3), keepdim=True)
        else:
            # temporal mean
            mean = calculate_mean(input, input_length, dim=-1, keepdim=True)
            # channel mean
            mean = torch.mean(mean, dim=-3, keepdim=True)

        return mean

    @classmethod
    def get_mean_std_time_channel(
        cls, input: torch.Tensor, input_length: Optional[torch.Tensor] = None, eps: float = 1e-10
    ) -> torch.Tensor:
        """Calculate mean and standard deviation across time and channel dimensions.

        Args:
            input: tensor with shape (B, C, F, T)
            input_length: tensor with shape (B,)

        Returns:
            Mean and standard deviation of the `input` calculated across time and
            channel dimension, each with shape (B, 1, F, 1).
        """
        assert input.ndim == 4, f'Expected input to have 4 dimensions, got {input.ndim}'

        if input_length is None:
            std, mean = torch.std_mean(input, dim=(-1, -3), unbiased=False, keepdim=True)
        else:
            mean = cls.get_mean_time_channel(input, input_length)
            std = (input - mean).pow(2)
            # temporal mean
            std = calculate_mean(std, input_length, dim=-1, keepdim=True)
            # channel mean
            std = torch.mean(std, dim=-3, keepdim=True)
            # final value
            std = torch.sqrt(std.clamp(eps))

        return mean, std

    @typecheck(
        input_types={
            'input': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            'input_length': NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            'output': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        },
    )
    def normalize_mean(self, input: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
        """Mean normalization for the input tensor.

        Args:
            input: input tensor
            input_length: valid length for each example

        Returns:
            Mean normalized input.
        """
        mean = self.get_mean_time_channel(input=input, input_length=input_length)
        output = input - mean
        return output

    @typecheck(
        input_types={
            'input': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            'input_length': NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            'output': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        },
    )
    def normalize_mean_var(self, input: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
        """Mean and variance normalization for the input tensor.

        Args:
            input: input tensor
            input_length: valid length for each example

        Returns:
            Mean and variance normalized input.
        """
        mean, std = self.get_mean_std_time_channel(input=input, input_length=input_length, eps=self.eps)
        output = (input - mean) / std
        return output

    @typecheck()
    def forward(self, input: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
        """Convert input batch of C-channel spectrograms into
        a batch of time-frequency features with dimension num_feat.
        The output number of channels may be the same as input, or
        reduced to 1, e.g., if averaging over magnitude and not appending individual IPDs.

        Args:
            input: Spectrogram for C channels with F subbands and N time frames, (B, C, F, N)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            num_feat_channels channels with num_feat features, shape (B, num_feat_channels, num_feat, N)
        """
        num_input_channels = input.size(1)

        # Magnitude spectrum
        if self.mag_reduction is None:
            mag = torch.abs(input)
        elif self.mag_reduction == 'abs_mean':
            mag = torch.abs(torch.mean(input, axis=1, keepdim=True))
        elif self.mag_reduction == 'mean_abs':
            mag = torch.mean(torch.abs(input), axis=1, keepdim=True)
        elif self.mag_reduction == 'rms':
            mag = torch.sqrt(torch.mean(torch.abs(input) ** 2, axis=1, keepdim=True))
        else:
            raise ValueError(f'Unexpected magnitude reduction {self.mag_reduction}')

        if self.mag_power is not None:
            mag = torch.pow(mag, self.mag_power)

        if self.mag_normalization == 'mean':
            # normalize mean across channels and time steps
            mag = self.normalize_mean(input=mag, input_length=input_length)
        elif self.mag_normalization == 'mean_var':
            # normalize mean and variance across channels and time steps
            mag = self.normalize_mean_var(input=mag, input_length=input_length)

        features = mag

        if self.use_ipd:
            if num_input_channels == 1:
                # no IPD for single-channel input
                ipd = torch.zeros_like(input, dtype=features.dtype, device=features.device)
            else:
                # Calculate IPD relative to the average spec
                spec_mean = torch.mean(input, axis=1, keepdim=True)  # channel average
                ipd = torch.angle(input) - torch.angle(spec_mean)
                # Modulo to [-pi, pi]
                ipd = wrap_to_pi(ipd)

                if self.ipd_normalization == 'mean':
                    # normalize mean across channels and time steps
                    # mean across time
                    ipd = self.normalize_mean(input=ipd, input_length=input_length)
                elif self.ipd_normalization == 'mean_var':
                    ipd = self.normalize_mean_var(input=ipd, input_length=input_length)

            # Concatenate to existing features
            features = torch.cat([features.expand(ipd.shape), ipd], axis=2)

        if self._num_channels is not None and features.size(1) != self._num_channels:
            raise RuntimeError(
                f'Number of channels in features {features.size(1)} is different than the configured number of channels {self._num_channels}'
            )

        return features, input_length
