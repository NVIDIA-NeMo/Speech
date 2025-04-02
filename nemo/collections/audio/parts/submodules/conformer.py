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

from typing import Dict

import einops
import torch

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import ChannelType, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging

__all__ = ['SpectrogramConformer']


class SpectrogramConformer(NeuralModule):
    """A Conformer-based model for processing complex-valued spectrograms.

    This model processes complex-valued inputs by stacking real and imaginary components
    along the channel dimension. The stacked tensor is processed using Conformer layers,
    and the output is projected back to generate real and imaginary components of the
    output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
        kwargs: additional arguments for ConformerEncoder
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input channels for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output channels for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Conformer-based estimator
        conformer_params = kwargs.copy()
        conformer_params['feat_in'] = conformer_params['feat_out'] = (
            2 * self.in_channels * kwargs['feat_in']
        )  # stack real and imag
        logging.debug('Conformer params: %s', conformer_params)
        self.conformer = ConformerEncoder(**conformer_params)

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def context_size(self):
        """Returns the attention context size used by the conformer encoder.

        The context size is a list of two integers [left_context, right_context] that defines
        how many frames to the left and right each frame can attend to in the self-attention
        layers.

        Returns:
            List[int]: The attention context size [left_context, right_context]
        """
        return self.conformer.att_context_size

    @context_size.setter
    def context_size(self, value):
        """Sets the attention context size used by the conformer encoder.

        The context size is a list of two integers [left_context, right_context] that defines
        how many frames to the left and right each frame can attend to in the self-attention
        layers.

        Args:
            value (List[int]): The attention context size [left_context, right_context]
        """
        self.conformer.set_default_att_context_size(value)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            # convolutional context
            "cache_last_channel": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
            "cache_last_time": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
            "cache_last_channel_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
            # convolutional context
            "cache_last_channel_next": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
            "cache_last_time_next": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
            "cache_last_channel_next_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, input, input_length=None, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        """Forward pass for the SpectrogramConformer model.

        This method processes complex-valued inputs by stacking real and imaginary components,
        passing the stacked tensor through Conformer layers, and projecting back to generate
        real and imaginary components of the output channels.
        """
        B, C_in, D, T = input.shape
        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary components
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI D T -> B (C RI D) T')

        # Conformer
        if cache_last_channel is None:
            # Not using caching mode
            output, output_length = self.conformer(audio_signal=input, length=input_length)
        else:
            # Using caching mode
            output, output_length, cache_last_channel, cache_last_time, cache_last_channel_len = self.conformer(
                audio_signal=input,
                length=input_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )

        # Output projection
        output = einops.rearrange(output, 'B (C RI D) T -> B (C RI) D T', C=self.in_channels, RI=2, D=D)
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = einops.rearrange(output, 'B (C RI) D T -> B C D T RI', C=self.out_channels, RI=2, D=D)
        output = torch.view_as_complex(output.contiguous())

        if cache_last_channel is None:
            return output, output_length
        else:
            return output, output_length, cache_last_channel, cache_last_time, cache_last_channel_len
