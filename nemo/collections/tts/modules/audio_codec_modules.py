# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from nemo.collections.tts.parts.utils.helpers import mask_sequence_tensor
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types.elements import AudioSignal, EncodedRepresentation, LengthsType, VoidType
from nemo.core.neural_types.neural_type import NeuralType


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def get_padding_2d(kernel_size: Tuple[int, int], dilation: Tuple[int, int]) -> Tuple[int, int]:
    paddings = (get_padding(kernel_size[0], dilation[0]), get_padding(kernel_size[1], dilation[1]))
    return paddings


def get_down_sample_padding(kernel_size: int, stride: int) -> int:
    return (kernel_size - stride + 1) // 2


def get_up_sample_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
    output_padding = (kernel_size - stride) % 2
    padding = (kernel_size - stride + 1) // 2
    return padding, output_padding


class Conv1dNorm(NeuralModule):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: Optional[int] = None
    ):
        super().__init__()
        if not padding:
            padding = get_padding(kernel_size)
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    def forward(self, inputs, lengths):
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, lengths)
        return out


class ConvTranspose1dNorm(NeuralModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding, output_padding = get_up_sample_padding(kernel_size, stride)
        conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            padding_mode="zeros",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    def forward(self, inputs, lengths):
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, lengths)
        return out


class Conv2dNorm(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        assert len(kernel_size) == len(dilation)
        padding = get_padding_2d(kernel_size, dilation)
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'H', 'T'), VoidType()),
        }

    @property
    def output_types(self):
        return {
            "out": [NeuralType(('B', 'C', 'H', 'T'), VoidType())],
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    def forward(self, inputs):
        return self.conv(inputs)
