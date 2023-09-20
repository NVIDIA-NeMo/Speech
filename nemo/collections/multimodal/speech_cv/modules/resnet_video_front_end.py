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

from collections import OrderedDict

import torch
from torch import nn

from nemo.collections.multimodal.speech_cv.parts.submodules.resnet import ResNet
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import LengthsType, NeuralType, VideoSignal


class ResNetVideoFrontEnd(NeuralModule):
    """
    Lip Reading / Visual Speech Recognition (VSR) ResNet Front-End Network

    Paper:
    'Audio-Visual Efficient Conformer for Robust Speech Recognition' by Burchi and Timofte
    https://arxiv.org/abs/2301.01456

    Args:
        in_channels: number of inputs video channels, 1 for grayscale and 3 for RGB
        model: model size in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
        dim_output: output feature dimension for linear projection after spacial average pooling
        out_channels_first: Whether outputs should have channels_first format (Batch, Dout, Time) or channels_last (Batch, Time, Dout)
    """

    def __init__(self, in_channels=1, model="ResNet18", dim_output=256, out_channels_first=True):
        super(ResNetVideoFrontEnd, self).__init__()

        self.front_end = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels, out_channels=64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)
            ),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            ResNet(include_stem=False, dim_output=dim_output, model=model),
        )

        self.out_channels_first = out_channels_first

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T', 'H', 'W'), VideoSignal()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def input_types_for_export(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "output_signal": NeuralType(('B', 'D', 'T'), NeuralType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    def forward(self, input_signal, length):

        # Front-End Network (Batch, Din, Time, Height, Width) -> (Batch, Dout, Time)
        input_signal = self.front_end(input_signal)

        # Transpose to channels_last format (Batch, Dout, Time) -> (Batch, Time, Dout)
        if not self.out_channels_first:
            input_signal = input_signal.transpose(1, 2)

        return input_signal, length
