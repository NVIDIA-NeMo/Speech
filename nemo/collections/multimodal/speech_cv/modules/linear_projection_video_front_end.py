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

from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import LengthsType, NeuralType, VideoSignal


class LinearProjectionVideoFrontEnd(NeuralModule):

    """
    Linear Projection Video Front-End for Lip Reading

    The spatial dimension is flattened and projected to dim_output using a Linear layer.
    This is equivalent to having a convolution layer with a kernel size of the size of the image.
    Circle crop can be used as pre-processing to crop the image as a circle around lips and ignore corner pixels

    Args:
        in_channels: number of inputs video channels, 1 for grayscale and 3 for RGB
        in_height: image height
        in_width: image width
        dim_output: output feature dimension for linear projection
        out_channels_first: Whether outputs should have channels_first format (Batch, Dout, Time) or channels_last (Batch, Time, Dout)
        circle_crop: crop the image as a circle before the Linear layer, default to False
        circle_radius: the circle radius, default to 1 for full circle
    
    """

    def __init__(
        self,
        in_channels,
        in_height,
        in_width,
        dim_output,
        out_channels_first=True,
        circle_crop=False,
        circle_radius=1.0,
    ):
        super(LinearProjectionVideoFrontEnd, self).__init__()

        self.out_channels_first = out_channels_first
        self.in_height = in_height
        self.in_width = in_width
        self.dim_output = dim_output
        self.in_channels = in_channels
        self.circle_crop = circle_crop
        self.circle_radius = circle_radius
        self.circle_indices = self.get_circle_indices()

        if self.dim_output is not None:
            if self.circle_crop:
                self.linear_proj = nn.Linear(in_channels * len(self.circle_indices), dim_output)
            else:
                self.linear_proj = nn.Linear(in_channels * in_height * in_width, dim_output)
        else:
            self.linear_proj = nn.Identity()

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

    def get_circle_indices(self):

        """ return image indices inside circle of radius circle_radius """

        # Create linspace
        linspace_height = (torch.linspace(0, 2, steps=self.in_height) - 1).abs()
        linspace_width = (torch.linspace(0, 2, steps=self.in_width) - 1).abs()

        # Repeat linspace along height/width
        linspace_height = linspace_height.unsqueeze(dim=-1).repeat(1, self.in_width).flatten()
        linspace_width = linspace_width.repeat(self.in_height)

        # Compute norm
        dist = torch.sqrt(linspace_height.square() + linspace_width.square())

        # Get circle indices
        circle_indices = torch.nonzero(dist <= self.circle_radius).squeeze(dim=-1)

        return circle_indices

    def forward(self, input_signal, length):

        # Permute (B, C, T, H, W) -> (B, T, H, W, C)
        input_signal = input_signal.permute(0, 2, 3, 4, 1)

        # Circle Crop
        if self.circle_crop:

            # Flatten height, width (B, T, H, W, C) -> (B, T, H*W, C)
            input_signal = input_signal.flatten(start_dim=2, end_dim=-2)

            # (B, T, H*W, C) -> (B, T, N circle, C)
            input_signal = input_signal[:, :, self.circle_indices]

            # Flatten circle and channels (B, T, N circle, C) -> (B, T, N)
            input_signal = input_signal.flatten(start_dim=2, end_dim=-1)

        # Flatten height, width and channels (B, T, H, W, C) -> (B, T, N)
        else:
            input_signal = input_signal.flatten(start_dim=2, end_dim=-1)

        # Project (B, T, N) -> (B, T, Dout)
        input_signal = self.linear_proj(input_signal)

        # Transpose to channels_last format (Batch, Dout, Time) -> (Batch, Time, Dout)
        if self.out_channels_first:
            output_signal = input_signal.transpose(1, 2)
        else:
            output_signal = input_signal

        return output_signal, length
