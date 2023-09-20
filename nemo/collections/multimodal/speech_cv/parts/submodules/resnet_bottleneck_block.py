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

import torch
from torch import nn
from torch.nn.modules.utils import _pair

from nemo.collections.multimodal.speech_cv.parts.submodules.conv2d import Conv2d


class ResNetBottleneckBlock(nn.Module):

    """ ResNet Bottleneck Residual Block used by ResNet50, ResNet101 and ResNet152 networks.
    References: "Deep Residual Learning for Image Recognition", He et al.
    https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        in_features,
        out_features,
        bottleneck_ratio,
        kernel_size,
        stride,
        weight_init="he_normal",
        bias_init="zeros",
    ):
        super(ResNetBottleneckBlock, self).__init__()

        # Assert
        assert in_features % bottleneck_ratio == 0

        # Convert to pair
        kernel_size = _pair(kernel_size)

        # layers
        self.layers = nn.Sequential(
            Conv2d(
                in_channels=in_features,
                out_channels=in_features // bottleneck_ratio,
                kernel_size=1,
                bias=False,
                weight_init=weight_init,
                bias_init=bias_init,
            ),
            nn.BatchNorm2d(in_features // bottleneck_ratio),
            nn.ReLU(),
            Conv2d(
                in_channels=in_features // bottleneck_ratio,
                out_channels=in_features // bottleneck_ratio,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                weight_init=weight_init,
                bias_init=bias_init,
                padding=((kernel_size[0] - 1) // 2, kernel_size[1] // 2),
            ),
            nn.BatchNorm2d(in_features // bottleneck_ratio),
            nn.ReLU(),
            Conv2d(
                in_channels=in_features // bottleneck_ratio,
                out_channels=out_features,
                kernel_size=1,
                bias=False,
                weight_init=weight_init,
                bias_init=bias_init,
            ),
            nn.BatchNorm2d(out_features),
        )

        # Joined Post Act
        self.joined_post_act = nn.ReLU()

        # Residual Block
        if torch.prod(torch.tensor(stride)) > 1 or in_features != out_features:
            self.residual = nn.Sequential(
                Conv2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    weight_init=weight_init,
                    bias_init=bias_init,
                ),
                nn.BatchNorm2d(out_features),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):

        # Forward Layers
        x = self.joined_post_act(self.layers(x) + self.residual(x))

        return x
