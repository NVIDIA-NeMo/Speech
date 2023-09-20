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

from torch import nn

from nemo.collections.multimodal.speech_cv.parts.submodules.conv2d import Conv2d
from nemo.collections.multimodal.speech_cv.parts.submodules.global_avg_pool2d import GlobalAvgPool2d
from nemo.collections.multimodal.speech_cv.parts.submodules.resnet_block import ResNetBlock
from nemo.collections.multimodal.speech_cv.parts.submodules.resnet_bottleneck_block import ResNetBottleneckBlock


class ResNet(nn.Module):

    """ ResNet (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)
    Models: 224 x 224
    ResNet18: 11,689,512 Params
    ResNet34: 21,797,672 Params
    ResNet50: 25,557,032 Params
    ResNet101: 44,549,160 Params
    Resnet152: 60,192,808 Params
    Reference: "Deep Residual Learning for Image Recognition" by He et al.
    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, dim_input=3, dim_output=1000, model="ResNet50", include_stem=True, include_head=True):
        super(ResNet, self).__init__()

        assert model in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

        if model == "ResNet18":
            dim_stem = 64
            dim_blocks = [64, 128, 256, 512]
            num_blocks = [2, 2, 2, 2]
            bottleneck = False
        elif model == "ResNet34":
            dim_stem = 64
            dim_blocks = [64, 128, 256, 512]
            num_blocks = [3, 4, 6, 3]
            bottleneck = False
        elif model == "ResNet50":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 6, 3]
            bottleneck = True
        elif model == "ResNet101":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 23, 3]
            bottleneck = True
        elif model == "ResNet152":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 8, 36, 3]
            bottleneck = True

        self.stem = (
            nn.Sequential(
                Conv2d(
                    in_channels=dim_input,
                    out_channels=dim_stem,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    weight_init="he_normal",
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=dim_stem),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            )
            if include_stem
            else nn.Identity()
        )

        # Blocks
        self.blocks = nn.ModuleList()
        for stage_id in range(4):

            for block_id in range(num_blocks[stage_id]):

                # Projection Block
                if block_id == 0:
                    if stage_id == 0:
                        stride = (1, 1)
                        bottleneck_ratio = 1
                        in_features = dim_stem
                    else:
                        stride = (2, 2)
                        bottleneck_ratio = 2
                        in_features = dim_blocks[stage_id - 1]
                # Default Block
                else:
                    stride = (1, 1)
                    in_features = dim_blocks[stage_id]
                    bottleneck_ratio = 4

                if bottleneck:
                    self.blocks.append(
                        ResNetBottleneckBlock(
                            in_features=in_features,
                            out_features=dim_blocks[stage_id],
                            bottleneck_ratio=bottleneck_ratio,
                            kernel_size=(3, 3),
                            stride=stride,
                        )
                    )
                else:
                    self.blocks.append(
                        ResNetBlock(
                            in_features=in_features,
                            out_features=dim_blocks[stage_id],
                            kernel_size=(3, 3),
                            stride=stride,
                        )
                    )

        # Head
        self.head = (
            nn.Sequential(
                GlobalAvgPool2d(),
                nn.Linear(in_features=dim_blocks[-1], out_features=dim_output)
                if dim_output is not None
                else nn.Identity(),
            )
            if include_head
            else nn.Identity()
        )

    def forward(self, x):

        # Is Video
        if x.dim() == 5:

            is_video = True
            batch_size = x.shape[0]
            video_frames = x.shape[2]

            # (B, Din, T, H, W) -> (B * T, Din, H, W)
            x = x.transpose(1, 2).flatten(start_dim=0, end_dim=1)

        else:
            is_video = False

        # (B, Din, H, W) -> (B, D0, H//4, W//4)
        x = self.stem(x)

        # (B, D0, H//4, W//4) -> (B, D4, H//32, W//32)
        for block in self.blocks:
            x = block(x)

        # (B, D4, H//32, W//32) -> (B, Dout)
        x = self.head(x)

        # Is Video
        if is_video:

            # (B * T, Dout) -> (B, Dout, T)
            if x.dim() == 2:
                x = x.reshape(batch_size, video_frames, -1).transpose(1, 2)

            # (B * T, D4, H//32, W//32) -> (B, D4, T, H//32, W//32)
            else:
                x = x.reshape(batch_size, video_frames, x.shape[1], x.shape[2], x.shape[3]).transpose(1, 2)

        return x
