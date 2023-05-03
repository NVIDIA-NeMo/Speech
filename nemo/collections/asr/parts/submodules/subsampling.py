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

import math

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from nemo.collections.asr.parts.submodules.causal_convs import CausalConv2D


class StackingSubsampling(torch.nn.Module):
    """Stacking subsampling which simply stacks consecutive frames to reduce the sampling rate
    Args:
        subsampling_factor (int): The subsampling factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        norm (bool): whether to use an MLP layer after the stacking along with normalization. default is False.
    """

    def __init__(self, subsampling_factor, feat_in, feat_out, norm=False):
        super(StackingSubsampling, self).__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out)
        if norm:
            self.pre_norm = LayerNorm(feat_in)
        else:
            self.pre_norm = None

    def get_sampling_frames(self):
        return self.subsampling_factor

    def get_streaming_cache_size(self):
        return 0

    def forward(self, x, lengths):
        b, t, h = x.size()
        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = self.proj_out(x)
        lengths = torch.div(lengths + pad_size, self.subsampling_factor, rounding_mode='floor')
        return x, lengths


class ConvSubsampling(torch.nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: Transformer-transducer: end-to-end speech recognition with self-attention (https://arxiv.org/pdf/1910.12977.pdf)
    Striding Subsampling: "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al. (https://ieeexplore.ieee.org/document/8462506)
    Args:
        subsampling (str): The subsampling technique from {"vggnet", "striding"}
        subsampling_factor (int): The subsampling factor which should be a power of 2
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self, subsampling, subsampling_factor, feat_in, feat_out, conv_channels, activation=nn.ReLU(), is_causal=False
    ):
        super(ConvSubsampling, self).__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal

        in_channels = 1
        layers = []

        if subsampling == 'vggnet':
            self._stride = 2
            self._kernel_size = 2
            self._ceil_mode = True

            self._left_padding = 0
            self._right_padding = 0

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.MaxPool2d(
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        ceil_mode=self._ceil_mode,
                    )
                )
                in_channels = conv_channels

        elif subsampling == 'dw_striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    )
                )
            else:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        )
                    )

                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            lengths=in_length,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x, lengths):
        lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        x = x.unsqueeze(1)

        x = self.conv(x)

        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, lengths

    def reset_parameters(self):
        # initialize weights
        if self._subsampling == 'dw_striding':
            with torch.no_grad():
                # init conv
                scale = 1.0 / self._kernel_size
                dw_max = (self._kernel_size ** 2) ** -0.5
                pw_max = self._conv_channels ** -0.5

                torch.nn.init.uniform_(self.conv[0].weight, -scale, scale)
                torch.nn.init.uniform_(self.conv[0].bias, -scale, scale)

                for idx in range(2, len(self.conv), 3):
                    torch.nn.init.uniform_(self.conv[idx].weight, -dw_max, dw_max)
                    torch.nn.init.uniform_(self.conv[idx].bias, -dw_max, dw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].weight, -pw_max, pw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].bias, -pw_max, pw_max)

                # init fc (80 * 64 = 5120 from https://github.com/kssteven418/Squeezeformer/blob/13c97d6cf92f2844d2cb3142b4c5bfa9ad1a8951/src/models/conformer_encoder.py#L487
                fc_scale = (self._feat_out * self._feat_in / self._sampling_num) ** -0.5
                torch.nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
                torch.nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class TimeReductionModule(nn.Module):
    """
    Squeezeformer Time Reduction procedure. Downsamples the audio by `stride` in the time dimension.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    def __init__(self, d_model: int, out_dim: int, kernel_size: int = 5, stride: int = 2):
        super().__init__()

        self.d_model = d_model
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = max(0, self.kernel_size - self.stride)

        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=d_model,
        )

        self.pw_conv = nn.Conv1d(
            in_channels=d_model, out_channels=out_dim, kernel_size=1, stride=1, padding=0, groups=1,
        )

        self.reset_parameters()

    def forward(self, x, att_mask=None, pad_mask=None):
        x = x.transpose(1, 2)  # [B, C, T]
        if pad_mask is not None:
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.dw_conv(x)
        x = self.pw_conv(x)

        x = x.transpose(1, 2)  # [B, T, C]

        B, T, D = x.size()
        if att_mask is not None and pad_mask is not None:
            att_mask = att_mask[:, :: self.stride, :: self.stride]
            pad_mask = pad_mask[:, :: self.stride]
            L = pad_mask.size(-1)
            x = torch.nn.functional.pad(x, (0, 0, 0, L - T))

        return x, att_mask, pad_mask

    def reset_parameters(self):
        dw_max = self.kernel_size ** -0.5
        pw_max = self.d_model ** -0.5

        with torch.no_grad():
            torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
            torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
            torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
            torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)


class SubsamplingReductionModule(nn.Module):
    """Downsamples the audio signal in time dimension."""

    def __init__(self, reduction: str, d_model: int, reduction_factor: int = 2):
        super().__init__()

        assert reduction in ['pooling', 'striding']

        self.reduction = reduction
        self.d_model = d_model
        self._sampling_num = int(math.log(reduction_factor, 2))

        if reduction == 'pooling':
            self.reduction_enc = nn.MaxPool1d(kernel_size=reduction_factor)
            self.padding = 0
            self.kernel_size = self.reduction_enc.kernel_size
            self.stride = self.reduction_enc.stride
        elif reduction == 'striding':
            self.reduction_enc = ConvSubsampling(
                subsampling='striding',
                subsampling_factor=reduction_factor,
                feat_in=d_model,
                feat_out=d_model,
                conv_channels=d_model,
                activation=nn.ReLU(),
                is_causal=False,
            )

    def forward(self, x, lengths):
        """Shapes:
            - x: [B, T, C]
            - lengths: [B]
        """

        if self.reduction == 'striding':
            x, lengths = self.reduction_enc(x=x, lengths=lengths)
        else:
            x = torch.transpose(x, 1, 2)  # [B, C, T]
            lengths = calc_length(
                lengths=lengths,
                all_paddings=self.padding,
                kernel_size=self.kernel_size,
                stride=self.stride,
                ceil_mode=False,
                repeat_num=self._sampling_num,
            )
            x = self.reduction_enc(x)
            x = torch.transpose(x, 1, 2)  # [B, T, C]

        return x, lengths
