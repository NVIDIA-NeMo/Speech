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

#  MIT License
#
#  Copyright (c) 2020 Jungil Kong
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

# The following functions/classes were based on code from https://github.com/jik876/hifi-gan:
# ResBlock1, ResBlock2, Generator, DiscriminatorP, DiscriminatorS, MultiScaleDiscriminator,
# MultiPeriodDiscriminator, init_weights, get_padding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types.elements import AudioSignal, MelSpectrogramType, VoidType
from nemo.core.neural_types.neural_type import NeuralType

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    __constants__ = ['lrelu_slope']

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.lrelu_slope = LRELU_SLOPE
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    __constants__ = ['lrelu_slope']

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(NeuralModule):
    __constants__ = ['lrelu_slope', 'num_kernels', 'num_upsamples']

    def __init__(
        self,
        resblock,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        initial_input_size=80,
        apply_weight_init_conv_pre=False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_input_size, upsample_initial_channel, 7, 1, padding=3))
        self.lrelu_slope = LRELU_SLOPE
        resblock = ResBlock1 if resblock == 1 else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            resblock_list = nn.ModuleList()
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                resblock_list.append(resblock(ch, k, d))
            self.resblocks.append(resblock_list)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        if apply_weight_init_conv_pre:
            self.conv_pre.apply(init_weights)

    @property
    def input_types(self):
        return {
            "x": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'S', 'T'), AudioSignal()),
        }

    @typecheck()
    def forward(self, x):
        x = self.conv_pre(x)
        for upsample_layer, resblock_group in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = upsample_layer(x)
            xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            for resblock in resblock_group:
                xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for group in self.resblocks:
            for block in group:
                block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(NeuralModule):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, debug=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        conv_ch = [32, 128, 512, 1024] if not debug else [8, 12, 32, 64]
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, conv_ch[0], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(conv_ch[0], conv_ch[1], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(conv_ch[1], conv_ch[2], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(conv_ch[2], conv_ch[3], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(conv_ch[3], conv_ch[3], (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(conv_ch[3], 1, (3, 1), 1, padding=(1, 0)))

    @property
    def input_types(self):
        return {
            "x": NeuralType(('B', 'S', 'T'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "decision": NeuralType(('B', 'T'), VoidType()),
            "feature_maps": [NeuralType(("B", "C", "H", "W"), VoidType())],
        }

    @typecheck()
    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(NeuralModule):
    def __init__(self, debug=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2, debug=debug),
                DiscriminatorP(3, debug=debug),
                DiscriminatorP(5, debug=debug),
                DiscriminatorP(7, debug=debug),
                DiscriminatorP(11, debug=debug),
            ]
        )

    @property
    def input_types(self):
        return {
            "y": NeuralType(('B', 'S', 'T'), AudioSignal()),
            "y_hat": NeuralType(('B', 'S', 'T'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "real_scores": [NeuralType(('B', 'T'), VoidType())],
            "fake_scores": [NeuralType(('B', 'T'), VoidType())],
            "real_feature_maps": [[NeuralType(("B", "C", "H", "W"), VoidType())]],
            "fake_feature_maps": [[NeuralType(("B", "C", "H", "W"), VoidType())]],
        }

    @typecheck()
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(NeuralModule):
    def __init__(self, use_spectral_norm=False, debug=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        conv_ch = [128, 256, 512, 1024] if not debug else [16, 32, 32, 64]
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, conv_ch[0], 15, 1, padding=7)),
                norm_f(Conv1d(conv_ch[0], conv_ch[0], 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(conv_ch[0], conv_ch[1], 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[1], conv_ch[2], 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[2], conv_ch[3], 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[3], conv_ch[3], 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[3], conv_ch[3], 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(conv_ch[3], 1, 3, 1, padding=1))

    @property
    def input_types(self):
        return {
            "x": NeuralType(('B', 'S', 'T'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "decision": NeuralType(('B', 'T'), VoidType()),
            "feature_maps": [NeuralType(("B", "C", "T"), VoidType())],
        }

    @typecheck()
    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(NeuralModule):
    def __init__(self, debug=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True, debug=debug),
                DiscriminatorS(debug=debug),
                DiscriminatorS(debug=debug),
            ]
        )
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    @property
    def input_types(self):
        return {
            "y": NeuralType(('B', 'S', 'T'), AudioSignal()),
            "y_hat": NeuralType(('B', 'S', 'T'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "real_scores": [NeuralType(('B', 'T'), VoidType())],
            "fake_scores": [NeuralType(('B', 'T'), VoidType())],
            "real_feature_maps": [[NeuralType(("B", "C", "T"), VoidType())]],
            "fake_feature_maps": [[NeuralType(("B", "C", "T"), VoidType())]],
        }

    @typecheck()
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
