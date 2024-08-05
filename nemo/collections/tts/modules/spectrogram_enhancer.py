# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The following is largely based on code from https://github.com/lucidrains/stylegan2-pytorch

import math
from functools import partial
from math import log2
from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange
from kornia.filters import filter2d

from nemo.collections.common.parts.utils import mask_sequence_tensor


class Blur(torch.nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class EqualLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleMapping(torch.nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()

        layers = []
        for _ in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), torch.nn.LeakyReLU(0.2, inplace=True)])

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class RGBBlock(torch.nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, channels=3):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = torch.nn.Linear(latent_dim, input_channel)

        out_filters = channels
        self.conv = Conv2DModulated(input_channel, out_filters, 1, demod=False)

        self.upsample = (
            torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                Blur(),
            )
            if upsample
            else None
        )

    def forward(self, x, prev_rgb, istyle):
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class Conv2DModulated(torch.nn.Module):
    """
    Modulated convolution.
    For details refer to [1]
    [1] Karras et. al. - Analyzing and Improving the Image Quality of StyleGAN (https://arxiv.org/abs/1912.04958)
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        kernel,
        demod=True,
        stride=1,
        dilation=1,
        eps=1e-8,
        **kwargs,
    ):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = torch.nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        torch.nn.init.kaiming_normal_(self.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights**2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class GeneratorBlock(torch.nn.Module):
    def __init__(
        self,
        latent_dim,
        input_channels,
        filters,
        upsample=True,
        upsample_rgb=True,
        channels=1,
    ):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if upsample else None

        self.to_style1 = torch.nn.Linear(latent_dim, input_channels)
        self.to_noise1 = torch.nn.Linear(1, filters)
        self.conv1 = Conv2DModulated(input_channels, filters, 3)

        self.to_style2 = torch.nn.Linear(latent_dim, filters)
        self.to_noise2 = torch.nn.Linear(1, filters)
        self.conv2 = Conv2DModulated(filters, filters, 3)

        self.activation = torch.nn.LeakyReLU(0.2, inplace=True)
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, channels)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        inoise = inoise[:, : x.shape[2], : x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 1, 2))
        noise2 = self.to_noise2(inoise).permute((0, 3, 1, 2))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = torch.nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, filters, 3, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(filters, filters, 3, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.downsample = (
            torch.nn.Sequential(Blur(), torch.nn.Conv2d(filters, filters, 3, padding=1, stride=2))
            if downsample
            else None
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class Generator(torch.nn.Module):
    def __init__(
        self, n_bands, latent_dim, style_depth, network_capacity=16, channels=1, fmap_max=512, start_from_zero=True
    ):
        super().__init__()
        self.image_size = n_bands
        self.latent_dim = latent_dim
        self.num_layers = int(log2(n_bands) - 1)
        self.style_depth = style_depth

        self.style_mapping = StyleMapping(self.latent_dim, self.style_depth, lr_mul=0.1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])

        self.initial_conv = torch.nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = torch.nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                channels=channels,
            )
            self.blocks.append(block)

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
        for block in self.blocks:
            torch.nn.init.zeros_(block.to_noise1.weight)
            torch.nn.init.zeros_(block.to_noise1.bias)
            torch.nn.init.zeros_(block.to_noise2.weight)
            torch.nn.init.zeros_(block.to_noise2.bias)

        initial_block_size = n_bands // self.upsample_factor, 1
        self.initial_block = torch.nn.Parameter(
            torch.randn((1, init_channels, *initial_block_size)), requires_grad=False
        )
        if start_from_zero:
            self.initial_block.data.zero_()

    def add_scaled_condition(self, target: torch.Tensor, condition: torch.Tensor, condition_lengths: torch.Tensor):
        *_, target_height, _ = target.shape
        *_, height, _ = condition.shape

        scale = height // target_height

        # scale appropriately
        condition = F.interpolate(condition, size=target.shape[-2:], mode="bilinear")

        # add and mask
        result = (target + condition) / 2
        result = mask_sequence_tensor(result, (condition_lengths / scale).ceil().long())

        return result

    @property
    def upsample_factor(self):
        return 2 ** sum(1 for block in self.blocks if block.upsample)

    def forward(self, condition: torch.Tensor, lengths: torch.Tensor, ws: List[torch.Tensor], noise: torch.Tensor):
        batch_size, _, _, max_length = condition.shape

        x = self.initial_block.expand(batch_size, -1, -1, max_length // self.upsample_factor)

        rgb = None
        x = self.initial_conv(x)

        for style, block in zip(ws, self.blocks):
            x, rgb = block(x, rgb, style, noise)

            x = self.add_scaled_condition(x, condition, lengths)
            rgb = self.add_scaled_condition(rgb, condition, lengths)

        return rgb


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        n_bands,
        network_capacity=16,
        channels=1,
        fmap_max=512,
    ):
        super().__init__()
        num_layers = int(log2(n_bands) - 1)
        num_init_filters = channels

        blocks = []
        filters = [num_init_filters] + [(network_capacity * 4) * (2**i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

        self.blocks = torch.nn.ModuleList(blocks)

        channel_last = filters[-1]
        latent_dim = channel_last

        self.final_conv = torch.nn.Conv2d(channel_last, channel_last, 3, padding=1)
        self.to_logit = torch.nn.Linear(latent_dim, 1)

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x, condition: torch.Tensor, lengths: torch.Tensor):
        for block in self.blocks:
            x = block(x)
            scale = condition.shape[-1] // x.shape[-1]
            x = mask_sequence_tensor(x, (lengths / scale).ceil().long())

        x = self.final_conv(x)

        scale = condition.shape[-1] // x.shape[-1]
        x = mask_sequence_tensor(x, (lengths / scale).ceil().long())

        x = x.mean(axis=-2)
        x = (x / rearrange(lengths / scale, "b -> b 1 1")).sum(axis=-1)
        x = self.to_logit(x)
        return x.squeeze()
