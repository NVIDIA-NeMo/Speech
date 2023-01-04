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

# BSD 3-Clause License
#
# Copyright (c) 2019, Seungwon Park 박승원
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The following functions/classes were based on code from https://github.com/mindslab-ai/univnet:
# Generator, DiscriminatorP, MultiPeriodDiscriminator, DiscriminatorR, MultiScaleDiscriminator,
# KernelPredictor, LVCBlock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types.elements import AudioSignal, MelSpectrogramType, VoidType
from nemo.core.neural_types.neural_type import NeuralType


class KernelPredictor(torch.nn.Module):
    ''' Kernel predictor for the location-variable convolutions'''

    def __init__(
        self,
        cond_channels,
        conv_in_channels,
        conv_out_channels,
        conv_layers,
        conv_kernel_size=3,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        kpnet_dropout=0.0,
        kpnet_nonlinear_activation="LeakyReLU",
        kpnet_nonlinear_activation_params={"negative_slope": 0.1},
    ):
        '''
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int): number of layers
        '''
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        kpnet_kernel_channels = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers  # l_w
        kpnet_bias_channels = conv_out_channels * conv_layers  # l_b

        self.input_conv = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=2, bias=True)),
            getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )

        self.residual_convs = nn.ModuleList()
        padding = (kpnet_conv_size - 1) // 2
        for _ in range(3):
            self.residual_convs.append(
                nn.Sequential(
                    nn.Dropout(kpnet_dropout),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True
                        )
                    ),
                    getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True
                        )
                    ),
                    getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
                )
            )
        self.kernel_conv = nn.utils.weight_norm(
            nn.Conv1d(kpnet_hidden_channels, kpnet_kernel_channels, kpnet_conv_size, padding=padding, bias=True)
        )
        self.bias_conv = nn.utils.weight_norm(
            nn.Conv1d(kpnet_hidden_channels, kpnet_bias_channels, kpnet_conv_size, padding=padding, bias=True)
        )

    def forward(self, c):
        '''
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        '''
        batch, _, cond_length = c.shape
        c = self.input_conv(c)
        for residual_conv in self.residual_convs:
            residual_conv.to(c.device)
            c = c + residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = k.contiguous().view(
            batch, self.conv_layers, self.conv_in_channels, self.conv_out_channels, self.conv_kernel_size, cond_length,
        )
        bias = b.contiguous().view(batch, self.conv_layers, self.conv_out_channels, cond_length,)

        return kernels, bias

    def remove_weight_norm(self):
        remove_weight_norm(self.input_conv[0])
        remove_weight_norm(self.kernel_conv)
        remove_weight_norm(self.bias_conv)
        for block in self.residual_convs:
            remove_weight_norm(block[1])
            remove_weight_norm(block[3])


class LVCBlock(torch.nn.Module):
    '''the location-variable convolutions'''

    def __init__(
        self,
        in_channels,
        cond_channels,
        stride,
        dilations=[1, 3, 9, 27],
        lReLU_slope=0.2,
        conv_kernel_size=3,
        cond_hop_length=256,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        kpnet_dropout=0.0,
    ):
        super().__init__()

        self.cond_hop_length = cond_hop_length
        self.conv_layers = len(dilations)
        self.conv_kernel_size = conv_kernel_size

        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=len(dilations),
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout,
            kpnet_nonlinear_activation_params={"negative_slope": lReLU_slope},
        )

        self.convt_pre = nn.Sequential(
            nn.LeakyReLU(lReLU_slope),
            nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    in_channels,
                    in_channels,
                    2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                    output_padding=stride % 2,
                )
            ),
        )

        self.conv_blocks = nn.ModuleList()
        for dilation in dilations:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(lReLU_slope),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels,
                            in_channels,
                            conv_kernel_size,
                            padding=dilation * (conv_kernel_size - 1) // 2,
                            dilation=dilation,
                        )
                    ),
                    nn.LeakyReLU(lReLU_slope),
                )
            )

    def forward(self, x, c):
        ''' forward propagation of the location-variable convolutions.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length)
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)

        Returns:
            Tensor: the output sequence (batch, in_channels, in_length)
        '''
        _, in_channels, _ = x.shape  # (B, c_g, L')

        x = self.convt_pre(x)  # (B, c_g, stride * L')
        kernels, bias = self.kernel_predictor(c)

        for i, conv in enumerate(self.conv_blocks):
            output = conv(x)  # (B, c_g, stride * L')

            k = kernels[:, i, :, :, :, :]  # (B, 2 * c_g, c_g, kernel_size, cond_length)
            b = bias[:, i, :, :]  # (B, 2 * c_g, cond_length)

            output = self.location_variable_convolution(
                output, k, b, hop_size=self.cond_hop_length
            )  # (B, 2 * c_g, stride * L'): LVC
            x = x + torch.sigmoid(output[:, :in_channels, :]) * torch.tanh(
                output[:, in_channels:, :]
            )  # (B, c_g, stride * L'): GAU

        return x

    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):
        ''' perform location-variable convolution operation on the input sequence (x) using the local convolution kernel
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length).
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length)
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length)
            dilation (int): the dilation of convolution.
            hop_size (int): the hop_size of the conditioning sequence.
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, _, in_length = x.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == (kernel_length * hop_size), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), 'constant', 0)  # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)  # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(
            3, dilation, dilation
        )  # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        x = x.unfold(4, kernel_size, 1)  # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        o = o + bias
        o = o.contiguous().view(batch, out_channels, -1)

        return o

    def remove_weight_norm(self):
        self.kernel_predictor.remove_weight_norm()
        remove_weight_norm(self.convt_pre[1])
        for block in self.conv_blocks:
            remove_weight_norm(block[1])


class Generator(NeuralModule):
    __constants__ = ['lrelu_slope', 'num_kernels', 'num_upsamples']

    def __init__(
        self,
        noise_dim,
        channel_size,
        dilations,
        strides,
        lrelu_slope,
        kpnet_conv_size,
        n_mel_channels=80,
        hop_length=256,
    ):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.channel_size = channel_size
        self.dilations = dilations
        self.strides = strides
        self.lrelu_slope = lrelu_slope
        self.kpnet_conv_size = kpnet_conv_size
        self.mel_channel = n_mel_channels
        self.hop_length = hop_length

        self.res_stack = nn.ModuleList()
        hop_length_lvc = 1
        for stride in self.strides:
            hop_length_lvc = stride * hop_length_lvc
            self.res_stack.append(
                LVCBlock(
                    self.channel_size,
                    self.mel_channel,
                    stride=stride,
                    dilations=self.dilations,
                    lReLU_slope=self.lrelu_slope,
                    cond_hop_length=hop_length_lvc,
                    kpnet_conv_size=self.kpnet_conv_size,
                )
            )

        assert (
            hop_length_lvc == self.hop_length
        ), "multiplied value of strides {} should match n_window_stride {}".format(self.strides, self.hop_length)

        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(self.noise_dim, self.channel_size, 7, padding=3, padding_mode='reflect')
        )

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(self.lrelu_slope),
            nn.utils.weight_norm(nn.Conv1d(self.channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

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
        # UnivNet starts with Gaussian noise
        z = torch.randn(x.size(0), self.noise_dim, x.size(2), dtype=x.dtype, device=x.device)
        z = self.conv_pre(z)  # (B, c_g, L)

        for res_block in self.res_stack:
            z = res_block(z, x)  # (B, c_g, L * s_0 * ... * s_i)

        z = self.conv_post(z)  # (B, 1, L * 256)

        return z

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                remove_weight_norm(layer)
        for res_block in self.res_stack:
            res_block.remove_weight_norm()


class DiscriminatorP(NeuralModule):
    def __init__(self, lrelu_slope, period, kernel_size=5, stride=3, use_spectral_norm=False, debug=False):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        conv_ch = [64, 128, 256, 512, 1024] if not debug else [8, 12, 32, 64, 128]
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, conv_ch[0], (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                norm_f(Conv2d(conv_ch[0], conv_ch[1], (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                norm_f(Conv2d(conv_ch[1], conv_ch[2], (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                norm_f(Conv2d(conv_ch[2], conv_ch[3], (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                norm_f(Conv2d(conv_ch[3], conv_ch[4], (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(conv_ch[4], 1, (3, 1), 1, padding=(1, 0)))

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
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(NeuralModule):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.lrelu_slope = cfg.lrelu_slope
        self.periods = cfg.periods
        assert len(self.periods) == 5, "MPD requires list of len=5, got {}".format(cfg.periods)
        self.kernel_size = cfg.kernel_size
        self.stride = cfg.stride
        self.use_spectral_norm = cfg.use_spectral_norm

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(
                    self.lrelu_slope,
                    self.periods[0],
                    self.kernel_size,
                    self.stride,
                    self.use_spectral_norm,
                    debug=debug,
                ),
                DiscriminatorP(
                    self.lrelu_slope,
                    self.periods[1],
                    self.kernel_size,
                    self.stride,
                    self.use_spectral_norm,
                    debug=debug,
                ),
                DiscriminatorP(
                    self.lrelu_slope,
                    self.periods[2],
                    self.kernel_size,
                    self.stride,
                    self.use_spectral_norm,
                    debug=debug,
                ),
                DiscriminatorP(
                    self.lrelu_slope,
                    self.periods[3],
                    self.kernel_size,
                    self.stride,
                    self.use_spectral_norm,
                    debug=debug,
                ),
                DiscriminatorP(
                    self.lrelu_slope,
                    self.periods[4],
                    self.kernel_size,
                    self.stride,
                    self.use_spectral_norm,
                    debug=debug,
                ),
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


class DiscriminatorR(NeuralModule):
    def __init__(self, cfg, resolution):
        super().__init__()

        self.resolution = resolution
        assert len(self.resolution) == 3, "MRD layer requires list with len=3, got {}".format(self.resolution)
        self.lrelu_slope = cfg.lrelu_slope

        norm_f = weight_norm if cfg.use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

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

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.view_as_real(
            torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        )  # [B, F, TT, 2] (Note: torch.stft() returns complex tensor [B, F, TT]; converted via view_as_real)
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(NeuralModule):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.resolutions
        assert (
            len(self.resolutions) == 3
        ), "MRD requires list of list with len=3, each element having a list with len=3. got {}".format(
            self.resolutions
        )
        self.discriminators = nn.ModuleList([DiscriminatorR(cfg, resolution) for resolution in self.resolutions])

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
