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


class SEANetResnetBlock(NeuralModule):
    def __init__(self, channels: int):
        super().__init__()
        self.activation = nn.ELU()
        hidden_channels = channels // 2
        self.pre_conv = Conv1dNorm(in_channels=channels, out_channels=channels, kernel_size=1)
        self.res_conv1 = Conv1dNorm(in_channels=channels, out_channels=hidden_channels, kernel_size=3)
        self.res_conv2 = Conv1dNorm(in_channels=hidden_channels, out_channels=channels, kernel_size=1)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T_input'), VoidType()),
            "lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": [NeuralType(('B', 'C', 'T_out'), VoidType())],
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.res_conv1.remove_weight_norm()
        self.res_conv2.remove_weight_norm()

    def forward(self, inputs, lengths):
        res = self.activation(inputs)
        res = self.res_conv1(res, lengths)
        res = self.activation(res)
        res = self.res_conv2(res, lengths)

        out = self.pre_conv(inputs, lengths) + res
        out = mask_sequence_tensor(out, lengths)
        return out


class SEANetRNN(NeuralModule):
    def __init__(self, dim: int, num_layers: int, rnn_type: str = "lstm", use_skip: bool = False):
        super().__init__()
        self.use_skip = use_skip
        if rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(input_size=dim, hidden_size=dim, num_layers=num_layers)
        elif rnn_type == "gru":
            self.rnn = torch.nn.GRU(input_size=dim, hidden_size=dim, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown RNN type {rnn_type}")

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

    def forward(self, inputs, lengths):
        inputs = rearrange(inputs, "B C T -> B T C")

        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed_inputs)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        if self.use_skip:
            out = out + inputs

        out = rearrange(out, "B T C -> B C T")
        return out


class SEANetEncoder(NeuralModule):
    def __init__(
        self,
        down_sample_rates: Iterable[int] = (2, 4, 5, 8),
        base_channels: int = 32,
        in_kernel_size: int = 7,
        out_kernel_size: int = 7,
        encoded_dim: int = 128,
        rnn_layers: int = 2,
        rnn_type: str = "lstm",
        rnn_skip: bool = True,
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.down_sample_rates = down_sample_rates
        self.activation = nn.ELU()
        self.pre_conv = Conv1dNorm(in_channels=1, out_channels=base_channels, kernel_size=in_kernel_size)

        in_channels = base_channels
        self.res_blocks = nn.ModuleList([])
        self.down_sample_conv_layers = nn.ModuleList([])
        for i, down_sample_rate in enumerate(self.down_sample_rates):
            res_block = SEANetResnetBlock(channels=in_channels)
            self.res_blocks.append(res_block)

            out_channels = 2 * in_channels
            kernel_size = 2 * down_sample_rate
            down_sample_conv = Conv1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=down_sample_rate,
                padding=get_down_sample_padding(kernel_size, down_sample_rate),
            )
            in_channels = out_channels
            self.down_sample_conv_layers.append(down_sample_conv)

        self.rnn = SEANetRNN(dim=in_channels, num_layers=rnn_layers, rnn_type=rnn_type, use_skip=rnn_skip)
        self.post_conv = Conv1dNorm(in_channels=in_channels, out_channels=encoded_dim, kernel_size=out_kernel_size)

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'C', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": [NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation())],
            "encoded_len": [NeuralType(tuple('B'), LengthsType())],
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()
        for down_sample_conv in self.down_sample_conv_layers:
            down_sample_conv.remove_weight_norm()

    def forward(self, audio, audio_len):
        encoded_len = audio_len
        audio = rearrange(audio, "B T -> B 1 T")
        # [B, C, T_audio]
        out = self.pre_conv(audio, encoded_len)
        for res_block, down_sample_conv, down_sample_rate in zip(
            self.res_blocks, self.down_sample_conv_layers, self.down_sample_rates
        ):
            # [B, C, T]
            out = res_block(out, encoded_len)
            out = self.activation(out)

            encoded_len = encoded_len // down_sample_rate
            # [B, 2 * C, T / down_sample_rate]
            out = down_sample_conv(out, encoded_len)

        out = self.rnn(out, encoded_len)
        out = self.activation(out)
        # [B, encoded_dim, T_encoded]
        encoded = self.post_conv(out, encoded_len)
        return encoded, encoded_len


class SEANetDecoder(NeuralModule):
    def __init__(
        self,
        up_sample_rates: Iterable[int] = (8, 5, 4, 2),
        base_channels: int = 512,
        in_kernel_size: int = 7,
        out_kernel_size: int = 3,
        encoded_dim: int = 128,
        rnn_layers: int = 2,
        rnn_type: str = "lstm",
        rnn_skip: bool = True,
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.up_sample_rates = up_sample_rates
        self.activation = nn.ELU()
        self.pre_conv = Conv1dNorm(in_channels=encoded_dim, out_channels=base_channels, kernel_size=in_kernel_size)
        self.rnn = SEANetRNN(dim=base_channels, num_layers=rnn_layers, rnn_type=rnn_type, use_skip=rnn_skip)

        in_channels = base_channels
        self.res_blocks = nn.ModuleList([])
        self.up_sample_conv_layers = nn.ModuleList([])
        for i, up_sample_rate in enumerate(self.up_sample_rates):
            out_channels = in_channels // 2
            kernel_size = 2 * up_sample_rate
            up_sample_conv = ConvTranspose1dNorm(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=up_sample_rate
            )
            in_channels = out_channels
            self.up_sample_conv_layers.append(up_sample_conv)

            res_block = SEANetResnetBlock(channels=in_channels)
            self.res_blocks.append(res_block)

        self.post_conv = Conv1dNorm(in_channels=in_channels, out_channels=1, kernel_size=out_kernel_size)
        self.out_activation = nn.Tanh()

    @property
    def input_types(self):
        return {
            "inputs": [NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation())],
            "input_len": [NeuralType(tuple('B'), LengthsType())],
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'C', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        for up_sample_conv in self.up_sample_conv_layers:
            up_sample_conv.remove_weight_norm()
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    def forward(self, inputs, input_len):
        audio_len = input_len
        # [B, C, T_encoded]
        out = self.pre_conv(inputs, audio_len)
        out = self.rnn(out, audio_len)
        for res_block, up_sample_conv, up_sample_rate in zip(
            self.res_blocks, self.up_sample_conv_layers, self.up_sample_rates
        ):
            audio_len *= up_sample_rate
            out = self.activation(out)
            # [B, C / 2, T * up_sample_rate]
            out = up_sample_conv(out, audio_len)
            out = res_block(out, audio_len)

        out = self.activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(out, audio_len)
        audio = self.out_activation(out)
        audio = rearrange(audio, "B 1 T -> B T")
        return audio, audio_len


class DiscriminatorSTFT(NeuralModule):
    def __init__(self, resolution, lrelu_slope=0.1):
        super().__init__()

        self.n_fft, self.hop_length, self.win_length = resolution
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.activation = nn.LeakyReLU(lrelu_slope)

        self.conv_layers = nn.ModuleList(
            [
                Conv2dNorm(2, 32, kernel_size=(3, 9)),
                Conv2dNorm(32, 32, kernel_size=(3, 9), dilation=(1, 1), stride=(1, 2)),
                Conv2dNorm(32, 32, kernel_size=(3, 9), dilation=(2, 1), stride=(1, 2)),
                Conv2dNorm(32, 32, kernel_size=(3, 9), dilation=(4, 1), stride=(1, 2)),
                Conv2dNorm(32, 32, kernel_size=(3, 3)),
            ]
        )
        self.conv_post = Conv2dNorm(32, 1, kernel_size=(3, 3))

    def stft(self, audio):
        # [B, fft, T_spec]
        out = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            return_complex=True,
        )
        out = rearrange(out, "B fft T -> B 1 T fft")
        # [batch, 2, T_spec, fft]
        out = torch.cat([out.real, out.imag], dim=1)
        return out

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores": NeuralType(('B', 'C', 'T_spec'), VoidType()),
            "fmap": [NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())],
        }

    def forward(self, audio):
        fmap = []

        # [batch, 2, T_spec, fft]
        out = self.stft(audio)
        for conv in self.conv_layers:
            # [batch, filters, T_spec, fft // 2**i]
            out = conv(out)
            out = self.activation(out)
            fmap.append(out)
        # [batch, 1, T_spec, fft // 8]
        scores = self.conv_post(out)
        fmap.append(scores)
        scores = rearrange(scores, "B 1 T C -> B C T")

        return scores, fmap


class MultiResolutionDiscriminatorSTFT(NeuralModule):
    def __init__(self, resolutions):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorSTFT(res) for res in resolutions])

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_spec'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_spec'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())]],
        }

    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []

        for disc in self.discriminators:
            score_real, fmap_real = disc(audio=audio_real)
            scores_real.append(score_real)
            fmaps_real.append(fmap_real)

            score_gen, fmap_gen = disc(audio=audio_gen)
            scores_gen.append(score_gen)
            fmaps_gen.append(fmap_gen)

        return scores_real, scores_gen, fmaps_real, fmaps_gen
