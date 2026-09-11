# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D, CausalConv2D
from nemo.core.utils.optional_libs import TRITON_AVAILABLE, triton_required
from nemo.utils import logging

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.triton.depthwise_conv import dw_conv2d
    from nemo.collections.asr.parts.triton.subsampling import fused_conv_relu_dw


class FeatureStacking(nn.Module):
    """Stacks consecutive input frames and projects to model dimension.

    Reduces the temporal resolution by ``subsampling_factor`` while increasing
    the feature dimension proportionally, then linearly projects back to
    ``feat_out``.

    Args:
        subsampling_factor: Number of consecutive frames to stack.
        feat_in: Input feature dimension.
        feat_out: Output feature dimension.
    """

    def __init__(self, subsampling_factor: int, feat_in: int, feat_out: int):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj = nn.Linear(subsampling_factor * feat_in, feat_out, bias=False)

    def compute_num_out_frames(self, in_frames):
        return (in_frames + self.subsampling_factor - 1) // self.subsampling_factor

    def get_sampling_frames(self):
        """Input frames consumed per output frame — probed by the cache-aware streaming utils."""
        return self.subsampling_factor

    def get_streaming_cache_size(self):
        """Input-frame look-back needed to reproduce the offline output. Stacking is
        non-overlapping, so a chunk aligned to ``subsampling_factor`` needs none."""
        return 0

    def forward(self, x, lengths):
        """
        Args:
            x: (B, C, T) input features.
            lengths: (B,) valid lengths per sample.
        Returns:
            x: (B, T', feat_out) stacked and projected features.
            lengths: (B,) updated lengths after subsampling.
        """
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        b, t, c = x.size()
        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        if pad_size > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_size))
        t_new = (t + pad_size) // self.subsampling_factor
        x = x.reshape(b, t_new, c * self.subsampling_factor)
        x = self.proj(x)
        lengths = self.compute_num_out_frames(lengths)
        return x, lengths


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
        subsampling (str): The subsampling technique from {"vggnet", "striding", "dw-striding"}
        subsampling_factor (int): The subsampling factor which should be a power of 2
        subsampling_conv_chunking_factor (int): Input chunking factor which can be -1 (no chunking)
        1 (auto) or a power of 2. Default is 1
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self,
        subsampling,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
        use_triton: bool | None = None,
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

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

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

        elif subsampling == 'striding_conv1d':

            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
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
                        CausalConv1D(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'dw_striding_conv1d':

            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2

            # Layer 1
            layers.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        groups=in_channels,
                    ),
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=feat_out if self._sampling_num == 1 else conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ),
                ]
            )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                layers.extend(
                    [
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        ),
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 2 else conv_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                        ),
                    ]
                )
                layers.append(activation)
                in_channels = conv_channels

        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        if subsampling in ["vggnet", "dw_striding", "striding"]:

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
            self.conv2d_subsampling = True
        elif subsampling in ["striding_conv1d", "dw_striding_conv1d"]:
            self.out = None
            self.conv2d_subsampling = False
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        self.conv = MaskedConvSequential(*layers)

        # The kernels implement `dw_striding`'s layout, [conv, act] + (sampling_num - 1) x
        # [dw, pw, act], with ReLU baked in; a factor of 2 stops after [conv, act], leaving no
        # depthwise to fuse.
        supported = subsampling == 'dw_striding' and self._sampling_num >= 2 and isinstance(activation, nn.ReLU)
        if use_triton and not supported:
            logging.warning(
                "use_triton=True was requested, but the fused kernels only cover dw_striding with "
                "subsampling_factor >= 4 and a ReLU activation, falling back to PyTorch instead."
            )
        self.conv.fuse_triton = supported and (TRITON_AVAILABLE if use_triton is None else use_triton)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x, lengths):
        out_lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        # Transpose to Channel First mode
        if not self.conv2d_subsampling:
            x = x.transpose(1, 2)

        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                # if subsampling_conv_chunking_factor is 1, we split only if needed
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
                if torch.numel(x) > x_ceil:
                    need_to_split = True
                else:
                    need_to_split = False
            else:
                # if subsampling_conv_chunking_factor > 1 we always split
                need_to_split = True

            if need_to_split:
                x, lengths, success = self.conv_split_by_batch(x, lengths)
                if not success:  # if unable to split by batch, try by channel
                    if self._subsampling == 'dw_striding':
                        # TODO: implement lengths inside conv_split_by_channel
                        x = self.conv_split_by_channel(x)
                        lengths = out_lengths
                    else:
                        x, lengths = self.conv(x, lengths)  # try anyway
            else:
                x, lengths = self.conv(x, lengths)
        else:
            x, lengths = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)

        return x, lengths

    def reset_parameters(self):
        # initialize weights
        if self._subsampling == 'dw_striding':
            with torch.no_grad():
                # init conv
                scale = 1.0 / self._kernel_size
                dw_max = (self._kernel_size**2) ** -0.5
                pw_max = self._conv_channels**-0.5

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

    def conv_split_by_batch(self, x, lengths):
        """Tries to split input by batch, run conv and concat results"""
        b, *_ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, lengths, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
            logging.debug(f'using manually set chunking factor: {cf}')
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2**p
            logging.debug(f'using auto set chunking factor: {cf}')

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, lengths, False

        logging.debug(f'conv subsampling: using split batch size {new_batch_size}')

        ans = [
            self.conv(chunk, ln)
            for chunk, ln in zip(
                torch.split(x, new_batch_size, 0),
                torch.split(lengths, new_batch_size, 0),
            )
        ]
        return torch.cat([a[0] for a in ans]), torch.cat([a[1] for a in ans]), True

    def conv_split_by_channel(self, x):
        """For dw convs, tries to split input by time, run conv and concat results"""

        # Note: this method doesn't use the convolution masking implemented in MaskedConvolutionSequential
        x = x.unsqueeze(0)
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
                logging.debug(f'using manually set chunking factor: {cf}')
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(torch.numel(x) / 2**31, 2))
                cf = 2**p
                logging.debug(f'using auto set chunking factor: {cf}')

            new_c = int(c // cf)
            if new_c == 0:
                logging.warning(f'chunking factor {cf} is too high; splitting down to one channel.')
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                logging.warning(f'chunking factor {cf} is too high; splitting down to one timestep.')
                new_t = 1

            logging.debug(f'conv dw subsampling: using split C size {new_c} and split T size {new_t}')
            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c, x)  # conv2D, depthwise

            # splitting pointwise convs by time
            x = torch.cat([self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_t, 2)], 2)  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk, pad=(self._kernel_size - 1, self._stride - 1, self._kernel_size - 1, self._stride - 1)
                )
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


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


def apply_channel_mask(tensor, mask):
    """Apply mask to tensor with channel dimension."""
    # tensor: (batch, channels, time, features)
    # mask: (batch, time, features)
    batch_size, channels, time, features = tensor.shape
    expanded_mask = mask.unsqueeze(1).expand(batch_size, channels, time, features)
    return tensor * expanded_mask


def calculate_conv_output_size(input_size: torch.Tensor, kernel_size: int, stride: int, padding: tuple[int, int]):
    """Calculate exact output size after convolution."""
    return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


class MaskedConvSequential(nn.Sequential):
    # Set by ConvSubsampling; off by default, so every other subsampling type stays on PyTorch.
    fuse_triton = False

    def forward(self, x, lengths):
        # Convert input (batch, time, features) to conv format
        x = x.unsqueeze(1)  # (batch, 1, time, features)
        current_lengths = lengths

        # Tracing and export cannot capture a Triton launch, the fused kernel returns no input
        # gradient, and its weight gradients accumulate through atomics, so their summation order
        # varies between runs.
        if (
            self.fuse_triton
            and x.is_cuda
            and not x.requires_grad
            and not torch.are_deterministic_algorithms_enabled()
            and not (torch.jit.is_tracing() or torch.compiler.is_exporting())
        ):
            x, current_lengths, mask = self._forward_fused(x, current_lengths)
        else:
            x, current_lengths, mask = self._forward_torch(x, current_lengths)

        # Final masking
        x = apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _forward_torch(self, x, current_lengths):
        mask = self._create_mask(x, current_lengths.long())

        # Process through each layer with mask propagation
        for i, layer in enumerate(self):
            # Apply current mask before layer
            x = apply_channel_mask(x, mask)

            # Apply layer
            x = layer(x)

            # Update lengths for stride operations with proper padding
            if hasattr(layer, 'stride') and layer.stride != (1, 1):
                current_lengths = calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], _layer_padding(layer)
                )
                mask = self._create_mask(x, current_lengths.long())

        return x, current_lengths, mask

    @triton_required
    def _forward_fused(self, x, current_lengths):
        """The `dw_striding` stack, with conv0 and the depthwise layers as Triton kernels.

        The stack is `[conv, act] + (sampling_num - 1) x [dw, pw, act]`. One kernel covers the
        leading `conv, act, dw`; the loop over `self[3:]` runs each depthwise as a kernel, each
        pointwise as a linear, and every other layer as itself. Lengths change only at the
        depthwise layers.

        Tensors are channels-last throughout, `(batch, time, freq, channels)`, and one permute at
        the end returns the `(batch, channels, time, freq)` the caller expects.

        The kernels read zeros beyond their input lengths and write zeros beyond their output
        lengths. Only the trailing pointwise and activation touch the padded tail, which
        `apply_channel_mask` clears at the end of `forward`.
        """
        conv0, _, first_depthwise, first_pointwise, activation = self[:5]
        # conv -> ReLU -> depthwise in one kernel; the intermediate never reaches memory.
        x, current_lengths = fused_conv_relu_dw(
            x,
            conv0.weight,
            conv0.bias,
            first_depthwise.weight,
            first_depthwise.bias,
            *_layer_padding(conv0),
            current_lengths,
        )
        x = _pointwise_block(x, first_pointwise, activation)

        body = self[5:]
        for i in range(0, len(body), 3):
            depthwise, pointwise, activation = body[i : i + 3]
            # The kernel masks its own output, so it needs the post-stride lengths.
            next_lengths = calculate_conv_output_size(
                current_lengths, depthwise.kernel_size[0], depthwise.stride[0], _layer_padding(depthwise)
            )
            x = dw_conv2d(
                x,
                depthwise.weight,
                depthwise.bias,
                depthwise.stride,
                *_layer_padding(depthwise),
                current_lengths,
                next_lengths,
            )
            current_lengths = next_lengths
            x = _pointwise_block(x, pointwise, activation)

        x = x.permute(0, 3, 1, 2)
        return x, current_lengths, self._create_mask(x, current_lengths.long())

    def _create_mask(self, tensor, lengths):
        """Create mask matching tensor dimensions."""
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)


def _layer_padding(layer):
    """The (start, end) padding of a convolution.

    nn.Conv2d's `.padding` is (pad_h, pad_w), one value per axis and symmetric within it, so the
    height value is both edges. CausalConv2D keeps its two edges on private attributes.
    """
    if hasattr(layer, "_left_padding"):
        return layer._left_padding, layer._right_padding
    return layer.padding[0], layer.padding[0]


def _is_depthwise(layer):
    """A depthwise convolution: one group per channel."""
    return isinstance(layer, nn.Conv2d) and layer.groups > 1


def _is_pointwise(layer):
    """A 1x1 convolution over all channels, which is a contraction over the channel axis alone."""
    return isinstance(layer, nn.Conv2d) and layer.groups == 1 and layer.kernel_size == (1, 1)


def _pointwise_block(x, conv, activation):
    # kernel_size=1 convs are pointwise, i.e. linear, but nn.Conv2d dispatches to much slower
    # cuBLAS kernels. flatten(1) on the weight is a free view, so checkpoints are unchanged.
    # F.linear on an N-D input returns a view of its 2D result. An in-place activation on a
    # view copies the whole tensor in backward, so x is flattened and the activation runs on
    # the 2D result itself.
    # TODO: remove the shape manipulation once https://github.com/pytorch/pytorch/pull/194077
    # is in the minimum required PyTorch version.
    b, t, f, c = x.shape
    x = nn.functional.linear(x.view(-1, c), conv.weight.flatten(1), conv.bias)
    return activation(x).view(b, t, f, -1)
