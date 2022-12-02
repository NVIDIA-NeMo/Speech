# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#
from typing import List

import torch
from numpy import inf
from torch import nn as nn
from torch.nn import functional as F

from nemo.collections.asr.parts.submodules.jasper import get_same_padding, init_weights


class StatsPoolLayer(nn.Module):
    """Statistics and time average pooling (TAP) layer

    This computes mean and, optionally, standard deviation statistics across the time dimension.

    Args:
        feat_in: Input features with shape [B, D, T]
        pool_mode: Type of pool mode. Supported modes are 'xvector' (mean and standard deviation) and 'tap' (time
            average pooling, i.e., mean)
        eps: Epsilon, minimum value before taking the square root, when using 'xvector' mode.
        biased: Whether to use the biased estimator for the standard deviation when using 'xvector' mode. The default
            for torch.Tensor.std() is True.

    Returns:
        Pooled statistics with shape [B, D].

    Raises:
        ValueError if an unsupported pooling mode is specified.
    """

    def __init__(self, feat_in: int, pool_mode: str = 'xvector', eps: float = 1e-10, biased: bool = True):
        super().__init__()
        supported_modes = {"xvector", "tap"}
        if pool_mode not in supported_modes:
            raise ValueError(f"Pool mode must be one of {supported_modes}; got '{pool_mode}'")
        self.pool_mode = pool_mode
        self.feat_in = feat_in
        self.eps = eps
        self.biased = biased
        if self.pool_mode == 'xvector':
            # Mean + std
            self.feat_in *= 2

    def forward(self, encoder_output, length=None):
        if length is None:
            mean = encoder_output.mean(dim=-1)  # Time Axis
            if self.pool_mode == 'xvector':
                std = encoder_output.std(dim=-1)
                pooled = torch.cat([mean, std], dim=-1)
            else:
                pooled = mean
        else:
            mask = make_seq_mask_like(like=encoder_output, lengths=length, valid_ones=False)
            encoder_output = encoder_output.masked_fill(mask, 0.0)
            # [B, D, T] -> [B, D]
            means = encoder_output.mean(dim=-1)
            # Re-scale to get padded means
            means = means * (encoder_output.shape[-1] / length).unsqueeze(-1)
            if self.pool_mode == "xvector":
                stds = (
                    encoder_output.sub(means.unsqueeze(-1))
                    .masked_fill(mask, 0.0)
                    .pow(2.0)
                    .sum(-1)  # [B, D, T] -> [B, D]
                    .div(length.view(-1, 1).sub(1 if self.biased else 0))
                    .clamp(min=self.eps)
                    .sqrt()
                )
                pooled = torch.cat((means, stds), dim=-1)
            else:
                pooled = means
        return pooled


@torch.jit.script_if_tracing
def make_seq_mask_like(
    like: torch.Tensor, lengths: torch.Tensor, valid_ones: bool = True, time_dim: int = -1
) -> torch.Tensor:
    mask = torch.arange(like.shape[time_dim], device=like.device).repeat(lengths.shape[0], 1).lt(lengths.unsqueeze(-1))
    # Match number of dims in `like` tensor
    for _ in range(like.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    # If time dim != -1, transpose to proper dim.
    if time_dim != -1:
        mask = mask.transpose(time_dim, -1)
    if not valid_ones:
        mask = ~mask
    return mask


def lens_to_mask(lens: List[int], max_len: int, device: str = None):
    """
    outputs masking labels for list of lengths of audio features, with max length of any 
    mask as max_len
    input:
        lens: list of lens
        max_len: max length of any audio feature
    output:
        mask: masked labels
        num_values: sum of mask values for each feature (useful for computing statistics later)
    """
    lens_mat = torch.arange(max_len).to(device)
    mask = lens_mat[:max_len].unsqueeze(0) < lens.unsqueeze(1)
    mask = mask.unsqueeze(1)
    num_values = torch.sum(mask, dim=2, keepdim=True)
    return mask, num_values


def get_statistics_with_mask(x: torch.Tensor, m: torch.Tensor, dim: int = 2, eps: float = 1e-10):
    """
    compute mean and standard deviation of input(x) provided with its masking labels (m)
    input:
        x: feature input 
        m: averaged mask labels 
    output:
        mean: mean of input features
        std: stadard deviation of input features
    """
    mean = torch.sum((m * x), dim=dim)
    std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
    return mean, std


class TDNNModule(nn.Module):
    """
    Time Delayed Neural Module (TDNN) - 1D
    input:
        inp_filters: input filter channels for conv layer
        out_filters: output filter channels for conv layer
        kernel_size: kernel weight size for conv layer
        dilation: dilation for conv layer
        stride: stride for conv layer
        padding: padding for conv layer (default None: chooses padding value such that input and output feature shape matches)
    output:
        tdnn layer output 
    """

    def __init__(
        self,
        inp_filters: int,
        out_filters: int,
        kernel_size: int = 1,
        dilation: int = 1,
        stride: int = 1,
        padding: int = None,
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size, stride=stride, dilation=dilation)

        self.conv_layer = nn.Conv1d(
            in_channels=inp_filters,
            out_channels=out_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_filters)

    def forward(self, x, length=None):
        x = self.conv_layer(x)
        x = self.activation(x)
        return self.bn(x)


class MaskedSEModule(nn.Module):
    """
    Squeeze and Excite module implementation with conv1d layers
    input:
        inp_filters: input filter channel size 
        se_filters: intermediate squeeze and excite channel output and input size
        out_filters: output filter channel size
        kernel_size: kernel_size for both conv1d layers
        dilation: dilation size for both conv1d layers

    output:
        squeeze and excite layer output
    """

    def __init__(self, inp_filters: int, se_filters: int, out_filters: int, kernel_size: int = 1, dilation: int = 1):
        super().__init__()
        self.se_layer = nn.Sequential(
            nn.Conv1d(inp_filters, se_filters, kernel_size=kernel_size, dilation=dilation,),
            nn.ReLU(),
            nn.BatchNorm1d(se_filters),
            nn.Conv1d(se_filters, out_filters, kernel_size=kernel_size, dilation=dilation,),
            nn.Sigmoid(),
        )

    def forward(self, input, length=None):
        if length is None:
            x = torch.mean(input, dim=2, keep_dim=True)
        else:
            max_len = input.size(2)
            mask, num_values = lens_to_mask(length, max_len=max_len, device=input.device)
            x = torch.sum((input * mask), dim=2, keepdim=True) / (num_values)

        out = self.se_layer(x)
        return out * input


class TDNNSEModule(nn.Module):
    """
    Modified building SE_TDNN group module block from ECAPA implementation for faster training and inference
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)
    inputs:
        inp_filters: input filter channel size 
        out_filters: output filter channel size
        group_scale: scale value to group wider conv channels (deafult:8)
        se_channels: squeeze and excite output channel size (deafult: 1024/8= 128)
        kernel_size: kernel_size for group conv1d layers (default: 1)
        dilation: dilation size for group conv1d layers  (default: 1)
    """

    def __init__(
        self,
        inp_filters: int,
        out_filters: int,
        group_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        init_mode: str = 'xavier_uniform',
    ):
        super().__init__()
        self.out_filters = out_filters
        padding_val = get_same_padding(kernel_size=kernel_size, dilation=dilation, stride=1)

        group_conv = nn.Conv1d(
            out_filters,
            out_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding_val,
            groups=group_scale,
        )
        self.group_tdnn_block = nn.Sequential(
            TDNNModule(inp_filters, out_filters, kernel_size=1, dilation=1),
            group_conv,
            nn.ReLU(),
            nn.BatchNorm1d(out_filters),
            TDNNModule(out_filters, out_filters, kernel_size=1, dilation=1),
        )

        self.se_layer = MaskedSEModule(out_filters, se_channels, out_filters)

        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, input, length=None):
        x = self.group_tdnn_block(input)
        x = self.se_layer(x, length)
        return x + input


class AttentivePoolLayer(nn.Module):
    """
    Attention pooling layer for pooling speaker embeddings
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)
    inputs:
        inp_filters: input feature channel length from encoder
        attention_channels: intermediate attention channel size
        kernel_size: kernel_size for TDNN and attention conv1d layers (default: 1)
        dilation: dilation size for TDNN and attention conv1d layers  (default: 1) 
    """

    def __init__(
        self,
        inp_filters: int,
        attention_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        eps: float = 1e-10,
    ):
        super().__init__()

        self.feat_in = 2 * inp_filters

        self.attention_layer = nn.Sequential(
            TDNNModule(inp_filters * 3, attention_channels, kernel_size=kernel_size, dilation=dilation),
            nn.Tanh(),
            nn.Conv1d(
                in_channels=attention_channels, out_channels=inp_filters, kernel_size=kernel_size, dilation=dilation,
            ),
        )
        self.eps = eps

    def forward(self, x, length=None):
        max_len = x.size(2)

        if length is None:
            length = torch.ones(x.shape[0], device=x.device)

        mask, num_values = lens_to_mask(length, max_len=max_len, device=x.device)

        # encoder statistics
        mean, std = get_statistics_with_mask(x, mask / num_values)
        mean = mean.unsqueeze(2).repeat(1, 1, max_len)
        std = std.unsqueeze(2).repeat(1, 1, max_len)
        attn = torch.cat([x, mean, std], dim=1)

        # attention statistics
        attn = self.attention_layer(attn)  # attention pass
        attn = attn.masked_fill(mask == 0, -inf)
        alpha = F.softmax(attn, dim=2)  # attention values, α
        mu, sg = get_statistics_with_mask(x, alpha)  # µ and ∑

        # gather
        return torch.cat((mu, sg), dim=1).unsqueeze(2)
