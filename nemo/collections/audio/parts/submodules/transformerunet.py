# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
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
# Copyright (c) 2023 Phil Wang
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

import math
from functools import partial
from typing import Dict, Optional

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import BoolType, FloatType, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging

__all__ = ['TransformerUNet']


class LearnedSinusoidalPosEmb(Module):
    """The sinusoidal Embedding to encode time conditional information"""

    def __init__(self, dim: int):
        super().__init__()
        if (dim % 2) != 0:
            raise ValueError(f"Input dimension {dim} is not divisible by 2!")
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
          t: input time tensor, shape (B)

        Return:
          fouriered: the encoded time conditional embedding, shape (B, D)
        """
        t = einops.rearrange(t, 'b -> b 1')
        freqs = t * einops.rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class ConvPositionEmbed(Module):
    """The Convolutional Embedding to encode time information of each frame"""

    def __init__(self, dim: int, kernel_size: int, groups: Optional[int] = None):
        super().__init__()
        if (kernel_size % 2) == 0:
            raise ValueError(f"Kernel size {kernel_size} is divisible by 2!")

        if groups is None:
            groups = dim

        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2), nn.GELU()
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor, shape (B, T, D)

        Return:
            out: output tensor with the same shape (B, T, D)
        """

        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(mask, 0.0)

        x = einops.rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        out = einops.rearrange(x, 'b c n -> b n c')

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class RMSNorm(Module):
    """The Root Mean Square Layer Normalization

    References:
      - Zhang et al., Root Mean Square Layer Normalization, 2019
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class AdaptiveRMSNorm(Module):
    """
    Adaptive Root Mean Square Layer Normalization given a conditional embedding.
    This enables the model to consider the conditional input during normalization.
    """

    def __init__(self, dim: int, cond_dim: Optional[int] = None):
        super().__init__()
        if cond_dim is None:
            cond_dim = dim
        self.scale = dim**0.5

        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta = nn.Linear(cond_dim, dim)

        # init adaptive normalization to identity

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        normed = F.normalize(x, dim=-1) * self.scale

        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma = einops.rearrange(gamma, 'B D -> B 1 D')
        beta = einops.rearrange(beta, 'B D -> B 1 D')

        return normed * gamma + beta


class GEGLU(Module):
    """The GeGLU activation implementation"""

    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def get_feedforward_layer(dim: int, mult: int = 4, dropout: float = 0.0):
    """
    Return a Feed-Forward layer for the Transformer Layer.
    GeGLU activation is used in this FF layer
    """
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim_inner, dim))


class TransformerUNet(NeuralModule):
    """
    Implementation of the transformer Encoder Model with U-Net structure used in
    VoiceBox and AudioBox

    References:
        Le et al., Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale, 2023
        Vyas et al., Audiobox: Unified Audio Generation with Natural Language Prompts, 2023
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int = 8,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        max_positions: int = 6000,
        adaptive_rmsnorm: bool = False,
        adaptive_rmsnorm_cond_dim_in: Optional[int] = None,
        use_unet_skip_connection: bool = True,
        skip_connect_scale: Optional[int] = None,
    ):
        """
        Args:
            dim: Embedding dimension
            depth: Number of Transformer Encoder Layers
            heads: Number of heads in MHA
            ff_mult: The multiplier for the feedforward dimension (ff_dim = ff_mult * dim)
            attn_dropout: dropout rate for the MHA layer
            ff_dropout: droupout rate for the feedforward layer
            max_positions: The maximum time length of the input during training and inference
            adaptive_rmsnorm: Whether to use AdaptiveRMS layer.
                Set to True if the model has a conditional embedding in forward()
            adaptive_rms_cond_dim_in: Dimension of the conditional embedding
            use_unet_skip_connection: Whether to use U-Net or not
            skip_connect_scale: The scale of the U-Net connection.
        """
        super().__init__()
        if (depth % 2) != 0:
            raise ValueError(f"Number of layers {depth} is not divisible by 2!")
        self.layers = nn.ModuleList([])
        self.init_alibi(max_positions=max_positions, heads=heads)

        if adaptive_rmsnorm:
            rmsnorm_class = partial(AdaptiveRMSNorm, cond_dim=adaptive_rmsnorm_cond_dim_in)
        else:
            rmsnorm_class = RMSNorm

        if skip_connect_scale is None:
            self.skip_connect_scale = 2**-0.5
        else:
            self.skip_connect_scale = skip_connect_scale

        for ind in range(depth):
            layer = ind + 1
            has_skip = use_unet_skip_connection and layer > (depth // 2)

            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(dim * 2, dim) if has_skip else None,
                        rmsnorm_class(dim=dim),
                        nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=heads,
                            dropout=attn_dropout,
                            batch_first=True,
                        ),
                        rmsnorm_class(dim=dim),
                        get_feedforward_layer(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.final_norm = RMSNorm(dim)

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tembedding dim:       %s', dim)
        logging.debug('\tNumber of Layer:     %s', depth)
        logging.debug('\tfeedforward dim:     %s', dim * ff_mult)
        logging.debug('\tnumber of heads:     %s', heads)
        logging.debug('\tDropout rate of MHA: %s', attn_dropout)
        logging.debug('\tDropout rate of FF:  %s', ff_dropout)
        logging.debug('\tnumber of heads:     %s', heads)
        logging.debug('\tmaximun time length: %s', max_positions)
        logging.debug('\tuse AdaptiveRMS:     %s', adaptive_rmsnorm)
        logging.debug('\tConditional  dim:    %s', adaptive_rmsnorm_cond_dim_in)
        logging.debug('\tUse UNet connection: %s', use_unet_skip_connection)
        logging.debug('\tskip connect scale:  %s', self.skip_connect_scale)

    def init_alibi(
        self,
        max_positions: int,
        heads: int,
    ):
        """Initialize the Alibi bias parameters

        References:
          - Press et al., Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, 2021
        """

        def get_slopes(n):
            ratio = 2 ** (-8 / n)
            return ratio ** torch.arange(1, n + 1)

        if not math.log2(heads).is_integer():
            logging.warning(
                "It is recommend to set number of attention heads to be the power of 2 for the Alibi bias!"
            )
            logging.warning(f"Current value of heads: {heads}")

        self.slopes = nn.Parameter(einops.rearrange(get_slopes(heads), "B -> B 1 1"))

        pos_matrix = (
            -1 * torch.abs(torch.arange(max_positions).unsqueeze(0) - torch.arange(max_positions).unsqueeze(1)).float()
        )
        pos_matrix = einops.rearrange(pos_matrix, "T1 T2 -> 1 T1 T2")
        self.register_buffer('pos_matrix', pos_matrix, persistent=False)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "x": NeuralType(('B', 'T', 'D'), FloatType()),
            "key_padding_mask": NeuralType(('B', 'T'), BoolType(), optional=True),
            "adaptive_rmsnorm_cond": NeuralType(('B', 'D'), FloatType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'T', 'D'), FloatType()),
        }

    @typecheck()
    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None, adaptive_rmsnorm_cond=None):
        """Forward pass of the model.

        Args:
            input: input tensor, shape (B, C, D, T)
            key_padding_mask: mask tensor indicating the padding parts, shape (B, T)
            adaptive_rmsnorm_cond: conditional input for the model, shape (B, D)
        """
        batch_size, seq_len, *_ = x.shape
        skip_connects = []
        alibi_bias = self.get_alibi_bias(batch_size=batch_size, seq_len=seq_len)

        rmsnorm_kwargs = dict()
        if adaptive_rmsnorm_cond is not None:
            rmsnorm_kwargs = dict(cond=adaptive_rmsnorm_cond)

        for skip_combiner, attn_prenorm, attn, ff_prenorm, ff in self.layers:

            if skip_combiner is None:
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop() * self.skip_connect_scale
                x = torch.cat((x, skip_connect), dim=-1)
                x = skip_combiner(x)

            attn_input = attn_prenorm(x, **rmsnorm_kwargs)
            if key_padding_mask is not None:
                # Since Alibi_bias is a float-type attn_mask, the padding_mask need to be float-type.
                float_key_padding_mask = key_padding_mask.float()
                float_key_padding_mask = float_key_padding_mask.masked_fill(key_padding_mask, float('-inf'))
            else:
                float_key_padding_mask = None

            attn_output, _ = attn(
                query=attn_input,
                key=attn_input,
                value=attn_input,
                key_padding_mask=float_key_padding_mask,
                need_weights=False,
                attn_mask=alibi_bias,
            )
            x = x + attn_output

            ff_input = ff_prenorm(x, **rmsnorm_kwargs)
            x = ff(ff_input) + x

        return self.final_norm(x)

    def get_alibi_bias(self, batch_size: int, seq_len: int):
        """
        Return the alibi_bias given batch size and seqence length
        """
        pos_matrix = self.pos_matrix[:, :seq_len, :seq_len]
        alibi_bias = pos_matrix * self.slopes
        alibi_bias = alibi_bias.repeat(batch_size, 1, 1)

        return alibi_bias


class SpectrogramTransformerUNet(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using TransformerUNet and the output is projected to generate real
    and imaginary components of the output channels.

    Convolutional Positional Embedding is applied for the input sequence
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        freq_dim: int = 256,
        dim: int = 1024,
        depth: int = 24,
        heads: int = 16,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        max_positions: int = 6000,
        time_hidden_dim: Optional[int] = None,
        conv_pos_embed_kernel_size: int = 31,
        conv_pos_embed_groups: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        dim_in = freq_dim * in_channels * 2

        if time_hidden_dim is None:
            time_hidden_dim = dim * 4

        self.proj_in = nn.Linear(dim_in, dim)

        self.sinu_pos_emb = nn.Sequential(LearnedSinusoidalPosEmb(dim), nn.Linear(dim, time_hidden_dim), nn.SiLU())

        self.conv_embed = ConvPositionEmbed(
            dim=dim, kernel_size=conv_pos_embed_kernel_size, groups=conv_pos_embed_groups
        )

        self.transformerunet = TransformerUNet(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            max_positions=max_positions,
            adaptive_rmsnorm=True,
            adaptive_rmsnorm_cond_dim_in=time_hidden_dim,
            use_unet_skip_connection=True,
        )

        # 2x the frequency dimension as the model operates in the complex-value domain
        dim_out = freq_dim * out_channels * 2

        self.proj_out = nn.Linear(dim, dim_out)

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)
        logging.debug('\tInput frequency dimension: %s', freq_dim)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @staticmethod
    def _get_key_padding_mask(input_length: torch.Tensor, max_length: int):
        """
        Return the self_attention masking according to the input length.
        0 indicates the frame is in the valid range, while 1 indicates the frame is a padding frame.
        Args:
          input_length: shape (B)
          max_length (int): The maximum length of the input sequence

        return:
          key_padding_mask: shape (B, T)
        """
        key_padding_mask = torch.arange(max_length).expand(len(input_length), max_length).to(input_length.device)
        key_padding_mask = key_padding_mask >= input_length.unsqueeze(1)
        return key_padding_mask

    @typecheck()
    def forward(self, input, input_length=None, condition=None):
        """Forward pass of the model.

        Args:
            input: input tensor, shape (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
        """
        # Stack real and imaginary components
        B, C_in, D, T = input.shape
        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI D T -> B T (C RI D)')

        x = self.proj_in(input)
        key_padding_mask = self._get_key_padding_mask(input_length, max_length=T)
        x = self.conv_embed(x, mask=key_padding_mask) + x

        if condition is None:
            raise NotImplementedError

        time_emb = self.sinu_pos_emb(condition)

        x = self.transformerunet(x=x, key_padding_mask=key_padding_mask, adaptive_rmsnorm_cond=time_emb)

        output = self.proj_out(x)
        output = einops.rearrange(output, "B T (C RI D) -> B C D T RI", C=self.out_channels, RI=2, D=D)
        output = torch.view_as_complex(output.contiguous())

        return output, input_length
