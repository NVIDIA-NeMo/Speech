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
import math
import os
import re
from abc import abstractmethod
from collections.abc import Iterable
from contextlib import nullcontext
from functools import partial
from typing import Iterable

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.multimodal.modules.stable_diffusion.attention import SpatialTransformer
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.util import (
    avg_pool_nd,
    build_timestep_embedding,
    checkpoint,
    conv_nd,
    default,
    exists,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from nemo.utils import logging

try:
    # FP8 related import
    import transformer_engine

    HAVE_TE = True

except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

try:
    from apex.contrib.group_norm import GroupNorm

    OPT_GROUP_NORM = True
except Exception:
    print('Fused optimized group norm has not been installed.')
    OPT_GROUP_NORM = False


def convert_module_to_dtype(module, dtype, enable_norm_layers=False):
    # Convert module parameters to dtype
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
        module.weight.data = module.weight.data.to(dtype)
        if module.bias is not None:
            module.bias.data = module.bias.data.to(dtype)

    if enable_norm_layers:
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, GroupNorm)):
            module.weight.data = module.weight.data.to(dtype)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(dtype)


def convert_module_to_fp16(module, enable_norm_layers=False):
    convert_module_to_dtype(module, torch.float16, enable_norm_layers)


def convert_module_to_fp32(module, enable_norm_layers=False):
    convert_module_to_dtype(module, torch.float32, enable_norm_layers)


def convert_module_to_fp8(model):
    def _set_module(model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)

    import copy

    from transformer_engine.pytorch.module import Linear as te_Linear

    for n, v in model.named_modules():
        if isinstance(v, torch.nn.Linear):
            # if n in ['class_embed', 'bbox_embed.layers.0', 'bbox_embed.layers.1', 'bbox_embed.layers.2']: continue
            logging.info(f'[INFO] Replace Linear: {n}, weight: {v.weight.shape}')
            if v.bias is None:
                is_bias = False
            else:
                is_bias = True
            newlinear = te_Linear(v.in_features, v.out_features, bias=is_bias)
            newlinear.weight = copy.deepcopy(v.weight)
            if v.bias is not None:
                newlinear.bias = copy.deepcopy(v.bias)
            _set_module(model, n, newlinear)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    This layer performs upsampling on the given input with the option to apply a convolution operation.
    The upsampling can be applied to 1D, 2D, or 3D signals, depending on the specified dimensions.

    Parameters:
        channels (int): The number of channels in both the inputs and outputs.
        use_conv (bool): A bool determining if a convolution is applied.
        dims (int): Specifies the dimensionality of the signal.
                    It can be 1, 2, or 3. If set to 3, upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_up=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(yuya): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)
        if self.dims == 3:
            t_factor = 1 if not self.third_up else 2
            x = F.interpolate(x, (t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'

    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=ks, stride=2)

    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    This layer performs downsampling on the given input and optionally applies a convolution operation.
    The downsampling can be applied to 1D, 2D, or 3D signals, with specific behavior for 3D signals.

    Parameters:
        channels (int): The number of channels in both the inputs and outputs.
        use_conv (bool): Determines whether a convolution is applied.
                         True to apply convolution, False otherwise.
        dims (int): Specifies the dimensionality of the signal.
                    It can be 1, 2, or 3. For 3D signals, downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_down=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that optionally changes the number of channels.

    Parameters:
        channels (int): The number of input channels.
        emb_channels (int): The number of timestep embedding channels.
        dropout (float): The rate of dropout to apply.
        out_channels (int, optional): The number of output channels. If not specified, the output channels
                                      will be the same as the input channels.
        use_conv (bool): If True and out_channels is specified, a spatial convolution is used instead of a
                         smaller 1x1 convolution to change the channels in the skip connection.
        dims (int): Determines if the signal is 1D, 2D, or 3D.
        use_checkpoint (bool): If True, gradient checkpointing is used on this module. This can save memory
                                at the cost of additional compute.
        up (bool): If True, the block is used for upsampling.
        down (bool): If True, the block is used for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        resblock_gn_groups=32,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels, act="silu", gn_groups=resblock_gn_groups),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        if self.skip_t_emb:
            logging.info(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(emb_channels, self.emb_out_channels),
            )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, act="silu", gn_groups=resblock_gn_groups),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        Parameters:
            x (Tensor): An input Tensor of shape [N x C x ...], where N is the batch size, C is the number of channels,
                        and '...' represents additional dimensions.
            emb (Tensor): A Tensor of timestep embeddings of shape [N x emb_channels], where emb_channels is the number
                          of embedding channels.

        Returns:
            Tensor: An output Tensor of shape [N x C x ...], representing the processed features.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = th.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, **kwargs):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV (Query-Key-Value) attention.

        Parameters:
            qkv (Tensor): An input tensor of shape [N x (3 * H * C) x T], where N is the batch size,
                          H is the number of attention heads, C is the channel size, and T is the sequence length.
                          This tensor includes queries, keys, and values concatenated together.

        Returns:
            Tensor: An output tensor of shape [N x (H * C) x T] after applying attention. This tensor
                    contains the processed information with the same sequence length but with modified features.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Parameters:
        in_channels (int): The number of channels in the input Tensor.
        model_channels (int): The base channel count for the model.
        out_channels (int): The number of channels in the output Tensor.
        num_res_blocks (int): The number of residual blocks per downsample.
        attention_resolutions (set/list/tuple): The downsampling rates at which attention is applied.
                                                For example, if this includes 4, attention is used at 4x downsampling.
        dropout (float): The dropout probability.
        channel_mult (list/tuple): A channel multiplier for each level of the UNet.
        conv_resample (bool): If True, use learned convolutions for upsampling and downsampling.
        dims (int): Determines if the signal is 1D, 2D, or 3D.
        num_classes (int, optional): If specified, the model becomes class-conditional with the given number of classes.
        use_checkpoint (bool): If True, use gradient checkpointing to reduce memory usage.
        num_heads (int): The number of attention heads in each attention layer.
        num_heads_channels (int, optional): If specified, overrides num_heads and uses a fixed channel width per attention head.
        num_heads_upsample (int, optional): Sets a different number of heads for upsampling. Deprecated.
        use_scale_shift_norm (bool): If True, use a FiLM-like conditioning mechanism.
        resblock_updown (bool): If True, use residual blocks for up/downsampling.
        use_new_attention_order (bool): If True, use a different attention pattern for potentially increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_gn_groups=32,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        offload_to_cpu=False,
        transformer_depth_middle=None,
        from_pretrained: str = None,
        from_NeMo=False,
        # It must be specified when from pretrained is not None. It indicates loading unet from NeMo trained ckpt or HF
        use_flash_attention: bool = False,
        unet_precision: str = "fp32",
        lora_network_alpha=None,
        timesteps=1000,
        use_te_fp8: bool = False,
    ):
        super().__init__()
        from omegaconf.listconfig import ListConfig

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), 'You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), 'You forgot to use the spatial transformer for your cross-attention conditioning...'

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(transformer_depth_middle, transformer_depth[-1])

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        # self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            logging.info(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.time_embeddings = torch.Tensor(build_timestep_embedding(model_channels, timesteps))
        if unet_precision == 'fp16-mixed' or unet_precision == 'fp16':
            self.time_embeddings = self.time_embeddings.to(torch.float16)

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                logging.info("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.adm_in_channels = adm_in_channels
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        self.use_te_fp8 = use_te_fp8
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        resblock_gn_groups=resblock_gn_groups,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                                use_flash_attention=use_flash_attention,
                                lora_network_alpha=lora_network_alpha,
                                use_te=self.use_te_fp8,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            resblock_gn_groups=resblock_gn_groups,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                resblock_gn_groups=resblock_gn_groups,
            ),
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth_middle,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    use_checkpoint=use_checkpoint,
                    use_flash_attention=use_flash_attention,
                    use_te=self.use_te_fp8,
                    lora_network_alpha=lora_network_alpha,
                )
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                resblock_gn_groups=resblock_gn_groups,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        resblock_gn_groups=resblock_gn_groups,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                                use_flash_attention=use_flash_attention,
                                lora_network_alpha=lora_network_alpha,
                                use_te=self.use_te_fp8,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            resblock_gn_groups=resblock_gn_groups,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, act="silu", gn_groups=resblock_gn_groups),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )

        if from_pretrained is not None:
            if from_pretrained.endswith('safetensors'):
                from safetensors.torch import load_file as load_safetensors

                state_dict = load_safetensors(from_pretrained)
            else:
                state_dict = torch.load(from_pretrained, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            missing_key, unexpected_keys, _, _ = self._load_pretrained_model(state_dict, from_NeMo=from_NeMo)
            if len(missing_key) > 0:
                logging.info(
                    'Following keys are missing during loading unet weights, which may lead to compromised image quality for a resumed training. Please check the checkpoint you provided.'
                )
                logging.info(f"Missing keys: {missing_key}")
                logging.info(f"Unexpected keys: {unexpected_keys}")

        if unet_precision == "fp16-mixed":  # AMP O2
            self.convert_to_fp16()
        elif unet_precision == 'fp16':
            self.convert_to_fp16(enable_norm_layers=True)
        elif self.use_te_fp8:
            assert unet_precision != 'fp16', "fp8 training can't work with fp16 O2 amp recipe"
            convert_module_to_fp8(self)

            fp8_margin = int(os.getenv("FP8_MARGIN", '0'))
            fp8_interval = int(os.getenv("FP8_INTERVAL", '1'))
            fp8_format = os.getenv("FP8_FORMAT", "hybrid")
            fp8_amax_history_len = int(os.getenv("FP8_HISTORY_LEN", '1024'))
            fp8_amax_compute_algo = os.getenv("FP8_COMPUTE_ALGO", 'max')
            fp8_wgrad = os.getenv("FP8_WGRAD", '1') == '1'

            fp8_format_dict = {
                'hybrid': transformer_engine.common.recipe.Format.HYBRID,
                'e4m3': transformer_engine.common.recipe.Format.E4M3,
            }
            fp8_format = fp8_format_dict[fp8_format]

            self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=fp8_margin,
                interval=fp8_interval,
                fp8_format=fp8_format,
                amax_history_len=fp8_amax_history_len,
                amax_compute_algo=fp8_amax_compute_algo,
                override_linear_precision=(False, False, not fp8_wgrad),
            )
            old_state_dict = self.state_dict()
            new_state_dict = self.te_fp8_key_mapping(old_state_dict)
            self.load_state_dict(new_state_dict, strict=False)

        self.unet_precision = unet_precision

    def _input_blocks_mapping(self, input_dict):
        res_dict = {}
        for key_, value_ in input_dict.items():
            id_0 = int(key_[13])
            if "resnets" in key_:
                id_1 = int(key_[23])
                target_id = 3 * id_0 + 1 + id_1
                post_fix = (
                    key_[25:]
                    .replace('time_emb_proj', 'emb_layers.1')
                    .replace('norm1', 'in_layers.0')
                    .replace('norm2', 'out_layers.0')
                    .replace('conv1', 'in_layers.2')
                    .replace('conv2', 'out_layers.3')
                    .replace('conv_shortcut', 'skip_connection')
                )
                res_dict["input_blocks." + str(target_id) + '.0.' + post_fix] = value_
            elif "attentions" in key_:
                id_1 = int(key_[26])
                target_id = 3 * id_0 + 1 + id_1
                post_fix = key_[28:]
                res_dict["input_blocks." + str(target_id) + '.1.' + post_fix] = value_
            elif "downsamplers" in key_:
                post_fix = key_[35:]
                target_id = 3 * (id_0 + 1)
                res_dict["input_blocks." + str(target_id) + '.0.op.' + post_fix] = value_
        return res_dict

    def _mid_blocks_mapping(self, mid_dict):
        res_dict = {}
        for key_, value_ in mid_dict.items():
            if "resnets" in key_:
                temp_key_ = (
                    key_.replace('time_emb_proj', 'emb_layers.1')
                    .replace('norm1', 'in_layers.0')
                    .replace('norm2', 'out_layers.0')
                    .replace('conv1', 'in_layers.2')
                    .replace('conv2', 'out_layers.3')
                    .replace('conv_shortcut', 'skip_connection')
                    .replace('middle_block.resnets.0', 'middle_block.0')
                    .replace('middle_block.resnets.1', 'middle_block.2')
                )
                res_dict[temp_key_] = value_
            elif "attentions" in key_:
                res_dict[key_.replace('attentions.0', '1')] = value_
        return res_dict

    def _other_blocks_mapping(self, other_dict):
        res_dict = {}
        for key_, value_ in other_dict.items():
            tmp_key = (
                key_.replace('conv_in', 'input_blocks.0.0')
                .replace('time_embedding.linear_1', 'time_embed.0')
                .replace('time_embedding.linear_2', 'time_embed.2')
                .replace('conv_norm_out', 'out.0')
                .replace('conv_out', 'out.2')
            )
            res_dict[tmp_key] = value_
        return res_dict

    def _output_blocks_mapping(self, output_dict):
        res_dict = {}
        for key_, value_ in output_dict.items():
            id_0 = int(key_[14])
            if "resnets" in key_:
                id_1 = int(key_[24])
                target_id = 3 * id_0 + id_1
                post_fix = (
                    key_[26:]
                    .replace('time_emb_proj', 'emb_layers.1')
                    .replace('norm1', 'in_layers.0')
                    .replace('norm2', 'out_layers.0')
                    .replace('conv1', 'in_layers.2')
                    .replace('conv2', 'out_layers.3')
                    .replace('conv_shortcut', 'skip_connection')
                )
                res_dict["output_blocks." + str(target_id) + '.0.' + post_fix] = value_
            elif "attentions" in key_:
                id_1 = int(key_[27])
                target_id = 3 * id_0 + id_1
                post_fix = key_[29:]
                res_dict["output_blocks." + str(target_id) + '.1.' + post_fix] = value_
            elif "upsamplers" in key_:
                post_fix = key_[34:]
                target_id = 3 * (id_0 + 1) - 1
                mid_str = '.2.conv.' if target_id != 2 else '.1.conv.'
                res_dict["output_blocks." + str(target_id) + mid_str + post_fix] = value_
        return res_dict

    def _sdxl_embedding_mapping(self, sdxl_dict):
        res_dict = {}
        for key_, value_ in sdxl_dict.items():
            new_key_ = (
                key_.replace('add_embedding.', 'label_emb.').replace('linear_1.', '0.0.').replace('linear_2.', '0.2.')
            )
            res_dict[new_key_] = value_
        return res_dict

    def _legacy_unet_ckpt_mapping(self, unet_dict):
        new_dict = {}
        key_map = {
            'transformer_blocks.0.norm1.weight': 'transformer_blocks.0.attn1.norm.weight',
            'transformer_blocks.0.norm1.bias': 'transformer_blocks.0.attn1.norm.bias',
            'transformer_blocks.0.norm2.weight': 'transformer_blocks.0.attn2.norm.weight',
            'transformer_blocks.0.norm2.bias': 'transformer_blocks.0.attn2.norm.bias',
            'transformer_blocks.0.norm3.weight': 'transformer_blocks.0.ff.net.0.weight',
            'transformer_blocks.0.norm3.bias': 'transformer_blocks.0.ff.net.0.bias',
            'transformer_blocks.0.ff.net.0.proj.weight': 'transformer_blocks.0.ff.net.1.proj.weight',
            'transformer_blocks.0.ff.net.0.proj.bias': 'transformer_blocks.0.ff.net.1.proj.bias',
            'transformer_blocks.0.ff.net.2.weight': 'transformer_blocks.0.ff.net.3.weight',
            'transformer_blocks.0.ff.net.2.bias': 'transformer_blocks.0.ff.net.3.bias',
        }

        pattern = re.compile(r'(input_blocks|output_blocks)\.[\d\w]+\.[\d\w]+\.')
        pattern_middle_block = re.compile(r'middle_block\.[\d\w]+\.')
        for old_key, value in unet_dict.items():
            match = pattern.match(old_key)
            match_middle = pattern_middle_block.match(old_key)
            if match or match_middle:
                prefix = match.group(0) if match else match_middle.group(0)
                suffix = old_key.split('.', 3)[-1] if match else old_key.split('.', 2)[-1]
                if suffix in key_map:
                    new_key = prefix + key_map[suffix]
                    new_dict[new_key] = value
                else:
                    new_dict[old_key] = value
            else:
                new_dict[old_key] = value

        return new_dict

    def te_fp8_key_mapping(self, unet_dict):
        new_state_dict = {}
        for key in unet_dict.keys():
            if 'extra_state' in key:
                continue

            ### LayerNormLinear
            # norm_to_q.layer_norm_{weight|bias} -> norm.{weight|bias}
            # norm_to_q.weight -> to_q.weight
            new_key = key.replace('attn1.norm.', 'attn1.norm_to_q.layer_norm_')
            new_key = new_key.replace(
                'attn1.to_q.weight',
                'attn1.norm_to_q.weight',
            )
            new_key = new_key.replace('attn2.norm.', 'attn2.norm_to_q.layer_norm_')
            new_key = new_key.replace(
                'attn2.to_q.weight',
                'attn2.norm_to_q.weight',
            )

            ### LayerNormMLP
            # ff.net.layer_norm_{weight|bias} -> ff.net.0.{weight|bias}
            # ff.net.fc1_{weight|bias} -> ff.net.1.proj.{weight|bias}
            # ff.net.fc2_{weight|bias} -> ff.net.3.{weight|bias}
            new_key = new_key.replace('ff.net.0.', 'ff.net.layer_norm_')
            new_key = new_key.replace('ff.net.1.proj.', 'ff.net.fc1_')
            new_key = new_key.replace('ff.net.3.', 'ff.net.fc2_')

            new_state_dict[new_key] = unet_dict[key]
        return new_state_dict

    def _state_key_mapping(self, state_dict: dict):

        res_dict = {}
        input_dict = {}
        mid_dict = {}
        output_dict = {}
        other_dict = {}
        sdxl_dict = {}
        for key_, value_ in state_dict.items():
            if "down_blocks" in key_:
                input_dict[key_.replace('down_blocks', 'input_blocks')] = value_
            elif "up_blocks" in key_:
                output_dict[key_.replace('up_blocks', 'output_blocks')] = value_
            elif "mid_block" in key_:
                mid_dict[key_.replace('mid_block', 'middle_block')] = value_
            elif "add_embedding" in key_:
                # SDXL related mapping
                sdxl_dict[key_] = value_
            else:
                other_dict[key_] = value_

        input_dict = self._input_blocks_mapping(input_dict)
        output_dict = self._output_blocks_mapping(output_dict)
        mid_dict = self._mid_blocks_mapping(mid_dict)
        other_dict = self._other_blocks_mapping(other_dict)
        sdxl_dict = self._sdxl_embedding_mapping(sdxl_dict)

        res_dict.update(input_dict)
        res_dict.update(output_dict)
        res_dict.update(mid_dict)
        res_dict.update(other_dict)
        res_dict.update(sdxl_dict)

        return res_dict

    def _load_pretrained_model(self, state_dict, ignore_mismatched_sizes=False, from_NeMo=False):
        state_dict = self._strip_unet_key_prefix(state_dict)
        if not from_NeMo:
            state_dict = self._state_key_mapping(state_dict)
        state_dict = self._legacy_unet_ckpt_mapping(state_dict)

        model_state_dict = self.state_dict()
        loaded_keys = [k for k in state_dict.keys()]
        expected_keys = list(model_state_dict.keys())
        original_loaded_keys = loaded_keys
        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # SDXL specific mapping
        if 'output_blocks.2.2.conv.bias' in missing_keys and 'output_blocks.2.1.conv.bias' in loaded_keys:
            state_dict['output_blocks.2.2.conv.bias'] = state_dict['output_blocks.2.1.conv.bias']
            state_dict['output_blocks.2.2.conv.weight'] = state_dict['output_blocks.2.1.conv.weight']

        if 'out.1.weight' in missing_keys and 'out.2.weight' in state_dict.keys():
            state_dict['out.1.weight'] = state_dict['out.2.weight']
            state_dict['out.1.bias'] = state_dict['out.2.bias']

        if (
            'input_blocks.1.0.in_layers.2.weight' in loaded_keys
            and 'input_blocks.1.0.in_layers.1.weight' in expected_keys
        ):
            # GroupNormOpt fuses activation function to one layer, thus the indexing of weights are shifted for following
            for key_ in missing_keys:
                try:
                    s = key_.split('.')
                    idx = int(s[-2])
                    new_key_ = ".".join(s[:-2] + [str(int(idx + 1))] + [s[-1]])
                    state_dict[key_] = state_dict[new_key_]
                except:
                    continue

        loaded_keys = list(state_dict.keys())
        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            error_msgs = self._load_state_dict_into_model(state_dict)
        return missing_keys, unexpected_keys, mismatched_keys, error_msgs

    # TODO MMY maybe combine these cases of key prefix
    def _strip_unet_key_prefix(self, state_dict):
        re_state_dict = {}
        for key_, value_ in state_dict.items():
            if key_.startswith('model.diffusion_model'):
                re_state_dict[key_.replace('model.diffusion_model.', '')] = value_
            elif key_.startswith('model.model.diffusion_model'):
                re_state_dict[key_.replace('model.model.diffusion_model.', '')] = value_
            elif key_.startswith('model._orig_mod.diffusion_model.'):
                re_state_dict[key_.replace('model._orig_mod.diffusion_model.', '')] = value_
            elif key_.startswith('model.model._orig_mod.diffusion_model.'):
                re_state_dict[key_.replace('model.model._orig_mod.diffusion_model.', '')] = value_
            elif key_.startswith('model.model.diffusion_model._orig_mod.'):
                re_state_dict[key_.replace('model.model.diffusion_model._orig_mod.', '')] = value_
            else:
                re_state_dict[key_] = value_
        return re_state_dict

    def _load_state_dict_into_model(self, state_dict):
        # Convert old format to new format if needed from a PyTorch state_dict
        # copy state_dict so _load_from_state_dict can modify it
        state_dict = state_dict.copy()
        error_msgs = []

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module: torch.nn.Module, prefix=""):
            args = (state_dict, prefix, {}, True, [], [], error_msgs)
            module._load_from_state_dict(*args)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self)

        return error_msgs

    def convert_to_fp16(self, enable_norm_layers=False):
        """
        Convert the torso of the model to float16.
        """
        self.apply(lambda module: convert_module_to_fp16(module=module, enable_norm_layers=enable_norm_layers))

    def _forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.

        Parameters:
            x (Tensor): An input tensor of shape [N x C x ...], where N is the batch size, C is the number of channels,
                        and '...' represents additional dimensions.
            timesteps (Tensor): A 1-D tensor representing a batch of timesteps.
            context (Tensor, optional): An optional tensor for additional conditioning, used via cross-attention.
            y (Tensor, optional): An optional 1-D tensor of labels of shape [N], used if the model is class-conditional.

        Returns:
            Tensor: An output tensor of shape [N x C x ...], representing the processed batch.
        """

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        if self.unet_precision == "fp16-mixed" or self.unet_precision == "fp16":
            x = x.type(torch.float16)
            if context is not None:
                context = context.type(torch.float16)

        t_emb = timestep_embedding(
            timesteps, self.model_channels, cached_embedding=self.time_embeddings.to(timesteps.device)
        )
        emb = self.time_embed(t_emb)
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(emb.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        with (
            transformer_engine.pytorch.fp8_autocast(
                enabled=self.use_te_fp8,
                fp8_recipe=self.fp8_recipe,
            )
            if self.use_te_fp8
            else nullcontext()
        ):
            out = self._forward(x, timesteps, context, y, **kwargs)
        return out


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        resblock_gn_groups=32,
        *args,
        **kwargs,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        resblock_gn_groups=resblock_gn_groups,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            resblock_gn_groups=resblock_gn_groups,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                resblock_gn_groups=resblock_gn_groups,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                resblock_gn_groups=resblock_gn_groups,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d((image_size // ds), ch, num_head_channels, out_channels),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_fp16)
        self.middle_block.apply(convert_module_to_fp16)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels), use_fp16=self.use_fp16)

        # future support
        if self.dtype == th.float32:
            self.dtype == x.dtype

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)
