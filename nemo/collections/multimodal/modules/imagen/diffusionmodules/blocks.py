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
"""
Adapted from:
https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
"""
import math
from abc import abstractmethod

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from nemo.collections.multimodal.modules.imagen.diffusionmodules import attention_alt

if attention_alt.USE_ALT:
    from nemo.collections.multimodal.modules.imagen.diffusionmodules.attention_alt import (
        QKVAttention,
        QKVMaskedAttention,
        QKVStableAttention,
        QKVStableMaskedAttention,
    )
else:
    from nemo.collections.multimodal.modules.imagen.diffusionmodules.attention import (
        QKVAttention,
        QKVMaskedAttention,
        QKVStableAttention,
        QKVStableMaskedAttention,
    )
from nemo.collections.multimodal.modules.imagen.diffusionmodules.layers import (
    Downsample,
    Upsample,
    UpsampleLearnable,
    conv_nd,
    linear,
    normalization,
    zero_module,
)


def check_cuda():
    if not th.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    cur_device = th.cuda.current_device()
    dprops = th.cuda.get_device_properties(cur_device)

    is_sm75 = dprops.major == 7 and dprops.minor == 5
    is_sm8x_or_later = dprops.major >= 8

    return is_sm75 or is_sm8x_or_later


try:
    from flash_attn import flash_attn_varlen_func, flash_attn_varlen_kvpacked_func

    flash_attn_installed = check_cuda()
except ImportError:
    flash_attn_installed = False


class TextConditionedBlock(nn.Module):
    r"""
    Any module where forward() takes text embeddings as arguments.
    """

    @abstractmethod
    def forward(self, x, text_emb, text_mask):
        """
        Apply the module to `x` given `text_emb` text embedding and 'text_mask' text valid mask.
        """


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class ConditionalSequential(nn.Sequential, TimestepBlock, TextConditionedBlock):
    r"""
    A sequential module that accepts timestep embeddings, text embedding and text mask in addition to the input x.
    Depending on the type of block, we either pass timestep embedding or text embeddings as inputs.
    """

    def forward(self, x, emb, text_emb, text_mask):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, TextConditionedBlock):
                x = layer(x, text_emb, text_mask)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
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
        learnable_upsampling=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if learnable_upsampling:
            upsample_fn = UpsampleLearnable
        else:
            upsample_fn = Upsample

        if up:
            self.h_upd = upsample_fn(channels, False, dims)
            self.x_upd = upsample_fn(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, emb)
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
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class EfficientResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    Follow Figure A.27 in Imagen Paper.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        skip_connection_scaling=False,
    ):
        super().__init__()

        out_channels = out_channels or channels

        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(), conv_nd(dims, channels, out_channels, 3, padding=1)
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * out_channels if use_scale_shift_norm else out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            zero_module(conv_nd(dims, out_channels, out_channels, 3, padding=1)),
        )

        self.shortcut = conv_nd(dims, channels, out_channels, 1)
        self.shortcut_scale = 1 / math.sqrt(2) if skip_connection_scaling else 1

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return h + self.shortcut(x) * self.shortcut_scale


class Block(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_scale_shift_norm=True,
        num_resblocks=2,
        attention_type=None,
        text_embed_dim=0,
        stable_attention=True,
        flash_attention=False,
        num_head_channels=-1,
        num_heads=8,
        dims=2,
        use_checkpoint=False,
        skip_connection_scaling=False,
    ):
        super().__init__()

        out_channels = out_channels or channels

        self.attention_type = attention_type
        self.text_embed_dim = text_embed_dim

        blocks = [
            EfficientResBlock(
                channels,
                emb_channels,
                out_channels=out_channels,
                use_scale_shift_norm=use_scale_shift_norm,
                dims=dims,
                use_checkpoint=use_checkpoint,
                skip_connection_scaling=skip_connection_scaling,
            )
        ]

        blocks += [
            EfficientResBlock(
                out_channels,
                emb_channels,
                out_channels=out_channels,
                use_scale_shift_norm=use_scale_shift_norm,
                dims=dims,
                use_checkpoint=use_checkpoint,
                skip_connection_scaling=skip_connection_scaling,
            )
            for _ in range(num_resblocks - 1)
        ]

        self.blocks = nn.ModuleList(blocks)

        # Attention blocks
        # Self - Self-attention blocks
        # fused - Single attention layer for fusing self and cross attention.
        if self.attention_type is not None:
            assert self.attention_type in ('self', 'cross', 'fused', 'stacked')
            attention_kwargs = dict()

            if self.attention_type == 'self':
                attention_fn = SelfAttentionBlock
            elif self.attention_type == 'cross':
                attention_fn = CrossAttentionBlock
                attention_kwargs['context_dim'] = self.text_embed_dim
            elif self.attention_type == 'stacked':
                attention_fn = StackedCrossAttentionBlock
                attention_kwargs['context_dim'] = self.text_embed_dim
            else:
                attention_fn = FusedCrossAttentionBlock
                attention_kwargs['context_dim'] = self.text_embed_dim

            self.attention_layer = attention_fn(
                out_channels,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_checkpoint=use_checkpoint,
                stable_attention=stable_attention,
                flash_attention=flash_attention,
                **attention_kwargs,
            )

    @abstractmethod
    def forward(self, x, emb, text_embed=None, text_mask=None):
        pass


class DBlock(Block):
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_scale_shift_norm=True,
        conv_down=True,
        stride=2,
        num_resblocks=2,
        attention_type=None,
        text_embed_dim=0,
        stable_attention=True,
        flash_attention=False,
        num_head_channels=-1,
        num_heads=8,
        dims=2,
        use_checkpoint=False,
        skip_connection_scaling=False,
    ):
        super().__init__(
            channels,
            emb_channels,
            out_channels=out_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            num_resblocks=num_resblocks,
            attention_type=attention_type,
            text_embed_dim=text_embed_dim,
            stable_attention=stable_attention,
            flash_attention=flash_attention,
            num_head_channels=num_head_channels,
            num_heads=num_heads,
            dims=dims,
            use_checkpoint=use_checkpoint,
            skip_connection_scaling=skip_connection_scaling,
        )

        self.conv_down = conv_down
        if self.conv_down:
            # self.conv = nn.Conv2d(channels, channels, 3, stride=stride, padding=1)
            self.conv = nn.Conv2d(channels, channels, 4, stride=stride, padding=1)

    def forward(self, x, emb, text_embed=None, text_mask=None):
        if self.conv_down:
            x = self.conv(x)

        for block in self.blocks:
            x = block(x, emb)

        if self.attention_type in ('cross', 'fused', 'stacked'):
            x = self.attention_layer(x, text_embed, text_mask)
        elif self.attention_type == 'self':
            x = self.attention_layer(x)

        return x


class UBlock(Block):
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_scale_shift_norm=True,
        conv_up=True,
        stride=2,
        num_resblocks=2,
        attention_type=None,
        text_embed_dim=0,
        stable_attention=True,
        flash_attention=False,
        num_head_channels=-1,
        num_heads=8,
        dims=2,
        use_checkpoint=False,
        skip_connection_scaling=False,
    ):
        super().__init__(
            channels,
            emb_channels,
            out_channels=out_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            num_resblocks=num_resblocks,
            attention_type=attention_type,
            text_embed_dim=text_embed_dim,
            stable_attention=stable_attention,
            flash_attention=flash_attention,
            num_head_channels=num_head_channels,
            num_heads=num_heads,
            dims=dims,
            use_checkpoint=use_checkpoint,
            skip_connection_scaling=skip_connection_scaling,
        )

        self.conv_up = conv_up
        if self.conv_up:
            self.conv = nn.ConvTranspose2d(out_channels, out_channels, 4, stride, 1)

    def forward(self, x, emb, text_embed=None, text_mask=None):
        for block in self.blocks:
            x = block(x, emb)

        if self.attention_type in ('cross', 'fused', 'stacked'):
            x = self.attention_layer(x, text_embed, text_mask)
        elif self.attention_type == 'self':
            x = self.attention_layer(x)

        if self.conv_up:
            x = self.conv(x)

        return x


class FusedCrossAttentionBlock(TextConditionedBlock):
    """
    An attention block that fuses self-attention and cross-attention
    in a single block.
    """

    def __init__(
        self,
        channels,
        context_dim,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        stable_attention=True,
        flash_attention=False,
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
        self.flash_attention = flash_attention
        self.norm = normalization(channels)
        self.norm_context = normalization(context_dim)
        self.norm_self = normalization(channels)

        # For image features
        self.q = conv_nd(1, channels, channels, 1)

        # For context
        self.kv_context = conv_nd(1, context_dim, channels * 2, 1)

        # For spatial
        self.kv_self = conv_nd(1, channels, channels * 2, 1)

        if flash_attention:
            assert flash_attn_installed, "FlashAttention is not installed."
            assert not stable_attention, "FlashAttention doesn't support the stable form."

        elif stable_attention:
            self.attention = QKVStableMaskedAttention(self.num_heads)
        else:
            self.attention = QKVMaskedAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, context, mask):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, context, mask)
        else:
            return self._forward(x, context, mask)

    def _forward(self, x, context, mask):

        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        q = self.q(self.norm(x))

        # Key-value pairs for self-attention
        kv_self = self.kv_self(self.norm_self(x))
        k_self, v_self = kv_self.chunk(2, dim=1)
        k_self = k_self.contiguous()
        v_self = v_self.contiguous()

        # Key-value pairs for cross-attention
        context = th.permute(context, (0, 2, 1))
        context_n = self.norm_context(context)
        kv_context = self.kv_context(context_n)
        k_context, v_context = kv_context.chunk(2, dim=1)
        k_context = k_context.contiguous()
        v_context = v_context.contiguous()

        # Appending key-value pairs
        k_full = th.cat([k_self, k_context], dim=2)
        v_full = th.cat([v_self, v_context], dim=2)

        if self.flash_attention:
            # q: b (h d) s, k_context: b (h d) s
            batch_size = q.shape[0]
            max_seqlen_q, max_seqlen_k = q.shape[2], q.shape[2] + k_context.shape[2]
            q = rearrange(q, 'b (h d) s -> (b s) h d', h=self.num_heads)

            mask_self = th.ones((batch_size, max_seqlen_q), device=q.device, dtype=th.bool)
            mask_context = mask.bool()
            mask_full = th.cat([mask_self, mask_context], dim=1)

            k_full_unpadded = k_full.transpose(1, 2)[mask_full]
            total_k = k_full_unpadded.shape[0]
            k_full_unpadded = k_full_unpadded.view(total_k, self.num_heads, -1)

            v_full_unpadded = v_full.transpose(1, 2)[mask_full]
            v_full_unpadded = v_full_unpadded.view(total_k, self.num_heads, -1)

            # (b s) t h d
            kv_full_unpadded = th.stack([k_full_unpadded, v_full_unpadded], dim=1)

            cu_seqlens_q = th.arange(
                0, (batch_size + 1) * max_seqlen_q, step=max_seqlen_q, dtype=th.int32, device=q.device
            )
            cu_seqlens_k = th.zeros((batch_size + 1), dtype=th.int32, device=k_full.device)
            cu_seqlens_k[1:] = th.cumsum(mask.sum(dim=1), dim=0)
            cu_seqlens_k += cu_seqlens_q

            out = flash_attn_varlen_kvpacked_func(
                q, kv_full_unpadded, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 0.0
            )
            h = rearrange(out, '(b s) h d -> b (h d) s', b=batch_size, h=self.num_heads)
        else:
            # Computing mask for self attention
            mask_self = th.ones(k_self.shape[0], q.shape[2], k_self.shape[2], device=mask.device)

            # Mask for cross attention
            mask_context = mask.view(mask.shape[0], 1, mask.shape[1])
            mask_context = mask_context.repeat(1, q.shape[2], 1)

            # Fused mask
            mask_full = th.cat([mask_self, mask_context], dim=2)
            mask_full = mask_full.to(th.bool)

            h, _ = self.attention(q, k_full, v_full, mask_full)

        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class SelfAttentionBlock(nn.Module):
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
        stable_attention=False,
        flash_attention=False,
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
        self.flash_attention = flash_attention
        if flash_attention:
            assert flash_attn_installed, "FlashAttention is not installed."
            assert not stable_attention, "FlashAttention doesn't support the stable form."
        elif stable_attention:
            self.attention = QKVStableAttention(self.num_heads)
        else:
            self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):

        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))

        if self.flash_attention:
            # qkv shape: (b, (3 h d) s), need to reshape to (b, s, h, d) for each q, k, v
            b, _, _ = qkv.shape
            h = self.num_heads
            q, k, v = qkv.chunk(3, dim=1)
            max_seqlen_q, max_seqlen_k = q.shape[2], k.shape[2]
            q = rearrange(q, 'b (h d) s -> (b s) h d', h=self.num_heads)
            k = rearrange(k, 'b (h d) s -> (b s) h d', h=self.num_heads)
            v = rearrange(v, 'b (h d) s -> (b s) h d', h=self.num_heads)
            cu_seqlens_q = th.arange(0, (b + 1) * max_seqlen_q, step=max_seqlen_q, dtype=th.int32, device=q.device)
            cu_seqlens_k = th.arange(0, (b + 1) * max_seqlen_k, step=max_seqlen_k, dtype=th.int32, device=k.device)
            h = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 0.0)
            h = rearrange(h, '(b s) h d -> b (h d) s', b=b, h=self.num_heads)
        else:
            h, _ = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


#########################################################################
# These are the attention blocks as implemented by Stable Diffusion
# https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/attention.py#L196


class CrossAttentionBlock(TextConditionedBlock):
    """
    An attention block that allows spatial positions to attend to context.
    In our case, context is the token-wise text embeddings.
    """

    def __init__(
        self,
        channels,
        context_dim,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        stable_attention=True,
        flash_attention=False,
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
        self.norm_context = normalization(context_dim)
        self.flash_attention = flash_attention
        # For image features
        self.q = conv_nd(1, channels, channels, 1)

        # For context
        self.kv = conv_nd(1, context_dim, channels * 2, 1)

        if flash_attention:
            assert flash_attn_installed, "FlashAttention is not installed."
            assert not stable_attention, "FlashAttention doesn't support the stable form."
        elif stable_attention:
            self.attention = QKVStableMaskedAttention(self.num_heads)
        else:
            self.attention = QKVMaskedAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, context, mask):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, context, mask)
        else:
            return self._forward(x, context, mask)

    def _forward(self, x, context, mask):

        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        q = self.q(self.norm(x))
        context = th.permute(context, (0, 2, 1))
        context_n = self.norm_context(context)
        kv = self.kv(context_n)
        k, v = kv.chunk(2, dim=1)
        k = k.contiguous()
        v = v.contiguous()

        if self.flash_attention:
            batch_size = q.shape[0]
            max_seqlen_q, max_seqlen_k = q.shape[2], k.shape[2]
            q = rearrange(q, 'b (h d) s -> (b s) h d', h=self.num_heads)
            mask = mask.to(th.bool)
            k_unpadded = k.transpose(1, 2)[mask]
            total_k = k_unpadded.shape[0]
            k_unpadded = k_unpadded.view(total_k, self.num_heads, -1)
            v_unpadded = v.transpose(1, 2)[mask]
            v_unpadded = v_unpadded.view(total_k, self.num_heads, -1)
            kv_unpadded = th.stack([k_unpadded, v_unpadded], dim=1)
            cu_seqlens_q = th.arange(
                0, (batch_size + 1) * max_seqlen_q, step=max_seqlen_q, dtype=th.int32, device=q.device
            )
            cu_seqlens_k = th.zeros((batch_size + 1), dtype=th.int32, device=q.device)
            cu_seqlens_k[1:] = th.cumsum(mask.sum(dim=1), dim=0)

            out = flash_attn_varlen_kvpacked_func(
                q, kv_unpadded, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 0.0
            )
            h = rearrange(out, '(b s) h d -> b (h d) s', b=batch_size, h=self.num_heads)
        else:
            # Computing mask for cross attention
            mask = mask.view(mask.shape[0], 1, mask.shape[1])
            mask = mask.repeat(1, q.shape[-1], 1)
            mask = mask.to(th.bool)

            h, _ = self.attention(q, k, v, mask)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.norm = normalization(dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        h = self.norm(x)

        # Reshape so that the channel dim moves to last
        # Linear function operates on the last dimension
        h = th.permute(h, (0, 2, 1))

        h = self.net(h)

        # Permute it back
        h = th.permute(h, (0, 2, 1))

        return (x + h).reshape(b, c, *spatial)


class StackedCrossAttentionBlock(TextConditionedBlock):
    """
    An attention block that stacks self-attention and cross-attention layers
    in a single block.
    """

    def __init__(
        self,
        channels,
        context_dim,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        stable_attention=True,
        flash_attention=False,
    ):
        super().__init__()
        self.proj_in = conv_nd(2, channels, channels, 1)
        self.norm = normalization(channels)
        self.use_checkpoint = use_checkpoint

        self.self_attention_block = SelfAttentionBlock(
            channels=channels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_checkpoint=use_checkpoint,
            stable_attention=stable_attention,
            flash_attention=flash_attention,
        )

        self.cross_attention_block = CrossAttentionBlock(
            channels=channels,
            context_dim=context_dim,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_checkpoint=use_checkpoint,
            stable_attention=stable_attention,
            flash_attention=flash_attention,
        )

        self.ff = FeedForward(dim=channels, glu=True)
        self.proj_out = zero_module(conv_nd(2, channels, channels, 1))

    def forward(self, x, context, mask):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, context, mask)
        else:
            return self._forward(x, context, mask)

    def _forward(self, x, context, mask):

        h = self.norm(x)
        h = self.proj_in(h)

        h = self.self_attention_block(h)
        h = self.cross_attention_block(h, context, mask)
        h = self.ff(h)

        h = self.proj_out(h)
        return h + x
