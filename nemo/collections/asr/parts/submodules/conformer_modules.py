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
#

import torch
from torch import nn as nn
from torch.nn import LayerNorm

from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.submodules.batchnorm import FusedBatchNorm1d
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
    RoPEMultiHeadAttention,
)
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.common.parts.utils import ModuleBlock, activation_registry, block_binds, build_module
from nemo.core.classes.mixins import AccessMixin

__all__ = ['ConformerConvolution', 'ConformerFeedForward', 'ConformerLayer', 'build_self_attention']


def build_self_attention(
    self_attn: ModuleBlock | None,
    self_attention_model: str,
    *,
    n_heads: int,
    d_model: int,
    dropout_att: float,
    att_context_size,
    pos_bias_u=None,
    pos_bias_v=None,
    use_bias: bool = True,
    use_pytorch_sdpa: bool = False,
    use_pytorch_sdpa_backends=None,
    global_tokens: int = 0,
    global_tokens_spacing: int = 1,
    global_attn_separate: bool = False,
    pos_enc=None,
) -> nn.Module:
    """Build the self-attention module of a Conformer layer.

    `ConformerLayer` builds its attention here, and `ConformerEncoder.change_attention_model` rebuilds it here,
    so a `self_attn` block keeps its settings through both.

    Args:
        self_attn: the attention block (a class, a callable or a Hydra partial),
            default the attention class of `self_attention_model`;
            the block's own arguments win over the defaults below
        self_attention_model: 'rel_pos', 'rel_pos_local_attn', 'abs_pos' or 'rope',
            which picks the default class and the arguments it receives
        n_heads: number of attention heads
        d_model: size of the features
        dropout_att: dropout of the attention weights
        att_context_size: left and right context; the left size bounds the cache,
            and local attention receives the whole context
        pos_bias_u: positional bias U of the relative-position models, shared across layers when given
        pos_bias_v: positional bias V, likewise
        use_bias: bias in the projections
        use_pytorch_sdpa: PyTorch scaled dot-product attention; local attention does not support it and is built without
        use_pytorch_sdpa_backends: its backends
        global_tokens: global tokens of local attention
        global_tokens_spacing: their spacing
        global_attn_separate: separate projections for the global tokens
        pos_enc: the rotary encoding that 'rope' attention needs

    Returns:
        The attention module.
    """
    attention_defaults = dict(
        n_head=n_heads,
        n_feat=d_model,
        dropout_rate=dropout_att,
        max_cache_len=att_context_size[0],
        use_bias=use_bias,
    )
    sdpa_defaults = dict(use_pytorch_sdpa=use_pytorch_sdpa, use_pytorch_sdpa_backends=use_pytorch_sdpa_backends or [])

    if self_attention_model == 'rel_pos':
        return build_module(
            self_attn or RelPositionMultiHeadAttention,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            **attention_defaults,
            **sdpa_defaults,
        )
    if self_attention_model == 'rel_pos_local_attn':
        return build_module(
            self_attn or RelPositionMultiHeadAttentionLongformer,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            att_context_size=att_context_size,
            global_tokens=global_tokens,
            global_tokens_spacing=global_tokens_spacing,
            global_attn_separate=global_attn_separate,
            **attention_defaults,
        )
    if self_attention_model == 'abs_pos':
        return build_module(self_attn or MultiHeadAttention, **attention_defaults, **sdpa_defaults)
    if self_attention_model == 'rope':
        if pos_enc is None:
            raise ValueError("'rope' attention requires a RotaryPositionalEncoding via pos_enc.")
        return build_module(
            self_attn or RoPEMultiHeadAttention, pos_enc=pos_enc, **attention_defaults, **sdpa_defaults
        )
    raise ValueError(
        f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
        f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos', 'rope']"
    )


class ConformerLayer(torch.nn.Module, AttentionAdapterModuleMixin, AccessMixin):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
                overlapping chunks. Attention context is determined by att_context_size parameter.
            'abs_pos': absolute positional embedding and Transformer
            'rope': rotary position embedding
            Default is rel_pos.
        global_tokens (int): number of tokens to be used for global attention.
            Only relevant if self_attention_model is 'rel_pos_local_attn'.
            Defaults to 0.
        global_tokens_spacing (int): how far apart the global tokens are
            Defaults to 1.
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        use_bias (bool): Apply bias to all Linear and Conv1d layers from each ConformerLayer to improve activation flow and stabilize training of huge models.
            Defaults to True.
        feed_forward: block of both feed-forward modules, default `ConformerFeedForward`;
            a block is a class, a callable or a Hydra partial (``_target_`` with ``_partial_: true``),
            built from the arguments the parent supplies, so a block section only names what deviates
        conv: block of the convolution module, default `ConformerConvolution`
        self_attn: block of the self-attention, default the attention class of `self_attention_model`
        norm: block of every layer norm, default `torch.nn.LayerNorm`
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
        pos_enc=None,
        feed_forward: ModuleBlock | None = None,
        conv: ModuleBlock | None = None,
        self_attn: ModuleBlock | None = None,
        norm: ModuleBlock | None = None,
    ):
        super(ConformerLayer, self).__init__()

        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        norm = norm or LayerNorm
        feed_forward = feed_forward or ConformerFeedForward

        # first feed forward module
        self.norm_feed_forward1 = build_module(norm, normalized_shape=d_model)
        self.feed_forward1 = build_module(feed_forward, d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        # convolution module
        self.norm_conv = build_module(norm, normalized_shape=d_model)
        conv_defaults = dict(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )
        if block_binds(conv, 'kernel_size') and not block_binds(conv, 'conv_context_size'):
            # a block with its own kernel derives its own padding
            conv_defaults['conv_context_size'] = None
        self.conv = build_module(conv or ConformerConvolution, **conv_defaults)

        # multi-headed self-attention module
        self.norm_self_att = build_module(norm, normalized_shape=d_model)
        self.self_attn = build_self_attention(
            self_attn,
            self_attention_model,
            n_heads=n_heads,
            d_model=d_model,
            dropout_att=dropout_att,
            att_context_size=att_context_size,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            use_bias=use_bias,
            use_pytorch_sdpa=self.use_pytorch_sdpa,
            use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            global_tokens=global_tokens,
            global_tokens_spacing=global_tokens_spacing,
            global_attn_separate=global_attn_separate,
            pos_enc=pos_enc,
        )

        # second feed forward module
        self.norm_feed_forward2 = build_module(norm, normalized_shape=d_model)
        self.feed_forward2 = build_module(feed_forward, d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = build_module(norm, normalized_shape=d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
            cache_last_channel (torch.tensor) : cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : cache for convolutional layers (B, d_model, T_cache)
        Returns:
            x (torch.Tensor): (B, T, d_model)
            cache_last_channel (torch.tensor) : next cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : next cache for convolutional layers (B, d_model, T_cache)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'rel_pos_local_attn':
            x = self.self_attn(query=x, key=x, value=x, pad_mask=pad_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model in ('abs_pos', 'rope'):
            # 'rope' rotates Q/K inside MultiHeadAttention via the attached pos_enc,
            # so the call site is identical to 'abs_pos' (no additive pos_emb).
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, cache=cache_last_channel)
        else:
            x = None

        if x is not None and cache_last_channel is not None:
            (x, cache_last_channel) = x

        residual = residual + self.dropout(x)

        if self.is_adapter_available():
            # Call the MHA adapters
            pack_input = {
                'x': residual,
                'loc': 'mha',
                'att_mask': att_mask,
                'pos_emb': pos_emb,
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            residual = pack_input['x']

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)

        if self.is_adapter_available():
            # Call the adapters
            pack_input = {
                'x': x,
                'loc': 'post',
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            x = pack_input['x']

        if self.is_access_enabled(getattr(self, "model_guid", None)) and self.access_cfg.get(
            'save_encoder_tensors', False
        ):
            self.register_accessible_tensor(name='encoder', tensor=x)
        if cache_last_channel is None:
            return x
        else:
            return x, cache_last_channel, cache_last_time


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution;
            an even kernel pads the sequence with (kernel_size - 1) // 2 frames on the left
            and kernel_size // 2 frames on the right, unless conv_context_size says otherwise
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
        use_bias (bool): Use bias in all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
            Defaults to True
        norm_eps: epsilon of the normalization layer
    """

    def __init__(
        self,
        d_model,
        kernel_size,
        norm_type='batch_norm',
        conv_context_size=None,
        pointwise_activation='glu_',
        use_bias=True,
        norm_eps: float = 1e-5,
    ):
        super(ConformerConvolution, self).__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.use_bias = use_bias

        if conv_context_size is None:
            conv_context_size = [(kernel_size - 1) // 2, kernel_size // 2]
        elif conv_context_size == 'causal':
            conv_context_size = [kernel_size - 1, 0]
        elif isinstance(conv_context_size, int):
            conv_context_size = [conv_context_size, conv_context_size]
        conv_context_size = list(conv_context_size)
        if len(conv_context_size) != 2 or conv_context_size[0] + conv_context_size[1] + 1 != kernel_size:
            raise ValueError(
                f"Invalid conv_context_size {conv_context_size} for kernel_size {kernel_size}: "
                "the left and right context plus one must equal the kernel size."
            )
        # the encoder reads the context of its layers' convolution for the streaming caches
        self.conv_context_size = conv_context_size

        if pointwise_activation in activation_registry:
            self.pointwise_activation = activation_registry[pointwise_activation]()
            dw_conv_input_dim = d_model * 2

            if hasattr(self.pointwise_activation, 'inplace'):
                self.pointwise_activation.inplace = True
        else:
            self.pointwise_activation = pointwise_activation
            dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=self.use_bias,
        )

        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim, eps=norm_eps)
        elif norm_type == 'instance_norm':
            self.batch_norm = nn.InstanceNorm1d(dw_conv_input_dim, eps=norm_eps)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim, eps=norm_eps)
        elif norm_type == 'fused_batch_norm':
            self.batch_norm = FusedBatchNorm1d(dw_conv_input_dim)
        elif norm_type.startswith('group_norm'):
            num_groups = int(norm_type.replace("group_norm", ""))
            self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model, eps=norm_eps)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

    @staticmethod
    def _apply_pointwise(x, conv):
        # kernel_size=1 convs are pointwise, i.e. linear, but nn.Conv1d dispatches to much slower
        # cuBLAS kernels. squeeze(-1) on the (O, I, 1) weight is a free view, so
        # the module stays an nn.Conv1d and checkpoints are unchanged.
        return nn.functional.linear(x, conv.weight.squeeze(-1), conv.bias)

    def forward(self, x, pad_mask=None, cache=None):
        x = self._apply_pointwise(x, self.pointwise_conv1)

        # Compute the activation function or use GLU for original Conformer
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=-1)
        else:
            x = self.pointwise_activation(x)

        # the depthwise conv and its streaming cache are channel-first
        x = x.transpose(1, 2)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
        else:
            x = self.batch_norm(x)
            x = x.transpose(1, 2)

        x = self.activation(x)
        x = self._apply_pointwise(x, self.pointwise_conv2)
        if cache is None:
            return x
        else:
            return x, cache

    def reset_parameters_conv(self):
        pw1_max = pw2_max = self.d_model**-0.5
        dw_max = self.kernel_size**-0.5

        with torch.no_grad():
            nn.init.uniform_(self.pointwise_conv1.weight, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv2.weight, -pw2_max, pw2_max)
            nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            if self.use_bias:
                nn.init.uniform_(self.pointwise_conv1.bias, -pw1_max, pw1_max)
                nn.init.uniform_(self.pointwise_conv2.bias, -pw2_max, pw2_max)
                nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)


class ConformerFeedForward(nn.Module):
    """Feed-forward module of the Conformer model.

    Args:
        d_model: input and output dimension
        d_ff: hidden dimension
        dropout: dropout probability after the activation
        activation: the activation module, or its name in the activation registry
            (`swish`, `relu`, `gelu`, `squared_relu`, ...); default `Swish`
        use_bias: bias in both linear layers
    """

    def __init__(self, d_model, d_ff, dropout, activation: nn.Module | str = Swish(), use_bias=True):
        super(ConformerFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = nn.Linear(d_model, d_ff, bias=self.use_bias)
        if isinstance(activation, str):
            activation = activation_registry[activation]()
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=self.use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters_ff(self):
        ffn1_max = self.d_model**-0.5
        ffn2_max = self.d_ff**-0.5
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear2.weight, -ffn2_max, ffn2_max)
            if self.use_bias:
                nn.init.uniform_(self.linear1.bias, -ffn1_max, ffn1_max)
                nn.init.uniform_(self.linear2.bias, -ffn2_max, ffn2_max)
