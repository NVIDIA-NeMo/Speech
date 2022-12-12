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
#
import torch
from torch import nn as nn
from torch.nn import LayerNorm

from nemo.collections.asr.parts.submodules.conformer_modules import ConformerConvolution, ConformerFeedForward
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
)
from nemo.collections.common.parts import adapter_modules
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin

__all__ = ['SqueezeformerLayer', 'ConformerFeedForward', 'SqueezeformerLayer']


class ScaleBiasLayer(torch.nn.Module):
    """
    Computes an affine transformation y = x * scale + bias, either learned via adaptive weights, or fixed.
    Efficient alternative to LayerNorm where we can avoid computing the mean and variance of the input, and
    just rescale the output of the previous layer.

    Args:
        d_model (int): input dimension of layer.
        adaptive_scale (bool): whether to learn the affine transformation parameters or not. If set to False,
            the scale is fixed to 1 and bias to 0, effectively performing a No-Op on the input.
            This is done for export compatibility.
    """

    def __init__(self, d_model: int, adaptive_scale: bool):
        super().__init__()
        self.adaptive_scale = adaptive_scale
        if adaptive_scale:
            self.scale = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer('scale', torch.ones(d_model), persistent=True)
            self.register_buffer('bias', torch.zeros(d_model), persistent=True)

    def forward(self, x):
        scale = self.scale.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)
        return x * scale + bias


class SqueezeformerLayer(torch.nn.Module, AdapterModuleMixin, AccessMixin):
    """A single block of the Squeezeformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        adaptive_scale (bool): Whether to scale the inputs to each component by affine `scale` and `bias` layer.
            Or use a fixed scale=1 and bias=0.
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        adaptive_scale: bool = True,
    ):
        super().__init__()

        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 1.0
        self.adaptive_scale = adaptive_scale

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.feed_forward1_scale = ScaleBiasLayer(d_model=d_model, adaptive_scale=adaptive_scale)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model, kernel_size=conv_kernel_size, norm_type=conv_norm_type, pointwise_activation='swish'
        )
        self.conv_scale = ScaleBiasLayer(d_model=d_model, adaptive_scale=adaptive_scale)

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att)
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'abs_pos']"
            )
        self.self_attn_scale = ScaleBiasLayer(d_model=d_model, adaptive_scale=adaptive_scale)

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.feed_forward2_scale = ScaleBiasLayer(d_model=d_model, adaptive_scale=adaptive_scale)

        self.dropout = nn.Dropout(dropout)
        # self.norm_out = LayerNorm(d_model)

        # initialize parameters properly
        self.reset_parameters()

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x

        x = self.self_attn_scale(x)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask)
        else:
            x = None
        x = residual + self.dropout(x)
        x = self.norm_self_att(x)
        residual = x

        if self.is_adapter_available():
            # Call the MHA adapters
            pack_ip = {
                'x': residual,
                'loc': 'mha',
                'att_mask': att_mask,
                'pos_emb': pos_emb,
            }
            pack_ip = self.forward_enabled_adapters(pack_ip)
            x = pack_ip['x']

        x = self.feed_forward1_scale(x)
        x = self.feed_forward1(x)
        x = residual + self.dropout(x) * self.fc_factor
        x = self.norm_feed_forward1(x)
        residual = x

        x = self.conv_scale(x)
        x = self.conv(x, pad_mask)
        x = residual + self.dropout(x)
        x = self.norm_conv(x)
        residual = x

        x = self.feed_forward2_scale(x)
        x = self.feed_forward2(x)
        x = residual + self.dropout(x) * self.fc_factor
        x = self.norm_feed_forward2(x)

        if self.is_adapter_available():
            # Call the adapters
            pack_ip = {
                'x': x,
                'loc': 'post',
            }
            pack_ip = self.forward_enabled_adapters(pack_ip)
            x = pack_ip['x']

        if self.is_access_enabled() and self.access_cfg.get('save_encoder_tensors', False):
            self.register_accessible_tensor(name='encoder', tensor=x)

        return x

    def forward_single_enabled_adapter_(
        self,
        input: dict,
        adapter_module: torch.nn.Module,
        *,
        adapter_name: str,
        adapter_strategy: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractAdapterStrategy',
    ):
        """
        Perform the forward step of a single adapter module on some input data.

        **Note**: Subclasses can override this method to accommodate more complicate adapter forward steps.

        Args:
            input: Dictionary of packed tensors. The dict should contain at least
                `x`: output tensor
                `loc`: Semantic location in module where this adapter was called
                `att_mask`: Optional, Attention mask
                `pos_emb`: Optional, Positional Embedding for Relative Positional Encoding.
                The output tensor of the calling module is the input to the first adapter, whose output
                is then chained to the next adapter until all adapters are consumed.
            adapter_module: The adapter module that is currently required to perform the forward pass.
            adapter_name: The resolved name of the adapter that is undergoing the current forward pass.
            adapter_strategy: A subclass of `AbstractAdapterStrategy`, that determines how the
                output of the adapter should be merged with the input, or if it should be merged at all.

        Returns:
            The result tensor, after the current active adapter has finished its forward pass.
        """
        # (input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin')
        x = input['x']
        loc = input['loc']
        att_mask = input.get('att_mask', None)
        pos_emb = input.get('pos_emb', None)

        if isinstance(adapter_module, adapter_modules.LinearAdapter) and loc == 'post':
            output = adapter_strategy(x, adapter_module, module=self)

        elif isinstance(adapter_module, MultiHeadAttention) and loc == 'mha':
            if self.self_attention_model == 'rel_pos':
                x = dict(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
                output = adapter_strategy(x, adapter_module, module=self)

            elif self.self_attention_model == 'abs_pos':
                x = dict(query=x, key=x, value=x, mask=att_mask)
                output = adapter_strategy(x, adapter_module, module=self)

            else:
                raise ValueError(f"Unsupported value of self_attention_model , provided {self.self_attention_model}!")

        else:
            # No adapter compatible, skip
            output = x

        input['x'] = output

        return input

    def reset_parameters(self):
        # Used for Squeezeformer initialization only
        self.feed_forward1.reset_parameters_ff()
        self.feed_forward2.reset_parameters_ff()
        self.conv.reset_parameters_conv()
