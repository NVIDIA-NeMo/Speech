# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from re import X
import torch
from torch import nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F

from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.submodules.batchnorm import FusedBatchNorm1d
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.common.parts.utils import activation_registry
from nemo.core.classes.mixins import AccessMixin

from nemo.utils import logging

__all__ = ['ConformerConvolution', 'ConformerFeedForward', 'ConformerLayer']


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
        conv_context_style="regular",
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
    ):
        super(ConformerLayer, self).__init__()

        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            conv_context_style=conv_context_style,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]

        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            self.self_attn = RelPositionMultiHeadAttentionLongformer(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                att_context_size=att_context_size,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                use_bias=use_bias,
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos']"
            )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None, dcc_chunk=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
            cache_last_channel (torch.tensor) : cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : cache for convolutional layers (B, d_model, T_cache)
            dcc_chunk (int) : chunk size for dynamic chunked convolution
        Returns:
            x (torch.Tensor): (B, T, d_model)
            cache_last_channel (torch.tensor) : next cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : next cache for convolutional layers (B, d_model, T_cache)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        # import ipdb; ipdb.set_trace()

        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'rel_pos_local_attn':
            x = self.self_attn(query=x, key=x, value=x, pad_mask=pad_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'abs_pos':
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
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time, dcc_chunk=dcc_chunk)
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
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
        use_bias (bool): Use bias in all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
            Defaults to True
    """

    def __init__(
        self,
        d_model,
        kernel_size,
        conv_context_style="regular",
        norm_type='batch_norm',
        conv_context_size=None,
        pointwise_activation='glu_',
        use_bias=True,
    ):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv_context_style = conv_context_style
        self.norm_type = norm_type
        self.use_bias = use_bias

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2
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
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        elif norm_type == 'instance_norm':
            self.batch_norm = nn.InstanceNorm1d(dw_conv_input_dim)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim)
        elif norm_type == 'fused_batch_norm':
            self.batch_norm = FusedBatchNorm1d(dw_conv_input_dim)
        elif norm_type.startswith('group_norm'):
            num_groups = int(norm_type.replace("group_norm", ""))
            self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
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

    def forward(self, x, pad_mask=None, cache=None, dcc_chunk=None):
        
        if dcc_chunk is not None:
            
            if dcc_chunk and self.conv_context_style == "regular":
                raise ValueError("dcc_chunk is not supported for regular convolution context style!")
            
            # apply dynamic chunked convolution with the config (only during training)
            chunk_size = dcc_chunk
            batch_size = x.size(0)
            if isinstance(self.conv_context_size, list):
                conv_context_size = self.conv_context_size[0]
            else:
                conv_context_size = self.conv_context_size

            if self.conv_context_style == "dcc_rc":
                right_dcc_context = conv_context_size
            else:
                right_dcc_context = 0

            # define right padding for the last chunk
            if x.shape[1] % chunk_size != 0:
                final_right_padding = chunk_size - (x.shape[1] % chunk_size) + right_dcc_context
                final_chunk_padding = chunk_size - (x.shape[1] % chunk_size)
            else:
                final_right_padding = right_dcc_context
                final_chunk_padding = 0

            x = x.transpose(1, 2) # [B, T, D] -> [B, D, T]
            x = self.pointwise_conv1(x)

            # Compute the activation function or use GLU for original Conformer
            if self.pointwise_activation == 'glu_':
                x = nn.functional.glu(x, dim=1)
            else:
                x = self.pointwise_activation(x)

            # if pad_mask is not None:
            #     x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

            # logging.warning("*********"*10)
            # logging.warning(f"x.shape: {x.shape}")
            # logging.warning(f"self.conv_context_size: {self.conv_context_size}")
            # logging.warning(f"conv_context_size: {conv_context_size}")
            # logging.warning(f"right_dcc_context: {right_dcc_context}")
            # logging.warning(f"final_right_padding: {final_right_padding}")
            # logging.warning(f"final_chunk_padding: {final_chunk_padding}")
            # logging.warning(f"chunk_size: {chunk_size}")

            # raise ValueError("Stop here")

            x = F.pad(x, (conv_context_size, final_right_padding), value=0) # [batch_size, in_channels, lc+t+final_right_padding]
            # logging.warning(f"x.shape after padding 1: {x.shape}")

            # split the tensor into chunks
            x = x.unfold(2, size=conv_context_size + chunk_size + right_dcc_context, step=chunk_size)

            # logging.warning(f"conv_context_size + chunk_size + right_dcc_context: {conv_context_size + chunk_size + right_dcc_context}")
            # logging.warning(f"x.shape after unfold: {x.shape}")

            # # -> [batch_size, in_channels, num_chunks, lc+chunk_size+rpad]
            x = F.pad(x, (0, conv_context_size), value=0)
            # logging.warning(f"x.shape after padding 2: {x.shape}")

            # -> [batch_size, num_chunks, in_channels, lc+chunk_size+rpad]
            x = x.transpose(1, 2)

            # -> [batch_size * num_chunks, in_channels, lc+chunk_size+rpad]
            x = x.flatten(start_dim=0, end_dim=1)

            # we are using only weigth from depthwise convolution
            x = F.conv1d(
                x,
                weight=self.depthwise_conv.weight,
                bias=self.depthwise_conv.bias,
                stride=self.depthwise_conv.stride,
                padding=0,
                dilation=self.depthwise_conv.dilation,
                groups=self.depthwise_conv.groups,
            )

            # logging.warning(f"x.shape after depthwise conv: {x.shape}")

            if self.norm_type == "layer_norm":
                x = x.transpose(1, 2)
                x = self.batch_norm(x)
                x = x.transpose(1, 2)
            else:
                x = self.batch_norm(x)

            x = self.activation(x)
            x = self.pointwise_conv2(x)

            # -> [batch_size * num_chunks, chunk_size+right_context, out_channels]
            x = x.transpose(1, 2)

            # # -> [batch_size * num_chunks, chunk_size, out_channels]
            if self.conv_context_style == "dcc_rc":
                x = x[:, :chunk_size, :]

            # -> [batch_size, num_chunks, chunk_size, out_channels]
            x = torch.unflatten(x, dim=0, sizes=(batch_size, -1))

            # -> [batch_size, t + final_right_padding, out_channels]
            x = torch.flatten(x, start_dim=1, end_dim=2)

            # -> [batch_size, t, out_channels]
            if final_chunk_padding > 0:
                x = x[:, :-final_chunk_padding, :]

        else:
            # original Conformer convolution with standard and causal padding
            x = x.transpose(1, 2)
            x = self.pointwise_conv1(x)

            # Compute the activation function or use GLU for original Conformer
            if self.pointwise_activation == 'glu_':
                x = nn.functional.glu(x, dim=1)
            else:
                x = self.pointwise_activation(x)

            if pad_mask is not None:
                x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

            x = self.depthwise_conv(x, cache=cache)
            if cache is not None:
                x, cache = x

            if self.norm_type == "layer_norm":
                x = x.transpose(1, 2)
                x = self.batch_norm(x)
                x = x.transpose(1, 2)
            else:
                x = self.batch_norm(x)

            x = self.activation(x)
            x = self.pointwise_conv2(x)

            x = x.transpose(1, 2)


        # # apply padding mask to the final convolution output
        # if pad_mask is not None:
        #     x = x.masked_fill(pad_mask.unsqueeze(2), 0.0)

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
    """
    feed-forward module of Conformer model.
    use_bias (bool): Apply bias to all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish(), use_bias=True):
        super(ConformerFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = nn.Linear(d_model, d_ff, bias=self.use_bias)
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
