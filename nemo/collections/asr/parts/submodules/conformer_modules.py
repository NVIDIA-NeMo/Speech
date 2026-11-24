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

from nemo.collections.asr.parts.packed_sequence import PackedSlotLayout
from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.submodules.batchnorm import FusedBatchNorm1d
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
    RoPEMultiHeadAttention,
)
from nemo.collections.asr.parts.submodules.packed_attention import PackedAttentionContext
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.common.parts.utils import ModuleBlock, activation_registry, block_binds, build_module
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.mixins.adapter_mixin_strategies import ResidualAddAdapterStrategy

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
        autocast_residual: under autocast, keep the residual stream and the layer norm inputs in the autocast dtype
            instead of float32, which halves what the plain layer norms save for the backward (another norm block
            saves what it saved before); default False, the float32 stream.
            With it the layer returns the autocast dtype instead of float32 (so do the encoder's output, its InterCTC
            captures and the ``save_encoder_tensors`` capture), the residual additions and the norm inputs are rounded
            to that dtype, and a branch output or adapter output of another dtype is cast back to it; streaming caches
            keep the dtype of the cache they came in with. Meant for bfloat16: float16's range may overflow a stream
            that float32 carries. Without autocast the option has no effect.
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
        autocast_residual: bool = False,
    ):
        super(ConformerLayer, self).__init__()

        self.autocast_residual = autocast_residual
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
        dtype = self._residual_dtype(x)
        if dtype is not None:
            x = x.to(dtype)
        residual = x
        x = self._norm(self.norm_feed_forward1, x, dtype)
        x = self.feed_forward1(x)
        residual = residual + self._dropout(x, dtype) * self.fc_factor

        x = self._norm(self.norm_self_att, residual, dtype)
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

        residual = residual + self._dropout(x, dtype)

        if self.is_adapter_available():
            # Call the MHA adapters
            pack_input = {
                'x': residual,
                'loc': 'mha',
                'att_mask': att_mask,
                'pos_emb': pos_emb,
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            residual = self._stream(pack_input['x'], dtype)

        x = self._norm(self.norm_conv, residual, dtype)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self._dropout(x, dtype)

        x = self._norm(self.norm_feed_forward2, residual, dtype)
        x = self.feed_forward2(x)
        residual = residual + self._dropout(x, dtype) * self.fc_factor

        x = self._norm(self.norm_out, residual, dtype)

        if self.is_adapter_available():
            # Call the adapters
            pack_input = {
                'x': x,
                'loc': 'post',
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            x = self._stream(pack_input['x'], dtype)

        if self.is_access_enabled(getattr(self, "model_guid", None)) and self.access_cfg.get(
            'save_encoder_tensors', False
        ):
            self.register_accessible_tensor(name='encoder', tensor=x)
        if cache_last_channel is None:
            return x
        else:
            return x, cache_last_channel, cache_last_time

    def forward_packed(
        self,
        x: torch.Tensor,
        *,
        attention_context: PackedAttentionContext,
        conv_layout: PackedSlotLayout,
        pos_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The layer over token-flat input of all utterances, with explicit boundaries instead of padding masks.

        Streaming caches, local attention with global tokens and the ``save_encoder_tensors`` capture
        are not supported in this form.
        Enabled adapters run on the token-flat states at the places where ``forward`` runs them,
        see ``sequence_packed_adapters_unsupported_reason`` for what is refused.

        Args:
            x: input signals, shape ``(T_total, d_model)``.
            attention_context: packing metadata for the self-attention, with the limited attention context if any.
            conv_layout: gapped layout for the depthwise convolution, see ``ConformerConvolution.forward_packed``.
            pos_emb: relative positional embedding ``(1, 2 * max_seqlen - 1, d_model)`` for ``rel_pos`` attention,
                the one of the window for ``rel_pos_local_attn``, else None.

        Returns:
            The layer output, shape ``(T_total, d_model)``.
        """
        reason = self.sequence_packed_adapters_unsupported_reason()
        if reason is not None:
            raise NotImplementedError(reason)

        dtype = self._residual_dtype(x)
        if dtype is not None:
            x = x.to(dtype)
        residual = x
        x = self._norm(self.norm_feed_forward1, x, dtype)
        x = self.feed_forward1(x)
        residual = residual + self._dropout(x, dtype) * self.fc_factor

        x = self._norm(self.norm_self_att, residual, dtype)
        x = self.self_attn.forward_packed(x, attention_context, pos_emb=pos_emb)
        residual = residual + self._dropout(x, dtype)

        if self.is_adapter_available():
            pack_input = {
                'x': residual,
                'loc': 'mha',
                'pos_emb': pos_emb,
                'attention_context': attention_context,
            }
            residual = self._stream(self.forward_enabled_adapters(pack_input)['x'], dtype)

        x = self._norm(self.norm_conv, residual, dtype)
        x = self.conv.forward_packed(x, conv_layout)
        residual = residual + self._dropout(x, dtype)

        x = self._norm(self.norm_feed_forward2, residual, dtype)
        x = self.feed_forward2(x)
        residual = residual + self._dropout(x, dtype) * self.fc_factor

        x = self._norm(self.norm_out, residual, dtype)
        if self.is_adapter_available():
            pack_input = {'x': x, 'loc': 'post'}
            x = self._stream(self.forward_enabled_adapters(pack_input)['x'], dtype)
        return x

    def _residual_dtype(self, x: torch.Tensor) -> torch.dtype | None:
        """The autocast dtype the residual stream follows with ``autocast_residual``, else None for the float32 stream."""
        if not self.autocast_residual or not torch.amp.is_autocast_available(x.device.type):
            return None
        if torch.is_autocast_enabled(x.device.type):
            return torch.get_autocast_dtype(x.device.type)
        return None

    @staticmethod
    def _stream(x: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
        """A tensor that joins the residual stream, in the stream's dtype.

        A branch or an adapter may return another dtype than the stream's (the attention avoids float16 autocast
        and returns bfloat16 or float32, an adapter's own norm returns float32), which would promote the stream.
        """
        return x if dtype is None else x.to(dtype)

    def _dropout(self, x: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
        """Dropout of a branch output, in the dtype of the residual stream."""
        return self.dropout(self._stream(x, dtype))

    def _norm(self, norm: nn.Module, x: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
        """A layer norm of the residual stream; on the autocast-dtype stream the plain norms run on that dtype.

        ``torch.nn.LayerNorm`` and ``torch.nn.RMSNorm`` themselves run outside autocast through their functional
        forms with the parameters cast to the stream's dtype, since their fused kernels take no float32 parameters
        with a half-precision input (the composite fallback would upcast and save float32 copies).
        Another norm block, a subclass with its own ``forward`` included, runs under autocast as on the float32
        stream, so its own kernels decide what it saves (a block around a plain norm saves what it saved before),
        and its output returns to the stream's dtype.
        """
        if dtype is None:
            return norm(x)
        # the keyword form: Dynamo binds autocast's arguments by name when it traces the context manager;
        # a subclass that keeps the plain forward takes the plain path
        if isinstance(norm, LayerNorm) and type(norm).forward is LayerNorm.forward:
            with torch.autocast(device_type=x.device.type, enabled=False):
                weight = norm.weight.to(dtype) if norm.weight is not None else None
                bias = norm.bias.to(dtype) if norm.bias is not None else None
                return nn.functional.layer_norm(x, norm.normalized_shape, weight, bias, norm.eps)
        if isinstance(norm, nn.RMSNorm) and type(norm).forward is nn.RMSNorm.forward:
            with torch.autocast(device_type=x.device.type, enabled=False):
                weight = norm.weight.to(dtype) if norm.weight is not None else None
                return nn.functional.rms_norm(x, norm.normalized_shape, weight, norm.eps)
        return norm(x).to(dtype)

    def sequence_packed_adapters_unsupported_reason(self) -> str | None:
        """Why ``forward_packed`` cannot run the enabled adapters of this layer, or None when it can.

        Linear adapters are pointwise and run on the token-flat states as they are,
        and the Conformer attention adapters attend within the utterance boundaries.
        Without a sequence-packed form: the attention adapters of the Transformer blocks,
        an attention adapter in a layer whose attention is neither ``rel_pos`` nor ``abs_pos``
        (``rope``, ``rel_pos_local_attn``), which the padded forward refuses as well,
        and in training mode a residual strategy with an auxiliary L2 loss (``l2_lambda`` above zero),
        which sums over the frames of a padded batch row and averages over the rows;
        it is refused whether or not the adapter's parameters and access configuration let the loss be computed.
        Disabled adapters do not run and are never a reason.
        """
        # a function-level import as in AttentionAdapterModuleMixin, to prevent a circular import
        from nemo.collections.asr.modules.transformer import transformer_modules

        for name in self.get_enabled_adapters():
            adapter = self.adapter_layer[name]
            if isinstance(adapter, transformer_modules.MultiHeadAttention):
                return f"The adapter {name!r} ({type(adapter).__name__}) has no sequence-packed form."
            if isinstance(adapter, MultiHeadAttention) and self.self_attention_model not in ('rel_pos', 'abs_pos'):
                # forward_single_enabled_adapter_ raises for it, in the padded forward as well
                return (
                    f"The attention adapter {name!r} runs with rel_pos and abs_pos attention only, "
                    f"not with self_attention_model={self.self_attention_model!r}."
                )
            strategy = adapter.adapter_strategy
            if self.training and isinstance(strategy, ResidualAddAdapterStrategy) and strategy.l2_lambda > 0.0:
                return (
                    f"The auxiliary L2 loss of the adapter {name!r} (l2_lambda={strategy.l2_lambda}) averages over "
                    "padded batch rows and has no sequence-packed form: set l2_lambda: 0 on its adapter_strategy."
                )
        return None


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
        # contiguous() avoids a strided-batched fp16 GEMM that hits illegal memory access on sm_120.
        return nn.functional.linear(x.contiguous(), conv.weight.squeeze(-1), conv.bias)

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

    def forward_packed(self, x: torch.Tensor, layout: PackedSlotLayout) -> torch.Tensor:
        """The module over token-flat input of all utterances.

        Batch normalization takes its statistics from the valid frames only.
        The padded forward includes the padded frames,
        which hold the bias and the boundary tails of the depthwise convolution,
        so the two forms differ in training mode by the share of padding in the batch and agree in evaluation mode.
        Instance and group normalization take the statistics of every utterance from its own frames,
        in both modes (see ``_utterance_norm_packed``),
        so an utterance gets the result of the padded forward on that utterance alone;
        the padded forward of a batch counts the padded frames of the shorter utterances into their statistics.
        Without any frame (a rank whose share of a batch is empty) the convolution is skipped
        and its parameters keep a zero gradient edge;
        a single valid frame, which ``BatchNorm1d`` refuses to normalize in training mode,
        is normalized to the bias (see ``_batch_norm_one_frame``).

        Args:
            x: input signals, shape ``(T_total, d_model)``.
            layout: gapped layout of the utterances with at least ``kernel_size - 1`` empty frames after each,
                so the depthwise convolution reads zeros beyond an utterance,
                as it does beyond the masked padding of a padded batch.

        Returns:
            The module output, shape ``(T_total, d_model)``.
        """
        x = self._apply_pointwise(x, self.pointwise_conv1)
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=-1)
        else:
            x = self.pointwise_activation(x)

        if x.shape[0] == 0:
            # no frame to convolve, and a batch without utterances has a layout without slots,
            # which the convolution cannot take; the kernel keeps its zero gradient edge
            # as the empty matmuls keep the other parameters'
            anchor = x.new_zeros(())
            for parameter in self.depthwise_conv.parameters():
                anchor = anchor + parameter.sum().to(anchor.dtype) * 0.0
            x = x + anchor
        else:
            # one channel-first sequence over the gapped slots; the empty slots are the zero padding between utterances
            slots = layout.scatter(x).transpose(0, 1).unsqueeze(0)
            slots = self.depthwise_conv(slots)
            x = layout.gather(slots.squeeze(0).transpose(0, 1))

        if self.norm_type == 'batch_norm' and self.batch_norm.training and x.shape[0] == 1:
            x = self._batch_norm_one_frame(x)
        elif self.norm_type == 'instance_norm' or self.norm_type.startswith('group_norm'):
            x = self._utterance_norm_packed(x, layout)
        else:
            x = self.batch_norm(x)
        x = self.activation(x)
        return self._apply_pointwise(x, self.pointwise_conv2)

    def sequence_packed_unsupported_reason(self) -> str | None:
        """Why ``forward_packed`` cannot run this module, or None when it can.

        The encoder asks the convolution block of every layer; this module runs every norm type.
        """
        return None

    def _batch_norm_one_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Training-mode batch normalization of a batch that holds one valid frame, which ``BatchNorm1d`` refuses.

        The frame is its own batch mean, so its normalized value is zero and the output is the bias,
        the limit of the training formula for a vanishing batch variance.
        The running mean moves towards the frame as it does for any batch,
        and the running variance keeps its value, since one frame has no unbiased variance.
        The frame and the weight keep their (zero) gradient edges.
        """
        batch_norm = self.batch_norm
        if batch_norm.track_running_stats and batch_norm.running_mean is not None:
            with torch.no_grad():
                batch_norm.num_batches_tracked += 1
                momentum = batch_norm.momentum
                if momentum is None:  # cumulative moving average
                    momentum = 1.0 / float(batch_norm.num_batches_tracked)
                frame = x[0].to(batch_norm.running_mean.dtype)
                batch_norm.running_mean.mul_(1.0 - momentum).add_(frame, alpha=momentum)
        return x * 0.0 + batch_norm.weight.to(x.dtype) * 0.0 + batch_norm.bias.to(x.dtype)

    def _utterance_norm_packed(self, x: torch.Tensor, layout: PackedSlotLayout) -> torch.Tensor:
        """Instance or group normalization of token-flat input ``(T_total, C)``, per utterance.

        Mean and variance of a group are taken over the frames of one utterance and the channels of the group,
        as ``InstanceNorm1d`` (one channel per group) and ``GroupNorm`` take them over ``(C / G, T)`` of a batch row.
        float16 and bfloat16 input is normalized in float32 and cast back:
        without autocast half-precision squares overflow and the sums lose the small terms,
        and in either case a mean rounded to half precision before the subtraction
        costs several percent of the output where the mean is large against the spread.
        A group with a single value, one frame of a one-channel group, which both modules refuse,
        has no variance and is normalized to the bias.
        """
        norm = self.batch_norm
        num_groups = norm.num_groups if isinstance(norm, nn.GroupNorm) else x.shape[1]
        # the lengths and utterance indices the layout was built from, the same tensors for every layer;
        # no step here reads a value back from the device
        sequence_ids, lengths = layout.sequence_ids, layout.lengths
        if sequence_ids is None or lengths is None:
            # a layout built without them: utterances without frames can share their slot start
            # with the next utterance, so a token belongs to the last utterance that starts at or before its slot
            sequence_ids = torch.bucketize(layout.token_slots, layout.slot_offsets[1:], right=True)
            lengths = sequence_ids.new_zeros(layout.slot_offsets.numel())
            lengths = lengths.index_add_(0, sequence_ids, torch.ones_like(sequence_ids))
        counts = lengths.clamp(min=1) * (x.shape[1] // num_groups)

        def utterance_mean(values: torch.Tensor) -> torch.Tensor:
            """(T_total, G, C / G) to the mean over the utterance's frames and the group, per token (T_total, G, 1)."""
            # the tokens of an utterance are consecutive; segment_reduce adds them in a fixed order,
            # which index_add on CUDA does not, and the lengths are those of the layout, so they need no check
            sums = torch.segment_reduce(values.sum(dim=2), "sum", lengths=lengths, unsafe=True)
            return _UtteranceValuePerToken.apply(sums / counts.unsqueeze(1), sequence_ids, lengths).unsqueeze(2)

        dtype = x.dtype
        x = x.to(torch.promote_types(dtype, torch.float32)).unflatten(1, (num_groups, -1))
        x = x - utterance_mean(x)
        x = (x * torch.rsqrt(utterance_mean(x.square()) + norm.eps)).flatten(1)
        if norm.weight is not None:
            x = x * norm.weight.to(x.dtype) + norm.bias.to(x.dtype)
        return x.to(dtype)

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


class _UtteranceValuePerToken(torch.autograd.Function):
    """One row per utterance ``(B, ...)`` repeated for every token of the utterance, ``(T_total, ...)``.

    The same values as ``values.index_select(0, sequence_ids)``.
    Its backward adds the token gradients of an utterance with ``segment_reduce``, in a fixed order,
    where the backward of ``index_select`` is an ``index_add``, whose result on CUDA changes from run to run.
    """

    @staticmethod
    def forward(ctx, values: torch.Tensor, sequence_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lengths)
        return values.index_select(0, sequence_ids)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (lengths,) = ctx.saved_tensors
        return torch.segment_reduce(grad_output, "sum", lengths=lengths, unsafe=True), None, None
