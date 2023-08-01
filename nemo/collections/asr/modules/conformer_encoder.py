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

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Set

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, open_dict

from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    LocalAttRelPositionalEncoding,
    MultiHeadAttention,
    PositionalEncoding,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling,
    StackingSubsampling,
    SubsamplingReductionModule,
)
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.asr.parts.utils.regularization_utils import compute_stochastic_depth_drop_probs
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, ChannelType, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging

__all__ = ['ConformerEncoder']


class ConformerEncoder(NeuralModule, StreamingEncoder, Exportable, AccessMixin):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling, choices=['vggnet', 'striding']
            Defaults to striding.
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_chunking_factor(int): optionally, force chunk inputs (helpful for large inputs)
            Should be power of 2, 1 (auto-chunking, default), or -1 (no chunking)
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to -1 which would set it to d_model.
        reduction (str, Optional): the method of reduction, choices=['pooling', 'striding']. If no value
            is passed, then no reduction is performed and the models runs with the original 4x subsampling.
        reduction_position (int, Optional): the index of the layer to apply reduction. If -1, apply reduction
            at the end.
        reduction_factor (int): the reduction factor which should be either 1 or a power of 2
            Defaults to 1.
        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
                overlapping chunks. Attention context is determined by att_context_size parameter.
            'abs_pos': absolute positional embedding and Transformer
            Default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaults to 5000
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        att_context_size (List[Union[List[int],int]]): specifies the context sizes on each side. Each context size should be a list of two integers like [100,100].
            A list of context sizes like [[100,100],[100,50]] can also be passed. -1 means unlimited context.
            Defaults to [-1,-1]
        att_context_probs (List[float]): a list of probabilities of each one of the att_context_size when a list of them is passed. If not specified, uniform distribution is being used.
            Defaults to None
        att_context_style (str): 'regular' or 'chunked_limited'.
            Defaults to 'regular'
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by sqrt(d_model)
            Defaults to True.
        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        conv_norm_type (str): the type of the normalization in the convolutional modules
            Defaults to 'batch_norm'.
        conv_context_size (list): it can be"causal" or a list of two integers while conv_context_size[0]+conv_context_size[1]+1==conv_kernel_size.
            None means [(conv_kernel_size-1)//2, (conv_kernel_size-1)//2], and 'causal' means [(conv_kernel_size-1), 0].
            Defaults to None.
        conv_dual_mode (bool): specifies if convolution should be dual mode when dual_offline mode is being used. When enables, the left half of the convolution kernel would get masked in streaming cases.
            Defaults to False
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
        dropout_pre_encoder (float): the dropout rate used before the encoder
            Defaults to 0.1.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
        stochastic_depth_drop_prob (float): if non-zero, will randomly drop
            layers during training. The higher this value, the more often layers
            are dropped. Defaults to 0.0.
        stochastic_depth_mode (str): can be either "linear" or "uniform". If
            set to "uniform", all layers have the same probability of drop. If
            set to "linear", the drop probability grows linearly from 0 for the
            first layer to the desired value for the final layer. Defaults to
            "linear".
        stochastic_depth_start_layer (int): starting layer for stochastic depth.
            All layers before this will never be dropped. Note that drop
            probability will be adjusted accordingly if mode is "linear" when
            start layer is > 1. Defaults to 1.
        global_tokens (int): number of tokens to be used for global attention.
            Only relevant if self_attention_model is 'rel_pos_local_attn'.
            Defaults to 0.
        global_tokens_spacing (int): how far apart the global tokens are
            Defaults to 1.
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.

    """

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        if self.export_cache_support:
            window_size = max_dim
            if self.streaming_cfg is not None:
                if isinstance(self.streaming_cfg.chunk_size, list):
                    chunk_size = self.streaming_cfg.chunk_size[1]
                else:
                    chunk_size = self.streaming_cfg.chunk_size
                if isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size[1]
                else:
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size
                window_size = chunk_size + pre_encode_cache_size
            input_example = torch.randn(max_batch, self._feat_in, window_size, device=dev)
            input_example_length = torch.randint(
                window_size // 4, window_size, (max_batch,), device=dev, dtype=torch.int64
            )
            cache_last_channel, cache_last_time, cache_last_channel_len = self.get_initial_cache_state(
                batch_size=max_batch, device=dev, max_dim=max_dim
            )
            all_input_example = tuple(
                [
                    input_example,
                    input_example_length,
                    cache_last_channel.transpose(0, 1),
                    cache_last_time.transpose(0, 1),
                    cache_last_channel_len,
                ]
            )
        else:
            input_example = torch.randn(max_batch, self._feat_in, max_dim, device=dev)
            input_example_length = torch.randint(max_dim // 4, max_dim, (max_batch,), device=dev, dtype=torch.int64)
            all_input_example = tuple([input_example, input_example_length])

        return all_input_example

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def input_types_for_export(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel": NeuralType(('B', 'D', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time": NeuralType(('B', 'D', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel_next": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time_next": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_next_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def output_types_for_export(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel_next": NeuralType(('B', 'D', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time_next": NeuralType(('B', 'D', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_next_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def disabled_deployment_input_names(self):
        if not self.export_cache_support:
            return set(["cache_last_channel", "cache_last_time", "cache_last_channel_len"])
        else:
            return set()

    @property
    def disabled_deployment_output_names(self):
        if not self.export_cache_support:
            return set(["cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"])
        else:
            return set()

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style='regular',
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
    ):
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        self.self_attention_model = self_attention_model
        self.global_tokens = global_tokens
        self.global_attn_separate = global_attn_separate
        self.global_tokens_spacing = global_tokens_spacing

        # Setting up the att_context_size
        (
            self.att_context_size_all,
            self.att_context_size,
            self.att_context_probs,
            self.conv_context_size,
        ) = self._calc_context_sizes(
            att_context_style=att_context_style,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            conv_context_size=conv_context_size,
            conv_kernel_size=conv_kernel_size,
        )

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        # Subsampling
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                # stacking_norm has an extra layer norm after stacking comparing to stacking
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    norm=True if subsampling == 'stacking_norm' else False,
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                    activation=nn.ReLU(True),
                    is_causal=causal_downsampling,
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        # Reduction
        if reduction and reduction_factor > 1:
            assert reduction_position >= -1 and reduction_position < n_layers
            self.reduction_subsampling = SubsamplingReductionModule(
                reduction=reduction, d_model=d_model, reduction_factor=reduction_factor,
            )
            self.reduction_position = reduction_position
        else:
            self.reduction_subsampling = None
            self.reduction_position = None

        self._feat_out = d_model

        # Biases for relative positional encoding
        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        # Positional encodings
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            if max(att_context_size) <= 0:
                raise ValueError("When using local attention, context size must be set > 0")
            self.pos_enc = LocalAttRelPositionalEncoding(
                att_context_size=att_context_size,
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout_pre_encoder, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                conv_context_size=self.conv_context_size,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_context_size=self.att_context_size,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

        self.setup_streaming_params()
        self.export_cache_support = False

        self.layer_drop_probs = compute_stochastic_depth_drop_probs(
            len(self.layers), stochastic_depth_drop_prob, stochastic_depth_mode, stochastic_depth_start_layer
        )
        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_capture_at_layers = None

    def forward_for_export(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        if cache_last_channel is not None:
            cache_last_channel = cache_last_channel.transpose(0, 1)
            cache_last_time = cache_last_time.transpose(0, 1)

        rets = self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        rets = self.streaming_post_process(rets, keep_all_outputs=False)
        if len(rets) == 2:
            return rets
        elif rets[2] is None and rets[3] is None and rets[4] is None:
            return (rets[0], rets[1])
        else:
            return (
                rets[0],
                rets[1],
                rets[2].transpose(0, 1),
                rets[3].transpose(0, 1),
                rets[4],
            )

    def streaming_post_process(self, rets, keep_all_outputs=True):
        if len(rets) == 2:
            return rets[0], rets[1], None, None, None

        (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len) = rets

        if cache_last_channel_next is not None and self.streaming_cfg.last_channel_cache_size >= 0:
            if self.streaming_cfg.last_channel_cache_size > 0:
                cache_last_channel_next = cache_last_channel_next[
                    :, :, -self.streaming_cfg.last_channel_cache_size :, :
                ]

        if self.streaming_cfg.valid_out_len > 0 and (not keep_all_outputs or self.att_context_style == "regular"):
            encoded = encoded[:, :, : self.streaming_cfg.valid_out_len]
            encoded_len = torch.clamp(encoded_len, max=self.streaming_cfg.valid_out_len)

        return (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len)

    @typecheck()
    def forward(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

    def forward_internal(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)

        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
            length = length.to(torch.int64)
            # self.streaming_cfg is set by setup_streaming_cfg(), called in the init
            if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        if self.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
                cache_last_channel_next.append(cache_last_channel_cur)
                cache_last_time_next.append(cache_last_time_cur)

            # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                # adjusting to match expectation
                if should_drop:
                    # that's not efficient, but it's hard to implement distributed
                    # version of dropping layers without deadlock or random seed meddling
                    # so multiplying the signal by 0 to ensure all weights get gradients
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    # not doing this operation if drop prob is 0 as it's identity in that case
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                # Don't update the audio_signal here because then it will again scale the audio_signal
                # and cause an increase in the WER
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )

            # saving tensors if required for interctc loss
            if self.is_access_enabled():
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    if self.out_proj is not None:
                        lth_audio_signal = self.out_proj(audio_signal)
                    # shape is the same as the shape of audio_signal output, i.e. [B, D, T]
                    self.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )
                    self.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return audio_signal, length

    def update_max_seq_length(self, seq_length: int, device):
        # Find global max audio length across all nodes
        if torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def set_max_audio_length(self, max_audio_length):
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        self.pos_enc.extend_pe(max_audio_length, device)

    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        if self.self_attention_model != "rel_pos_local_attn":
            att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)

            if self.att_context_style == "regular":
                if att_context_size[0] >= 0:
                    att_mask = att_mask.triu(diagonal=-att_context_size[0])
                if att_context_size[1] >= 0:
                    att_mask = att_mask.tril(diagonal=att_context_size[1])
            elif self.att_context_style == "chunked_limited":
                # When right context is unlimited, just the left side of the masking need to get updated
                if att_context_size[1] == -1:
                    if att_context_size[0] >= 0:
                        att_mask = att_mask.triu(diagonal=-att_context_size[0])
                else:
                    chunk_size = att_context_size[1] + 1
                    # left_chunks_num specifies the number of chunks to be visible by each chunk on the left side
                    if att_context_size[0] >= 0:
                        left_chunks_num = att_context_size[0] // chunk_size
                    else:
                        left_chunks_num = 10000

                    chunk_idx = torch.arange(0, max_audio_length, dtype=torch.int, device=att_mask.device)
                    chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                    diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                    chunked_limited_mask = torch.logical_and(
                        torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
                    )
                    att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))
        else:
            att_mask = None

        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            # pad_mask_for_att_mask is the mask which helps to ignore paddings
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
            pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
            # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
            att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def enable_pad_mask(self, on=True):
        # On inference, user may choose to disable pad mask
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask

    def _calc_context_sizes(
        self, att_context_size, att_context_probs, att_context_style, conv_context_size, conv_kernel_size
    ):
        # convert att_context_size to a standard list of lists
        if att_context_size:
            att_context_size_all = list(att_context_size)
            if isinstance(att_context_size_all[0], int):
                att_context_size_all = [att_context_size_all]
            for i, att_cs in enumerate(att_context_size_all):
                if isinstance(att_cs, ListConfig):
                    att_context_size_all[i] = list(att_cs)
                if att_context_style == "chunked_limited":
                    if att_cs[0] > 0 and att_cs[0] % (att_cs[1] + 1) > 0:
                        raise ValueError(f"att_context_size[{i}][0] % (att_context_size[{i}][1] + 1) should be zero!")
                    if att_cs[1] < 0 and len(att_context_size_all) <= 1:
                        raise ValueError(
                            f"Right context (att_context_size[{i}][1]) can not be unlimited for chunked_limited style!"
                        )
        else:
            att_context_size_all = [[-1, -1]]

        if att_context_probs:
            if len(att_context_probs) != len(att_context_size_all):
                raise ValueError("The size of the att_context_probs should be the same as att_context_size.")
            att_context_probs = list(att_context_probs)
            if sum(att_context_probs) != 1:
                raise ValueError(
                    "The sum of numbers in att_context_probs should be equal to one to be a distribution."
                )
        else:
            att_context_probs = [1.0 / len(att_context_size_all)] * len(att_context_size_all)

        if conv_context_size is not None:
            if isinstance(conv_context_size, ListConfig):
                conv_context_size = list(conv_context_size)
            if not isinstance(conv_context_size, list) and not isinstance(conv_context_size, str):
                raise ValueError(
                    f"Invalid conv_context_size! It should be the string 'causal' or a list of two integers."
                )
            if conv_context_size == "causal":
                conv_context_size = [conv_kernel_size - 1, 0]
            else:
                if conv_context_size[0] + conv_context_size[1] + 1 != conv_kernel_size:
                    raise ValueError(f"Invalid conv_context_size: {self.conv_context_size}!")
        else:
            conv_context_size = [(conv_kernel_size - 1) // 2, (conv_kernel_size - 1) // 2]
        return att_context_size_all, att_context_size_all[0], att_context_probs, conv_context_size

    def set_default_att_context_size(self, att_context_size):
        if att_context_size not in self.att_context_size_all:
            logging.warning(
                f"att_context_size={att_context_size} is not among the list of the supported look-aheads: {self.att_context_size_all}"
            )
        if att_context_size is not None:
            self.att_context_size = att_context_size

    def setup_streaming_params(
        self,
        chunk_size: int = None,
        shift_size: int = None,
        left_chunks: int = None,
        att_context_size: list = None,
        max_context: int = 10000,
    ):
        """
            This function sets the needed values and parameters to perform streaming. The configuration would be stored in self.streaming_cfg.
            The streaming configuration is needed to simulate streaming inference.
            Args:
                chunk_size (int): overrides the chunk size
                shift_size (int): overrides the shift size for chunks
                left_chunks (int): overrides the number of left chunks visible to each chunk
                max_context (int): the value used for the cache size of last_channel layers if left context is set to infinity (-1)
                    Defaults to -1 (means feat_out is d_model)
        """
        streaming_cfg = CacheAwareStreamingConfig()

        # When att_context_size is not specified, it uses the default_att_context_size
        if att_context_size is None:
            att_context_size = self.att_context_size

        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError("chunk_size needs to be a number larger or equal to one.")
            lookahead_steps = chunk_size - 1
            streaming_cfg.cache_drop_size = chunk_size - shift_size
        elif self.att_context_style == "chunked_limited":
            lookahead_steps = att_context_size[1]
            streaming_cfg.cache_drop_size = 0
        elif self.att_context_style == "regular":
            lookahead_steps = att_context_size[1] * self.n_layers + self.conv_context_size[1] * self.n_layers
            streaming_cfg.cache_drop_size = lookahead_steps
        else:
            streaming_cfg.cache_drop_size = 0
            lookahead_steps = None

        if chunk_size is None:
            streaming_cfg.last_channel_cache_size = att_context_size[0] if att_context_size[0] >= 0 else max_context
        else:
            if left_chunks is None:
                raise ValueError("left_chunks can not be None when chunk_size is set.")
            streaming_cfg.last_channel_cache_size = left_chunks * chunk_size

        if hasattr(self.pre_encode, "get_sampling_frames"):
            sampling_frames = self.pre_encode.get_sampling_frames()
        else:
            sampling_frames = 0

        if isinstance(sampling_frames, list):
            streaming_cfg.chunk_size = [
                sampling_frames[0] + self.subsampling_factor * lookahead_steps,
                sampling_frames[1] + self.subsampling_factor * lookahead_steps,
            ]
        else:
            streaming_cfg.chunk_size = sampling_frames * (1 + lookahead_steps)

        if isinstance(sampling_frames, list):
            streaming_cfg.shift_size = [
                sampling_frames[0] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
                sampling_frames[1] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
            ]
        else:
            streaming_cfg.shift_size = sampling_frames * (1 + lookahead_steps - streaming_cfg.cache_drop_size)

        if isinstance(streaming_cfg.shift_size, list):
            streaming_cfg.valid_out_len = (
                streaming_cfg.shift_size[1] - sampling_frames[1]
            ) // self.subsampling_factor + 1
        else:
            streaming_cfg.valid_out_len = streaming_cfg.shift_size // self.subsampling_factor

        if hasattr(self.pre_encode, "get_streaming_cache_size"):
            streaming_cfg.pre_encode_cache_size = self.pre_encode.get_streaming_cache_size()
        else:
            streaming_cfg.pre_encode_cache_size = 0

        if isinstance(streaming_cfg.pre_encode_cache_size, list):
            if streaming_cfg.pre_encode_cache_size[1] >= 1:
                streaming_cfg.drop_extra_pre_encoded = (
                    1 + (streaming_cfg.pre_encode_cache_size[1] - 1) // self.subsampling_factor
                )
            else:
                streaming_cfg.drop_extra_pre_encoded = 0
        else:
            streaming_cfg.drop_extra_pre_encoded = streaming_cfg.pre_encode_cache_size // self.subsampling_factor

        for m in self.layers.modules():
            if hasattr(m, "_max_cache_len"):
                if isinstance(m, MultiHeadAttention):
                    m.cache_drop_size = streaming_cfg.cache_drop_size
                if isinstance(m, CausalConv1D):
                    m.cache_drop_size = streaming_cfg.cache_drop_size

        self.streaming_cfg = streaming_cfg

    def get_initial_cache_state(self, batch_size=1, dtype=torch.float32, device=None, max_dim=0):
        if device is None:
            device = next(self.parameters()).device
        if max_dim > 0:
            create_tensor = torch.randn
        else:
            create_tensor = torch.zeros
        last_time_cache_size = self.conv_context_size[0]
        cache_last_channel = create_tensor(
            (len(self.layers), batch_size, self.streaming_cfg.last_channel_cache_size, self.d_model,),
            device=device,
            dtype=dtype,
        )
        cache_last_time = create_tensor(
            (len(self.layers), batch_size, self.d_model, last_time_cache_size), device=device, dtype=dtype,
        )
        if max_dim > 0:
            cache_last_channel_len = torch.randint(
                0,
                min(max_dim, self.streaming_cfg.last_channel_cache_size),
                (batch_size,),
                device=device,
                dtype=torch.int64,
            )
            for i in range(batch_size):
                cache_last_channel[:, i, cache_last_channel_len[i] :, :] = 0
                # what is the right rule to zero out cache_last_time?
                if cache_last_channel_len[i] == 0:
                    cache_last_time[:, i, :, :] = 0
        else:
            cache_last_channel_len = torch.zeros(batch_size, device=device, dtype=torch.int64)
        return cache_last_channel, cache_last_time, cache_last_channel_len

    def change_attention_model(
        self,
        self_attention_model: str = None,
        att_context_size: List[int] = None,
        update_config: bool = True,
        device: torch.device = None,
    ):

        """
        Update the self_attention_model which changes the positional encoding and attention layers.

        Args:
            self_attention_model (str): type of the attention layer and positional encoding
                'rel_pos': relative positional embedding and Transformer-XL
                'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
                    overlapping windows. Attention context is determined by att_context_size parameter.
                'abs_pos': absolute positional embedding and Transformer
                If None is provided, the self_attention_model isn't changed. Defaults to None.
            att_context_size (List[int]): List of 2 ints corresponding to left and right attention context sizes,
                or None to keep as it is. Defaults to None.
            update_config (bool): Whether to update the config or not with the new attention model.
                Defaults to True.
            device (torch.device): If provided, new layers will be moved to the device.
                Defaults to None.
        """

        if att_context_size:
            att_context_size = list(att_context_size)
        else:
            att_context_size = self.att_context_size

        if self_attention_model is None:
            self_attention_model = self.self_attention_model

        if self_attention_model == 'rel_pos_local_attn' and max(att_context_size) <= 0:
            raise ValueError("When using local attention, context size must be set > 0")

        if self_attention_model == "rel_pos":
            new_pos_enc = RelPositionalEncoding(
                d_model=self._cfg.d_model,
                dropout_rate=self._cfg.dropout,
                max_len=self._cfg.pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=self._cfg.dropout_emb,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            new_pos_enc = LocalAttRelPositionalEncoding(
                att_context_size=att_context_size,
                d_model=self._cfg.d_model,
                dropout_rate=self._cfg.dropout,
                max_len=self._cfg.pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=self._cfg.dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            new_pos_enc = PositionalEncoding(
                d_model=self._cfg.d_model,
                dropout_rate=self._cfg.dropout,
                max_len=self._cfg.pos_emb_max_len,
                xscale=self.xscale,
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        if device is not None:
            new_pos_enc = new_pos_enc.to(device=device)
        del self.pos_enc
        self.pos_enc = new_pos_enc
        self.self_attention_model = self_attention_model
        self.att_context_size = att_context_size
        self.set_max_audio_length(self.pos_emb_max_len)

        for name, m in self.named_modules():
            if type(m) == ConformerLayer:
                if self_attention_model == 'rel_pos':
                    new_attn = RelPositionMultiHeadAttention(
                        n_head=self._cfg.n_heads,
                        n_feat=self._cfg.d_model,
                        dropout_rate=self._cfg.dropout_att,
                        max_cache_len=att_context_size[0],
                        pos_bias_u=None,
                        pos_bias_v=None,
                    )
                elif self_attention_model == 'rel_pos_local_attn':
                    new_attn = RelPositionMultiHeadAttentionLongformer(
                        n_head=self._cfg.n_heads,
                        n_feat=self._cfg.d_model,
                        dropout_rate=self._cfg.dropout_att,
                        max_cache_len=att_context_size[0],
                        att_context_size=att_context_size,
                        pos_bias_u=None,
                        pos_bias_v=None,
                    )
                elif self_attention_model == 'abs_pos':
                    new_attn = MultiHeadAttention(
                        n_head=self._cfg.n_heads,
                        n_feat=self._cfg.d_model,
                        dropout_rate=self._cfg.dropout_att,
                        max_cache_len=att_context_size[0],
                    )
                else:
                    raise ValueError(
                        f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                        f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos']"
                    )
                if device is not None:
                    new_attn = new_attn.to(device=device)
                new_attn.load_state_dict(m.self_attn.state_dict(), strict=False)
                del m.self_attn
                m.self_attn = new_attn
                m.self_attention_model = self_attention_model

        if update_config:
            with open_dict(self._cfg):
                self._cfg.self_attention_model = self_attention_model
                self._cfg.att_context_size = att_context_size

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        """
        Update the conv_chunking_factor (int) 
        Default is 1 (auto)
        Set it to -1 (disabled) or to a specific value (power of 2) if you OOM in the conv subsampling layers


        Args:
            subsampling_conv_chunking_factor (int)
        """

        if not hasattr(self.pre_encode, "change_subsampling_conv_chunking_factor"):
            logging.info("Model pre_encoder doesn't have a change_subsampling_conv_chunking_factor method ")
            return

        self.pre_encode.change_subsampling_conv_chunking_factor(
            subsampling_conv_chunking_factor=subsampling_conv_chunking_factor
        )


class ConformerEncoderAdapter(ConformerEncoder, adapter_mixins.AdapterModuleMixin):

    # Higher level forwarding
    def add_adapter(self, name: str, cfg: dict):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        for conformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            conformer_layer.add_adapter(name, cfg)

    def is_adapter_available(self) -> bool:
        return any([conformer_layer.is_adapter_available() for conformer_layer in self.layers])

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for conformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            conformer_layer.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for conformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            names.update(conformer_layer.get_enabled_adapters())

        names = sorted(list(names))
        return names

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.d_model)
        return cfg

    def get_accepted_adapter_types(self,) -> Set[type]:
        types = super().get_accepted_adapter_types()

        if len(types) == 0:
            self.set_accepted_adapter_types(
                [
                    adapter_utils.LINEAR_ADAPTER_CLASSPATH,
                    adapter_utils.MHA_ADAPTER_CLASSPATH,
                    adapter_utils.RELMHA_ADAPTER_CLASSPATH,
                ]
            )
            types = self.get_accepted_adapter_types()
        return types


"""
Register any additional information
"""
if adapter_mixins.get_registered_adapter(ConformerEncoder) is None:
    adapter_mixins.register_adapter(base_class=ConformerEncoder, adapter_class=ConformerEncoderAdapter)


@dataclass
class ConformerChangeConfig:
    # Change self_attention_model for Conformer
    # Options:
    #  'rel_pos': relative positional embedding and Transformer-XL
    #  'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
    #   overlapping chunks. Attention context is determined by att_context_size parameter.
    #  'abs_pos': absolute positional embedding and Transformer
    # If None is provided, self_attention_model is not changed.
    self_attention_model: Optional[str] = None

    # Change the attention context size by providing 2 integers,
    # corresponding to left and right context, or -1 for full context.
    # If None is provided, the attention context size isn't changed.
    att_context_size: Optional[List[int]] = None
