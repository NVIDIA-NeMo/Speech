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
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    PositionalEncoding,
    RelPositionalEncoding,
)
from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling,
    StackingSubsampling,
    SubsamplingReductionModule,
)
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, ChannelType, LengthsType, NeuralType, SpectrogramType

__all__ = ['ConformerEncoder']


class ConformerEncoder(NeuralModule, StreamingEncoder, Exportable):
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
            'abs_pos': absolute positional embedding and Transformer
            default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaulst to 5000
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by sqrt(d_model)
            Defaults to True.
        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        conv_norm_type (str): the type of the normalization in the convolutional modules
            Defaults to 'batch_norm'.
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
        dropout_pre_encoder (float): the dropout rate used before the encoder
            Defaults to 0.1.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
    """

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(dev)
        input_example_length = torch.randint(1, max_dim, (max_batch,)).to(dev)

        if hasattr(self, 'export_cache_support') and self.export_cache_support:
            cache_last_channel = torch.randn(self.n_layers, max_batch, max_dim, self.d_model).to(dev)
            cache_last_time = torch.randn(self.n_layers, max_batch, self.d_model, self.conv_context_size[0]).to(dev)
            all_input_example = tuple([input_example, input_example_length, cache_last_channel, cache_last_time])
        else:
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
            }
        )

    @property
    def disabled_deployment_input_names(self):
        if not self.export_cache_support:
            return set(["cache_last_channel", "cache_last_time"])
        else:
            return set()

    @property
    def disabled_deployment_output_names(self):
        if not self.export_cache_support:
            return set(["cache_last_channel_next", "cache_last_time_next"])
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
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=None,
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
    ):
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.scale = math.sqrt(self.d_model)
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor

        if att_context_size:
            self.att_context_size = list(att_context_size)
        else:
            self.att_context_size = [-1, -1]

        if isinstance(conv_context_size, ListConfig):
            conv_context_size = list(conv_context_size)

        if conv_context_size is not None:
            if (
                not isinstance(conv_context_size, list)
                and not isinstance(conv_context_size, str)
                and not isinstance(conv_context_size, ListConfig)
            ):
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
        self.conv_context_size = conv_context_size

        if att_context_style == "chunked_limited":
            # the left context for self-attention in chunked_limited mode should be dividable by the right context
            # right context=att_context_size[1]+1, and left_context=self.att_context_size[0]
            if self.att_context_size[0] > 0 and self.att_context_size[0] % (self.att_context_size[1] + 1) > 0:
                raise ValueError("att_context_size[0] % (att_context_size[1] + 1) should be zero!")
            if self.att_context_size[1] < 0:
                raise ValueError("Right context can not be unlimited for chunked_limited style!")
            self.chunk_size = self.att_context_size[1] + 1

            # left_chunks_num specifies the number of chunks to be visible by each chunk on the left side
            if self.att_context_size[0] >= 0:
                self.left_chunks_num = self.att_context_size[0] // self.chunk_size
            else:
                self.left_chunks_num = 100000

        elif att_context_style == "regular":
            self.chunk_size = None
        else:
            raise ValueError("Invalid att_context_style!")

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
                    activation=nn.ReLU(),
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

        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
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

        att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)
        if self.chunk_size is None:
            if self.att_context_size[0] >= 0:
                att_mask = att_mask.triu(diagonal=-self.att_context_size[0])
            if self.att_context_size[1] >= 0:
                att_mask = att_mask.tril(diagonal=self.att_context_size[1])
        else:
            chunk_idx = torch.arange(0, max_audio_length, dtype=torch.int, device=att_mask.device)
            chunk_idx = torch.div(chunk_idx, self.chunk_size, rounding_mode="trunc")
            diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
            chunked_limited_mask = torch.logical_and(
                torch.le(diff_chunks, self.left_chunks_num), torch.ge(diff_chunks, 0)
            )
            att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))

        if hasattr(self, 'att_mask'):
            self.att_mask = att_mask
        else:
            self.register_buffer('att_mask', att_mask, persistent=False)

    @typecheck()
    def forward(self, audio_signal, length, cache_last_channel=None, cache_last_time=None):
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        max_audio_length: int = audio_signal.size(-1)

        if length is None:
            length = audio_signal.new_full(
                audio_signal.size(0), max_audio_length, dtype=torch.int32, device=audio_signal.device
            )

        if cache_last_time is not None:
            cache_last_time_next = torch.zeros_like(cache_last_time)
        else:
            cache_last_time_next = None
        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
            # self.streaming_cfg is set by setup_streaming_cfg(), called in the init
            if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                # TODO: find a better solution
                length = (length - self.streaming_cfg.drop_extra_pre_encoded).float()
                length = torch.clip(length, min=0).int()

        max_audio_length = audio_signal.size(1)

        if self.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

        if cache_last_channel is not None:
            last_channel_num, bs, cache_len, channel_size = cache_last_channel.size()
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            cache_last_channel_next = torch.zeros(
                (last_channel_num, bs, cache_len + cache_keep_size, channel_size),
                device=cache_last_channel.device,
                dtype=cache_last_channel.dtype,
            )
            max_audio_length += cache_len
            padding_length = length + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(max_audio_length, padding_length, audio_signal.device)

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            att_mask = att_mask[:, cache_len:]

        for lth, layer in enumerate(self.layers):
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_next=cache_last_channel_next,
                cache_last_time_next=cache_last_time_next,
            )

            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                # Don't update the audio_signal here because then it will again scale the audio_signal
                # and cause an increase in the WER
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(max_audio_length, length, audio_signal.device)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if cache_last_channel is not None:
            return audio_signal, length, cache_last_channel_next, cache_last_time_next
        else:
            return audio_signal, length

    def _create_masks(self, max_audio_length, padding_length, device):
        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        # pad_mask_for_att_mask is the mask which helps to ignore paddings
        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
        # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
        att_mask = self.att_mask[:, :max_audio_length, :max_audio_length]
        # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
        att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))

        pad_mask = ~pad_mask
        att_mask = ~att_mask

        return pad_mask, att_mask

    def enable_pad_mask(self, on=True):
        # On inference, user may choose to disable pad mask
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask

    def setup_streaming_params(
        self, chunk_size: int = None, shift_size: int = None, left_chunks: int = None, max_context: int = 10000
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
        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError("chunk_size needs to be a number larger or equal to one.")
            lookahead_steps = chunk_size - 1
            streaming_cfg.cache_drop_size = chunk_size - shift_size
        elif self.att_context_style == "chunked_limited":
            lookahead_steps = self.att_context_size[1]
            streaming_cfg.cache_drop_size = 0
        elif self.att_context_style == "regular":
            lookahead_steps = self.att_context_size[1] * self.n_layers + self.conv_context_size[1] * self.n_layers
            streaming_cfg.cache_drop_size = lookahead_steps
        else:
            streaming_cfg.cache_drop_size = 0
            lookahead_steps = None

        if chunk_size is None:
            streaming_cfg.last_channel_cache_size = (
                self.att_context_size[0] if self.att_context_size[0] >= 0 else max_context
            )
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

        # counting the number of the layers need caching
        streaming_cfg.last_channel_num = 0
        streaming_cfg.last_time_num = 0
        for m in self.layers.modules():
            if hasattr(m, "_max_cache_len"):
                if isinstance(m, MultiHeadAttention):
                    m._cache_id = streaming_cfg.last_channel_num
                    m.cache_drop_size = streaming_cfg.cache_drop_size
                    streaming_cfg.last_channel_num += 1

                if isinstance(m, CausalConv1D):
                    m._cache_id = streaming_cfg.last_time_num
                    m.cache_drop_size = streaming_cfg.cache_drop_size
                    streaming_cfg.last_time_num += 1

        self.streaming_cfg = streaming_cfg

    def get_initial_cache_state(self, batch_size=1, dtype=torch.float32, device=None):
        if device is None:
            device = next(self.parameters()).device
        last_time_cache_size = self.conv_context_size[0]
        cache_last_channel = torch.zeros(
            (self.streaming_cfg.last_channel_num, batch_size, 0, self.d_model), device=device, dtype=dtype
        )
        cache_last_time = torch.zeros(
            (self.streaming_cfg.last_time_num, batch_size, self.d_model, last_time_cache_size),
            device=device,
            dtype=dtype,
        )

        return cache_last_channel, cache_last_time


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


"""
Register any additional information
"""
if adapter_mixins.get_registered_adapter(ConformerEncoder) is None:
    adapter_mixins.register_adapter(base_class=ConformerEncoder, adapter_class=ConformerEncoderAdapter)
