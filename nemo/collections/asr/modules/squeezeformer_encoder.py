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

import math
from collections import OrderedDict
from typing import List, Optional, Set

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig

from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, RelPositionalEncoding
from nemo.collections.asr.parts.submodules.squeezeformer_modules import SqueezeformerLayer
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, StackingSubsampling, TimeReductionModule
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType

__all__ = ['SqueezeformerEncoder']


class SqueezeformerEncoder(NeuralModule, Exportable, AccessMixin):
    """
    The encoder for ASR model of Squeezeformer.
    Based on this paper:
    'Squeezeformer: An Efficient Transformer for Automatic Speech Recognition' by Sehoon Kim et al.
    https://arxiv.org/abs/2206.00888

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling, choices=['vggnet', 'striding', 'dw_striding']
            Defaults to dw_striding.
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to -1 which would set it to d_model.
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
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
        adaptive_scale (bool): Whether to scale the inputs to each component by affine `scale` and `bias` layer.
            Or use a fixed scale=1 and bias=0.
        time_reduce_idx (int): Optional integer index of a layer where a time reduction operation will occur.
            All operations beyond this point will only occur at the reduced resolution.
        time_recovery_idx (int): Optional integer index of a layer where the time recovery operation will occur.
            All operations beyond this point will occur at the original resolution (resolution after
            primary downsampling). If no value is provided, assumed to be the last layer.
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
        return tuple([input_example, input_example_length])

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    def __init__(
        self,
        feat_in: int,
        n_layers: int,
        d_model: int,
        feat_out: int = -1,
        subsampling: str = 'dw_striding',
        subsampling_factor: int = 4,
        subsampling_conv_channels: int = -1,
        ff_expansion_factor: int = 4,
        self_attention_model: str = 'rel_pos',
        n_heads: int = 4,
        att_context_size: Optional[List[int]] = None,
        xscaling: bool = True,
        untie_biases: bool = True,
        pos_emb_max_len: int = 5000,
        conv_kernel_size: int = 31,
        conv_norm_type: str = 'batch_norm',
        dropout: float = 0.1,
        dropout_emb: float = 0.1,
        dropout_att: float = 0.0,
        adaptive_scale: bool = True,
        time_reduce_idx: Optional[int] = None,
        time_recovery_idx: Optional[int] = None,
    ):
        super().__init__()

        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self._feat_in = feat_in
        if att_context_size:
            self.att_context_size = att_context_size
        else:
            self.att_context_size = [-1, -1]

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None
        self.adaptive_scale = adaptive_scale

        self.time_reduce_idx = time_reduce_idx
        if time_reduce_idx is not None:
            if time_recovery_idx is None:
                self.time_recovery_idx = n_layers - 1  # recover at last layer
            else:
                self.time_recovery_idx = time_recovery_idx  # recover at given layer

        if self.time_reduce_idx is not None:
            if self.time_reduce_idx < 0 or self.time_recovery_idx >= n_layers:
                raise ValueError(f"Time reduce index must lie between [0, {n_layers})")
            if self.time_recovery_idx < 0 or self.time_recovery_idx >= n_layers:
                raise ValueError(f"Time recovery index must lie between [0, {n_layers})")

        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling == 'stacking':
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor, feat_in=feat_in, feat_out=d_model
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    activation=nn.ReLU(),
                )
                # For Squeezeformer, initialize the parameters as required.
                self.pre_encode.reset_parameters()
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

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
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = SqueezeformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                adaptive_scale=adaptive_scale,
            )
            self.layers.append(layer)

        # Time Reduction and Recovery layer setup
        self.time_reduce_layer = None
        self.time_recovery_layer = None
        self.time_reduce_pos_enc = None
        # Add time reduction layer
        if self.time_reduce_idx is not None:
            self.time_reduce_layer = TimeReductionModule(d_model, d_model, kernel_size=5, stride=2)
            self.time_recovery_layer = nn.Linear(d_model, d_model)

            # Chose same type of positional encoding as the originally determined above
            if self_attention_model == "rel_pos":
                self.time_reduce_pos_enc = RelPositionalEncoding(
                    d_model=d_model, dropout_rate=0.0, max_len=pos_emb_max_len, xscale=None, dropout_rate_emb=0.0,
                )
            else:
                self.time_reduce_pos_enc = PositionalEncoding(
                    d_model=d_model, dropout_rate=0.0, max_len=pos_emb_max_len, xscale=None, dropout_rate_emb=0.0
                )

        self.pre_ln = nn.LayerNorm(d_model)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_capture_at_layers = None

    def set_max_audio_length(self, max_audio_length):
        """ Sets maximum input length.
            Pre-calculates internal seq_range mask.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        seq_range = torch.arange(0, self.max_audio_length, device=device)
        if hasattr(self, 'seq_range'):
            self.seq_range = seq_range
        else:
            self.register_buffer('seq_range', seq_range, persistent=False)
        self.pos_enc.extend_pe(max_audio_length, device)

        if self.time_reduce_pos_enc is not None:
            self.time_reduce_pos_enc.extend_pe(max_audio_length, device)

    @typecheck()
    def forward(self, audio_signal, length=None):
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_for_export(audio_signal=audio_signal, length=length)

    @typecheck()
    def forward_for_export(self, audio_signal, length):
        max_audio_length: int = audio_signal.size(-1)

        if max_audio_length > self.max_audio_length:
            self.set_max_audio_length(max_audio_length)

        if length is None:
            length = audio_signal.new_full(
                audio_signal.size(0), max_audio_length, dtype=torch.int32, device=self.seq_range.device
            )

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(audio_signal, length)

        audio_signal, pos_emb = self.pos_enc(audio_signal)
        # adjust size
        max_audio_length = audio_signal.size(1)
        # Create the self-attention and padding masks

        pad_mask = self.make_pad_mask(max_audio_length, length)
        att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
        if self.att_context_size[0] >= 0:
            att_mask = att_mask.triu(diagonal=-self.att_context_size[0])
        if self.att_context_size[1] >= 0:
            att_mask = att_mask.tril(diagonal=self.att_context_size[1])
        att_mask = ~att_mask

        if self.use_pad_mask:
            pad_mask = ~pad_mask
        else:
            pad_mask = None

        # Create cache of activations for the time reduction step
        # Note: NeMo codebase allows only a single time reduction step to occur
        recovery_activation_cache = []

        audio_signal = self.pre_ln(audio_signal)
        for lth, layer in enumerate(self.layers):
            # Perform time reduction
            if self.time_reduce_layer is not None and lth == self.time_reduce_idx:
                # Perform time reduction
                recovery_activation_cache.append((audio_signal, att_mask, pad_mask, pos_emb))
                audio_signal, att_mask, pad_mask = self.time_reduce_layer(
                    x=audio_signal, att_mask=att_mask, pad_mask=pad_mask
                )
                # Only update PE, not the original audio_signal
                _, pos_emb = self.time_reduce_pos_enc(audio_signal)

            # Perform time recovery
            if self.time_recovery_layer is not None and lth == self.time_recovery_idx:
                recovery_audio_signal, att_mask, pad_mask, pos_emb = recovery_activation_cache.pop(0)
                # repeat interleaved values for 2x seq length
                audio_signal = torch.repeat_interleave(audio_signal, repeats=2, dim=1)

                B, T, D = recovery_audio_signal.size()
                audio_signal = audio_signal[:, :T, :]  # Slice off the exact T timesteps as original cache value
                audio_signal = self.time_recovery_layer(audio_signal)  # learn non linear mapping
                audio_signal = recovery_audio_signal + audio_signal  # learn just the residual

            audio_signal = layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)

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

        audio_signal = torch.transpose(audio_signal, 1, 2)
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

    def make_pad_mask(self, max_audio_length, seq_lens):
        """Make masking for padding."""
        mask = self.seq_range[:max_audio_length].expand(seq_lens.size(0), -1) < seq_lens.unsqueeze(-1)
        return mask

    def enable_pad_mask(self, on=True):
        # On inference, user may chose to disable pad mask
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask


class SqueezeformerEncoderAdapter(SqueezeformerEncoder, adapter_mixins.AdapterModuleMixin):

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
if adapter_mixins.get_registered_adapter(SqueezeformerEncoder) is None:
    adapter_mixins.register_adapter(base_class=SqueezeformerEncoder, adapter_class=SqueezeformerEncoderAdapter)
