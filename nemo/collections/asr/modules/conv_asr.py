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
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from nemo.collections.asr.parts.submodules.jasper import (
    JasperBlock,
    MaskedConv1d,
    ParallelBlock,
    SqueezeExcite,
    init_weights,
    jasper_activations,
)
from nemo.collections.asr.parts.submodules.tdnn_attention import (
    AttentivePoolLayer,
    StatsPoolLayer,
    TDNNModule,
    TDNNSEModule,
)
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    LogitsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['ConvASRDecoder', 'ConvASREncoder', 'ConvASRDecoderClassification']


class ConvASREncoder(NeuralModule, Exportable, AccessMixin):
    """
    Convolutional encoder for ASR models. With this class you can implement JasperNet and QuartzNet models.

    Based on these papers:
        https://arxiv.org/pdf/1904.03288.pdf
        https://arxiv.org/pdf/1910.10261.pdf
    """

    def _prepare_for_export(self, **kwargs):
        m_count = 0
        for name, m in self.named_modules():
            if isinstance(m, MaskedConv1d):
                m.use_mask = False
                m_count += 1

        Exportable._prepare_for_export(self, **kwargs)
        logging.warning(f"Turned off {m_count} masked convolutions")

    def input_example(self, max_batch=1, max_dim=8192):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        device = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim, device=device)
        lens = torch.full(size=(input_example.shape[0],), fill_value=max_dim, device=device)
        return tuple([input_example, lens])

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
        jasper,
        activation: str,
        feat_in: int,
        normalization_mode: str = "batch",
        residual_mode: str = "add",
        norm_groups: int = -1,
        conv_mask: bool = True,
        frame_splicing: int = 1,
        init_mode: Optional[str] = 'xavier_uniform',
        quantize: bool = False,
    ):
        super().__init__()
        if isinstance(jasper, ListConfig):
            jasper = OmegaConf.to_container(jasper)

        activation = jasper_activations[activation]()

        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        feat_in = feat_in * frame_splicing

        self._feat_in = feat_in

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for layer_idx, lcfg in enumerate(jasper):
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            heads = lcfg.get('heads', -1)
            residual_mode = lcfg.get('residual_mode', residual_mode)
            se = lcfg.get('se', False)
            se_reduction_ratio = lcfg.get('se_reduction_ratio', 8)
            se_context_window = lcfg.get('se_context_size', -1)
            se_interpolation_mode = lcfg.get('se_interpolation_mode', 'nearest')
            kernel_size_factor = lcfg.get('kernel_size_factor', 1.0)
            stride_last = lcfg.get('stride_last', False)
            future_context = lcfg.get('future_context', -1)
            encoder_layers.append(
                JasperBlock(
                    feat_in,
                    lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'],
                    residual=lcfg['residual'],
                    groups=groups,
                    separable=separable,
                    heads=heads,
                    residual_mode=residual_mode,
                    normalization=normalization_mode,
                    norm_groups=norm_groups,
                    activation=activation,
                    residual_panes=dense_res,
                    conv_mask=conv_mask,
                    se=se,
                    se_reduction_ratio=se_reduction_ratio,
                    se_context_window=se_context_window,
                    se_interpolation_mode=se_interpolation_mode,
                    kernel_size_factor=kernel_size_factor,
                    stride_last=stride_last,
                    future_context=future_context,
                    quantize=quantize,
                    layer_idx=layer_idx,
                )
            )
            feat_in = lcfg['filters']

        self._feat_out = feat_in

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

        self.max_audio_length = 0

    @typecheck()
    def forward(self, audio_signal, length):
        self.update_max_sequence_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        s_input, length = self.encoder(([audio_signal], length))
        if length is None:
            return s_input[-1]

        return s_input[-1], length

    def update_max_sequence_length(self, seq_length: int, device):
        # Find global max audio length across all nodes
        if torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            if seq_length < 5000:
                seq_length = seq_length * 2
            elif seq_length < 10000:
                seq_length = seq_length * 1.5
            self.max_audio_length = seq_length

            device = next(self.parameters()).device
            seq_range = torch.arange(0, self.max_audio_length, device=device)
            if hasattr(self, 'seq_range'):
                self.seq_range = seq_range
            else:
                self.register_buffer('seq_range', seq_range, persistent=False)

            # Update all submodules
            for name, m in self.named_modules():
                if isinstance(m, MaskedConv1d):
                    m.update_masked_length(self.max_audio_length, seq_range=self.seq_range)
                elif isinstance(m, SqueezeExcite):
                    m.set_max_len(self.max_audio_length, seq_range=self.seq_range)


class ParallelConvASREncoder(NeuralModule, Exportable):
    """
    Convolutional encoder for ASR models with parallel blocks. CarneliNet can be implemented with this class.
    """

    def _prepare_for_export(self):
        m_count = 0
        for m in self.modules():
            if isinstance(m, MaskedConv1d):
                m.use_mask = False
                m_count += 1
        logging.warning(f"Turned off {m_count} masked convolutions")

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        return set(["length"])

    @property
    def disabled_deployment_output_names(self):
        """Implement this method to return a set of output names disabled for export"""
        return set(["encoded_lengths"])

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

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
        jasper,
        activation: str,
        feat_in: int,
        normalization_mode: str = "batch",
        residual_mode: str = "add",
        norm_groups: int = -1,
        conv_mask: bool = True,
        frame_splicing: int = 1,
        init_mode: Optional[str] = 'xavier_uniform',
        aggregation_mode: Optional[str] = None,
        quantize: bool = False,
    ):
        super().__init__()
        if isinstance(jasper, ListConfig):
            jasper = OmegaConf.to_container(jasper)

        activation = jasper_activations[activation]()
        feat_in = feat_in * frame_splicing

        self._feat_in = feat_in

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in jasper:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            heads = lcfg.get('heads', -1)
            residual_mode = lcfg.get('residual_mode', residual_mode)
            se = lcfg.get('se', False)
            se_reduction_ratio = lcfg.get('se_reduction_ratio', 8)
            se_context_window = lcfg.get('se_context_size', -1)
            se_interpolation_mode = lcfg.get('se_interpolation_mode', 'nearest')
            kernel_size_factor = lcfg.get('kernel_size_factor', 1.0)
            stride_last = lcfg.get('stride_last', False)
            aggregation_mode = lcfg.get('aggregation_mode', 'sum')
            block_dropout = lcfg.get('block_dropout', 0.0)
            parallel_residual_mode = lcfg.get('parallel_residual_mode', 'sum')

            parallel_blocks = []
            for kernel_size in lcfg['kernel']:
                parallel_blocks.append(
                    JasperBlock(
                        feat_in,
                        lcfg['filters'],
                        repeat=lcfg['repeat'],
                        kernel_size=[kernel_size],
                        stride=lcfg['stride'],
                        dilation=lcfg['dilation'],
                        dropout=lcfg['dropout'],
                        residual=lcfg['residual'],
                        groups=groups,
                        separable=separable,
                        heads=heads,
                        residual_mode=residual_mode,
                        normalization=normalization_mode,
                        norm_groups=norm_groups,
                        activation=activation,
                        residual_panes=dense_res,
                        conv_mask=conv_mask,
                        se=se,
                        se_reduction_ratio=se_reduction_ratio,
                        se_context_window=se_context_window,
                        se_interpolation_mode=se_interpolation_mode,
                        kernel_size_factor=kernel_size_factor,
                        stride_last=stride_last,
                        quantize=quantize,
                    )
                )
            if len(parallel_blocks) == 1:
                encoder_layers.append(parallel_blocks[0])
            else:
                encoder_layers.append(
                    ParallelBlock(
                        parallel_blocks,
                        aggregation_mode=aggregation_mode,
                        block_dropout_prob=block_dropout,
                        residual_mode=parallel_residual_mode,
                        in_filters=feat_in,
                        out_filters=lcfg['filters'],
                    )
                )
            feat_in = lcfg['filters']

        self._feat_out = feat_in

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    @typecheck()
    def forward(self, audio_signal, length=None):
        s_input, length = self.encoder(([audio_signal], length))
        if length is None:
            return s_input[-1]

        return s_input[-1], length


class ConvASRDecoder(NeuralModule, Exportable, adapter_mixins.AdapterModuleMixin):
    """Simple ASR Decoder for use with CTC-based models such as JasperNet and QuartzNet

     Based on these papers:
        https://arxiv.org/pdf/1904.03288.pdf
        https://arxiv.org/pdf/1910.10261.pdf
        https://arxiv.org/pdf/2005.04290.pdf
    """

    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        return OrderedDict({"logprobs": NeuralType(('B', 'T', 'D'), LogprobsType())})

    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform", vocabulary=None):
        super().__init__()

        if vocabulary is None and num_classes < 0:
            raise ValueError(
                f"Neither of the vocabulary and num_classes are set! At least one of them need to be set."
            )

        if num_classes <= 0:
            num_classes = len(vocabulary)
            logging.info(f"num_classes of ConvASRDecoder is set to the size of the vocabulary: {num_classes}.")

        if vocabulary is not None:
            if num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, it's length should be equal to the num_classes. Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True)
        )
        self.apply(lambda x: init_weights(x, mode=init_mode))

        accepted_adapters = [adapter_utils.LINEAR_ADAPTER_CLASSPATH]
        self.set_accepted_adapter_types(accepted_adapters)

        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0

    @typecheck()
    def forward(self, encoder_output):
        # Adapter module forward step
        if self.is_adapter_available():
            encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]
            encoder_output = self.forward_enabled_adapters(encoder_output)
            encoder_output = encoder_output.transpose(1, 2)  # [B, C, T]

        if self.temperature != 1.0:
            return torch.nn.functional.log_softmax(
                self.decoder_layers(encoder_output).transpose(1, 2) / self.temperature, dim=-1
            )
        return torch.nn.functional.log_softmax(self.decoder_layers(encoder_output).transpose(1, 2), dim=-1)

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    def _prepare_for_export(self, **kwargs):
        m_count = 0
        for m in self.modules():
            if type(m).__name__ == "MaskedConv1d":
                m.use_mask = False
                m_count += 1
        if m_count > 0:
            logging.warning(f"Turned off {m_count} masked convolutions")
        Exportable._prepare_for_export(self, **kwargs)

    # Adapter method overrides
    def add_adapter(self, name: str, cfg: DictConfig):
        # Update the config with correct input dim
        cfg = self._update_adapter_cfg_input_dim(cfg)
        # Add the adapter
        super().add_adapter(name=name, cfg=cfg)

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self._feat_in)
        return cfg

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def num_classes_with_blank(self):
        return self._num_classes


class ConvASRDecoderReconstruction(NeuralModule, Exportable):
    """ASR Decoder for reconstructing masked regions of spectrogram
    """

    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        if self.apply_softmax:
            return OrderedDict({"out": NeuralType(('B', 'T', 'D'), LogprobsType())})
        else:
            return OrderedDict({"out": NeuralType(('B', 'T', 'D'), AcousticEncodedRepresentation())})

    def __init__(
        self,
        feat_in,
        feat_out,
        feat_hidden,
        stride_layers=0,
        non_stride_layers=0,
        kernel_size=11,
        init_mode="xavier_uniform",
        activation="relu",
        stride_transpose=True,
        apply_softmax=False,
    ):
        super().__init__()

        if ((stride_layers + non_stride_layers) > 0) and (kernel_size < 3 or kernel_size % 2 == 0):
            raise ValueError("Kernel size in this decoder needs to be >= 3 and odd when using at least 1 conv layer.")

        activation = jasper_activations[activation]()

        self.feat_in = feat_in
        self.feat_out = feat_out
        self.feat_hidden = feat_hidden

        self.decoder_layers = [nn.Conv1d(self.feat_in, self.feat_hidden, kernel_size=1, bias=True)]
        for i in range(stride_layers):
            self.decoder_layers.append(activation)
            if stride_transpose:
                self.decoder_layers.append(
                    nn.ConvTranspose1d(
                        self.feat_hidden,
                        self.feat_hidden,
                        kernel_size,
                        stride=2,
                        padding=(kernel_size - 3) // 2 + 1,
                        output_padding=1,
                        bias=True,
                        groups=self.feat_hidden,
                    )
                )
            else:
                self.decoder_layers.append(
                    nn.Conv1d(
                        self.feat_hidden,
                        self.feat_hidden,
                        kernel_size,
                        stride=2,
                        padding=(kernel_size - 1) // 2,
                        bias=True,
                        groups=self.feat_hidden,
                    )
                )
            self.decoder_layers.append(nn.Conv1d(self.feat_hidden, self.feat_hidden, kernel_size=1, bias=True))
            self.decoder_layers.append(nn.BatchNorm1d(self.feat_hidden, eps=1e-3, momentum=0.1))
        for i in range(non_stride_layers):
            self.decoder_layers.append(activation)
            self.decoder_layers.append(
                nn.Conv1d(
                    self.feat_hidden,
                    self.feat_hidden,
                    kernel_size,
                    bias=True,
                    groups=self.feat_hidden,
                    padding=kernel_size // 2,
                )
            )
            self.decoder_layers.append(nn.Conv1d(self.feat_hidden, self.feat_hidden, kernel_size=1, bias=True))
            self.decoder_layers.append(nn.BatchNorm1d(self.feat_hidden, eps=1e-3, momentum=0.1))

        self.decoder_layers.append(activation)
        self.decoder_layers.append(nn.Conv1d(self.feat_hidden, self.feat_out, kernel_size=1, bias=True))

        self.decoder_layers = nn.Sequential(*self.decoder_layers)
        self.apply_softmax = apply_softmax

        self.apply(lambda x: init_weights(x, mode=init_mode))

    @typecheck()
    def forward(self, encoder_output):
        out = self.decoder_layers(encoder_output).transpose(-2, -1)
        if self.apply_softmax:
            out = torch.nn.functional.log_softmax(out, dim=-1)
        return out

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    def _prepare_for_export(self, **kwargs):
        m_count = 0
        for m in self.modules():
            if type(m).__name__ == "MaskedConv1d":
                m.use_mask = False
                m_count += 1
        if m_count > 0:
            logging.warning(f"Turned off {m_count} masked convolutions")
        Exportable._prepare_for_export(self, **kwargs)


class ConvASRDecoderClassification(NeuralModule, Exportable):
    """Simple ASR Decoder for use with classification models such as JasperNet and QuartzNet

     Based on these papers:
        https://arxiv.org/pdf/2005.04290.pdf
    """

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        return OrderedDict({"logits": NeuralType(('B', 'D'), LogitsType())})

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        init_mode: Optional[str] = "xavier_uniform",
        return_logits: bool = True,
        pooling_type='avg',
    ):
        super().__init__()

        self._feat_in = feat_in
        self._return_logits = return_logits
        self._num_classes = num_classes

        if pooling_type == 'avg':
            self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        elif pooling_type == 'max':
            self.pooling = torch.nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError('Pooling type chosen is not valid. Must be either `avg` or `max`')

        self.decoder_layers = torch.nn.Sequential(torch.nn.Linear(self._feat_in, self._num_classes, bias=True))
        self.apply(lambda x: init_weights(x, mode=init_mode))

    @typecheck()
    def forward(self, encoder_output):
        batch, in_channels, timesteps = encoder_output.size()

        encoder_output = self.pooling(encoder_output).view(batch, in_channels)  # [B, C]
        logits = self.decoder_layers(encoder_output)  # [B, num_classes]

        if self._return_logits:
            return logits

        return torch.nn.functional.softmax(logits, dim=-1)

    @property
    def num_classes(self):
        return self._num_classes


class ECAPAEncoder(NeuralModule, Exportable):
    """
    Modified ECAPA Encoder layer without Res2Net module for faster training and inference which achieves
    better numbers on speaker diarization tasks
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)

    input:
        feat_in: input feature shape (mel spec feature shape)
        filters: list of filter shapes for SE_TDNN modules
        kernel_sizes: list of kernel shapes for SE_TDNN modules
        dilations: list of dilations for group conv se layer
        scale: scale value to group wider conv channels (deafult:8)

    output:
        outputs : encoded output
        output_length: masked output lengths
    """

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
        filters: list,
        kernel_sizes: list,
        dilations: list,
        scale: int = 8,
        init_mode: str = 'xavier_uniform',
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TDNNModule(feat_in, filters[0], kernel_size=kernel_sizes[0], dilation=dilations[0]))

        for i in range(len(filters) - 2):
            self.layers.append(
                TDNNSEModule(
                    filters[i],
                    filters[i + 1],
                    group_scale=scale,
                    se_channels=128,
                    kernel_size=kernel_sizes[i + 1],
                    dilation=dilations[i + 1],
                )
            )
        self.feature_agg = TDNNModule(filters[-1], filters[-1], kernel_sizes[-1], dilations[-1])
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal, length=None):
        x = audio_signal
        outputs = []

        for layer in self.layers:
            x = layer(x, length=length)
            outputs.append(x)

        x = torch.cat(outputs[1:], dim=1)
        x = self.feature_agg(x)
        return x, length


class SpeakerDecoder(NeuralModule, Exportable):
    """
    Speaker Decoder creates the final neural layers that maps from the outputs
    of Jasper Encoder to the embedding layer followed by speaker based softmax loss.
    Args:
        feat_in (int): Number of channels being input to this module
        num_classes (int): Number of unique speakers in dataset
        emb_sizes (list) : shapes of intermediate embedding layers (we consider speaker embbeddings from 1st of this layers)
                Defaults to [1024,1024]
        pool_mode (str) : Pooling strategy type. options are 'xvector','tap', 'attention'
                Defaults to 'xvector (mean and variance)'
                tap (temporal average pooling: just mean)
                attention (attention based pooling)

        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self.input_feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    @property
    def input_types(self):
        return OrderedDict(
            {
                "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "length": NeuralType(('B',), LengthsType(), optional=True),
            }
        )

    @property
    def output_types(self):
        return OrderedDict(
            {
                "logits": NeuralType(('B', 'D'), LogitsType()),
                "embs": NeuralType(('B', 'D'), AcousticEncodedRepresentation()),
            }
        )

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        emb_sizes: Optional[Union[int, list]] = 256,
        pool_mode: str = 'xvector',
        angular: bool = False,
        attention_channels: int = 128,
        init_mode: str = "xavier_uniform",
    ):
        super().__init__()
        self.angular = angular
        self.emb_id = 2
        bias = False if self.angular else True
        emb_sizes = [emb_sizes] if type(emb_sizes) is int else emb_sizes

        self._num_classes = num_classes
        self.pool_mode = pool_mode.lower()
        if self.pool_mode == 'xvector' or self.pool_mode == 'tap':
            self._pooling = StatsPoolLayer(feat_in=feat_in, pool_mode=self.pool_mode)
            affine_type = 'linear'
        elif self.pool_mode == 'attention':
            self._pooling = AttentivePoolLayer(inp_filters=feat_in, attention_channels=attention_channels)
            affine_type = 'conv'

        shapes = [self._pooling.feat_in]
        for size in emb_sizes:
            shapes.append(int(size))

        emb_layers = []
        for shape_in, shape_out in zip(shapes[:-1], shapes[1:]):
            layer = self.affine_layer(shape_in, shape_out, learn_mean=False, affine_type=affine_type)
            emb_layers.append(layer)

        self.emb_layers = nn.ModuleList(emb_layers)

        self.final = nn.Linear(shapes[-1], self._num_classes, bias=bias)

        self.apply(lambda x: init_weights(x, mode=init_mode))

    def affine_layer(
        self, inp_shape, out_shape, learn_mean=True, affine_type='conv',
    ):
        if affine_type == 'conv':
            layer = nn.Sequential(
                nn.BatchNorm1d(inp_shape, affine=True, track_running_stats=True),
                nn.Conv1d(inp_shape, out_shape, kernel_size=1),
            )

        else:
            layer = nn.Sequential(
                nn.Linear(inp_shape, out_shape),
                nn.BatchNorm1d(out_shape, affine=learn_mean, track_running_stats=True),
                nn.ReLU(),
            )

        return layer

    @typecheck()
    def forward(self, encoder_output, length=None):
        pool = self._pooling(encoder_output, length)
        embs = []

        for layer in self.emb_layers:
            pool, emb = layer(pool), layer[: self.emb_id](pool)
            embs.append(emb)

        pool = pool.squeeze(-1)
        if self.angular:
            for W in self.final.parameters():
                W = F.normalize(W, p=2, dim=1)
            pool = F.normalize(pool, p=2, dim=1)

        out = self.final(pool)

        return out, embs[-1].squeeze(-1)


class ConvASREncoderAdapter(ConvASREncoder, adapter_mixins.AdapterModuleMixin):

    # Higher level forwarding
    def add_adapter(self, name: str, cfg: dict):
        for jasper_block in self.encoder:  # type: adapter_mixins.AdapterModuleMixin
            cfg = self._update_adapter_cfg_input_dim(jasper_block, cfg)

            jasper_block.set_accepted_adapter_types(self.get_accepted_adapter_types())
            jasper_block.add_adapter(name, cfg)

    def is_adapter_available(self) -> bool:
        return any([jasper_block.is_adapter_available() for jasper_block in self.encoder])

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for jasper_block in self.encoder:  # type: adapter_mixins.AdapterModuleMixin
            jasper_block.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for jasper_block in self.encoder:  # type: adapter_mixins.AdapterModuleMixin
            names.update(jasper_block.get_enabled_adapters())

        names = sorted(list(names))
        return names

    def _update_adapter_cfg_input_dim(self, block: JasperBlock, cfg):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=block.planes)
        return cfg

    def get_accepted_adapter_types(self,) -> Set[type]:
        types = super().get_accepted_adapter_types()

        if len(types) == 0:
            self.set_accepted_adapter_types(
                [adapter_utils.LINEAR_ADAPTER_CLASSPATH,]
            )
            types = self.get_accepted_adapter_types()
        return types


@dataclass
class JasperEncoderConfig:
    filters: int = MISSING
    repeat: int = MISSING
    kernel: List[int] = MISSING
    stride: List[int] = MISSING
    dilation: List[int] = MISSING
    dropout: float = MISSING
    residual: bool = MISSING

    # Optional arguments
    groups: int = 1
    separable: bool = False
    heads: int = -1
    residual_mode: str = "add"
    residual_dense: bool = False
    se: bool = False
    se_reduction_ratio: int = 8
    se_context_size: int = -1
    se_interpolation_mode: str = 'nearest'
    kernel_size_factor: float = 1.0
    stride_last: bool = False


@dataclass
class ConvASREncoderConfig:
    _target_: str = 'nemo.collections.asr.modules.ConvASREncoder'
    jasper: Optional[List[JasperEncoderConfig]] = field(default_factory=list)
    activation: str = MISSING
    feat_in: int = MISSING
    normalization_mode: str = "batch"
    residual_mode: str = "add"
    norm_groups: int = -1
    conv_mask: bool = True
    frame_splicing: int = 1
    init_mode: Optional[str] = "xavier_uniform"


@dataclass
class ConvASRDecoderConfig:
    _target_: str = 'nemo.collections.asr.modules.ConvASRDecoder'
    feat_in: int = MISSING
    num_classes: int = MISSING
    init_mode: Optional[str] = "xavier_uniform"
    vocabulary: Optional[List[str]] = field(default_factory=list)


@dataclass
class ConvASRDecoderClassificationConfig:
    _target_: str = 'nemo.collections.asr.modules.ConvASRDecoderClassification'
    feat_in: int = MISSING
    num_classes: int = MISSING
    init_mode: Optional[str] = "xavier_uniform"
    return_logits: bool = True
    pooling_type: str = 'avg'


"""
Register any additional information
"""
if adapter_mixins.get_registered_adapter(ConvASREncoder) is None:
    adapter_mixins.register_adapter(base_class=ConvASREncoder, adapter_class=ConvASREncoderAdapter)
