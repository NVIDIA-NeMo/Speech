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

from collections import OrderedDict

import torch
import torch.distributed
import torch.nn as nn

from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, StackingSubsampling
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType

__all__ = ['RNNEncoder']


class RNNEncoder(NeuralModule, Exportable):
    """
    The RNN-based encoder for ASR models.
    Followed the architecture suggested in the following paper:
    'STREAMING END-TO-END SPEECH RECOGNITION FOR MOBILE DEVICES' by Yanzhang He et al.
    https://arxiv.org/pdf/1811.06621.pdf


    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of RNN
        d_model (int): the hidden size of the model
        proj_size (int): the size of the output projection after each RNN layer
        rnn_type (str): the type of the RNN layers, choices=['lstm, 'gru', 'rnn']
        bidirectional (float): specifies whether RNN layers should be bidirectional or not
            Defaults to True.
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling, choices=['stacking, 'vggnet', 'striding']
            Defaults to stacking.
        subsampling_factor (int): the subsampling factor
            Defaults to 4.
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module for vggnet and striding
            Defaults to -1 which would set it to d_model.
        dropout (float): the dropout rate used between all layers
            Defaults to 0.2.
    """

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(16, self._feat_in, 256).to(next(self.parameters()).device)
        input_example_length = torch.randint(0, 256, (16,)).to(next(self.parameters()).device)
        return tuple([input_example, input_example_length])

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
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
        proj_size: int = -1,
        rnn_type: str = 'lstm',
        bidirectional: bool = True,
        subsampling: str = 'striding',
        subsampling_factor: int = 4,
        subsampling_conv_channels: int = -1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.d_model = d_model
        self._feat_in = feat_in

        if subsampling_conv_channels == -1:
            subsampling_conv_channels = proj_size
        if subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=proj_size,
                    norm=True if 'norm' in subsampling else False,
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=proj_size,
                    conv_channels=subsampling_conv_channels,
                    activation=nn.ReLU(),
                )
        else:
            self.pre_encode = nn.Linear(feat_in, proj_size)

        self._feat_out = proj_size

        self.layers = nn.ModuleList()

        SUPPORTED_RNN = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
        if rnn_type not in SUPPORTED_RNN:
            raise ValueError(f"rnn_type can be one from the following:{SUPPORTED_RNN.keys()}")
        else:
            rnn_module = SUPPORTED_RNN[rnn_type]

        for i in range(n_layers):
            rnn_proj_size = proj_size // 2 if bidirectional else proj_size
            if rnn_type == "lstm":
                layer = rnn_module(
                    input_size=self._feat_out,
                    hidden_size=d_model,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                    proj_size=rnn_proj_size,
                )
            self.layers.append(layer)
            self.layers.append(nn.LayerNorm(proj_size))
            self.layers.append(nn.Dropout(p=dropout))
            self._feat_out = proj_size

    @typecheck()
    def forward(self, audio_signal, length=None):
        max_audio_length: int = audio_signal.size(-1)

        if length is None:
            length = audio_signal.new_full(
                audio_signal.size(0), max_audio_length, dtype=torch.int32, device=self.seq_range.device
            )

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(audio_signal, length)

        for lth, layer in enumerate(self.layers):
            audio_signal = layer(audio_signal)
            if isinstance(audio_signal, tuple):
                audio_signal, _ = audio_signal

        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal, length
