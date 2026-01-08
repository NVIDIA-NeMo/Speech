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

import torch
import torch.nn as nn

from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LogprobsType, NeuralType

__all__ = ['LSTMDecoder']


class LSTMDecoder(NeuralModule, Exportable):
    """
    Simple LSTM Decoder for ASR models
    Args:
        feat_in (int): size of the input features
        num_classes (int): the size of the vocabulary
        lstm_hidden_size (int): hidden size of the LSTM layers
        vocabulary (vocab): The vocabulary
        bidirectional (bool): default is False. Whether LSTMs are bidirectional or not
        num_layers (int): default is 1. Number of LSTM layers stacked
    """

    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        return OrderedDict({"logprobs": NeuralType(('B', 'T', 'D'), LogprobsType())})

    def __init__(
        self,
        feat_in,
        num_classes,
        lstm_hidden_size,
        vocabulary=None,
        bidirectional=False,
        num_layers=1,
        add_blank=True,
    ):
        super().__init__()

        if vocabulary is not None:
            if num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, it's length should be equal to the num_classes. "
                    f"Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1 if add_blank else num_classes

        self.lstm_layer = nn.LSTM(
            input_size=feat_in,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_hidden_size = 2 * lstm_hidden_size if bidirectional else lstm_hidden_size
        self.linear_layer = torch.nn.Linear(in_features=lstm_hidden_size, out_features=self._num_classes)

    @typecheck()
    def forward(self, encoder_output):
        output = encoder_output.transpose(1, 2)
        output, _ = self.lstm_layer(output)
        output = self.linear_layer(output)
        return torch.nn.functional.log_softmax(output, dim=-1)

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def num_classes_with_blank(self):
        return self._num_classes
