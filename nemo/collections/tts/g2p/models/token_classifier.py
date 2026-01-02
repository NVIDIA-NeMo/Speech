# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, Optional

import torch
from torch import nn as nn

from nemo.collections.common.parts import MultiLayerPerceptron, transformer_weights_init
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types import ChannelType, LogitsType, LogprobsType, NeuralType

__all__ = ['Classifier', 'TokenClassifier']


class Classifier(NeuralModule, Exportable):
    """
    A baseclass for modules to perform various classification tasks.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module input ports.
        We implement it here since all NLP classifiers have the same inputs
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the Classifier base module.
        Args:
            hidden_size: the size of the hidden dimension
            dropout: dropout to apply to the input hidden states
        """
        super().__init__()
        self._hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

    def post_init(self, use_transformer_init: bool):
        """
        Common post-processing to be called at the end of concrete Classifiers init methods
        Args:
          use_transformer_init : whether or not to apply transformer_weights_init
        """
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        example = torch.randn(max_batch, max_dim, self._hidden_size).to(sample.device).to(sample.dtype)
        return tuple([example])

    def save_to(self, save_path: str):
        """
        Saves the module to the specified path.
        Args:
            save_path: Path to where to save the module.
        """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """
        Restores the module from the specified path.
        Args:
            restore_path: Path to restore the module from.
        """
        pass


class TokenClassifier(Classifier):
    """
    A module to perform token level classification tasks such as Named entity recognition.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module output ports.
        """
        if not self.log_softmax:
            return {"logits": NeuralType(('B', 'T', 'C'), LogitsType())}
        else:
            return {"log_probs": NeuralType(('B', 'T', 'C'), LogprobsType())}

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        activation: str = 'relu',
        log_softmax: bool = True,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
    ) -> None:
        """
        Initializes the Token Classifier module.

        Args:
            hidden_size: the size of the hidden dimension
            num_classes: number of classes
            num_layers: number of fully connected layers in the multilayer perceptron (MLP)
            activation: activation to usee between fully connected layers in the MLP
            log_softmax: whether to apply softmax to the output of the MLP
            dropout: dropout to apply to the input hidden states
            use_transformer_init: whether to initialize the weights of the classifier head with the same approach used in Transformer
        """
        super().__init__(hidden_size=hidden_size, dropout=dropout)
        self.log_softmax = log_softmax
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, num_layers=num_layers, activation=activation, log_softmax=log_softmax
        )
        self.post_init(use_transformer_init=use_transformer_init)

    @typecheck()
    def forward(self, hidden_states):
        """
        Performs the forward step of the module.
        Args:
            hidden_states: batch of hidden states (for example, from the BERT encoder module)
                [BATCH_SIZE x SEQ_LENGTH x HIDDEN_SIZE]
        Returns: logits value for each class [BATCH_SIZE x SEQ_LENGTH x NUM_CLASSES]
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states)
        return logits
