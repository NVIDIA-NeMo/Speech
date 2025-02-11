# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule

__all__ = ['SortformerModules']


class SortformerModules(NeuralModule, Exportable):
    """
    A class including auxiliary functions for Sortformer models.
    This class contains and will contain the following functions that performs streaming features,
    and any neural layers that are not included in the NeMo neural modules (e.g. Transformer, Fast-Conformer).
    """

    def init_weights(self, m):
        """Init weights for linear layers."""
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(
        self,
        num_spks: int = 4,
        hidden_size: int = 192,
        dropout_rate: float = 0.5,
        fc_d_model: int = 512,
        tf_d_model: int = 192,
    ):
        """
        Args:
            num_spks (int):
                Max number of speakers that are processed by the model.
            hidden_size (int):
                Number of hidden units in sequence models and intermediate layers.
            dropout_rate (float):
                Dropout rate for linear layers, CNN and LSTM.
            fc_d_model (int):
                Dimension of the embedding vectors.
            tf_d_model (int):
                Dimension of the embedding vectors.
        """
        super().__init__()
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.hidden_size = tf_d_model
        self.unit_n_spks: int = num_spks
        self.hidden_to_spks = nn.Linear(2 * self.hidden_size, self.unit_n_spks)
        self.first_hidden_to_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.single_hidden_to_spks = nn.Linear(self.hidden_size, self.unit_n_spks)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_proj = nn.Linear(self.fc_d_model, self.tf_d_model)

    def length_to_mask(self, lengths, max_length):
        """
        Convert length values to encoder mask input tensor

        Args:
            lengths (torch.Tensor): tensor containing lengths (frame counts) of sequences
            max_length (int): maximum length (frame count) of the sequences in the batch

        Returns:
            mask (torch.Tensor): tensor of shape (batch_size, max_len) containing 0's
                                in the padded region and 1's elsewhere
        """
        batch_size = lengths.shape[0]
        arange = torch.arange(max_length, device=lengths.device)
        mask = arange.expand(batch_size, max_length) < lengths.unsqueeze(1)
        return mask

    def forward_speaker_sigmoids(self, hidden_out):
        """
        A set of layers for predicting speaker probabilities with a sigmoid activation function.

        Args:
            hidden_out (torch.Tensor): tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            preds (torch.Tensor): tensor of shape (batch_size, seq_len, num_spks) containing speaker probabilities
        """
        hidden_out = self.dropout(F.relu(hidden_out))
        hidden_out = self.first_hidden_to_hidden(hidden_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        spk_preds = self.single_hidden_to_spks(hidden_out)
        preds = nn.Sigmoid()(spk_preds)
        return preds
