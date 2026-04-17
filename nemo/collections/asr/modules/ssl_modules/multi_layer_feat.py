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

from typing import List, Optional, Tuple

import torch
import torch.distributed
import torch.nn as nn

from nemo.collections.asr.modules import (
    AudioToMelSpectrogramPreprocessor,
    ConformerEncoder,
    ConformerMultiLayerFeatureExtractor,
)
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.mixins import AccessMixin


class Aggregator(nn.Module):
    AVAILABLE_POOLING = ["cat", "sum", "mean", "avg", "max", "min", "none", "weighted_sum"]

    def __init__(self, mode, weights, layer_idx_list, channel_idx: int = 1):
        """
        Args:
            mode: Aggregation mode. One of ["cat", "sum", "mean", "avg", "max", "min", "none", "weighted_sum"]
            weights: Weights for weighted sum aggregation. If None, weights are initialized to 1/num_layers.
            layer_idx_list: List of layer indices to aggregate.
            channel_idx: Channel dimension index of the input tensors.
        """
        super().__init__()
        self.mode = mode
        self.channel_idx = channel_idx
        self.weights = weights
        if self.mode not in self.AVAILABLE_POOLING:
            raise ValueError(f"Unknown mode `{self.mode}`, available modes are {self.AVAILABLE_POOLING}")
        if self.mode == "weighted_sum" and self.weights is None:
            self.weights = nn.Parameter(torch.ones(len(layer_idx_list)) / len(layer_idx_list))

    def _forward_for_weighted_sum(
        self, encoded: List[torch.Tensor], encoded_len: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_weighted = [encoded[i] * self.weights[i] for i in range(len(encoded))]
        encoded_weighted = torch.sum(torch.stack(encoded_weighted, dim=-1), dim=-1)
        return encoded_weighted, encoded_len[0]

    def forward(
        self, encoded: List[torch.Tensor], encoded_len: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoded: List of tensors of shape [B, D, T] representing the encoded features from different layers.
            encoded_len: List of tensors of shape [B] representing the lengths of the encoded features.
        Returns:
            aggregated: Aggregated tensor of shape [B, D, T] representing the aggregated features.
            aggregated_len: Tensor of shape [B] representing the lengths of the aggregated features.
        """

        if self.mode == "cat":
            return torch.cat(encoded, dim=self.channel_idx), encoded_len[0]
        elif self.mode == "sum":
            return torch.cat([x.unsqueeze(-1) for x in encoded], dim=-1).sum(dim=-1), encoded_len[0]
        elif self.mode == "mean" or self.mode == "avg":
            return torch.cat([x.unsqueeze(-1) for x in encoded], dim=-1).mean(dim=-1), encoded_len[0]
        elif self.mode == "max":
            return torch.cat([x.unsqueeze(-1) for x in encoded], dim=-1).max(dim=-1), encoded_len[0]
        elif self.mode == "min":
            return torch.cat([x.unsqueeze(-1) for x in encoded], dim=-1).min(dim=-1), encoded_len[0]
        elif self.mode == "none":
            return encoded, encoded_len
        elif self.mode == "weighted_sum":
            return self._forward_for_weighted_sum(encoded, encoded_len)
        else:
            raise ValueError(f"Unknown mode {self.mode}")


class ConformerMultiLayerFeaturePreprocessor(NeuralModule, Exportable, AccessMixin):
    """
    This class is used to replace the AudioToMelSpectrogramPreprocessor such that
    the input to the actual model encoder is the multi-layer features from a pre-trained ConformerEncoder.
    """

    def __init__(
        self,
        aggregator: nn.Module,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        encoder: ConformerEncoder,
        spec_augment=None,
        layer_idx_list: Optional[List[int]] = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.spec_augmentation = spec_augment
        self.feature_extractor = ConformerMultiLayerFeatureExtractor(
            encoder=encoder, aggregator=aggregator, layer_idx_list=layer_idx_list
        )
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.freeze()

    def forward(self, input_signal, length):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            length: Vector of length B, that contains the individual lengths of the audio
                sequences.
        Returns:
            encoded: A tensor of shape [B, D, T], where D represents the number of
                feature dimensions extracted from the audio signal, and T represents the
                number of timesteps in the processed audio signal.
            encoded_len: A tensor of shape [B], that contains the lengths of the audio sequences.
        """

        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal,
            length=length,
        )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.feature_extractor(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len
