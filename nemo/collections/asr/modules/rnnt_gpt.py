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

import copy
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import open_dict

from nemo.collections.asr.modules import SpectrogramAugmentation, rnnt_abstract
from nemo.collections.asr.modules.transformer import TransformerEmbedding, TransformerEncoder
from nemo.collections.asr.parts.utils import Hypothesis
from nemo.core.classes import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AdapterModuleMixin
from nemo.core.neural_types import LabelsType, LengthsType, NeuralType
from nemo.utils.decorators import experimental


class GPTDecoderState:
    transformer_state: torch.Tensor  # list[torch.Tensor] -> torch.Tensor
    lengths: torch.Tensor

    def __init__(self, transformer_state: list[torch.Tensor], prev_state: Optional["GPTDecoderState"] = None):
        self.transformer_state = torch.stack(transformer_state)
        batch_size = transformer_state[0].shape[0]
        device = transformer_state[0].device
        if prev_state is None:
            self.lengths = torch.ones([batch_size], device=device, dtype=torch.long)
        else:
            self.lengths = prev_state.lengths + 1
            # fix added state at the last index
            self.transformer_state[:, torch.arange(batch_size, device=device), self.lengths - 1] = (
                self.transformer_state[:, :, -1].clone()
            )

    def filter_(self, active_mask: torch.Tensor):
        if active_mask.sum() == active_mask.shape[0]:
            return  # nothing to filter
        assert active_mask.shape[0] == self.lengths.shape[0]
        self.transformer_state = self.transformer_state[:, active_mask]
        self.lengths = self.lengths[active_mask]
        self._fix_shape()

    def _fix_shape(self):
        # empty state
        if self.lengths.shape[0] == 0:
            return
        max_length = self.lengths.max()
        if max_length >= self.transformer_state[0].shape[1]:
            return  # nothing to fix
        self.transformer_state = torch.narrow(self.transformer_state, dim=2, start=0, length=max_length)

    def reduce_length_(self, blank_mask: torch.Tensor):
        self.lengths -= blank_mask.to(torch.long)
        self._fix_shape()

    def get_mask(self):
        mask = (
            torch.arange(self.transformer_state[0].shape[1], device=self.lengths.device)[None, :]
            < self.lengths[:, None]
        )
        return mask


@experimental
class GPTTransducerDecoder(rnnt_abstract.AbstractRNNTDecoder, Exportable, AdapterModuleMixin):
    def __init__(
        self,
        vocab_size: int,
        embedding_layer: Dict[str, Any],
        predictor_transformer: Dict[str, Any],
        spec_augment: Optional[Dict[str, Any]] = None,
        blank_as_pad=True,
    ):
        super().__init__(vocab_size=vocab_size, blank_idx=vocab_size, blank_as_pad=blank_as_pad)
        if not self.blank_as_pad:
            raise NotImplementedError("blank_as_pad=False not implemented")
        embedding_layer = copy.deepcopy(embedding_layer)  # make local copy
        with open_dict(embedding_layer):
            embedding_layer["vocab_size"] = vocab_size + 1
        self.embedding = TransformerEmbedding(**embedding_layer)
        self.spec_augment = SpectrogramAugmentation(**spec_augment) if spec_augment is not None else None
        self.prediction_network = TransformerEncoder(**predictor_transformer)
        self.blank_idx = vocab_size  # last symbol - as in other RNN-T models

    @property
    def input_types(self):
        return {
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
        }

    def forward(self, targets: torch.Tensor, target_length: torch.Tensor):
        input_ids = targets
        input_lengths = target_length
        batch_size = input_ids.shape[0]
        blank_prefix = torch.full(
            (batch_size, 1), fill_value=self.blank_idx, device=input_ids.device, dtype=input_ids.dtype
        )
        input_ids = torch.cat((blank_prefix, input_ids), dim=1)
        # `<=` => prefix
        input_mask = torch.arange(input_lengths.max() + 1, device=input_ids.device)[None, :] <= input_lengths[:, None]
        input_embed = self.embedding(input_ids)
        if self.spec_augment is not None and self.training:
            with typecheck.disable_semantic_checks():
                input_embed_spaug = self.spec_augment(
                    input_spec=input_embed.transpose(1, 2).detach(), length=input_lengths
                ).transpose(1, 2)
                input_embed[input_embed_spaug == 0.0] = 0.0
        decoder_output = self.prediction_network(
            encoder_states=input_embed,
            encoder_mask=input_mask,
            encoder_mems_list=None,
            return_mems=False,
        )
        return decoder_output.transpose(1, 2), None, input_lengths

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, GPTDecoderState]:
        if y is None:
            raise NotImplementedError
        if add_sos:
            raise NotImplementedError
        return self.predict_step(input_ids=y, state=state)

    def predict_stateless(
        self, labels: torch.Tensor, labels_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        if labels.dtype != torch.long:
            labels = labels.long()
        output, *additional_outputs = self.forward(
            targets=torch.narrow(labels, 1, 0, labels_lengths.max()),
            target_length=labels_lengths,
        )
        # take last elements for batch
        last_labels = output.transpose(1, 2)[torch.arange(output.shape[0]), labels_lengths].unsqueeze(1)
        return (last_labels, *additional_outputs)

    def predict_step(
        self, input_ids: torch.Tensor, state: Optional[GPTDecoderState] = None
    ) -> Tuple[torch.Tensor, GPTDecoderState]:
        batch_size = input_ids.shape[0]
        if input_ids.dim() == 2:
            assert input_ids.shape[1] == 1
            input_ids = input_ids.squeeze(1)
        device = input_ids.device
        input_mask = torch.full((batch_size, 1), fill_value=True, device=device, dtype=torch.bool)
        # input_mask = torch.arange(1, device=device)[None, :] <= input_lengths[:, None]
        if state is None:
            input_embed = self.embedding(input_ids.unsqueeze(1), start_pos=0)
            *transformer_state, decoder_output = self.prediction_network(
                encoder_states=input_embed,
                encoder_mask=input_mask,
                encoder_mems_list=None,
                return_mems=True,
                memory_mask=None,
            )
            new_state = GPTDecoderState(transformer_state=transformer_state, prev_state=None)
            return decoder_output[:, -1].unsqueeze(1), new_state

        # not first step, state is not None
        input_embed = self.embedding(input_ids.unsqueeze(1), start_pos=state.lengths)
        *transformer_state, decoder_output = self.prediction_network(
            encoder_states=input_embed,
            encoder_mask=input_mask,
            encoder_mems_list=state.transformer_state,
            return_mems=True,
            memory_mask=state.get_mask(),
        )
        next_state = GPTDecoderState(transformer_state=transformer_state, prev_state=state)
        return decoder_output[:, -1].unsqueeze(1), next_state

    def initialize_state(self, y: torch.Tensor) -> Optional[list[torch.Tensor]]:
        return None

    def score_hypothesis(
        self, hypothesis: Hypothesis, cache: Dict[Tuple[int], Any]
    ) -> Tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def state_size_is_fixed(self) -> bool:
        return False
