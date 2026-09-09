# ! /usr/bin/python
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


from dataclasses import dataclass
from typing import List, Optional

import torch
from omegaconf import DictConfig

from nemo.collections.asr.modules.transformer import (
    BeamSearchSequenceGenerator,
    GreedySequenceGenerator,
    TopKSequenceGenerator,
)
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes.module import NeuralModule


@dataclass
class SequenceGeneratorConfig:
    type: str = "greedy"  # choices=[greedy, topk, beam]
    max_sequence_length: int = 512
    max_delta_length: int = -1
    temperature: float = 1.0  # for top-k sampling
    beam_size: int = 1  # K for top-k sampling, N for beam search
    len_pen: float = 0.0  # for beam-search


class SequenceGenerator:
    """
    Wrapper class for sequence generators for NeMo transformers.
    """

    TYPE_GREEDY = "greedy"
    TYPE_TOPK = "topk"
    TYPE_BEAM = "beam"
    SEARCHER_TYPES = [TYPE_GREEDY, TYPE_TOPK, TYPE_BEAM]

    def __init__(
        self,
        cfg: DictConfig,
        embedding: NeuralModule,
        decoder: NeuralModule,
        log_softmax: NeuralModule,
        tokenizer: TokenizerSpec,
    ) -> None:
        super().__init__()

        self._type = cfg.get("type", "greedy")
        self.tokenizer = tokenizer
        self.pad_id = getattr(tokenizer, "pad_id", 0)
        self.eos_id = getattr(tokenizer, "eos_id", -1)
        self.bos_id = getattr(tokenizer, "bos_id", -1)
        common_args = {
            "pad": self.pad_id,
            "bos": self.bos_id,
            "eos": self.eos_id,
            "max_sequence_length": cfg.get("max_sequence_length", 512),
            "max_delta_length": cfg.get("max_delta_length", -1),
            "batch_size": cfg.get("batch_size", 1),
        }
        if self._type == self.TYPE_GREEDY:
            self.generator = GreedySequenceGenerator(embedding, decoder, log_softmax, **common_args)
        elif self._type == self.TYPE_TOPK:
            beam_size = cfg.get("beam_size", 1)
            temperature = cfg.get("temperature", 1.0)
            self.generator = TopKSequenceGenerator(
                embedding, decoder, log_softmax, beam_size, temperature, **common_args
            )
        elif self._type == self.TYPE_BEAM:
            beam_size = cfg.get("beam_size", 1)
            len_pen = cfg.get("len_pen", 0.0)
            self.generator = BeamSearchSequenceGenerator(
                embedding, decoder, log_softmax, beam_size, len_pen, **common_args
            )
        else:
            raise ValueError(
                f"Sequence Generator only supports one of {self.SEARCH_TYPES}, but got {self._type} instead."
            )

    def __call__(
        self,
        encoder_states: torch.Tensor,
        encoder_input_mask: torch.Tensor = None,
        return_beam_scores: bool = False,
        pad_max_len: Optional[int] = None,
        return_length: bool = False,
    ):
        """
        Generate sequence tokens given the input encoder states and masks.
        Params:
        -   encoder_states: a torch Tensor of shape BxTxD
        -   encoder_input_mask: a binary tensor of shape BxTxD
        -   return_beam_scores: whether to return beam scores
        -   pad_max_len: optional int, set it to pad all sequence to the same length
        -   return_length: whether to return the lengths for generated sequences (shape B)
        Returns:
        -   generated tokens tensor of shape BxT
        """
        predictions = self.generator(
            encoder_hidden_states=encoder_states,
            encoder_input_mask=encoder_input_mask,
            return_beam_scores=return_beam_scores,
        )

        if pad_max_len:
            predictions = pad_sequence(predictions, pad_max_len, self.pad_id)

        if return_length:
            return predictions, self.get_seq_length(predictions)

        return predictions

    def get_seq_length(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Get sequence length.
        Params:
        -   seq: batched sequence tensor of shape BxTxD
        Returns:
        -   tensor of shape B, where each element is the length of the sequence
        """
        lengths = seq.size(1) * torch.ones(seq.size(0), device=seq.device).long()
        pos = (seq == self.eos_id).long().nonzero()
        seq_lengths = torch.scatter(lengths, dim=0, index=pos[:, 0], src=pos[:, 1])
        return seq_lengths

    def decode_semantics_from_tokens(self, seq_tokens: torch.Tensor) -> List[str]:
        """
        Decode tokens into strings
        Rarams:
        -   seq_tokens: integer tensor of shape BxT
        Returns:
        -   list of strings
        """
        semantics_list = []
        # Drop sequence tokens to CPU
        seq_tokens = seq_tokens.detach().long().cpu()
        seq_lengths = self.get_seq_length(seq_tokens)
        # iterate over batch
        for ind in range(seq_tokens.shape[0]):
            tokens = seq_tokens[ind].numpy().tolist()
            length = seq_lengths[ind].long().cpu().item()
            tokens = tokens[:length]
            text = "".join(self.tokenizer.tokenizer.decode_ids(tokens))
            semantics_list.append(text)
        return semantics_list


def get_seq_length(seq: torch.Tensor, eos_id: int) -> torch.Tensor:
    """
    Get sequence length.
    Params:
    -   seq: batched sequence tensor of shape BxTxD
    -   eos_id: integer representing the end of sentence
    Returns:
    -   tensor of shape B, where each element is the length of the sequence
    """
    lengths = seq.size(1) * torch.ones(seq.size(0), device=seq.device).long()
    pos = (seq == eos_id).long().nonzero()
    seq_lengths = torch.scatter(lengths, dim=0, index=pos[:, 0], src=pos[:, 1])
    return seq_lengths


def pad_sequence(seq: torch.Tensor, max_len: int, pad_token: int = 0) -> torch.Tensor:
    """
    Params:
        - seq: integer token sequences of shape BxT
        - max_len: integer for max sequence length
        - pad_token: integer token for padding
    Returns:
        - padded sequence of shape B x max_len
    """
    batch = seq.size(0)
    curr_len = seq.size(1)
    if curr_len >= max_len:
        return seq

    padding = torch.zeros(batch, max_len - curr_len, dtype=seq.dtype, device=seq.device).fill_(pad_token)
    return torch.cat([seq, padding], dim=1)


def get_seq_mask(seq: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """
    Get the sequence mask based on the actual length of each sequence
    Params:
        - seq: tensor of shape [BxLxD]
        - seq_len: tensor of shape [B]
    Returns:
        - binary mask of shape [BxL]
    """
    mask = torch.arange(seq.size(1))[None, :].to(seq.device) < seq_lens[:, None]
    return mask.to(seq.device, dtype=bool)
