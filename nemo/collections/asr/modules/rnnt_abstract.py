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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.core import NeuralModule


class AbstractRNNTJoint(NeuralModule, ABC):
    """
    An abstract RNNT Joint framework, which can possibly integrate with GreedyRNNTInfer and BeamRNNTInfer classes.
    Represents the abstract RNNT Joint network, which accepts the acoustic model and prediction network
    embeddings in order to compute the joint of the two prior to decoding the output sequence.
    """

    @abstractmethod
    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the joint step of the network.

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the RNNT blank token).

        NOTE:
            The implementation of this model is slightly modified from the original paper.
            The original paper proposes the following steps :
            (enc, dec) -> Expand + Concat + Sum [B, T, U, H1+H2] -> Forward through joint hidden [B, T, U, H] -- *1
            *1 -> Forward through joint final [B, T, U, V + 1].

            We instead split the joint hidden into joint_hidden_enc and joint_hidden_dec and act as follows:
            enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- *1
            dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- *2
            (*1, *2) -> Sum [B, T, U, H] -> Forward through joint final [B, T, U, V + 1].

        Args:
            f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
            g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]

        Returns:
            Logits / log softmaxed tensor of shape (B, T, U, V + 1).
        """
        raise NotImplementedError()

    @property
    def num_classes_with_blank(self):
        raise NotImplementedError()

    @property
    def num_extra_outputs(self):
        raise NotImplementedError()


class AbstractRNNTDecoder(NeuralModule, ABC):
    """
    An abstract RNNT Decoder framework, which can possibly integrate with GreedyRNNTInfer and BeamRNNTInfer classes.
    Represents the abstract RNNT Prediction/Decoder stateful network, which performs autoregressive decoding
    in order to construct the output sequence.

    Args:
        vocab_size: Size of the vocabulary, excluding the RNNT blank token.
        blank_idx: Index of the blank token. Can be 0 or size(vocabulary).
        blank_as_pad: Bool flag, whether to allocate an additional token in the Embedding layer
            of this module in order to treat all RNNT `blank` tokens as pad tokens, thereby letting
            the Embedding layer batch tokens more efficiently.

            It is mandatory to use this for certain Beam RNNT Infer methods - such as TSD, ALSD.
            It is also more efficient to use greedy batch decoding with this flag.
    """

    def __init__(self, vocab_size, blank_idx, blank_as_pad):
        super().__init__()

        self.vocab_size = vocab_size
        self.blank_idx = blank_idx  # first or last index of vocabulary
        self.blank_as_pad = blank_as_pad

        if blank_idx not in [0, vocab_size]:
            raise ValueError("`blank_idx` must be either 0 or the final token of the vocabulary")

    @abstractmethod
    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Stateful prediction of scores and state for a (possibly null) tokenset.
        This method takes various cases into consideration :
        - No token, no state - used for priming the RNN
        - No token, state provided - used for blank token scoring
        - Given token, states - used for scores + new states

        Here:
        B - batch size
        U - label length
        H - Hidden dimension size of RNN
        L - Number of RNN layers

        Args:
            y: Optional torch tensor of shape [B, U] of dtype long which will be passed to the Embedding.
                If None, creates a zero tensor of shape [B, 1, H] which mimics output of pad-token on Embedding.

            state: An optional list of states for the RNN. Eg: For LSTM, it is the state list length is 2.
                Each state must be a tensor of shape [L, B, H].
                If None, and during training mode and `random_state_sampling` is set, will sample a
                normal distribution tensor of the above shape. Otherwise, None will be passed to the RNN.

            add_sos: bool flag, whether a zero vector describing a "start of signal" token should be
                prepended to the above "y" tensor. When set, output size is (B, U + 1, H).

            batch_size: An optional int, specifying the batch size of the `y` tensor.
                Can be infered if `y` and `state` is None. But if both are None, then batch_size cannot be None.

        Returns:
            A tuple  (g, hid) such that -

            If add_sos is False:
                g: (B, U, H)
                hid: (h, c) where h is the final sequence hidden state and c is the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)

            If add_sos is True:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)

        """
        raise NotImplementedError()

    @abstractmethod
    def initialize_state(self, y: torch.Tensor) -> List[torch.Tensor]:
        """
        Initialize the state of the RNN layers, with same dtype and device as input `y`.

        Args:
            y: A torch.Tensor whose device the generated states will be placed on.

        Returns:
            List of torch.Tensor, each of shape [L, B, H], where
                L = Number of RNN layers
                B = Batch size
                H = Hidden size of RNN.
        """
        raise NotImplementedError()

    @abstractmethod
    def score_hypothesis(
        self, hypothesis: Hypothesis, cache: Dict[Tuple[int], Any]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Similar to the predict() method, instead this method scores a Hypothesis during beam search.
        Hypothesis is a dataclass representing one hypothesis in a Beam Search.

        Args:
            hypothesis: Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.

        Returns:
            Returns a tuple (y, states, lm_token) such that:
            y is a torch.Tensor of shape [1, 1, H] representing the score of the last token in the Hypothesis.
            state is a list of RNN states, each of shape [L, 1, H].
            lm_token is the final integer token of the hypothesis.
        """
        raise NotImplementedError()

    def batch_score_hypothesis(
        self, hypotheses: List[Hypothesis], cache: Dict[Tuple[int], Any], batch_states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Used for batched beam search algorithms. Similar to score_hypothesis method.

        Args:
            hypothesis: List of Hypotheses. Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.
            batch_states: List of torch.Tensor which represent the states of the RNN for this batch.
                Each state is of shape [L, B, H]

        Returns:
            Returns a tuple (b_y, b_states, lm_tokens) such that:
            b_y is a torch.Tensor of shape [B, 1, H] representing the scores of the last tokens in the Hypotheses.
            b_state is a list of list of RNN states, each of shape [L, B, H].
                Represented as B x List[states].
            lm_token is a list of the final integer tokens of the hypotheses in the batch.
        """
        raise NotImplementedError()

    def batch_initialize_states(self, batch_states: List[torch.Tensor], decoder_states: List[List[torch.Tensor]]):
        """
        Create batch of decoder states.

       Args:
           batch_states (list): batch of decoder states
              ([L x (B, H)], [L x (B, H)])

           decoder_states (list of list): list of decoder states
               [B x ([L x (1, H)], [L x (1, H)])]

       Returns:
           batch_states (tuple): batch of decoder states
               ([L x (B, H)], [L x (B, H)])
       """
        raise NotImplementedError()

    def batch_select_state(self, batch_states: List[torch.Tensor], idx: int) -> List[List[torch.Tensor]]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (list): batch of decoder states
                ([L x (B, H)], [L x (B, H)])

            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                ([L x (1, H)], [L x (1, H)])
        """
        raise NotImplementedError()

    def batch_concat_states(self, batch_states: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Concatenate a batch of decoder state to a packed state.

        Args:
            batch_states (list): batch of decoder states
                B x ([L x (H)], [L x (H)])

        Returns:
            (tuple): decoder states
                (L x B x H, L x B x H)
        """
        raise NotImplementedError()

    def batch_copy_states(
        self,
        old_states: List[torch.Tensor],
        new_states: List[torch.Tensor],
        ids: List[int],
        value: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """Copy states from new state to old state at certain indices.
        
        Args:
            old_states(list): packed decoder states
                (L x B x H, L x B x H)

            new_states: packed decoder states
                (L x B x H, L x B x H)

            ids (list): List of indices to copy states at.

            value (optional float): If a value should be copied instead of a state slice, a float should be provided

        Returns:
            batch of decoder states with partial copy at ids (or a specific value).
                (L x B x H, L x B x H)
        """
        raise NotImplementedError()
