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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.parts.submodules import stateless_net
from nemo.collections.asr.parts.utils import adapter_utils, rnnt_utils
from nemo.collections.common.parts import rnn
from nemo.core.classes import adapter_mixins, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AdapterModuleMixin
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    ElementType,
    EmbeddedTextType,
    LabelsType,
    LengthsType,
    LogprobsType,
    LossType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging


class StatelessTransducerDecoder(rnnt_abstract.AbstractRNNTDecoder, Exportable):
    """A Stateless Neural Network Transducer Decoder / Prediction Network.
    An RNN-T Decoder/Prediction stateless network that simply takes concatenation of embeddings of the history tokens as the output.

    Args:
        prednet: A dict-like object which contains the following key-value pairs.
            pred_hidden: int specifying the hidden dimension of the prediction net.

            dropout: float, set to 0.0 by default. Optional dropout applied at the end of the final LSTM RNN layer.

        vocab_size: int, specifying the vocabulary size of the embedding layer of the Prediction network,
            excluding the RNNT blank token.

        context_size: int, specifying the size of the history context used for this decoder.

        normalization_mode: Can be either None, 'layer'. By default, is set to None.
            Defines the type of normalization applied to the RNN layer.

    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
            "states": [NeuralType(('B', 'T'), LabelsType(), optional=True)],
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "prednet_lengths": NeuralType(tuple('B'), LengthsType()),
            "states": [NeuralType(('B', 'T'), LabelsType(), optional=True)],
        }

    def input_example(self, max_batch=1, max_dim=1):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        length = max_dim
        targets = torch.full(fill_value=self.blank_idx, size=(max_batch, length), dtype=torch.int32).to(
            next(self.parameters()).device
        )
        target_length = torch.randint(0, length, size=(max_batch,), dtype=torch.int32).to(
            next(self.parameters()).device
        )
        states = tuple(self.initialize_state(targets.float()))
        return (targets, target_length, states)

    def _prepare_for_export(self, **kwargs):
        self._rnnt_export = True
        super()._prepare_for_export(**kwargs)

    def __init__(
        self,
        prednet: Dict[str, Any],
        vocab_size: int,
        context_size: int = 1,
        normalization_mode: Optional[str] = None,
    ):
        # Required arguments
        self.pred_hidden = prednet['pred_hidden']
        self.blank_idx = vocab_size
        self.context_size = context_size

        # Initialize the model (blank token increases vocab size by 1)
        super().__init__(vocab_size=vocab_size, blank_idx=self.blank_idx, blank_as_pad=True)

        # Optional arguments
        dropout = prednet.get('dropout', 0.0)

        self.prediction = self._predict_modules(
            **{
                "context_size": context_size,
                "vocab_size": vocab_size,
                "emb_dim": self.pred_hidden,
                "blank_idx": self.blank_idx,
                "normalization_mode": normalization_mode,
                "dropout": dropout,
            }
        )
        self._rnnt_export = False

    @typecheck()
    def forward(self, targets, target_length, states=None):
        # y: (B, U)
        y = rnn.label_collate(targets)

        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        if self._rnnt_export:
            add_sos = False
        else:
            add_sos = True

        g, state = self.predict(y, state=states, add_sos=add_sos)  # (B, U, D)
        g = g.transpose(1, 2)  # (B, D, U)

        return g, target_length, state

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Stateful prediction of scores and state for a tokenset.

        Here:
        B - batch size
        U - label length
        C - context size for stateless decoder
        D - total embedding size

        Args:
            y: Optional torch tensor of shape [B, U] of dtype long which will be passed to the Embedding.
                If None, creates a zero tensor of shape [B, 1, D] which mimics output of pad-token on Embedding.

            state: An optional one-element list of one tensor. The tensor is used to store previous context labels.
                The tensor uses type long and is of shape [B, C].

            add_sos: bool flag, whether a zero vector describing a "start of signal" token should be
                prepended to the above "y" tensor. When set, output size is (B, U + 1, D).

            batch_size: An optional int, specifying the batch size of the `y` tensor.
                Can be infered if `y` and `state` is None. But if both are None, then batch_size cannot be None.

        Returns:
            A tuple  (g, state) such that -

            If add_sos is False:
                g: (B, U, D)
                state: [(B, C)] storing the history context including the new words in y.

            If add_sos is True:
                g: (B, U + 1, D)
                state: [(B, C)] storing the history context including the new words in y.

        """
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            if y.device != device:
                y = y.to(device)

            y, state = self.prediction(y, state)

        else:
            # Y is not provided, assume zero tensor with shape [B, 1, D] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            B, U, D = y.shape
            start = torch.zeros((B, 1, D), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, D)
        else:
            start = None  # makes del call later easier

        del start
        return y, state

    def _predict_modules(self, **kwargs):
        """
        Prepare the trainable parameters of the Prediction Network.

        Args:
            vocab_size: Vocab size (excluding the blank token).
            pred_n_hidden: Hidden size of the RNNs.
            norm: Type of normalization to perform in RNN.
            dropout: Whether to apply dropout to RNN.
        """

        net = stateless_net.StatelessNet(**kwargs)
        return net

    def score_hypothesis(
        self, hypothesis: rnnt_utils.Hypothesis, cache: Dict[Tuple[int], Any]
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
        if hypothesis.dec_state is not None:
            device = hypothesis.dec_state[0].device
        else:
            _p = next(self.parameters())
            device = _p.device

        # parse "blank" tokens in hypothesis
        if len(hypothesis.y_sequence) > 0 and hypothesis.y_sequence[-1] == self.blank_idx:
            blank_state = True
        else:
            blank_state = False

        # Convert last token of hypothesis to torch.Tensor
        target = torch.full([1, 1], fill_value=hypothesis.y_sequence[-1], device=device, dtype=torch.long)
        lm_token = target[:, -1]  # [1]

        # Convert current hypothesis into a tuple to preserve in cache
        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            # Obtain score for target token and new states
            if blank_state:
                y, new_state = self.predict(None, state=None, add_sos=False, batch_size=1)  # [1, 1, H]

            else:
                y, new_state = self.predict(
                    target, state=hypothesis.dec_state, add_sos=False, batch_size=1
                )  # [1, 1, H]

            y = y[:, -1:, :]  # Extract just last state : [1, 1, H]
            cache[sequence] = (y, new_state)

        return y, new_state, lm_token

    def initialize_state(self, y: torch.Tensor) -> List[torch.Tensor]:
        batch = y.size(0)
        state = [torch.ones([batch, self.context_size], dtype=torch.long, device=y.device) * self.blank_idx]
        return state

    def batch_initialize_states(self, batch_states: List[torch.Tensor], decoder_states: List[List[torch.Tensor]]):
        """
        Create batch of decoder states.

       Args:
           batch_states (list): batch of decoder states
              ([(B, H)])

           decoder_states (list of list): list of decoder states
               [B x ([(1, C)]]

       Returns:
           batch_states (tuple): batch of decoder states
               ([(B, C)])
       """
        new_state = torch.stack([s[0] for s in decoder_states])

        return [new_state]

    def batch_select_state(self, batch_states: List[torch.Tensor], idx: int) -> List[List[torch.Tensor]]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (list): batch of decoder states
                [(B, C)]

            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                [(C)]
        """
        if batch_states is not None:
            states = batch_states[0][idx]
            states = (
                states.long()
            )  # beam search code assumes the batch_states tensor is always of float type, so need conversion
            return [states]
        else:
            return None

    def batch_concat_states(self, batch_states: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Concatenate a batch of decoder state to a packed state.

        Args:
            batch_states (list): batch of decoder states
                B x ([(C)]

        Returns:
            (tuple): decoder states
                [(B x C)]
        """
        state_list = []
        batch_list = []
        for sample_id in range(len(batch_states)):
            tensor = torch.stack(batch_states[sample_id])  # [1, H]
            batch_list.append(tensor)

        state_tensor = torch.cat(batch_list, 0)  # [B, H]
        state_list.append(state_tensor)

        return state_list

    def batch_copy_states(
        self,
        old_states: List[torch.Tensor],
        new_states: List[torch.Tensor],
        ids: List[int],
        value: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """Copy states from new state to old state at certain indices.

        Args:
            old_states: packed decoder states
                single element list of (B x C)

            new_states: packed decoder states
                single element list of (B x C)

            ids (list): List of indices to copy states at.

            value (optional float): If a value should be copied instead of a state slice, a float should be provided

        Returns:
            batch of decoder states with partial copy at ids (or a specific value).
                (B x C)
        """

        if value is None:
            old_states[0][ids, :] = new_states[0][ids, :]

        return old_states

    def batch_score_hypothesis(
        self, hypotheses: List[rnnt_utils.Hypothesis], cache: Dict[Tuple[int], Any], batch_states: List[torch.Tensor]
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
        final_batch = len(hypotheses)
        if final_batch == 0:
            raise ValueError("No hypotheses was provided for the batch!")

        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        tokens = []
        process = []
        done = [None for _ in range(final_batch)]

        # For each hypothesis, cache the last token of the sequence and the current states
        for i, hyp in enumerate(hypotheses):
            sequence = tuple(hyp.y_sequence)

            if sequence in cache:
                done[i] = cache[sequence]
            else:
                tokens.append(hyp.y_sequence[-1])
                process.append((sequence, hyp.dec_state))

        if process:
            batch = len(process)

            # convert list of tokens to torch.Tensor, then reshape.
            tokens = torch.tensor(tokens, device=device, dtype=torch.long).view(batch, -1)
            dec_states = self.initialize_state(tokens)  # [B, C]
            dec_states = self.batch_initialize_states(dec_states, [d_state for seq, d_state in process])

            y, dec_states = self.predict(
                tokens, state=dec_states, add_sos=False, batch_size=batch
            )  # [B, 1, H], List([L, 1, H])

            dec_states = tuple(state.to(dtype=dtype) for state in dec_states)

        # Update done states and cache shared by entire batch.
        j = 0
        for i in range(final_batch):
            if done[i] is None:
                # Select sample's state from the batch state list
                new_state = self.batch_select_state(dec_states, j)

                # Cache [1, H] scores of the current y_j, and its corresponding state
                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        # Set the incoming batch states with the new states obtained from `done`.
        batch_states = self.batch_initialize_states(batch_states, [d_state for y_j, d_state in done])

        # Create batch of all output scores
        # List[1, 1, H] -> [B, 1, H]
        batch_y = torch.stack([y_j for y_j, d_state in done])

        # Extract the last tokens from all hypotheses and convert to a tensor
        lm_tokens = torch.tensor([h.y_sequence[-1] for h in hypotheses], device=device, dtype=torch.long).view(
            final_batch
        )

        return batch_y, batch_states, lm_tokens


class RNNTDecoder(rnnt_abstract.AbstractRNNTDecoder, Exportable, AdapterModuleMixin):
    """A Recurrent Neural Network Transducer Decoder / Prediction Network (RNN-T Prediction Network).
    An RNN-T Decoder/Prediction network, comprised of a stateful LSTM model.

    Args:
        prednet: A dict-like object which contains the following key-value pairs.
            pred_hidden: int specifying the hidden dimension of the prediction net.
            pred_rnn_layers: int specifying the number of rnn layers.

            Optionally, it may also contain the following:
            forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)
            t_max: int value, set to None by default. If an int is specified, performs Chrono Initialization
                of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
                of training.
                Reference:
                [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)
            weights_init_scale: Float scale of the weights after initialization. Setting to lower than one
                sometimes helps reduce variance between runs.
            hidden_hidden_bias_scale: Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
                the default behaviour.
            dropout: float, set to 0.0 by default. Optional dropout applied at the end of the final LSTM RNN layer.

        vocab_size: int, specifying the vocabulary size of the embedding layer of the Prediction network,
            excluding the RNNT blank token.

        normalization_mode: Can be either None, 'batch' or 'layer'. By default, is set to None.
            Defines the type of normalization applied to the RNN layer.

        random_state_sampling: bool, set to False by default. When set, provides normal-distribution
            sampled state tensors instead of zero tensors during training.
            Reference:
            [Recognizing long-form speech using streaming end-to-end models](https://arxiv.org/abs/1910.11455)

        blank_as_pad: bool, set to True by default. When set, will add a token to the Embedding layer of this
            prediction network, and will treat this token as a pad token. In essence, the RNNT pad token will
            be treated as a pad token, and the embedding layer will return a zero tensor for this token.

            It is set by default as it enables various batch optimizations required for batched beam search.
            Therefore, it is not recommended to disable this flag.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
            "states": [NeuralType(('D', 'B', 'D'), ElementType(), optional=True)],  # must always be last
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "prednet_lengths": NeuralType(tuple('B'), LengthsType()),
            "states": [NeuralType((('D', 'B', 'D')), ElementType(), optional=True)],  # must always be last
        }

    def input_example(self, max_batch=1, max_dim=1):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        length = max_dim
        targets = torch.full(fill_value=self.blank_idx, size=(max_batch, length), dtype=torch.int32).to(
            next(self.parameters()).device
        )
        target_length = torch.randint(0, length, size=(max_batch,), dtype=torch.int32).to(
            next(self.parameters()).device
        )
        states = tuple(self.initialize_state(targets.float()))
        return (targets, target_length, states)

    def _prepare_for_export(self, **kwargs):
        self._rnnt_export = True
        super()._prepare_for_export(**kwargs)

    def __init__(
        self,
        prednet: Dict[str, Any],
        vocab_size: int,
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
        blank_as_pad: bool = True,
    ):
        # Required arguments
        self.pred_hidden = prednet['pred_hidden']
        self.pred_rnn_layers = prednet["pred_rnn_layers"]
        self.blank_idx = vocab_size

        # Initialize the model (blank token increases vocab size by 1)
        super().__init__(vocab_size=vocab_size, blank_idx=self.blank_idx, blank_as_pad=blank_as_pad)

        # Optional arguments
        forget_gate_bias = prednet.get('forget_gate_bias', 1.0)
        t_max = prednet.get('t_max', None)
        weights_init_scale = prednet.get('weights_init_scale', 1.0)
        hidden_hidden_bias_scale = prednet.get('hidden_hidden_bias_scale', 0.0)
        dropout = prednet.get('dropout', 0.0)
        self.random_state_sampling = random_state_sampling

        self.prediction = self._predict_modules(
            vocab_size=vocab_size,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            pred_rnn_layers=self.pred_rnn_layers,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            norm=normalization_mode,
            weights_init_scale=weights_init_scale,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            dropout=dropout,
            rnn_hidden_size=prednet.get("rnn_hidden_size", -1),
        )
        self._rnnt_export = False

    @typecheck()
    def forward(self, targets, target_length, states=None):
        # y: (B, U)
        y = rnn.label_collate(targets)

        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        if self._rnnt_export:
            add_sos = False
        else:
            add_sos = True

        g, states = self.predict(y, state=states, add_sos=add_sos)  # (B, U, D)
        g = g.transpose(1, 2)  # (B, D, U)

        return g, target_length, states

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        add_sos: bool = True,
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
                If None, creates a zero tensor of shape [B, 1, H] which mimics output of pad-token on EmbeddiNg.

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
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            if y.device != device:
                y = y.to(device)

            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            # Y is not provided, assume zero tensor with shape [B, 1, H] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        # If in training mode, and random_state_sampling is set,
        # initialize state to random normal distribution tensor.
        if state is None:
            if self.random_state_sampling and self.training:
                state = self.initialize_state(y)

        # Forward step through RNN
        y = y.transpose(0, 1)  # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # (B, U + 1, H)

        del y, start, state

        # Adapter module forward step
        if self.is_adapter_available():
            g = self.forward_enabled_adapters(g)

        return g, hid

    def _predict_modules(
        self,
        vocab_size,
        pred_n_hidden,
        pred_rnn_layers,
        forget_gate_bias,
        t_max,
        norm,
        weights_init_scale,
        hidden_hidden_bias_scale,
        dropout,
        rnn_hidden_size,
    ):
        """
        Prepare the trainable parameters of the Prediction Network.

        Args:
            vocab_size: Vocab size (excluding the blank token).
            pred_n_hidden: Hidden size of the RNNs.
            pred_rnn_layers: Number of RNN layers.
            forget_gate_bias: Whether to perform unit forget gate bias.
            t_max: Whether to perform Chrono LSTM init.
            norm: Type of normalization to perform in RNN.
            weights_init_scale: Float scale of the weights after initialization. Setting to lower than one
                sometimes helps reduce variance between runs.
            hidden_hidden_bias_scale: Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
                the default behaviour.
            dropout: Whether to apply dropout to RNN.
            rnn_hidden_size: the hidden size of the RNN, if not specified, pred_n_hidden would be used
        """
        if self.blank_as_pad:
            embed = torch.nn.Embedding(vocab_size + 1, pred_n_hidden, padding_idx=self.blank_idx)
        else:
            embed = torch.nn.Embedding(vocab_size, pred_n_hidden)

        layers = torch.nn.ModuleDict(
            {
                "embed": embed,
                "dec_rnn": rnn.rnn(
                    input_size=pred_n_hidden,
                    hidden_size=rnn_hidden_size if rnn_hidden_size > 0 else pred_n_hidden,
                    num_layers=pred_rnn_layers,
                    norm=norm,
                    forget_gate_bias=forget_gate_bias,
                    t_max=t_max,
                    dropout=dropout,
                    weights_init_scale=weights_init_scale,
                    hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                    proj_size=pred_n_hidden if pred_n_hidden < rnn_hidden_size else 0,
                ),
            }
        )
        return layers

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
        batch = y.size(0)
        if self.random_state_sampling and self.training:
            state = [
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            ]

        else:
            state = [
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            ]
        return state

    def score_hypothesis(
        self, hypothesis: rnnt_utils.Hypothesis, cache: Dict[Tuple[int], Any]
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
        if hypothesis.dec_state is not None:
            device = hypothesis.dec_state[0].device
        else:
            _p = next(self.parameters())
            device = _p.device

        # parse "blank" tokens in hypothesis
        if len(hypothesis.y_sequence) > 0 and hypothesis.y_sequence[-1] == self.blank_idx:
            blank_state = True
        else:
            blank_state = False

        # Convert last token of hypothesis to torch.Tensor
        target = torch.full([1, 1], fill_value=hypothesis.y_sequence[-1], device=device, dtype=torch.long)
        lm_token = target[:, -1]  # [1]

        # Convert current hypothesis into a tuple to preserve in cache
        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            # Obtain score for target token and new states
            if blank_state:
                y, new_state = self.predict(None, state=None, add_sos=False, batch_size=1)  # [1, 1, H]

            else:
                y, new_state = self.predict(
                    target, state=hypothesis.dec_state, add_sos=False, batch_size=1
                )  # [1, 1, H]

            y = y[:, -1:, :]  # Extract just last state : [1, 1, H]
            cache[sequence] = (y, new_state)

        return y, new_state, lm_token

    def batch_score_hypothesis(
        self, hypotheses: List[rnnt_utils.Hypothesis], cache: Dict[Tuple[int], Any], batch_states: List[torch.Tensor]
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
        final_batch = len(hypotheses)

        if final_batch == 0:
            raise ValueError("No hypotheses was provided for the batch!")

        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        tokens = []
        process = []
        done = [None for _ in range(final_batch)]

        # For each hypothesis, cache the last token of the sequence and the current states
        for i, hyp in enumerate(hypotheses):
            sequence = tuple(hyp.y_sequence)

            if sequence in cache:
                done[i] = cache[sequence]
            else:
                tokens.append(hyp.y_sequence[-1])
                process.append((sequence, hyp.dec_state))

        if process:
            batch = len(process)

            # convert list of tokens to torch.Tensor, then reshape.
            tokens = torch.tensor(tokens, device=device, dtype=torch.long).view(batch, -1)
            dec_states = self.initialize_state(tokens.to(dtype=dtype))  # [L, B, H]
            dec_states = self.batch_initialize_states(dec_states, [d_state for seq, d_state in process])

            y, dec_states = self.predict(
                tokens, state=dec_states, add_sos=False, batch_size=batch
            )  # [B, 1, H], List([L, 1, H])

            dec_states = tuple(state.to(dtype=dtype) for state in dec_states)

        # Update done states and cache shared by entire batch.
        j = 0
        for i in range(final_batch):
            if done[i] is None:
                # Select sample's state from the batch state list
                new_state = self.batch_select_state(dec_states, j)

                # Cache [1, H] scores of the current y_j, and its corresponding state
                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        # Set the incoming batch states with the new states obtained from `done`.
        batch_states = self.batch_initialize_states(batch_states, [d_state for y_j, d_state in done])

        # Create batch of all output scores
        # List[1, 1, H] -> [B, 1, H]
        batch_y = torch.stack([y_j for y_j, d_state in done])

        # Extract the last tokens from all hypotheses and convert to a tensor
        lm_tokens = torch.tensor([h.y_sequence[-1] for h in hypotheses], device=device, dtype=torch.long).view(
            final_batch
        )

        return batch_y, batch_states, lm_tokens

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
        # LSTM has 2 states
        new_states = [[] for _ in range(len(decoder_states[0]))]
        for layer in range(self.pred_rnn_layers):
            for state_id in range(len(decoder_states[0])):
                # batch_states[state_id][layer] = torch.stack([s[state_id][layer] for s in decoder_states])
                new_state_for_layer = torch.stack([s[state_id][layer] for s in decoder_states])
                new_states[state_id].append(new_state_for_layer)

        for state_id in range(len(decoder_states[0])):
            new_states[state_id] = torch.stack([state for state in new_states[state_id]])

        return new_states

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
        if batch_states is not None:
            state_list = []
            for state_id in range(len(batch_states)):
                states = [batch_states[state_id][layer][idx] for layer in range(self.pred_rnn_layers)]
                state_list.append(states)

            return state_list
        else:
            return None

    def batch_concat_states(self, batch_states: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Concatenate a batch of decoder state to a packed state.

        Args:
            batch_states (list): batch of decoder states
                B x ([L x (H)], [L x (H)])

        Returns:
            (tuple): decoder states
                (L x B x H, L x B x H)
        """
        state_list = []

        for state_id in range(len(batch_states[0])):
            batch_list = []
            for sample_id in range(len(batch_states)):
                tensor = torch.stack(batch_states[sample_id][state_id])  # [L, H]
                tensor = tensor.unsqueeze(0)  # [1, L, H]
                batch_list.append(tensor)

            state_tensor = torch.cat(batch_list, 0)  # [B, L, H]
            state_tensor = state_tensor.transpose(1, 0)  # [L, B, H]
            state_list.append(state_tensor)

        return state_list

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
        for state_id in range(len(old_states)):
            if value is None:
                old_states[state_id][:, ids, :] = new_states[state_id][:, ids, :]
            else:
                old_states[state_id][:, ids, :] *= 0.0
                old_states[state_id][:, ids, :] += value

        return old_states

    # Adapter method overrides
    def add_adapter(self, name: str, cfg: DictConfig):
        # Update the config with correct input dim
        cfg = self._update_adapter_cfg_input_dim(cfg)
        # Add the adapter
        super().add_adapter(name=name, cfg=cfg)

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.pred_hidden)
        return cfg


class RNNTJoint(rnnt_abstract.AbstractRNNTJoint, Exportable, AdapterModuleMixin):
    """A Recurrent Neural Network Transducer Joint Network (RNN-T Joint Network).
    An RNN-T Joint network, comprised of a feedforward model.

    Args:
        jointnet: A dict-like object which contains the following key-value pairs.
            encoder_hidden: int specifying the hidden dimension of the encoder net.
            pred_hidden: int specifying the hidden dimension of the prediction net.
            joint_hidden: int specifying the hidden dimension of the joint net
            activation: Activation function used in the joint step. Can be one of
                ['relu', 'tanh', 'sigmoid'].

            Optionally, it may also contain the following:
            dropout: float, set to 0.0 by default. Optional dropout applied at the end of the joint net.

        num_classes: int, specifying the vocabulary size that the joint network must predict,
            excluding the RNNT blank token.

        vocabulary: Optional list of strings/tokens that comprise the vocabulary of the joint network.
            Unused and kept only for easy access for character based encoding RNNT models.

        log_softmax: Optional bool, set to None by default. If set as None, will compute the log_softmax()
            based on the value provided.

        preserve_memory: Optional bool, set to False by default. If the model crashes due to the memory
            intensive joint step, one might try this flag to empty the tensor cache in pytorch.

            Warning: This will make the forward-backward pass much slower than normal.
            It also might not fix the OOM if the GPU simply does not have enough memory to compute the joint.

        fuse_loss_wer: Optional bool, set to False by default.

            Fuses the joint forward, loss forward and
            wer forward steps. In doing so, it trades of speed for memory conservation by creating sub-batches
            of the provided batch of inputs, and performs Joint forward, loss forward and wer forward (optional),
            all on sub-batches, then collates results to be exactly equal to results from the entire batch.

            When this flag is set, prior to calling forward, the fields `loss` and `wer` (either one) *must*
            be set using the `RNNTJoint.set_loss()` or `RNNTJoint.set_wer()` methods.

            Further, when this flag is set, the following argument `fused_batch_size` *must* be provided
            as a non negative integer. This value refers to the size of the sub-batch.

            When the flag is set, the input and output signature of `forward()` of this method changes.
            Input - in addition to `encoder_outputs` (mandatory argument), the following arguments can be provided.
                - decoder_outputs (optional). Required if loss computation is required.
                - encoder_lengths (required)
                - transcripts (optional). Required for wer calculation.
                - transcript_lengths (optional). Required for wer calculation.
                - compute_wer (bool, default false). Whether to compute WER or not for the fused batch.

            Output - instead of the usual `joint` log prob tensor, the following results can be returned.
                - loss (optional). Returned if decoder_outputs, transcripts and transript_lengths are not None.
                - wer_numerator + wer_denominator (optional). Returned if transcripts, transcripts_lengths are provided
                    and compute_wer is set.

        fused_batch_size: Optional int, required if `fuse_loss_wer` flag is set. Determines the size of the
            sub-batches. Should be any value below the actual batch size per GPU.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "encoder_outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "decoder_outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "encoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "transcripts": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "transcript_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "compute_wer": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        if not self._fuse_loss_wer:
            return {
                "outputs": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            }

        else:
            return {
                "loss": NeuralType(elements_type=LossType(), optional=True),
                "wer": NeuralType(elements_type=ElementType(), optional=True),
                "wer_numer": NeuralType(elements_type=ElementType(), optional=True),
                "wer_denom": NeuralType(elements_type=ElementType(), optional=True),
            }

    def _prepare_for_export(self, **kwargs):
        self._fuse_loss_wer = False
        self.log_softmax = False
        super()._prepare_for_export(**kwargs)

    def input_example(self, max_batch=1, max_dim=8192):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        B, T, U = max_batch, max_dim, max_batch
        encoder_outputs = torch.randn(B, self.encoder_hidden, T).to(next(self.parameters()).device)
        decoder_outputs = torch.randn(B, self.pred_hidden, U).to(next(self.parameters()).device)
        return (encoder_outputs, decoder_outputs)

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        return set(["encoder_lengths", "transcripts", "transcript_lengths", "compute_wer"])

    def __init__(
        self,
        jointnet: Dict[str, Any],
        num_classes: int,
        num_extra_outputs: int = 0,
        vocabulary: Optional[List] = None,
        log_softmax: Optional[bool] = None,
        preserve_memory: bool = False,
        fuse_loss_wer: bool = False,
        fused_batch_size: Optional[int] = None,
        experimental_fuse_loss_wer: Any = None,
    ):
        super().__init__()

        self.vocabulary = vocabulary

        self._vocab_size = num_classes
        self._num_extra_outputs = num_extra_outputs
        self._num_classes = num_classes + 1 + num_extra_outputs  # 1 is for blank

        if experimental_fuse_loss_wer is not None:
            # Override fuse_loss_wer from deprecated argument
            fuse_loss_wer = experimental_fuse_loss_wer

        self._fuse_loss_wer = fuse_loss_wer
        self._fused_batch_size = fused_batch_size

        if fuse_loss_wer and (fused_batch_size is None):
            raise ValueError("If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!")

        self._loss = None
        self._wer = None

        # Log softmax should be applied explicitly only for CPU
        self.log_softmax = log_softmax
        self.preserve_memory = preserve_memory

        if preserve_memory:
            logging.warning(
                "`preserve_memory` was set for the Joint Model. Please be aware this will severely impact "
                "the forward-backward step time. It also might not solve OOM issues if the GPU simply "
                "does not have enough memory to compute the joint."
            )

        # Required arguments
        self.encoder_hidden = jointnet['encoder_hidden']
        self.pred_hidden = jointnet['pred_hidden']
        self.joint_hidden = jointnet['joint_hidden']
        self.activation = jointnet['activation']

        # Optional arguments
        dropout = jointnet.get('dropout', 0.0)

        self.pred, self.enc, self.joint_net = self._joint_net_modules(
            num_classes=self._num_classes,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            enc_n_hidden=self.encoder_hidden,
            joint_n_hidden=self.joint_hidden,
            activation=self.activation,
            dropout=dropout,
        )

        # Flag needed for RNNT export support
        self._rnnt_export = False

        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0

    @typecheck()
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: Optional[torch.Tensor],
        encoder_lengths: Optional[torch.Tensor] = None,
        transcripts: Optional[torch.Tensor] = None,
        transcript_lengths: Optional[torch.Tensor] = None,
        compute_wer: bool = False,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # encoder = (B, D, T)
        # decoder = (B, D, U) if passed, else None
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)

        if decoder_outputs is not None:
            decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        if not self._fuse_loss_wer:
            if decoder_outputs is None:
                raise ValueError(
                    "decoder_outputs passed is None, and `fuse_loss_wer` is not set. "
                    "decoder_outputs can only be None for fused step!"
                )

            out = self.joint(encoder_outputs, decoder_outputs)  # [B, T, U, V + 1]
            return out

        else:
            # At least the loss module must be supplied during fused joint
            if self._loss is None or self._wer is None:
                raise ValueError("`fuse_loss_wer` flag is set, but `loss` and `wer` modules were not provided! ")

            # If fused joint step is required, fused batch size is required as well
            if self._fused_batch_size is None:
                raise ValueError("If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!")

            # When using fused joint step, both encoder and transcript lengths must be provided
            if (encoder_lengths is None) or (transcript_lengths is None):
                raise ValueError(
                    "`fuse_loss_wer` is set, therefore encoder and target lengths " "must be provided as well!"
                )

            losses = []
            target_lengths = []
            batch_size = int(encoder_outputs.size(0))  # actual batch size

            # Iterate over batch using fused_batch_size steps
            for batch_idx in range(0, batch_size, self._fused_batch_size):
                begin = batch_idx
                end = min(begin + self._fused_batch_size, batch_size)

                # Extract the sub batch inputs
                # sub_enc = encoder_outputs[begin:end, ...]
                # sub_transcripts = transcripts[begin:end, ...]
                sub_enc = encoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))
                sub_transcripts = transcripts.narrow(dim=0, start=begin, length=int(end - begin))

                sub_enc_lens = encoder_lengths[begin:end]
                sub_transcript_lens = transcript_lengths[begin:end]

                # Sub transcripts does not need the full padding of the entire batch
                # Therefore reduce the decoder time steps to match
                max_sub_enc_length = sub_enc_lens.max()
                max_sub_transcript_length = sub_transcript_lens.max()

                if decoder_outputs is not None:
                    # Reduce encoder length to preserve computation
                    # Encoder: [sub-batch, T, D] -> [sub-batch, T', D]; T' < T
                    if sub_enc.shape[1] != max_sub_enc_length:
                        sub_enc = sub_enc.narrow(dim=1, start=0, length=int(max_sub_enc_length))

                    # sub_dec = decoder_outputs[begin:end, ...]  # [sub-batch, U, D]
                    sub_dec = decoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))  # [sub-batch, U, D]

                    # Reduce decoder length to preserve computation
                    # Decoder: [sub-batch, U, D] -> [sub-batch, U', D]; U' < U
                    if sub_dec.shape[1] != max_sub_transcript_length + 1:
                        sub_dec = sub_dec.narrow(dim=1, start=0, length=int(max_sub_transcript_length + 1))

                    # Perform joint => [sub-batch, T', U', V + 1]
                    sub_joint = self.joint(sub_enc, sub_dec)

                    del sub_dec

                    # Reduce transcript length to correct alignment
                    # Transcript: [sub-batch, L] -> [sub-batch, L']; L' <= L
                    if sub_transcripts.shape[1] != max_sub_transcript_length:
                        sub_transcripts = sub_transcripts.narrow(dim=1, start=0, length=int(max_sub_transcript_length))

                    # Compute sub batch loss
                    # preserve loss reduction type
                    loss_reduction = self.loss.reduction

                    # override loss reduction to sum
                    self.loss.reduction = None

                    # compute and preserve loss
                    loss_batch = self.loss(
                        log_probs=sub_joint,
                        targets=sub_transcripts,
                        input_lengths=sub_enc_lens,
                        target_lengths=sub_transcript_lens,
                    )
                    losses.append(loss_batch)
                    target_lengths.append(sub_transcript_lens)

                    # reset loss reduction type
                    self.loss.reduction = loss_reduction

                else:
                    losses = None

                # Update WER for sub batch
                if compute_wer:
                    sub_enc = sub_enc.transpose(1, 2)  # [B, T, D] -> [B, D, T]
                    sub_enc = sub_enc.detach()
                    sub_transcripts = sub_transcripts.detach()

                    # Update WER on each process without syncing
                    self.wer.update(sub_enc, sub_enc_lens, sub_transcripts, sub_transcript_lens)

                del sub_enc, sub_transcripts, sub_enc_lens, sub_transcript_lens

            # Reduce over sub batches
            if losses is not None:
                losses = self.loss.reduce(losses, target_lengths)

            # Collect sub batch wer results
            if compute_wer:
                # Sync and all_reduce on all processes, compute global WER
                wer, wer_num, wer_denom = self.wer.compute()
                self.wer.reset()
            else:
                wer = None
                wer_num = None
                wer_denom = None

            return losses, wer, wer_num, wer_denom

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
        # f = [B, T, H1]
        f = self.enc(f)
        f.unsqueeze_(dim=2)  # (B, T, 1, H)

        # g = [B, U, H2]
        g = self.pred(g)
        g.unsqueeze_(dim=1)  # (B, 1, U, H)

        inp = f + g  # [B, T, U, H]

        del f, g

        # Forward adapter modules on joint hidden
        if self.is_adapter_available():
            inp = self.forward_enabled_adapters(inp)

        res = self.joint_net(inp)  # [B, T, U, V + 1]

        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()

        # If log_softmax is automatic
        if self.log_softmax is None:
            if not res.is_cuda:  # Use log softmax only if on CPU
                if self.temperature != 1.0:
                    res = (res / self.temperature).log_softmax(dim=-1)
                else:
                    res = res.log_softmax(dim=-1)
        else:
            if self.log_softmax:
                if self.temperature != 1.0:
                    res = (res / self.temperature).log_softmax(dim=-1)
                else:
                    res = res.log_softmax(dim=-1)

        return res

    def _joint_net_modules(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
        """
        Prepare the trainable modules of the Joint Network

        Args:
            num_classes: Number of output classes (vocab size) excluding the RNNT blank token.
            pred_n_hidden: Hidden size of the prediction network.
            enc_n_hidden: Hidden size of the encoder network.
            joint_n_hidden: Hidden size of the joint network.
            activation: Activation of the joint. Can be one of [relu, tanh, sigmoid]
            dropout: Dropout value to apply to joint.
        """
        pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)

        if activation not in ['relu', 'sigmoid', 'tanh']:
            raise ValueError("Unsupported activation for joint step - please pass one of " "[relu, sigmoid, tanh]")

        activation = activation.lower()

        if activation == 'relu':
            activation = torch.nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif activation == 'tanh':
            activation = torch.nn.Tanh()

        layers = (
            [activation]
            + ([torch.nn.Dropout(p=dropout)] if dropout else [])
            + [torch.nn.Linear(joint_n_hidden, num_classes)]
        )
        return pred, enc, torch.nn.Sequential(*layers)

    # Adapter method overrides
    def add_adapter(self, name: str, cfg: DictConfig):
        # Update the config with correct input dim
        cfg = self._update_adapter_cfg_input_dim(cfg)
        # Add the adapter
        super().add_adapter(name=name, cfg=cfg)

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.joint_hidden)
        return cfg

    @property
    def num_classes_with_blank(self):
        return self._num_classes

    @property
    def num_extra_outputs(self):
        return self._num_extra_outputs

    @property
    def loss(self):
        return self._loss

    def set_loss(self, loss):
        if not self._fuse_loss_wer:
            raise ValueError("Attempting to set loss module even though `fuse_loss_wer` is not set!")

        self._loss = loss

    @property
    def wer(self):
        return self._wer

    def set_wer(self, wer):
        if not self._fuse_loss_wer:
            raise ValueError("Attempting to set WER module even though `fuse_loss_wer` is not set!")

        self._wer = wer

    @property
    def fuse_loss_wer(self):
        return self._fuse_loss_wer

    def set_fuse_loss_wer(self, fuse_loss_wer, loss=None, metric=None):
        self._fuse_loss_wer = fuse_loss_wer

        self._loss = loss
        self._wer = metric

    @property
    def fused_batch_size(self):
        return self._fuse_loss_wer

    def set_fused_batch_size(self, fused_batch_size):
        self._fused_batch_size = fused_batch_size


class RNNTDecoderJoint(torch.nn.Module, Exportable):
    """
    Utility class to export Decoder+Joint as a single module
    """

    def __init__(self, decoder, joint):
        super().__init__()
        self.decoder = decoder
        self.joint = joint

    @property
    def input_types(self):
        state_type = NeuralType(('D', 'B', 'D'), ElementType())
        mytypes = {
            'encoder_outputs': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
            'input_states_1': state_type,
            'input_states_2': state_type,
        }

        return mytypes

    def input_example(self, max_batch=1, max_dim=1):
        decoder_example = self.decoder.input_example(max_batch=max_batch, max_dim=max_dim)
        state1, state2 = decoder_example[-1]
        return tuple([self.joint.input_example()[0]]) + decoder_example[:2] + (state1, state2)

    @property
    def output_types(self):
        return {
            "outputs": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "prednet_lengths": NeuralType(tuple('B'), LengthsType()),
            "output_states_1": NeuralType((('D', 'B', 'D')), ElementType()),
            "output_states_2": NeuralType((('D', 'B', 'D')), ElementType()),
        }

    def forward(self, encoder_outputs, targets, target_length, input_states_1, input_states_2):
        decoder_outputs = self.decoder(targets, target_length, (input_states_1, input_states_2))
        decoder_output = decoder_outputs[0]
        decoder_length = decoder_outputs[1]
        input_states_1, input_states_2 = decoder_outputs[2][0], decoder_outputs[2][1]
        joint_output = self.joint(encoder_outputs, decoder_output)
        return (joint_output, decoder_length, input_states_1, input_states_2)


class RNNTDecoderJointSSL(torch.nn.Module):
    def __init__(self, decoder, joint):
        super().__init__()
        self.decoder = decoder
        self.joint = joint

    @property
    def needs_labels(self):
        return True

    @property
    def input_types(self):
        return {
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"log_probs": NeuralType(('B', 'T', 'D'), SpectrogramType())}

    def forward(self, encoder_output, targets, target_lengths):

        decoder, target_length, states = self.decoder(targets=targets, target_length=target_lengths)
        log_probs = self.joint(encoder_outputs=encoder_output, decoder_outputs=decoder)

        return log_probs


class SampledRNNTJoint(RNNTJoint):
    """A Sampled Recurrent Neural Network Transducer Joint Network (RNN-T Joint Network).
    An RNN-T Joint network, comprised of a feedforward model, where the vocab size will be sampled instead
    of computing the full vocabulary joint.

    Args:
        jointnet: A dict-like object which contains the following key-value pairs.
            encoder_hidden: int specifying the hidden dimension of the encoder net.
            pred_hidden: int specifying the hidden dimension of the prediction net.
            joint_hidden: int specifying the hidden dimension of the joint net
            activation: Activation function used in the joint step. Can be one of
                ['relu', 'tanh', 'sigmoid'].

            Optionally, it may also contain the following:
            dropout: float, set to 0.0 by default. Optional dropout applied at the end of the joint net.

        num_classes: int, specifying the vocabulary size that the joint network must predict,
            excluding the RNNT blank token.

        n_samples: int, specifies the number of tokens to sample from the vocabulary space,
            excluding the RNNT blank token. If a given value is larger than the entire vocabulary size,
            then the full vocabulary will be used.

        vocabulary: Optional list of strings/tokens that comprise the vocabulary of the joint network.
            Unused and kept only for easy access for character based encoding RNNT models.

        log_softmax: Optional bool, set to None by default. If set as None, will compute the log_softmax()
            based on the value provided.

        preserve_memory: Optional bool, set to False by default. If the model crashes due to the memory
            intensive joint step, one might try this flag to empty the tensor cache in pytorch.

            Warning: This will make the forward-backward pass much slower than normal.
            It also might not fix the OOM if the GPU simply does not have enough memory to compute the joint.

        fuse_loss_wer: Optional bool, set to False by default.

            Fuses the joint forward, loss forward and
            wer forward steps. In doing so, it trades of speed for memory conservation by creating sub-batches
            of the provided batch of inputs, and performs Joint forward, loss forward and wer forward (optional),
            all on sub-batches, then collates results to be exactly equal to results from the entire batch.

            When this flag is set, prior to calling forward, the fields `loss` and `wer` (either one) *must*
            be set using the `RNNTJoint.set_loss()` or `RNNTJoint.set_wer()` methods.

            Further, when this flag is set, the following argument `fused_batch_size` *must* be provided
            as a non negative integer. This value refers to the size of the sub-batch.

            When the flag is set, the input and output signature of `forward()` of this method changes.
            Input - in addition to `encoder_outputs` (mandatory argument), the following arguments can be provided.
                - decoder_outputs (optional). Required if loss computation is required.
                - encoder_lengths (required)
                - transcripts (optional). Required for wer calculation.
                - transcript_lengths (optional). Required for wer calculation.
                - compute_wer (bool, default false). Whether to compute WER or not for the fused batch.

            Output - instead of the usual `joint` log prob tensor, the following results can be returned.
                - loss (optional). Returned if decoder_outputs, transcripts and transript_lengths are not None.
                - wer_numerator + wer_denominator (optional). Returned if transcripts, transcripts_lengths are provided
                    and compute_wer is set.

        fused_batch_size: Optional int, required if `fuse_loss_wer` flag is set. Determines the size of the
            sub-batches. Should be any value below the actual batch size per GPU.
    """

    def __init__(
        self,
        jointnet: Dict[str, Any],
        num_classes: int,
        n_samples: int,
        vocabulary: Optional[List] = None,
        log_softmax: Optional[bool] = None,
        preserve_memory: bool = False,
        fuse_loss_wer: bool = False,
        fused_batch_size: Optional[int] = None,
    ):
        super().__init__(
            jointnet=jointnet,
            num_classes=num_classes,
            vocabulary=vocabulary,
            log_softmax=log_softmax,
            preserve_memory=preserve_memory,
            fuse_loss_wer=fuse_loss_wer,
            fused_batch_size=fused_batch_size,
        )
        self.n_samples = n_samples
        self.register_buffer('blank_id', torch.tensor([self.num_classes_with_blank - 1]), persistent=False)

    @typecheck()
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: Optional[torch.Tensor],
        encoder_lengths: Optional[torch.Tensor] = None,
        transcripts: Optional[torch.Tensor] = None,
        transcript_lengths: Optional[torch.Tensor] = None,
        compute_wer: bool = False,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # If in inference mode, revert to basic RNNT Joint behaviour.
        # Sampled RNNT is only used for training.
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            # Simply call full tensor joint
            return super().forward(
                encoder_outputs=encoder_outputs,
                decoder_outputs=decoder_outputs,
                encoder_lengths=encoder_lengths,
                transcripts=transcripts,
                transcript_lengths=transcript_lengths,
                compute_wer=compute_wer,
            )

        if transcripts is None or transcript_lengths is None:
            logging.warning(
                "Sampled RNNT Joint currently only works with `fuse_loss_wer` set to True, "
                "and when `fused_batch_size` is a positive integer."
            )
            raise ValueError(
                "Sampled RNNT loss only works when the transcripts are provided during training."
                "Please ensure that you correctly pass the `transcripts` and `transcript_lengths`."
            )

        # encoder = (B, D, T)
        # decoder = (B, D, U) if passed, else None
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)

        if decoder_outputs is not None:
            decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        # At least the loss module must be supplied during fused joint
        if self._loss is None or self._wer is None:
            raise ValueError("`fuse_loss_wer` flag is set, but `loss` and `wer` modules were not provided! ")

        # If fused joint step is required, fused batch size is required as well
        if self._fused_batch_size is None:
            raise ValueError("If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!")

        # When using fused joint step, both encoder and transcript lengths must be provided
        if (encoder_lengths is None) or (transcript_lengths is None):
            raise ValueError(
                "`fuse_loss_wer` is set, therefore encoder and target lengths " "must be provided as well!"
            )

        losses = []
        target_lengths = []
        batch_size = int(encoder_outputs.size(0))  # actual batch size

        # Iterate over batch using fused_batch_size steps
        for batch_idx in range(0, batch_size, self._fused_batch_size):
            begin = batch_idx
            end = min(begin + self._fused_batch_size, batch_size)

            # Extract the sub batch inputs
            # sub_enc = encoder_outputs[begin:end, ...]
            # sub_transcripts = transcripts[begin:end, ...]
            sub_enc = encoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))
            sub_transcripts = transcripts.narrow(dim=0, start=begin, length=int(end - begin))

            sub_enc_lens = encoder_lengths[begin:end]
            sub_transcript_lens = transcript_lengths[begin:end]

            # Sub transcripts does not need the full padding of the entire batch
            # Therefore reduce the decoder time steps to match
            max_sub_enc_length = sub_enc_lens.max()
            max_sub_transcript_length = sub_transcript_lens.max()

            if decoder_outputs is not None:
                # Reduce encoder length to preserve computation
                # Encoder: [sub-batch, T, D] -> [sub-batch, T', D]; T' < T
                if sub_enc.shape[1] != max_sub_enc_length:
                    sub_enc = sub_enc.narrow(dim=1, start=0, length=int(max_sub_enc_length))

                # sub_dec = decoder_outputs[begin:end, ...]  # [sub-batch, U, D]
                sub_dec = decoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))  # [sub-batch, U, D]

                # Reduce decoder length to preserve computation
                # Decoder: [sub-batch, U, D] -> [sub-batch, U', D]; U' < U
                if sub_dec.shape[1] != max_sub_transcript_length + 1:
                    sub_dec = sub_dec.narrow(dim=1, start=0, length=int(max_sub_transcript_length + 1))

                # Reduce transcript length to correct alignment
                # Transcript: [sub-batch, L] -> [sub-batch, L']; L' <= L
                if sub_transcripts.shape[1] != max_sub_transcript_length:
                    sub_transcripts = sub_transcripts.narrow(dim=1, start=0, length=int(max_sub_transcript_length))

                # Perform sampled joint => [sub-batch, T', U', {V' < V} + 1}]
                sub_joint, sub_transcripts_remapped = self.sampled_joint(
                    sub_enc, sub_dec, transcript=sub_transcripts, transcript_lengths=sub_transcript_lens
                )

                del sub_dec

                # Compute sub batch loss
                # preserve loss reduction type
                loss_reduction = self.loss.reduction

                # override loss reduction to sum
                self.loss.reduction = None

                # override blank idx in order to map to new vocabulary space
                # in the new vocabulary space, we set the mapping of the RNNT Blank from index V+1 to 0
                # So the loss here needs to be updated accordingly.
                # TODO: See if we can have some formal API for rnnt loss to update inner blank index.
                cached_blank_id = self.loss._loss.blank
                self.loss._loss.blank = 0

                # compute and preserve loss
                loss_batch = self.loss(
                    log_probs=sub_joint,
                    targets=sub_transcripts_remapped,  # Note: We have to use remapped transcripts here !
                    input_lengths=sub_enc_lens,
                    target_lengths=sub_transcript_lens,  # Note: Even after remap, the transcript lengths remain intact.
                )
                losses.append(loss_batch)
                target_lengths.append(sub_transcript_lens)

                # reset loss reduction type and blank id
                self.loss.reduction = loss_reduction
                self.loss._loss.blank = cached_blank_id

            else:
                losses = None

            # Update WER for sub batch
            if compute_wer:
                sub_enc = sub_enc.transpose(1, 2)  # [B, T, D] -> [B, D, T]
                sub_enc = sub_enc.detach()
                sub_transcripts = sub_transcripts.detach()

                # Update WER on each process without syncing
                self.wer.update(sub_enc, sub_enc_lens, sub_transcripts, sub_transcript_lens)

            del sub_enc, sub_transcripts, sub_enc_lens, sub_transcript_lens

        # Reduce over sub batches
        if losses is not None:
            losses = self.loss.reduce(losses, target_lengths)

        # Collect sub batch wer results
        if compute_wer:
            # Sync and all_reduce on all processes, compute global WER
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()
        else:
            wer = None
            wer_num = None
            wer_denom = None

        return losses, wer, wer_num, wer_denom

    def sampled_joint(
        self, f: torch.Tensor, g: torch.Tensor, transcript: torch.Tensor, transcript_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the sampled joint step of the network.

        # Reference
        - [Memory-Efficient Training of RNN-Transducer with Sampled Softmax](https://arxiv.org/abs/2203.16868)

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the RNNT blank token).
        S = Sample size of vocabulary.

        NOTE:
            The implementation of this joint model is slightly modified from the original paper.
            The original paper proposes the following steps :
            (enc, dec) -> Expand + Concat + Sum [B, T, U, H1+H2] -> Forward through joint hidden [B, T, U, H] -- *1
            *1 -> Forward through joint final [B, T, U, V + 1].

            We instead split the joint hidden into joint_hidden_enc and joint_hidden_dec and act as follows:
            enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- *1
            dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- *2
            (*1, *2) -> Sum [B, T, U, H] -> Sample Vocab V_Pos (for target tokens) and V_Neg ->
            (V_Neg is sampled not uniformly by as a rand permutation of all vocab tokens, then eliminate
            all Intersection(V_Pos, V_Neg) common tokens to avoid duplication of loss) ->
            Concat new Vocab V_Sampled = Union(V_Pos, V_Neg)
            -> Forward partially through the joint final to create [B, T, U, V_Sampled]

        Args:
            f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
            g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]
            transcript: Batch of transcripts. A torch.Tensor of shape [B, U]
            transcript_lengths: Batch of lengths of the transcripts. A torch.Tensor of shape [B]

        Returns:
            Logits / log softmaxed tensor of shape (B, T, U, V + 1).
        """
        # If under inference mode, ignore sampled joint and compute full joint.
        if self.training is False or torch.is_grad_enabled() is False or torch.is_inference_mode_enabled():
            # Simply call full tensor joint
            return super().joint(f=f, g=g)

        # Compute sampled softmax
        # f = [B, T, H1]
        f = self.enc(f)
        f.unsqueeze_(dim=2)  # (B, T, 1, H)

        # g = [B, U, H2]
        g = self.pred(g)
        g.unsqueeze_(dim=1)  # (B, 1, U, H)

        inp = f + g  # [B, T, U, H]

        del f, g

        # Forward adapter modules on joint hidden
        if self.is_adapter_available():
            inp = self.forward_enabled_adapters(inp)

        # Do partial forward of joint net (skipping the final linear)
        for module in self.joint_net[:-1]:
            inp = module(inp)  # [B, T, U, H]

        # Begin compute of sampled RNNT joint
        with torch.no_grad():
            # gather true labels
            transcript_vocab_ids = torch.unique(transcript)

            # augment with blank token id
            transcript_vocab_ids = torch.cat([self.blank_id, transcript_vocab_ids])

            # Remap the transcript label ids to new positions of label ids (in the transcript_vocab_ids)
            # This is necessary cause the RNNT loss doesnt care about the value, only the position of the ids
            # of the transcript tokens. We can skip this step for noise samples cause those are only used for softmax
            # estimation, not for computing actual label.
            # From `https://stackoverflow.com/a/68969697` - bucketize algo.
            t_ids = torch.arange(transcript_vocab_ids.size(0), device='cpu')
            mapping = {k: v for k, v in zip(transcript_vocab_ids.to('cpu'), t_ids)}

            # From `https://stackoverflow.com/questions/13572448`.
            palette, key = zip(*mapping.items())

            t_device = transcript.device
            key = torch.tensor(key, device=t_device)
            palette = torch.tensor(palette, device=t_device)

            # This step maps old token id to new token id in broadcasted manner.
            # For example, if original transcript tokens were [2, 1, 4, 5, 4, 1]
            # But after computing the unique token set of above we get
            # transcript_vocab_ids = [1, 2, 4, 5]  # note: pytorch returns sorted unique values thankfully
            # Then we get the index map of the new vocab ids as:
            # {0: 1, 1: 2, 2: 4, 3: 5}
            # Now we need to map the original transcript tokens to new vocab id space
            # So we construct the inverted map as follow :
            # {1: 0, 2: 1, 4: 2, 5: 3}
            # Then remap the original transcript tokens to new token ids
            # new_transcript = [1, 0, 2, 3, 2, 0]
            index = torch.bucketize(transcript.ravel(), palette)
            transcript = key[index].reshape(transcript.shape)
            transcript = transcript.to(t_device)

        # Extract out partial weight tensor and bias tensor of just the V_Pos vocabulary from the full joint.
        true_weights = self.joint_net[-1].weight[transcript_vocab_ids, :]
        true_bias = self.joint_net[-1].bias[transcript_vocab_ids]

        # Compute the transcript joint scores (only of vocab V_Pos)
        transcript_scores = torch.matmul(inp, true_weights.transpose(0, 1)) + true_bias

        # Construct acceptance criteria in vocab space, reject all tokens in Intersection(V_Pos, V_Neg)
        with torch.no_grad():
            # Instead of uniform sample, first we create arange V (ignoring blank), then randomly shuffle
            # this range of ids, then subset `n_samples` amount of vocab tokens out of the permuted tensor.
            # This is good because it guarentees that no token will ever be repeated in V_Neg;
            # which dramatically complicates loss calculation.
            # Further more, with this strategy, given a `n_samples` > V + 1; we are guarenteed to get the
            # V_Samples = V (i.e., full vocabulary will be used in such a case).
            # Useful to debug cases where you expect sampled vocab to get exact same training curve as
            # full vocab.
            sample_ids = torch.randperm(n=self.num_classes_with_blank - 1, device=transcript_scores.device)[
                : self.n_samples
            ]

            # We need to compute the intersection(V_Pos, V_Neg), then eliminate the intersection arguments
            # from inside V_Neg.

            # First, compute the pairwise commonality to find index inside `sample_ids` which match the token id
            # inside transcript_vocab_ids.
            # Note: It is important to ignore the hardcoded RNNT Blank token injected at id 0 of the transcript
            # vocab ids, otherwise the blank may occur twice, once for RNNT blank and once as negative sample,
            # doubling the gradient of the RNNT blank token.
            reject_samples = torch.where(transcript_vocab_ids[1:, None] == sample_ids[None, :])

            # Let accept samples be a set of ids which is a subset of sample_ids
            # such that intersection(V_Pos, accept_samples) is a null set.
            accept_samples = sample_ids.clone()

            # In order to construct such an accept_samples tensor, first we construct a bool map
            # and fill all the indices where there is a match inside of sample_ids.
            # reject_samples is a tuple (transcript_vocab_position, sample_position) which gives a
            # many to many map between N values of transript and M values of sample_ids.
            # We dont care about transcript side matches, only the ids inside of sample_ids that matched.
            sample_mask = torch.ones_like(accept_samples, dtype=torch.bool)
            sample_mask[reject_samples[1]] = False

            # Finally, compute the subset of tokens by selecting only those sample_ids which had no matches
            accept_samples = accept_samples[sample_mask]

        # Extract out partial weight tensor and bias tensor of just the V_Neg vocabulary from the full joint.
        sample_weights = self.joint_net[-1].weight[accept_samples, :]
        sample_bias = self.joint_net[-1].bias[accept_samples]

        # Compute the noise joint scores (only of vocab V_Neg) to be used for softmax
        # The quality of this sample determines the quality of the softmax gradient.
        # We use naive algo broadcasted over batch, but it is more efficient than sample level computation.
        # One can increase `n_samples` for better estimation of rejection samples and its gradient.
        noise_scores = torch.matmul(inp, sample_weights.transpose(0, 1)) + sample_bias

        # Finally, construct the sampled joint as the V_Sampled = Union(V_Pos, V_Neg)
        # Here, we simply concatenate the two tensors to construct the joint with V_Sampled vocab
        # because before we have properly asserted that Intersection(V_Pos, V_Neg) is a null set.
        res = torch.cat([transcript_scores, noise_scores], dim=-1)

        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()

        # If log_softmax is automatic
        if self.log_softmax is None:
            if not res.is_cuda:  # Use log softmax only if on CPU
                res = res.log_softmax(dim=-1)
        else:
            if self.log_softmax:
                res = res.log_softmax(dim=-1)

        return res, transcript


# Add the adapter compatible modules to the registry
for cls in [RNNTDecoder, RNNTJoint, SampledRNNTJoint]:
    if adapter_mixins.get_registered_adapter(cls) is None:
        adapter_mixins.register_adapter(cls, cls)  # base class is adapter compatible itself
