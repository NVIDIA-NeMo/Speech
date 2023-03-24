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

from nemo.collections.asr.modules import rnnt
from nemo.collections.asr.parts.utils.rnnt_utils import HATJointOutput

from nemo.utils import logging


class HATJoint(rnnt.RNNTJoint):
    """A Hybrid Autoregressive Transducer Joint Network (HAT Joint Network).
    A HAT Joint network, comprised of a feedforward model.

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
            excluding the HAT blank token.

        vocabulary: Optional list of strings/tokens that comprise the vocabulary of the joint network.
            Unused and kept only for easy access for character based encoding HAT models.

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
            be set using the `HATJoint.set_loss()` or `HATJoint.set_wer()` methods.

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
        num_extra_outputs: int = 0,
        vocabulary: Optional[List] = None,
        log_softmax: Optional[bool] = None,
        preserve_memory: bool = False,
        fuse_loss_wer: bool = False,
        fused_batch_size: Optional[int] = None,
        experimental_fuse_loss_wer: Any = None,
    ):
        super().__init__(
            jointnet=jointnet,
            num_classes=num_classes,
            num_extra_outputs=num_extra_outputs,
            vocabulary=vocabulary,
            log_softmax=log_softmax,
            preserve_memory=preserve_memory,
            fuse_loss_wer=fuse_loss_wer,
            fused_batch_size=fused_batch_size,
            experimental_fuse_loss_wer=experimental_fuse_loss_wer,
        )

        self.pred, self.enc, self.joint_net, self.blank_pred = self._joint_hat_net_modules(
            num_classes=self._vocab_size,  # non blank symbol
            pred_n_hidden=self.pred_hidden,
            enc_n_hidden=self.encoder_hidden,
            joint_n_hidden=self.joint_hidden,
            activation=self.activation,
            dropout=jointnet.get('dropout', 0.0),
        )
        self._return_hat_ilm = False

    @property
    def return_hat_ilm(self):
        return self._return_hat_ilm

    @return_hat_ilm.setter
    def return_hat_ilm(self, hat_subtract_ilm):
        self._return_hat_ilm = hat_subtract_ilm

    def joint(self, f: torch.Tensor, g: torch.Tensor) -> Union[torch.Tensor, HATJointOutput]:
        """
        Compute the joint step of the network.

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the HAT blank token).

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
            Log softmaxed tensor of shape (B, T, U, V + 1).
            Internal LM probability (B, 1, U, V) -- in case of return_ilm==True.
        """
        # f = [B, T, H1]
        f = self.enc(f)
        f.unsqueeze_(dim=2)  # (B, T, 1, H)

        # g = [B, U, H2]
        g = self.pred(g)
        g.unsqueeze_(dim=1)  # (B, 1, U, H)

        inp = f + g  # [B, T, U, H]

        del f

        # Forward adapter modules on joint hidden
        if self.is_adapter_available():
            inp = self.forward_enabled_adapters(inp)

        blank_logprob = self.blank_pred(inp)  # [B, T, U, 1]
        label_logit = self.joint_net(inp)  # [B, T, U, V]

        del inp

        label_logprob = label_logit.log_softmax(dim=-1)
        scale_prob = torch.clamp(1 - torch.exp(blank_logprob), min=1e-6)
        label_logprob_scaled = torch.log(scale_prob) + label_logprob  # [B, T, U, V]

        res = torch.cat((label_logprob_scaled, blank_logprob), dim=-1).contiguous()  # [B, T, U, V+1]

        if self.return_hat_ilm:
            ilm_logprobs = self.joint_net(g).log_softmax(dim=-1)  # [B, 1, U, V]
            res = HATJointOutput(hat_logprobs=res, ilm_logprobs=ilm_logprobs)

        del g, blank_logprob, label_logprob, label_logit, scale_prob, label_logprob_scaled

        if self.preserve_memory:
            torch.cuda.empty_cache()

        return res

    def _joint_hat_net_modules(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
        """
        Prepare the trainable modules of the Joint Network

        Args:
            num_classes: Number of output classes (vocab size) excluding the HAT blank token.
            pred_n_hidden: Hidden size of the prediction network.
            enc_n_hidden: Hidden size of the encoder network.
            joint_n_hidden: Hidden size of the joint network.
            activation: Activation of the joint. Can be one of [relu, tanh, sigmoid]
            dropout: Dropout value to apply to joint.
        """
        pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)
        blank_pred = torch.nn.Sequential(
            torch.nn.Tanh(), torch.nn.Dropout(p=dropout), torch.nn.Linear(joint_n_hidden, 1), torch.nn.LogSigmoid()
        )

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
        return pred, enc, torch.nn.Sequential(*layers), blank_pred
