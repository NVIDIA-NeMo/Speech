# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn.functional as F

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LengthsType, LogprobsType, LossType, ProbsType
from nemo.core.neural_types.neural_type import NeuralType


class ForwardSumLoss(Loss):
    def __init__(self, blank_logprob=-1, loss_scale=1.0):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob
        self.loss_scale = loss_scale

    @property
    def input_types(self):
        return {
            "attn_logprob": NeuralType(('B', 'S', 'T_spec', 'T_text'), LogprobsType()),
            "in_lens": NeuralType(tuple('B'), LengthsType()),
            "out_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "forward_sum_loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        max_key_len = attn_logprob.size(-1)

        # Reorder input to [query_len, batch_size, key_len]
        attn_logprob = attn_logprob.squeeze(1)
        attn_logprob = attn_logprob.permute(1, 0, 2)

        # Add blank label
        attn_logprob = F.pad(input=attn_logprob, pad=(1, 0, 0, 0, 0, 0), value=self.blank_logprob)

        # Convert to log probabilities
        # Note: Mask out probs beyond key_len
        key_inds = torch.arange(max_key_len + 1, device=attn_logprob.device, dtype=torch.long)
        attn_logprob.masked_fill_(key_inds.view(1, 1, -1) > key_lens.view(1, -1, 1), -1e15)  # key_inds >= key_lens+1
        attn_logprob = self.log_softmax(attn_logprob)

        # Target sequences
        target_seqs = key_inds[1:].unsqueeze(0)
        target_seqs = target_seqs.repeat(key_lens.numel(), 1)

        # Evaluate CTC loss
        cost = self.ctc_loss(attn_logprob, target_seqs, input_lengths=query_lens, target_lengths=key_lens)
        cost *= self.loss_scale

        return cost


class BinLoss(Loss):
    def __init__(self, loss_scale=1.0):
        super().__init__()
        self.loss_scale = loss_scale

    @property
    def input_types(self):
        return {
            "hard_attention": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
            "soft_attention": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
        }

    @property
    def output_types(self):
        return {
            "bin_loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        loss = -log_sum / hard_attention.sum()
        loss *= self.loss_scale
        return loss
