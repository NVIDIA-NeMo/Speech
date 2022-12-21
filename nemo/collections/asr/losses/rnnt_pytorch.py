# ! /usr/bin/python
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

import torch

from nemo.core.classes import Loss
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType


class RNNTLossPytorch(Loss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, blank, reduction):
        super().__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(self, acts, labels, act_lens, label_lens):
        acts = torch.log_softmax(acts, -1)
        forward_logprob = self.compute_forward_prob(acts, labels, act_lens, label_lens)
        losses = -forward_logprob
        if self.reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.reduction == 'mean':
            losses = torch.div(losses, label_lens).mean()
        elif self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean_volume':
            losses = losses.sum() / label_lens.sum()  # same as above but longer samples weigh more

        return losses

    def compute_forward_prob(self, acts, labels, act_lens, label_lens):
        B, T, U, _ = acts.shape

        log_alpha = torch.zeros(B, T, U)
        log_alpha = log_alpha.to(acts.device)

        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        # this is the base case: (t=0, u=0) with log-alpha = 0.
                        log_alpha[:, t, u] = 0.0
                    else:
                        # this is case for (t = 0, u > 0), reached by (t, u - 1)
                        # emitting a blank symbol.
                        log_alpha[:, t, u] = log_alpha[:, t - 1, u] + acts[:, t - 1, 0, self.blank]
                else:
                    if t == 0:
                        # in case of (u > 0, t = 0), this is only reached from
                        # (t, u - 1) with a label emission.
                        gathered = torch.gather(
                            acts[:, t, u - 1], dim=1, index=labels[:, u - 1].view(-1, 1).type(torch.int64)
                        ).reshape(-1)
                        log_alpha[:, t, u] = log_alpha[:, t, u - 1] + gathered.to(log_alpha.device)
                    else:
                        # here both t and u are > 0, this state is reachable
                        # with two possibilities: (t - 1, u) with a blank emission
                        # or (t, u - 1) with a label emission.
                        log_alpha[:, t, u] = torch.logsumexp(
                            torch.stack(
                                [
                                    log_alpha[:, t - 1, u] + acts[:, t - 1, u, self.blank],
                                    log_alpha[:, t, u - 1]
                                    + torch.gather(
                                        acts[:, t, u - 1], dim=1, index=labels[:, u - 1].view(-1, 1).type(torch.int64)
                                    ).reshape(-1),
                                ]
                            ),
                            dim=0,
                        )

        log_probs = []
        for b in range(B):
            # here we need to add the final blank emission weights.
            to_append = (
                log_alpha[b, act_lens[b] - 1, label_lens[b]] + acts[b, act_lens[b] - 1, label_lens[b], self.blank]
            )
            log_probs.append(to_append)
        log_prob = torch.stack(log_probs)

        return log_prob


class MultiblankRNNTLossPytorch(Loss):
    """
    Pure Python implementation of multi-blank transducer loss (https://arxiv.org/pdf/2211.03541.pdf)
    """

    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, blank, big_blank_durations, reduction, sigma):
        super().__init__()
        self.blank = blank
        self.big_blank_durations = big_blank_durations
        self.reduction = reduction
        self.sigma = sigma

    def forward(self, acts, labels, act_lens, label_lens):
        acts = torch.log_softmax(acts, -1) - self.sigma
        forward_logprob = self.compute_forward_prob(acts, labels, act_lens, label_lens)

        losses = -forward_logprob
        if self.reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.reduction == 'mean':
            losses = torch.div(losses, label_lens).mean()
        elif self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean_volume':
            losses = losses.sum() / label_lens.sum()  # same as above but longer samples weigh more

        return losses

    def compute_forward_prob(self, acts, labels, act_lens, label_lens):
        B, T, U, _ = acts.shape

        log_alpha = torch.zeros(B, T, U, device=acts.device)
        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        # this is the base case: (t=0, u=0) with log-alpha = 0.
                        log_alpha[:, t, u] = 0.0
                    else:
                        # this is case for (t = 0, u > 0), reached by (t, u - d)
                        # emitting a blank symbol of duration d.
                        log_alpha[:, t, u] = log_alpha[:, t - 1, u] + acts[:, t - 1, 0, self.blank]
                        for i, d in enumerate(self.big_blank_durations):
                            if t >= d:
                                tt = log_alpha[:, t - d, u] + acts[:, t - d, 0, self.blank - 1 - i]
                                log_alpha[:, t, u] = torch.logsumexp(
                                    torch.stack([1.0 * log_alpha[:, t, u], tt]), dim=0
                                )

                else:
                    if t == 0:
                        # in case of (u > 0, t = 0), this is only reached from
                        # (t, u - 1) with a label emission.
                        gathered = torch.gather(
                            acts[:, t, u - 1], dim=1, index=labels[:, u - 1].view(-1, 1).type(torch.int64)
                        ).reshape(-1)
                        log_alpha[:, t, u] = log_alpha[:, t, u - 1] + gathered
                    else:
                        # here both t and u are > 0, this state is reachable
                        # with two possibilities: (t - d, u) with emission of
                        # blank with duration d, or (t, u - 1) with a label emission.

                        # first we take care of the standard blank.
                        log_alpha[:, t, u] = torch.logsumexp(
                            torch.stack(
                                [
                                    log_alpha[:, t - 1, u] + acts[:, t - 1, u, self.blank],
                                    log_alpha[:, t, u - 1]
                                    + torch.gather(
                                        acts[:, t, u - 1], dim=1, index=labels[:, u - 1].view(-1, 1).type(torch.int64)
                                    ).reshape(-1),
                                ]
                            ),
                            dim=0,
                        )

                        # now we go over all big blanks. They need to be considered if current t >= blank duration d.
                        for i, d in enumerate(self.big_blank_durations):
                            if t >= d:
                                tt = log_alpha[:, t - d, u] + acts[:, t - d, u, self.blank - 1 - i]
                                log_alpha[:, t, u] = torch.logsumexp(
                                    torch.stack([1.0 * log_alpha[:, t, u], tt]), dim=0
                                )

        log_probs = []
        for b in range(B):
            # here we need to add the final blank emission weights, which needs
            # to consider all possible blank durations.
            to_append = (
                log_alpha[b, act_lens[b] - 1, label_lens[b]] + acts[b, act_lens[b] - 1, label_lens[b], self.blank]
            )

            for i, d in enumerate(self.big_blank_durations):
                if act_lens[b] >= d:
                    tt = (
                        log_alpha[b, act_lens[b] - d, label_lens[b]]
                        + acts[b, act_lens[b] - d, label_lens[b], self.blank - 1 - i]
                    )
                    to_append = torch.logsumexp(torch.stack([1.0 * to_append, tt]), dim=0)

            log_probs.append(to_append)
        log_prob = torch.stack(log_probs)

        return log_prob
