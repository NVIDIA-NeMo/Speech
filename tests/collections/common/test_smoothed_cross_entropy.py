# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest
import torch

from nemo.collections.common.losses import SmoothedCrossEntropyLoss

SMOOTHING = 0.1


def smoothed_targets(labels: torch.Tensor, num_labels: int, smoothing: float) -> torch.Tensor:
    """The label-smoothed target distribution: ``1 - smoothing`` on the label, the rest spread evenly."""
    targets = torch.full(labels.shape + (num_labels,), smoothing / (num_labels - 1))
    return targets.scatter(-1, labels.unsqueeze(-1), 1.0 - smoothing)


class TestSmoothedCrossEntropyLossGradientOnly:
    @pytest.mark.unit
    @pytest.mark.parametrize("batch,time,num_labels", [(2, 4, 7), (3, 17, 1026)])
    def test_reports_the_plain_loss_with_the_smoothed_gradient(self, batch, time, num_labels):
        """The second case has the decoder shape of a 1,024-piece vocabulary with boundary labels."""
        torch.manual_seed(1)
        logits = torch.randn(batch, time, num_labels)
        labels = torch.randint(num_labels, (batch, time))
        lengths = torch.tensor([time, time - 1, time // 2][:batch])  # unequal sequence lengths
        mask = (torch.arange(time)[None, :] < lengths[:, None]).long()

        smoothed_loss = SmoothedCrossEntropyLoss(label_smoothing=SMOOTHING)
        gradient_only = SmoothedCrossEntropyLoss(label_smoothing=SMOOTHING, label_smoothing_gradient_only=True)
        plain_loss = SmoothedCrossEntropyLoss(label_smoothing=0.0)

        reference = logits.clone().requires_grad_()
        smoothed_loss(log_probs=reference.log_softmax(-1), labels=labels, output_mask=mask).backward()
        candidate = logits.clone().requires_grad_()
        value = gradient_only(log_probs=candidate.log_softmax(-1), labels=labels, output_mask=mask)
        value.backward()
        plain = plain_loss(log_probs=logits.log_softmax(-1), labels=labels, output_mask=mask)

        torch.testing.assert_close(value, plain)
        torch.testing.assert_close(candidate.grad, reference.grad)

    @pytest.mark.unit
    def test_per_token_losses_keep_the_smoothed_gradient(self):
        num_labels = 7
        torch.manual_seed(2)
        logits = torch.randn(3, 2, num_labels)
        labels = torch.randint(num_labels, (3, 2))
        mask = torch.ones(3, 2)
        loss = SmoothedCrossEntropyLoss(
            label_smoothing=SMOOTHING, label_smoothing_gradient_only=True, per_token_reduction=False
        )
        candidate = logits.clone().requires_grad_()
        per_token = loss(log_probs=candidate.log_softmax(-1), labels=labels, output_mask=mask)
        assert per_token.shape == (3, 2)
        per_token.sum().backward()
        reference = logits.clone().requires_grad_()
        targets = smoothed_targets(labels, num_labels, SMOOTHING)
        (-(targets * reference.log_softmax(-1)).sum(-1)).sum().backward()
        torch.testing.assert_close(candidate.grad, reference.grad)
