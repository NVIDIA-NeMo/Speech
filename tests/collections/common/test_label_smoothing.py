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

from nemo.collections.common.losses import label_smoothed_log_prob_gradient

NUM_LABELS = 7
SMOOTHING = 0.1


def smoothed_targets(labels: torch.Tensor, num_labels: int, smoothing: float) -> torch.Tensor:
    """The label-smoothed target distribution: ``1 - smoothing`` on the label, the rest spread evenly."""
    targets = torch.full(labels.shape + (num_labels,), smoothing / (num_labels - 1))
    return targets.scatter(-1, labels.unsqueeze(-1), 1.0 - smoothing)


class TestLabelSmoothedLogProbGradient:
    @pytest.mark.unit
    def test_forward_is_the_identity_and_zero_smoothing_is_a_no_op(self):
        log_probs = torch.randn(3, NUM_LABELS).log_softmax(-1).requires_grad_()
        assert label_smoothed_log_prob_gradient(log_probs, 0.0) is log_probs
        assert torch.equal(label_smoothed_log_prob_gradient(log_probs, SMOOTHING), log_probs)

    @pytest.mark.unit
    @pytest.mark.parametrize("batch,time,num_labels", [(2, 5, NUM_LABELS), (3, 17, 1026)])
    def test_gradient_equals_the_smoothed_cross_entropy_gradient(self, batch, time, num_labels):
        """The decoder shape of a 1,024-piece vocabulary with boundary labels is the second case."""
        torch.manual_seed(0)
        logits = torch.randn(batch, time, num_labels)
        labels = torch.randint(num_labels, (batch, time))
        weights = torch.rand(batch, time)  # per-position loss scales, as a normalization would apply
        weights[:, time // 2 :] *= torch.arange(batch)[:, None] > 0  # one sequence shorter than the others

        plain = logits.clone().requires_grad_()
        log_probs = label_smoothed_log_prob_gradient(plain.log_softmax(-1), SMOOTHING)
        nll = -(log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * weights).sum()
        nll.backward()

        smoothed = logits.clone().requires_grad_()
        targets = smoothed_targets(labels, num_labels, SMOOTHING)
        ce = -((targets * smoothed.log_softmax(-1)).sum(-1) * weights).sum()
        ce.backward()

        torch.testing.assert_close(plain.grad, smoothed.grad)
        assert nll.item() != pytest.approx(ce.item())

    @pytest.mark.unit
    def test_gradient_on_log_probs_is_the_smoothed_target_distribution(self):
        log_probs = torch.zeros(2, NUM_LABELS, requires_grad=True)
        labels = torch.tensor([1, 4])
        out = label_smoothed_log_prob_gradient(log_probs, SMOOTHING)
        (-out.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * torch.tensor([1.0, 0.5])).sum().backward()
        expected = -smoothed_targets(labels, NUM_LABELS, SMOOTHING) * torch.tensor([[1.0], [0.5]])
        torch.testing.assert_close(log_probs.grad, expected)

    @pytest.mark.unit
    def test_masked_positions_keep_a_zero_gradient(self):
        log_probs = torch.zeros(3, NUM_LABELS, requires_grad=True)
        out = label_smoothed_log_prob_gradient(log_probs, SMOOTHING)
        mask = torch.tensor([1.0, 0.0, 1.0])
        (-out[:, 2] * mask).sum().backward()
        assert torch.equal(log_probs.grad[1], torch.zeros(NUM_LABELS))
        torch.testing.assert_close(log_probs.grad[0].sum(), torch.tensor(-1.0))

    @pytest.mark.unit
    def test_excluded_labels_receive_no_smoothing_mass(self):
        blank = NUM_LABELS - 1
        log_probs = torch.zeros(2, NUM_LABELS, requires_grad=True)
        out = label_smoothed_log_prob_gradient(log_probs, SMOOTHING, exclude_labels=[blank])
        labels = torch.tensor([2, blank])
        (-out.gather(-1, labels.unsqueeze(-1))).sum().backward()
        grad = log_probs.grad
        # a regular target: 1 - smoothing on the label, the mass spread over the other regular labels, none on blank
        torch.testing.assert_close(grad[0, 2], torch.tensor(-(1.0 - SMOOTHING)))
        assert grad[0, blank] == 0.0
        torch.testing.assert_close(grad[0, 0], torch.tensor(-SMOOTHING / (NUM_LABELS - 2)))
        torch.testing.assert_close(grad[0].sum(), torch.tensor(-1.0))
        # the excluded label as target keeps its full gradient and spreads nothing
        torch.testing.assert_close(grad[1, blank], torch.tensor(-1.0))
        assert torch.equal(grad[1, :blank], torch.zeros(blank))

    @pytest.mark.unit
    def test_invalid_inputs_are_rejected(self):
        log_probs = torch.zeros(2, NUM_LABELS)
        with pytest.raises(ValueError, match=r"smoothing must be in \[0, 1\]"):
            label_smoothed_log_prob_gradient(log_probs, 1.5)
        with pytest.raises(ValueError, match=r"smoothing must be in \[0, 1\]"):
            label_smoothed_log_prob_gradient(log_probs, -0.1)
        with pytest.raises(ValueError, match="scalar tensor"):
            label_smoothed_log_prob_gradient(log_probs, torch.tensor([SMOOTHING, SMOOTHING]))
        with pytest.raises(ValueError, match="at least two labels"):
            label_smoothed_log_prob_gradient(torch.zeros(2, 1), SMOOTHING)
        with pytest.raises(ValueError, match=r"exclude_labels must be in \[0, 7\)"):
            label_smoothed_log_prob_gradient(log_probs, SMOOTHING, exclude_labels=[NUM_LABELS])
        with pytest.raises(ValueError, match=r"exclude_labels must be in \[0, 7\)"):
            label_smoothed_log_prob_gradient(log_probs, SMOOTHING, exclude_labels=[-1])
        with pytest.raises(ValueError, match="at least two participating labels"):
            label_smoothed_log_prob_gradient(log_probs, SMOOTHING, exclude_labels=list(range(NUM_LABELS - 1)))
        # the boundaries of the smoothing range are valid
        assert label_smoothed_log_prob_gradient(log_probs, 0.0) is log_probs
        assert torch.equal(label_smoothed_log_prob_gradient(log_probs, 1.0), log_probs)

    @pytest.mark.unit
    def test_tensor_smoothing_and_label_dimension(self):
        smoothing = torch.tensor(SMOOTHING)
        log_probs = torch.zeros(NUM_LABELS, 2, requires_grad=True)
        out = label_smoothed_log_prob_gradient(log_probs, smoothing, dim=0)
        (-out[3]).sum().backward()
        expected = -smoothed_targets(torch.tensor([3, 3]), NUM_LABELS, SMOOTHING).t()
        torch.testing.assert_close(log_probs.grad, expected)
