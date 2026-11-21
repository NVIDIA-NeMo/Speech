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

from nemo.collections.common.losses import CrossEntropyLoss, NLLLoss


class TestCrossEntropyLosses:
    @pytest.mark.unit
    def test_nll_loss_default_2d_typecheck(self):
        """Verify NLLLoss default 2D classification log_probs_ndim=2 passes typecheck."""
        loss_fn = NLLLoss(log_probs_ndim=2)
        log_probs = torch.randn(4, 10).log_softmax(dim=-1)
        labels = torch.randint(0, 10, (4,))
        loss = loss_fn(log_probs=log_probs, labels=labels)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    @pytest.mark.unit
    def test_nll_loss_mask_parameter_aliases(self):
        """Verify NLLLoss accepts both loss_mask and output_mask."""
        loss_fn = NLLLoss(log_probs_ndim=3)
        log_probs = torch.randn(2, 5, 10).log_softmax(dim=-1)
        labels = torch.randint(0, 10, (2, 5))
        mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])

        loss_with_loss_mask = loss_fn(log_probs=log_probs, labels=labels, loss_mask=mask)
        loss_with_output_mask = loss_fn(log_probs=log_probs, labels=labels, output_mask=mask)

        torch.testing.assert_close(loss_with_loss_mask, loss_with_output_mask)

    @pytest.mark.unit
    def test_cross_entropy_mask_parameter_aliases(self):
        """Verify CrossEntropyLoss accepts both loss_mask and output_mask."""
        loss_fn = CrossEntropyLoss(logits_ndim=3)
        logits = torch.randn(2, 5, 10)
        labels = torch.randint(0, 10, (2, 5))
        mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])

        loss_with_loss_mask = loss_fn(logits=logits, labels=labels, loss_mask=mask)
        loss_with_output_mask = loss_fn(logits=logits, labels=labels, output_mask=mask)

        torch.testing.assert_close(loss_with_loss_mask, loss_with_output_mask)

    @pytest.mark.unit
    def test_cross_entropy_empty_mask_3d(self):
        """Verify CrossEntropyLoss does not crash when all elements are masked out in 3D tensors."""
        loss_fn = CrossEntropyLoss(logits_ndim=3)
        logits = torch.randn(2, 5, 10, requires_grad=True)
        labels = torch.randint(0, 10, (2, 5))
        mask = torch.zeros(2, 5, dtype=torch.bool)

        loss = loss_fn(logits=logits, labels=labels, loss_mask=mask)
        assert loss.item() == 0.0
        loss.backward()
        assert logits.grad is not None
        assert torch.all(logits.grad == 0.0)

    @pytest.mark.unit
    def test_nll_loss_empty_mask_3d(self):
        """Verify NLLLoss does not crash when all elements are masked out in 3D tensors."""
        loss_fn = NLLLoss(log_probs_ndim=3)
        log_probs = torch.randn(2, 5, 10, requires_grad=True).log_softmax(dim=-1)
        labels = torch.randint(0, 10, (2, 5))
        mask = torch.zeros(2, 5, dtype=torch.bool)

        loss = loss_fn(log_probs=log_probs, labels=labels, loss_mask=mask)
        assert loss.item() == 0.0

    @pytest.mark.unit
    def test_empty_mask_reduction_none(self):
        """Verify empty mask with reduction='none' returns an empty tensor."""
        ce = CrossEntropyLoss(logits_ndim=3, reduction='none')
        nll = NLLLoss(log_probs_ndim=3, reduction='none')
        logits = torch.randn(2, 5, 10)
        log_probs = logits.log_softmax(dim=-1)
        labels = torch.randint(0, 10, (2, 5))
        mask = torch.zeros(2, 5, dtype=torch.bool)

        ce_loss = ce(logits=logits, labels=labels, loss_mask=mask)
        assert ce_loss.shape == torch.Size([0])

        nll_loss = nll(log_probs=log_probs, labels=labels, loss_mask=mask)
        assert nll_loss.shape == torch.Size([0])
