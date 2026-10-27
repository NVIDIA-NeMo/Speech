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

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output


class TestPackedCTCLoss:
    """`forward_packed` against `forward`: loss and gradient parity, edge cases and input checks."""

    @pytest.mark.unit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum", "mean_batch", "mean_volume"])
    def test_loss_and_gradient_match_padded(self, reduction):
        torch.manual_seed(7)
        lengths = torch.tensor([8, 5, 2])
        target_lengths = torch.tensor([3, 1, 0])
        targets = torch.tensor([[1, 1, 2], [2, 0, 0], [0, 0, 0]])
        dense_logits = torch.randn(3, 11, 5, dtype=torch.float64, requires_grad=True)
        packed_logits = dense_logits.detach().clone().requires_grad_()
        criterion = CTCLoss(num_classes=4, reduction=reduction)
        expected = criterion(
            log_probs=dense_logits.log_softmax(-1),
            targets=targets,
            input_lengths=lengths,
            target_lengths=target_lengths,
        )
        packed = pack_encoder_output(packed_logits.log_softmax(-1), lengths)
        actual = criterion.forward_packed(packed, targets, target_lengths)
        torch.testing.assert_close(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(packed_logits.grad, dense_logits.grad)

    @pytest.mark.unit
    @pytest.mark.parametrize("zero_infinity", [False, True])
    def test_impossible_alignment(self, zero_infinity):
        logits = torch.tensor([[[1.0, 0.5, -0.5]]], requires_grad=True)
        lengths = torch.tensor([1])
        targets = torch.tensor([[0, 0]])
        target_lengths = torch.tensor([2])
        criterion = CTCLoss(num_classes=2, zero_infinity=zero_infinity)
        packed = pack_encoder_output(logits.log_softmax(-1), lengths)
        actual = criterion.forward_packed(packed, targets, target_lengths)
        if zero_infinity:
            assert actual.item() == 0.0
            actual.backward()
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        else:
            assert torch.isposinf(actual)

    @pytest.mark.unit
    def test_rejects_padded_log_probs(self):
        criterion = CTCLoss(num_classes=2)
        with pytest.raises(TypeError, match="PackedEncoderActivations"):
            criterion.forward_packed(torch.zeros(1, 3, 3), torch.zeros(1, 1, dtype=torch.long), torch.tensor([1]))
