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

from nemo.collections.common.losses.multi_similarity_loss import MultiSimilarityLoss


class TestMultiSimilarityLoss:
    @pytest.mark.unit
    def test_multi_similarity_loss_with_hard_pairs(self):
        torch.manual_seed(42)
        loss_fn = MultiSimilarityLoss(scale_pos=2.0, scale_neg=40.0, margin=0.1)
        # 4 samples, 2 classes (2 samples each)
        logits = torch.randn(4, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])

        loss = loss_fn(logits=logits, labels=labels)
        assert loss.ndim == 0
        assert loss > 0.0
        assert torch.isfinite(loss)

        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    @pytest.mark.unit
    def test_multi_similarity_loss_empty_hard_pairs_cpu(self):
        """When no hard positive/negative pairs exist (e.g. all labels identical), loss must be 0 on CPU."""
        torch.manual_seed(42)
        loss_fn = MultiSimilarityLoss()
        logits = torch.randn(4, 16, requires_grad=True)
        # All samples have the same label, so no negative pairs exist
        labels = torch.tensor([0, 0, 0, 0])

        loss = loss_fn(logits=logits, labels=labels)
        assert loss.device == logits.device
        assert loss.dtype == logits.dtype
        assert loss.item() == 0.0

        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
        assert torch.all(logits.grad == 0.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_multi_similarity_loss_dtypes(self, dtype):
        loss_fn = MultiSimilarityLoss()
        logits = torch.randn(4, 8, dtype=dtype, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 0])

        loss = loss_fn(logits=logits, labels=labels)
        assert loss.dtype == dtype
        assert loss.device == logits.device
