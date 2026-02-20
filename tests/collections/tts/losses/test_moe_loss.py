# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Unit tests for MoE loss modules.
"""

import pytest
import torch

from nemo.collections.tts.losses.moe_loss import MoEAuxiliaryLoss, MoELoadBalancingLoss, MoERouterZLoss


@pytest.mark.unit
class TestMoELosses:
    """Test MoE loss modules."""

    def test_load_balancing_loss(self):
        """Test MoELoadBalancingLoss."""
        loss_fn = MoELoadBalancingLoss(num_experts=8, loss_scale=0.01)

        router_probs = torch.softmax(torch.randn(2, 10, 8), dim=-1)
        loss = loss_fn(router_probs=router_probs)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_router_z_loss(self):
        """Test MoERouterZLoss."""
        loss_fn = MoERouterZLoss(loss_scale=0.001)

        router_logits = torch.randn(2, 10, 8)
        loss = loss_fn(router_logits=router_logits)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_auxiliary_loss(self):
        """Test MoEAuxiliaryLoss returns tuple of 3 values."""
        loss_fn = MoEAuxiliaryLoss(
            num_experts=8,
            load_balancing_loss_scale=0.01,
            router_z_loss_scale=0.001,
        )

        router_logits = torch.randn(2, 10, 8)
        router_probs = torch.softmax(router_logits, dim=-1)

        result = loss_fn(router_logits=router_logits, router_probs=router_probs)
        assert len(result) == 3, "MoEAuxiliaryLoss should return tuple of 3 values"

        load_balancing_loss, router_z_loss, total_loss = result

        # Check values are scalars
        assert load_balancing_loss.ndim == 0
        assert router_z_loss.ndim == 0
        assert total_loss.ndim == 0

        # Check total is sum of components
        expected_total = load_balancing_loss + router_z_loss
        assert torch.allclose(total_loss, expected_total)
