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

import torch
import torch.nn.functional as F

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LossType, ProbsType
from nemo.core.neural_types.neural_type import NeuralType


def compute_expert_usage(router_probs: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute expert usage statistics from router probabilities.

    Args:
        router_probs (torch.Tensor): Router probabilities of shape (B, T, num_experts).
            Padded positions should be masked to zero.
        x_mask (torch.Tensor, optional): Mask of shape (B, T) where 1=valid, 0=padding.
            If provided, computes average only over valid tokens. If None, uses all positions.

    Returns:
        torch.Tensor: Average probability of routing to each expert of shape (num_experts,)
    """
    if x_mask is not None:
        # Sum over valid tokens only
        valid_probs_sum = (router_probs * x_mask.unsqueeze(-1)).sum(dim=[0, 1])  # (num_experts,)
        num_valid_tokens = x_mask.sum()
        expert_usage = valid_probs_sum / (num_valid_tokens + 1e-9)
    else:
        # Average over all positions
        expert_usage = router_probs.mean(dim=[0, 1])

    return expert_usage


class MoELoadBalancingLoss(Loss):
    """
    Load balancing auxiliary loss for Mixture of Experts.
    Encourages uniform distribution of tokens across experts.

    Based on Switch Transformer paper: https://arxiv.org/abs/2101.03961
    """

    def __init__(self, num_experts: int, loss_scale: float = 0.01):
        """
        Args:
            num_experts (int): Number of experts in the MoE layer
            loss_scale (float): Scaling factor for the loss
        """
        super().__init__()
        self.num_experts = num_experts
        self.loss_scale = loss_scale

    @property
    def input_types(self):
        return {
            "router_probs": NeuralType(('B', 'T', 'D'), ProbsType()),  # D = num_experts
            "x_mask": NeuralType(('B', 'T'), ProbsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, router_probs, x_mask=None):
        """
        Args:
            router_probs (torch.Tensor): Router probabilities of shape (B, T, num_experts).
                For padded positions, probabilities should be masked to zero.
            x_mask (torch.Tensor, optional): Mask of shape (B, T) where 1=valid, 0=padding.
                If provided, computes mean only over valid tokens. If None, uses all positions.

        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        if self.loss_scale == 0.0:
            return torch.tensor(0.0, device=router_probs.device, dtype=router_probs.dtype)

        # Compute average probability of routing to each expert (over valid tokens only)
        expert_usage = compute_expert_usage(router_probs, x_mask)  # (num_experts,)

        # Target is uniform distribution
        target = torch.ones_like(expert_usage) / self.num_experts

        # L2 loss to encourage uniform distribution
        loss = F.mse_loss(expert_usage, target)

        return loss * self.loss_scale


class MoERouterZLoss(Loss):
    """
    Router z-loss for Mixture of Experts.
    Encourages smaller logits for numerical stability.

    Based on ST-MoE paper: https://arxiv.org/abs/2202.08906
    """

    def __init__(self, loss_scale: float = 0.001):
        """
        Args:
            loss_scale (float): Scaling factor for the loss
        """
        super().__init__()
        self.loss_scale = loss_scale

    @property
    def input_types(self):
        return {
            "router_logits": NeuralType(('B', 'T', 'D'), ProbsType()),  # D = num_experts
            "x_mask": NeuralType(('B', 'T'), ProbsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, router_logits, x_mask=None):
        """
        Args:
            router_logits (torch.Tensor): Router logits of shape (B, T, num_experts).
                For padded positions, logits should be masked to zero.
            x_mask (torch.Tensor, optional): Mask of shape (B, T) where 1=valid, 0=padding.
                If provided, computes mean only over valid tokens. If None, uses all positions.

        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        if self.loss_scale == 0.0:
            return torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)

        # Log of sum of exponentials (logsumexp for numerical stability)
        log_z = torch.logsumexp(router_logits, dim=-1)  # (B, T)

        # Compute mean over valid tokens only
        if x_mask is not None:
            # Only average over valid positions
            log_z_squared = log_z**2
            z_loss = (log_z_squared * x_mask).sum() / (x_mask.sum() + 1e-9)
        else:
            z_loss = torch.mean(log_z**2)

        return z_loss * self.loss_scale


class MoEAuxiliaryLoss(Loss):
    """
    Combined auxiliary loss for Mixture of Experts.
    Includes both load balancing loss and router z-loss.
    """

    def __init__(
        self,
        num_experts: int,
        load_balancing_loss_scale: float = 0.01,
        router_z_loss_scale: float = 0.001,
    ):
        """
        Args:
            num_experts (int): Number of experts in the MoE layer
            load_balancing_loss_scale (float): Scale for load balancing loss
            router_z_loss_scale (float): Scale for router z-loss
        """
        super().__init__()
        self.num_experts = num_experts

        self.load_balancing_loss = MoELoadBalancingLoss(
            num_experts=num_experts,
            loss_scale=load_balancing_loss_scale,
        )

        self.router_z_loss = MoERouterZLoss(
            loss_scale=router_z_loss_scale,
        )

    @property
    def input_types(self):
        return {
            "router_logits": NeuralType(('B', 'T', 'D'), ProbsType()),
            "router_probs": NeuralType(('B', 'T', 'D'), ProbsType()),
            "x_mask": NeuralType(('B', 'T'), ProbsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "load_balancing_loss": NeuralType(elements_type=LossType()),
            "router_z_loss": NeuralType(elements_type=LossType()),
            "total_loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, router_logits, router_probs, x_mask=None):
        """
        Compute combined MoE auxiliary losses.

        Args:
            router_logits (torch.Tensor): Router logits before softmax of shape (B, T, num_experts).
                For padded positions, logits should be masked to zero.
            router_probs (torch.Tensor): Router probabilities after softmax of shape (B, T, num_experts).
                For padded positions, probabilities should be masked to zero.
            x_mask (torch.Tensor, optional): Mask of shape (B, T) where 1=valid, 0=padding.
                If provided, losses are computed only over valid tokens. If None, uses all positions.

        Returns:
            Tuple of (load_balancing_loss, router_z_loss, total_loss)
                All are scalar tensors. If loss_scale=0.0, returns zero without computation.
        """
        load_balancing_loss = self.load_balancing_loss(router_probs=router_probs, x_mask=x_mask)
        router_z_loss = self.router_z_loss(router_logits=router_logits, x_mask=x_mask)
        total_loss = load_balancing_loss + router_z_loss

        return load_balancing_loss, router_z_loss, total_loss
