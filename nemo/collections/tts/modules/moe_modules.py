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

from typing import Callable, Tuple

import torch
import torch.nn.functional as F

from nemo.collections.tts.modules.ffn_modules import ConvolutionLayer


class MoERouter(torch.nn.Module):
    """
    Router for Mixture of Experts that selects which experts to use for each token.
    Supports multiple routing strategies including top-k and Sinkhorn routing.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        router_jitter_noise: float = 0.0,
        routing_strategy: str = "top_k",  # "top_k" or "sinkhorn"
    ):
        """
        Args:
            d_model (int): Model dimension
            num_experts (int): Number of experts
            top_k (int): Number of experts to select per token
            router_jitter_noise (float): Add noise to router logits for exploration during training
            routing_strategy (str): Strategy for routing ("top_k" or "sinkhorn")
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router_jitter_noise = router_jitter_noise
        self.routing_strategy = routing_strategy
        assert routing_strategy in ["top_k", "sinkhorn"], "Invalid routing strategy"

        # Router is a simple linear layer that outputs logits for each expert
        self.router = torch.nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing decisions for each token.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
            x_mask (torch.Tensor): Mask tensor of shape (B, T) where 1=valid token, 0=padding

        Returns:
            Tuple containing:
                - expert_weights (torch.Tensor): Normalized weights for selected experts of shape (B, T, top_k).
                    For padded positions, weights are set to 0.
                - expert_indices (torch.Tensor): Indices of selected experts of shape (B, T, top_k).
                    For padded positions, indices are set to -1 (sentinel value).
                - router_logits (torch.Tensor): Raw router logits of shape (B, T, num_experts).
                    Padded positions are masked to zero.
                - router_probs (torch.Tensor): Router probabilities after softmax of shape (B, T, num_experts).
                    Padded positions are masked to zero.
        """
        # Compute router logits: (B, T, num_experts)
        router_logits = self.router(x * x_mask.unsqueeze(-1))

        # Add jitter noise during training for exploration
        if self.training and self.router_jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.router_jitter_noise
            router_logits = router_logits + noise

        # Mask router logits to ensure padded positions remain zero
        router_logits = router_logits * x_mask.unsqueeze(-1)

        # Compute routing probabilities for each token.
        # Padded positions with logits of [0, 0, ..., 0] will produce a uniform softmax ([1/n, ..., 1/n]);
        # this is acceptable, since we require valid probabilities for top-k selection and normalization.
        # Sinkhorn routing is used only during training for balancing, while at inference simple softmax is used for efficiency.
        if self.routing_strategy == "sinkhorn" and self.training:
            router_probs = self._sinkhorn_routing(router_logits, x_mask)
        else:
            router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        # expert_weights: (B, T, top_k), expert_indices: (B, T, top_k)
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize weights to sum to 1
        # For padded positions: uniform probs -> 1/top_k
        # For valid positions: normal routing weights
        # Avoid division by zero when all weights are zero.
        weight_sums = expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights / torch.where(weight_sums > 0, weight_sums, torch.ones_like(weight_sums))

        # Mask expert_weights and expert_indices for padded positions
        # Set expert_indices to -1 for padding so they don't match any valid expert (0 to num_experts-1)
        # This prevents padded tokens from being processed through experts
        expert_weights = expert_weights * x_mask.unsqueeze(-1)  # Zero out weights for padding
        expert_indices = expert_indices.masked_fill(~x_mask.unsqueeze(-1).bool(), -1)  # Set to -1 for padding

        # Mask router_probs for return
        router_probs = router_probs * x_mask.unsqueeze(-1)

        return expert_weights, expert_indices, router_logits, router_probs

    @staticmethod
    def _sinkhorn_routing(
        logits: torch.Tensor, x_mask: torch.Tensor, num_iters: int = 100, e_tol: float = 1e-3
    ) -> torch.Tensor:
        """
        Padding-aware Sinkhorn routing with convergence checking.

        This implementation:
        1. Extracts only valid (non-padded) tokens before Sinkhorn
        2. Applies Sinkhorn-Knopp algorithm with convergence criterion
        3. Re-pads the output to original shape

        The algorithm computes a doubly stochastic matrix by iteratively
        normalizing rows and columns using diagonal scaling factors.

        Args:
            logits (torch.Tensor): Router logits of shape (B, T, num_experts)
            x_mask (torch.Tensor): Mask of shape (B, T) where 1=valid token, 0=padding
            num_iters (int): Maximum number of Sinkhorn iterations (default: 100)
            e_tol (float): Convergence tolerance for scaling factors (default: 1e-3)

        Returns:
            torch.Tensor: Routing probabilities of shape (B, T, num_experts)
                Valid tokens: doubly stochastic probabilities
                Padded tokens: zeros
        """
        B, T, E = logits.shape

        # Extract valid tokens (exclude padding)
        valid_mask = x_mask.view(-1).bool()  # (B*T,)
        valid_logits = logits.view(B * T, E)[valid_mask]  # (N, E) where N = number of valid tokens

        if valid_logits.numel() == 0:
            # All tokens are padding, return zeros
            return torch.zeros_like(logits)

        # Numerical stability: subtract max per row to prevent exp overflow.
        # This is similar to the log-sum-exp trick used in softmax.
        # For Sinkhorn, subtracting a constant per row doesn't change the final
        # doubly-stochastic result since both row and column normalizations will
        # absorb the scaling factor.
        valid_logits_stable = valid_logits - valid_logits.max(dim=-1, keepdim=True).values

        # Apply exp to get cost matrix (must be positive for Sinkhorn)
        K = torch.exp(valid_logits_stable)  # (N, E)

        # Initialize diagonal scaling factors
        d1 = torch.ones(K.size(0), device=K.device, dtype=K.dtype)  # Row scaling (N,)
        d2 = torch.ones(K.size(1), device=K.device, dtype=K.dtype)  # Column scaling (E,)

        # Sinkhorn-Knopp iterations with convergence check
        for _ in range(num_iters):
            d1_old = d1.clone()

            # Update row scaling: d1[i] = 1 / sum_j(K[i,j] * d2[j])
            d1 = 1.0 / (torch.matmul(K, d2) + 1e-9)

            # Update column scaling: d2[j] = 1 / sum_i(K[i,j] * d1[i])
            d2 = 1.0 / (torch.matmul(K.t(), d1) + 1e-9)

            # Clamp scaling factors to prevent numerical instability from accumulating
            d1 = torch.clamp(d1, min=1e-9, max=1e9)
            d2 = torch.clamp(d2, min=1e-9, max=1e9)

            # Check convergence based on change in scaling factors
            err = torch.mean(torch.abs(d1_old - d1))
            if err < e_tol:
                break

        # Compute scaled matrix using broadcasting (avoids materializing NxN diagonal matrices):
        # P = diag(d1) @ K @ diag(d2)  =>  P[i, j] = d1[i] * K[i, j] * d2[j]
        P = (d1[:, None] * K) * d2[None, :]  # (N, E)

        # Final row normalization to ensure each row sums to 1 (valid probability distribution)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-9)  # (N, E)

        # Re-pad to original shape
        result = torch.zeros(B * T, E, device=logits.device, dtype=logits.dtype)
        result[valid_mask] = P
        result = result.view(B, T, E)

        return result


class PositionwiseConvFFMoE(torch.nn.Module):
    """
    Mixture of Experts version of `PositionwiseConvFF`.
    Uses multiple expert FFN networks with a learned router.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        p_dropout: float,
        num_experts: int = 8,
        top_k_experts: int = 2,
        kernel_size: int = 1,
        bias: bool = False,
        is_causal: bool = True,
        non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
        router_jitter_noise: float = 0.0,
        routing_strategy: str = "top_k",
    ):
        """
        Args:
            d_model (int): Input and output dimension
            d_ffn (int): Hidden dimension of FFN (usually 4 * d_model, or d_model for param-matched MoE)
            p_dropout (float): Dropout probability
            num_experts (int): Number of expert networks
            top_k_experts (int): Number of experts to use per token
            kernel_size (int): Convolution kernel size. Must be 1 for MoE so that each expert
                is a standard pointwise linear FFN (Conv1d with kernel_size=1 is equivalent to
                nn.Linear applied independently at each position).
            bias (bool): Whether to use bias in convolution layers
            is_causal (bool): Whether to use causal convolution
            non_linearity (Callable): Activation function
            router_jitter_noise (float): Noise for router exploration
            routing_strategy (str): Routing strategy ("top_k" or "sinkhorn")
        """
        if kernel_size != 1:
            raise ValueError(
                f"`PositionwiseConvFFMoE` requires kernel_size=1, got {kernel_size}. "
                f"Each MoE expert must be a pointwise linear FFN (Conv1d with kernel_size=1 == nn.Linear). "
                f"kernel_size > 1 is not supported because (1) standard MoE experts are linear layers, "
                f"and (2) MoE dispatch gathers tokens from arbitrary (batch, time) positions, so "
                f"Conv1d with kernel_size > 1 would mix non-adjacent tokens."
            )
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.non_linearity = non_linearity

        # Router for expert selection
        self.router = MoERouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k_experts,
            router_jitter_noise=router_jitter_noise,
            routing_strategy=routing_strategy,
        )

        # Create multiple expert FFN networks
        self.experts = torch.nn.ModuleList()
        for _ in range(num_experts):
            expert = torch.nn.ModuleDict(
                {
                    'proj': ConvolutionLayer(d_model, d_ffn, bias=bias, kernel_size=kernel_size, is_causal=is_causal),
                    'o_net': ConvolutionLayer(d_ffn, d_model, bias=bias, kernel_size=kernel_size, is_causal=is_causal),
                }
            )
            self.experts.append(expert)

        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Mixture of Experts feedforward layer.

        For each valid token (x_mask=1), routes to top_k experts based on router predictions.
        Padded tokens (x_mask=0) are assigned expert_indices=-1 and are not processed through any expert,
        ensuring they remain zero in the output.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
            x_mask (torch.Tensor): Mask tensor of shape (B, T) where 1=valid token, 0=padding

        Returns:
            Tuple containing:
                - output (torch.Tensor): Output tensor of shape (B, T, C).
                    Valid tokens contain weighted combination of top_k expert outputs.
                    Padded positions remain zero (never processed by experts).
                - router_logits (torch.Tensor): Raw router logits for auxiliary loss of shape (B, T, num_experts).
                    Padded positions are masked to zero.
                - router_probs (torch.Tensor): Router probabilities for auxiliary loss of shape (B, T, num_experts).
                    Padded positions are masked to zero.
                - expert_indices (torch.Tensor): Selected expert indices of shape (B, T, top_k).
                    For padded positions, indices are -1. For computing expert selection statistics.
        """
        # Get expert routing from router
        expert_weights, expert_indices, router_logits, router_probs = self.router(x, x_mask)
        # expert_weights: (B, T, top_k)
        # expert_indices: (B, T, top_k)
        # router_logits: (B, T, num_experts)
        # router_probs: (B, T, num_experts)

        # Vectorized dispatch: flatten all (token, expert-slot) assignments once,
        # sort by expert to get contiguous slices, then process each expert on its slice.
        B, T, C = x.shape
        top_k = expert_indices.shape[-1]

        # Flatten token dimension: (B*T, C)
        x_flat = x.view(-1, C)
        num_tokens = x_flat.size(0)  # B * T

        # Flatten routing assignments to 1-D vectors:
        #   assign_expert: (num_tokens * top_k,)  — which expert each assignment targets
        #   assign_weight: (num_tokens * top_k, 1) — routing weight for each assignment
        assign_expert = expert_indices.reshape(-1)
        assign_weight = expert_weights.reshape(-1, 1)

        # Map each assignment back to its source token index (0 .. num_tokens-1).
        # token_indices: (num_tokens * top_k,)
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)

        # Filter out padding assignments (expert_indices == -1 for padded positions).
        # This is required because torch.bincount does not accept negative values,
        # and padded tokens should not be processed by any expert.
        valid_assign_mask = assign_expert != -1
        assign_expert = assign_expert[valid_assign_mask]
        assign_weight = assign_weight[valid_assign_mask]
        token_indices = token_indices[valid_assign_mask]

        # Initialize flat output buffer.
        output_flat = torch.zeros_like(x_flat)

        if assign_expert.numel() > 0:
            # Sort assignments by expert so each expert's tokens form a contiguous slice.
            sorted_expert, sort_idx = torch.sort(assign_expert)
            sorted_token_indices = token_indices[sort_idx]
            sorted_weights = assign_weight[sort_idx]

            # Compute per-expert assignment counts and slice boundaries.
            counts = torch.bincount(sorted_expert, minlength=self.num_experts)
            offsets = counts.cumsum(0)
            starts = torch.zeros_like(offsets)
            starts[1:] = offsets[:-1]

            # Process each expert on its contiguous slice of assignments.
            for expert_idx in range(self.num_experts):
                count = counts[expert_idx].item()
                if count == 0:
                    continue

                start = starts[expert_idx].item()
                end = start + count

                expert_token_idx = sorted_token_indices[start:end]
                expert_token_weights = sorted_weights[start:end]  # (N_assign, 1)

                # Gather tokens for this expert: (N_assign, C)
                # Note: expert_token_idx values are in [0, B*T-1] (token-space indices, not assignment-space indices),
                # we can safely index into x_flat (B*T, C) with these indices.
                expert_tokens = x_flat[expert_token_idx]

                # Add batch dimension expected by conv layers: (1, N_assign, C)
                expert_tokens = expert_tokens.unsqueeze(0)

                # Apply expert FFN
                expert_out = self.non_linearity(self.experts[expert_idx]['proj'](expert_tokens.transpose(1, 2)))
                expert_out = self.dropout(self.experts[expert_idx]['o_net'](expert_out).transpose(1, 2))
                expert_out = expert_out.squeeze(0)  # (N_assign, C)

                # Weight and accumulate back to the source token positions.
                expert_out = expert_out * expert_token_weights
                output_flat.index_add_(0, expert_token_idx, expert_out)

        # Reshape back to (B, T, C)
        output = output_flat.view(B, T, C)

        return output, router_logits, router_probs, expert_indices
