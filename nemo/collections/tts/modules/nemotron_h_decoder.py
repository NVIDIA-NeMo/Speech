# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
NemotronH model implementation for use as a decoder backbone in TTS models.
Ported from: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/blob/main/modeling_nemotron_h.py

This is a hybrid Mamba2/Attention model that can be configured with different
layer types (Mamba, Attention, MLP, MoE) via the hybrid_override_pattern config.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from nemo.utils import logging


# Try to import optimized kernels, fall back to pure PyTorch if unavailable
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined

    MAMBA_SSM_AVAILABLE = True
except ImportError:
    selective_state_update = None
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None
    MAMBA_SSM_AVAILABLE = False

try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn

    RMSNORM_FN_AVAILABLE = True
except ImportError:
    rmsnorm_fn = None
    RMSNORM_FN_AVAILABLE = False

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None
    CAUSAL_CONV1D_AVAILABLE = False

try:
    from transformers.utils.import_utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

    if is_flash_attn_2_available():
        from transformers.modeling_flash_attention_utils import _flash_attention_forward

        FLASH_ATTN_AVAILABLE = True
    else:
        _flash_attention_forward = None
        FLASH_ATTN_AVAILABLE = False
except ImportError:
    is_flash_attn_2_available = None
    is_flash_attn_greater_or_equal_2_10 = None
    _flash_attention_forward = None
    FLASH_ATTN_AVAILABLE = False


# Check if fast path is available (all optimized kernels present)
IS_FAST_PATH_AVAILABLE = all(
    [
        MAMBA_SSM_AVAILABLE,
        CAUSAL_CONV1D_AVAILABLE,
        selective_state_update is not None,
        mamba_chunk_scan_combined is not None,
        causal_conv1d_fn is not None,
    ]
)


def get_activation_fn(activation: str):
    """Get activation function by name."""
    if activation == "silu" or activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    else:
        raise ValueError(f"Unsupported activation: {activation}")


@dataclass
class NemotronHConfig:
    """
    Configuration class for NemotronH model.

    This configuration controls the hybrid Mamba2/Attention architecture.
    The layer types are specified via hybrid_override_pattern where:
    - 'M' = Mamba2 layer
    - '*' = Attention layer
    - '-' = MLP layer
    - 'E' = MoE layer
    """

    # Model dimensions
    hidden_size: int = 1536
    num_hidden_layers: int = 24
    vocab_size: int = 131072

    # Attention config
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: Optional[int] = None
    attention_dropout: float = 0.0
    attention_bias: bool = False
    max_position_embeddings: int = 4096

    # Mamba config
    mamba_num_heads: int = 64
    mamba_head_dim: int = 64
    ssm_state_size: int = 128
    conv_kernel: int = 4
    n_groups: int = 8
    chunk_size: int = 256
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: Tuple[float, float] = (0.0, float("inf"))
    mamba_hidden_act: str = "silu"
    use_conv_bias: bool = True
    use_bias: bool = False

    # MLP config
    intermediate_size: int = 4096
    mlp_hidden_act: str = "silu"
    mlp_bias: bool = False

    # MoE config (if using MoE layers)
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 1024
    moe_shared_expert_intermediate_size: int = 2048
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True

    # Layer pattern: M=Mamba, *=Attention, -=MLP, E=MoE
    # Example: "M*M*M*M*" = alternating Mamba and Attention
    hybrid_override_pattern: str = "M*M*M*M*M*M*M*M*M*M*M*M*"

    # Normalization
    layer_norm_epsilon: float = 1e-5
    residual_in_fp32: bool = True

    # Initialization
    initializer_range: float = 0.02
    rescale_prenorm_residual: bool = True

    # Output
    use_cache: bool = True
    use_return_dict: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
    num_logits_to_keep: int = 1

    # Attention implementation
    _attn_implementation: str = "sdpa"  # "eager", "sdpa", or "flash_attention_2"

    def __post_init__(self):
        # Derive layers_block_type from hybrid_override_pattern
        pattern_map = {'M': 'mamba', '*': 'attention', '-': 'mlp', 'E': 'moe'}
        self.layers_block_type = [pattern_map.get(c, 'mamba') for c in self.hybrid_override_pattern]

        # Ensure num_hidden_layers matches pattern length
        if len(self.layers_block_type) != self.num_hidden_layers:
            # Extend or truncate pattern to match num_hidden_layers
            if len(self.layers_block_type) < self.num_hidden_layers:
                # Repeat pattern
                full_pattern = self.hybrid_override_pattern * (
                    self.num_hidden_layers // len(self.hybrid_override_pattern) + 1
                )
                self.hybrid_override_pattern = full_pattern[: self.num_hidden_layers]
                self.layers_block_type = [pattern_map.get(c, 'mamba') for c in self.hybrid_override_pattern]
            else:
                self.layers_block_type = self.layers_block_type[: self.num_hidden_layers]
                self.hybrid_override_pattern = self.hybrid_override_pattern[: self.num_hidden_layers]

        # Set head_dim if not specified
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class NemotronHOutput:
    """Output class for NemotronH model."""

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None  # HybridMambaAttentionDynamicCache
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class NemotronHCausalLMOutput:
    """Output class for NemotronH causal LM."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class HybridMambaAttentionDynamicCache:
    """
    A dynamic cache that handles both attention cache (with seq_len dimension)
    and mamba cache (with constant shape regardless of seq_len).
    """

    def __init__(self, config: NemotronHConfig, batch_size: int, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.has_previous_state = False
        self.conv_kernel_size = config.conv_kernel

        intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        ssm_state_size = config.ssm_state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = []
        self.ssm_states = []
        self.key_cache = []
        self.value_cache = []
        self.transformer_layers = []

        for i in range(config.num_hidden_layers):
            if config.layers_block_type[i] == "mamba":
                self.conv_states.append(
                    torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
                )
                self.ssm_states.append(
                    torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
                )
            else:
                self.conv_states.append(torch.tensor([[]] * batch_size, device=device))
                self.ssm_states.append(torch.tensor([[]] * batch_size, device=device))
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2] if self.key_cache[layer_idx].dim() > 2 else 0

    def update_conv_state(self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False):
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states[layer_idx].device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states[layer_idx].device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def reset(self):
        """Reset all cache states to zero."""
        for i in range(len(self.conv_states)):
            if self.conv_states[i].numel() > 0:
                self.conv_states[i].zero_()
            if self.ssm_states[i].numel() > 0:
                self.ssm_states[i].zero_()
        for i in range(len(self.key_cache)):
            if self.key_cache[i].numel() > 0:
                self.key_cache[i].zero_()
            if self.value_cache[i].numel() > 0:
                self.value_cache[i].zero_()


class NemotronHRMSNorm(nn.Module):
    """RMSNorm implementation for NemotronH."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


class MambaRMSNormGated(nn.Module):
    """Gated RMSNorm for Mamba layers."""

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, hidden_states: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Only use Triton kernel if available AND tensors are on CUDA
        use_triton = RMSNORM_FN_AVAILABLE and rmsnorm_fn is not None and hidden_states.is_cuda

        if use_triton:
            return rmsnorm_fn(
                x=hidden_states,
                weight=self.weight,
                bias=None,
                z=gate,
                eps=self.variance_epsilon,
                group_size=self.group_size,
                norm_before_gate=False,
            )
        else:
            # Fallback: simple RMSNorm + gating (works on CPU and GPU)
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            hidden_states = (self.weight.to(torch.float32) * hidden_states).to(input_dtype)
            if gate is not None:
                hidden_states = hidden_states * F.silu(gate)
            return hidden_states


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """Pad tensor on seq_len dim (dim=1)."""
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """Pad and reshape tensor into chunks."""
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    """Compute segment sum for SSM."""
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """Zero out hidden states for padding tokens."""
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


class NemotronHMamba2Mixer(nn.Module):
    """
    Mamba2 mixer layer implementation.
    Computes state space model operations for sequence modeling.
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.mamba_num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.mamba_hidden_act
        self.act = get_activation_fn(config.mamba_hidden_act)
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.n_groups = config.n_groups
        self.head_dim = config.mamba_head_dim
        self.chunk_size = config.chunk_size
        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)

        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.norm = MambaRMSNormGated(
            self.intermediate_size, eps=self.layer_norm_epsilon, group_size=self.intermediate_size // self.n_groups
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Only use CUDA kernels if available AND tensors are on CUDA
        if IS_FAST_PATH_AVAILABLE and hidden_states.is_cuda:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2

        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            # Cached forward (single token)
            _, _, gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
                [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )

            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )

            A = -torch.exp(self.A_log.float())
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)

            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[:, None, ...]
        else:
            # Full sequence forward
            A = -torch.exp(self.A_log.float())
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

            if self.training and cache_params is None:
                out = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=False,
                    **dt_limit_kwargs,
                )
            else:
                _, _, gate, hidden_states_B_C, dt = projected_states.split(
                    [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
                )

                if cache_params is not None:
                    hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                    conv_states = F.pad(
                        hidden_states_B_C_transposed,
                        (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                    )
                    cache_params.update_conv_state(
                        layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True
                    )

                if self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
                    )
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)

                hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )

                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    dt,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )

                if ssm_state is not None and cache_params is not None:
                    cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

                scan_output = scan_output.view(batch_size, seq_len, -1)
                scan_output = self.norm(scan_output, gate)
                out = self.out_proj(scan_output)

        return out

    def torch_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Pure PyTorch implementation (slower but works without CUDA kernels)."""
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)

        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        # Convolution
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            cache_params.update_conv_state(
                layer_idx=self.layer_idx, new_conv_state=hidden_states_B_C, cache_init=False
            )
            conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)
            hidden_states_B_C = torch.sum(conv_states * self.conv1d.weight.squeeze(1), dim=-1)
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = F.pad(
                    hidden_states_B_C_transposed,
                    (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                )
                cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True)
            hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        # SSM
        A = -torch.exp(self.A_log.float())

        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            # Single step SSM update
            cache_device = cache_params.ssm_states[self.layer_idx].device
            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)
            dt = F.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])

            A_expanded = (
                A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            )
            dA = (torch.exp(dt[..., None] * A_expanded)).to(device=cache_device)

            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            dB = dt[..., None] * B[..., None, :]

            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)

            cache_params.update_ssm_state(
                layer_idx=self.layer_idx, new_ssm_state=cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])

            ssm_states = cache_params.ssm_states[self.layer_idx].to(device=C.device, dtype=C.dtype)
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # Full sequence SSM (chunked)
            dt = F.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)

            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            hidden_states = hidden_states * dt[..., None]
            A_dt = A.to(hidden_states.dtype) * dt

            hidden_states, A_dt, B, C = [
                reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A_dt, B, C)
            ]

            A_dt = A_dt.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A_dt, dim=-1)
            L = torch.exp(segment_sum(A_dt))

            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
            G = G_intermediate.sum(dim=-1)
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

            if cache_params is not None and cache_position is not None and cache_position[0] > 0:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=states.device)
            else:
                previous_states = torch.zeros_like(states[:, :1])

            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
            decay_chunk = decay_chunk.transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            state_decay_out = torch.exp(A_cumsum)
            C_times_states = C[..., None, :] * states[:, :, None, ...]
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

            y = Y_diag + Y_off
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
            y = y + D_residual

            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)

            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

        scan_output = self.norm(y, gate)
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for multi-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class NemotronHAttention(nn.Module):
    """Multi-headed attention for NemotronH."""

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class NemotronHFlashAttention2(NemotronHAttention):
    """
    FlashAttention2 path for NemotronH attention.

    Falls back to eager/SDPA attention if flash-attn is not installed.
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self._flash_attn_uses_top_left_mask = (
            not is_flash_attn_greater_or_equal_2_10() if is_flash_attn_greater_or_equal_2_10 is not None else True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if not FLASH_ATTN_AVAILABLE or _flash_attention_forward is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Query is [B, T, H, D] for flash-attn helper.
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        # Keep key/value as [B, H_kv, T, D] while updating cache.
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Convert key/value to [B, T, H, D] for flash-attn helper.
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self.config, "sliding_window", None),
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


NEMOTRONH_ATTENTION_CLASSES = {
    "eager": NemotronHAttention,
    "sdpa": NemotronHAttention,
    "flash_attention_2": NemotronHFlashAttention2,
}


class NemotronHMLP(nn.Module):
    """MLP layer for NemotronH."""

    def __init__(
        self, config: NemotronHConfig, intermediate_size: Optional[int] = None, layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = get_activation_fn(config.mlp_hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class NemotronHTopkRouter(nn.Module):
    """
    Top-k router for Mixture of Experts.

    Routes tokens to the top-k experts based on learned routing weights.
    Supports grouped routing where experts are divided into groups and
    top-k groups are selected first, then top-k experts within those groups.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))
        nn.init.normal_(self.weight, mean=0.0, std=config.initializer_range)

    @torch.no_grad()
    def get_topk_indices(self, scores: torch.Tensor) -> torch.Tensor:
        """Get top-k expert indices using grouped routing."""
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)

        # Compute group scores by taking top-2 within each group and summing
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )

        # Select top-k groups
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Create mask for experts in selected groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )

        # Zero out scores for experts not in selected groups
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        # Select top-k experts from remaining
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            topk_indices: Indices of selected experts (batch_size * seq_len, top_k)
            topk_weights: Weights for selected experts (batch_size * seq_len, top_k)
        """
        hidden_states = hidden_states.view(-1, self.config.hidden_size)

        # Compute router logits and convert to probabilities via sigmoid
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()

        # Get top-k expert indices
        topk_indices = self.get_topk_indices(scores)

        # Gather weights for selected experts
        topk_weights = scores.gather(1, topk_indices)

        # Optionally normalize weights
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        # Apply routing scaling factor
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights


class NemotronHMOE(nn.Module):
    """
    Mixture of Experts layer for NemotronH.

    Combines multiple expert MLPs with a router that selects which experts
    to use for each token. Also includes shared experts that are always used.
    """

    def __init__(self, config: NemotronHConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Create routed experts
        self.experts = nn.ModuleList(
            [
                NemotronHMLP(config, intermediate_size=config.moe_intermediate_size, layer_idx=layer_idx)
                for _ in range(config.n_routed_experts)
            ]
        )

        # Router for selecting experts
        self.gate = NemotronHTopkRouter(config)

        # Shared experts (always used)
        self.shared_experts = NemotronHMLP(
            config=config, intermediate_size=config.moe_shared_expert_intermediate_size, layer_idx=layer_idx
        )

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply mixture of experts to hidden states.

        Args:
            hidden_states: Input tensor of shape (batch_size * seq_len, hidden_size)
            topk_indices: Expert indices of shape (batch_size * seq_len, top_k)
            topk_weights: Expert weights of shape (batch_size * seq_len, top_k)

        Returns:
            Output tensor of shape (batch_size * seq_len, hidden_size)
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)

        # Create one-hot mask for expert selection
        expert_mask = F.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)  # (num_experts, batch*seq, top_k)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                # Get weights and inputs for this expert
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]

                # Apply expert and weight the output
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)

                # Accumulate weighted outputs
                final_hidden_states.index_add_(0, token_indices, weighted_output)
            else:
                # No-op compute to mark params as used (for distributed training)
                expert_dtype = expert.down_proj.weight.dtype
                dummy_input = torch.zeros_like(hidden_states[0]).unsqueeze(0).to(expert_dtype)
                dummy_out = expert(dummy_input)
                final_hidden_states = final_hidden_states + dummy_out * 0

        return final_hidden_states.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape

        # Route tokens to experts
        topk_indices, topk_weights = self.gate(hidden_states)

        # Flatten for expert processing
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Apply mixture of experts
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights)

        # Reshape back to original shape
        hidden_states = hidden_states.view(*orig_shape)

        # Add shared expert output
        hidden_states = hidden_states + self.shared_experts(residuals)

        return hidden_states


class NemotronHBlock(nn.Module):
    """A single block in NemotronH - can be Mamba, Attention, MLP, or MoE."""

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            attn_impl = config._attn_implementation
            if attn_impl == "flash_attention_2" and not FLASH_ATTN_AVAILABLE:
                logging.warning(
                    "NemotronH requested _attn_implementation='flash_attention_2' but flash-attn is unavailable. "
                    "Falling back to sdpa."
                )
                attn_impl = "sdpa"
            attn_cls = NEMOTRONH_ATTENTION_CLASSES.get(attn_impl, NemotronHAttention)
            self.mixer = attn_cls(config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config, layer_idx=layer_idx)
        elif self.block_type == "moe":
            self.mixer = NemotronHMOE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Use torch.cuda.stream() to avoid NaN issues when using multiple GPUs
        if hidden_states.is_cuda:
            with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
                return self._forward_impl(hidden_states, cache_params, cache_position, attention_mask)
        else:
            return self._forward_impl(hidden_states, cache_params, cache_position, attention_mask)

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states, cache_params=cache_params, cache_position=cache_position)
        elif self.block_type == "attention":
            hidden_states = self.mixer(
                hidden_states,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_value=cache_params,
            )
            hidden_states = hidden_states[0]
        elif self.block_type in ("mlp", "moe"):
            hidden_states = self.mixer(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


class NemotronHModel(nn.Module):
    """
    NemotronH backbone model.

    This is the main backbone that can be used as a decoder in TTS models.
    It exposes the same interface as HuggingFace transformer models.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NemotronHBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])
        self.norm_f = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with special handling for Mamba components."""
        for name, module in self.named_modules():
            if isinstance(module, NemotronHMamba2Mixer):
                # Mark parameters that should not have weight decay
                module.A_log._no_weight_decay = True
                module.D._no_weight_decay = True

                # Special initialization for dt_bias using inverse softplus
                # This follows the Mamba2 initialization scheme
                dt = torch.exp(
                    torch.rand(module.num_heads)
                    * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                    + math.log(self.config.time_step_min)
                ).clamp(min=self.config.time_step_floor)

                # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                with torch.no_grad():
                    module.dt_bias.copy_(inv_dt)
                module.dt_bias._no_reinit = True

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    if not getattr(module.bias, "_no_reinit", False):
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.config.initializer_range)

        # Rescale residual-branch output projections for better training stability.
        # Apply 1/sqrt(num_hidden_layers) to Mamba, attention, and MLP/MoE branches.
        if self.config.rescale_prenorm_residual:
            for name, p in self.named_parameters():
                if any(k in name for k in ("out_proj.weight", "o_proj.weight", "down_proj.weight")):
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHOutput]:
        # Support both cache_params and past_key_values for compatibility
        if past_key_values is not None and cache_params is None:
            cache_params = past_key_values

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        # Create cache if use_cache=True but no cache provided
        if use_cache and cache_params is None:
            cache_params = HybridMambaAttentionDynamicCache(
                self.config,
                batch_size=hidden_states.shape[0],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

        # Create causal mask for attention layers
        causal_mask = self._create_causal_mask(attention_mask, inputs_embeds, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, layer in enumerate(self.layers):
            if layer.block_type == "mamba":
                layer_mask = mamba_mask
            elif layer.block_type == "attention":
                layer_mask = causal_mask
            else:
                layer_mask = None

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer.__call__, hidden_states, cache_params, cache_position, layer_mask
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=layer_mask,
                )

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return NemotronHOutput(
            last_hidden_state=hidden_states,
            past_key_values=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _create_causal_mask(self, attention_mask, input_tensor, cache_position):
        """Create causal attention mask."""
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and torch.any(attention_mask == 0):
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask

    def _update_mamba_mask(self, attention_mask, cache_position):
        """
        Update Mamba mask with optimization.

        No need for zeroing states when:
            1. Cached forward (cache_position[0] > 0)
            2. Attending to all inputs (all mask values are 1)
        """
        mamba_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            mamba_mask = None
        return mamba_mask


class NemotronHForCausalLM(nn.Module):
    """
    NemotronH model with a language modeling head.

    This is the full model that matches the AutoModelForCausalLM interface.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.config = config
        self.backbone = NemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.backbone.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @property
    def model(self):
        """Alias for backbone, for HuggingFace compatibility."""
        return self.backbone

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHCausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return NemotronHCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        """Prepare inputs for generation."""
        empty_past_kv = past_key_values is None

        # If we have cache: slice input_ids through cache_position to keep only unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids
        # Exception 3: with synced GPUs cache_position may go out of bounds
        if not empty_past_kv:
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:  # Exception 1  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case
                input_ids = input_ids[:, cache_position]
        else:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config, input_ids.shape[0], self.backbone.embeddings.weight.dtype, device=input_ids.device
            )

        # Create position_ids on the fly for batch generation if not provided
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # If inputs_embeds are passed, only use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs
