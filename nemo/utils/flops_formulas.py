# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from nemo.collections.common.parts.perf_metrics_utils import LLM_VOCAB_SIZE_MAP


@dataclass
class FLOPSConfig:
    """Contains the model hparams needed for FLOPS computations"""

    gbs: int
    enc_seq_len: Optional[int] = None
    hs: Optional[int] = None
    layers: Optional[int] = None
    ffn_hs: Optional[int] = None
    attention_heads: Optional[int] = None
    moe_router_topk: Optional[int] = None
    query_groups: Optional[int] = None
    kv_channels: Optional[int] = None
    img_seq_len: Optional[int] = None
    img_h: Optional[int] = None
    img_w: Optional[int] = None
    in_channels: Optional[int] = None
    patch_dim: Optional[int] = None
    class_token_len: Optional[int] = None
    projector_type: Optional[str] = None
    inp_s: Optional[int] = None
    model_pattern: Optional[str] = None
    vocab_size: Optional[int] = None
    model_channels: Optional[int] = None
    vec_in_dim: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_head_dim: Optional[int] = None
    qk_pos_emb_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    moe_layer_freq: Union[int, List[int]] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_ffn_hidden_size: Optional[int] = None
    mtp_num_layers: Optional[int] = None
    causal_self_attn: Optional[bool] = None
    is_hybrid_model: bool = False
    hybrid_override_pattern: Optional[str] = None
    mamba_state_dim: Optional[int] = None
    mamba_head_dim: Optional[int] = None
    mamba_num_groups: Optional[int] = None
    mamba_num_heads: Optional[int] = None
    # SWA configs
    window_attn_skip_freq: Optional[Union[int, List[int]]] = None
    window_size: Optional[Tuple[int, int]] = (128, 0)


def gpt3(config: FLOPSConfig):
    """Model FLOPs for GPT3 family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["gpt3"]
    causal_self_attn = True

    return (
        24 * config.gbs * config.enc_seq_len * config.hs * config.hs
        + 4 * config.gbs * config.enc_seq_len * config.enc_seq_len * config.hs * (0.5 if causal_self_attn else 1)
    ) * (3 * config.layers) + (6 * config.gbs * config.enc_seq_len * config.hs * vocab_size)


def llama2(config: FLOPSConfig):
    """Model FLOPs for llama2 family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["llama2"]
    causal_self_attn = True

    return (
        config.gbs
        * config.enc_seq_len
        * config.layers
        * config.hs
        * config.hs
        * (
            12
            + (12 * config.query_groups / config.attention_heads)
            + (18 * config.ffn_hs / config.hs)
            + (12 * config.enc_seq_len / config.hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (config.layers * config.hs))
        )
    )


def llama3(config: FLOPSConfig):
    """Model FLOPs for llama3 family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["llama3"]
    causal_self_attn = True

    return (
        config.gbs
        * config.enc_seq_len
        * config.layers
        * config.hs
        * config.hs
        * (
            12
            + (12 * config.query_groups / config.attention_heads)
            + (18 * config.ffn_hs / config.hs)
            + (12 * config.enc_seq_len / config.hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (config.layers * config.hs))
        )
    )


def nemotron(config: FLOPSConfig):
    """Model FLOPs for nemotron family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["nemotron"]
    causal_self_attn = True

    return (
        config.gbs
        * config.enc_seq_len
        * config.layers
        * config.hs
        * config.hs
        * (
            12
            + (12 * config.query_groups / config.attention_heads)
            + (12 * config.ffn_hs / config.hs)
            + (12 * config.enc_seq_len / config.hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (config.layers * config.hs))
        )
    )


def mixtral(config: FLOPSConfig):
    """Model FLOPs for mixtral family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["mixtral"]
    causal_self_attn = True

    return (
        config.gbs
        * config.enc_seq_len
        * config.layers
        * config.hs
        * config.hs
        * (
            12
            + (12 * config.query_groups / config.attention_heads)
            + (18 * config.moe_router_topk * config.ffn_hs / config.hs)
            + (12 * config.enc_seq_len / config.hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (config.layers * config.hs))
        )
    )


def qwen3(config: FLOPSConfig):
    """Model FLOPs for Qwen3 family"""
    causal_self_attn = True
    seq_len = config.enc_seq_len
    hidden_size = config.hs
    gated_linear_multiplier = 2
    query_projection_to_hidden_size_ratio = config.kv_channels * config.attention_heads / hidden_size

    # attention flops for GQA
    attention_flops = (
        3
        * 2
        * config.gbs
        * config.layers
        * seq_len
        * hidden_size
        * hidden_size
        * query_projection_to_hidden_size_ratio
        * (
            (config.query_groups / config.attention_heads * 2 + 1)  # QKV gemm
            + (seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1))  # attention
            + 1  # attention proj gemm
        )
    )

    mlp_ffn_hidden_size = config.ffn_hs
    if hasattr(config, "moe_ffn_hidden_size") and config.moe_ffn_hidden_size is not None:
        mlp_ffn_hidden_size = config.moe_ffn_hidden_size * config.moe_router_topk

    # mlp flops
    mlp_flops = (
        3
        * 2
        * config.gbs
        * config.layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * mlp_ffn_hidden_size
    )

    # vocab flops
    vocab_flops = 3 * 2 * config.gbs * seq_len * hidden_size * config.vocab_size

    return attention_flops + mlp_flops + vocab_flops


def bert(config: FLOPSConfig):
    """Model FLOPs for BERT family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["bert"]

    return (
        72
        * config.gbs
        * config.layers
        * config.enc_seq_len
        * config.hs
        * config.hs
        * (1 + (config.enc_seq_len / (6 * config.hs)) + (vocab_size / (12 * config.hs * config.layers)))
    )


def transformer(config: FLOPSConfig):
    """Calculate FLOPs for a standard Transformer model.
    Note: This does not cover encoder-decoder models.
    """
    # Extract parameters from config
    batch_size = config.gbs
    hidden_size = config.hs
    seq_length = config.enc_seq_len
    num_layers = config.layers
    num_attention_heads = config.attention_heads
    ffn_hidden_size = config.ffn_hs
    vocab_size = config.vocab_size

    if vocab_size is None:
        raise ValueError("vocab_size is required for transformer FLOPs calculation")

    # Handle optional parameters with reasonable defaults
    query_groups = config.query_groups if config.query_groups is not None else num_attention_heads
    causal_self_attn = config.causal_self_attn if config.causal_self_attn is not None else False
    moe_router_topk = config.moe_router_topk if config.moe_router_topk is not None else 0
    kv_channels = hidden_size // num_attention_heads  # Standard dimension per head

    # Calculate query projection size and ratio
    query_projection_size = kv_channels * num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / hidden_size

    # MoE parameters - simplified for NeMo config
    # In this implementation, we assume all layers are dense if num_experts is None
    if moe_router_topk == 0:
        num_dense_layers = num_layers
        num_moe_layers = 0
        num_experts_routed_to = 0
    else:
        # Simplified MoE handling - assuming uniform distribution of MoE layers
        # This can be expanded based on NeMo's actual MoE implementation
        num_moe_layers = num_layers // 2  # Simplified assumption
        num_dense_layers = num_layers - num_moe_layers
        num_experts_routed_to = moe_router_topk

    # Handle SwiGLU vs standard GELU/ReLU
    # Default to standard activation (no SwiGLU)
    gated_linear_multiplier = 1

    # Define the expansion factor as described in the paper
    # 3x: Each GEMM needs forward pass, backward wgrad, and backward dgrad
    # 2x: GEMMs are stacked twice in standard Transformer architectures
    # 2x: A GEMM of m*n with n*k requires 2mnk floating-point operations
    expansion_factor = 3 * 2 * 2
    # Attention
    if not causal_self_attn:
        attention_component = (
            1
            + (query_groups / num_attention_heads)
            # Only half of the attention matrix is non-zero and needs to be multiplied with V
            + (seq_length / hidden_size)  # If causal self attn -> divide by 2.
        ) * query_projection_to_hidden_size_ratio
    else:
        attention_component = (
            1
            + (query_groups / num_attention_heads)
            # Only half of the attention matrix is non-zero and needs to be multiplied with V
            + (seq_length / hidden_size / 2)  # If causal self attn -> divide by 2.
        ) * query_projection_to_hidden_size_ratio

    # Calculate total FLOPs
    total_flops = (
        expansion_factor
        * batch_size
        * seq_length
        * num_layers
        * hidden_size
        * hidden_size
        * (
            attention_component
            # MLP component
            + (
                (
                    # Dense layers
                    (ffn_hidden_size * num_dense_layers)
                    +
                    # MoE layers
                    (
                        (
                            # Routed experts
                            ffn_hidden_size
                            * num_experts_routed_to
                            # Note: Shared experts are not implemented in this version
                        )
                        * num_moe_layers
                    )
                )
                * gated_linear_multiplier
                / (num_layers * hidden_size)
            )
            # Logit component
            + (vocab_size / (2 * num_layers * hidden_size))
        )
    )

    return total_flops


def clip_vit_l(config: FLOPSConfig):
    """Model FLOPs for CLIP ViT"""

    if config.img_seq_len is None:
        config.img_seq_len = (config.img_h * config.img_w) / (
            config.patch_dim * config.patch_dim
        ) + config.class_token_len
    return config.gbs * config.layers * config.hs * config.hs * config.img_seq_len * (
        24 + (4 * config.img_seq_len / config.hs)
    ) + (2 * config.gbs * config.hs * config.in_channels * config.img_h * config.img_w)


def neva_projection(config: FLOPSConfig):
    """Model FLOPs for NeVA Projection"""

    if "mlp" in config.projector_type:
        return 6 * config.gbs * config.img_seq_len * config.ffn_hs * (config.inp_s + config.hs)
    elif config.projector_type == "affine":
        return 6 * config.gbs * config.img_seq_len * config.inp_s * config.hs
    else:
        raise ValueError(
            f"NeVA Projections FLOPs calculator only supports 'mlp', 'mcore_mlp'"
            f" or 'affine' projector_type but found {config.projector_type}"
        )


def flux(config: FLOPSConfig):
    """Model FLOPs for FLUX"""

    hs = config.hs
    seq_len = config.model_channels + config.inp_s
    base_factor = 6 * config.gbs  # common multiplier for most terms

    # Joint layer computations
    joint_layer_flops = (
        base_factor
        * config.layers[0]
        * (
            10 * hs * hs  # hidden size operations
            + 2 * hs * (config.model_channels + config.inp_s) * (1 + hs * 7)  # channel and context joint attention
            + 2 * (config.model_channels + config.inp_s) * hs  # final projection
        )
    )

    # Single layer computations
    single_layer_flops = (
        base_factor
        * config.layers[1]
        * seq_len
        * hs
        * (
            3  # linear Y
            + 1  # Modulation
            + 4 * hs  # Linear computations
            + (3 * hs + 2 * seq_len)  # attention operations
            + 5 * hs  # feed-forward
            + 1  # Modulation
        )
    )

    # Embedding and projection layers
    other_flops = base_factor * (
        config.inp_s * config.in_channels * hs  # image embedding
        + config.inp_s * hs * config.model_channels  # text embedding
        + config.vec_in_dim * hs
        + hs * hs  # vector embedding
        + 2 * (config.model_channels * hs + hs * hs)  # guidance + timestep embedding
        + (config.inp_s * config.in_channels * hs) / config.gbs  # final projection
    )

    return joint_layer_flops + single_layer_flops + other_flops


def deepseekv3(config: FLOPSConfig):
    """Model FLOPs for DeepSeek V3"""

    # self-attention flops
    bmm1_flops = (
        0.5 * (config.qk_head_dim + config.qk_pos_emb_head_dim) * config.attention_heads * (config.enc_seq_len**2)
    )
    bmm2_flops = 0.5 * config.v_head_dim * config.attention_heads * (config.enc_seq_len**2)
    per_input_attention_flops = 6 * (bmm1_flops + bmm2_flops) * config.layers
    if config.mtp_num_layers is not None:
        per_input_attention_flops += 6 * (bmm1_flops + bmm2_flops) * config.mtp_num_layers

    # linear layer flops
    per_layer_mla_params = config.hs * config.q_lora_rank + config.q_lora_rank * (
        (config.qk_head_dim + config.qk_pos_emb_head_dim) * config.attention_heads
    )  # Q
    per_layer_mla_params += config.hs * config.qk_pos_emb_head_dim  # K^R
    per_layer_mla_params += config.hs * config.kv_lora_rank + config.kv_lora_rank * (
        (config.qk_head_dim + config.v_head_dim) * config.attention_heads
    )  # K^C and V^C
    per_layer_mla_params += config.v_head_dim * config.attention_heads * config.hs  # Proj
    mla_params = per_layer_mla_params * config.layers
    if config.mtp_num_layers is not None:
        mla_params += per_layer_mla_params * config.mtp_num_layers

    dense_layer_ffn_params = config.hs * config.ffn_hs * 3  # gated linear unit
    per_shared_expert_params = config.hs * config.moe_shared_expert_intermediate_size * 3
    per_selected_expert_params = config.hs * config.moe_ffn_hidden_size * 3
    ffn_params = 0

    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.layers)]
    else:
        moe_layer_pattern = config.moe_layer_freq
    for i in moe_layer_pattern:
        if i == 0:
            ffn_params += dense_layer_ffn_params
        else:
            ffn_params += per_shared_expert_params + (per_selected_expert_params * config.moe_router_topk)
    if config.mtp_num_layers is not None:
        for i in range(config.mtp_num_layers):
            ffn_params += per_shared_expert_params + (per_selected_expert_params * config.moe_router_topk)
    per_input_params = mla_params + ffn_params
    per_input_linear_flops = 6 * per_input_params * config.enc_seq_len

    # vocab flops
    per_input_vocab_flops = 6 * config.vocab_size * config.hs * config.enc_seq_len
    if config.mtp_num_layers is not None:
        for i in range(config.mtp_num_layers):
            per_input_vocab_flops += 6 * config.vocab_size * config.hs * config.enc_seq_len
            per_input_vocab_flops += 6 * config.hs * 2 * config.hs * config.enc_seq_len

    return (per_input_attention_flops + per_input_linear_flops + per_input_vocab_flops) * config.gbs


def _nemotronh_mlp_layer_flops(config: FLOPSConfig):
    """Model FLOPs for MLP layer. Assume gated linear unit."""
    return 6 * config.gbs * config.enc_seq_len * config.hs * config.ffn_hs * 3


def _non_mla_attn_layer_flops(config: FLOPSConfig):
    """Model FLOPs for attention layer"""
    return (
        6
        * config.gbs
        * config.enc_seq_len
        * config.hs
        * (
            config.hs  # Q
            + config.query_groups / config.attention_heads * config.hs * 2  # KV
            + config.enc_seq_len / 2 * 2
            + config.hs
        )
    )


def _mamba_layer_flops(config: FLOPSConfig):
    """Model FLOPs for Mamba layer. We ignore part of the flops of scan because the
    chunk size is not known from model config."""
    assert config.mamba_state_dim is not None
    assert config.mamba_head_dim is not None

    if config.mamba_num_heads:
        nheads = config.mamba_num_heads
    else:
        nheads = 2 * config.hs // config.mamba_head_dim  # default expand is 2
    d_in = nheads * config.mamba_head_dim
    return (
        (
            6
            * config.gbs
            * config.enc_seq_len
            * config.hs
            * (2 * d_in + 2 * config.mamba_num_groups * config.mamba_state_dim + nheads)
        )
        + (3 * 2 * config.gbs * config.enc_seq_len * d_in * config.mamba_state_dim)
        + (6 * config.gbs * config.enc_seq_len * d_in * config.hs)
    )


def _hybrid_model_flops(config: FLOPSConfig):
    """Model FLOPs for hybrid model"""
    assert config.is_hybrid_model == True
    assert config.hybrid_override_pattern is not None

    num_attn_layers, num_mamba_layers, num_mlp_layers = 0, 0, 0
    for c in config.hybrid_override_pattern:
        if c == 'M':
            num_mamba_layers += 1
        elif c == '-':
            num_mlp_layers += 1
        elif c == '*':
            num_attn_layers += 1
    return (
        num_attn_layers * _non_mla_attn_layer_flops(config)
        + num_mamba_layers * _mamba_layer_flops(config)
        + num_mlp_layers * _nemotronh_mlp_layer_flops(config)
        + 6 * config.gbs * config.enc_seq_len * config.hs * config.vocab_size
    )


def nemotronh(config: FLOPSConfig):
    """Model FLOPs for NemotronH"""
    return _hybrid_model_flops(config)


def attention_flops_calculator(
    seqlen,
    hidden_size,
    num_attention_heads,
    num_query_groups,
    kv_channels: Optional[int] = None,
    is_swa: bool = False,
    swa_window_size: int = 128,
):
    """Calculate the flops for the attention part."""
    kv_channels = kv_channels or (hidden_size // num_attention_heads)

    linear_qkv = seqlen * hidden_size * (kv_channels * (num_attention_heads + num_query_groups * 2))

    linear_proj = seqlen * hidden_size * (kv_channels * num_attention_heads)

    if is_swa:
        attention_mask_nz_elem = (
            swa_window_size * (swa_window_size + 1) / 2 + (seqlen - swa_window_size) * swa_window_size
        )
        attention = num_attention_heads * (attention_mask_nz_elem * kv_channels) * 2
    else:
        bmm_k = kv_channels
        bmm_b = num_attention_heads
        attention_mask_nz_elem = seqlen * (seqlen + 1) / 2
        attention = bmm_b * attention_mask_nz_elem * bmm_k * 2

    return (linear_qkv + linear_proj + attention) * 6


def moe_mlp_flops_calculator(
    seqlen,
    hidden_size,
    moe_ffn_hidden_size,
    moe_router_topk,
    gated_linear_unit: bool = True,
):
    """Calculate the flops for the MLP"""
    total_num_tokens = seqlen * moe_router_topk
    linear_fc1 = total_num_tokens * hidden_size * moe_ffn_hidden_size * (2 if gated_linear_unit else 1)
    linear_fc2 = total_num_tokens * moe_ffn_hidden_size * hidden_size
    return (linear_fc1 + linear_fc2) * 6


def loss_flops_calculator(
    seqlen,
    hidden_size,
    vocab_size,
):
    """Calculate the flops for the loss"""
    return (seqlen * hidden_size * vocab_size) * 6


def gpt_oss_flops_calculator(
    gbs,
    num_layers,
    seqlen,
    hidden_size,
    num_attention_heads,
    num_query_groups,
    moe_ffn_hidden_size,
    moe_router_topk,
    vocab_size,
    kv_channels: Optional[int] = None,
    swa_window_size: int = 128,
    window_attn_skip_freq: Optional[int] = 2,
):
    """Calculate the flops for the GPT-OSS model"""
    flops = 0
    for i in range(num_layers):
        if i % window_attn_skip_freq == 0:
            flops += attention_flops_calculator(
                seqlen,
                hidden_size,
                num_attention_heads,
                num_query_groups,
                kv_channels,
                is_swa=False,
            )
        else:
            flops += attention_flops_calculator(
                seqlen,
                hidden_size,
                num_attention_heads,
                num_query_groups,
                kv_channels,
                is_swa=True,
                swa_window_size=swa_window_size,
            )
        flops += moe_mlp_flops_calculator(
            seqlen,
            hidden_size,
            moe_ffn_hidden_size,
            moe_router_topk,
        )
    flops += loss_flops_calculator(seqlen, hidden_size, vocab_size)
    flops *= gbs
    return flops


def gpt_oss(config: FLOPSConfig):
    """Model FLOPs for GPT-OSS"""
    return gpt_oss_flops_calculator(
        gbs=config.gbs,
        num_layers=config.layers,
        seqlen=config.enc_seq_len,
        hidden_size=config.hs,
        num_attention_heads=config.attention_heads,
        num_query_groups=config.query_groups,
        moe_ffn_hidden_size=config.moe_ffn_hidden_size,
        moe_router_topk=config.moe_router_topk,
        vocab_size=config.vocab_size,
        kv_channels=config.kv_channels,
        swa_window_size=config.window_size[0] if config.window_size is not None else 128,
        window_attn_skip_freq=config.window_attn_skip_freq,
    )
