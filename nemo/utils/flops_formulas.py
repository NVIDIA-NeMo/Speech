# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

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
    img_seq_len: Optional[int] = None
    img_h: Optional[int] = None
    img_w: Optional[int] = None
    in_channels: Optional[int] = None
    patch_dim: Optional[int] = None
    class_token_len: Optional[int] = None
    projector_type: Optional[str] = None
    inp_s: Optional[int] = None


def gpt3(config: FLOPSConfig):
    """Model FLOPs for GPT3 family"""

    vocab_size = LLM_VOCAB_SIZE_MAP["gpt3"]

    return (
        24 * config.gbs * config.enc_seq_len * config.hs * config.hs
        + 4 * config.gbs * config.enc_seq_len * config.enc_seq_len * config.hs
    ) * (3 * config.layers) + (6 * config.gbs * config.enc_seq_len * config.hs * vocab_size)


def llama2(config: FLOPSConfig):
    """Model FLOPs for llama2 family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["llama2"]

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
            + (12 * config.enc_seq_len / config.hs)
            + (6 * vocab_size / (config.layers * config.hs))
        )
    )


def llama3(config: FLOPSConfig):
    """Model FLOPs for llama3 family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["llama3"]

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
            + (12 * config.enc_seq_len / config.hs)
            + (6 * vocab_size / (config.layers * config.hs))
        )
    )


def nemotron(config: FLOPSConfig):
    """Model FLOPs for nemotron family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["nemotron"]

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
            + (12 * config.enc_seq_len / config.hs)
            + (6 * vocab_size / (config.layers * config.hs))
        )
    )


def mixtral(config: FLOPSConfig):
    """Model FLOPs for mixtral family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["mixtral"]

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
            + (12 * config.enc_seq_len / config.hs)
            + (6 * vocab_size / (config.layers * config.hs))
        )
    )


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
