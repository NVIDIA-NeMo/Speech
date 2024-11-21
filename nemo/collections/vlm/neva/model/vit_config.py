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

from nemo.collections.llm.fn.activation import openai_gelu, quick_gelu

from nemo.collections.vlm.neva.model.base import CLIPViTConfig


@dataclass
class CLIPViTL_14_336_Config(CLIPViTConfig):
    """Clip vit large patch14 config"""

    vision_model_type = "clip"
    patch_dim = 14
    img_h = 336
    img_w = 336
    num_layers = 24
    num_attention_heads = 16
    add_bias_linear = True
    add_qkv_bias = True
    hidden_size = 1024
    hidden_dropout = 0.0
    attention_dropout = 0.0
    ffn_hidden_size = 4096
    gated_linear_unit = False
    activation_func = quick_gelu
    kv_channels = 64
    num_query_groups = 16
    layernorm_zero_centered_gamma = False
    apply_query_key_layer_scaling = False
    bias_activation_fusion = False
    bias_dropout_fusion = False
    attention_softmax_in_fp32 = True
    normalization = 'LayerNorm'
    apply_rope_fusion = False


@dataclass
class SigLIPViT400M_14_384_Config(CLIPViTConfig):
    """Siglip so400m patch14 384 config"""

    vision_model_type = "siglip"
    patch_dim = 14
    img_h = 384
    img_w = 384
    num_layers = 27
    num_attention_heads = 16
    add_bias_linear = True
    add_qkv_bias = True
    hidden_size = 1152
    hidden_dropout = 0.0
    attention_dropout = 0.0
    ffn_hidden_size = 4304
    gated_linear_unit = False
    activation_func = openai_gelu
    kv_channels = 72
    num_query_groups = 16
    layernorm_zero_centered_gamma = False
    apply_query_key_layer_scaling = False
    bias_activation_fusion = False
    bias_dropout_fusion = False
    attention_softmax_in_fp32 = True
    normalization = 'LayerNorm'
    apply_rope_fusion = False
    qk_layernorm = False
    layernorm_epsilon = 1e-6
