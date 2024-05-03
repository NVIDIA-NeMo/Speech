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

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from nemo.collections.nlp.models.language_modeling.megatron.griffin.recurrent_layer import (
    RecurrentBlock,
    RecurrentBlockSubmodules,
)
from nemo.collections.nlp.models.language_modeling.megatron.griffin.recurrent_module import (
    RGLRU,
    Conv1D,
    RecurrentLayer,
    RecurrentLayerSubmodules,
)

griffin_mqa_layer_with_transformer_engine_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear,),
        ),
        mlp_bda=get_bias_dropout_add,
    ),
)

griffin_recurrent_layer_with_transformer_engine_spec = ModuleSpec(
    module=RecurrentBlock,
    submodules=RecurrentBlockSubmodules(
        recurrent_layer=ModuleSpec(
            module=RecurrentLayer,
            submodules=RecurrentLayerSubmodules(
                linear_in=TELayerNormColumnParallelLinear,
                linear_out=TERowParallelLinear,
                conv_1d=Conv1D,
                rg_lru=RGLRU,
            ),
        ),
        recurrent_bda=get_bias_dropout_add,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear,),
        ),
        mlp_bda=get_bias_dropout_add,
    ),
)
