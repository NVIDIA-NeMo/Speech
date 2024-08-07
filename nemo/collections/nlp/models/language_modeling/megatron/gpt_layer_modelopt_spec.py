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

try:
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.custom_layers.transformer_engine import TENorm
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError) as e:

    from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

    TransformerLayer = TransformerLayerSubmodules = ApexGuardDefaults
    MLP = MLPSubmodules = ModuleSpec = IdentityOp = ApexGuardDefaults
    AttnMaskType = DotProductAttention = TENorm = ApexGuardDefaults
    ColumnParallelLinear = RowParallelLinear = SelfAttention = SelfAttentionSubmodules = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False
    IMPORT_ERROR = e


# Use this spec for Model Optimizer PTQ and TensorRT-LLM export
def get_gpt_layer_modelopt_spec(num_experts: int = None) -> ModuleSpec:
    """Mix the native spec with TENorm.

    This is essentially the native local spec except for the layernorm implementation
    is using TENorm from Transformer-Engine. This TENorm supports both FusedLayerNorm and RMSNorm and
    prevents the apex dependency.
    """
    if not HAVE_MEGATRON_CORE:
        raise Exception(IMPORT_ERROR)

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=_get_mlp_module_spec(num_experts=num_experts),
            mlp_bda=get_bias_dropout_add,
            # Map TE-layernorm-fusion keys back
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                **({'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_'} if num_experts is None else {}),
            },
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(num_experts: int = None, moe_grouped_gemm: bool = False) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return ModuleSpec(
            module=MoELayer,
            submodules=(
                MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                )
                if not moe_grouped_gemm
                else None
            ),
        )
