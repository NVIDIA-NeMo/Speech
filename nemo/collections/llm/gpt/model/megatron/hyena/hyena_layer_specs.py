# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from nemo.collections.llm.gpt.model.megatron.hyena.hyena_block import HyenaStack, HyenaStackSubmodules
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_layer import HyenaLayer, HyenaLayerSubmodules
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_mixer import HyenaMixer, HyenaMixerSubmodules

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

    from nemo.collections.llm.gpt.model.megatron.hyena.te_compat import (
        Linear,
        RMSNormLinear,
        RMSNormTELinearFp8,
        TELayerNormColumnParallelLinearFp8,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

    def _raise_te_import_error(*args, **kwargs):
        raise ImportError("Transformer Engine is not installed")

    # NeMo has a number of tests that make sure that you can initialize some modules without TE installed.
    TENorm = _raise_te_import_error
    TELayerNormColumnParallelLinear = _raise_te_import_error
    TERowParallelLinear = _raise_te_import_error
    TEDotProductAttention = _raise_te_import_error


def get_hyena_stack_spec(
    use_te=HAVE_TE,
    vortex_style_fp8=False,
    unfused_rmsnorm=False,
    plain_row_linear=False,
):
    """Construct desired Hyena stack spec based on given parameters."""
    if use_te:
        row_linear = TERowParallelLinear
        col_linear = TELayerNormColumnParallelLinear
        pre_layernorm = IdentityOp  # fused Norm+Linear, so no pre norm
        core_attention = TEDotProductAttention
    else:
        row_linear = RowParallelLinear
        col_linear = ColumnParallelLinear
        pre_layernorm = TENorm  # would raise error if attempt to use
        core_attention = DotProductAttention

    if unfused_rmsnorm:
        col_linear = RMSNormLinear

    if vortex_style_fp8:
        if unfused_rmsnorm:
            dense_projection = RMSNormTELinearFp8
        else:
            dense_projection = TELayerNormColumnParallelLinearFp8
    else:
        dense_projection = col_linear

    if plain_row_linear:
        row_linear = Linear

    return ModuleSpec(
        module=HyenaStack,
        submodules=HyenaStackSubmodules(
            hyena_layer=ModuleSpec(
                module=HyenaLayer,
                submodules=HyenaLayerSubmodules(
                    mixer=ModuleSpec(
                        module=HyenaMixer,
                        submodules=HyenaMixerSubmodules(dense_projection=dense_projection, dense=row_linear),
                    ),
                    hyena_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=pre_layernorm,
                    mlp=ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(linear_fc1=col_linear, linear_fc2=row_linear),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=pre_layernorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=col_linear,
                            core_attention=core_attention,
                            linear_proj=row_linear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=pre_layernorm,
                    mlp=ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(linear_fc1=col_linear, linear_fc2=row_linear),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
        ),
    )


hyena_stack_spec_no_te = get_hyena_stack_spec(use_te=False)  # for tests/debugging
