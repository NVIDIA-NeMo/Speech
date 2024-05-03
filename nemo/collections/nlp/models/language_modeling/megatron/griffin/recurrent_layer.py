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
from typing import Union

from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor
from torch import Tensor


@dataclass
class RecurrentBlockSubmodules:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    recurrent_layer: Union[ModuleSpec, type] = IdentityOp
    recurrent_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp


class RecurrentBlock(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: RecurrentBlockSubmodules,
        layer_idx=None,
        residual_in_fp32=False,
        **kwargs,
    ):
        """
        Top level Mamba Layer
        """
        super().__init__(config)
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = config.hidden_dropout

        self.input_layernorm = build_module(submodules.input_layernorm, dim=self.config.hidden_size)

        self.recurrent_layer = build_module(
            submodules.recurrent_layer,
            self.config,
            width=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            lru_width=self.config.hidden_size,
            conv1d_temporal_width=4,
            final_w_init_variance_scale=1.0,
        )

        self.recurrent_bda = build_module(submodules.recurrent_bda)

        self.pre_mlp_layernorm = build_module(submodules.pre_mlp_layernorm, dim=self.config.hidden_size)

        self.mlp = build_module(submodules.mlp, config=self.config)

        self.mlp_bda = build_module(submodules.mlp_bda)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, inference_params=None, **kwargs):

        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Reccurent block.
        recurrent_output_with_bias = self.recurrent_layer(input_layernorm_output)

        hidden_states = self.recurrent_bda(self.training, self.config.bias_dropout_fusion)(
            recurrent_output_with_bias, residual, self.hidden_dropout
        )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
            mlp_output_with_bias, residual, self.hidden_dropout
        )

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
