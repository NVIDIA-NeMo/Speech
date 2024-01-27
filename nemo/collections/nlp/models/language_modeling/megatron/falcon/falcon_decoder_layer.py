# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

try:
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
    from megatron.core.utils import make_viewless_tensor

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

    TransformerLayer = ApexGuardDefaults
    TransformerConfig = ApexGuardDefaults
    TransformerLayerSubmodules = ApexGuardDefaults
    AttnMaskType = ApexGuardDefaults()

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class FalconTransformerLayer(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
        
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        self_attn_mask_type=AttnMaskType.padding,
    ):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)

        if hasattr(self.config, 'new_decoder_architecture'):
            self.new_decoder_architecture = self.config.new_decoder_architecture
        else:
            self.new_decoder_architecture = None
        if hasattr(self.config, 'parallel_attention'):
            self.parallel_attention = self.config.parallel_attention
        else:
            self.parallel_attention = None

        if self.new_decoder_architecture or self.parallel_attention:
            self.post_self_attn_layernorm = None
        else:
            self.post_self_attn_layernorm = build_module(
                submodules.post_self_attn_layernorm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        if self.new_decoder_architecture:
            self.pre_mlp_layernorm = build_module(
                submodules.pre_mlp_layernorm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.pre_mlp_layernorm = None

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        mlp_ln_output = None
        if self.new_decoder_architecture:
            mlp_ln_output = self.pre_mlp_layernorm(hidden_states)

        input_layernorm_output = self.input_layernorm(hidden_states)

        input_mlp_ln = input_layernorm_output

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.config.hidden_dropout
            )

        if not self.new_decoder_architecture:
            if self.parallel_attention:
                layernorm_output = input_mlp_ln
            else:
                residual = hidden_states
                layernorm_output = self.post_self_attn_layernorm(hidden_states)

        else:
            layernorm_output = mlp_ln_output

        mlp_output_with_bias = self.mlp(layernorm_output)

        # falcon specific:
        if self.new_decoder_architecture or self.parallel_attention:
            mlp_output = mlp_output_with_bias[0]
            attn_output = attention_output_with_bias[0]
            mlp_output_without_bias = mlp_output + attn_output
            mlp_output_with_bias = (mlp_output_without_bias, None)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.config.hidden_dropout
            )

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, context
