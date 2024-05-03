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

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from nemo.collections.nlp.models.language_modeling.megatron.griffin.griffin_layer_spec import (
    griffin_mqa_layer_with_transformer_engine_spec,
    griffin_recurrent_layer_with_transformer_engine_spec,
)


def get_griffin_layers(num_layers):
    dict_spec = {
        "Recurrent_Layer": griffin_recurrent_layer_with_transformer_engine_spec,
        "Attention_Layer": griffin_mqa_layer_with_transformer_engine_spec,
    }

    griffin_layers = []
    for i in range(num_layers):
        if i % 3 == 2:
            griffin_layers.append(dict_spec["Attention_Layer"])
        else:
            griffin_layers.append(dict_spec["Recurrent_Layer"])

    return griffin_layers


def create_block(
    config, layer_spec, layer_idx,
):
    block = build_module(layer_spec, config,)
    block.layer_number = layer_idx + 1
    return block


class GriffinStack(LanguageModule):
    def __init__(
        self, config: TransformerConfig,
    ):

        super().__init__(config)
        self.config = config
        self.griffin_layers = get_griffin_layers(self.config.num_layers)

        self.layers = nn.ModuleList(
            [create_block(self.config, layer_spec, layer_idx=i,) for i, layer_spec in enumerate(self.griffin_layers)]
        )
        self.final_layernorm = TENorm(
            config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon,
        )

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):

        for layer in self.layers:

            hidden_states, _ = layer(hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
