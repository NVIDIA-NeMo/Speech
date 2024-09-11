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
from torch import Tensor, nn

from nemo.collections.nlp.models.language_modeling.megatron.griffin.griffin_layer_spec import (
    griffin_mqa_layer_with_transformer_engine_spec,
    griffin_recurrent_layer_with_transformer_engine_spec,
)
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.extensions.transformer_engine import TENorm, te_checkpoint
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False


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
    config,
    layer_spec,
    layer_idx,
):
    block = build_module(
        layer_spec,
        config,
    )
    block.layer_number = layer_idx + 1
    return block


class GriffinStack(LanguageModule):
    def __init__(
        self,
        config: TransformerConfig,
    ):

        super().__init__(config)
        self.config = config
        self.griffin_layers = get_griffin_layers(self.config.num_layers)

        self.layers = nn.ModuleList(
            [
                create_block(
                    self.config,
                    layer_spec,
                    layer_idx=i,
                )
                for i, layer_spec in enumerate(self.griffin_layers)
            ]
        )
        self.final_layernorm = TENorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.num_layers = len(self.layers)

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=None,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    packed_seq_params,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    packed_seq_params,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                hidden_states, context = checkpoint_handler(custom(l, l + self.config.recompute_num_layers))

                l += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for l in range(self.num_layers):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if l >= recompute_skip_num_layers and l < self.config.recompute_num_layers + recompute_skip_num_layers:
                    hidden_states, context = checkpoint_handler(custom(l, l + 1))
                else:
                    hidden_states, context = custom(l, l + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        packed_seq_params,
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):

        if (
            self.config.recompute_granularity == 'full'
            and self.training
            and not self.config.activations_checkpoint_recurrent
        ):
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
            )
        else:
            for layer in self.layers:

                hidden_states, _ = layer(hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
