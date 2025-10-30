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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Union

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor
from torch import Tensor, nn

from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_hybrid_layer_allocation import Symbols as LayerSymbols
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_hybrid_layer_allocation import allocate_layers

try:
    from megatron.core.extensions.transformer_engine import TEDelayedScaling, TENorm, te_checkpoint

    HAVE_TE = True
    LayerNormImpl = TENorm

except ImportError:
    HAVE_TE = False

    try:
        from apex.normalization import FusedLayerNorm

        LayerNormImpl = FusedLayerNorm

    except ImportError:
        from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

        LayerNormImpl = WrappedTorchLayerNorm


HYENA_LAYER_MAP = {
    LayerSymbols.HYENA_SHORT: "hyena_short_conv",
    LayerSymbols.HYENA_MEDIUM: "hyena_medium_conv",
    LayerSymbols.HYENA: "hyena",
}


@dataclass
class HyenaStackSubmodules:
    """
    A class for the module specs for the HyenaStack.
    """

    hyena_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp


class HyenaStack(MegatronModule):
    """
    A class for the HyenaStack.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        hyena_config: HyenaConfig,
        hybrid_override_pattern,
        max_sequence_length,
        submodules: HyenaStackSubmodules,
        pre_process: bool = True,
        post_process: bool = True,
        post_layer_norm: bool = False,
        pg_collection=None,
    ) -> None:

        super().__init__(config=transformer_config)

        self.transformer_config = transformer_config
        self.hyena_config = hyena_config
        self.submodules = submodules
        self.hybrid_override_pattern = hybrid_override_pattern
        self.pre_process = pre_process
        self.post_process = post_process
        self.post_layer_norm = post_layer_norm
        self.pg_collection = pg_collection

        # Required for pipeline parallel schedules
        self.input_tensor = None

        layer_type_list = allocate_layers(self.transformer_config.num_layers, self.hybrid_override_pattern)

        pp_layer_offset = 0
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            pp_layer_offset, layer_type_list = self._select_layers_for_pipeline_parallel(layer_type_list)

        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(layer_type_list):
            if layer_type in HYENA_LAYER_MAP:
                layer = build_module(
                    submodules.hyena_layer,
                    self.transformer_config,
                    self.hyena_config,
                    operator_type=HYENA_LAYER_MAP.get(layer_type),
                    max_sequence_length=max_sequence_length,
                    layer_number=i + 1 + pp_layer_offset,
                    pg_collection=self.pg_collection,
                )
            elif layer_type == LayerSymbols.ATTENTION:
                # Transformer layers apply their own pp_layer_offset
                layer = build_module(
                    submodules.attention_layer,
                    config=self.transformer_config,
                    layer_number=i + 1,
                    pg_collection=self.pg_collection,
                )
            else:
                assert True, "unexpected layer_type"
            self.layers.append(layer)

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_norm = TENorm(
                config=self.transformer_config,
                hidden_size=self.transformer_config.hidden_size,
                eps=self.transformer_config.layernorm_epsilon,
            )
        # Required for activation recomputation
        self.num_layers_per_pipeline_rank = len(self.layers)

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def _select_layers_for_pipeline_parallel(self, layer_type_list):
        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
        num_layers_per_pipeline_rank = (
            self.transformer_config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
        )

        assert getattr(self.transformer_config, 'virtual_pipeline_model_parallel_size', None) is None, (
            "The Hyena hybrid model does not currently support " "virtual/interleaved pipeline parallelism"
        )

        offset = pipeline_rank * num_layers_per_pipeline_rank
        selected_list = layer_type_list[offset : offset + num_layers_per_pipeline_rank]

        return offset, selected_list

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(hidden_states, attention_mask, context, context_mask, rotary_pos_emb):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        attention_bias=attention_bias,
                        inference_context=None,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
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
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                upper_layer_idx = min(layer_idx + self.config.recompute_num_layers, self.num_layers_per_pipeline_rank)
                hidden_states, context = checkpoint_handler(custom(layer_idx, upper_layer_idx))
                new_n_layers = upper_layer_idx - layer_idx
                layer_idx += new_n_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states, attention_mask, context, context_mask, rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """Forward pass for the HyenaStack."""
        inference_context = deprecate_inference_params(inference_context, inference_params)
        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.transformer_config.fp8:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.transformer_config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.transformer_config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.transformer_config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.transformer_config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=False, tp_only_amax_red=self.transformer_config.tp_only_amax_red
                )
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        with fp8_context, rng_context:

            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            else:
                for layer in self.layers:
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        inference_context=inference_context,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                    )

            # The attention layer (currently a simplified transformer layer)
            # outputs a tuple of (hidden_states, context). Context is intended
            # for cross-attention, and is not needed in our model.
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)
        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """
        Returns a sharded state dictionary for the current object.

        This function constructs a sharded state dictionary by iterating over the layers
        in the current object, computing the sharded state dictionary for each layer,
        and combining the results into a single dictionary.

        Parameters:
            prefix (str): The prefix to use for the state dictionary keys.
            sharded_offsets (tuple): The sharded offsets to use for the state dictionary.
            metadata (dict): Additional metadata to use when computing the sharded state dictionary.

        Returns:
            dict: The sharded state dictionary for the current object.
        """

        sharded_state_dict = {}
        layer_prefix = f'{prefix}layers.'

        for local_layer_idx, layer in enumerate(self.layers):

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{local_layer_idx}.'  # module list index in HyenaBlock

            sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(state_dict_prefix, sharded_pp_offset, metadata)

            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata)
                )

        return sharded_state_dict
