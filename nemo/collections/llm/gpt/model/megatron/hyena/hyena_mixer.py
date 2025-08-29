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

import logging
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from einops import rearrange
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default

from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import (
    B2BCausalConv1dModule,
    ParallelCausalDepthwiseConv1dWithState,
    ParallelHyenaOperator,
    ParallelShortHyenaOperator,
    divide,
)

logger = logging.getLogger(__name__)

try:
    from transformer_engine.common.recipe import DelayedScaling, Format
except ImportError:

    def DelayedScaling(*args, **kwargs):
        """Not imported: DelayedScaling. An error will be raised if this is called."""
        raise ImportError("transformer_engine not installed. Using default recipe.")

    def Format(*args, **kwargs):
        """Not imported: Format. An error will be raised if this is called."""
        raise ImportError("transformer_engine not installed. Using default recipe.")

    class _te:
        """If this dummy module is accessed, a not imported error will be raised."""

        def __getattribute__(self, name: str) -> None:
            """Not imported: te. An error will be raised if this is called like a module."""
            raise ImportError("transformer_engine not installed. Using default recipe.")

    te = _te()  # if a user accesses anything in this module, an error will be raised
    logger.warning("WARNING: transformer_engine not installed. Using default recipe.")

try:
    from subquadratic_ops.rearrange import rearrange as subquadratic_ops_rearrange
except ImportError:

    def subquadratic_ops_rearrange(*args, **kwargs):
        """Not imported: subquadratic_ops_rearrange. An error will be raised if this is called."""
        raise ImportError("subquadratic_ops not installed. subquadratic_ops_rearrange is not available.")


def set_format_recipe():
    """Set the fp8 format recipe. for Hyena."""
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    return fp8_recipe


@dataclass
class HyenaMixerSubmodules:
    """Contains the module specs for the input and output linear layers."""

    dense_projection: Union[ModuleSpec, type] = None
    dense: Union[ModuleSpec, type] = None


class HyenaMixer(MegatronModule):
    """A class for the HyenaMixer."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        hyena_config: HyenaConfig,
        max_sequence_length,
        submodules,
        layer_number=1,
        operator_type="H",
    ):

        super().__init__(transformer_config)
        self.transformer_config = transformer_config
        self.hyena_config = hyena_config
        self.operator_type = operator_type
        self.layer_number = layer_number
        self.grouped_attention = self.hyena_config.grouped_attention

        self.fast_conv_proj = self.hyena_config.fast_conv_proj
        self.fast_conv_mixer = self.hyena_config.fast_conv_mixer

        # Use b2b causal conv1d
        self.use_subquadratic_ops = self.transformer_config.use_subquadratic_ops

        # Per attention head and per partition values.
        assert torch.distributed.is_initialized()
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        world_size: int = get_tensor_model_parallel_world_size()

        # Width expansion for Hyena
        self.hyena_width_expansion = self.hyena_config.hyena_width_expansion

        # we might expand the hidden size for hyena
        self.input_size = self.transformer_config.hidden_size
        self.hidden_size = int(self.transformer_config.hidden_size * self.hyena_width_expansion)

        # ensures parallizable
        if self.hyena_width_expansion > 1:
            multiple_of = 32
            self.hidden_size = int(multiple_of * ((self.hidden_size + multiple_of - 1) // multiple_of))

        # checks on the hidden size divisibility
        assert (
            self.hidden_size % world_size == 0
        ), f"Hidden size {self.hidden_size} is not divisible by the world size {world_size}"
        self.hidden_size_per_partition = divide(self.hidden_size, world_size)
        self.proj_groups = self.hyena_config.proj_groups

        self.tie_projection_weights = self.hyena_config.tie_projection_weights

        self.grouped_proj_size = self.transformer_config.hidden_size // self.proj_groups

        # Strided linear layer.
        if self.tie_projection_weights:
            # we'll repeat the output 3 times instead
            projections_size = self.hidden_size
        else:
            projections_size = 3 * self.hidden_size

        # qkv projections
        self.dense_projection = build_module(
            submodules.dense_projection,
            self.input_size,
            projections_size,
            config=self.transformer_config,
            init_method=self.transformer_config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='fc1',
        )

        hyena_proj_groups = self.proj_groups if not self.grouped_attention else 1
        grouped_proj_size = self.hidden_size_per_partition // hyena_proj_groups

        self.hyena_proj_conv = ParallelCausalDepthwiseConv1dWithState(
            self.hidden_size_per_partition + 2 * grouped_proj_size,
            self.transformer_config,
            self.hyena_config,
            kernel_size=self.hyena_config.short_conv_L,
            init_method=transformer_config.init_method,
            bias=False,  # bias not currently supported (self.hyena_config.conv_proj_bias),
            use_fast_causal_conv=self.fast_conv_proj,
        )

        if self.operator_type == "hyena_short_conv":
            self.num_groups = self.hyena_config.num_groups_hyena_short
            self.num_groups_per_tp_rank = self.num_groups // self.model_parallel_size

            self.mixer = ParallelShortHyenaOperator(
                self.hidden_size,  # pass hidden size here to avoid recalculating
                self.transformer_config,
                self.hyena_config,
                self.transformer_config.init_method,
                short_conv_class=ParallelCausalDepthwiseConv1dWithState,
                use_fast_causal_conv=self.fast_conv_mixer,
                use_conv_bias=self.transformer_config.use_short_conv_bias,
            )

            if self.use_subquadratic_ops:
                # Create a wrapper module that doesn't register parameters
                # Use the existing weights from the original model
                self.b2b_kernel = B2BCausalConv1dModule(
                    self.hyena_proj_conv,
                    self.mixer,
                    operator_type=self.operator_type,
                )

        if self.operator_type in [
            "hyena",
            "hyena_medium_conv",
        ]:
            if self.operator_type == "hyena_medium_conv":
                self.num_groups = self.hyena_config.num_groups_hyena_medium
            else:
                self.num_groups = self.hyena_config.num_groups_hyena
            self.num_groups_per_tp_rank = self.num_groups // self.model_parallel_size

            # subquadratic_ops LI layer is handled internally in the ParallelHyenaOperator
            # by transformer_configs.use_subquadratic_ops
            self.mixer = ParallelHyenaOperator(
                self.hidden_size,  # pass hidden size here to avoid recalculating
                self.transformer_config,
                self.hyena_config,
                self.transformer_config.init_method,
                operator_type,
                max_sequence_length,
            )

            if self.use_subquadratic_ops and self.operator_type == "hyena_medium_conv":
                # Create a wrapper module that doesn't register parameters
                # Use the existing weights from the original model
                self.b2b_kernel = B2BCausalConv1dModule(
                    self.hyena_proj_conv,
                    self.mixer,
                    operator_type=self.operator_type,
                )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.dropout_p = self.transformer_config.attention_dropout
        self.attention_dropout = nn.Dropout(self.dropout_p)

        # When using non-parallel row linears, we allow PyTorch's Linear to
        # add bias: this is faster for TP=1 inference. For other cases (and
        # training), a more complex path is used, where bias is added as a
        # separate step.
        dense_skip_bias_add = not self.transformer_config.plain_row_linear

        self.dense = build_module(
            submodules.dense,
            self.hidden_size,
            self.input_size,
            config=self.transformer_config,
            init_method=self.transformer_config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=dense_skip_bias_add,
            is_expert=False,
            tp_comm_buffer_name='fc2',
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharded state dictionary for the HyenaMixer."""
        sharded_state_dict = {}
        # Submodules
        for name, module in self.named_children():
            if name != 'attention_dropout' and name != 'b2b_kernel':  # Don't register b2b_kernel (it's a wrapper)
                module_sharded_sd = sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata)

                sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict

    def forward(self, x, layer_past=None, inference_context=None, _hyena_use_cp=True):
        """Applies the Hyena sequence mixing operation to input embeddings.

        Args:
            x: Input tensor of shape [L, B, D] (seq_len, batch_size, hidden_dim)
            layer_past: Past layer state for inference (default: None)
            inference_context: Parameters for inference (default: None)
            _hyena_use_cp: Whether to use context parallelism (default: True)

        Returns:
            Tuple of (output tensor, bias)
        """
        # CP control
        if _hyena_use_cp:
            cp_group = get_context_parallel_group()
        else:
            cp_group = None

        if cp_group is not None and get_context_parallel_world_size() > 1:
            _proj_use_cp = True
        else:
            _proj_use_cp = False

        features, _ = self.dense_projection(x)
        if self.use_subquadratic_ops:
            features = subquadratic_ops_rearrange(features, bhl_to_lbh=False)
        else:
            features = rearrange(features, "l b d -> b d l").contiguous()

        if (
            self.use_subquadratic_ops
            and self.operator_type in ["hyena_short_conv", "hyena_medium_conv"]
            and inference_context is None
        ):
            # todo: support inference_context for b2b_kernel
            # Use the B2BCausalConv1dModule wrapper with the existing weights from the original model
            z = self.b2b_kernel(features, _use_cp=_proj_use_cp)
        else:
            features = self.hyena_proj_conv(
                features, _use_cp=_proj_use_cp, inference_context=inference_context
            )  # [B, D, L]
            x1, x2, v = rearrange(features, "b (g dg p) l -> b (g dg) p l", p=3, g=self.num_groups_per_tp_rank).unbind(
                dim=2
            )
            z = self.mixer(x1, x2, v, _hyena_use_cp=_proj_use_cp, inference_context=inference_context)

        if self.use_subquadratic_ops:
            z = subquadratic_ops_rearrange(z, bhl_to_lbh=True)
        else:
            z = rearrange(z, "b d l -> l b d").contiguous()
        y, bias = self.dense(z)
        return y, bias
