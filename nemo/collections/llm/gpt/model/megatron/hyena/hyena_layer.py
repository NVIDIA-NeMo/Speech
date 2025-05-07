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

from dataclasses import dataclass
from typing import Optional, Union

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig


@dataclass
class HyenaLayerSubmodules:
    """Submodules for the HyenaLayer."""

    norm: Union[ModuleSpec, type] = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    hyena_bda: Union[ModuleSpec, type] = IdentityOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityOp


class HyenaLayer(MegatronModule):
    """Top level Hyena Layer."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        hyena_config: HyenaConfig,
        operator_type,
        max_sequence_length,
        submodules: HyenaLayerSubmodules,
        layer_number: int = 1,
        residual_in_fp32=False,
    ):
        """
        Top level Hyena Layer
        """
        super().__init__(config=transformer_config)
        self.transformer_config = transformer_config
        self.hyena_config = hyena_config
        self.layer_number = layer_number
        self.hidden_dropout = transformer_config.hidden_dropout
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = build_module(
            submodules.mixer,
            self.transformer_config,
            self.hyena_config,
            max_sequence_length,
            layer_number=layer_number,
            operator_type=operator_type,
        )
        self.norm = build_module(
            submodules.norm,
            self.transformer_config,
            self.transformer_config.hidden_size,
            eps=self.transformer_config.layernorm_epsilon,
        )

        self.hyena_bda = build_module(submodules.hyena_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.transformer_config,
            hidden_size=self.transformer_config.hidden_size,
            eps=self.transformer_config.layernorm_epsilon,
        )

        self.mlp = build_module(submodules.mlp, config=self.transformer_config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        self.mlp_bda = build_module(submodules.mlp_bda)

        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,  # Not used in HyenaLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Tensor = None,  # Not used in HyenaLayer
    ):
        """Forward pass for the HyenaLayer."""
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = hidden_states.to(dtype=self.transformer_config.params_dtype)
        hidden_states = self.norm(hidden_states)

        mixer_out_with_bias = self.mixer(hidden_states, inference_context=inference_context)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.hyena_bda(self.training, self.transformer_config.bias_dropout_fusion)(
                mixer_out_with_bias, residual, self.hidden_dropout
            )

        residual = hidden_states

        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.transformer_config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states
