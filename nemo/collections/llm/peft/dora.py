# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_tp_sharded_tensor_for_checkpoint
from torch import nn

from nemo.collections.llm.peft.module_matcher import ModuleMatcher
from nemo.collections.llm.peft.utils import ParallelLinearAdapter, get_adapter_attributes_from_linear
from nemo.lightning.pytorch.callbacks.peft import PEFT, AdapterWrapper
from nemo.utils import logging


class ParallelLinearDoRAAdapter(ParallelLinearAdapter):
    """
    Adapter class for DoRA to handle the additional weight_magnitude parameter
    """

    def init_weight_magnitude(self, value):
        """
        Initialize weight_magnitude with shape (d,), where d is the output dim of the linear layer
        """
        self.weight_magnitude = nn.Parameter(value, requires_grad=True)

    def get_weight_magnitude(self):
        """
        Public function to get the weight magnitude parameter
        """
        return self.weight_magnitude

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Sharded state dict implementation for DoRA adapter.
        Weight magnitude is TP sharded for linear_qkv and linear_fc1 only.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        magnitude_key = f"{prefix}weight_magnitude"
        if self.input_is_parallel:
            # RPL output is gathered, so weight_magnitude is not sharded for TP
            magnitude_sharded_tensor = make_sharded_tensor_for_checkpoint(
                self.weight_magnitude, magnitude_key, prepend_offsets=sharded_offsets
            )
        else:
            # CPL output is sharded, so weight_magnitude is sharded for TP
            magnitude_sharded_tensor = make_tp_sharded_tensor_for_checkpoint(
                self.weight_magnitude, magnitude_key, 0, prepend_offsets=sharded_offsets
            )
        sharded_state_dict[magnitude_key] = magnitude_sharded_tensor

        return sharded_state_dict


class DoRALinear(AdapterWrapper):
    """
    An adapter wrapper that is designed to be used with DoRA
    It extends the AdapterWrapper class to provide a specific implementation of the forward method.
    """

    def __init__(self, to_wrap: nn.Module, adapter: ParallelLinearDoRAAdapter):
        super().__init__(to_wrap, adapter)
        self.adapter: ParallelLinearDoRAAdapter
        self.scaling = adapter.alpha / adapter.dim
        self.adapter.init_weight_magnitude(self._get_weight_norm())

    def _get_weight_norm(self):
        if self.adapter.input_is_parallel:
            linear_out_weight = gather_from_tensor_model_parallel_region(self.adapter.linear_out.weight.T).T
            linear_in_weight = self.adapter.linear_in.weight
        else:
            linear_out_weight = self.adapter.linear_out.weight
            linear_in_weight = gather_from_tensor_model_parallel_region(self.adapter.linear_in.weight.T).T

        weight = self.to_wrap.weight + self.scaling * linear_out_weight @ linear_in_weight
        return torch.linalg.norm(weight, dim=1).to(weight.dtype).detach()

    def forward(self, x):
        """
        Forward method for DoRA

          mag_norm_scale * (linear_output + adapter_output)
        = ||W_0 + B_0 A_0|| / ||W_0 + B A|| * (W_0 x + B A x)
        = ||W_0 + B_0 A_0|| ((W_0 + B A) / ||W_0 + B A||) x
        = m ((W_0 + B A) / ||W_0 + B A||) x
        = equation 5 in DoRA paper

        When dropout is used, equation becomes
          W_0 x + (m /||W_0 + B A|| - 1) W_0 dropout(x) + m /||W_0 + B A|| B A dropout(x)
        = ...
        = m /||W_0 + B A|| (W_0 x + B A dropout(x)) + (m /||W_0 + B A|| - 1) W_0 (dropout(x) - x)

        """
        linear_output, bias, layernorm_output = self.base_linear_forward(x)
        adapter_output = self.adapter(layernorm_output.contiguous())

        # mag_norm_scale is  ||W_0 + B_0 A_0|| / ||W_0 + B A||  (scaling in front of BA not shown)
        mag_norm_scale = (self.adapter.get_weight_magnitude() / self._get_weight_norm()).view(1, 1, -1)
        if self.adapter.dropout is None or not self.training:
            dropout_correction = 0
        else:
            dropout_correction = (mag_norm_scale - 1) * self.base_linear_forward(
                self.adapter.dropout(layernorm_output) - layernorm_output
            )[0]

        return (
            mag_norm_scale * (linear_output + adapter_output.reshape(linear_output.shape)) + dropout_correction,
            bias,
        )


@dataclass
class DoRA(PEFT, ModuleMatcher):
    """
    Implements the DoRA (Weight-Decomposed LowRank Adaptation) module for parameter-efficient fine-tuning.

    DoRA decomposes pre-trained weight into magnitude and direction, and uses a low-rank projection in the
    directional component to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of DoRA to specific modules within the model architecture.

    Args:
        See LoRA class for a detailed explanation of the arguments.

    Example:
    --------
        >>> from nemo.collections import llm
        >>> lora = llm.peft.DoRA(target_modules=['linear_qkv', 'linear_proj'], dim=32, alpha=64)
        >>> model = llm.Mistral7BModel(model_transform=lora)
        >>> # (set up trainer and data)
        >>> trainer.fit(model, data)

    References:
    -----------
        Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng,
        Min-Hung Chen (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. arXiv preprint arXiv:2402.09353.
        https://arxiv.org/abs/2402.09353
    )
    """

    target_modules: List[str] = field(
        default_factory=lambda: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    )
    dim: int = 32
    alpha: int = 64
    dropout: float = 0.0
    dropout_position: Literal['pre', 'post'] = 'pre'
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"

    def __post_init__(self):
        assert self.dropout_position == "pre", (
            "DoRA only supports pre-adapter dropout at this time." "Please set DoRA(..., dropout_position='pre')"
        )

    def transform(self, m: nn.Module, name=None, prefix=None):
        """
        Applies DoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply DoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with DoRA applied, or the original module if not a target.
        """
        if (ans := self.match(m, name, prefix)) is not None:
            (match, full_name) = ans
            input_is_parallel, in_features, out_features, disable_sp_comm, base_linear_is_parallel = (
                get_adapter_attributes_from_linear(m)
            )
            logging.info(f"Adding DoRA to: {full_name}")
            adapter = ParallelLinearDoRAAdapter(
                in_features,
                out_features,
                self.dim,
                base_linear_name=full_name,
                activation='identity',
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(m, "config", None),
                alpha=self.alpha,
                disable_sequence_parallel_comm=disable_sp_comm,
                base_linear_is_parallel=base_linear_is_parallel,
            )
            return DoRALinear(m, adapter)
        return m
