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

import math
from dataclasses import dataclass, field
from typing import List, Literal

import torch
from torch import nn

from nemo.collections.llm.peft.utils import get_adapter_attributes_from_linear, is_expert_linear, wildcard_match
from nemo.lightning.pytorch.callbacks.peft import PEFT, AdapterWrapper
from nemo.utils import logging


class LoRALinear(AdapterWrapper):
    """An adapter wrapper that adds the output of the adapter to the output of the wrapped module.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(self, x):
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x)
        adapter_output = self.adapter(layernorm_output.contiguous())
        return linear_output + adapter_output, bias


class LinearAdapter(nn.Module):
    """
    A simple LoRA linear module for non-megatron models.
    """

    def __init__(
        self, orig_linear, dim=8, alpha=32, dropout=0.1, dropout_position='post', lora_A_init_method='xavier'
    ):
        super(LinearAdapter, self).__init__()
        assert isinstance(orig_linear, nn.Linear)

        self.orig_linear = orig_linear
        self.dim = dim
        self.scale = alpha / dim

        # Freezer
        device = self.orig_linear.weight.device
        self.orig_linear.weight.requires_grad = False
        if self.orig_linear.bias is not None:
            self.orig_linear.bias.requires_grad = False

        in_features = self.orig_linear.in_features
        out_features = self.orig_linear.out_features
        dtype = self.orig_linear.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros((in_features, dim), dtype=dtype, device=device))
        self.lora_b = nn.Parameter(torch.zeros((dim, out_features), dtype=dtype, device=device))
        if lora_A_init_method == 'xavier':
            torch.nn.init.uniform_(self.lora_a)
        else:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        assert dropout_position in ['pre', 'post'], dropout_position
        self.dropout_position = dropout_position

    def forward(self, x):
        # pylint: disable=C0115,C0116
        res = self.orig_linear(x)
        if self.dropout_position == 'pre':
            x = self.dropout(x)
        lora_res = x @ self.lora_a
        lora_res = lora_res @ self.lora_b
        lora_res = lora_res * self.scale
        if self.dropout_position == 'post':
            lora_res = self.dropout(lora_res)
        return res + lora_res


@dataclass
class LoRA(PEFT):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        a2a_experimental (bool): Enables the experimental All-to-All (A2A) communication strategy. Defaults to False.

    Example:
    --------
        >>> from nemo.collections import llm
        >>> lora = llm.peft.LoRA(target_modules=['linear_qkv', 'linear_proj'], dim=32)
        >>> model = llm.Mistral7BModel(model_transform=lora)
        >>> # (set up trainer and data)
        >>> trainer.fit(model, data)

    References:
    -----------
        Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).
        LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.
        https://arxiv.org/abs/2106.09685

    )
    """

    target_modules: List[str] = field(
        default_factory=lambda: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal['pre', 'post'] = 'pre'
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False

    def transform(self, m: nn.Module, name=None, prefix=None):
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import ParallelLinearAdapter

        full_name = f"{prefix}.{name}" if prefix else name
        if name in self.target_modules or any(wildcard_match(pattern, full_name) for pattern in self.target_modules):
            if isinstance(m, nn.Linear):
                return LinearAdapter(
                    m, dim=self.dim, alpha=self.alpha, dropout=self.dropout, lora_A_init_method=self.lora_A_init_method
                )

            input_is_parallel, in_features, out_features = get_adapter_attributes_from_linear(m)
            logging.info(f"Adding lora to: {full_name}")
            adapter = ParallelLinearAdapter(
                in_features,
                out_features,
                self.dim,
                activation='identity',
                norm_position=None,
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(m, "config", None),
                alpha=self.alpha,
                is_expert=is_expert_linear(full_name),
                a2a_experimental=self.a2a_experimental,
            )
            return LoRALinear(m, adapter)
        return m


class LoRAMerge(PEFT):
    """
    Implements the LoRA weight merge for parameter-efficient fine-tuning.

    Example:
    --------
        >>> from nemo.collections.llm.peft.lora import LoRAMerge
        >>> lora_merge = LoRAMerge()
        >>> merged_model = lora_merge(trainer.strategy.megatron_parallel)
    """

    @torch.no_grad()
    def transform(self, m: nn.Module, name=None, prefix=None):
        """
        Merges the LoRA adapter with the base model weights.

        Args:
            m (nn.Module): The module to apply LoRA merge to.
            name (str, optional): Name of the module to merge. Defaults to None.
            prefix (str, optional): Prefix for the module name. Defaults to None.

        Returns:
            nn.Module: The modified module with the LoRA adapter merged into the base model weights.
        """

        if not isinstance(m, LoRALinear):
            return m
        logging.info(f'merging {(prefix if prefix else "") + "." + (name if name else "")}')
        base_weight = m.to_wrap.weight
        lora_weight = (
            m.adapter.alpha
            / m.adapter.dim
            * m.adapter.linear_out.weight.to(base_weight.device)
            @ m.adapter.linear_in.weight.to(base_weight.device)
        )
        merged_weight = base_weight + lora_weight
        m.to_wrap.weight.data = merged_weight
        return m
