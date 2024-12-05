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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

import torch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from torch import nn

from nemo.collections.llm.peft.lora import LinearAdapter, LoRALinear
from nemo.collections.llm.peft.utils import get_adapter_attributes_from_linear, is_expert_linear, wildcard_match
from nemo.lightning.pytorch.callbacks.peft import PEFT, AdapterWrapper
from nemo.utils import logging


class ModuleDict(nn.ModuleDict):
    """
    nn.ModuleDict with a sharded_state_dict implementation for checkpointing
    """

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> "ShardedStateDict":
        """Retrieve the sharded state dictionary of the wrapped module and adapter.

        This method is used for distributed checkpointing, combining the sharded states
        of both the main module and the adapter.

        Args:
            prefix (str): A prefix added to parameter and buffer names. Defaults to ''.
            sharded_offsets (Tuple[Tuple[int, int, int]]): Offsets for sharded parameters.
                                                           Defaults to an empty tuple.
            metadata (Optional[dict]): Additional metadata for the sharded state.
                                       Defaults to None.

        Returns:
            ShardedStateDict: The combined sharded state dictionary.
        """
        sharded_state_dict = {}
        for key, layer in self.items():
            sharded_state_dict.update(layer.sharded_state_dict(f"{prefix}{key}.", sharded_offsets, metadata))
        return sharded_state_dict


class LoRALinearSplitQKV(AdapterWrapper):
    """An adapter wrapper for `linear_qkv` where q, k, v are three separate adapters.
    This module that adds the output of the adapters to the output of the wrapped module while taking care of shape.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(self, x):
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x)
        query = self.adapter.adapter_q(layernorm_output)
        key = self.adapter.adapter_k(layernorm_output)
        value = self.adapter.adapter_v(layernorm_output)

        query_4d = query.reshape(query.shape[0], query.shape[1], -1, self.to_wrap.config.kv_channels)
        key_4d = key.reshape(key.shape[0], key.shape[1], -1, self.to_wrap.config.kv_channels)
        value_4d = value.reshape(value.shape[0], value.shape[1], -1, self.to_wrap.config.kv_channels)

        qkv_4d = torch.cat([query_4d, key_4d, value_4d], dim=2)
        adapter_output = qkv_4d.reshape(qkv_4d.shape[0], qkv_4d.shape[1], -1)

        return linear_output + adapter_output, bias


class LoRALinearSplitFC1UpGate(AdapterWrapper):
    """An adapter wrapper for `linear_fc1` where up_proj and gate_proj are two separate adapters.
    This module that adds the output of the adapters to the output of the wrapped module while taking care of shape.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(self, x):
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x)
        adapter_output_gate = self.adapter.adapter_gate(layernorm_output)
        adapter_output_up = self.adapter.adapter_up(layernorm_output)
        adapter_output = torch.cat([adapter_output_gate, adapter_output_up], dim=2)
        return linear_output + adapter_output, bias


@dataclass
class CanonicalLoRA(PEFT):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.
    Canonical LoRA applies LoRA on Q, K, V projection matrices separately, as well as Up and Gate projection
    matrices separately. This follows more closely with Huggingface's implementation of LoRA.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_q', 'linear_k', 'linear_v', 'linear_proj',
                                           'linear_fc1_up', 'linear_fc1_gate', 'linear_fc2'].
                - 'linear_q', 'linear_k', 'linear_v': Apply LoRA to the linear layer used for query, key, and value
                        projections in self-attention. This is fused into one matrix in NeMo LoRA, but left as three
                        separate matrices in Canonical LoRA.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1_up', 'linear_fc1_proj': Apply LoRA to the Up proj and Gate proj layers.
                        These two together constitute the first fully-connected layer in MLP in NeMo LoRA.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_q', '*.layers.1.*.linear_q'] to add LoRA to only linear_q
                on the first two layers.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.

    Example:
    --------
        >>> from nemo.collections import llm
        >>> lora = llm.peft.CanonicalLoRA(target_modules=['linear_q', 'linear_k', 'linear_v', 'linear_fc1_up'], dim=32)
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
        default_factory=lambda: [
            'linear_q',
            'linear_k',
            'linear_v',
            'linear_proj',
            'linear_fc1_up',
            'linear_fc1_gate',
            'linear_fc2',
        ]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal['pre', 'post'] = 'pre'
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"

    def __post_init__(self):
        """
        Construct a mapping from the target module as supported in LoRA() to the specific parts of the layer for which
        adapter is applied.

        For example, if user specifies target_module = ['linear_q', 'linear_k', 'linear_proj', 'linear_fc1_up'], then
        canonical_lora_mapping = {
            "linear_qkv": {'linear_q', 'linear_k'},
            "linear_proj": {'linear_proj'},  # the value of this key does not matter
            "linear_fc1": {'linear_fc1_up'},
        }

        If user specifies target_module = ['*.layers.0.*.linear_q', '*.layers.1.*.linear_q'], then
        canonical_lora_mapping = {
            "'*.layers.0.*.linear_qkv'": {'linear_q'},
            "'*.layers.1.*.linear_qkv'": {'linear_q'},
        }

        """
        self.canonical_mapping: Dict[str, Set] = defaultdict(set)
        for target in self.target_modules:
            assert not target.endswith("linear_qkv"), (
                "Canonical LoRA does not support target 'linear_qkv'. Either use 'linear_qkv' with LoRA() or "
                "use ['linear_q', 'linear_k', 'linear_v'] with Canonical LoRA"
            )
            assert not target.endswith("linear_fc1"), (
                "Canonical LoRA does not support target 'linear_fc1'. Either use 'linear_fc1' with LoRA() or "
                "use ['linear_fc1_up', 'linear_fc1_gate'] with Canonical LoRA"
            )

            if 'linear_q' in target:
                self.canonical_mapping[target.replace('linear_q', 'linear_qkv')].add('linear_q')
            elif 'linear_k' in target:
                self.canonical_mapping[target.replace('linear_k', 'linear_qkv')].add('linear_k')
            elif 'linear_v' in target:
                self.canonical_mapping[target.replace('linear_v', 'linear_qkv')].add('linear_v')
            elif 'linear_fc1_up' in target:
                self.canonical_mapping[target.replace('linear_fc1_up', 'linear_fc1')].add('linear_fc1_up')
            elif 'linear_fc1_gate' in target:
                self.canonical_mapping[target.replace('linear_fc1_gate', 'linear_fc1')].add('linear_fc1_gate')
            else:
                self.canonical_mapping[target].add(target)

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

        """
        Find the element in canonical_mapping which 
        1) matches the current `name` exactly, OR
        2) matches the current `full_name` with regex
        match is None if current module name doesn't match the specified targets.
        """
        match = None
        for pattern in self.canonical_mapping:
            if name == pattern or wildcard_match(pattern, full_name):
                match = pattern
                break

        if match is not None:
            if isinstance(m, nn.Linear):
                return LinearAdapter(
                    m, dim=self.dim, alpha=self.alpha, dropout=self.dropout, lora_A_init_method=self.lora_A_init_method
                )

            input_is_parallel, in_features, out_features = get_adapter_attributes_from_linear(m)

            adapter_kwargs = dict(
                dim=self.dim,
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
            )
            if name in ['linear_proj', 'linear_fc2']:
                adapter = ParallelLinearAdapter(in_features, out_features, **adapter_kwargs)
                logging.info(f"Adding lora to: {full_name}")
                return LoRALinear(m, adapter)

            canonical_submodules = self.canonical_mapping[match]
            logging.info(f"Adding lora to: {full_name} ({canonical_submodules})")
            if name == 'linear_qkv':
                adapter_q, adapter_k, adapter_v = None, None, None
                kv_out_features = m.config.kv_channels * m.config.num_query_groups
                if 'linear_q' in canonical_submodules:
                    adapter_q = ParallelLinearAdapter(in_features, in_features, **adapter_kwargs)
                if 'linear_k' in canonical_submodules:
                    adapter_k = ParallelLinearAdapter(in_features, kv_out_features, **adapter_kwargs)
                if 'linear_v' in canonical_submodules:
                    adapter_v = ParallelLinearAdapter(in_features, kv_out_features, **adapter_kwargs)
                adapters = ModuleDict({'adapter_q': adapter_q, 'adapter_k': adapter_k, 'adapter_v': adapter_v})
                return LoRALinearSplitQKV(m, adapters)

            if name == 'linear_fc1':
                adapter_up, adapter_gate = None, None
                if 'linear_fc1_up' in canonical_submodules:
                    adapter_up = ParallelLinearAdapter(in_features, out_features // 2, **adapter_kwargs)
                if 'linear_fc1_gate' in canonical_submodules:
                    adapter_gate = ParallelLinearAdapter(in_features, out_features // 2, **adapter_kwargs)
                adapters = ModuleDict({'adapter_up': adapter_up, 'adapter_gate': adapter_gate})
                return LoRALinearSplitFC1UpGate(m, adapters)

        return m
