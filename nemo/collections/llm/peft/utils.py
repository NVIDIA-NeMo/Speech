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

import math
import re
from importlib.metadata import version
from typing import Optional

import packaging
import torch
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from torch import nn

from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil
from nemo.collections.common.parts.utils import activation_registry
from nemo.core.classes.mixins import adapter_mixin_strategies
from nemo.utils.import_utils import safe_import_from

TEColumnParallelLinear, HAVE_TE_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TEColumnParallelLinear"
)
TELayerNormColumnParallelLinear, HAVE_TE_LN_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine",
    "TELayerNormColumnParallelLinear",
)
TEColumnParallelGroupedLinear, HAVE_TE_COL_GRP_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TEColumnParallelGroupedLinear"
)
TERowParallelLinear, HAVE_TE_ROW_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelLinear"
)
TERowParallelGroupedLinear, HAVE_TE_ROW_GRP_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelGroupedLinear"
)
TELinear, HAVE_TE_LINEAR = safe_import_from("megatron.core.extensions.transformer_engine", "TELinear")
HAVE_TE = all(
    (
        HAVE_TE_COL_LINEAR,
        HAVE_TE_LN_COL_LINEAR,
        HAVE_TE_ROW_LINEAR,
        HAVE_TE_LINEAR,
        HAVE_TE_COL_GRP_LINEAR,
        HAVE_TE_ROW_GRP_LINEAR,
    )
)

MixedFusedLayerNorm, HAVE_APEX = safe_import_from("apex.normalization.fused_layer_norm", "MixedFusedLayerNorm")

TECL = (TEColumnParallelLinear, TELayerNormColumnParallelLinear, TEColumnParallelGroupedLinear)
TERL = (TERowParallelLinear, TERowParallelGroupedLinear)


def get_adapter_attributes_from_linear(m: nn.Module):
    """
    Return input_is_parallel, in_features, out_feature attributes based on implementation of the base layer.
    """
    disable_sequence_parallel_comm = not m.config.sequence_parallel
    base_linear_is_parallel = True
    if HAVE_TE and any(isinstance(m, te_column_parallel) for te_column_parallel in TECL):
        input_is_parallel = False
        # m.in_features and m.out_features are divided by tp_size already,
        # but in_features and out_features passed to ParallelLinearAdapter are not.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        in_features = m.in_features
        out_features = m.out_features * tp_size

        if isinstance(m, TELayerNormColumnParallelLinear):
            # LoRA is applied after layernorm, so layernorm output must be returned
            m.return_layernorm_output = True
            # perf optimization for LoRA + SP
            if hasattr(m, "ub_overlap_ag"):
                ub_overlap_ag = m.ub_overlap_ag
            elif hasattr(m, "ub_overlap_ag_fprop"):
                ub_overlap_ag = m.ub_overlap_ag_fprop
            else:
                ub_overlap_ag = False
            if m.config.sequence_parallel and not ub_overlap_ag:
                m.return_layernorm_output_gathered = True
                te_version = packaging.version.Version(version("transformer-engine"))
                if te_version >= packaging.version.Version("1.5.0dev") and (
                    not getattr(m.config, "tp_comm_overlap", False)
                    or getattr(m.config, "tp_comm_overlap_disable_qkv", False)
                ):
                    # TE 1.5 introduces the option `return_layernorm_output_gathered`, so the all gather
                    # in the forward method is not needed, so disable sp communications
                    # unless TP communication overlap is used
                    disable_sequence_parallel_comm = True
    elif HAVE_TE and any(isinstance(m, te_row_parallel) for te_row_parallel in TERL):
        input_is_parallel = True
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        in_features = m.in_features * tp_size
        out_features = m.out_features
    elif HAVE_TE and isinstance(m, TELinear):  # parallel_mode="duplicated"
        input_is_parallel = False
        in_features = m.in_features
        out_features = m.out_features
        base_linear_is_parallel = False
    elif isinstance(m, ColumnParallelLinear):
        input_is_parallel = False
        in_features = m.input_size
        out_features = m.output_size
    elif isinstance(m, RowParallelLinear):
        input_is_parallel = True
        in_features = m.input_size
        out_features = m.output_size
    else:
        raise NotImplementedError(f"Layer type is unrecognized for LoRA: {type(m)}")

    return input_is_parallel, in_features, out_features, disable_sequence_parallel_comm, base_linear_is_parallel


def is_expert_linear(fqn):
    """
    Return whether the current base module is an expert linear module.
    See ParallelLinearAdapter.is_expert for usage details.
    """
    return re.match(r'.*mlp\..*experts.*\.linear_fc[1-2]$', fqn) is not None


def wildcard_match(pattern, key):
    """
    Return whether the pattern (target module to add LoRA) matches the key (model weight name).

    Example:
    --------
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.0.self_attention.linear_qkv")
        True
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.1.self_attention.linear_qkv")
        False
    """
    if key is None:
        return None
    regex_pattern = re.compile("^" + pattern.replace("*", "(.*)") + "$")
    match = regex_pattern.match(key)
    return match is not None


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def init_method_kaiming_uniform(val):
    """ """

    def init_(tensor):
        return nn.init.kaiming_uniform_(tensor, a=val)

    return init_


def init_method_const(val):
    """ """

    def init_(tensor):
        return nn.init.constant_(tensor, val)

    return init_


def pad_seq_to_mult(x, mult):
    """ """

    if x.shape[0] % mult == 0:
        return x, 0
    pad_len = mult - (x.shape[0] % mult)
    with torch.no_grad():
        # pad at the tail
        x = nn.functional.pad(x, (0, 0, 0, pad_len))
    return x, pad_len


def unpad_seq_to_mult(x, pad_len):
    """ """
    if pad_len <= 0:
        return x
    with torch.no_grad():
        # prune tail padding
        return x[:-pad_len, :]


class _All2AllHp2Sp(torch.autograd.Function):
    """
    All-2-All from Hidden Parallel to Sequence Parallel
    This is a temporary workaround and can be updated in the future
    TODO: Move the functionality to MCore
    """

    @staticmethod
    def forward(ctx, input_):
        """ """

        world_size = parallel_state.get_tensor_model_parallel_world_size()
        group = parallel_state.get_tensor_model_parallel_group()
        send_list = list(input_.chunk(world_size, dim=0))
        send_list = [tensor.contiguous() for tensor in send_list]
        receive_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
        torch.distributed.all_to_all(receive_list, send_list, group=group)
        x = torch.cat(receive_list, dim=-1)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        """ """

        world_size = parallel_state.get_tensor_model_parallel_world_size()
        group = parallel_state.get_tensor_model_parallel_group()
        send_list = list(grad_output.chunk(world_size, dim=-1))
        send_list = [tensor.contiguous() for tensor in send_list]
        receive_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
        torch.distributed.all_to_all(receive_list, send_list, group=group)
        x = torch.cat(receive_list, dim=0)

        return x


def all2all_hp2sp(input_):
    """ """
    return _All2AllHp2Sp.apply(input_)


class ParallelLinearAdapter(nn.Module, AdapterModuleUtil):
    """ """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        base_linear_name: str,
        activation: str = 'swish',
        column_init_method: str = 'xavier',
        row_init_method: str = 'zero',
        input_is_parallel: bool = False,
        dropout: float = 0.0,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        alpha: float | None = None,
        dropout_position: str = 'post',
        a2a_experimental: bool = False,
        is_expert: bool = False,
        disable_sequence_parallel_comm: bool = True,
        dropout_recompute: bool = False,
        base_linear_is_parallel: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.base_linear_name = base_linear_name
        self.activation = activation_registry[activation]()
        self.dim = dim
        self.alpha = alpha if alpha is not None else self.dim
        self.input_is_parallel = input_is_parallel
        self.dropout_position = dropout_position
        self.use_a2a = a2a_experimental
        self.is_expert = is_expert

        # megatron_gpt_peft_models will provide this arg, but deprecated ones do not.
        # in case this arg is not provided, use the dummy default config.
        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()
        _sequence_parallel = model_parallel_config.sequence_parallel
        model_parallel_config.sequence_parallel = False  # SP is irrelevant for the lora linear layer
        self.config = model_parallel_config

        if input_is_parallel:
            self.linear_in = RowParallelLinear(
                in_features,
                dim,
                config=model_parallel_config,
                input_is_parallel=True,
                skip_bias_add=True,
                bias=False,
                init_method=self._get_init_fn(column_init_method),
            )
        else:
            self.linear_in = ColumnParallelLinear(
                in_features,
                dim,
                config=model_parallel_config,
                bias=False,
                gather_output=True,
                init_method=self._get_init_fn(column_init_method),
                disable_grad_reduce=_sequence_parallel,
            )

        # (@adithyare) we use this option to mirror the behavior
        # a column parallel layer with two low-rank column parallel layers
        # if the original column parallel layer uses gather_output=False,
        # then we will use the self.liner_out layer defined below.
        lin_out_gather_output = True if input_is_parallel else False
        if self.use_a2a and input_is_parallel and _sequence_parallel:
            lin_out_gather_output = False

        if not base_linear_is_parallel:
            lin_out_gather_output = True

        self.linear_out = ColumnParallelLinear(
            dim,
            out_features,
            config=model_parallel_config,
            bias=False,
            gather_output=lin_out_gather_output,
            init_method=self._get_init_fn(row_init_method),
        )

        if dropout > 0.0:
            if dropout_recompute:
                import thunder

                self.dropout = thunder.jit(nn.Dropout(dropout))
            else:
                self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # cast all parameters when using amp O2 training
        if model_parallel_config.bf16:
            self.bfloat16()
        elif model_parallel_config.fp16:
            self.half()

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_mixin_strategies.ReturnResultAdapterStrategy())

        # revert config change in case it is read elsewhere
        model_parallel_config.sequence_parallel = _sequence_parallel
        self.disable_sequence_parallel_comm = disable_sequence_parallel_comm
        if not _sequence_parallel:
            self.disable_sequence_parallel_comm = True

        if not base_linear_is_parallel:
            self.disable_sequence_parallel_comm = True

    def _get_init_fn(self, init_method: str):
        if init_method == 'xavier':
            init_fn = nn.init.xavier_normal_
        elif init_method == 'normal':
            init_fn = init_method_normal(0.2)
        elif init_method == 'kaiming':
            init_fn = init_method_kaiming_uniform(math.sqrt(5))
        elif init_method == "zero":
            init_fn = init_method_const(0.0)
        else:
            raise NotImplementedError("out_init_method should be zero, normal, kaiming or xavier")
        return init_fn

    def forward(self, x):
        """ """

        if self.dropout is not None and self.dropout_position == 'pre':
            x = self.dropout(x)

        pad_len = 0
        if self.is_expert:
            x, pad_len = pad_seq_to_mult(x, self.config.tensor_model_parallel_size)

        if not self.disable_sequence_parallel_comm and not self.input_is_parallel and not self.is_expert:
            # for attention_qkv and linear_fc1
            # layernorm before lora is impacted by sequence parallel,
            # hence seq dim need to be gathered right before lora linear layers
            # this function also handles the backward pass correctly
            x = gather_from_sequence_parallel_region(x)

        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        x, _ = self.linear_in(x)  # (@adithyare) ColumnLinear returns output and bias, we are ignoring the bias term.

        x = self.activation(x)

        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        x, _ = self.linear_out(x)

        if not self.disable_sequence_parallel_comm and self.input_is_parallel and not self.is_expert:
            # for attention_dense and linear_fc2
            # layernorm after lora is impacted by sequence parallel,
            # hence seq dim need to be scattered right after lora linear layers
            # this function also handles the backward pass correctly
            if self.use_a2a:
                # all2all hidden_size / TP to seq_len / TP
                x = all2all_hp2sp(x)
            else:
                x = scatter_to_sequence_parallel_region(x)

        # Add dropout if available
        if self.dropout is not None and self.dropout_position == 'post':
            x = self.dropout(x)

        x = x * (self.alpha / self.dim)

        if pad_len > 0:
            # Remove MoE padding.
            x = unpad_seq_to_mult(x, pad_len)

        return x

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Sharded state dict for LoRA adapter. Special treatment is given to the linear_fc1 adapter
        since TP is sharded separately for the two logical matrices (gate and up)
        """
        sharded_state_dict = {}
        linear_in_sd = self.linear_in.sharded_state_dict(f"{prefix}linear_in.", sharded_offsets, metadata)
        linear_out_sd = self.linear_out.sharded_state_dict(f"{prefix}linear_out.", sharded_offsets, metadata)

        if 'linear_fc1' in self.base_linear_name:
            for k, v in linear_out_sd.items():
                if k in (f'{prefix}linear_out.weight', f'{prefix}linear_out.bias'):
                    linear_out_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets)

        sharded_state_dict.update(linear_in_sd)
        sharded_state_dict.update(linear_out_sd)
        return sharded_state_dict
