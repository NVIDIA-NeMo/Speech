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
import re
from typing import Optional

import torch
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
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
TERowParallelLinear, HAVE_TE_ROW_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelLinear"
)
HAVE_TE = all((HAVE_TE_COL_LINEAR, HAVE_TE_LN_COL_LINEAR, HAVE_TE_ROW_LINEAR))

MixedFusedLayerNorm, HAVE_APEX = safe_import_from("apex.normalization.fused_layer_norm", "MixedFusedLayerNorm")


def get_adapter_attributes_from_linear(m: nn.Module):
    """
    Return input_is_parallel, in_features, out_feature attributes based on implementation of the base layer.
    """
    if HAVE_TE and isinstance(m, TEColumnParallelLinear) or isinstance(m, TELayerNormColumnParallelLinear):
        input_is_parallel = False
        # m.in_features and m.out_features are divided by tp_size already,
        # but in_features and out_features passed to ParallelLinearAdapter are not.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        in_features = m.in_features
        out_features = m.out_features * tp_size
        # LoRA is applied after layernorm, so layernorm output must be returned
        m.return_layernorm_output = True
        # perf optimization for LoRA + SP
        if m.config.sequence_parallel and not m.ub_overlap_ag:
            m.return_layernorm_output_gathered = True
    elif HAVE_TE and isinstance(m, TERowParallelLinear):
        input_is_parallel = True
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        in_features = m.in_features * tp_size
        out_features = m.out_features
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

    return input_is_parallel, in_features, out_features


def is_expert_linear(fqn):
    """
    Return whether the current base module is an expert linear module.
    See ParallelLinearAdapter.is_expert for usage details.
    """
    return re.match('.*mlp\.experts\.local_experts.[0-9]+\.linear_fc[1-2]$', fqn) is not None


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
        activation: str = 'swish',
        norm_position: Optional[str] = 'post',
        norm_type: Optional[str] = 'mixedfusedlayernorm',
        column_init_method: str = 'xavier',
        # TODO: (@adithyare) should rename this to input_init_method to be more precise.
        row_init_method: str = 'zero',
        # TODO: (@adithyare) should rename this to output_init_method to be more precise.
        gather_output: bool = True,
        input_is_parallel: bool = False,
        # NOTE: (@ertkonuk) we need this for LoRA adapters that are applied to RowParallelLinear layers
        dropout: float = 0.0,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        alpha: float | None = None,
        dropout_position: str = 'post',
        a2a_experimental: bool = False,
        # TODO: should rename this or make it a default feature
        is_expert: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.activation = activation_registry[activation]()
        self.norm_position = norm_position
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
        self._sequence_parallel = model_parallel_config.sequence_parallel
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
                disable_grad_reduce=self._sequence_parallel,
            )
        if gather_output:
            self.linear_out = RowParallelLinear(
                dim,
                out_features,
                config=model_parallel_config,
                bias=False,
                init_method=self._get_init_fn(row_init_method),
                input_is_parallel=False,
                skip_bias_add=True,
            )
        else:
            # (@adithyare) we use this option to mirror the behavior
            # a column parallel layer with two low-rank column parallel layers
            # if the original column parallel layer uses gather_output=False,
            # then we will use the self.liner_out layer defined below.
            lin_out_gather_output = True if input_is_parallel else False
            if self.use_a2a and input_is_parallel and self._sequence_parallel:
                lin_out_gather_output = False
            self.linear_out = ColumnParallelLinear(
                dim,
                out_features,
                config=model_parallel_config,
                bias=False,
                gather_output=lin_out_gather_output,
                init_method=self._get_init_fn(row_init_method),
            )

        if self.norm_position in ["pre", "post"]:
            ln_features = in_features if self.norm_position == "pre" else out_features
            if norm_type == 'mixedfusedlayernorm':
                assert HAVE_APEX, "Apex is required to use MixedFusedLayerNorm"
                self.layer_norm = MixedFusedLayerNorm(ln_features, 1e-5, sequence_parallel_enbaled=False)
            elif norm_type == 'layernorm':
                self.layer_norm = nn.LayerNorm(ln_features)
            else:
                raise NotImplementedError("norm_type should be either mixedfusedlayernorm or layernorm")
        else:
            self.layer_norm = None

        if dropout > 0.0:
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
        model_parallel_config.sequence_parallel = self._sequence_parallel
        if self._sequence_parallel and not input_is_parallel:
            from importlib.metadata import version

            import packaging

            te_version = packaging.version.Version(version("transformer-engine"))
            if te_version >= packaging.version.Version("1.5.0dev") and (
                not self.input_is_parallel
                and (
                    not getattr(model_parallel_config, "tp_comm_overlap", False)
                    or getattr(model_parallel_config, "tp_comm_overlap_disable_qkv", False)
                )
            ):
                # TE 1.5 introduces the option `return_layernorm_output_gathered`, so the all gather
                # in the forward method is not needed, so set self._sequence_parallel to False
                # unless TP communication overlap is used
                self._sequence_parallel = False

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

    def adapter_unfreeze(
        self,
    ):
        """
        Can be customized to allow for selective training of only some params in the PEFT.
        """
        super().adapter_unfreeze()

    def forward(self, x):
        """ """

        if self.dropout is not None and self.dropout_position == 'pre':
            x = self.dropout(x)

        pad_len = 0
        if self.is_expert:
            x, pad_len = pad_seq_to_mult(x, self.config.tensor_model_parallel_size)

        if self.norm_position == 'pre':
            x = self.layer_norm(x)
        if self._sequence_parallel and not self.input_is_parallel:
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

        if self._sequence_parallel and self.input_is_parallel and not self.is_expert:
            # for attention_dense and linear_fc2
            # layernorm after lora is impacted by sequence parallel,
            # hence seq dim need to be scattered right after lora linear layers
            # this function also handles the backward pass correctly
            if self.use_a2a:
                # all2all hidden_size / TP to seq_len / TP
                x = all2all_hp2sp(x)
            else:
                x = scatter_to_sequence_parallel_region(x)

        if self.norm_position == 'post':
            x = self.layer_norm(x)

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
        """ """

        sharded_state_dict = {}
        sharded_state_dict.update(self.linear_in.sharded_state_dict(f"{prefix}linear_in.", sharded_offsets, metadata))
        sharded_state_dict.update(
            self.linear_out.sharded_state_dict(f"{prefix}linear_out.", sharded_offsets, metadata)
        )
        return sharded_state_dict
