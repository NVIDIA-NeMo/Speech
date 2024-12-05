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

import re
from megatron.core import parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from torch import nn

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
