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

import torch.nn as nn

try:
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TERowParallelLinear
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import ModuleSpec

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

    ModuleSpec = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False

from nemo.collections.nlp.modules.common.hyena.hyena import (
    CausalDepthWiseConv1d,
    HyenaOperator,
    HyenaOperatorSubmodules,
)
from nemo.collections.nlp.modules.common.hyena.hyena_filter import (
    ExponentialModulation,
    HyenaFilter,
    HyenaFilterSubmodules,
    PositionalEmbedding,
    Sin,
)


def get_hyena_layer_with_transformer_engine_spec(hyena_cfg):
    if not HAVE_MEGATRON_CORE:
        raise ImportError(
            "megatron-core was not found. "
            "Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
        )

    return ModuleSpec(
        module=HyenaOperator,
        params=hyena_cfg,
        submodules=HyenaOperatorSubmodules(
            in_proj=TELayerNormColumnParallelLinear,
            short_filter=CausalDepthWiseConv1d,
            implicit_filter=ModuleSpec(
                module=HyenaFilter,
                submodules=HyenaFilterSubmodules(
                    positional_embedding=PositionalEmbedding,
                    linear=nn.Linear,
                    activation=Sin,
                    modulation=ExponentialModulation,
                ),
            ),
            out_proj=TERowParallelLinear,
        ),
    )


def get_gpt_layer_with_te_and_hyena_spec(hyena_cfg):
    spec = get_gpt_layer_with_transformer_engine_spec()
    spec.submodules.self_attention = get_hyena_layer_with_transformer_engine_spec(hyena_cfg)
    return spec
