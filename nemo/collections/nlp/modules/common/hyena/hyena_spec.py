import torch.nn as nn
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TERowParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import ModuleSpec

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
