# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer based language model."""
from nemo.collections.nlp.modules.common.megatron.megatron_perceiver_encoders import MegatronPerceiverEncoderModule
from nemo.collections.nlp.modules.common.megatron.megatron_transformer_encoder import MegatronTransformerEncoderModule
from nemo.collections.nlp.modules.common.megatron.retrieval_transformer import (
    MegatronRetrievalTransformerEncoderModule,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    init_method_normal,
    scaled_init_method_normal,
)

try:
    from MeCab import Model

    HAVE_MECAB = True
except (ImportError, ModuleNotFoundError):
    HAVE_MECAB = False

try:
    from apex.transformer.enums import AttnMaskType, ModelType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    ModelType = ApexGuardDefaults()

try:
    from megatron.core import ModelParallelConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

__all__ = []

AVAILABLE_ENCODERS = ["transformer", "perceiver", "retro"]


def get_encoder_model(
    config: ModelParallelConfig,
    arch,
    hidden_size,
    ffn_hidden_size,
    num_layers,
    num_attention_heads,
    apply_query_key_layer_scaling=False,
    kv_channels=None,
    init_method=None,
    scaled_init_method=None,
    encoder_attn_mask_type=AttnMaskType.padding,
    pre_process=True,
    post_process=True,
    init_method_std=0.02,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    ffn_dropout=0.0,
    precision=16,
    fp32_residual_connection=False,
    activations_checkpoint_method=None,
    activations_checkpoint_num_layers=1,
    activations_checkpoint_granularity=None,
    layernorm_epsilon=1e-5,
    bias_activation_fusion=True,
    bias_dropout_add_fusion=True,
    masked_softmax_fusion=True,
    persist_layer_norm=False,
    openai_gelu=False,
    activation="gelu",
    onnx_safe=False,
    bias=True,
    normalization="layernorm",
    headscale=False,
    transformer_block_type="pre_ln",
    hidden_steps=32,
    parent_model_type=ModelType.encoder_or_decoder,
    layer_type=None,
    chunk_size=64,
    num_self_attention_per_cross_attention=1,
    layer_number_offset=0,  # this is use only for attention norm_factor scaling
    megatron_legacy=False,
    normalize_attention_scores=True,
    sequence_parallel=False,
    num_moe_experts=1,
    moe_frequency=1,
    moe_dropout=0.0,
    turn_off_rop=False,  # turn off the RoP positional embedding
    version=1,  # model version
    position_embedding_type='learned_absolute',
    use_flash_attention=False,
):
    """Build language model and return along with the key to save."""

    if kv_channels is None:
        assert (
            hidden_size % num_attention_heads == 0
        ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
        kv_channels = hidden_size // num_attention_heads

    if init_method is None:
        init_method = init_method_normal(init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

    if arch == "transformer":
        # Language encoder.
        encoder = MegatronTransformerEncoderModule(
            config=config,
            init_method=init_method,
            output_layer_init_method=scaled_init_method,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            encoder_attn_mask_type=encoder_attn_mask_type,
            pre_process=pre_process,
            post_process=post_process,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            layernorm_epsilon=layernorm_epsilon,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            activation=activation,
            bias=bias,
            normalization=normalization,
            transformer_block_type=transformer_block_type,
            headscale=headscale,
            parent_model_type=parent_model_type,
            megatron_legacy=megatron_legacy,
            normalize_attention_scores=normalize_attention_scores,
            num_moe_experts=num_moe_experts,
            moe_frequency=moe_frequency,
            moe_dropout=moe_dropout,
            position_embedding_type=position_embedding_type,
            use_flash_attention=use_flash_attention,
        )
    elif arch == "retro":
        encoder = MegatronRetrievalTransformerEncoderModule(
            config=config,
            init_method=init_method,
            output_layer_init_method=scaled_init_method,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            layer_type=layer_type,
            ffn_hidden_size=ffn_hidden_size,
            pre_process=pre_process,
            post_process=post_process,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            layernorm_epsilon=layernorm_epsilon,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            activation=activation,
            bias=bias,
            normalization=normalization,
            transformer_block_type=transformer_block_type,
            parent_model_type=parent_model_type,
            chunk_size=chunk_size,
            layer_number_offset=layer_number_offset,
            megatron_legacy=megatron_legacy,
            normalize_attention_scores=normalize_attention_scores,
            turn_off_rop=turn_off_rop,
            version=version,
        )
    elif arch == "perceiver":
        encoder = MegatronPerceiverEncoderModule(
            config=config,
            init_method=init_method,
            output_layer_init_method=scaled_init_method,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            encoder_attn_mask_type=encoder_attn_mask_type,
            pre_process=pre_process,
            post_process=post_process,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            layernorm_epsilon=layernorm_epsilon,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            activation=activation,
            bias=bias,
            normalization=normalization,
            transformer_block_type=transformer_block_type,
            headscale=headscale,
            parent_model_type=parent_model_type,
            hidden_steps=hidden_steps,
            num_self_attention_per_cross_attention=num_self_attention_per_cross_attention,
            megatron_legacy=megatron_legacy,
            normalize_attention_scores=normalize_attention_scores,
        )
    else:
        raise ValueError(f"Unknown encoder arch = {arch}. Available encoder arch = {AVAILABLE_ENCODERS}")

    return encoder
