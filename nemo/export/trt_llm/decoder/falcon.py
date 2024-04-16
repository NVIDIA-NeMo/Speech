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


from typing import Optional

from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.models.falcon.model import FalconDecoderLayer
from tensorrt_llm.models.modeling_utils import PretrainedConfig, QuantConfig
from typing_extensions import override

from nemo.export.trt_llm.decoder.decoder import DecoderLayerBuilder, DecoderLayerConfigBuilder
from nemo.export.trt_llm.model_config import (
    LINEAR_COLUMN,
    LINEAR_ROW,
    AttentionConfig,
    LayernormConfig,
    LinearConfig,
    MLPConfig,
)


class FALCONDecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    """The FALCON implementation of the DecoderLayerConfigBuilder."""

    @override
    def hidden_act_fn(self, layer):
        return layer.mlp.act_fn

    @override
    def infer_num_attention_heads(self, layer):
        return layer.self_attn.num_heads

    @override
    def infer_num_kv_heads(self, layer):
        return layer.self_attn.num_key_value_heads

    @override
    def infer_max_position_embeddings(self, layer):
        return layer.self_attn.max_position_embeddings

    @override
    def build_input_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.input_layernorm, dtype=self.dtype)

    @override
    def build_mlp_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.mlp_layernorm, dtype=self.dtype)

    @override
    def build_attention(self, layer) -> AttentionConfig:
        config = AttentionConfig()
        config.qkv = LinearConfig.from_qkv_nn_modules(
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        config.dense = LinearConfig.from_nn_module(
            layer.self_attn.o_proj, LINEAR_ROW, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )

        return config

    @override
    def build_mlp(self, layer) -> MLPConfig:
        config = MLPConfig()
        config.fc = LinearConfig.from_nn_module(
            layer.mlp.gate_proj, LINEAR_COLUMN, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )
        config.proj = LinearConfig.from_nn_module(
            layer.mlp.down_proj, LINEAR_ROW, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )
        config.gate = LinearConfig.from_nn_module(
            layer.mlp.up_proj, LINEAR_COLUMN, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )

        return config

    @override
    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        return LayernormConfig.from_nn_module(layer.post_attention_layernorm, dtype=self.dtype)


class FALCONDecoderLayerBuilder(DecoderLayerBuilder):
    """The FALCON implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        # Falcon 7B: parallel_attention=True, new_decoder_architecture=False
        # Falcon 40B/180B: parallel_attention=True, new_decoder_architecture=True
        config = PretrainedConfig(
            architecture=None,
            dtype=self.dtype,
            logits_dtype=self.dtype,
            vocab_size=layer.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=non_gated_version(self.hidden_act),
            intermediate_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            norm_epsilon=layer.norm_epsilon,
            position_embedding_type="rope_gpt_neox",
            world_size=self.tensor_parallel,
            tp_size=self.tensor_parallel,
            pp_size=1,
            quantization=QuantConfig(),
            max_lora_rank=layer.max_lora_rank,
            use_parallel_embedding=False,
        )

        # No other way to pass in model variant config, determine model variant by num_layers (7B: 32 layers)
        config.set_if_not_exist('new_decoder_architecture', False if self.num_layers == 32 else True)
        config.set_if_not_exist('parallel_attention', True)
        config.set_if_not_exist('layernorm_epsilon', 1e-5)
        config.set_if_not_exist('bias', False)
        config.set_if_not_exist('moe_num_experts', 0)

        return FalconDecoderLayer(config=config, layer_idx=self.layer_id,)
