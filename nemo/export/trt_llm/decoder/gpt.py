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

from tensorrt_llm.layers import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.models.gpt.model import GPTDecoderLayer
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


class GPTDecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    """The GPT2 implementation of the DecoderLayerConfigBuilder."""

    @override
    def hidden_act_fn(self, layer):
        return layer.mlp.act

    @override
    def infer_num_attention_heads(self, layer):
        return layer.attn.num_heads

    @override
    def infer_max_position_embeddings(self, layer):
        return layer.attn.bias.shape[2]

    @override
    def build_input_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.ln_1, dtype=self.dtype)

    @override
    def build_attention(self, layer) -> AttentionConfig:
        config = AttentionConfig()
        config.qkv = LinearConfig.from_qkv_nn_modules(
            [layer.attn.c_attn], rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )

        config.dense = LinearConfig.from_nn_module(
            layer.attn.c_proj, LINEAR_ROW, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )

        return config

    @override
    def build_mlp(self, layer) -> MLPConfig:
        config = MLPConfig()
        config.fc = LinearConfig.from_nn_module(
            layer.mlp.c_fc, LINEAR_COLUMN, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )
        config.proj = LinearConfig.from_nn_module(
            layer.mlp.c_proj, LINEAR_ROW, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )

        return config

    @override
    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        return LayernormConfig.from_nn_module(layer.ln_2, dtype=self.dtype)


class GPTDecoderLayerBuilder(DecoderLayerBuilder):
    """The GPT implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        rotary_pct = layer.rotary_pct

        position_embedding_type = "rope_gpt_neox" if layer.position_embedding_type == "rope" else "learned_absolute"

        assert not (position_embedding_type == "rope_gpt_neox" and rotary_pct == 0.0)

        bias_qkv = layer.attention.qkv.bias is not None

        rotary_scaling = None
        if layer.rotary_scaling is not None:
            rotary_scaling = {"type": "linear", "factor": float(layer.rotary_scaling)}

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
            hidden_act=self.hidden_act,
            intermediate_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            norm_epsilon=layer.norm_epsilon,
            position_embedding_type=position_embedding_type,
            world_size=self.tensor_parallel,
            tp_size=self.tensor_parallel,
            pp_size=1,
            max_lora_rank=layer.max_lora_rank,
            quantization=QuantConfig(),
        )

        config.set_if_not_exist('hidden_act', self.hidden_act)
        config.set_if_not_exist('apply_query_key_layer_scaling', False)
        config.set_if_not_exist('bias', bias_qkv)
        config.set_if_not_exist('rotary_base', layer.rotary_base)
        config.set_if_not_exist('rotary_scaling', rotary_scaling)
        config.set_if_not_exist('rotary_pct', rotary_pct)
        config.set_if_not_exist('moe_num_experts', 0)

        return GPTDecoderLayer(config=config, layer_idx=self.layer_id,)
