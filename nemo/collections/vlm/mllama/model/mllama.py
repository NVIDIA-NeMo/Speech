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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed
from megatron.core.transformer import TransformerConfig
from torch import Tensor

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.vlm.mllama.model.base import (
    CrossAttentionTextConfig,
    CrossAttentionVisionConfig,
    MLlamaModel,
    MLlamaModelConfig,
)
from nemo.lightning import io, teardown
from nemo.lightning.io.state import _ModelState
from nemo.lightning.pytorch.utils import dtype_from_hf

# pylint: disable=C0115,C0116,C0301


@dataclass
class MLlamaConfig11B(MLlamaModelConfig):
    language_model_config: Optional[TransformerConfig] = field(default_factory=lambda: CrossAttentionTextConfig())
    vision_model_config: Optional[TransformerConfig] = field(
        default_factory=lambda: CrossAttentionVisionConfig(vision_chunk_size=448)
    )


@dataclass
class MLlamaConfig11BInstruct(MLlamaModelConfig):
    language_model_config: Optional[TransformerConfig] = field(default_factory=lambda: CrossAttentionTextConfig())
    vision_model_config: Optional[TransformerConfig] = field(
        default_factory=lambda: CrossAttentionVisionConfig(vision_chunk_size=560)
    )


@dataclass
class MLlamaConfig90B(MLlamaModelConfig):
    language_model_config: Optional[TransformerConfig] = field(
        default_factory=lambda: CrossAttentionTextConfig(
            hidden_size=8192,
            ffn_hidden_size=28672,
            num_attention_heads=64,
            num_layers=80,
            num_cross_attention_layers=20,
        )
    )
    vision_model_config: Optional[TransformerConfig] = field(
        default_factory=lambda: CrossAttentionVisionConfig(vision_chunk_size=560, text_hidden_size=8192)
    )


@dataclass
class MLlamaConfig90BInstruct(MLlamaConfig90B):
    pass


@io.model_importer(MLlamaModel, "hf")
class HFMLlamaImporter(io.ModelConnector["MLlamaModel", MLlamaModel]):
    def init(self) -> MLlamaModel:
        return MLlamaModel(self.config, tokenizer=self.tokenizer)

    def local_path(self, base_path: Optional[Path] = None) -> Path:
        # note: this entire function is for debugging
        output_path = super().local_path(base_path)
        return output_path

    def apply(self, output_path: Path) -> Path:
        from transformers import MllamaForConditionalGeneration

        source = MllamaForConditionalGeneration.from_pretrained(str(self), torch_dtype='auto')

        state_dict = _rename_xattn_layer_nums_hf(source.state_dict())
        source = _ModelState(state_dict)
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Mllama model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {}
        transforms = []
        mapping.update(
            {
                "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
                "language_model.model.xattn_layers.*.cross_attn.o_proj.weight": "language_model.decoder.xattn_layers.*.cross_attention.linear_proj.weight",
                "language_model.model.xattn_layers.*.cross_attn.q_proj.weight": "language_model.decoder.xattn_layers.*.cross_attention.linear_q.weight",
                "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
                "language_model.lm_head.weight": "language_model.output_layer.weight",
                "language_model.model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "language_model.model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
                "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "language_model.model.xattn_layers.*.cross_attn.k_norm.weight": "language_model.decoder.xattn_layers.*.cross_attention.k_layernorm.weight",
                "language_model.model.xattn_layers.*.input_layernorm.weight": "language_model.decoder.xattn_layers.*.cross_attention.linear_q.layer_norm_weight",
                "language_model.model.xattn_layers.*.cross_attn.q_norm.weight": "language_model.decoder.xattn_layers.*.cross_attention.q_layernorm.weight",
                "language_model.model.xattn_layers.*.post_attention_layernorm.weight": "language_model.decoder.xattn_layers.*.mlp.linear_fc1.layer_norm_weight",
                "language_model.model.xattn_layers.*.mlp.down_proj.weight": "language_model.decoder.xattn_layers.*.mlp.linear_fc2.weight",
            }
        )

        transforms.extend(
            [
                io.state_transform(
                    source_key="language_model.model.xattn_layers.*.cross_attn_attn_gate",
                    target_key="language_model.decoder.xattn_layers.*.gate_attn",
                    fn=_import_gate,
                ),
                io.state_transform(
                    source_key="language_model.model.xattn_layers.*.cross_attn_mlp_gate",
                    target_key="language_model.decoder.xattn_layers.*.gate_ffn",
                    fn=_import_gate,
                ),
                io.state_transform(
                    source_key=(
                        "language_model.model.layers.*.self_attn.q_proj.weight",
                        "language_model.model.layers.*.self_attn.k_proj.weight",
                        "language_model.model.layers.*.self_attn.v_proj.weight",
                    ),
                    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    fn=_import_text_qkv,
                ),
                io.state_transform(
                    source_key=(
                        "language_model.model.layers.*.mlp.gate_proj.weight",
                        "language_model.model.layers.*.mlp.up_proj.weight",
                    ),
                    target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    fn=_import_simple_concat,
                ),
                io.state_transform(
                    source_key=(
                        "language_model.model.xattn_layers.*.cross_attn.k_proj.weight",
                        "language_model.model.xattn_layers.*.cross_attn.v_proj.weight",
                    ),
                    target_key="language_model.decoder.xattn_layers.*.cross_attention.linear_kv.weight",
                    fn=_import_text_kv,
                ),
                io.state_transform(
                    source_key=(
                        "language_model.model.xattn_layers.*.mlp.gate_proj.weight",
                        "language_model.model.xattn_layers.*.mlp.up_proj.weight",
                    ),
                    target_key="language_model.decoder.xattn_layers.*.mlp.linear_fc1.weight",
                    fn=_import_simple_concat,
                ),
                io.state_transform(
                    source_key="language_model.model.embed_tokens.weight",
                    target_key=(
                        "language_model.embedding.word_embeddings.weight",
                        "language_model.learnable_embedding.weight",
                    ),
                    fn=_import_embedding_hf,
                ),
            ]
        )

        v = "vision_model.vision_encoder"
        mapping.update(
            {
                "vision_model.global_transformer.layers.*.self_attn.o_proj.weight": f"{v}.global_transformer.layers.*.self_attention.linear_proj.weight",
                "vision_model.global_transformer.layers.*.gate_attn": f"{v}.global_transformer.layers.*.gate_attn",
                "vision_model.global_transformer.layers.*.gate_ffn": f"{v}.global_transformer.layers.*.gate_ffn",
                "vision_model.global_transformer.layers.*.input_layernorm.bias": f"{v}.global_transformer.layers.*.input_layernorm.bias",
                "vision_model.global_transformer.layers.*.input_layernorm.weight": f"{v}.global_transformer.layers.*.input_layernorm.weight",
                "vision_model.global_transformer.layers.*.post_attention_layernorm.bias": f"{v}.global_transformer.layers.*.pre_mlp_layernorm.bias",
                "vision_model.global_transformer.layers.*.post_attention_layernorm.weight": f"{v}.global_transformer.layers.*.pre_mlp_layernorm.weight",
                "vision_model.global_transformer.layers.*.mlp.fc1.bias": f"{v}.global_transformer.layers.*.mlp.linear_fc1.bias",
                "vision_model.global_transformer.layers.*.mlp.fc1.weight": f"{v}.global_transformer.layers.*.mlp.linear_fc1.weight",
                "vision_model.global_transformer.layers.*.mlp.fc2.bias": f"{v}.global_transformer.layers.*.mlp.linear_fc2.bias",
                "vision_model.global_transformer.layers.*.mlp.fc2.weight": f"{v}.global_transformer.layers.*.mlp.linear_fc2.weight",
                "vision_model.transformer.layers.*.self_attn.o_proj.weight": f"{v}.transformer.layers.*.self_attention.linear_proj.weight",
                "vision_model.transformer.layers.*.input_layernorm.bias": f"{v}.transformer.layers.*.input_layernorm.bias",
                "vision_model.transformer.layers.*.input_layernorm.weight": f"{v}.transformer.layers.*.input_layernorm.weight",
                "vision_model.transformer.layers.*.post_attention_layernorm.bias": f"{v}.transformer.layers.*.pre_mlp_layernorm.bias",
                "vision_model.transformer.layers.*.post_attention_layernorm.weight": f"{v}.transformer.layers.*.pre_mlp_layernorm.weight",
                "vision_model.transformer.layers.*.mlp.fc1.bias": f"{v}.transformer.layers.*.mlp.linear_fc1.bias",
                "vision_model.transformer.layers.*.mlp.fc1.weight": f"{v}.transformer.layers.*.mlp.linear_fc1.weight",
                "vision_model.transformer.layers.*.mlp.fc2.bias": f"{v}.transformer.layers.*.mlp.linear_fc2.bias",
                "vision_model.transformer.layers.*.mlp.fc2.weight": f"{v}.transformer.layers.*.mlp.linear_fc2.weight",
                "vision_model.class_embedding": f"{v}.class_embedding",
                "vision_model.gated_positional_embedding.embedding": f"{v}.positional_embedding",
                "vision_model.gated_positional_embedding.tile_embedding.weight": f"{v}.gated_tile_positional_embedding.weight",
                "vision_model.gated_positional_embedding.gate": f"{v}.gated_positional_embedding_gate",
                "vision_model.layernorm_post.bias": f"{v}.ln_post.bias",
                "vision_model.layernorm_post.weight": f"{v}.ln_post.weight",
                "vision_model.layernorm_pre.bias": f"{v}.ln_pre.bias",
                "vision_model.layernorm_pre.weight": f"{v}.ln_pre.weight",
                "vision_model.post_tile_positional_embedding.embedding.weight": f"{v}.post_tile_pos_embed.embedding.weight",
                "vision_model.post_tile_positional_embedding.gate": f"{v}.post_tile_pos_embed.gate",
                "vision_model.pre_tile_positional_embedding.embedding.weight": f"{v}.pre_tile_pos_embed.embedding.weight",
                "vision_model.pre_tile_positional_embedding.gate": f"{v}.pre_tile_pos_embed.gate",
                "multi_modal_projector.bias": "vision_model.vision_projection.encoder.bias",
                "multi_modal_projector.weight": "vision_model.vision_projection.encoder.weight",
            }
        )
        transforms.extend(
            [
                io.state_transform(
                    source_key=(
                        "vision_model.global_transformer.layers.*.self_attn.q_proj.weight",
                        "vision_model.global_transformer.layers.*.self_attn.k_proj.weight",
                        "vision_model.global_transformer.layers.*.self_attn.v_proj.weight",
                    ),
                    target_key=(f"{v}.global_transformer.layers.*.self_attention.linear_qkv.weight"),
                    fn=_import_vision_qkv,
                ),
                io.state_transform(
                    source_key=(
                        "vision_model.transformer.layers.*.self_attn.q_proj.weight",
                        "vision_model.transformer.layers.*.self_attn.k_proj.weight",
                        "vision_model.transformer.layers.*.self_attn.v_proj.weight",
                    ),
                    target_key=(f"{v}.transformer.layers.*.self_attention.linear_qkv.weight"),
                    fn=_import_vision_qkv,
                ),
                io.state_transform(
                    source_key="vision_model.patch_embedding.weight",
                    target_key=f"{v}.conv1._linear.weight",
                    fn=_import_patch_embedding_hf,
                ),
            ]
        )

        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> MLlamaModelConfig:
        from transformers import AutoConfig

        source = AutoConfig.from_pretrained(str(self))

        return MLlamaModelConfig(
            language_model_config=self._language_model_config(source),
            vision_model_config=self._vision_model_config(source),
        )

    def _language_model_config(self, source) -> Optional[CrossAttentionTextConfig]:
        def _calculate_num_layers(num_hidden_layers, cross_attention_layers):
            return num_hidden_layers - len(cross_attention_layers)

        return CrossAttentionTextConfig(
            rotary_base=source.text_config.rope_theta,
            seq_length=8192,
            num_layers=_calculate_num_layers(
                source.text_config.num_hidden_layers, source.text_config.cross_attention_layers
            ),
            num_cross_attention_layers=len(source.text_config.cross_attention_layers),
            hidden_size=source.text_config.hidden_size,
            ffn_hidden_size=source.text_config.intermediate_size,
            num_attention_heads=source.text_config.num_attention_heads,
            num_query_groups=source.text_config.num_key_value_heads,
            vocab_size=source.text_config.vocab_size,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

    def _vision_model_config(self, source) -> Optional[CrossAttentionVisionConfig]:
        return CrossAttentionVisionConfig(
            num_layers=source.vision_config.num_hidden_layers,
            hidden_size=source.vision_config.hidden_size,
            num_attention_heads=source.vision_config.attention_heads,
            vision_chunk_size=source.vision_config.image_size,
            vision_max_num_chunks=source.vision_config.max_num_tiles,
            text_hidden_size=source.text_config.hidden_size,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )


def _rename_xattn_layer_nums_hf(source: Dict):
    def convert_layer_num(match):
        layer_num = int(match.group(1))
        cross_num = (layer_num - 3) // (cross_attention_frequency + 1)
        if (layer_num - 3) % (cross_attention_frequency + 1) == 0:
            new_layer_num = cross_num * cross_attention_frequency + 3
            return f'xattn_layers.{new_layer_num}.'

        new_layer_num = layer_num - cross_num - 1
        return f'layers.{new_layer_num}.'

    cross_attention_frequency = 4

    output_dict = {}
    for k, v in source.items():
        if "language_model" in k:
            output_dict[re.sub(r"layers\.(\d+)\.", convert_layer_num, k)] = v
        else:
            output_dict[k] = v
    return output_dict


def _import_embedding_hf(a):
    return torch.split(a, a.shape[0] - 8, dim=0)


def _import_patch_embedding_hf(a):
    return a.reshape(a.shape[0], -1)


def _import_gate(gate):
    return gate[0:1]


def _import_vision_qkv(ctx: io.TransformCTX, q, k, v):
    vision_config = ctx.target.config.vision_model_config

    head_num = vision_config.num_attention_heads
    num_query_groups = vision_config.num_query_groups
    head_size = vision_config.kv_channels
    hidden_size = vision_config.hidden_size
    return _merge_qkv(q, k, v, head_num, num_query_groups, head_size, hidden_size)


def _import_text_qkv(ctx: io.TransformCTX, q, k, v):
    text_config = ctx.target.config.language_model_config

    head_num = text_config.num_attention_heads
    num_query_groups = text_config.num_query_groups
    head_size = text_config.kv_channels
    hidden_size = text_config.hidden_size
    return _merge_qkv(q, k, v, head_num, num_query_groups, head_size, hidden_size)


def _import_text_kv(ctx: io.TransformCTX, k, v):
    text_config = ctx.target.config.language_model_config

    head_num = text_config.num_attention_heads
    num_query_groups = text_config.num_query_groups
    head_size = text_config.kv_channels
    hidden_size = text_config.hidden_size
    return _merge_kv(k, v, head_num, num_query_groups, head_size, hidden_size)


def _merge_kv(k: Tensor, v: Tensor, head_num: int, num_query_groups: int, head_size: int, hidden_size: int):
    old_tensor_shape = k.size()
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    kv_weights = torch.stack((k, v), dim=1)
    kv_weights = kv_weights.reshape(-1, *new_kv_tensor_shape[1:])
    assert kv_weights.ndim == 3, kv_weights.shape
    assert kv_weights.shape[0] == 2 * num_query_groups, kv_weights.shape
    assert kv_weights.shape[1] == head_size, kv_weights.shape
    assert kv_weights.shape[2] == old_tensor_shape[1], kv_weights.shape

    kv_weights = kv_weights.reshape([head_size * 2 * num_query_groups, hidden_size])
    return kv_weights


def _merge_qkv(
    q: Tensor, k: Tensor, v: Tensor, head_num: int, num_query_groups: int, head_size: int, hidden_size: int
):
    heads_per_group = head_num // num_query_groups
    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


def _split_qkv(qkv, head_num: int, num_query_groups: int, head_size: int, hidden_size: int):
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = qkv.reshape([qkv_total_dim, head_size, hidden_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

    return q_proj, k_proj, v_proj


def _import_simple_concat(a, b):
    # for both (w1, w3) -> fc1, and (wk, wv) -> wkv
    return torch.cat((a, b), dim=0)


def _rename_xattn_layer_nums(source: Dict):
    def convert_layer_num(match):
        new_layer_num = int(match.group(1)) * 4 + 3
        return f'.{new_layer_num}.'

    output_dict = {}
    for k, v in source.items():
        if "cross_attention_layers" in k:
            output_dict[re.sub(r"\.(\d+)\.", convert_layer_num, k)] = v
        else:
            output_dict[k] = v
    return output_dict
