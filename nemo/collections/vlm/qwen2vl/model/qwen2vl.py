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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

import torch
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm import Qwen2Config, Qwen2Config1P5B, Qwen2Config7B, Qwen2Config72B
from nemo.collections.vlm.qwen2vl.model.base import Qwen2VLConfig, Qwen2VLModel, Qwen2VLVisionConfig
from nemo.collections.vlm.vision import MultimodalProjectorConfig
from nemo.lightning import io, teardown

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


# Note: these Qwen2VL configs are copied from the corresponding HF model. You may need to modify the parameter for
# your own needs
@dataclass
class Qwen2VLConfig2B(Qwen2VLConfig):
    """Qwen2VL Config 2B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen2Config1P5B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen2VLVisionConfig(num_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(input_size=5120, hidden_size=1536, ffn_hidden_size=5120)
    )


@dataclass
class Qwen2VLConfig7B(Qwen2VLConfig):
    """Qwen2VL Config 7B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen2Config7B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen2VLVisionConfig(num_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(input_size=5120, hidden_size=3584, ffn_hidden_size=5120)
    )


@dataclass
class Qwen2VLConfig72B(Qwen2VLConfig):
    """Qwen2VL Config 72B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen2Config72B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen2VLVisionConfig(num_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(input_size=5120, hidden_size=8192, ffn_hidden_size=5120)
    )


@io.model_importer(Qwen2VLModel, "hf")
class HFQwen2VLImporter(io.ModelConnector["Qwen2VLForConditionalGeneration", Qwen2VLModel]):
    """Qwen2VL Model HF Importer"""

    def init(self) -> Qwen2VLModel:
        # pylint: disable=C0115,C0116
        return Qwen2VLModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        # pylint: disable=C0115,C0116
        from transformers import Qwen2VLForConditionalGeneration

        source = Qwen2VLForConditionalGeneration.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        print(f"Converted Qwen2VL model to Nemo, saving to {output_path}")
        # for name, param in target.named_parameters():
        #     print(name, param.shape)
        self.nemo_save(output_path, trainer)

        print(f"Converted Qwen2VL model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116,C0301
        mapping = {
            "visual.patch_embed.proj.weight": "vision_model.conv1.weight",
            "visual.blocks.*.norm1.weight": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "visual.blocks.*.norm1.bias": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "visual.blocks.*.norm2.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "visual.blocks.*.norm2.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "visual.blocks.*.attn.proj.weight": "vision_model.decoder.layers.*.self_attention.linear_proj.weight",
            "visual.blocks.*.attn.proj.bias": "vision_model.decoder.layers.*.self_attention.linear_proj.bias",
            "visual.blocks.*.mlp.fc1.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.weight",
            "visual.blocks.*.mlp.fc1.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.bias",
            "visual.blocks.*.mlp.fc2.weight": "vision_model.decoder.layers.*.mlp.linear_fc2.weight",
            "visual.blocks.*.mlp.fc2.bias": "vision_model.decoder.layers.*.mlp.linear_fc2.bias",
            "visual.merger.ln_q.weight": "vision_model.decoder.final_layernorm.weight",
            "visual.merger.ln_q.bias": "vision_model.decoder.final_layernorm.bias",
            "model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "language_model.decoder.final_layernorm.weight",
            "lm_head.weight": "language_model.output_layer.weight",
        }

        if "vision_projection.encoder.linear_fc1.weight" in target.module.state_dict().keys():
            mapping.update(
                {
                    "visual.merger.mlp.0.weight": "vision_projection.encoder.linear_fc1.weight",
                    "visual.merger.mlp.0.bias": "vision_projection.encoder.linear_fc1.bias",
                    "visual.merger.mlp.2.weight": "vision_projection.encoder.linear_fc2.weight",
                    "visual.merger.mlp.2.bias": "vision_projection.encoder.linear_fc2.bias",
                }
            )
        elif "vision_projection.0.weight" in target.module.state_dict().keys():
            mapping.update(
                {
                    "visual.merger.mlp.0.weight": "vision_projection.0.weight",
                    "visual.merger.mlp.0.bias": "vision_projection.0.bias",
                    "visual.merger.mlp.2.weight": "vision_projection.2.weight",
                    "visual.merger.mlp.2.bias": "vision_projection.2.bias",
                }
            )
        else:
            raise KeyError("Unable to map vision projection keys.")

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                _import_language_qkv,
                _import_language_qkv_bias,
                _import_vision_qkv,
                _import_vision_qkv_bias,
                _import_linear_fc1,
            ],
        )

    @property
    def tokenizer(self) -> "AutoTokenizer":
        # pylint: disable=C0115,C0116
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> Qwen2VLConfig:
        # pylint: disable=C0115,C0116
        from transformers import Qwen2VLConfig as HFQwen2VLConfig

        hf_config = HFQwen2VLConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            # pylint: disable=C0115,C0116
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        language_transformer_config = Qwen2Config(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            num_query_groups=hf_config.num_key_value_heads,
            rotary_base=hf_config.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(hf_config.vocab_size),
            share_embeddings_and_output_weights=False,
            vocab_size=hf_config.vocab_size,
        )

        # Use MCore instead of Pytorch
        vision_transformer_config = Qwen2VLVisionConfig()
        merge_hidden_size = hf_config.vision_config.embed_dim * (hf_config.vision_config.spatial_merge_size**2)
        vision_projection_config = MultimodalProjectorConfig(
            input_size=merge_hidden_size,
            hidden_size=hf_config.vision_config.hidden_size,
            ffn_hidden_size=merge_hidden_size,
            projector_type="mcore_mlp",
        )

        output = Qwen2VLConfig(
            language_transformer_config=language_transformer_config,
            vision_transformer_config=vision_transformer_config,
            vision_projection_config=vision_projection_config,
            vision_feature_layer=-1,
        )

        return output


def import_qkv(q, k, v, head_num, num_query_groups, heads_per_group, hidden_size, head_size):
    # pylint: disable=C0115,C0116
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


@io.state_transform(
    source_key=("visual.blocks.*.attn.qkv.weight",),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_vision_qkv(ctx: io.TransformCTX, hf_qkv_weights):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.vision_transformer_config

    slice = int(hf_qkv_weights.shape[0] / 3)
    assert slice == megatron_config.hidden_size
    q = hf_qkv_weights[:slice, :]
    k = hf_qkv_weights[slice : slice * 2, :]
    v = hf_qkv_weights[slice * 2 :, :]

    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )


@io.state_transform(
    source_key=("visual.blocks.*.attn.qkv.bias",),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_vision_qkv_bias(ctx: io.TransformCTX, hf_qkv_bias):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.vision_transformer_config

    slice = int(hf_qkv_bias.shape[0] / 3)
    assert slice == megatron_config.hidden_size

    q_bias = hf_qkv_bias[:slice]
    k_bias = hf_qkv_bias[slice : slice * 2]
    v_bias = hf_qkv_bias[slice * 2 :]

    return import_qkv(
        q_bias.unsqueeze(-1),
        k_bias.unsqueeze(-1),
        v_bias.unsqueeze(-1),
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=1,
        head_size=megatron_config.kv_channels,
    ).squeeze(-1)


@io.state_transform(
    source_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_language_qkv(ctx: io.TransformCTX, q, k, v):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.language_transformer_config
    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )


@io.state_transform(
    source_key=(
        "model.layers.*.self_attn.q_proj.bias",
        "model.layers.*.self_attn.k_proj.bias",
        "model.layers.*.self_attn.v_proj.bias",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_language_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.language_transformer_config
    return import_qkv(
        q_bias.unsqueeze(-1),
        k_bias.unsqueeze(-1),
        v_bias.unsqueeze(-1),
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=1,
        head_size=megatron_config.kv_channels,
    ).squeeze(-1)


@io.state_transform(
    source_key=("vision_model.embeddings.class_embedding",),
    target_key="vision_model.class_token",
)
def _import_cls_token(ctx: io.TransformCTX, cls_token):
    # pylint: disable=C0115,C0116
    return cls_token.reshape(1, 1, -1)


@io.state_transform(
    source_key=(
        "model.layers.*.mlp.gate_proj.weight",
        "model.layers.*.mlp.up_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    # pylint: disable=C0115,C0116
    return torch.cat((down, gate), axis=0)
