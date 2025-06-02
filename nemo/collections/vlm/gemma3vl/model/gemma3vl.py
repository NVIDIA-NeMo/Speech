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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from nemo.collections.llm.gpt.model.gemma3 import (
    Gemma3Config,
    Gemma3Config1B,
    Gemma3Config4B,
    Gemma3Config12B,
    Gemma3Config27B,
)
from nemo.collections.vlm.gemma3vl.model.base import Gemma3VLConfig, Gemma3VLModel
from nemo.collections.vlm.gemma3vl.model.vision import Gemma3VLMultimodalProjectorConfig, Gemma3VLVisionConfig
from nemo.collections.vlm.neva.model.llava import export_qkv, export_qkv_bias, import_qkv
from nemo.lightning import io, teardown
from nemo.lightning.io.state import TransformFns

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


@dataclass
class Gemma3VLConfig4B(Gemma3VLConfig):
    """Gemma3 VL config 4B"""

    language_transformer_config: Gemma3Config = field(default_factory=lambda: Gemma3Config4B())
    vision_transformer_config: Gemma3VLVisionConfig = field(default_factory=lambda: Gemma3VLVisionConfig())
    vision_projection_config: Gemma3VLMultimodalProjectorConfig = field(
        default_factory=lambda: Gemma3VLMultimodalProjectorConfig(input_size=1152, hidden_size=2560)
    )


@dataclass
class Gemma3VLConfig12B(Gemma3VLConfig):
    """Gemma3 VL config 12B"""

    language_transformer_config: Gemma3Config = field(default_factory=lambda: Gemma3Config12B())
    vision_transformer_config: Gemma3VLVisionConfig = field(default_factory=lambda: Gemma3VLVisionConfig())
    vision_projection_config: Gemma3VLMultimodalProjectorConfig = field(
        default_factory=lambda: Gemma3VLMultimodalProjectorConfig(input_size=1152, hidden_size=3840)
    )


@dataclass
class Gemma3VLConfig27B(Gemma3VLConfig):
    """Gemma3 VL config 27B"""

    language_transformer_config: Gemma3Config = field(default_factory=lambda: Gemma3Config27B())
    vision_transformer_config: Gemma3VLVisionConfig = field(default_factory=lambda: Gemma3VLVisionConfig())
    vision_projection_config: Gemma3VLMultimodalProjectorConfig = field(
        default_factory=lambda: Gemma3VLMultimodalProjectorConfig(input_size=1152, hidden_size=5376)
    )


@io.model_importer(Gemma3VLModel, "hf")
class Gemma3VLImporter(io.ModelConnector["Gemma3ForConditionalGeneration", Gemma3VLModel]):
    """Gemma3 VL model HF importer"""

    def init(self) -> Gemma3VLModel:
        return Gemma3VLModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        # pylint: disable=C0115,C0116
        from transformers import Gemma3ForConditionalGeneration

        source = Gemma3ForConditionalGeneration.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted HF Gemma3VL model to NeMo, saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        # pylint: disable=C0301,C0116
        mapping = {
            # vision model
            "vision_tower.vision_model.embeddings.patch_embedding.weight": "vision_model.conv1.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias": "vision_model.conv1.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight": "vision_model.position_embeddings.weight",
            "vision_tower.vision_model.post_layernorm.weight": "vision_model.ln_post.weight",
            "vision_tower.vision_model.post_layernorm.bias": "vision_model.ln_post.bias",
            "vision_tower.vision_model.encoder.layers.*.layer_norm1.weight": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "vision_tower.vision_model.encoder.layers.*.layer_norm1.bias": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.weight": "vision_model.decoder.layers.*.self_attention.linear_proj.weight",
            "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.bias": "vision_model.decoder.layers.*.self_attention.linear_proj.bias",
            "vision_tower.vision_model.encoder.layers.*.layer_norm2.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "vision_tower.vision_model.encoder.layers.*.layer_norm2.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "vision_tower.vision_model.encoder.layers.*.mlp.fc1.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.weight",
            "vision_tower.vision_model.encoder.layers.*.mlp.fc1.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.bias",
            "vision_tower.vision_model.encoder.layers.*.mlp.fc2.weight": "vision_model.decoder.layers.*.mlp.linear_fc2.weight",
            "vision_tower.vision_model.encoder.layers.*.mlp.fc2.bias": "vision_model.decoder.layers.*.mlp.linear_fc2.bias",
            # vision projector
            "multi_modal_projector.mm_soft_emb_norm.weight": "vision_projection.mm_soft_embed_norm.weight",
            # text model
            "language_model.model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
            "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "language_model.model.layers.*.self_attn.q_norm.weight": "language_model.decoder.layers.*.self_attention.q_layernorm.weight",
            "language_model.model.layers.*.self_attn.k_norm.weight": "language_model.decoder.layers.*.self_attention.k_layernorm.weight",
            "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
            "language_model.model.layers.*.post_attention_layernorm.weight": (
                "language_model.decoder.layers.*.self_attention.linear_proj.post_layernorm.weight"
            ),
            "language_model.model.layers.*.pre_feedforward_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "language_model.model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
            "language_model.model.layers.*.post_feedforward_layernorm.weight": (
                "language_model.decoder.layers.*.mlp.linear_fc2.post_layernorm.weight"
            ),
            "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
        }
        transforms = [
            _import_vision_qkv,
            _import_vision_qkv_bias,
            _import_language_qkv,
            io.state_transform(
                source_key="multi_modal_projector.mm_input_projection_weight",
                target_key="vision_projection.proj.weight",
                fn=_vision_projector_permute,
            ),
            io.state_transform(
                source_key=(
                    "language_model.model.layers.*.mlp.gate_proj.weight",
                    "language_model.model.layers.*.mlp.up_proj.weight",
                ),
                target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        # pylint: disable=C0115,C0116
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> Gemma3VLConfig:
        # pylint: disable=C0115,C0116
        from transformers import Gemma3Config as HFGemma3Config

        name = str(self)
        source = HFGemma3Config.from_pretrained(name)
        source_text = source.text_config
        source_vision = source.vision_config

        if source_text.num_hidden_layers == 26:
            language_transformer_config = Gemma3Config1B()
        elif source_text.num_hidden_layers == 34:
            language_transformer_config = Gemma3Config4B()
        elif source_text.num_hidden_layers == 48:
            language_transformer_config = Gemma3Config12B()
        elif source_text.num_hidden_layers == 62:
            language_transformer_config = Gemma3Config27B()
        else:
            raise ValueError(f"Unrecognized import model: {name}")
        vision_transformer_config = Gemma3VLVisionConfig()
        vision_projection_config = Gemma3VLMultimodalProjectorConfig(
            input_size=source_vision.hidden_size,
            hidden_size=source_text.hidden_size,
        )

        output = Gemma3VLConfig(
            language_transformer_config=language_transformer_config,
            vision_transformer_config=vision_transformer_config,
            vision_projection_config=vision_projection_config,
        )
        return output


@io.model_exporter(Gemma3VLModel, "hf")
class Gemma3VLExporter(io.ModelConnector[Gemma3VLModel, "Gemma3ForConditionalGeneration"]):
    """Export Gemma3 VL to HF"""

    def init(self):
        from transformers import Gemma3ForConditionalGeneration
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return Gemma3ForConditionalGeneration._from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        # pylint: disable=C0115,C0116
        target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116,C0301
        mapping = {
            # vision model
            "vision_model.conv1.weight": "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_model.conv1.bias": "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_model.position_embeddings.weight": "vision_tower.vision_model.embeddings.position_embedding.weight",
            "vision_model.ln_post.weight": "vision_tower.vision_model.post_layernorm.weight",
            "vision_model.ln_post.bias": "vision_tower.vision_model.post_layernorm.bias",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "vision_tower.vision_model.encoder.layers.*.layer_norm1.weight",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "vision_tower.vision_model.encoder.layers.*.layer_norm1.bias",
            "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.weight",
            "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "vision_tower.vision_model.encoder.layers.*.layer_norm2.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "vision_tower.vision_model.encoder.layers.*.layer_norm2.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc1.weight": "vision_tower.vision_model.encoder.layers.*.mlp.fc1.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc1.bias": "vision_tower.vision_model.encoder.layers.*.mlp.fc1.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "vision_tower.vision_model.encoder.layers.*.mlp.fc2.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "vision_tower.vision_model.encoder.layers.*.mlp.fc2.bias",
            # vision projector
            "vision_projection.mm_soft_embed_norm.weight": "multi_modal_projector.mm_soft_emb_norm.weight",
            # text model
            "language_model.embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "language_model.model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "language_model.model.layers.*.self_attn.k_norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "language_model.model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "language_model.model.layers.*.post_attention_layernorm.weight"
            ),
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.model.layers.*.pre_feedforward_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": (
                "language_model.model.layers.*.post_feedforward_layernorm.weight"
            ),
            "language_model.decoder.final_layernorm.weight": "language_model.model.norm.weight",
        }

        transforms = [
            _export_vision_qkv,
            _export_vision_qkv_bias,
            _export_language_qkv,
            io.state_transform(
                source_key="vision_projection.proj.weight",
                target_key="multi_modal_projector.mm_input_projection_weight",
                fn=_vision_projector_permute,
            ),
            io.state_transform(
                source_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                target_key=(
                    "language_model.model.layers.*.mlp.gate_proj.weight",
                    "language_model.model.layers.*.mlp.up_proj.weight",
                ),
                fn=TransformFns.split_fc1,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self):
        # pylint: disable=C0115,C0116
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self):
        # pylint: disable=C0115,C0116
        source: Gemma3VLConfig = io.load_context(str(self), subpath="model.config")
        source_text: Gemma3Config = source.language_transformer_config
        source_vision: Gemma3VLVisionConfig = source.vision_transformer_config

        from transformers import Gemma3Config as HFGemma3Config
        from transformers import Gemma3TextConfig as HFGemma3TextConfig
        from transformers import SiglipVisionConfig as HFGemma3VisionConfig

        output_text = HFGemma3TextConfig(
            architectures=["Gemma3ForCausalLM"],
            num_hidden_layers=source_text.num_layers,
            hidden_size=source_text.hidden_size,
            intermediate_size=source_text.ffn_hidden_size,
            num_attention_heads=source_text.num_attention_heads,
            head_dim=source_text.kv_channels,
            hidden_activation="gelu_pytorch_tanh",
            max_position_embeddings=source_text.seq_length,
            initializer_range=source_text.init_method_std,
            rms_norm_eps=source_text.layernorm_epsilon,
            num_key_value_heads=source_text.num_query_groups,
            vocab_size=source_text.vocab_size,
            rope_theta=source_text.rotary_base[1],
            rope_local_base_freq=source_text.rotary_base[0],
        )
        if source_text.num_layers == 62:  # 27B
            output_text.query_pre_attn_scalar = 168
        else:
            output_text.query_pre_attn_scalar = output_text.head_dim

        output_vision = HFGemma3VisionConfig(
            hidden_size=source_vision.hidden_size,
            image_size=source_vision.img_h,
            intermediate_size=source_vision.ffn_hidden_size,
            num_attention_heads=source_vision.num_attention_heads,
            num_hidden_layers=source_vision.num_layers,
            patch_size=source_vision.patch_dim,
            vision_use_head=False,
        )

        output = HFGemma3Config(text_config=output_text, vision_config=output_vision)
        return output


@staticmethod
def _vision_projector_permute(ctx: io.TransformCTX, x):
    return torch.permute(x, (1, 0))


@io.state_transform(
    source_key=(
        "language_model.model.layers.*.self_attn.q_proj.weight",
        "language_model.model.layers.*.self_attn.k_proj.weight",
        "language_model.model.layers.*.self_attn.v_proj.weight",
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
    source_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "language_model.model.layers.*.self_attn.q_proj.weight",
        "language_model.model.layers.*.self_attn.k_proj.weight",
        "language_model.model.layers.*.self_attn.v_proj.weight",
    ),
)
def _export_language_qkv(ctx: io.TransformCTX, qkv):
    # pylint: disable=C0115,C0116
    hf_config = ctx.target.config.text_config
    return export_qkv(
        qkv,
        head_num=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_key_value_heads,
        heads_per_group=hf_config.num_attention_heads // hf_config.num_key_value_heads,
        hidden_size=hf_config.hidden_size,
        head_size=hf_config.head_dim,
    )


@io.state_transform(
    source_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.weight",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_vision_qkv(ctx: io.TransformCTX, q, k, v):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.vision_transformer_config
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
    source_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.weight",
    ),
)
def _export_vision_qkv(ctx: io.TransformCTX, qkv):
    # pylint: disable=C0115,C0116
    hf_config = ctx.target.config.vision_config
    return export_qkv(
        qkv,
        head_num=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_attention_heads,
        heads_per_group=1,
        hidden_size=hf_config.hidden_size,
        head_size=hf_config.hidden_size // hf_config.num_attention_heads,
    )


@io.state_transform(
    source_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.bias",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_vision_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.vision_transformer_config
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
    source_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
    target_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.bias",
    ),
)
def _export_vision_qkv_bias(ctx: io.TransformCTX, qkv_bias):
    # pylint: disable=C0115,C0116
    hf_config = ctx.target.config.vision_config
    return export_qkv_bias(
        qkv_bias,
        head_num=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_attention_heads,
        heads_per_group=1,
        head_size=hf_config.hidden_size // hf_config.num_attention_heads,
    )
