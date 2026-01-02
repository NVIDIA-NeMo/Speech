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
from typing import Dict, Optional, Tuple

import torch
import torch.distributed
from megatron.core.transformer import TransformerConfig
from torch import Tensor
from transformers import MllamaConfig as HFMllamaConfig
from transformers import MllamaForConditionalGeneration
from transformers.models.mllama.configuration_mllama import MllamaTextConfig, MllamaVisionConfig

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.vlm.mllama.model.base import (
    CrossAttentionTextConfig,
    CrossAttentionVisionConfig,
    MLlamaModel,
    MLlamaModelConfig,
)
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_distributed_model_weights
from nemo.lightning import io, teardown
from nemo.lightning.io.state import _ModelState
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging

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
        source = MllamaForConditionalGeneration.from_pretrained(str(self), torch_dtype="auto")

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
                "model.language_model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
                "model.language_model.xattn_layers.*.cross_attn.o_proj.weight": "language_model.decoder.xattn_layers.*.cross_attention.linear_proj.weight",
                "model.language_model.xattn_layers.*.cross_attn.q_proj.weight": "language_model.decoder.xattn_layers.*.cross_attention.linear_q.weight",
                "model.language_model.norm.weight": "language_model.decoder.final_layernorm.weight",
                "lm_head.weight": "language_model.output_layer.weight",
                "model.language_model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "model.language_model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
                "model.language_model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "model.language_model.xattn_layers.*.cross_attn.k_norm.weight": "language_model.decoder.xattn_layers.*.cross_attention.k_layernorm.weight",
                "model.language_model.xattn_layers.*.input_layernorm.weight": "language_model.decoder.xattn_layers.*.cross_attention.linear_q.layer_norm_weight",
                "model.language_model.xattn_layers.*.cross_attn.q_norm.weight": "language_model.decoder.xattn_layers.*.cross_attention.q_layernorm.weight",
                "model.language_model.xattn_layers.*.post_attention_layernorm.weight": "language_model.decoder.xattn_layers.*.mlp.linear_fc1.layer_norm_weight",
                "model.language_model.xattn_layers.*.mlp.down_proj.weight": "language_model.decoder.xattn_layers.*.mlp.linear_fc2.weight",
            }
        )

        transforms.extend(
            [
                io.state_transform(
                    source_key="model.language_model.xattn_layers.*.cross_attn_attn_gate",
                    target_key="language_model.decoder.xattn_layers.*.gate_attn",
                    fn=_import_gate,
                ),
                io.state_transform(
                    source_key="model.language_model.xattn_layers.*.cross_attn_mlp_gate",
                    target_key="language_model.decoder.xattn_layers.*.gate_ffn",
                    fn=_import_gate,
                ),
                io.state_transform(
                    source_key=(
                        "model.language_model.layers.*.self_attn.q_proj.weight",
                        "model.language_model.layers.*.self_attn.k_proj.weight",
                        "model.language_model.layers.*.self_attn.v_proj.weight",
                    ),
                    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    fn=_import_text_qkv,
                ),
                io.state_transform(
                    source_key=(
                        "model.language_model.layers.*.mlp.gate_proj.weight",
                        "model.language_model.layers.*.mlp.up_proj.weight",
                    ),
                    target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    fn=_import_simple_concat,
                ),
                io.state_transform(
                    source_key=(
                        "model.language_model.xattn_layers.*.cross_attn.k_proj.weight",
                        "model.language_model.xattn_layers.*.cross_attn.v_proj.weight",
                    ),
                    target_key="language_model.decoder.xattn_layers.*.cross_attention.linear_kv.weight",
                    fn=_import_text_kv,
                ),
                io.state_transform(
                    source_key=(
                        "model.language_model.xattn_layers.*.mlp.gate_proj.weight",
                        "model.language_model.xattn_layers.*.mlp.up_proj.weight",
                    ),
                    target_key="language_model.decoder.xattn_layers.*.mlp.linear_fc1.weight",
                    fn=_import_simple_concat,
                ),
                io.state_transform(
                    source_key="model.language_model.embed_tokens.weight",
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
                "model.vision_model.global_transformer.layers.*.self_attn.o_proj.weight": f"{v}.global_transformer.layers.*.self_attention.linear_proj.weight",
                "model.vision_model.global_transformer.layers.*.gate_attn": f"{v}.global_transformer.layers.*.gate_attn",
                "model.vision_model.global_transformer.layers.*.gate_ffn": f"{v}.global_transformer.layers.*.gate_ffn",
                "model.vision_model.global_transformer.layers.*.input_layernorm.bias": f"{v}.global_transformer.layers.*.input_layernorm.bias",
                "model.vision_model.global_transformer.layers.*.input_layernorm.weight": f"{v}.global_transformer.layers.*.input_layernorm.weight",
                "model.vision_model.global_transformer.layers.*.post_attention_layernorm.bias": f"{v}.global_transformer.layers.*.pre_mlp_layernorm.bias",
                "model.vision_model.global_transformer.layers.*.post_attention_layernorm.weight": f"{v}.global_transformer.layers.*.pre_mlp_layernorm.weight",
                "model.vision_model.global_transformer.layers.*.mlp.fc1.bias": f"{v}.global_transformer.layers.*.mlp.linear_fc1.bias",
                "model.vision_model.global_transformer.layers.*.mlp.fc1.weight": f"{v}.global_transformer.layers.*.mlp.linear_fc1.weight",
                "model.vision_model.global_transformer.layers.*.mlp.fc2.bias": f"{v}.global_transformer.layers.*.mlp.linear_fc2.bias",
                "model.vision_model.global_transformer.layers.*.mlp.fc2.weight": f"{v}.global_transformer.layers.*.mlp.linear_fc2.weight",
                "model.vision_model.transformer.layers.*.self_attn.o_proj.weight": f"{v}.transformer.layers.*.self_attention.linear_proj.weight",
                "model.vision_model.transformer.layers.*.input_layernorm.bias": f"{v}.transformer.layers.*.input_layernorm.bias",
                "model.vision_model.transformer.layers.*.input_layernorm.weight": f"{v}.transformer.layers.*.input_layernorm.weight",
                "model.vision_model.transformer.layers.*.post_attention_layernorm.bias": f"{v}.transformer.layers.*.pre_mlp_layernorm.bias",
                "model.vision_model.transformer.layers.*.post_attention_layernorm.weight": f"{v}.transformer.layers.*.pre_mlp_layernorm.weight",
                "model.vision_model.transformer.layers.*.mlp.fc1.bias": f"{v}.transformer.layers.*.mlp.linear_fc1.bias",
                "model.vision_model.transformer.layers.*.mlp.fc1.weight": f"{v}.transformer.layers.*.mlp.linear_fc1.weight",
                "model.vision_model.transformer.layers.*.mlp.fc2.bias": f"{v}.transformer.layers.*.mlp.linear_fc2.bias",
                "model.vision_model.transformer.layers.*.mlp.fc2.weight": f"{v}.transformer.layers.*.mlp.linear_fc2.weight",
                "model.vision_model.class_embedding": f"{v}.class_embedding",
                "model.vision_model.gated_positional_embedding.embedding": f"{v}.positional_embedding",
                "model.vision_model.gated_positional_embedding.tile_embedding.weight": f"{v}.gated_tile_positional_embedding.weight",
                "model.vision_model.gated_positional_embedding.gate": f"{v}.gated_positional_embedding_gate",
                "model.vision_model.layernorm_post.bias": f"{v}.ln_post.bias",
                "model.vision_model.layernorm_post.weight": f"{v}.ln_post.weight",
                "model.vision_model.layernorm_pre.bias": f"{v}.ln_pre.bias",
                "model.vision_model.layernorm_pre.weight": f"{v}.ln_pre.weight",
                "model.vision_model.post_tile_positional_embedding.embedding.weight": f"{v}.post_tile_pos_embed.embedding.weight",
                "model.vision_model.post_tile_positional_embedding.gate": f"{v}.post_tile_pos_embed.gate",
                "model.vision_model.pre_tile_positional_embedding.embedding.weight": f"{v}.pre_tile_pos_embed.embedding.weight",
                "model.vision_model.pre_tile_positional_embedding.gate": f"{v}.pre_tile_pos_embed.gate",
                "model.multi_modal_projector.bias": "vision_model.vision_projection.encoder.bias",
                "model.multi_modal_projector.weight": "vision_model.vision_projection.encoder.weight",
            }
        )
        transforms.extend(
            [
                io.state_transform(
                    source_key=(
                        "model.vision_model.global_transformer.layers.*.self_attn.q_proj.weight",
                        "model.vision_model.global_transformer.layers.*.self_attn.k_proj.weight",
                        "model.vision_model.global_transformer.layers.*.self_attn.v_proj.weight",
                    ),
                    target_key=(f"{v}.global_transformer.layers.*.self_attention.linear_qkv.weight"),
                    fn=_import_vision_qkv,
                ),
                io.state_transform(
                    source_key=(
                        "model.vision_model.transformer.layers.*.self_attn.q_proj.weight",
                        "model.vision_model.transformer.layers.*.self_attn.k_proj.weight",
                        "model.vision_model.transformer.layers.*.self_attn.v_proj.weight",
                    ),
                    target_key=(f"{v}.transformer.layers.*.self_attention.linear_qkv.weight"),
                    fn=_import_vision_qkv,
                ),
                io.state_transform(
                    source_key="model.vision_model.patch_embedding.weight",
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
                source.text_config.num_hidden_layers,
                source.text_config.cross_attention_layers,
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


@io.model_exporter(MLlamaModel, "hf")
class HFMLlamaExporter(io.ModelConnector[MLlamaModel, "MllamaForConditionalGeneration"]):
    """
    Exporter class for converting NeMo MLlama model to HuggingFace format.

    Inherits:
        io.ModelConnector: Connector interface to handle setup, save, and load using the Lightning framework.

    Methods:
        init: Initializes a new HuggingFace MLlama model instance.
        apply: Converts the NeMo model to HuggingFace format and saves it.
        convert_state: Maps and transforms the state dictionary from NeMo to HuggingFace format.
        config: Generates and returns the HuggingFace MLlama config for the model.
    """

    def init(self, dtype=torch.bfloat16) -> "MllamaForConditionalGeneration":
        """
        Initializes a HuggingFace MllamaForConditionalGeneration model.

        Args:
            dtype: The data type to use for the model (default: torch.bfloat16)

        Returns:
            MllamaForConditionalGeneration: A HuggingFace MLlama model initialized with the configuration.
        """
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return MllamaForConditionalGeneration._from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        """
        Converts the NeMo MLlama model to HuggingFace format and saves it to the specified path.

        Args:
            output_path (Path): The path where the converted HuggingFace model will be saved.

        Returns:
            Path: The output path where the HuggingFace model was saved.
        """
        logging.info("Loading MLlama NeMo checkpoint. This may take a while...")
        source, source_config = self.ckpt_load(self)
        logging.info("MLlama NeMo checkpoint loaded.")
        logging.info("Initializing the HF model..")
        target = self.init()
        logging.info("Start Converting the model..")
        target = self.convert_state(source, target, source_config)
        target = target.cpu()
        target.save_pretrained(output_path)

        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        print(f"Converted MLlama model saved to {output_path}")

        return output_path

    def convert_state(self, source, target, source_config):
        # pylint: disable=C0115,C0116,line-too-long
        """
        Maps and transforms the state dictionary from NeMo to HuggingFace format.

        Args:
            source: The source NeMo model.
            target: The target HuggingFace model.

        Returns:
            The target HuggingFace model with the converted state.
        """
        source = self._modify_mllama_source_state(source, source_config)
        mapping = {}
        transforms = []
        # Define the state mapping from NeMo to HuggingFace
        mapping.update(
            {
                "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
                "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
                "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
                "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
                "language_model.decoder.xattn_layers.*.cross_attention.q_layernorm.weight": "model.language_model.layers.*.cross_attn.q_norm.weight",
                "language_model.decoder.xattn_layers.*.cross_attention.linear_q.weight": "model.language_model.layers.*.cross_attn.q_proj.weight",
                "language_model.decoder.xattn_layers.*.cross_attention.k_layernorm.weight": "model.language_model.layers.*.cross_attn.k_norm.weight",
                "language_model.decoder.xattn_layers.*.cross_attention.linear_proj.weight": "model.language_model.layers.*.cross_attn.o_proj.weight",
                "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
                "language_model.output_layer.weight": "lm_head.weight",
            }
        )
        transforms.extend(
            [
                io.state_transform(
                    source_key="language_model.decoder.xattn_layers.*.gate_attn",
                    target_key="model.language_model.layers.*.cross_attn_attn_gate",
                    fn=_export_gate,
                ),
                io.state_transform(
                    source_key="language_model.decoder.xattn_layers.*.gate_ffn",
                    target_key="model.language_model.layers.*.cross_attn_mlp_gate",
                    fn=_export_gate,
                ),
                io.state_transform(
                    source_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    target_key=(
                        "model.language_model.layers.*.self_attn.q_proj.weight",
                        "model.language_model.layers.*.self_attn.k_proj.weight",
                        "model.language_model.layers.*.self_attn.v_proj.weight",
                    ),
                    fn=_export_text_qkv,
                ),
                io.state_transform(
                    source_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    target_key=(
                        "model.language_model.layers.*.mlp.gate_proj.weight",
                        "model.language_model.layers.*.mlp.up_proj.weight",
                    ),
                    fn=_export_simple_split,
                ),
                io.state_transform(
                    source_key="language_model.decoder.xattn_layers.*.cross_attention.linear_kv.weight",
                    target_key=(
                        "model.language_model.layers.*.cross_attn.k_proj.weight",
                        "model.language_model.layers.*.cross_attn.v_proj.weight",
                    ),
                    fn=_export_text_kv,
                ),
                io.state_transform(
                    source_key=(
                        "language_model.embedding.word_embeddings.weight",
                        "language_model.learnable_embedding.weight",
                    ),
                    target_key="model.language_model.embed_tokens.weight",
                    fn=_export_embedding_hf,
                ),
            ]
        )
        v = "vision_model.vision_encoder"
        mapping.update(
            {
                f"{v}.global_transformer.layers.*.self_attention.linear_proj.weight": "model.vision_model.global_transformer.layers.*.self_attn.o_proj.weight",
                f"{v}.global_transformer.layers.*.gate_attn": "model.vision_model.global_transformer.layers.*.gate_attn",
                f"{v}.global_transformer.layers.*.gate_ffn": "model.vision_model.global_transformer.layers.*.gate_ffn",
                f"{v}.global_transformer.layers.*.input_layernorm.bias": "model.vision_model.global_transformer.layers.*.input_layernorm.bias",
                f"{v}.global_transformer.layers.*.input_layernorm.weight": "model.vision_model.global_transformer.layers.*.input_layernorm.weight",
                f"{v}.global_transformer.layers.*.pre_mlp_layernorm.bias": "model.vision_model.global_transformer.layers.*.post_attention_layernorm.bias",
                f"{v}.global_transformer.layers.*.pre_mlp_layernorm.weight": "model.vision_model.global_transformer.layers.*.post_attention_layernorm.weight",
                f"{v}.global_transformer.layers.*.mlp.linear_fc1.bias": "model.vision_model.global_transformer.layers.*.mlp.fc1.bias",
                f"{v}.global_transformer.layers.*.mlp.linear_fc1.weight": "model.vision_model.global_transformer.layers.*.mlp.fc1.weight",
                f"{v}.global_transformer.layers.*.mlp.linear_fc2.bias": "model.vision_model.global_transformer.layers.*.mlp.fc2.bias",
                f"{v}.global_transformer.layers.*.mlp.linear_fc2.weight": "model.vision_model.global_transformer.layers.*.mlp.fc2.weight",
                f"{v}.transformer.layers.*.self_attention.linear_proj.weight": "model.vision_model.transformer.layers.*.self_attn.o_proj.weight",
                f"{v}.transformer.layers.*.input_layernorm.bias": "model.vision_model.transformer.layers.*.input_layernorm.bias",
                f"{v}.transformer.layers.*.input_layernorm.weight": "model.vision_model.transformer.layers.*.input_layernorm.weight",
                f"{v}.transformer.layers.*.pre_mlp_layernorm.bias": "model.vision_model.transformer.layers.*.post_attention_layernorm.bias",
                f"{v}.transformer.layers.*.pre_mlp_layernorm.weight": "model.vision_model.transformer.layers.*.post_attention_layernorm.weight",
                f"{v}.transformer.layers.*.mlp.linear_fc1.bias": "model.vision_model.transformer.layers.*.mlp.fc1.bias",
                f"{v}.transformer.layers.*.mlp.linear_fc1.weight": "model.vision_model.transformer.layers.*.mlp.fc1.weight",
                f"{v}.transformer.layers.*.mlp.linear_fc2.bias": "model.vision_model.transformer.layers.*.mlp.fc2.bias",
                f"{v}.transformer.layers.*.mlp.linear_fc2.weight": "model.vision_model.transformer.layers.*.mlp.fc2.weight",
                f"{v}.class_embedding": "model.vision_model.class_embedding",
                f"{v}.positional_embedding": "model.vision_model.gated_positional_embedding.embedding",
                f"{v}.gated_tile_positional_embedding.weight": "model.vision_model.gated_positional_embedding.tile_embedding.weight",
                f"{v}.gated_positional_embedding_gate": "model.vision_model.gated_positional_embedding.gate",
                f"{v}.ln_post.bias": "model.vision_model.layernorm_post.bias",
                f"{v}.ln_post.weight": "model.vision_model.layernorm_post.weight",
                f"{v}.ln_pre.bias": "model.vision_model.layernorm_pre.bias",
                f"{v}.ln_pre.weight": "model.vision_model.layernorm_pre.weight",
                f"{v}.post_tile_pos_embed.embedding.weight": "model.vision_model.post_tile_positional_embedding.embedding.weight",
                f"{v}.post_tile_pos_embed.gate": "model.vision_model.post_tile_positional_embedding.gate",
                f"{v}.pre_tile_pos_embed.embedding.weight": "model.vision_model.pre_tile_positional_embedding.embedding.weight",
                f"{v}.pre_tile_pos_embed.gate": "model.vision_model.pre_tile_positional_embedding.gate",
                "vision_model.vision_projection.encoder.bias": "model.multi_modal_projector.bias",
                "vision_model.vision_projection.encoder.weight": "model.multi_modal_projector.weight",
            }
        )
        transforms.extend(
            [
                io.state_transform(
                    source_key=(f"{v}.global_transformer.layers.*.self_attention.linear_qkv.weight"),
                    target_key=(
                        "model.vision_model.global_transformer.layers.*.self_attn.q_proj.weight",
                        "model.vision_model.global_transformer.layers.*.self_attn.k_proj.weight",
                        "model.vision_model.global_transformer.layers.*.self_attn.v_proj.weight",
                    ),
                    fn=_export_vision_qkv,
                ),
                io.state_transform(
                    source_key=(f"{v}.transformer.layers.*.self_attention.linear_qkv.weight"),
                    target_key=(
                        "model.vision_model.transformer.layers.*.self_attn.q_proj.weight",
                        "model.vision_model.transformer.layers.*.self_attn.k_proj.weight",
                        "model.vision_model.transformer.layers.*.self_attn.v_proj.weight",
                    ),
                    fn=_export_vision_qkv,
                ),
                io.state_transform(
                    source_key=f"{v}.conv1._linear.weight",
                    target_key="model.vision_model.patch_embedding.weight",
                    fn=_export_patch_embedding_hf,
                ),
            ]
        )
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "TokenizerSpec":
        """
        Gets the tokenizer from the loaded model context.

        Returns:
            The tokenizer specification.
        """
        return io.load_context(str(self), subpath="model").tokenizer

    def ckpt_load(self, path: Path) -> Tuple[Dict, Dict]:
        """
        This function loads the state dict directly from a distributed checkpoint, and modify the state dict
        so that it is consistent with the key names you would get from loading the checkpoint into a model.
        This is a more memory-efficient method to obtain a state dict without initializing the nemo model.

        Args:
            path (Path): The path from which the model will be loaded.

        Returns
        -------
            Tuple[Dict, Dict]: The loaded state dict and the yaml config dict.
        """
        config = io.load_context(str(self), subpath="model.config")
        dist_ckpt_folder = path / "weights"
        state_dict = {}

        langauge_layers = config.language_model_config.num_layers
        vision_layers = config.vision_model_config.num_layers
        distributed_model_weights = load_distributed_model_weights(dist_ckpt_folder, True).items()
        for k, v in distributed_model_weights:
            if "_extra_state" in k:
                continue
            new_k = k.replace("module.", "")
            if "layers" in new_k and (v.size(0) == langauge_layers or v.size(0) == vision_layers):
                # Only split layers
                for i in range(v.size(0)):
                    state_dict[new_k.replace("layers", f"layers.{str(i)}")] = v[i]
            elif "global_transformer.layers" in new_k:
                for i in range(v.size(0)):
                    state_dict[new_k.replace("layers", f"layers.{str(i)}")] = v[i]
            state_dict[new_k] = v
        return state_dict, config

    def _modify_mllama_source_state(self, state_dict, source_config):
        """
        - Modify state dict to integrate cross-attention layers into self-attention layer.
        e.g. 11B: 32 self-attn + 8 cross-attn -> 40 layers, 90B: 80 self-attn + 20 cross-attn -> 100 layers
        - Change the layer index to match the cross_attention_layers in the model config.
        e.g. 11B: [3, 7, 11, 15, 19, 23, 27, 31] -> [3, 8, 13, 18, 23, 28, 33, 38]

        Args:
            state_dict: Source model state dict
            source_config: Model config dict

        Returns:
            _ModelState: Modified state
        """

        def convert_layer_num(match):
            layer_num = int(match.group(1))
            x_num = (layer_num - 3) // (cross_attention_frequency)
            if (layer_num - 3) % (cross_attention_frequency) == 0:
                new_layer_num = x_num + layer_num
                return f".{new_layer_num}."
            raise ValueError(
                f"Unexpected layer_num: {layer_num} (does not align with cross_attention_frequency={cross_attention_frequency})"
            )

        text_config = source_config.language_model_config
        cross_attention_frequency = text_config.num_layers // text_config.num_cross_attention_layers
        total_num_layer = text_config.num_layers + text_config.num_cross_attention_layers
        prefix = "language_model.decoder"

        new_state_dict = {}
        # Integrating layer indexes of self-attention and cross-attention
        for i in range(total_num_layer):
            cross_num = (i - 3) // (cross_attention_frequency + 1)
            if (i - 3) % (cross_attention_frequency + 1) == 0:
                xattn_index = cross_num * cross_attention_frequency + 3
                new_state_dict[f"{prefix}.layers.{i}.mlp.linear_fc1.layer_norm_weight"] = state_dict.pop(
                    f"{prefix}.xattn_layers.{xattn_index}.mlp.linear_fc1.layer_norm_weight"
                )
                new_state_dict[f"{prefix}.layers.{i}.mlp.linear_fc2.weight"] = state_dict.pop(
                    f"{prefix}.xattn_layers.{xattn_index}.mlp.linear_fc2.weight"
                )
                new_state_dict[f"{prefix}.layers.{i}.self_attention.linear_qkv.layer_norm_weight"] = state_dict.pop(
                    f"{prefix}.xattn_layers.{xattn_index}.cross_attention.linear_q.layer_norm_weight"
                )
                new_state_dict[f"{prefix}.layers.{i}.mlp.linear_fc1.weight"] = state_dict.pop(
                    f"{prefix}.xattn_layers.{xattn_index}.mlp.linear_fc1.weight"
                )
            else:
                attn_index = i - cross_num - 1
                new_state_dict[f"{prefix}.layers.{i}.mlp.linear_fc1.layer_norm_weight"] = state_dict.pop(
                    f"{prefix}.layers.{attn_index}.mlp.linear_fc1.layer_norm_weight"
                )
                new_state_dict[f"{prefix}.layers.{i}.mlp.linear_fc2.weight"] = state_dict.pop(
                    f"{prefix}.layers.{attn_index}.mlp.linear_fc2.weight"
                )
                new_state_dict[f"{prefix}.layers.{i}.self_attention.linear_qkv.layer_norm_weight"] = state_dict.pop(
                    f"{prefix}.layers.{attn_index}.self_attention.linear_qkv.layer_norm_weight"
                )
                new_state_dict[f"{prefix}.layers.{i}.mlp.linear_fc1.weight"] = state_dict.pop(
                    f"{prefix}.layers.{attn_index}.mlp.linear_fc1.weight"
                )

        for k, v in new_state_dict.items():
            state_dict[k] = v

        new_state_dict = {}
        # Align the cross-attention layer index with HF
        for k, v in state_dict.items():
            if "xattn_layers" in k:
                new_state_dict[re.sub(r"\.(\d+)\.", convert_layer_num, k)] = v
            else:
                new_state_dict[k] = v

        source = _ModelState(new_state_dict)
        return source

    @property
    def config(self) -> "HFMllamaConfig":
        """
        Generates the configuration for the HuggingFace MLlama model based on the NeMo model.

        Returns:
            HFMllamaConfig: A configuration object for the HuggingFace MLlama model.
        """
        source = io.load_context(str(self), subpath="model.config")
        vision_model_config = source.vision_model_config
        language_config = source.language_model_config

        vision_config = MllamaVisionConfig(
            num_hidden_layers=vision_model_config.num_layers,
            hidden_size=vision_model_config.hidden_size,
            attention_heads=vision_model_config.num_attention_heads,
            image_size=vision_model_config.vision_chunk_size,
            max_num_tiles=vision_model_config.vision_max_num_chunks,
            torch_dtype="bfloat16",
        )
        cross_attention_layers = [
            x + i
            for i, x in enumerate(language_config._init_fusion_schedule(language_config.num_cross_attention_layers))
        ]
        # Create text config for HuggingFace model
        text_config = MllamaTextConfig(
            rope_theta=language_config.rotary_base,
            num_hidden_layers=language_config.num_layers + language_config.num_cross_attention_layers,
            tie_word_embeddings=language_config.share_embeddings_and_output_weights,
            cross_attention_layers=cross_attention_layers,
            hidden_size=language_config.hidden_size,
            intermediate_size=language_config.ffn_hidden_size,
            num_attention_heads=language_config.num_attention_heads,
            num_key_value_heads=language_config.num_query_groups,
            vocab_size=language_config.vocab_size,
            rope_scaling={
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            eos_token_id=[128001, 128008, 128009],
            torch_dtype="bfloat16",
        )
        # Create the MllamaConfig for HuggingFace
        return HFMllamaConfig(vision_config=vision_config, text_config=text_config, torch_dtype="bfloat16")


def _rename_xattn_layer_nums_hf(source: Dict):
    def convert_layer_num(match):
        layer_num = int(match.group(1))
        cross_num = (layer_num - 3) // (cross_attention_frequency + 1)
        if (layer_num - 3) % (cross_attention_frequency + 1) == 0:
            new_layer_num = cross_num * cross_attention_frequency + 3
            return f"xattn_layers.{new_layer_num}."

        new_layer_num = layer_num - cross_num - 1
        return f"layers.{new_layer_num}."

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


def _import_simple_concat(a, b):
    # for both (w1, w3) -> fc1, and (wk, wv) -> wkv
    return torch.cat((a, b), dim=0)


def _merge_kv(
    k: Tensor,
    v: Tensor,
    head_num: int,
    num_query_groups: int,
    head_size: int,
    hidden_size: int,
):
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
    q: Tensor,
    k: Tensor,
    v: Tensor,
    head_num: int,
    num_query_groups: int,
    head_size: int,
    hidden_size: int,
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


def _split_kv(
    kv: Tensor,
    head_num: int,
    num_query_groups: int,
    head_size: int,
    hidden_size: int,
):
    kv_total_dim = 2 * num_query_groups

    linear_kv = kv.reshape([kv_total_dim, head_size, hidden_size])

    k_slice = torch.arange(0, kv_total_dim, 2)
    v_slice = torch.arange(1, kv_total_dim, 2)

    k_proj = linear_kv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_kv[v_slice].reshape(-1, hidden_size).cpu()

    return k_proj, v_proj


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


def _export_gate(gate):
    return gate[0:1]


def _export_patch_embedding_hf(a):
    return a.reshape(a.shape[0], 3, 14, 14)


def _export_vision_qkv(ctx: io.TransformCTX, qkv):
    vision_config = ctx.target.config.vision_config

    head_num = vision_config.attention_heads
    num_query_groups = vision_config.attention_heads
    hidden_size = vision_config.hidden_size
    head_size = hidden_size // head_num
    return _split_qkv(qkv, head_num, num_query_groups, head_size, hidden_size)


def _export_text_kv(ctx: io.TransformCTX, kv):
    text_config = ctx.target.config.text_config

    head_num = text_config.num_attention_heads
    num_query_groups = text_config.num_key_value_heads
    hidden_size = text_config.hidden_size
    head_size = hidden_size // head_num
    return _split_kv(kv, head_num, num_query_groups, head_size, hidden_size)


def _export_text_qkv(ctx: io.TransformCTX, qkv):
    text_config = ctx.target.config.text_config

    head_num = text_config.num_attention_heads
    num_query_groups = text_config.num_key_value_heads
    hidden_size = text_config.hidden_size
    head_size = hidden_size // head_num
    return _split_qkv(qkv, head_num, num_query_groups, head_size, hidden_size)


def _export_simple_split(linear_fc1):
    """Splits NeMo's fused MLP linear_fc1 weight into gate_proj and up_proj for HuggingFace format."""
    gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)
    return gate_proj, up_proj


def _export_embedding_hf(word_embeddings, learnable_embedding):
    """Transforms the word embeddings from NeMo to HuggingFace format."""
    return torch.cat((word_embeddings, learnable_embedding), dim=0)
