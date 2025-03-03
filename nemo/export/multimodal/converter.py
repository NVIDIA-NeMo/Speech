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


import torch
from transformers import AutoProcessor, MllamaConfig
from transformers.models.mllama.configuration_mllama import MllamaTextConfig, MllamaVisionConfig

from nemo import lightning as nl
from nemo.collections import vlm


def split_qkv_weight(qkv_weight, model_config):
    """Split attention qkv from nemo to hf format"""
    hidden_size = model_config.hidden_size
    head_num = model_config.num_attention_heads
    num_query_groups = model_config.num_query_groups or head_num
    head_size = model_config.kv_channels or (hidden_size // head_num)
    heads_per_group = head_num // num_query_groups
    qkv_weight = qkv_weight.reshape(-1, head_size, hidden_size)
    q_weight = torch.empty((head_num, head_size, hidden_size), device=qkv_weight.device)
    k_weight = torch.empty((num_query_groups, head_size, hidden_size), device=qkv_weight.device)
    v_weight = torch.empty((num_query_groups, head_size, hidden_size), device=qkv_weight.device)

    qkv_index = 0
    for i in range(num_query_groups):
        q_weight[i * heads_per_group : (i + 1) * heads_per_group, :, :] = qkv_weight[
            qkv_index : qkv_index + heads_per_group, :, :
        ]
        qkv_index += heads_per_group
        k_weight[i, :, :] = qkv_weight[qkv_index, :, :]
        qkv_index += 1
        v_weight[i, :, :] = qkv_weight[qkv_index, :, :]
        qkv_index += 1

    return [('q_proj', q_weight), ('k_proj', k_weight), ('v_proj', v_weight)]


def split_kv_weight(kv_weight, model_config):
    """Split cross attention qkv from nemo to hf format"""
    hidden_size = model_config.hidden_size
    head_num = model_config.num_attention_heads
    num_query_groups = model_config.num_query_groups or head_num
    head_size = model_config.kv_channels or (hidden_size // head_num)
    kv_weight = kv_weight.reshape(-1, head_size, hidden_size)
    k_weight = torch.empty((num_query_groups, head_size, hidden_size), device=kv_weight.device)
    v_weight = torch.empty((num_query_groups, head_size, hidden_size), device=kv_weight.device)

    kv_index = 0
    for i in range(num_query_groups):
        k_weight[i, :, :] = kv_weight[kv_index, :, :]
        kv_index += 1
        v_weight[i, :, :] = kv_weight[kv_index, :, :]
        kv_index += 1

    return [('k_proj', k_weight), ('v_proj', v_weight)]


def split_gate_weight(gate_weight):
    """Split linear fc to gate"""
    gate_weight = torch.chunk(gate_weight, 2, axis=0)

    return [('gate_proj', gate_weight[0]), ('up_proj', gate_weight[1])]


def convert_mllama_config(source_vision, source_text):
    """Convert nemo mllama config to hf config"""
    vision_config = MllamaVisionConfig(
        num_hidden_layers=source_vision.num_layers,
        hidden_size=source_vision.hidden_size,
        attention_heads=source_vision.num_attention_heads,
        image_size=source_vision.vision_chunk_size,
        max_num_tiles=source_vision.vision_max_num_chunks,
        torch_dtype="bfloat16",
    )

    cross_attention_layers = [
        x + i for i, x in enumerate(source_text._init_fusion_schedule(source_text.num_cross_attention_layers))
    ]
    text_config = MllamaTextConfig(
        rope_theta=source_text.rotary_base,
        num_hidden_layers=source_text.num_layers + source_text.num_cross_attention_layers,
        cross_attention_layers=cross_attention_layers,
        hidden_size=source_text.hidden_size,
        intermediate_size=source_text.ffn_hidden_size,
        num_attention_heads=source_text.num_attention_heads,
        num_key_value_heads=source_text.num_query_groups,
        vocab_size=source_text.vocab_size,
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

    return MllamaConfig(vision_config, text_config, torch_dtype="bfloat16")


def convert_mllama_nemo_to_hf(checkpoint_path, processor_name):
    """Convert nemo mllama to hf state dict and config"""
    processor = AutoProcessor.from_pretrained(processor_name)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        ckpt_load_optimizer=False,
        ckpt_save_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=1,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    fabric = trainer.to_fabric()

    tokenizer = processor.tokenizer
    model = vlm.MLlamaModel(vlm.MLlamaConfig11BInstruct(), tokenizer=tokenizer)
    config = model.config
    vision_model_config = config.vision_model_config
    language_model_config = config.language_model_config
    model = fabric.load_model(checkpoint_path, model)
    model = model.module.module.module.module

    state_dict = model.state_dict()
    del model

    v = "vision_model.vision_encoder"
    key_map = [
        ("vision_model.class_embedding", f"{v}.class_embedding"),
        ("vision_model.gated_positional_embedding.embedding", f"{v}.positional_embedding"),
        (
            "vision_model.gated_positional_embedding.tile_embedding.weight",
            f"{v}.gated_tile_positional_embedding.weight",
        ),
        ("vision_model.gated_positional_embedding.gate", f"{v}.gated_positional_embedding_gate"),
        ("vision_model.layernorm_post.bias", f"{v}.ln_post.bias"),
        ("vision_model.layernorm_post.weight", f"{v}.ln_post.weight"),
        ("vision_model.layernorm_pre.bias", f"{v}.ln_pre.bias"),
        ("vision_model.layernorm_pre.weight", f"{v}.ln_pre.weight"),
        ("vision_model.post_tile_positional_embedding.embedding.weight", f"{v}.post_tile_pos_embed.embedding.weight"),
        ("vision_model.post_tile_positional_embedding.gate", f"{v}.post_tile_pos_embed.gate"),
        ("vision_model.pre_tile_positional_embedding.embedding.weight", f"{v}.pre_tile_pos_embed.embedding.weight"),
        ("vision_model.pre_tile_positional_embedding.gate", f"{v}.pre_tile_pos_embed.gate"),
        ("multi_modal_projector.bias", "vision_model.vision_projection.encoder.bias"),
        ("multi_modal_projector.weight", "vision_model.vision_projection.encoder.weight"),
        ("language_model.model.norm.weight", "language_model.decoder.final_layernorm.weight"),
        ("language_model.lm_head.weight", "language_model.output_layer.weight"),
    ]

    for i in range(vision_model_config.num_layers):
        key_map.extend(
            [
                (
                    f"vision_model.transformer.layers.{i}.self_attn.o_proj.weight",
                    f"{v}.transformer.layers.{i}.self_attention.linear_proj.weight",
                ),
                (
                    f"vision_model.transformer.layers.{i}.input_layernorm.bias",
                    f"{v}.transformer.layers.{i}.input_layernorm.bias",
                ),
                (
                    f"vision_model.transformer.layers.{i}.input_layernorm.weight",
                    f"{v}.transformer.layers.{i}.input_layernorm.weight",
                ),
                (
                    f"vision_model.transformer.layers.{i}.post_attention_layernorm.bias",
                    f"{v}.transformer.layers.{i}.pre_mlp_layernorm.bias",
                ),
                (
                    f"vision_model.transformer.layers.{i}.post_attention_layernorm.weight",
                    f"{v}.transformer.layers.{i}.pre_mlp_layernorm.weight",
                ),
                (
                    f"vision_model.transformer.layers.{i}.mlp.fc1.bias",
                    f"{v}.transformer.layers.{i}.mlp.linear_fc1.bias",
                ),
                (
                    f"vision_model.transformer.layers.{i}.mlp.fc1.weight",
                    f"{v}.transformer.layers.{i}.mlp.linear_fc1.weight",
                ),
                (
                    f"vision_model.transformer.layers.{i}.mlp.fc2.bias",
                    f"{v}.transformer.layers.{i}.mlp.linear_fc2.bias",
                ),
                (
                    f"vision_model.transformer.layers.{i}.mlp.fc2.weight",
                    f"{v}.transformer.layers.{i}.mlp.linear_fc2.weight",
                ),
            ]
        )

    for i in range(vision_model_config.num_global_layers):
        key_map.extend(
            [
                (
                    f"vision_model.global_transformer.layers.{i}.self_attn.o_proj.weight",
                    f"{v}.global_transformer.layers.{i}.self_attention.linear_proj.weight",
                ),
                (
                    f"vision_model.global_transformer.layers.{i}.gate_attn",
                    f"{v}.global_transformer.layers.{i}.gate_attn",
                ),
                (
                    f"vision_model.global_transformer.layers.{i}.gate_ffn",
                    f"{v}.global_transformer.layers.{i}.gate_ffn",
                ),
                (
                    f"vision_model.global_transformer.layers.{i}.input_layernorm.bias",
                    f"{v}.global_transformer.layers.{i}.input_layernorm.bias",
                ),
                (
                    f"vision_model.global_transformer.layers.{i}.input_layernorm.weight",
                    f"{v}.global_transformer.layers.{i}.input_layernorm.weight",
                ),
                (
                    f"vision_model.global_transformer.layers.{i}.post_attention_layernorm.bias",
                    f"{v}.global_transformer.layers.{i}.pre_mlp_layernorm.bias",
                ),
                (
                    f"vision_model.global_transformer.layers.{i}.post_attention_layernorm.weight",
                    f"{v}.global_transformer.layers.{i}.pre_mlp_layernorm.weight",
                ),
                (
                    f"vision_model.global_transformer.layers.{i}.mlp.fc1.bias",
                    f"{v}.global_transformer.layers.{i}.mlp.linear_fc1.bias",
                ),
                (
                    f"vision_model.global_transformer.layers.{i}.mlp.fc1.weight",
                    f"{v}.global_transformer.layers.{i}.mlp.linear_fc1.weight",
                ),
                (
                    f"vision_model.global_transformer.layers.{i}.mlp.fc2.bias",
                    f"{v}.global_transformer.layers.{i}.mlp.linear_fc2.bias",
                ),
                (
                    f"vision_model.global_transformer.layers.{i}.mlp.fc2.weight",
                    f"{v}.global_transformer.layers.{i}.mlp.linear_fc2.weight",
                ),
            ]
        )

    cross_attention_frequency = language_model_config.num_layers // language_model_config.num_cross_attention_layers
    toal_num_layer = language_model_config.num_layers + language_model_config.num_cross_attention_layers
    prefix = "language_model.decoder"
    for i in range(toal_num_layer):
        cross_num = (i - 3) // (cross_attention_frequency + 1)
        if (i - 3) % (cross_attention_frequency + 1) == 0:
            xattn_index = cross_num * cross_attention_frequency + 3
            key_map.extend(
                [
                    (
                        f"language_model.model.layers.{i}.cross_attn.o_proj.weight",
                        f"{prefix}.xattn_layers.{xattn_index}.cross_attention.linear_proj.weight",
                    ),
                    (
                        f"language_model.model.layers.{i}.cross_attn.q_proj.weight",
                        f"{prefix}.xattn_layers.{xattn_index}.cross_attention.linear_q.weight",
                    ),
                    (
                        f"language_model.model.layers.{i}.cross_attn.k_norm.weight",
                        f"{prefix}.xattn_layers.{xattn_index}.cross_attention.k_layernorm.weight",
                    ),
                    (
                        f"language_model.model.layers.{i}.input_layernorm.weight",
                        f"{prefix}.xattn_layers.{xattn_index}.cross_attention.linear_q.layer_norm_weight",
                    ),
                    (
                        f"language_model.model.layers.{i}.cross_attn.q_norm.weight",
                        f"{prefix}.xattn_layers.{xattn_index}.cross_attention.q_layernorm.weight",
                    ),
                    (
                        f"language_model.model.layers.{i}.post_attention_layernorm.weight",
                        f"{prefix}.xattn_layers.{xattn_index}.mlp.linear_fc1.layer_norm_weight",
                    ),
                    (
                        f"language_model.model.layers.{i}.mlp.down_proj.weight",
                        f"{prefix}.xattn_layers.{xattn_index}.mlp.linear_fc2.weight",
                    ),
                    (
                        f"language_model.model.layers.{i}.cross_attn_attn_gate",
                        f"{prefix}.xattn_layers.{xattn_index}.gate_attn",
                    ),
                    (
                        f"language_model.model.layers.{i}.cross_attn_mlp_gate",
                        f"{prefix}.xattn_layers.{xattn_index}.gate_ffn",
                    ),
                ]
            )
        else:
            attn_index = i - cross_num - 1
            key_map.extend(
                [
                    (
                        f"language_model.model.layers.{i}.self_attn.o_proj.weight",
                        f"{prefix}.layers.{attn_index}.self_attention.linear_proj.weight",
                    ),
                    (
                        f"language_model.model.layers.{i}.post_attention_layernorm.weight",
                        f"{prefix}.layers.{attn_index}.mlp.linear_fc1.layer_norm_weight",
                    ),
                    (
                        f"language_model.model.layers.{i}.mlp.down_proj.weight",
                        f"{prefix}.layers.{attn_index}.mlp.linear_fc2.weight",
                    ),
                    (
                        f"language_model.model.layers.{i}.input_layernorm.weight",
                        f"{prefix}.layers.{attn_index}.self_attention.linear_qkv.layer_norm_weight",
                    ),
                ]
            )

    new_state_dict = {}
    for new_key, old_key in key_map:
        new_state_dict[new_key] = state_dict[old_key]

    def convert_vision_qkv_weight(state_dict, vision_model_config):
        hidden_size = vision_model_config.hidden_size

        new_state_dict = {}
        for i in range(vision_model_config.num_layers):
            qkv_weights = state_dict[
                f"vision_model.vision_encoder.transformer.layers.{i}.self_attention.linear_qkv.weight"
            ]

            for name, weight in split_qkv_weight(qkv_weights, vision_model_config):
                new_key = f'vision_model.transformer.layers.{i}.self_attn.{name}.weight'
                new_state_dict[new_key] = weight.reshape(-1, hidden_size)

        for i in range(vision_model_config.num_global_layers):
            qkv_weights = state_dict[
                f"vision_model.vision_encoder.global_transformer.layers.{i}.self_attention.linear_qkv.weight"
            ]

            for name, weight in split_qkv_weight(qkv_weights, vision_model_config):
                new_key = f'vision_model.global_transformer.layers.{i}.self_attn.{name}.weight'
                new_state_dict[new_key] = weight.reshape(-1, hidden_size)

        return new_state_dict

    def convert_patch_embeding(state_dict):
        conv1_weight = state_dict["vision_model.vision_encoder.conv1._linear.weight"]
        return {"vision_model.patch_embedding.weight": conv1_weight.reshape(conv1_weight.shape[0], 3, 14, 14)}

    def convert_language_qkv_weight(state_dict, language_model_config):
        hidden_size = language_model_config.hidden_size
        new_state_dict = {}
        for i in range(toal_num_layer):
            cross_num = (i - 3) // (cross_attention_frequency + 1)
            if (i - 3) % (cross_attention_frequency + 1) == 0:
                xattn_index = cross_num * cross_attention_frequency + 3
                kv_weights = state_dict[f"{prefix}.xattn_layers.{xattn_index}.cross_attention.linear_kv.weight"]
                for name, weight in split_kv_weight(kv_weights, language_model_config):
                    new_key = f"language_model.model.layers.{i}.cross_attn.{name}.weight"
                    new_state_dict[new_key] = weight.reshape(-1, hidden_size)
            else:
                attn_index = i - cross_num - 1
                qkv_weights = state_dict[f"{prefix}.layers.{attn_index}.self_attention.linear_qkv.weight"]
                for name, weight in split_qkv_weight(qkv_weights, language_model_config):
                    new_key = f"language_model.model.layers.{i}.self_attn.{name}.weight"
                    new_state_dict[new_key] = weight.reshape(-1, hidden_size)

        return new_state_dict

    def convert_gate(state_dict):
        new_state_dict = {}
        for i in range(toal_num_layer):
            cross_num = (i - 3) // (cross_attention_frequency + 1)
            if (i - 3) % (cross_attention_frequency + 1) == 0:
                xattn_index = cross_num * cross_attention_frequency + 3
                gate_weight = state_dict[f"{prefix}.xattn_layers.{xattn_index}.mlp.linear_fc1.weight"]
            else:
                attn_index = i - cross_num - 1
                gate_weight = state_dict[f"{prefix}.layers.{attn_index}.mlp.linear_fc1.weight"]

            for name, weight in split_gate_weight(gate_weight):
                new_key = f"language_model.model.layers.{i}.mlp.{name}.weight"
                new_state_dict[new_key] = weight

        return new_state_dict

    def convert_embedding(state_dict):
        word_embeddings = state_dict["language_model.embedding.word_embeddings.weight"]
        learnable_embedding = state_dict["language_model.learnable_embedding.weight"]

        return {"language_model.model.embed_tokens.weight": torch.cat((word_embeddings, learnable_embedding), dim=0)}

    new_state_dict.update(convert_vision_qkv_weight(state_dict, vision_model_config))
    new_state_dict.update(convert_patch_embeding(state_dict))
    new_state_dict.update(convert_language_qkv_weight(state_dict, language_model_config))
    new_state_dict.update(convert_gate(state_dict))
    new_state_dict.update(convert_embedding(state_dict))

    return new_state_dict, convert_mllama_config(vision_model_config, language_model_config)
