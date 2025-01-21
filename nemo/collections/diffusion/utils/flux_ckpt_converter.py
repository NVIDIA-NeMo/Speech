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
import os

import torch
from safetensors.torch import load_file as load_safetensors


def _import_qkv_bias(transformer_config, qb, kb, vb):

    head_num = transformer_config.num_attention_heads
    num_query_groups = transformer_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = transformer_config.hidden_size
    head_num = transformer_config.num_attention_heads
    head_size = hidden_size // head_num

    new_q_bias_tensor_shape = (head_num, head_size)
    new_kv_bias_tensor_shape = (num_query_groups, head_size)

    qb = qb.view(*new_q_bias_tensor_shape)
    kb = kb.view(*new_kv_bias_tensor_shape)
    vb = vb.view(*new_kv_bias_tensor_shape)

    qkv_bias_l = []
    for i in range(num_query_groups):
        qkv_bias_l.append(qb[i * heads_per_group : (i + 1) * heads_per_group, :])
        qkv_bias_l.append(kb[i : i + 1, :])
        qkv_bias_l.append(vb[i : i + 1, :])

    qkv_bias = torch.cat(qkv_bias_l)
    qkv_bias = qkv_bias.reshape([head_size * (head_num + 2 * num_query_groups)])

    return qkv_bias


def _import_qkv(transformer_config, q, k, v):

    head_num = transformer_config.num_attention_heads
    num_query_groups = transformer_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = transformer_config.hidden_size
    head_num = transformer_config.num_attention_heads
    head_size = hidden_size // head_num

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


flux_key_mapping = {
    'double_blocks': {
        'norm1.linear.weight': 'adaln.adaLN_modulation.1.weight',
        'norm1.linear.bias': 'adaln.adaLN_modulation.1.bias',
        'norm1_context.linear.weight': 'adaln_context.adaLN_modulation.1.weight',
        'norm1_context.linear.bias': 'adaln_context.adaLN_modulation.1.bias',
        'attn.norm_q.weight': 'self_attention.q_layernorm.weight',
        'attn.norm_k.weight': 'self_attention.k_layernorm.weight',
        'attn.norm_added_q.weight': 'self_attention.added_q_layernorm.weight',
        'attn.norm_added_k.weight': 'self_attention.added_k_layernorm.weight',
        'attn.to_out.0.weight': 'self_attention.linear_proj.weight',
        'attn.to_out.0.bias': 'self_attention.linear_proj.bias',
        'attn.to_add_out.weight': 'self_attention.added_linear_proj.weight',
        'attn.to_add_out.bias': 'self_attention.added_linear_proj.bias',
        'ff.net.0.proj.weight': 'mlp.linear_fc1.weight',
        'ff.net.0.proj.bias': 'mlp.linear_fc1.bias',
        'ff.net.2.weight': 'mlp.linear_fc2.weight',
        'ff.net.2.bias': 'mlp.linear_fc2.bias',
        'ff_context.net.0.proj.weight': 'context_mlp.linear_fc1.weight',
        'ff_context.net.0.proj.bias': 'context_mlp.linear_fc1.bias',
        'ff_context.net.2.weight': 'context_mlp.linear_fc2.weight',
        'ff_context.net.2.bias': 'context_mlp.linear_fc2.bias',
    },
    'single_blocks': {
        'norm.linear.weight': 'adaln.adaLN_modulation.1.weight',
        'norm.linear.bias': 'adaln.adaLN_modulation.1.bias',
        'proj_mlp.weight': 'mlp.linear_fc1.weight',
        'proj_mlp.bias': 'mlp.linear_fc1.bias',
        # 'proj_out.weight': 'proj_out.weight',
        # 'proj_out.bias': 'proj_out.bias',
        'attn.norm_q.weight': 'self_attention.q_layernorm.weight',
        'attn.norm_k.weight': 'self_attention.k_layernorm.weight',
    },
    'norm_out.linear.bias': 'norm_out.adaLN_modulation.1.bias',
    'norm_out.linear.weight': 'norm_out.adaLN_modulation.1.weight',
    'proj_out.bias': 'proj_out.bias',
    'proj_out.weight': 'proj_out.weight',
    'time_text_embed.guidance_embedder.linear_1.bias': 'guidance_embedding.in_layer.bias',
    'time_text_embed.guidance_embedder.linear_1.weight': 'guidance_embedding.in_layer.weight',
    'time_text_embed.guidance_embedder.linear_2.bias': 'guidance_embedding.out_layer.bias',
    'time_text_embed.guidance_embedder.linear_2.weight': 'guidance_embedding.out_layer.weight',
    'x_embedder.bias': 'img_embed.bias',
    'x_embedder.weight': 'img_embed.weight',
    'time_text_embed.timestep_embedder.linear_1.bias': 'timestep_embedding.time_embedder.in_layer.bias',
    'time_text_embed.timestep_embedder.linear_1.weight': 'timestep_embedding.time_embedder.in_layer.weight',
    'time_text_embed.timestep_embedder.linear_2.bias': 'timestep_embedding.time_embedder.out_layer.bias',
    'time_text_embed.timestep_embedder.linear_2.weight': 'timestep_embedding.time_embedder.out_layer.weight',
    'context_embedder.bias': 'txt_embed.bias',
    'context_embedder.weight': 'txt_embed.weight',
    'time_text_embed.text_embedder.linear_1.bias': 'vector_embedding.in_layer.bias',
    'time_text_embed.text_embedder.linear_1.weight': 'vector_embedding.in_layer.weight',
    'time_text_embed.text_embedder.linear_2.bias': 'vector_embedding.out_layer.bias',
    'time_text_embed.text_embedder.linear_2.weight': 'vector_embedding.out_layer.weight',
    'controlnet_x_embedder.weight': 'controlnet_x_embedder.weight',
    'controlnet_x_embedder.bias': 'controlnet_x_embedder.bias',
}


def flux_transformer_converter(ckpt_path=None, transformer_config=None):
    # pylint: disable=C0116
    diffuser_state_dict = {}
    if os.path.isdir(ckpt_path):
        files = os.listdir(ckpt_path)
        for file in files:
            if file.endswith('.safetensors'):
                loaded_dict = load_safetensors(os.path.join(ckpt_path, file))
                diffuser_state_dict.update(loaded_dict)
    elif os.path.isfile(ckpt_path):
        diffuser_state_dict = load_safetensors(ckpt_path)
    else:
        raise FileNotFoundError("Please provide a valid ckpt path.")
    new_state_dict = {}
    num_single_blocks = -1
    num_double_blocks = -1
    for key, value in diffuser_state_dict.items():
        if 'attn.to_q' in key or 'attn.to_k' in key or 'attn.to_v' in key:
            continue
        if 'attn.add_q_proj' in key or 'attn.add_k_proj' in key or 'attn.add_v_proj' in key:
            continue
        if key.startswith('transformer_blocks'):
            temp = key.split('.')
            idx, k = temp[1], '.'.join(temp[2:])
            num_double_blocks = max(int(idx), num_double_blocks)
            new_key = '.'.join(['double_blocks', idx, flux_key_mapping['double_blocks'][k]])
        elif key.startswith('single_transformer_blocks'):
            if 'proj_out' in key:
                continue
            temp = key.split('.')
            idx, k = temp[1], '.'.join(temp[2:])
            num_single_blocks = max(int(idx), num_single_blocks)
            new_key = '.'.join(['single_blocks', idx, flux_key_mapping['single_blocks'][k]])
        elif key.startswith('controlnet_blocks'):
            new_key = 'controlnet_double_blocks.' + '.'.join(key.split('.')[1:])
        else:
            new_key = flux_key_mapping[key]
        new_state_dict[new_key] = value
    for i in range(num_double_blocks + 1):
        new_key = f'double_blocks.{str(i)}.self_attention.linear_qkv.weight'
        qk, kk, vk = [f'transformer_blocks.{str(i)}.attn.to_{n}.weight' for n in ('q', 'k', 'v')]
        new_state_dict[new_key] = _import_qkv(
            transformer_config, diffuser_state_dict[qk], diffuser_state_dict[kk], diffuser_state_dict[vk]
        )
        new_key = f'double_blocks.{str(i)}.self_attention.linear_qkv.bias'
        qk, kk, vk = [f'transformer_blocks.{str(i)}.attn.to_{n}.bias' for n in ('q', 'k', 'v')]
        new_state_dict[new_key] = _import_qkv_bias(
            transformer_config, diffuser_state_dict[qk], diffuser_state_dict[kk], diffuser_state_dict[vk]
        )
        new_key = f'double_blocks.{str(i)}.self_attention.added_linear_qkv.weight'
        qk, kk, vk = [f'transformer_blocks.{str(i)}.attn.add_{n}_proj.weight' for n in ('q', 'k', 'v')]
        new_state_dict[new_key] = _import_qkv(
            transformer_config, diffuser_state_dict[qk], diffuser_state_dict[kk], diffuser_state_dict[vk]
        )
        new_key = f'double_blocks.{str(i)}.self_attention.added_linear_qkv.bias'
        qk, kk, vk = [f'transformer_blocks.{str(i)}.attn.add_{n}_proj.bias' for n in ('q', 'k', 'v')]
        new_state_dict[new_key] = _import_qkv_bias(
            transformer_config, diffuser_state_dict[qk], diffuser_state_dict[kk], diffuser_state_dict[vk]
        )

    for i in range(num_single_blocks + 1):
        new_key = f'single_blocks.{str(i)}.self_attention.linear_qkv.weight'
        qk, kk, vk = [f'single_transformer_blocks.{str(i)}.attn.to_{n}.weight' for n in ('q', 'k', 'v')]
        new_state_dict[new_key] = _import_qkv(
            transformer_config, diffuser_state_dict[qk], diffuser_state_dict[kk], diffuser_state_dict[vk]
        )
        new_key = f'single_blocks.{str(i)}.self_attention.linear_qkv.bias'
        qk, kk, vk = [f'single_transformer_blocks.{str(i)}.attn.to_{n}.bias' for n in ('q', 'k', 'v')]
        new_state_dict[new_key] = _import_qkv_bias(
            transformer_config, diffuser_state_dict[qk], diffuser_state_dict[kk], diffuser_state_dict[vk]
        )

        (
            new_state_dict[f'single_blocks.{str(i)}.mlp.linear_fc2.weight'],
            new_state_dict[f'single_blocks.{str(i)}.self_attention.linear_proj.weight'],
        ) = (
            diffuser_state_dict[f'single_transformer_blocks.{str(i)}.proj_out.weight'].detach()[:, 3072:].clone(),
            diffuser_state_dict[f'single_transformer_blocks.{str(i)}.proj_out.weight'].detach()[:, :3072].clone(),
        )

        new_state_dict[f'single_blocks.{str(i)}.mlp.linear_fc2.bias'] = (
            diffuser_state_dict[f'single_transformer_blocks.{str(i)}.proj_out.bias'].detach().clone()
        )
        new_state_dict[f'single_blocks.{str(i)}.self_attention.linear_proj.bias'] = (
            diffuser_state_dict[f'single_transformer_blocks.{str(i)}.proj_out.bias'].detach().clone()
        )

    return new_state_dict
