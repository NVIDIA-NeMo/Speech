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

"""
Requires to install: `pip install orbax jax flax jaxlib`
Requires to clone: https://github.com/google-deepmind/gemma.git
Required to set: `export PYTHONPATH=/path/to/google/gemma_jax:$PYTHONPATH`
   python3 /opt/NeMo/scripts/nlp_language_modeling/convert_gemma_jax_to_nemo.py \
   --input_name_or_path /path/to/gemma/checkpoints/jax/7b \
   --output_path /path/to/gemma-7b.nemo \
   --tokenizer_path /path/to/tokenizer.model
"""

import os
import os.path as osp
from argparse import ArgumentParser

import jax
import torch
from gemma.params import load_params, nest_params, param_remapper
from omegaconf import OmegaConf
from transformer import TransformerConfig

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging


def create_rename_keys(num_hidden_layers):
    rename_keys = []
    for i in range(num_hidden_layers):
        # Attention layers
        rename_keys.extend(
            [
                (
                    f"transformer.layer_{i}.attn.attn_vec_einsum.w",
                    f"model.decoder.layers.{i}.self_attention.linear_proj.weight",
                ),
                (
                    f"transformer.layer_{i}.attn.qkv_einsum.w",
                    f"model.decoder.layers.{i}.self_attention.linear_qkv.weight",
                ),
                (
                    f"transformer.layer_{i}.attn.kv_einsum.w",
                    f"model.decoder.layers.{i}.self_attention.linear_kv.weight",
                ),
                (f"transformer.layer_{i}.attn.q_einsum.w", f"model.decoder.layers.{i}.self_attention.linear_q.weight"),
                # MLP and LayerNorm
                (f"transformer.layer_{i}.mlp.gating_einsum", f"model.decoder.layers.{i}.mlp.linear_fc1.weight"),
                (f"transformer.layer_{i}.mlp.linear", f"model.decoder.layers.{i}.mlp.linear_fc2.weight"),
                (
                    f"transformer.layer_{i}.pre_attention_norm.scale",
                    f"model.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight",
                ),
                (
                    f"transformer.layer_{i}.pre_ffw_norm.scale",
                    f"model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight",
                ),
            ]
        )

    # Non layer dependent keys
    rename_keys.extend(
        [
            ("transformer.embedder.input_embedding", "model.embedding.word_embeddings.weight"),
            ("transformer.final_norm.scale", "model.decoder.final_layernorm.weight"),
        ]
    )

    return rename_keys


def rename_model_keys(model_state_dict, rename_keys):
    """
    Rename keys in the model's state dictionary based on the provided mappings.

    Parameters:
    model_state_dict (dict): The state dictionary of the model.
    rename_keys (list): A list of tuples with the mapping (old_key, new_key).

    Returns:
    dict: A new state dictionary with updated key names.
    """

    # Create a new state dictionary with updated key names
    new_state_dict = {}

    # Track keys from the original state dict to ensure all are processed
    remaining_keys = set(model_state_dict.keys())

    # Iterate over the rename mappings
    for old_key, new_key in rename_keys:
        if old_key in model_state_dict:
            # Rename the key and remove it from the tracking set
            new_state_dict[new_key] = model_state_dict[old_key]
            remaining_keys.remove(old_key)

    # Check if any keys were not converted from old to new
    for old_key in remaining_keys:
        print(f"Warning: Key '{old_key}' was not converted.")

    return new_state_dict


def adjust_tensor_shapes(model, nemo_state_dict):
    """
    Adapt tensor shapes in the state dictionary to ensure compatibility with a different model structure.

    Parameters:
    nemo_state_dict (dict): The state dictionary of the model.

    Returns:
    dict: The updated state dictionary with modified tensor shapes for compatibility.
    """

    # Note: For 'key' and 'value' weight and biases, NeMo uses a consolidated tensor 'query_key_value'.
    for key_ in list(nemo_state_dict.keys()):
        if 'self_attention.linear_proj' in key_:
            weight = nemo_state_dict[key_]
            nemo_state_dict[key_] = weight.reshape(-1, weight.shape[2]).transpose(0, 1)
        if 'mlp.linear_fc1.weight' in key_:
            weight = nemo_state_dict[key_]
            nemo_state_dict[key_] = weight.transpose(1, 2).reshape(-1, weight.shape[1])
        if 'mlp.linear_fc2.weight' in key_:
            nemo_state_dict[key_] = nemo_state_dict[key_].transpose(0, 1)
        if 'layernorm.weight' in key_ or 'layer_norm_weight' in key_:
            nemo_state_dict[key_] = nemo_state_dict[key_] + 1.0
        if 'self_attention.linear_qkv.weight' in key_:
            weight_qkv = nemo_state_dict[key_]
            # [3, head_num, hidden_dim, head_size] -> [head_num, 3, head_dim, hidden_dim] -> [-1, hidden_dim]
            nemo_state_dict[key_] = weight_qkv.permute(1, 0, 3, 2).reshape(-1, weight_qkv.shape[2])
        if 'self_attention.linear_q.weight' in key_:
            key_q = key_
            key_kv = key_.replace('self_attention.linear_q', 'self_attention.linear_kv')
            key_qkv = key_.replace('self_attention.linear_q', 'self_attention.linear_qkv')
            # [head_num, hidden_dim, head_size] -> [head_num, head_size, hidden_dim]
            q_weight = nemo_state_dict[key_q].transpose(1, 2)

            # [2, num_query_groups, hidden_dim, head_size] -> 2 * [num_query_groups, head_size, hidden_dim]
            k_weight, v_weight = nemo_state_dict[key_kv]
            k_weight = k_weight.transpose(1, 2)
            v_weight = v_weight.transpose(1, 2)
            head_num, head_size, hidden_size = q_weight.shape
            num_query_groups = k_weight.shape[0]
            heads_per_group = head_num // num_query_groups

            qkv_weight = torch.empty((0, head_size, hidden_size), device=q_weight.device)
            for i in range(num_query_groups):
                qkv_weight = torch.cat((qkv_weight, q_weight[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
                qkv_weight = torch.cat((qkv_weight, k_weight[i : i + 1, :, :]))
                qkv_weight = torch.cat((qkv_weight, v_weight[i : i + 1, :, :]))
            qkv_weight = qkv_weight.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])
            nemo_state_dict[key_qkv] = qkv_weight
            del nemo_state_dict[key_q], nemo_state_dict[key_kv]

    return nemo_state_dict


def adjust_nemo_config(model_config, ref_config):
    model_config["num_layers"] = ref_config["num_layers"]
    model_config["hidden_size"] = ref_config["embed_dim"]
    model_config["ffn_hidden_size"] = ref_config["hidden_dim"]
    model_config["num_attention_heads"] = ref_config["num_heads"]
    model_config["num_query_groups"] = ref_config["num_kv_heads"]
    model_config["override_vocab_size"] = ref_config["num_embed"]
    model_config["kv_channels"] = ref_config["head_dim"]
    return model_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input_name_or_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), '../../examples/nlp/language_modeling/conf/megatron_gemma_config.yaml'
        ),
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Path to output .nemo file.")
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weight saved"
    )

    args = parser.parse_args()
    return args


def convert(args):
    logging.info(f"Loading checkpoint from jax: `{args.input_name_or_path}`")
    jax_params = param_remapper(load_params(osp.realpath(args.input_name_or_path)))
    # Convert Jax checkpoint to Pytorch state dict
    old_state_dict = {}
    for k, v in jax_params.items():
        for sub_k, sub_v in v.items():
            new_k = k.replace("/", ".") + "." + sub_k
            old_state_dict[new_k] = torch.from_numpy(jax.device_get(sub_v).astype('float32'))
    jax_params = nest_params(jax_params)
    jax_config = TransformerConfig.from_params(jax_params, num_embed=256128)

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.model = adjust_nemo_config(nemo_config.model, jax_config.__dict__)

    nemo_config.trainer["precision"] = args.precision
    nemo_config.model.tokenizer["model"] = args.tokenizer_path
    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronGPTModel(nemo_config.model, trainer)

    rename_keys = create_rename_keys(nemo_config.model.num_layers)
    new_state_dict = rename_model_keys(model_state_dict=old_state_dict, rename_keys=rename_keys)
    nemo_state_dict = adjust_tensor_shapes(model, new_state_dict)
    model.load_state_dict(nemo_state_dict, strict=False)

    dtype = torch_dtype_from_precision(args.precision)
    model = model.to(dtype=dtype)
    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
