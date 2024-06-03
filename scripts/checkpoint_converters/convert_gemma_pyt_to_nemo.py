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
Requires to install: `pip install fairscale==0.4.13 immutabledict==4.1.0 tensorstore==0.1.45`
Requires to clone: https://github.com/google/gemma_pytorch.git
Required to set: `export PYTHONPATH=/path/to/google/gemma_pytorchh:$PYTHONPATH`
   python3 /opt/NeMo/scripts/nlp_language_modeling/convert_gemma_pyt_to_nemo.py \
   --input_name_or_path /path/to/gemma/checkpoints/pyt/7b.ckpt \
   --output_path /path/to/gemma-7b.nemo \
   --tokenizer_path /path/to/tokenizer.model
"""

import contextlib
import os
from argparse import ArgumentParser

import torch
from gemma.config import get_config_for_2b, get_config_for_7b
from gemma.model import CausalLM
from gemma.tokenizer import Tokenizer
from omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging

PAD_TOKEN_ID = -1


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def create_rename_keys(num_hidden_layers):
    rename_keys = []
    for i in range(num_hidden_layers):
        # Attention layers
        rename_keys.extend(
            [
                (
                    f"model.layers.{i}.self_attn.o_proj.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_proj.weight",
                ),
                (
                    f"model.layers.{i}.self_attn.qkv_proj.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_qkv.weight",
                ),
                # MLP and LayerNorm
                (f"model.layers.{i}.mlp.gate_proj.weight", f"model.decoder.layers.{i}.mlp.linear_fc1_gate.weight"),
                (f"model.layers.{i}.mlp.up_proj.weight", f"model.decoder.layers.{i}.mlp.linear_fc1_proj.weight"),
                (f"model.layers.{i}.mlp.down_proj.weight", f"model.decoder.layers.{i}.mlp.linear_fc2.weight"),
                (
                    f"model.layers.{i}.input_layernorm.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight",
                ),
                (
                    f"model.layers.{i}.post_attention_layernorm.weight",
                    f"model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight",
                ),
            ]
        )

    # Non layer dependent keys
    rename_keys.extend(
        [
            ("embedder.weight", "model.embedding.word_embeddings.weight"),
            ("model.norm.weight", "model.decoder.final_layernorm.weight"),
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
    model_config = model.cfg
    num_query_groups = model_config["num_query_groups"]
    head_num = model_config["num_attention_heads"]
    head_size = model_config["kv_channels"]
    hidden_size = model_config["hidden_size"]
    heads_per_group = head_num // num_query_groups

    # Note: For 'key' and 'value' weight and biases, NeMo uses a consolidated tensor 'query_key_value'.
    for key_ in list(nemo_state_dict.keys()):
        if 'mlp.linear_fc1_gate.weight' in key_:
            key_gate = key_
            key_proj = key_.replace('mlp.linear_fc1_gate.weight', 'mlp.linear_fc1_proj.weight')
            new_key = key_.replace('mlp.linear_fc1_gate.weight', 'mlp.linear_fc1.weight')
            gate_weight = nemo_state_dict[key_gate]
            proj_weight = nemo_state_dict[key_proj]
            nemo_state_dict[new_key] = torch.cat((gate_weight, proj_weight))
        if 'layernorm.weight' in key_ or 'layer_norm_weight' in key_:
            nemo_state_dict[key_] = nemo_state_dict[key_] + 1.0
        if 'self_attention.linear_qkv.weight' in key_:
            qkv_weight = nemo_state_dict[key_]
            # [(head_num + 2 * num_query_groups) * head_size, hidden_size]
            # -> [head_num, head_size, hidden_size], 2 * [num_query_groups, head_size, hidden_size]
            q_weight, k_weight, v_weight = qkv_weight.split(
                [head_num * head_size, num_query_groups * head_size, num_query_groups * head_size],
                dim=0,
            )
            q_weight = q_weight.reshape(head_num, head_size, hidden_size)
            k_weight = k_weight.reshape(num_query_groups, head_size, hidden_size)
            v_weight = v_weight.reshape(num_query_groups, head_size, hidden_size)

            qkv_weight = torch.empty((0, head_size, hidden_size), device=q_weight.device)
            for i in range(num_query_groups):
                qkv_weight = torch.cat((qkv_weight, q_weight[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
                qkv_weight = torch.cat((qkv_weight, k_weight[i : i + 1, :, :]))
                qkv_weight = torch.cat((qkv_weight, v_weight[i : i + 1, :, :]))
            qkv_weight = qkv_weight.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])
            nemo_state_dict[key_] = qkv_weight

    return nemo_state_dict


def adjust_nemo_config(model_config, ref_config):
    model_config.tokenizer["model"] = ref_config["tokenizer"]  # ref_config["_input_name_or_path"]
    model_config["encoder_seq_length"] = ref_config["max_position_embeddings"]
    model_config["num_layers"] = ref_config["num_hidden_layers"]
    model_config["ffn_hidden_size"] = ref_config["intermediate_size"]
    model_config["hidden_size"] = ref_config["hidden_size"]
    model_config["num_attention_heads"] = ref_config["num_attention_heads"]
    model_config["num_query_groups"] = ref_config["num_key_value_heads"]
    model_config["kv_channels"] = ref_config["head_dim"]
    if "model_vocab_size" in ref_config:
        model_config["override_vocab_size"] = ref_config["model_vocab_size"]
    model_config["layernorm_epsilon"] = ref_config["rms_norm_eps"]
    return model_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input_name_or_path", type=str)
    parser.add_argument("--tokenizer_path", default=None, type=str)
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
    logging.info(f"Loading checkpoint from PyT Gemma: `{args.input_name_or_path}`")
    if "2b" in args.input_name_or_path:
        pyt_config = get_config_for_2b()
    else:
        pyt_config = get_config_for_7b()
    if args.tokenizer_path is not None:
        pyt_config.tokenizer = args.tokenizer_path

    device = torch.device("cuda")
    pyt_config.dtype = "bfloat16" if args.precision == "bf16" else "float32"

    with _set_default_tensor_type(pyt_config.get_dtype()):
        pyt_model = CausalLM(pyt_config)
        pyt_model.load_weights(args.input_name_or_path)
        pyt_model = pyt_model.to(device).eval()
    tokenizer = Tokenizer(pyt_config.tokenizer)
    logging.info("Model loading done")

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.model = adjust_nemo_config(nemo_config.model, pyt_config.__dict__)

    nemo_config.trainer["precision"] = args.precision
    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronGPTModel(nemo_config.model, trainer)

    rename_keys = create_rename_keys(nemo_config.model.num_layers)
    old_state_dict = pyt_model.state_dict()
    new_state_dict = rename_model_keys(model_state_dict=old_state_dict, rename_keys=rename_keys)

    nemo_state_dict = adjust_tensor_shapes(model, new_state_dict)
    model.load_state_dict(nemo_state_dict, strict=False)

    logging.info(f'=' * 50)

    # Mock inputs
    prompts = [
        "The capital of France is",
    ]
    top_ps = [1.0]
    top_ks = [1]
    output_len = 1
    batch_size = len(prompts)

    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    min_prompt_len = min(len(p) for p in prompt_tokens)
    max_prompt_len = max(len(p) for p in prompt_tokens)
    max_seq_len = max_prompt_len + output_len

    # build KV caches
    kv_caches = []
    for _ in range(pyt_config.num_hidden_layers):
        k_cache = torch.zeros(
            size=(batch_size, max_seq_len, pyt_config.num_key_value_heads, pyt_config.head_dim),
            dtype=pyt_config.get_dtype(),
            device=device,
        )
        v_cache = torch.zeros(
            size=(batch_size, max_seq_len, pyt_config.num_key_value_heads, pyt_config.head_dim),
            dtype=pyt_config.get_dtype(),
            device=device,
        )
        kv_caches.append((k_cache, v_cache))

    # prepare inputs
    token_ids_tensor = torch.full((batch_size, max_seq_len), PAD_TOKEN_ID, dtype=torch.int64)
    input_token_ids_tensor = torch.full((batch_size, min_prompt_len), PAD_TOKEN_ID, dtype=torch.int64)
    for i, p in enumerate(prompt_tokens):
        token_ids_tensor[i, : len(p)] = torch.tensor(p)
        input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])
    input_token_ids_tensor = input_token_ids_tensor.to(device)
    input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64).to(device)
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    top_ps_tensor = torch.FloatTensor(top_ps).to(device)
    top_ks_tensor = torch.LongTensor(top_ks).to(device)

    pyt_outputs = pyt_model(
        input_token_ids=input_token_ids_tensor,
        input_positions=input_positions_tensor,
        kv_write_indices=None,
        kv_caches=kv_caches,
        mask=curr_mask_tensor,
        output_positions=output_positions_tensor,
        temperatures=None,
        top_ps=top_ps_tensor,
        top_ks=top_ks_tensor,
    )
    nemo_outputs = model(
        tokens=input_token_ids_tensor,
        text_position_ids=input_positions_tensor,
        attention_mask=curr_mask_tensor,
        labels=None,
    )
    assert torch.argmax(nemo_outputs[0, -1], dim=-1) == pyt_outputs, "Predicted next token not match."

    dtype = torch_dtype_from_precision(args.precision)
    model = model.to(dtype=dtype)
    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
