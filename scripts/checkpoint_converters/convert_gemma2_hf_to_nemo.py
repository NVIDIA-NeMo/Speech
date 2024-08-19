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
Requires HF transformers updated to v4.42 to support Gemma 2 Models

    huggingface-cli login
    >>> from huggingface_hub import snapshot_download
    >>> snapshot_download(repo_id="google/gemma-2-9b", local_dir="/path/to/gemma2/checkpoints/hf/9b")

    python3 /opt/NeMo/scripts/checkpoint_converters/convert_gemma2_hf_to_nemo.py \
    --input_name_or_path /path/to/gemma2/checkpoints/hf/9b \
    --output_path /path/to/gemma2-9b.nemo \
    --tokenizer_path /path/to/gemma2/checkpoints/hf/9b/tokenizer.model
    [--cpu]

If you encounter a torch.cuda.OutOfMemoryError, try converting on CPU with --cpu.
"""

import os
from argparse import ArgumentParser

import torch

from megatron.core import parallel_state
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
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
                    f"model.layers.{i}.self_attn.o_proj.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_proj.weight",
                ),
                (
                    f"model.layers.{i}.self_attn.q_proj.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_q.weight",
                ),
                (
                    f"model.layers.{i}.self_attn.k_proj.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_k.weight",
                ),
                (
                    f"model.layers.{i}.self_attn.v_proj.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_v.weight",
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
                    f"model.layers.{i}.pre_feedforward_layernorm.weight",
                    f"model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight",
                ),
                (
                    f"model.layers.{i}.post_attention_layernorm.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_proj.post_layernorm.weight",
                ),
                (
                    f"model.layers.{i}.post_feedforward_layernorm.weight",
                    f"model.decoder.layers.{i}.mlp.linear_fc2.post_layernorm.weight",
                ),
            ]
        )

    # Non layer dependent keys
    rename_keys.extend(
        [
            ("model.embed_tokens.weight", "model.embedding.word_embeddings.weight"),
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
    hidden_size = model_config["hidden_size"]
    head_size = model_config["kv_channels"]
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
            nemo_state_dict[key_] = nemo_state_dict[key_]
        if 'self_attention.linear_q.weight' in key_:
            key_q = key_
            key_k = key_.replace('linear_q', 'linear_k')
            key_v = key_.replace('linear_q', 'linear_v')
            key_qkv = key_.replace('linear_q', 'linear_qkv')

            # [(head_num + 2 * num_query_groups) * head_size, hidden_size]
            # -> [head_num, head_size, hidden_size], 2 * [num_query_groups, head_size, hidden_size]
            q_weight, k_weight, v_weight = nemo_state_dict[key_q], nemo_state_dict[key_k], nemo_state_dict[key_v]
            q_weight = q_weight.reshape(head_num, head_size, hidden_size)
            k_weight = k_weight.reshape(num_query_groups, head_size, hidden_size)
            v_weight = v_weight.reshape(num_query_groups, head_size, hidden_size)

            qkv_weight = torch.empty((0, head_size, hidden_size), device=q_weight.device)
            for i in range(num_query_groups):
                qkv_weight = torch.cat((qkv_weight, q_weight[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
                qkv_weight = torch.cat((qkv_weight, k_weight[i : i + 1, :, :]))
                qkv_weight = torch.cat((qkv_weight, v_weight[i : i + 1, :, :]))
            qkv_weight = qkv_weight.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])
            nemo_state_dict[key_qkv] = qkv_weight
            del nemo_state_dict[key_q], nemo_state_dict[key_k], nemo_state_dict[key_v]

    return nemo_state_dict


def adjust_nemo_config(model_config, ref_config):
    model_config["encoder_seq_length"] = ref_config["max_position_embeddings"]
    model_config["num_layers"] = ref_config["num_hidden_layers"]
    model_config["ffn_hidden_size"] = ref_config["intermediate_size"]
    model_config["hidden_size"] = ref_config["hidden_size"]
    model_config["num_attention_heads"] = ref_config["num_attention_heads"]
    model_config["num_query_groups"] = ref_config["num_key_value_heads"]
    model_config["kv_channels"] = ref_config["head_dim"]
    model_config["layernorm_epsilon"] = ref_config["rms_norm_eps"]
    model_config["window_size"] = (ref_config["sliding_window_size"], 0)
    model_config["layernorm_zero_centered_gamma"] = True
    model_config["name"] = 'megatron_gemma2'
    model_config['mcore_customization_config'] = {
        "attn_logit_softcapping": ref_config["attn_logit_softcapping"],
        "final_logit_softcapping": ref_config["final_logit_softcapping"],
        "query_pre_attn_scalar": ref_config["query_pre_attn_scalar"],
    }
    return model_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input_name_or_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), '../../examples/nlp/language_modeling/conf/megatron_gemma2_config.yaml'
        ),
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Path to output .nemo file.")
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weight saved"
    )
    parser.add_argument("--run_verification", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    return args


def verify(nemo_model, hf_tokenizer, hf_model):
    # Verifications
    input_texts = [
        'query: how much protein should a female eat',
    ]
    logging.info(f"Running verifications {input_texts} ...")

    # Tokenize the input texts
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    batch_dict = hf_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict_cuda = {k: v.cuda() for k, v in batch_dict.items()}
    hf_model = hf_model.cuda().eval()
    nemo_model = nemo_model.eval()

    hf_outputs = hf_model(**batch_dict_cuda, output_hidden_states=True)

    parallel_state._set_global_memory_buffer()
    ids = batch_dict_cuda['input_ids']

    id_tensors = [torch.unsqueeze(torch.LongTensor(id_list), dim=0) for id_list in ids.cpu()]

    masks_and_position_ids = [
        get_ltor_masks_and_position_ids(id_tensor, hf_tokenizer.eos_token, False, False, False)
        for id_tensor in id_tensors
    ]
    for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
        attn_mask, _, pos_ids = attn_mask_and_pos_ids
        outputs = nemo_model(
            tokens=tokens.cuda(), text_position_ids=pos_ids.cuda(), attention_mask=attn_mask.cuda(), labels=None
        )

    hf_next_token = hf_outputs.logits[0, -1].argmax()
    next_token = outputs.squeeze()[-1].argmax()

    logging.info(f"HF predicted next token is: '{hf_tokenizer._convert_id_to_token(hf_next_token)}'.")
    logging.info(f"NeMo predicted next token is: '{hf_tokenizer._convert_id_to_token(next_token)}'.")
    assert (
        hf_next_token == next_token
    ), f'prediction mismatch: {hf_tokenizer.decode(hf_next_token)} != {hf_tokenizer.decode(next_token)}'


def convert(args):
    logging.info(f"Loading checkpoint from HF Gemma 2: `{args.input_name_or_path}`")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.input_name_or_path)
    hf_model = AutoModelForCausalLM.from_pretrained(args.input_name_or_path)
    logging.info("HF Model loading done.")

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.model = adjust_nemo_config(nemo_config.model, hf_model.config.__dict__)
    nemo_config.model.tokenizer["model"] = args.tokenizer_path

    nemo_config.trainer["precision"] = args.precision
    if args.cpu:
        nemo_config.model['use_cpu_initialization'] = True
        nemo_config.trainer['accelerator'] = 'cpu'
    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronGPTModel(nemo_config.model, trainer)

    rename_keys = create_rename_keys(nemo_config.model.num_layers)
    old_state_dict = hf_model.state_dict()
    new_state_dict = rename_model_keys(model_state_dict=old_state_dict, rename_keys=rename_keys)

    nemo_state_dict = adjust_tensor_shapes(model, new_state_dict)
    model.load_state_dict(nemo_state_dict, strict=False)

    if args.run_verification and not args.cpu:
        logging.info(f'=' * 100)
        verify(model, hf_tokenizer, hf_model)
        logging.info(f'=' * 100)

    dtype = torch_dtype_from_precision(args.precision)
    model = model.to(dtype=dtype)
    model.cfg.use_cpu_initialization = False
    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
