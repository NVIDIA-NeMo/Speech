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
   python3 /opt/NeMo/scripts/nlp_language_modeling/convert_gemma_hf_to_nemo.py \
   --input_name_or_path /path/to/llava-v1.5-7b.nemo \
   --hf_input_path llava-hf/llava-1.5-7b-hf \
   --hf_output_path=/path/to/hf_updated_checkpoint
"""

import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from transformers import LlamaTokenizer, LlavaForConditionalGeneration

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging


def create_rename_keys(num_hidden_layers):
    rename_keys = []
    for i in range(num_hidden_layers):
        # Attention layers
        rename_keys.extend(
            [
                (
                    f"language_model.model.layers.{i}.self_attn.o_proj.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_proj.weight",
                ),
                (
                    f"language_model.model.layers.{i}.self_attn.q_proj.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_q.weight",
                ),
                (
                    f"language_model.model.layers.{i}.self_attn.k_proj.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_k.weight",
                ),
                (
                    f"language_model.model.layers.{i}.self_attn.v_proj.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_v.weight",
                ),
                # MLP and LayerNorm
                (
                    f"language_model.model.layers.{i}.mlp.gate_proj.weight",
                    f"model.decoder.layers.{i}.mlp.linear_fc1_gate.weight",
                ),
                (
                    f"language_model.model.layers.{i}.mlp.up_proj.weight",
                    f"model.decoder.layers.{i}.mlp.linear_fc1_proj.weight",
                ),
                (
                    f"language_model.model.layers.{i}.mlp.down_proj.weight",
                    f"model.decoder.layers.{i}.mlp.linear_fc2.weight",
                ),
                (
                    f"language_model.model.layers.{i}.input_layernorm.weight",
                    f"model.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight",
                ),
                (
                    f"language_model.model.layers.{i}.post_attention_layernorm.weight",
                    f"model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight",
                ),
            ]
        )

    rename_keys.extend(
        [
            (
                "multi_modal_projector.linear_1.weight",
                "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.0.weight",
            ),
            (
                "multi_modal_projector.linear_1.bias",
                "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.0.bias",
            ),
            (
                "multi_modal_projector.linear_2.weight",
                "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.2.weight",
            ),
            (
                "multi_modal_projector.linear_2.bias",
                "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.2.bias",
            ),
            ("language_model.model.embed_tokens.weight", "model.embedding.word_embeddings.weight"),
            ("language_model.model.norm.weight", "model.decoder.final_layernorm.weight"),
            ("language_model.lm_head.weight", "model.output_layer.weight"),
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
    for new_key, old_key in rename_keys:
        if old_key in model_state_dict:
            # Rename the key and remove it from the tracking set
            new_state_dict[new_key] = model_state_dict[old_key]
            remaining_keys.remove(old_key)

    # Check if any keys were not converted from old to new
    for old_key in remaining_keys:
        print(f"Warning: Key '{old_key}' was not converted.")

    return new_state_dict


def reverse_adjust_tensor_shapes(model, hf_model, nemo_state_dict):
    """
    Reverse the tensor adjustments made in the state dictionary to retrieve the original model structure.

    Parameters:
    model (torch.nn.Module): The model instance to reference the state dictionary.
    nemo_state_dict (dict): The state dictionary containing the adjusted tensors.

    Returns:
    dict: The updated state dictionary with original tensor shapes and structures.
    """
    model_config = model.cfg
    head_num = model_config["num_attention_heads"]
    hidden_size = model_config["hidden_size"]
    head_size = model_config["kv_channels"]
    if "num_query_groups" in model_config and model_config["num_query_groups"] is not None:
        num_query_groups = model_config["num_query_groups"]
    else:
        num_query_groups = head_num
    if head_size is None:
        head_size = hidden_size // head_num
    heads_per_group = head_num // num_query_groups
    vocab_size = hf_model.config.vocab_size

    for key_ in list(nemo_state_dict.keys()):
        if 'word_embeddings.weight' in key_ or 'output_layer.weight' in key_:
            # Reverse padding
            loaded_weight = model.state_dict()[key_]
            nemo_state_dict[key_] = loaded_weight[:vocab_size]

        if 'mlp.linear_fc1.weight' in key_:
            new_key_gate = key_.replace('mlp.linear_fc1.weight', 'mlp.linear_fc1_gate.weight')
            new_key_proj = key_.replace('mlp.linear_fc1.weight', 'mlp.linear_fc1_proj.weight')

            # Split concatenated gate and projection weights
            combined_weight = nemo_state_dict[key_]
            gate_weight, proj_weight = torch.chunk(combined_weight, 2, dim=0)
            nemo_state_dict[new_key_gate] = gate_weight
            nemo_state_dict[new_key_proj] = proj_weight
            del nemo_state_dict[key_]

        if 'self_attention.linear_qkv.weight' in key_:
            key_qkv = key_
            key_q = key_qkv.replace('linear_qkv', 'linear_q')
            key_k = key_qkv.replace('linear_qkv', 'linear_k')
            key_v = key_qkv.replace('linear_qkv', 'linear_v')
            qkv_weight = nemo_state_dict[key_qkv].reshape(-1, head_size, hidden_size)
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

            nemo_state_dict[key_q] = q_weight.reshape(head_num * head_size, hidden_size)
            nemo_state_dict[key_k] = k_weight.reshape(num_query_groups * head_size, hidden_size)
            nemo_state_dict[key_v] = v_weight.reshape(num_query_groups * head_size, hidden_size)

            del nemo_state_dict[key_qkv]

    return nemo_state_dict


def adjust_nemo_config(model_config, ref_config):
    model_config.mm_cfg.mm_mlp_adapter_type = "mlp2x_gelu"
    if ref_config["vision_config"].image_size == 336:
        model_config.mm_cfg.vision_encoder.from_pretrained = "openai/clip-vit-large-patch14-336"
        model_config.data.image_token_len = 576
    else:
        model_config.mm_cfg.vision_encoder.from_pretrained = "openai/clip-vit-large-patch14"
        model_config.data.image_token_len = 256

    ref_config = ref_config['text_config'].__dict__
    model_config["encoder_seq_length"] = ref_config["max_position_embeddings"]
    model_config["num_layers"] = ref_config["num_hidden_layers"]
    model_config["ffn_hidden_size"] = ref_config["intermediate_size"]
    model_config["hidden_size"] = ref_config["hidden_size"]
    model_config["num_attention_heads"] = ref_config["num_attention_heads"]
    model_config["num_query_groups"] = ref_config["num_key_value_heads"]
    model_config["layernorm_epsilon"] = ref_config["rms_norm_eps"]
    model_config["init_method_std"] = ref_config["initializer_range"]
    model_config["kv_channels"] = ref_config.get(
        "head_dim", model_config["hidden_size"] // model_config["num_attention_heads"]
    )
    if ref_config.get("rope_scaling") is not None:
        if ref_config["rope_scaling"]["type"] == "linear":
            model_config["seq_len_interpolation_factor"] = ref_config["rope_scaling"]["factor"]
        else:
            raise ValueError("Only linear rope scaling type is supported now")
    model_config["use_cpu_initialization"] = True

    return model_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to .nemo file or extracted folder",
    )
    parser.add_argument(
        "--hf_input_path",
        type=str,
        default=None,
        help="A HF model path, " "e.g. a folder containing https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main",
    )
    parser.add_argument(
        "--hf_output_path",
        type=str,
        default=None,
        help="Output HF model path, " "with the same format as above but user's own weights",
    )
    parser.add_argument("--skip_verification", action="store_true")

    args = parser.parse_args()
    return args


def convert(args):
    logging.info(f"Loading checkpoint from HF Llava: `{args.hf_input_path}`")
    hf_tokenizer = LlamaTokenizer.from_pretrained(args.hf_input_path)
    hf_model = LlavaForConditionalGeneration.from_pretrained(args.hf_input_path)
    logging.info("HF Model loading done.")

    nemo_config = OmegaConf.load(
        os.path.join(os.path.dirname(__file__), '../../examples/multimodal/multimodal_llm/neva/conf/llava_config.yaml')
    )
    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronNevaModel.restore_from(
        restore_path=args.input_name_or_path,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
    )

    rename_keys = create_rename_keys(model.cfg.num_layers)
    old_state_dict = model.state_dict()
    nemo_state_dict = reverse_adjust_tensor_shapes(model, hf_model, old_state_dict)
    hf_state_dict = rename_model_keys(model_state_dict=nemo_state_dict, rename_keys=rename_keys)

    hf_model.load_state_dict(hf_state_dict, strict=False)

    logging.info(f'=' * 100)
    if not args.skip_verification:
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
        model = model.cuda().eval()

        hf_outputs = hf_model(**batch_dict_cuda, output_hidden_states=True)
        ids = batch_dict_cuda['input_ids']

        id_tensors = [torch.unsqueeze(torch.LongTensor(id_list), dim=0) for id_list in ids.cpu()]

        masks_and_position_ids = [
            get_ltor_masks_and_position_ids(id_tensor, hf_tokenizer.eos_token, False, False, False)
            for id_tensor in id_tensors
        ]
        for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
            attn_mask, _, pos_ids = attn_mask_and_pos_ids

            outputs = model(
                tokens=tokens.cuda(), text_position_ids=pos_ids.cuda(), attention_mask=attn_mask.cuda(), labels=None
            )

        hf_next_token = hf_outputs.logits[0, -1].argmax()
        next_token = outputs.squeeze()[-1].argmax()

        logging.info(f"HF predicted next token is: '{hf_tokenizer._convert_id_to_token(int(hf_next_token))}'.")
        logging.info(f"NeMo predicted next token is: '{hf_tokenizer._convert_id_to_token(int(next_token))}'.")
        assert (
            hf_next_token == next_token
        ), f'prediction mismatch: {hf_tokenizer.decode(hf_next_token)} != {hf_tokenizer.decode(next_token)}'
        logging.info(f'=' * 100)

    hf_model.save_pretrained(args.hf_output_path)
    logging.info(f"Full HF model saved to {args.hf_output_path}")


if __name__ == '__main__':
    args = get_args()
    convert(args)
