# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

r"""
Conversion script to convert NeMo Mistral-7B checkpoints into HuggingFace checkpoint.
  Example to run this conversion script:
    python3 convert_mistral_7b_nemo_to_hf.py \
     --input_name_or_path <path_to_nemo_checkpoints_folder> \
     --output_path <path_to_output_hf_file>
"""

from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn
from pytorch_lightning.trainer.trainer import Trainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path", type=str, default=None, required=True, help="Path to NeMo Mistral-7B checkpoint"
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output HF checkpoint.")
    parser.add_argument('--hf_model_name', type=str, default="mistralai/Mistral-7B-v0.1", help="Name of HF checkpoint")
    parser.add_argument("--precision", type=str, default="32", help="Model precision")
    args = parser.parse_args()
    return args


def load_config(hf_model_name, nemo_config):
    hf_config = AutoConfig.from_pretrained(hf_model_name)
    # SWA; nemo_config.window_size is list [left-bound, right-bound]
    hf_config.sliding_window = nemo_config.window_size[0]
    hf_config.max_position_embeddings = nemo_config.encoder_seq_length
    hf_config.num_hidden_layers = nemo_config.num_layers
    hf_config.hidden_size = nemo_config.hidden_size
    hf_config.intermediate_size = nemo_config.ffn_hidden_size
    hf_config.num_attention_heads = nemo_config.num_attention_heads
    hf_config.max_position_embeddings = nemo_config.max_position_embeddings
    hf_config.initializer_range = nemo_config.init_method_std
    hf_config.rms_norm_eps = nemo_config.layernorm_epsilon
    hf_config.num_key_value_heads = nemo_config.num_query_groups
    if nemo_config.activation == 'fast-swiglu':
        hf_config.activation = 'silu'
    else:
        logging.warning(f"Got unknown activation function {nemo_config.activation}")

    hf_config.rope_theta = nemo_config['rotary_base']
    return hf_config


def convert(in_file, precision=None, cpu_only=True) -> None:
    """
    Convert NeMo checkpoint to HF checkpoint
    """

    logging.info(f'Loading NeMo checkpoint from: {in_file}')

    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    model_config = MegatronGPTModel.restore_from(in_file, trainer=dummy_trainer, return_config=True)
    model_config.tensor_model_parallel_size = 1
    model_config.pipeline_model_parallel_size = 1
    if cpu_only:
        map_location = torch.device('cpu')
        model_config.use_cpu_initialization = True
    else:
        map_location = None

    if cpu_only:
        logging.info("******** Loading model on CPU. This will take a significant amount of time.")
    model = MegatronGPTModel.restore_from(
        in_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
    )
    ckpt = model.state_dict()
    nemo_config = model.cfg

    mcore_gpt = nemo_config.mcore_gpt

    if precision is None:
        precision = model.cfg.precision
    if precision in [32, "32"]:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        logging.warning(f"Precision string {precision} is not recognized, falling back to fp32")
        dtype = torch.float32  # fallback
    param_to_weights = lambda param: param.to(dtype)

    state_dict = OrderedDict()

    hf_embed_weight_name = f'model.embed_tokens.weight'
    if mcore_gpt:
        embed_weights_base_name = f'model.embedding.word_embeddings.weight'
    else:
        embed_weights_base_name = f'model.language_model.embedding.word_embeddings.weight'
    state_dict[hf_embed_weight_name] = param_to_weights(ckpt[embed_weights_base_name])

    head_num = model.cfg.num_attention_heads
    if nemo_config.num_query_groups is None or nemo_config.num_query_groups == head_num:
        num_query_groups = head_num
    else:
        num_query_groups = nemo_config.num_query_groups
        assert head_num % num_query_groups == 0, 'head_num must be divisible by num_query_groups'
    if mcore_gpt:
        assert nemo_config.activation.startswith('fast-'), 'mcore only supports fast version of gated linear unit.'

    hidden_size = model.cfg.hidden_size
    num_layers = model.cfg.num_layers
    num_query_groups = model.cfg.get("num_query_groups", head_num)  # different num_query_groups for 70B

    head_size = hidden_size // head_num
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    # Embedding
    embed_weight = model.state_dict()[f'model.embedding.word_embeddings.weight']
    embed_weights_base_name = f'model.embed_tokens.weight'
    state_dict[embed_weights_base_name] = param_to_weights(embed_weight)

    for l in range(int(num_layers)):
        print(f"converting layer {l}")

        qkv_weights = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.weight']
        qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

        for name, slice in [('q_proj', q_slice), ('k_proj', k_slice), ('v_proj', v_slice)]:
            weight_name = f'model.layers.{l}.self_attn.{name}.weight'
            state_dict[weight_name] = param_to_weights(qkv_weights[slice].reshape(-1, hidden_size))

        # attention dense
        hf_o_weight_name = f'model.layers.{l}.self_attn.o_proj.weight'
        if mcore_gpt:
            o_weight_base_name = f'model.decoder.layers.{l}.self_attention.linear_proj.weight'
        else:
            o_weight_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.weight'
        state_dict[hf_o_weight_name] = param_to_weights(ckpt[o_weight_base_name])

        # # MLP
        if mcore_gpt:
            mlp_down_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.weight'
        else:
            raise Exception("not implemented")
        gate_proj_weight, up_proj_weight = torch.chunk(ckpt[mlp_down_base_name], 2, dim=0)
        hf_gate_proj_name = f'model.layers.{l}.mlp.gate_proj.weight'
        hf_up_proj_name = f'model.layers.{l}.mlp.up_proj.weight'
        state_dict[hf_gate_proj_name] = param_to_weights(gate_proj_weight)
        state_dict[hf_up_proj_name] = param_to_weights(up_proj_weight)

        hf_mlp_up_weight_name = f'model.layers.{l}.mlp.down_proj.weight'
        if mcore_gpt:
            mlp_up_base_name = f'model.decoder.layers.{l}.mlp.linear_fc2.weight'
        else:
            raise Exception("not implemented")
        state_dict[hf_mlp_up_weight_name] = param_to_weights(ckpt[mlp_up_base_name])

        # LayerNorm
        hf_input_ln_weight_name = f'model.layers.{l}.input_layernorm.weight'
        if mcore_gpt:
            input_ln_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'
        else:
            input_ln_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.weight'
        state_dict[hf_input_ln_weight_name] = param_to_weights(ckpt[input_ln_base_name])

        hf_post_attn_ln_weight_name = f'model.layers.{l}.post_attention_layernorm.weight'
        if mcore_gpt:
            post_attn_ln_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'
        else:
            post_attn_ln_base_name = f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight'
        state_dict[hf_post_attn_ln_weight_name] = param_to_weights(ckpt[post_attn_ln_base_name])

    hf_final_ln_weight_name = 'model.norm.weight'
    if mcore_gpt:
        final_ln_base_name = 'model.decoder.final_layernorm.weight'
    else:
        final_ln_base_name = 'model.language_model.encoder.final_layernorm.weight'
    state_dict[hf_final_ln_weight_name] = param_to_weights(ckpt[final_ln_base_name])

    hf_output_layer_weight_name = 'lm_head.weight'
    if mcore_gpt:
        output_layer_base_name = 'model.output_layer.weight'
    else:
        output_layer_base_name = 'model.language_model.output_layer.weight'
    state_dict[hf_output_layer_weight_name] = param_to_weights(ckpt[output_layer_base_name])
    return state_dict, nemo_config


if __name__ == '__main__':
    args = get_args()
    hf_state_dict, nemo_config = convert(args.input_name_or_path, args.precision)

    config = load_config(args.hf_model_name, nemo_config)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(hf_state_dict)
    model.save_pretrained(args.output_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    hf_tokenizer.save_pretrained(args.output_path)
    logging.info(f'HF checkpoint saved to: {args.output_path}')
