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
Conversion script to convert NeMo starcoder2 checkpoints into HuggingFace checkpoint.
  Example to run this conversion script:
    python3 convert_nemo_starcoder2to_hf.py \
     --input_name_or_path <path_to_nemo_checkpoints_folder> \
     --output_path <path_to_output_hf_file>
"""

from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn
from lightning.pytorch.trainer.trainer import Trainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path", type=str, default=None, required=True, help="Path to NeMo Starcoder2 checkpoint"
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output HF checkpoint.")
    parser.add_argument(
        '--hf-model-name',
        type=str,
        default=None,
        required=True,
        help="Name of HF checkpoint. e.g. a folder containing https://huggingface.co/bigcode/starcoder2-15b/tree/main",
    )
    parser.add_argument("--precision", type=str, default="32", help="Model precision")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Load model in cpu only. Useful if the model cannot fit in GPU memory, but this option makes the conversion script significantly slower.",
    )
    args = parser.parse_args()
    return args


def load_config(hf_model_name, nemo_config):
    hf_config = AutoConfig.from_pretrained(hf_model_name)
    # SWA; nemo_config.window_size is list [left-bound, right-bound]
    # SC2 is pretrained with no SWA, long-context tuning model is using SWA
    hf_config.sliding_window = nemo_config.window_size[0] if 'window_size' in nemo_config else None
    hf_config.max_position_embeddings = nemo_config.encoder_seq_length
    hf_config.num_hidden_layers = nemo_config.num_layers
    hf_config.hidden_size = nemo_config.hidden_size
    hf_config.intermediate_size = nemo_config.ffn_hidden_size
    hf_config.num_attention_heads = nemo_config.num_attention_heads
    hf_config.initializer_range = nemo_config.init_method_std
    hf_config.rms_norm_eps = nemo_config.layernorm_epsilon
    hf_config.num_key_value_heads = nemo_config.num_query_groups
    if nemo_config.activation == 'gelu':
        hf_config.hidden_act = 'gelu_pytorch_tanh'
    else:
        logging.warning(f"Got unknown activation function {nemo_config.activation}")

    hf_config.rope_theta = nemo_config['rotary_base'] if 'rotary_base' in nemo_config else 10000
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
    model_config.name = "te_gpt"
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

    head_num = nemo_config.num_attention_heads
    if nemo_config.num_query_groups is None or nemo_config.num_query_groups == head_num:
        num_query_groups = head_num
    else:
        num_query_groups = nemo_config.num_query_groups
        assert head_num % num_query_groups == 0, 'head_num must be divisible by num_query_groups'

    hidden_size = model.cfg.hidden_size
    head_num = model.cfg.num_attention_heads
    num_layers = model.cfg.num_layers
    num_query_groups = model.cfg.get("num_query_groups", head_num)  # different num_query_groups for 70B

    head_size = model.cfg.get("kv_channels") or (hidden_size // head_num)  # equivalent to hf's head_dim
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    # Embedding
    embed_weight = model.state_dict()[f'model.embedding.word_embeddings.weight']
    embed_weights_base_name = f'model.embed_tokens.weight'
    state_dict[embed_weights_base_name] = param_to_weights(embed_weight)

    has_bias = nemo_config.bias

    for l in range(int(num_layers)):
        print(f"converting layer {l}")

        qkv_weights = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.weight']
        qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])
        if has_bias:
            qkv_bias = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.bias']
            qkv_bias = qkv_bias.reshape([qkv_total_dim, head_size])

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
            if has_bias:
                bias_name = f'model.layers.{l}.self_attn.{name}.bias'
                state_dict[bias_name] = param_to_weights(qkv_bias[slice].reshape(-1))

        # attention dense
        hf_o_weight_name = f'model.layers.{l}.self_attn.o_proj.weight'
        hf_o_bias_name = f'model.layers.{l}.self_attn.o_proj.bias'
        if mcore_gpt:
            o_weight_base_name = f'model.decoder.layers.{l}.self_attention.linear_proj.weight'
            o_bias_base_name = f'model.decoder.layers.{l}.self_attention.linear_proj.bias'
        else:
            o_weight_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.weight'
            o_bias_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.bias'
        state_dict[hf_o_weight_name] = param_to_weights(ckpt[o_weight_base_name])
        if has_bias:
            state_dict[hf_o_bias_name] = param_to_weights(ckpt[o_bias_base_name])

        # # MLP
        if mcore_gpt:
            mlp_down_base_name_weight = f'model.decoder.layers.{l}.mlp.linear_fc1.weight'
            mlp_down_base_name_bias = f'model.decoder.layers.{l}.mlp.linear_fc1.bias'
        else:
            raise Exception("not implemented")
        hf_mlp_cfc_weight_name = f'model.layers.{l}.mlp.c_fc.weight'
        hf_mlp_cfc_bias_name = f'model.layers.{l}.mlp.c_fc.bias'
        state_dict[hf_mlp_cfc_weight_name] = param_to_weights(ckpt[mlp_down_base_name_weight])
        if has_bias:
            state_dict[hf_mlp_cfc_bias_name] = param_to_weights(ckpt[mlp_down_base_name_bias])

        hf_mlp_up_base_name_weight = f'model.layers.{l}.mlp.c_proj.weight'
        hf_mlp_up_base_name_bias = f'model.layers.{l}.mlp.c_proj.bias'
        if mcore_gpt:
            mlp_up_base_name_weight = f'model.decoder.layers.{l}.mlp.linear_fc2.weight'
            mlp_up_base_name_bias = f'model.decoder.layers.{l}.mlp.linear_fc2.bias'
        else:
            raise Exception("not implemented")
        state_dict[hf_mlp_up_base_name_weight] = param_to_weights(ckpt[mlp_up_base_name_weight])
        if has_bias:
            state_dict[hf_mlp_up_base_name_bias] = param_to_weights(ckpt[mlp_up_base_name_bias])

        # LayerNorm
        hf_input_ln_weight_name = f'model.layers.{l}.input_layernorm.weight'
        hf_input_ln_bias_name = f'model.layers.{l}.input_layernorm.bias'
        if mcore_gpt:
            input_ln_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'
            input_ln_base_name_bias = f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_bias'
        else:
            input_ln_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.weight'
            input_ln_base_name_bias = f'model.language_model.encoder.layers.{l}.input_layernorm.bias'
        state_dict[hf_input_ln_weight_name] = param_to_weights(ckpt[input_ln_base_name])
        if has_bias:
            state_dict[hf_input_ln_bias_name] = param_to_weights(ckpt[input_ln_base_name_bias])

        hf_post_attn_ln_weight_name = f'model.layers.{l}.post_attention_layernorm.weight'
        hf_post_attn_ln_bias_name = f'model.layers.{l}.post_attention_layernorm.bias'
        if mcore_gpt:
            post_attn_ln_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'
            post_attn_ln_base_name_bias = f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_bias'
        else:
            post_attn_ln_base_name = f'model.language_model.encoder.layers.{l}.mlp.linear_fc1.layer_norm.weight'
            post_attn_ln_base_name_bias = f'model.language_model.encoder.layers.{l}.mlp.linear_fc1.layer_norm.bias'

        state_dict[hf_post_attn_ln_weight_name] = param_to_weights(ckpt[post_attn_ln_base_name])
        if has_bias:
            state_dict[hf_post_attn_ln_bias_name] = param_to_weights(ckpt[post_attn_ln_base_name_bias])

    hf_final_ln_weight_name = 'model.norm.weight'
    hf_final_ln_bias_name = 'model.norm.bias'
    if mcore_gpt:
        final_ln_base_name = 'model.decoder.final_layernorm.weight'
        final_ln_base_name_bias = 'model.decoder.final_layernorm.bias'
    else:
        final_ln_base_name = 'model.language_model.encoder.final_layernorm.weight'
        final_ln_base_name_bias = 'model.language_model.encoder.final_layernorm.bias'
    state_dict[hf_final_ln_weight_name] = param_to_weights(ckpt[final_ln_base_name])
    if has_bias:
        state_dict[hf_final_ln_bias_name] = param_to_weights(ckpt[final_ln_base_name_bias])

    hf_output_layer_weight_name = 'lm_head.weight'
    if mcore_gpt:
        output_layer_base_name = 'model.output_layer.weight'
    else:
        output_layer_base_name = 'model.language_model.output_layer.weight'
    state_dict[hf_output_layer_weight_name] = param_to_weights(ckpt[output_layer_base_name])
    return state_dict, nemo_config


if __name__ == '__main__':
    args = get_args()
    hf_state_dict, nemo_config = convert(args.input_name_or_path, args.precision, args.cpu_only)

    config = load_config(args.hf_model_name, nemo_config)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(hf_state_dict, strict=True)
    model.save_pretrained(args.output_path)
    hf_tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoder2-tokenizer')
    hf_tokenizer.save_pretrained(args.output_path)
    logging.info(f'HF checkpoint saved to: {args.output_path}')
