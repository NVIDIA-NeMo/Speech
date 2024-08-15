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
Conversion script to convert Huggingface LLaMA checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python convert_llama_hf_to_nemo.py \
     --input_name_or_path <path_to_hf_checkpoints_folder> \
     --output_path <path_to_output_nemo_file>
     --precision bf16 
     --apply_rope_scaling True
"""

import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.collections.nlp.parts.utils_funcs import load_state_dict_helper, torch_dtype_from_precision
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface LLaMA checkpoints",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output to dict dir")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), '../../examples/nlp/language_modeling/conf/megatron_llama_config.yaml'
        ),
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument(
        "--apply_rope_scaling",
        type=bool,
        default=True,
        required=False,
        help="Apply scaling for RoPE frequencies",
    )
    parser.add_argument("--precision", type=str, default="16", help="Model precision")
    args = parser.parse_args()
    return args


def load_config(args, llama_config):
    nemo_config = OmegaConf.load(args.hparams_file).model

    if llama_config.get('rope_theta', None):
        nemo_config['rotary_base'] = llama_config['rope_theta']
    nemo_config.encoder_seq_length = llama_config['max_position_embeddings']
    nemo_config.num_layers = int(llama_config['num_hidden_layers'])
    nemo_config.hidden_size = llama_config['hidden_size']
    nemo_config.ffn_hidden_size = llama_config['intermediate_size']
    nemo_config.num_attention_heads = llama_config['num_attention_heads']
    nemo_config.max_position_embeddings = llama_config['max_position_embeddings']
    nemo_config.init_method_std = llama_config['initializer_range']
    nemo_config.layernorm_epsilon = llama_config['rms_norm_eps']
    if 'num_key_value_heads' in llama_config:
        nemo_config.num_query_groups = llama_config['num_key_value_heads']
    nemo_config.use_cpu_initialization = True
    nemo_config.activation = 'fast-swiglu'
    nemo_config.megatron_amp_O2 = True

    # Tokenizer config
    if 'tokenizer_model' in llama_config:
        nemo_config.tokenizer.model = llama_config['tokenizer_model']
    else:
        # Llama3 uses converted TikToken Tokenizer
        tokenizer_dict = {
            'library': 'huggingface',
            'type': args.input_name_or_path,
            'use_fast': True,
        }
        nemo_config.tokenizer = tokenizer_dict

    if llama_config['rope_scaling'] is not None:
        if llama_config['rope_scaling']['type'] == 'linear':
            nemo_config['seq_len_interpolation_factor'] = llama_config['rope_scaling']['factor']
        else:
            raise ValueError("Only linear rope scaling type is supported now")
    if llama_config['rope_theta'] is not None:
        nemo_config['rotary_base'] = llama_config['rope_theta']

    base = 128
    while llama_config['vocab_size'] % base != 0:
        base //= 2
    nemo_config.make_vocab_size_divisible_by = base

    return nemo_config


def convert(args):
    logging.info(f"loading checkpoint {args.input_name_or_path}")
    import torch

    model = LlamaForCausalLM.from_pretrained(
        args.input_name_or_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    hf_config = vars(model.config)
    if os.path.exists(f'{args.input_name_or_path}/tokenizer.model'):
        tokenizer = LlamaTokenizer.from_pretrained(args.input_name_or_path)
        hf_config['tokenizer_model'] = str(tokenizer.vocab_file)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.input_name_or_path)

    print("named parameters:")
    for name, param in model.named_parameters():
        print(f"- {name}")

    nemo_config = load_config(args, hf_config)
    nemo_config.scale_positional_embedding = args.apply_rope_scaling

    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))
    elif args.precision in ["bf16", "bf16-mixed"]:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = args.precision
        else:
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            precision = args.precision[2:]  # prune bf in string
    else:
        precision = args.precision

    plugins = []
    if precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=nemo_config.get('native_amp_init_scale', 2**32),
                growth_interval=nemo_config.get('native_amp_growth_interval', 1000),
                hysteresis=nemo_config.get('hysteresis', 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'

        if nemo_config.get('megatron_amp_O2', False):
            print('HALF PRECISION')
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    nemo_config.precision = precision
    print(f"nemo_config: {nemo_config}")

    # Remove precision arg, since with PTL >= 2.1 both precision and precision plugin cannot exist together.
    trainer = Trainer(plugins=plugins, accelerator='cpu', strategy=NLPDDPStrategy())

    hidden_size = hf_config["hidden_size"]
    head_num = hf_config["num_attention_heads"]
    head_size = hidden_size // head_num
    num_layers = hf_config["num_hidden_layers"]

    mcore_gpt = nemo_config.mcore_gpt

    assert mcore_gpt == nemo_config.get(
        'transformer_engine', False
    ), "mcore_gpt transformer_engine must be enabled (or disabled) together."

    param_to_weights = lambda param: param.float()

    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()

    embed_weight = model.state_dict()[f'model.embed_tokens.weight']
    if mcore_gpt:
        embed_weights_base_name = f'model.embedding.word_embeddings.weight'
    else:
        embed_weights_base_name = f'model.language_model.embedding.word_embeddings.weight'
    checkpoint['state_dict'][embed_weights_base_name] = param_to_weights(embed_weight)

    # in hf, this is defined as register_buffer(..., persistent=False) so it won't be in the state dict
    if f'model.layers.0.self_attn.rotary_emb.inv_freq' in model.state_dict():
        rotary_embed_weight = model.state_dict()[f'model.layers.0.self_attn.rotary_emb.inv_freq']
        if mcore_gpt:
            rotary_embed_weight_base_name = f'model.rotary_pos_emb.inv_freq'
        else:
            rotary_embed_weight_base_name = f'model.language_model.rotary_pos_emb.inv_freq'
        checkpoint['state_dict'][rotary_embed_weight_base_name] = param_to_weights(rotary_embed_weight)

    if nemo_config.num_query_groups is None or nemo_config.num_query_groups == head_num:
        num_query_groups = head_num
    else:
        num_query_groups = nemo_config.num_query_groups
        assert head_num % num_query_groups == 0, 'head_num must be divisible by num_query_groups'
    if mcore_gpt:
        assert nemo_config.activation.startswith('fast-'), 'mcore only supports fast version of gated linear unit.'

    for l in range(int(num_layers)):
        print(f"converting layer {l}")
        old_tensor_shape = model.state_dict()[f'model.layers.{l}.self_attn.q_proj.weight'].size()
        new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
        new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]
        q = model.state_dict()[f'model.layers.{l}.self_attn.q_proj.weight'].view(*new_q_tensor_shape)
        k = model.state_dict()[f'model.layers.{l}.self_attn.k_proj.weight'].view(*new_kv_tensor_shape)
        v = model.state_dict()[f'model.layers.{l}.self_attn.v_proj.weight'].view(*new_kv_tensor_shape)
        qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])
        heads_per_group = head_num // num_query_groups
        for i in range(num_query_groups):
            qkv_weights = torch.cat((qkv_weights, q[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
            qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
            qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
        qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])
        if mcore_gpt:
            qkv_weights_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.weight'
        else:
            qkv_weights_base_name = f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.weight'
        checkpoint['state_dict'][qkv_weights_base_name] = param_to_weights(qkv_weights)

        # attention dense
        o_weight = model.state_dict()[f'model.layers.{l}.self_attn.o_proj.weight']
        if mcore_gpt:
            o_weight_base_name = f'model.decoder.layers.{l}.self_attention.linear_proj.weight'
        else:
            o_weight_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.weight'
        checkpoint['state_dict'][o_weight_base_name] = param_to_weights(o_weight)

        # MLP
        mlp_down_weight = model.state_dict()[f'model.layers.{l}.mlp.gate_proj.weight']
        mlp_gate_weight = model.state_dict()[f'model.layers.{l}.mlp.up_proj.weight']
        if mcore_gpt:
            mlp_down_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.weight'
        else:
            mlp_down_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.weight'
        mlp_down_weight = torch.cat((mlp_down_weight, mlp_gate_weight), axis=0)
        checkpoint['state_dict'][mlp_down_base_name] = param_to_weights(mlp_down_weight)

        mlp_up_weight = model.state_dict()[f'model.layers.{l}.mlp.down_proj.weight']
        if mcore_gpt:
            mlp_up_base_name = f'model.decoder.layers.{l}.mlp.linear_fc2.weight'
        else:
            mlp_up_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.weight'
        checkpoint['state_dict'][mlp_up_base_name] = param_to_weights(mlp_up_weight)

        # LayerNorm
        input_ln_weight = model.state_dict()[f'model.layers.{l}.input_layernorm.weight']
        if mcore_gpt:
            input_ln_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'
        else:
            input_ln_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.weight'
        checkpoint['state_dict'][input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = model.state_dict()[f'model.layers.{l}.post_attention_layernorm.weight']
        if mcore_gpt:
            post_attn_ln_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'
        else:
            post_attn_ln_base_name = f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight'
        checkpoint['state_dict'][post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        print(f"done layer {l}")

    final_ln_weight = model.state_dict()[f'model.norm.weight']
    if mcore_gpt:
        final_ln_base_name = f'model.decoder.final_layernorm.weight'
    else:
        final_ln_base_name = f'model.language_model.encoder.final_layernorm.weight'
    checkpoint['state_dict'][final_ln_base_name] = param_to_weights(final_ln_weight)

    output_layer_weight = model.state_dict()[f'lm_head.weight']
    if mcore_gpt:
        output_layer_base_name = f'model.output_layer.weight'
    else:
        output_layer_base_name = f'model.language_model.output_layer.weight'
    checkpoint['state_dict'][output_layer_base_name] = param_to_weights(output_layer_weight)

    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config

    del model
    import gc

    gc.collect()

    if nemo_config.get('megatron_amp_O2', False):
        keys = list(checkpoint['state_dict'].keys())
        print('convert to O2')
        for key in keys:
            checkpoint['state_dict'][key.replace('model.', 'model.module.', 1)] = checkpoint['state_dict'].pop(key)

    # os.mkdir(args.output_path, exist_ok=True)
    for key in checkpoint['state_dict']:
        print(f'Saving {key} in {checkpoint["state_dict"][key].dtype}..')
        save_location = f'{args.output_path}/{key[13:]}.pt'
        torch.save(checkpoint['state_dict'][key], save_location)


if __name__ == '__main__':
    args = get_args()
    convert(args)
