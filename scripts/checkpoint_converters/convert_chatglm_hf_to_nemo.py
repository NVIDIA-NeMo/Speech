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
Conversion script to convert Huggingface ChatGLM2/ChatGLM3 checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python convert_chatglm_hf_to_nemo.py \
     --input_name_or_path <path_to_hf_checkpoints_folder> \
     --output_path <path_to_output_nemo_file>
"""

import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from transformers import AutoModel, AutoTokenizer

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
        help="Path to Huggingface ChatGLM2/ChatGLM3 checkpoints",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), '../../examples/nlp/language_modeling/conf/megatron_chatglm_config.yaml'
        ),
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )

    parser.add_argument("--precision", type=str, default="16", help="Model precision")
    args = parser.parse_args()
    return args


def load_config(args, chatglm_config):
    nemo_config = OmegaConf.load(args.hparams_file).model
    nemo_config.encoder_seq_length = chatglm_config['seq_length']
    nemo_config.num_layers = int(chatglm_config['num_layers'])
    nemo_config.hidden_size = chatglm_config['hidden_size']
    nemo_config.ffn_hidden_size = chatglm_config['ffn_hidden_size']
    nemo_config.num_attention_heads = chatglm_config['num_attention_heads']
    nemo_config.max_position_embeddings = chatglm_config['seq_length']
    if 'multi_query_attention' in chatglm_config:
        if chatglm_config['multi_query_attention'] and 'multi_query_group_num' in chatglm_config:
            nemo_config.num_query_groups = chatglm_config['multi_query_group_num']
    nemo_config.attention_dropout = chatglm_config['attention_dropout']
    nemo_config.hidden_dropout = chatglm_config['hidden_dropout']
    nemo_config.layernorm_epsilon = chatglm_config['layernorm_epsilon']
    if 'apply_residual_connection_post_layernorm' in chatglm_config:
        if chatglm_config['apply_residual_connection_post_layernorm']:
            nemo_config.transformer_block_type = 'post_ln'
        else:
            nemo_config.transformer_block_type = 'pre_ln'
    nemo_config.use_cpu_initialization = True
    nemo_config.activation = 'fast-swiglu'
    nemo_config.tokenizer.model = chatglm_config['tokenizer_model']
    # base = 128
    # while chatglm_config['padded_vocab_size'] % base != 0:
    #     base //= 2
    # nemo_config.make_vocab_size_divisible_by = base
    nemo_config.override_vocab_size = chatglm_config['padded_vocab_size']

    return nemo_config


def convert(args):
    logging.info(f"loading checkpoint {args.input_name_or_path}")
    model = AutoModel.from_pretrained(args.input_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.input_name_or_path, trust_remote_code=True)
    hf_config = vars(model.config)
    hf_config['tokenizer_model'] = str(tokenizer.vocab_file)
    print(f"hf_config: {hf_config}")
    print("named parameters:")
    for name, param in model.named_parameters():
        print(f"hf - {name}", param.shape)

    nemo_config = load_config(args, hf_config)

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
                init_scale=nemo_config.get('native_amp_init_scale', 2 ** 32),
                growth_interval=nemo_config.get('native_amp_growth_interval', 1000),
                hysteresis=nemo_config.get('hysteresis', 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'

        if nemo_config.get('megatron_amp_O2', False):
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    nemo_config.precision = precision

    trainer = Trainer(plugins=plugins, accelerator='cpu', strategy=NLPDDPStrategy())

    hidden_size = hf_config["hidden_size"]
    head_num = hf_config["num_attention_heads"]
    head_size = hidden_size // head_num
    num_layers = hf_config["num_layers"]

    mcore_gpt = nemo_config.mcore_gpt

    assert mcore_gpt == nemo_config.get(
        'transformer_engine', False
    ), "mcore_gpt transformer_engine must be enabled (or disabled) together."

    param_to_weights = lambda param: param.float()

    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()

    embed_weight = model.state_dict()[f'transformer.embedding.word_embeddings.weight']
    if mcore_gpt:
        embed_weights_base_name = f'model.embedding.word_embeddings.weight'
    else:
        embed_weights_base_name = f'model.language_model.embedding.word_embeddings.weight'
    checkpoint['state_dict'][embed_weights_base_name] = param_to_weights(embed_weight)

    if nemo_config.num_query_groups is None or nemo_config.num_query_groups == head_num:
        num_query_groups = head_num
    else:
        num_query_groups = nemo_config.num_query_groups
        assert head_num % num_query_groups == 0, 'head_num must be divisible by num_query_groups'
    heads_per_group = head_num // num_query_groups

    if mcore_gpt:
        assert nemo_config.activation.startswith('fast-'), 'mcore only supports fast version of gated linear unit.'

    for l in range(int(num_layers)):
        print(f"converting layer {l}")

        hf_qkv_weights = model.state_dict()[f'transformer.encoder.layers.{l}.self_attention.query_key_value.weight']
        old_tensor_shape = hf_qkv_weights.size()
        new_q_tensor_shape = (head_num, head_size, old_tensor_shape[1])
        new_kv_tensor_shape = (num_query_groups, head_size, old_tensor_shape[1])
        q, k, v = hf_qkv_weights.split(
            [head_num * head_size, num_query_groups * head_size, num_query_groups * head_size], dim=0
        )
        q = q.view(*new_q_tensor_shape)
        k = k.view(*new_kv_tensor_shape)
        v = v.view(*new_kv_tensor_shape)
        qkv_weights = torch.empty((0, head_size, old_tensor_shape[1]))
        for i in range(num_query_groups):
            qkv_weights = torch.cat((qkv_weights, q[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
            qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
            qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
        qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

        hf_qkv_bias = model.state_dict()[f'transformer.encoder.layers.{l}.self_attention.query_key_value.bias']
        new_q_tensor_shape = (head_num, head_size)
        new_kv_tensor_shape = (num_query_groups, head_size)
        q, k, v = hf_qkv_bias.split(
            [head_num * head_size, num_query_groups * head_size, num_query_groups * head_size], dim=0
        )
        q = q.view(*new_q_tensor_shape)
        k = k.view(*new_kv_tensor_shape)
        v = v.view(*new_kv_tensor_shape)
        qkv_bias = torch.empty((0, head_size))
        for i in range(num_query_groups):
            qkv_bias = torch.cat((qkv_bias, q[i * heads_per_group : (i + 1) * heads_per_group, :]))
            qkv_bias = torch.cat((qkv_bias, k[i : i + 1, :]))
            qkv_bias = torch.cat((qkv_bias, v[i : i + 1, :]))
        qkv_bias = qkv_bias.reshape([head_size * (head_num + 2 * num_query_groups),])

        if mcore_gpt:
            qkv_weights_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.weight'
            qkv_bias_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.bias'
        else:
            qkv_weights_base_name = f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.weight'
            qkv_bias_base_name = f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.bias'
        checkpoint['state_dict'][qkv_weights_base_name] = param_to_weights(qkv_weights)
        checkpoint['state_dict'][qkv_bias_base_name] = param_to_weights(qkv_bias)

        # attention dense
        o_weight = model.state_dict()[f'transformer.encoder.layers.{l}.self_attention.dense.weight']
        if mcore_gpt:
            o_weight_base_name = f'model.decoder.layers.{l}.self_attention.linear_proj.weight'
        else:
            o_weight_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.weight'
        checkpoint['state_dict'][o_weight_base_name] = param_to_weights(o_weight)

        # MLP
        mlp_down_weight = model.state_dict()[f'transformer.encoder.layers.{l}.mlp.dense_h_to_4h.weight']
        if mcore_gpt:
            mlp_down_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.weight'
        else:
            mlp_down_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.weight'
        checkpoint['state_dict'][mlp_down_base_name] = param_to_weights(mlp_down_weight)

        mlp_up_weight = model.state_dict()[f'transformer.encoder.layers.{l}.mlp.dense_4h_to_h.weight']
        if mcore_gpt:
            mlp_up_base_name = f'model.decoder.layers.{l}.mlp.linear_fc2.weight'
        else:
            mlp_up_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.weight'
        checkpoint['state_dict'][mlp_up_base_name] = param_to_weights(mlp_up_weight)

        # LayerNorm
        input_ln_weight = model.state_dict()[f'transformer.encoder.layers.{l}.input_layernorm.weight']
        if mcore_gpt:
            input_ln_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'
        else:
            input_ln_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.weight'
        checkpoint['state_dict'][input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = model.state_dict()[f'transformer.encoder.layers.{l}.post_attention_layernorm.weight']
        if mcore_gpt:
            post_attn_ln_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'
        else:
            post_attn_ln_base_name = f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight'
        checkpoint['state_dict'][post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        print(f"done layer {l}")

    final_ln_weight = model.state_dict()[f'transformer.encoder.final_layernorm.weight']
    if mcore_gpt:
        final_ln_base_name = f'model.decoder.final_layernorm.weight'
    else:
        final_ln_base_name = f'model.language_model.encoder.final_layernorm.weight'
    checkpoint['state_dict'][final_ln_base_name] = param_to_weights(final_ln_weight)

    output_layer_weight = model.state_dict()[f'transformer.output_layer.weight']
    if mcore_gpt:
        output_layer_base_name = f'model.output_layer.weight'
    else:
        output_layer_base_name = f'model.language_model.output_layer.weight'
    checkpoint['state_dict'][output_layer_base_name] = param_to_weights(output_layer_weight)

    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config

    del model

    if nemo_config.get('megatron_amp_O2', False):
        keys = list(checkpoint['state_dict'].keys())
        for key in keys:
            checkpoint['state_dict'][key.replace('model.', 'model.module.', 1)] = checkpoint['state_dict'].pop(key)

    model = load_state_dict_helper(MegatronGPTModel, nemo_config, trainer, checkpoint['state_dict'])

    model._save_restore_connector = NLPSaveRestoreConnector()

    # cast to target precision and disable cpu init
    dtype = torch_dtype_from_precision(precision)
    model = model.to(dtype=dtype)
    model.cfg.use_cpu_initialization = False

    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
