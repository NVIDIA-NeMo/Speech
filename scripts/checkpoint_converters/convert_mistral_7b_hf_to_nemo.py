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
Conversion script to convert HuggingFace Mistral-7B checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python convert_mistral_7b_hf_to_nemo.py \
     --input_name_or_path <path_to_mistral_checkpoints_folder> \
     --output_path <path_to_output_nemo_file> 
"""


import json
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn
from omegaconf import OmegaConf
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.trainer.trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface Mistral-7b checkpoints",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--precision", type=str, default="bf16", help="Model precision")
    args = parser.parse_args()
    return args


def load_model(cls, checkpoint, strict, **kwargs):
    try:
        if 'cfg' in kwargs:
            model = ptl_load_state(cls, checkpoint, strict=strict, **kwargs)
        else:
            model = cls(cfg=checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY], **kwargs)
            for name, module in model.named_parameters():
                if name in checkpoint['state_dict']:
                    module.data = checkpoint['state_dict'][name]
                    checkpoint['state_dict'].pop(name)
                else:
                    print(f"Unexpected key: {name} not in checkpoint but in model.")

            for name, buffer in model.named_buffers():
                if name in checkpoint['state_dict']:
                    buffer.data = checkpoint['state_dict'][name]
                    checkpoint['state_dict'].pop(name)

            if len(checkpoint['state_dict'].keys()) != 0:
                raise RuntimeError(
                    f"Additional keys: {checkpoint['state_dict'].keys()} in checkpoint but not in model."
                )

            # register the artifacts
            cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
            if cfg.tokenizer.model is not None:
                model.register_artifact("tokenizer.tokenizer_model", cfg.tokenizer.model)
            if cfg.tokenizer.vocab_file is not None:
                model.register_artifact("tokenizer.vocab_file", cfg.tokenizer.vocab_file)
            if cfg.tokenizer.merge_file is not None:
                model.register_artifact("tokenizer.merge_file", cfg.tokenizer.merge_file)
    finally:
        cls._set_model_restore_state(is_being_restored=False)
    return model


def load_config(mistral_config, tokenizer, config_path):
    nemo_config = OmegaConf.load(
        os.path.join(os.path.dirname(__file__), '../../examples/nlp/language_modeling/conf/megatron_llama_config.yaml')
    ).model
    # akoumparouli: verify this.
    if mistral_config.get('sliding_window', None) is not None:
        nemo_config.encoder_seq_length = mistral_config['sliding_window']
    else:
        nemo_config.encoder_seq_length = mistral_config['max_position_embeddings']
    nemo_config.num_layers = int(mistral_config['num_hidden_layers'])
    nemo_config.hidden_size = mistral_config['hidden_size']
    nemo_config.ffn_hidden_size = mistral_config['intermediate_size']
    nemo_config.num_attention_heads = mistral_config['num_attention_heads']
    nemo_config.max_position_embeddings = mistral_config['max_position_embeddings']
    if mistral_config.get('sliding_window', None) is not None:
        nemo_config.window_size = [mistral_config['sliding_window'], 0]
    nemo_config.init_method_std = mistral_config['initializer_range']
    # RMSNorm's epsilon.
    nemo_config.layernorm_epsilon = mistral_config['rms_norm_eps']
    nemo_config.normalization = 'rmsnorm'

    if 'num_key_value_heads' in mistral_config:
        nemo_config.num_query_groups = mistral_config['num_key_value_heads']
    nemo_config.use_cpu_initialization = True
    # Mistral uses SiLU, but it is the same as swish with beta = 1.
    nemo_config.activation = 'fast-swiglu'

    # Tokenizer config
    if hasattr(tokenizer, 'vocab_file'):
        nemo_config.tokenizer.model = tokenizer.vocab_file
    else:
        # Load tekken.json, extract the 'vocab' field & write it to file.
        vocab_path = os.path.join(config_path, 'tekken.json')
        assert os.path.exists(vocab_path), f"Expected {vocab_path} to exist"
        with open(vocab_path, 'rt') as fp:
            tok_vocab = json.load(fp)
        vocab_output_path = '/tmp/tekken.json'
        if os.path.exists(vocab_output_path):
            os.remove(vocab_output_path)
        with open(vocab_output_path, 'wt') as fp:
            json.dump(tok_vocab['vocab'], fp)
        assert os.path.exists(vocab_output_path), f"Expected {vocab_output_path} to exist"
        assert os.path.getsize(vocab_output_path) > 0, f"Expected {vocab_output_path} to be non-empty"

        tokenizer_dict = {
            'library': 'tiktoken',
            'type': 'tiktoken',
            'vocab_file': vocab_output_path,
            'model': None,
            'merge_file': None,
            'delimiter': None,
            'sentencepiece_legacy': False,
        }
        nemo_config.tokenizer = tokenizer_dict

    # TODO(@akoumparouli): rope_scaling.
    nemo_config['rotary_base'] = mistral_config['rope_theta']

    base = 128
    while mistral_config['vocab_size'] % base != 0:
        base //= 2
    nemo_config.make_vocab_size_divisible_by = base

    return nemo_config


def load_mistral_ckpt(in_dir):
    params_file = os.path.join(in_dir, 'config.json')
    assert os.path.exists(params_file)
    with open(params_file, 'r') as fp:
        model_args = json.load(fp)

    model = AutoModelForCausalLM.from_pretrained(in_dir)
    ckpt = model.state_dict()

    tokenizer = AutoTokenizer.from_pretrained(in_dir)
    assert tokenizer.vocab_size == model_args['vocab_size']
    return model_args, ckpt, tokenizer


def convert(args):
    logging.info(f"loading checkpoint {args.input_name_or_path}")

    model_args, ckpt, tokenizer = load_mistral_ckpt(args.input_name_or_path)
    nemo_config = load_config(model_args, tokenizer, args.input_name_or_path)
    logging.info(f"loaded checkpoint {args.input_name_or_path}")

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
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    if precision == 32:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32  # fallback

    nemo_config.precision = precision
    logging.info(f"nemo_config: {nemo_config}")

    trainer = Trainer(plugins=plugins, accelerator='cpu', strategy=NLPDDPStrategy())

    hidden_size = nemo_config.hidden_size
    head_num = nemo_config.num_attention_heads
    head_size = model_args.get('head_dim', hidden_size // head_num)
    # Set this explictly because 2407 does not use hidden_size // num_attention_heads
    nemo_config.kv_channels = head_size
    num_layers = nemo_config.num_layers

    mcore_gpt = nemo_config.mcore_gpt

    assert mcore_gpt == nemo_config.get(
        'transformer_engine', False
    ), "mcore_gpt transformer_engine must be enabled (or disabled) together."

    param_to_weights = lambda param: param.float()

    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()

    embed_weight = ckpt[f'model.embed_tokens.weight']
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
    if mcore_gpt:
        assert nemo_config.activation.startswith('fast-'), 'mcore only supports fast version of gated linear unit.'

    for l in range(int(num_layers)):
        print(f"converting layer {l}")
        old_tensor_shape = ckpt[f'model.layers.{l}.self_attn.q_proj.weight'].size()
        new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
        new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

        q = ckpt[f'model.layers.{l}.self_attn.q_proj.weight'].view(*new_q_tensor_shape)
        k = ckpt[f'model.layers.{l}.self_attn.k_proj.weight'].view(*new_kv_tensor_shape)
        v = ckpt[f'model.layers.{l}.self_attn.v_proj.weight'].view(*new_kv_tensor_shape)

        # Note: we assume wq & wk have been appropriately transposed to work with
        # NeMo/Megatron's rotary embedding. The reference checkpoint/implementation
        # will not work OotB without transposing wq/wk matrices.
        heads_per_group = head_num // num_query_groups
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
        if mcore_gpt:
            qkv_weights_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.weight'
        else:
            qkv_weights_base_name = f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.weight'
        checkpoint['state_dict'][qkv_weights_base_name] = param_to_weights(qkv_weights)

        # attention dense
        o_weight = ckpt[f'model.layers.{l}.self_attn.o_proj.weight']
        if mcore_gpt:
            o_weight_base_name = f'model.decoder.layers.{l}.self_attention.linear_proj.weight'
        else:
            o_weight_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.weight'
        checkpoint['state_dict'][o_weight_base_name] = param_to_weights(o_weight)

        # MLP
        mlp_down_weight = ckpt[f'model.layers.{l}.mlp.gate_proj.weight']
        mlp_gate_weight = ckpt[f'model.layers.{l}.mlp.up_proj.weight']
        if mcore_gpt:
            mlp_down_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.weight'
        else:
            mlp_down_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.weight'
        mlp_down_weight = torch.cat((mlp_down_weight, mlp_gate_weight), axis=0)
        checkpoint['state_dict'][mlp_down_base_name] = param_to_weights(mlp_down_weight)

        mlp_up_weight = ckpt[f'model.layers.{l}.mlp.down_proj.weight']
        if mcore_gpt:
            mlp_up_base_name = f'model.decoder.layers.{l}.mlp.linear_fc2.weight'
        else:
            mlp_up_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.weight'
        checkpoint['state_dict'][mlp_up_base_name] = param_to_weights(mlp_up_weight)

        # LayerNorm
        input_ln_weight = ckpt[f'model.layers.{l}.input_layernorm.weight']
        if mcore_gpt:
            input_ln_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'
        else:
            input_ln_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.weight'
        checkpoint['state_dict'][input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = ckpt[f'model.layers.{l}.post_attention_layernorm.weight']
        if mcore_gpt:
            post_attn_ln_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'
        else:
            post_attn_ln_base_name = f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight'
        checkpoint['state_dict'][post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        print(f"done layer {l}")

    final_ln_weight = ckpt[f'model.norm.weight']
    if mcore_gpt:
        final_ln_base_name = f'model.decoder.final_layernorm.weight'
    else:
        final_ln_base_name = f'model.language_model.encoder.final_layernorm.weight'
    checkpoint['state_dict'][final_ln_base_name] = param_to_weights(final_ln_weight)

    output_layer_weight = ckpt[f'lm_head.weight']
    if mcore_gpt:
        output_layer_base_name = f'model.output_layer.weight'
    else:
        output_layer_base_name = f'model.language_model.output_layer.weight'
    checkpoint['state_dict'][output_layer_base_name] = param_to_weights(output_layer_weight)

    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config
    del ckpt

    if nemo_config.get('megatron_amp_O2', False):
        keys = list(checkpoint['state_dict'].keys())
        for key in keys:
            checkpoint['state_dict'][key.replace('model.', 'model.module.', 1)] = checkpoint['state_dict'].pop(key)

    model = load_model(MegatronGPTModel, checkpoint, strict=False, trainer=trainer)

    model._save_restore_connector = NLPSaveRestoreConnector()

    # cast to target precision and disable cpu init
    model = model.to(dtype=dtype)
    model.cfg.use_cpu_initialization = False

    if getattr(tokenizer, 'chat_template', None) is not None:
        import hashlib

        assert (
            hashlib.md5(tokenizer.chat_template.encode('utf-8')).hexdigest() == "0b629f783db54e02509999196956ff40"
        ), "Got unkown chat template"
        from omegaconf import OmegaConf, open_dict

        with open_dict(model.cfg):
            model.cfg.tokenizer.chat_template = OmegaConf.create(
                {
                    'prefix': "{_bos_}",
                    'roles': {'User': "[INST] {_content_} [/INST]", 'Assistant': "{_content_}{_eos_}"},
                }
            )

    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
