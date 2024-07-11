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
Script to convert LLaVA checkpoints into .nemo file.
This script depend on llava github project: 
https://github.com/haotian-liu/LLaVA/tree/main

If you want to convert huggingface LLaVA checkpoint such as llava-hf/llava-1.5-7b-hf,
you should check `NeMo/scripts/checkpoint_converters/convert_llava_hf_to_nemo.py`

Example to run this conversion script:
  python convert_hf_llava_to_neva.py \
   --in-file <path_to_hf_checkpoints_folder> \
   --out-file <path_to_output_nemo_file> \
   --tokenizer-model <path_to_sp_tokenizer_model> \
   --conv-template llama_2 # nvgpt, llama_2, v1, llama_3 (vicuna)
"""

import json
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from llava import LlavaLlamaForCausalLM
from omegaconf import OmegaConf
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.trainer.trainer import Trainer
from safetensors import safe_open
from transformers import LlamaTokenizer

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
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
        "--in-file",
        type=str,
        default=None,
        required=True,
        help="Path to LLaVA checkpoints",
    )
    parser.add_argument("--out-file", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument(
        "--conv-template",
        type=str,
        default="llama_2",
        required=False,
        help="Conversation template: nvgpt, llama_2, v1 (vicuna)",
    )
    parser.add_argument(
        "--tokenizer-model", type=str, default=None, required=False, help="Path to sentencepiece tokenizer model."
    )
    parser.add_argument("--precision", type=str, default="32", help="Model precision")
    parser.add_argument("--config-file", type=str, default="llava_config.yaml")
    parser.add_argument(
        "--mm-projector-ckpt-dir",
        type=str,
        default=None,
        help="Path to multimodal projector checkpoint directory \
                        This will overlap the projector weights in in-file hf checkpoint",
    )
    parser.add_argument("--mm-vision-tower", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
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
                    if module.data.shape != checkpoint['state_dict'][name].shape:
                        print(
                            f"WARNING: Auto padding {name} from {checkpoint['state_dict'][name].shape} to {module.data.shape}"
                        )
                        module.data[
                            : checkpoint['state_dict'][name].size(0), : checkpoint['state_dict'][name].size(1)
                        ] = checkpoint['state_dict'][name]
                    else:
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


def load_config(args, llava_config):
    nemo_config = OmegaConf.load(os.path.join(os.path.dirname(__file__), 'conf', args.config_file)).model
    nemo_config.mm_cfg.mm_mlp_adapter_type = llava_config.get('mm_projector_type', 'linear')

    mm_vision_tower = llava_config.get('mm_vision_tower', 'openai/clip-vit-large-patch14')

    if args.mm_vision_tower is not None:
        mm_vision_tower = args.mm_vision_tower

    nemo_config.mm_cfg.vision_encoder.from_pretrained = mm_vision_tower
    if args.mm_vision_tower is not None:
        config_file = os.path.join(args.mm_vision_tower, "config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                vision_model_config = json.load(f)
                nemo_config.mm_cfg.vision_encoder["model_type"] = vision_model_config.get("model_type", 'clip')
                crop_size = vision_model_config.get("image_size", 224)
                nemo_config.mm_cfg.vision_encoder.crop_size = [crop_size, crop_size]
    else:
        if '336' in mm_vision_tower:
            nemo_config.data.image_token_len = 576
            nemo_config.mm_cfg.vision_encoder.crop_size = [336, 336]
        else:
            nemo_config.data.image_token_len = 256
            nemo_config.mm_cfg.vision_encoder.crop_size = [224, 224]
        nemo_config.mm_cfg.vision_encoder.patch_dim = 14

    nemo_config.encoder_seq_length = llava_config['max_position_embeddings']
    nemo_config.num_layers = int(llava_config['num_hidden_layers'])
    nemo_config.hidden_size = llava_config['hidden_size']
    nemo_config.ffn_hidden_size = llava_config['intermediate_size']
    nemo_config.num_attention_heads = llava_config['num_attention_heads']
    nemo_config.max_position_embeddings = llava_config['max_position_embeddings']
    nemo_config.init_method_std = llava_config['initializer_range']
    nemo_config.layernorm_epsilon = llava_config['rms_norm_eps']
    if 'num_key_value_heads' in llava_config:
        nemo_config.num_query_groups = llava_config['num_key_value_heads']
    nemo_config.use_cpu_initialization = True
    nemo_config.activation = 'fast-swiglu'
    nemo_config.data.conv_template = args.conv_template
    nemo_config.data.image_aspect_ratio = llava_config.get('image_aspect_ratio', 'square')
    if args.model_type is None:
        nemo_config.mm_cfg.model_type = args.conv_template
    else:
        nemo_config.mm_cfg.model_type = args.model_type
    if args.tokenizer_model is None:
        if 'tokenizer_model' in llava_config:
            nemo_config.tokenizer.library = 'sentencepiece'
            nemo_config.tokenizer.model = llava_config['tokenizer_model']
        else:
            # Llama3 uses converted TikToken Tokenizer
            tokenizer_dict = {'library': 'huggingface', 'type': args.in_file, 'use_fast': True, 'model': None}
            nemo_config.tokenizer.update(tokenizer_dict)
    else:
        # if tokenizer_model is directory
        if os.path.isdir(args.tokenizer_model):
            tokenizer_dict = {'library': 'huggingface', 'type': args.tokenizer_model, 'use_fast': True, 'model': None}
            nemo_config.tokenizer.update(tokenizer_dict)
        else:
            nemo_config.tokenizer.library = 'sentencepiece'
            nemo_config.tokenizer.model = args.tokenizer_model
    if llava_config['rope_scaling'] is not None:
        if llava_config['rope_scaling']['type'] == 'linear':
            nemo_config['seq_len_interpolation_factor'] = llava_config['rope_scaling']['factor']
        else:
            raise ValueError("Only linear rope scaling type is supported now")
    if llava_config.get('rope_theta', None):
        nemo_config['rotary_base'] = llava_config['rope_theta']

    base = 128
    while llava_config['vocab_size'] % base != 0:
        base //= 2
    nemo_config.make_vocab_size_divisible_by = base

    return nemo_config


def convert(args):
    logging.info(f"loading checkpoint {args.in_file}")
    model = LlavaLlamaForCausalLM.from_pretrained(args.in_file)
    hf_config = vars(model.config)
    if os.path.exists(f'{args.in_file}/tokenizer.model'):
        tokenizer = LlamaTokenizer.from_pretrained(args.in_file)
        hf_config['tokenizer_model'] = str(tokenizer.vocab_file)

    for name, param in model.named_parameters():
        print(f"- {name}")

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
    print(f"nemo_config: {nemo_config}")

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

    # Multimodal projection
    if mcore_gpt:
        mm_projection_layer_base_name = (
            f'model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector'
        )
    else:
        mm_projection_layer_base_name = (
            f'model.language_model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector'
        )
    for key in model.state_dict():
        if 'mm_projector' in key:
            mm_projection_layer_suffix = key.split('mm_projector')[1]
            checkpoint['state_dict'][f'{mm_projection_layer_base_name}{mm_projection_layer_suffix}'] = (
                param_to_weights(model.state_dict()[key])
            )

    # Replace or add the projection weights
    proj_ckpt = None
    if args.mm_projector_ckpt_dir is not None:
        if os.path.exists(args.mm_projector_ckpt_dir):
            ckpt_path = os.path.join(args.mm_projector_ckpt_dir, "mm_projector.bin")
            if os.path.exists(ckpt_path):
                proj_ckpt = torch.load(ckpt_path)
            else:
                ckpt_path = os.path.join(args.mm_projector_ckpt_dir, "model.safetensors")
                proj_ckpt = {}
                with safe_open(ckpt_path, framework="pt", device="cuda") as f:
                    for key in f.keys():
                        new_key = key.replace("layers.", "mm_projector.")
                        proj_ckpt[new_key] = f.get_tensor(key)
        else:
            raise FileNotFoundError(f"mm_projector_ckpt_dir {args.mm_projector_ckpt_dir} does not exist.")
        for key in proj_ckpt.keys():
            if 'mm_projector' in key:
                mm_projection_layer_suffix = key.split('mm_projector')[1]
                checkpoint['state_dict'][f'{mm_projection_layer_base_name}{mm_projection_layer_suffix}'] = (
                    param_to_weights(proj_ckpt[key])
                )

        proj_conf_file = open(os.path.join(args.mm_projector_ckpt_dir, "config.json"))

        proj_conf = json.load(proj_conf_file)
        if proj_conf['mm_projector_type'] != nemo_config.mm_cfg.mm_mlp_adapter_type:
            logging.warning(
                f"Overriding mm_projector_type from {nemo_config.mm_cfg.mm_mlp_adapter_type} to {proj_conf['mm_projector_type']}"
            )
            nemo_config.mm_cfg.mm_mlp_adapter_type = proj_conf['mm_projector_type']
        proj_conf_file.close()
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

    checkpoint[MegatronNevaModel.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config

    del model

    if nemo_config.get('megatron_amp_O2', False):
        keys = list(checkpoint['state_dict'].keys())
        for key in keys:
            checkpoint['state_dict'][key.replace('model.', 'model.module.', 1)] = checkpoint['state_dict'].pop(key)

    model = load_model(MegatronNevaModel, checkpoint, strict=False, trainer=trainer)

    model._save_restore_connector = NLPSaveRestoreConnector()

    # cast to target precision and disable cpu init
    model = model.to(dtype=dtype)
    model.cfg.use_cpu_initialization = False

    model.save_to(args.out_file)
    logging.info(f'NeMo model saved to: {args.out_file}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
