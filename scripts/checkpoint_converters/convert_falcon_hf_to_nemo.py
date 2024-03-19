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

"""
Conversion script to convert Huggingface Falcon 1B/7B/40B/180B checkpoints into nemo checkpoint.

This script will generate a Megatron model with TP=1 and PP=1. The new dist ckpt format does not require
user to run additional script to set the TP/PP values manually.
    
Example to run this conversion script:
```
    python convert_falcon_hf_to_nemo.py \
     --input_name_or_path /path/to/hf/checkpoints/folder \
     --output_path /path/to/output/nemo/file \
     --precision <precision of converted nemo model>
```
"""

import argparse
import os
import time
from typing import Dict

import pytorch_lightning as pl
import torch
import yaml
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, FalconConfig

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging


def convert_state_dict(state_dict: Dict[str, torch.Tensor], amp: bool = False):
    def get_new_key(old_key):
        if old_key == "transformer.word_embeddings.weight":
            return "embedding.word_embeddings.weight"
        elif old_key.startswith("transformer.ln_f"):
            return old_key.replace("transformer.ln_f", "decoder.final_layernorm")
        elif old_key.startswith("lm_head"):
            return old_key.replace("lm_head", "output_layer")

        # For the rest, a base transformation
        key = old_key.replace("transformer.h", "decoder.layers")

        # Handling the layer normalization replacements
        if falcon_config.new_decoder_architecture:
            key = key.replace("ln_attn", "input_layernorm")
            key = key.replace("ln_mlp", "pre_mlp_layernorm")
        else:
            key = key.replace("input_layernorm", "input_layernorm")
            if not falcon_config.parallel_attn:
                key = key.replace("post_attention_layernorm", "post_self_attn_layernorm")

        key = key.replace("self_attention.dense", "self_attention.linear_proj")
        key = key.replace("self_attention.query_key_value", "self_attention.linear_qkv")
        key = key.replace("dense_h_to_4h", "linear_fc1")
        key = key.replace("dense_4h_to_h", "linear_fc2")
        return key

    new_dict = {}
    # amp O2 mode has different state dict name
    prefix = "model.module." if amp else "model."

    for old_key, val in state_dict.items():
        new_key = get_new_key(old_key)
        new_key = prefix + new_key
        new_dict[new_key] = val

    return new_dict


def load_falcon_config(args) -> FalconConfig:
    """ Helper utility to load FalconConfig.

    Legacy Falcon-7B and Falcon-40B are not compatible with `transformers.FalconConfig` and
    `transformers.FalconModel`. need to manually set the config values
    and force to `falcon` model type. 
    """
    config = FalconConfig.from_pretrained(args.input_name_or_path)
    if config.model_type == 'RefinedWeb':
        mappings = {
            "num_hidden_layers": config.n_layer,
            "num_attention_heads": config.n_head,
            "num_kv_heads": config.n_head_kv,
            "new_decoder_architecture": True,
        }
    elif config.model_type == 'RefinedWebModel':
        mappings = {
            "num_hidden_layers": config.n_layer,
            "num_attention_heads": config.n_head,
            "num_kv_heads": 1 if config.multi_query else config.n_head,
            "new_decoder_architecture": False,
        }
    else:
        return config

    for key, value in mappings.items():
        setattr(config, key, value)

    config.model_type = 'falcon'
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        required=True,
        help="Path to Falcon variants checkpoint from HuggingFace hub or local dir",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to dir where to store output .nemo file")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), '../../examples/nlp/language_modeling/conf/megatron_falcon_config.yaml'
        ),
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )
    parser.add_argument("--cuda", action="store_true", help="Put Nemo model onto GPU prior to saving")

    args = parser.parse_args()

    falcon_config = load_falcon_config(args)
    with open(args.hparams_file, "r", encoding="utf_8") as f:
        orig_cfg = yaml.safe_load(f)

    model_dict = orig_cfg["model"]

    if "data" in model_dict:
        del model_dict["data"]

    override_model_dict = {
        "micro_batch_size": 1,
        "global_batch_size": 1,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "megatron_amp_O2": False,
        "transformer_engine": True,
        "use_cpu_initialization": not args.cuda,
        "normalization": "layernorm",
        "mcore_gpt": True,
        "num_query_groups": None,  # MHA
        "hidden_size": falcon_config.hidden_size,
        "encoder_seq_length": falcon_config.max_position_embeddings,
        "max_position_embeddings": falcon_config.max_position_embeddings,
        "num_layers": falcon_config.num_hidden_layers,
        "num_attention_heads": falcon_config.num_attention_heads,
        "ffn_hidden_size": falcon_config.hidden_size * 4,
        "layernorm_epsilon": falcon_config.layer_norm_epsilon,
        "pre_process": True,
        "post_process": True,
        "apply_query_key_layer_scaling": False,
        "bias": falcon_config.bias,
        "transformer_block_type": "pre_ln",
        "fp32_residual_connection": False,
        "hidden_dropout": falcon_config.hidden_dropout,
        "attention_dropout": falcon_config.attention_dropout,
        "ffn_dropout": 0,
        "share_embeddings_and_output_weights": False,
        "position_embedding_type": "rope",
        "precision": args.precision,
        "init_method_std": falcon_config.initializer_range,
        "activation": "gelu",
        "bias_activation_fusion": False,
        "bias_dropout_add_fusion": False,
        "seq_len_interpolation_factor": None,
    }

    mcore_customization_config_dict = {
        "new_decoder_architecture": falcon_config.new_decoder_architecture,
        "parallel_attention": falcon_config.parallel_attn,
    }

    tokenizer_dict = {
        "library": "huggingface",
        "type": args.input_name_or_path,
        "use_fast": True,
    }
    trainer_dict = {
        "devices": 1,
        "num_nodes": 1,
        "accelerator": "gpu" if args.cuda else "cpu",
        "precision": args.precision,
        "logger": False,
        "enable_checkpointing": False,
        "max_epochs": -1,
        "max_steps": 100000,
        "log_every_n_steps": 10,
        "val_check_interval": 100,
        "limit_val_batches": 50,
        "limit_test_batches": 500,
        "accumulate_grad_batches": 1,
        "gradient_clip_val": 1.0,
        "benchmark": False,
        "enable_model_summary": False,
        "strategy": NLPDDPStrategy(),
    }

    # Additional logic for position_embedding_type = alibi
    if falcon_config.alibi:
        try:
            raise ValueError(
                "Alibi is not yet supported in Megatron Core, \
                force to use RoPE will generate suboptimal responses"
            )
        except ValueError as e:
            print(e)

    # Additional logic for num_query_groups
    if override_model_dict.get("num_query_groups") is None:
        if falcon_config.new_decoder_architecture:
            override_model_dict["num_query_groups"] = falcon_config.num_kv_heads
        elif falcon_config.multi_query:
            override_model_dict["num_query_groups"] = 1

    # Additional logic for bias fusion
    if falcon_config.bias:
        override_model_dict["bias_activation_fusion"] = True
        override_model_dict["bias_dropout_add_fusion"] = True

    # Addtional logic for rope scaling
    if falcon_config.rope_scaling is not None:
        if falcon_config.rope_scaling.type == 'linear':
            override_model_dict['seq_len_interpolation_factor'] = falcon_config.rope_scaling.factor
        else:
            raise ValueError("Only linear rope scaling type is supported now")

    model_dict.update(override_model_dict)
    model_dict["tokenizer"] = tokenizer_dict
    model_dict["name"] = 'megatron_falcon_gpt'
    model_dict["mcore_customization_config"] = mcore_customization_config_dict

    omega_cfg = OmegaConf.create(model_dict)

    trainer = pl.Trainer(**trainer_dict)

    logging.info("Creating Megatron model...")
    tik = time.time()
    model = MegatronGPTModel(omega_cfg, trainer)

    logging.info("Loading HuggingFace model...")
    model_hf = AutoModelForCausalLM.from_pretrained(args.input_name_or_path)

    state_dict_hf = model_hf.state_dict()
    convert_dict = convert_state_dict(state_dict_hf, amp=omega_cfg.megatron_amp_O2)

    logging.info("Loading state dict...")
    missing_keys, unexpected_keys = model.load_state_dict(convert_dict, strict=False)

    if missing_keys:
        # Keys ending with '_extra_state' are related to Transformer Engine internals
        missing_keys_non_extra = [key for key in missing_keys if not key.endswith("_extra_state")]
        if missing_keys_non_extra:
            logging.critical("Missing keys were detected during the load, something has gone wrong. Aborting.")
            raise RuntimeError(f"Missing keys: \n{missing_keys_non_extra}")

    if unexpected_keys:
        logging.critical("Unexpected keys were detected which should not happen. Aborting.")
        raise RuntimeError(f"Unexpected keys: \n{unexpected_keys}")

    logging.info("Saving model...")

    # We make sure that the tokenizer can be instantiated later regardless of args.input_name_or_path
    if falcon_config.new_decoder_architecture:
        model.cfg.tokenizer.update(type="tiiuae/falcon-40b")
    elif falcon_config.multi_query:
        model.cfg.tokenizer.update(type="tiiuae/falcon-7b")
    elif falcon_config.alibi and falcon_config.num_hidden_layers == 36:
        model.cfg.tokenizer.update(type="tiiuae/falcon-rw-7b")
    else:
        model.cfg.tokenizer.update(type="tiiuae/falcon-rw-1b")

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = model.to(dtype=dtype)
    model.cfg.update(use_cpu_initialization=False)
    model.save_to(args.output_path)
    logging.info(f'Done. NeMo model saved to: {args.output_path}')
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logging.info(f'nemo model created and saved. Total time: {t}')
