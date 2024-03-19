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
A script to convert the BigCode StarCoder checkpoints from HuggingFace to Megatron GPTModel.
This script is hardcoded specifically for the StarCoder pretrained models only, and is not
generalisable to any other models.

This script will load and convert the model entirely on CPU for OOM safety, but it is
possible to initialize the model on GPU before the save down. You can do this by adding --cuda
parameter to this script call.

This script requires that you have downloaded the StarCoder checkpoint from HuggingFace.
This can be done using Git with the following command:
```bash
git clone https://huggingface.co/bigcode/starcoder
```
Note that downloading this particular checkpoint requires authentication with a HuggingFace token.

The script will generate a Megatron model with TP=1 and PP=1. If you need different TP/PP
values, then after running this script, please use the following script to set whatever
TP/PP configuration you want:
    NeMo/examples/nlp/language_modeling/megatron_change_num_partitions.py

This script also requires a baseline config file from which to override default parameters.
You can specify the location of this file using the -c argument. Please use the config below
to correctly configure creating GPT-2 model in Megatron:
    NeMo/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml


Here is an example usage command:
```python
python convert_starcoder_hf_to_nemo.py \
    --input_name_or_path /path/to/starcoder \
    --output_path /path/to/save.nemo
```
"""

import argparse
import os
from typing import Dict

import pytorch_lightning as pl
import torch
import yaml
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.collections.nlp.parts.utils_funcs import load_state_dict_helper
from nemo.utils import logging


def convert_state_dict(state_dict: Dict[str, torch.Tensor], amp: bool = False):
    def get_new_key(old_key):
        if old_key == "transformer.wte.weight":
            return "embedding.word_embeddings.weight"
        if old_key == "transformer.wpe.weight":
            return "embedding.position_embeddings.weight"
        elif old_key.startswith("transformer.ln_f"):
            return old_key.replace("transformer.ln_f", "decoder.final_layernorm")
        elif old_key.startswith("lm_head"):
            return old_key.replace("lm_head", "output_layer")
        else:
            p1 = old_key.replace("transformer.h", "decoder.layers")
            p2 = p1.replace("ln_1.", "self_attention.linear_qkv.layer_norm_")
            p3 = p2.replace("attn.c_proj", "self_attention.linear_proj")
            p4 = p3.replace("attn.c_attn", "self_attention.linear_qkv")
            p5 = p4.replace("ln_2.", "mlp.linear_fc1.layer_norm_")
            p6 = p5.replace("c_fc", "linear_fc1")
            p7 = p6.replace("c_proj", "linear_fc2")
            return p7

    new_dict = {}
    prefix = "model.module." if amp else "model."

    for old_key, val in state_dict.items():
        new_key = get_new_key(old_key)
        new_key = prefix + new_key
        new_dict[new_key] = val

    return new_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        required=True,
        help="Path to Starcoder checkpoint from HuggingFace hub or local dir",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to dir where to store output .nemo file")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), '../../examples/nlp/language_modeling/conf/megatron_gpt_config.yaml'
        ),
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )
    parser.add_argument("--cuda", action="store_true", help="Put Nemo model onto GPU prior to saving")
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        raise FileNotFoundError(f"Output directory '{args.output_path}' does not exist")

    hf_config = AutoConfig.from_pretrained(args.input_name_or_path)

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
        "num_query_groups": 1,  # MQA
        "hidden_size": hf_config.n_embd,
        "encoder_seq_length": hf_config.n_positions,
        "max_position_embeddings": hf_config.n_positions,
        "num_layers": hf_config.n_layer,
        "cpu_offloading_num_layers": hf_config.n_layer - 1,  # @chcui temp workaround before m-lm !1124 is merged
        "num_attention_heads": hf_config.n_head,
        "ffn_hidden_size": hf_config.n_inner,
        "layernorm_epsilon": hf_config.layer_norm_epsilon,
        "pre_process": True,
        "post_process": True,
        "apply_query_key_layer_scaling": True,
        "bias": True,
        "transformer_block_type": "pre_ln",
        "fp32_residual_connection": False,
        "hidden_dropout": hf_config.summary_first_dropout,
        "attention_dropout": hf_config.attn_pdrop,
        "ffn_dropout": 0,
        "share_embeddings_and_output_weights": False,
        "position_embedding_type": "learned_absolute",
        "normalize_attention_scores": True,
        "precision": args.precision,
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

    model_dict.update(override_model_dict)
    model_dict["tokenizer"] = tokenizer_dict

    omega_cfg = OmegaConf.create(model_dict)

    trainer = pl.Trainer(**trainer_dict)

    logging.info("Loading HuggingFace model...")
    model_hf = AutoModelForCausalLM.from_pretrained(args.input_name_or_path)
    logging.info(f"Loaded model:\n{model_hf}")

    state_dict_hf = model_hf.state_dict()
    convert_dict = convert_state_dict(state_dict_hf, amp=omega_cfg.megatron_amp_O2)

    logging.info("Creating Megatron model...")
    omega_cfg.cpu_offloading_num_layers = 0
    model = load_state_dict_helper(MegatronGPTModel, omega_cfg, trainer, convert_dict)
    logging.info(f"Created model:\n{model}")

    logging.info("Saving model...")
    # We make sure that the tokenizer can be instantiated later regardless of args.input_name_or_path
    model.cfg.tokenizer.update(type="bigcode/starcoder")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = model.to(dtype=dtype)
    model.cfg.update(use_cpu_initialization=False)
    model.save_to(os.path.join(args.output_path, "megatron_starcoder_tp1_pp1.nemo"))
    logging.info("Done.")
