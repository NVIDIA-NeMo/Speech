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

import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from pytorch_lightning import Trainer
from transformers import AutoModelForCausalLM

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging

"""
Script to convert a falcon checkpoint in nemo (mcore path) into a HuggingFace checkpoint.
This script can be used to 1) generate only the HF weights, or 2) generate an entire HF model folder.

1) Generate only HF weights from a nemo file:

    python convert_falcon_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin
    
2) Generate the full HF model folder

    python convert_falcon_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin \
    --hf_input_path /path/to/input_hf_folder \
    --hf_output_path /path/to/output_hf_folder

    Use the --cpu-only flag if the model cannot fit in the GPU (e.g. falcon 180b). 
    However this option makes the conversion script significantly slower.
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path", type=str, required=True, help="Path to .nemo file",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to HF .bin file")
    parser.add_argument(
        "--hf_input_path",
        type=str,
        default=None,
        help="A HF model path, "
        "e.g. a folder containing https://huggingface.co/meta-falcon/falcon-2-7b-hf/tree/main",
    )
    parser.add_argument(
        "--hf_output_path",
        type=str,
        default=None,
        help="Output HF model path, " "with the same format as above but user's own weights",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision of output weights."
        "Defaults to precision of the input nemo weights (model.cfg.trainer.precision)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Load model in cpu only. Useful if the model cannot fit in GPU memory, "
        "but this option makes the conversion script significantly slower.",
    )
    args = parser.parse_args()
    return args


def convert(input_nemo_file, output_hf_file, precision=None, cpu_only=False) -> None:
    """
    Convert NeMo weights to HF weights
    """
    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    if cpu_only:
        map_location = torch.device('cpu')
        model_config = MegatronGPTModel.restore_from(input_nemo_file, trainer=dummy_trainer, return_config=True)
        model_config.use_cpu_initialization = True
        model_config.tensor_model_parallel_size = 1
    else:
        map_location, model_config = None, None

    if cpu_only:
        logging.info("******** Loading model on CPU. This will take a significant amount of time.")
    model = MegatronGPTModel.restore_from(
        input_nemo_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
    )
    if precision is None:
        precision = model.cfg.precision
    try:
        dtype = torch_dtype_from_precision(precision)
    except ValueError as e:
        logging.warning(str(e) + f", precision string '{precision}' is not recognized, falling back to fp32")
        dtype = torch.float32  # fallback

    param_to_weights = lambda param: param.to(dtype)
    checkpoint = OrderedDict()

    def get_original_key(new_key):
        new_key = new_key[len(prefix) :]

        if new_key.startswith("embedding.word_embeddings.weight"):
            return "transformer.word_embeddings.weight"
        elif new_key.startswith("decoder.final_layernorm"):
            return new_key.replace("decoder.final_layernorm", "transformer.ln_f")
        elif new_key.startswith("output_layer"):
            return new_key.replace("output_layer", "lm_head")

        key = new_key.replace("decoder.layers", "transformer.h")

        if model.cfg.mcore_customization_config.new_decoder_architecture:
            key = key.replace("input_layernorm", "ln_attn")
            key = key.replace("pre_mlp_layernorm", "ln_mlp")
        else:
            key = key.replace("input_layernorm", "input_layernorm")
            if not model.cfg.mcore_customization_config.parallel_attention:
                key = key.replace("post_self_attn_layernorm", "post_attention_layernorm")

        key = key.replace("self_attention.linear_proj", "self_attention.dense")
        key = key.replace("self_attention.linear_qkv", "self_attention.query_key_value")
        key = key.replace("linear_fc1", "dense_h_to_4h")
        key = key.replace("linear_fc2", "dense_4h_to_h")
        return key

    prefix = 'model.module.' if any(k.startswith('model.module.') for k in model.state_dict()) else 'model.'

    for key, value in model.state_dict().items():
        if '_extra_state' in key:
            continue
        orig_key = get_original_key(key)
        checkpoint[orig_key] = param_to_weights(value)

    os.makedirs(os.path.dirname(output_hf_file), exist_ok=True)
    torch.save(checkpoint, output_hf_file)
    logging.info(f"Weights reverted and saved to {output_hf_file}")


def replace_hf_weights(weights_file, input_hf_path, output_hf_path):
    model = AutoModelForCausalLM.from_pretrained(input_hf_path, local_files_only=True)
    nemo_exported = torch.load(weights_file)

    model.load_state_dict(nemo_exported)
    model.save_pretrained(output_hf_path)
    logging.info(f"Full HF model saved to {output_hf_path}")


if __name__ == '__main__':
    args = get_args()
    convert(args.input_name_or_path, args.output_path, precision=args.precision, cpu_only=args.cpu_only)
    if args.hf_input_path and args.hf_output_path:
        replace_hf_weights(args.output_path, args.hf_input_path, args.hf_output_path)
    else:
        logging.info("`hf-in-path` and/or `hf-out-path` not provided, not generating full HF model.")
        logging.info(f".bin file is saved to {args.output_path}")
