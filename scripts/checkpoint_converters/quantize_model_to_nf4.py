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

from argparse import ArgumentParser
from typing import List

import torch
from lightning.pytorch import Trainer
from torch import nn

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.modules.common.megatron.adapters.qlora import nf4_quantize
from nemo.collections.nlp.parts.nlp_overrides import MegatronHalfPrecisionPlugin, NLPDDPStrategy
from nemo.utils import logging

'''
This script quantizes the weights of linear layers to NF4 precision, then saves them in BF16 precision.
The resulting model will have the same format as the input, but have weights compatible with adapters trained
with QLoRA. 
Flow of QLoRA inference
- Path 1 (online quantize): similar to training, set eval peft_scheme to 'qlora' and linear layers will be quantized 
  immediately after model loading. This is applicable to framework inference only.
- Path 2 (offline quantize): run this script to get a new pretrained base model, then set eval `peft_scheme` to `lora`.
Path 1 and Path 2 yield identical inference results, but Path 2 enables deployment of a QLoRA model without further 
changes downstream.

Example usage:
python scripts/checkpoint_converters/quantize_model_to_nf4.py \
--input_name_or_path <base_nemo_model> \
--output_path <quantized_nemo_model> \
--target_modules linear_qkv,linear_proj,linear_fc1,linear_fc2
'''


def corrupt_linear_weight_(model: nn.Module, target_modules: List[str]):
    """
    Corrupt the linear weights of a model as specified by quantize_targets
    "Corrupting" refers to quantizing the linear weights to NF4 then casting back to BF16
    """
    state_dict = model.state_dict()
    keys = state_dict.keys()
    for k in keys:
        if any(f"{l}.weight" in k for l in target_modules):
            # Convert a BF16 tensor to NF4 then back to BF16
            state_dict[k] = nf4_quantize(state_dict[k]).dequantize()
    model.load_state_dict(state_dict)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        required=True,
        help="Path to .nemo base model checkpoint",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to output quantized .nemo file.")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="linear_qkv,linear_proj,linear_fc1,linear_fc2",
        help="Comma separated list of which linear module(s) to quantize",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    dummy_trainer = Trainer(
        devices=1,
        accelerator='gpu',
        strategy=NLPDDPStrategy(),
        plugins=[MegatronHalfPrecisionPlugin(precision='bf16-mixed', device='cuda')],
    )
    model = MegatronGPTSFTModel.restore_from(args.input_name_or_path, trainer=dummy_trainer).to(torch.bfloat16)
    corrupt_linear_weight_(model, args.target_modules.split(','))

    model.save_to(args.output_path)
    logging.info(f"Quantized model saved to {args.output_path}")
