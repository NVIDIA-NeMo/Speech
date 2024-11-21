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
import argparse
from dataclasses import dataclass

from nemo.collections import llm


@dataclass
class Llama3ConfigCI(llm.Llama3Config8B):
    seq_length: int = 2048
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 8


def get_args():
    parser = argparse.ArgumentParser(description='Merge LoRA weights with base LLM')
    parser.add_argument('--lora_checkpoint_path', type=str, help="Path to finetuned LORA checkpoint")
    parser.add_argument('--output_path', type=str, help="Path to save merged checkpoint")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    llm.peft.merge_lora(
        lora_checkpoint_path=args.lora_checkpoint_path,
        output_path=args.output_path,
    )
