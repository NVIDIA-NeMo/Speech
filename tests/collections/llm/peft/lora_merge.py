# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.collections import llm


def get_args():
    parser = argparse.ArgumentParser(description='Merge LoRA weights with base LLM')
    parser.add_argument('--lora_checkpoint_path', type=str, help="Path to finetuned LORA checkpoint")
    parser.add_argument('--output_path', type=str, help="Path to save merged checkpoint")
    parser.add_argument('--legacy_ckpt', action="store_true", help="Load ckpt saved with TE < 1.14")
    parser.add_argument('--trust_remote_code', action="store_true", help="If trust_remore_code")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    llm.peft.merge_lora(
        lora_checkpoint_path=args.lora_checkpoint_path,
        output_path=args.output_path,
        legacy_ckpt=args.legacy_ckpt,
        trust_remote_code=args.trust_remote_code,
    )
