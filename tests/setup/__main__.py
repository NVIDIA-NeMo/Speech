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
import os

from .data.create_sample_jsonl import create_sample_jsonl
from .models.create_hf_model import create_hf_model

print("Setup test data and models...")

parser = argparse.ArgumentParser("Setup test data and models.")
parser.add_argument("--save_dir", required=True, help="Root save directory for artifacts")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files and directories")
args = parser.parse_args()

print(f"Arguments are: {vars(args)}")

os.makedirs(args.save_dir, exist_ok=True)

create_sample_jsonl(
    output_file=os.path.join(args.save_dir, "test_quantization", "test.json"), overwrite=args.overwrite,
)

create_hf_model(
    model_name_or_path="/home/TestData/nlp/megatron_llama/llama-ci-hf",
    output_dir=os.path.join(args.save_dir, "megatron_llama/llama-ci-hf-tiny"),
    config_updates={"hidden_size": 256, "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 4},
    overwrite=args.overwrite,
)
print("Setup done.")
