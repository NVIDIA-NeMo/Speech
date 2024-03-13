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
import json
import os

from typing import Any, Dict, Optional

import transformers

"""
Create a randomly initialized HuggingFace model for testing purposes.

Model can be specified by name or path for creating its config and tokenizer using
HuggingFace transformers AutoConfig and AutoTokenizer functions.

Parameter config_updates can be used to override specific model config fields to make
it smaller, for example, by changing number of layers or hidden layers dimensionality,
making it adequate for testing purposes. This parameter should be specified as
a dictionary that can be parsed using json.loads method.

Example usage for Llama2 model (requires HF login):
```
python tests/setup/models/create_tiny_hf_model.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --output_dir tiny_llama2_hf \
  --config_updates '{"hidden_size": 128, "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 4}'
```
"""


def get_hf_model_class(hf_config):
    """Get HuggingFace model class from config."""
    if len(hf_config.architectures) > 1:
        print(f"More than one model architecture available, choosing 1st: {hf_config.architectures}")
    model_name = hf_config.architectures[0]
    model_class = getattr(transformers, model_name)
    return model_class


def create_hf_model(
    model_name_or_path: str, output_dir: str, config_updates: Optional[Dict[str, Any]] = None, overwrite: bool = False
):
    """Create HuggingFace model with optional config updates."""
    if os.path.isdir(output_dir) and not overwrite:
        print(f"Output directory {output_dir} exists and overwrite flag is not set so exiting.")
        return

    hf_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    model_class = get_hf_model_class(hf_config)

    if config_updates is not None:
        hf_config.update(config_updates)
    print(hf_config)

    model = model_class(hf_config)
    print(model)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model to {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print("OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create a HuggingFace model (random initialization) for testing purposes.")
    parser.add_argument(
        "--model_name_or_path", required=True, help="Model name or local path with model config and tokenizer",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory",
    )
    parser.add_argument(
        "--config_updates", type=json.loads, help="Parameter updates in JSON format to overwrite for model config",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite file if it exists",
    )
    args = parser.parse_args()
    create_hf_model(args.model_name_or_path, args.output_dir, args.config_updates)
