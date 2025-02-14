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

"""
Convert a NeMo 2.0 checkpoint to NeMo 1.0 for TRTLLM export.
Example to run this conversion script:
```
    python /opt/NeMo/scripts/scripts/export/convert_nemo2_for_export.py \
     --input_path /path/to/nemo2/ckpt \
     --output_path /path/to/output \
     --tokenizer_type huggingface \
     --tokenizer_name meta-llama/Llama-3.1-8B \
     --symbolic_link=True
```
"""

import os
import shutil
from argparse import ArgumentParser

from omegaconf import OmegaConf

from nemo.lightning import io


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to nemo 2.0 checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="huggingface",
        help="Type of tokenizer",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Name or path of tokenizer",
    )
    parser.add_argument(
        "--symbolic_link",
        type=bool,
        default=True,
        help="Whether to use symbiloc link for model weights",
    )

    args = parser.parse_args()
    return args


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    weight_path = os.path.join(output_path, "model_weights")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print(f"Remove existing {output_path}")

    os.makedirs(output_path, exist_ok=True)

    config = io.load_context(input_path, subpath="model.config")

    config_dict = {}
    for k, v in config.__dict__.items():
        if isinstance(v, (float, int, str, bool)):
            config_dict[k] = v
        elif k == "activation_func":
            config_dict["activation"] = v.__name__

    if config_dict.get("num_moe_experts") is None:
        config_dict["num_moe_experts"] = 0
        config_dict["moe_router_topk"] = 0
    if config_dict["activation"] == "silu":
        config_dict["activation"] = "fast-swiglu"

    config_dict["mcore_gpt"] = True
    config_dict["max_position_embeddings"] = config_dict.get("seq_length")
    config_dict["tokenizer"] = {
        "library": args.tokenizer_type,
        "type": args.tokenizer_name,
        "use_fast": True,
    }

    yaml_config = OmegaConf.create(config_dict)
    OmegaConf.save(config=yaml_config, f=os.path.join(output_path, "model_config.yaml"))

    if args.symbolic_link:
        os.symlink(input_path, weight_path)
    else:
        os.makedirs(weight_path, exist_ok=True)
        for file in os.listdir(input_path):
            source_path = os.path.join(input_path, file)
            target_path = os.path.join(weight_path, file)
            shutil.copy(source_path, target_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
