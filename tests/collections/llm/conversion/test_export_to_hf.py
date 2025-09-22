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
from pathlib import Path

import torch
import transformers

from nemo.collections import llm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-path", type=str, default="/root/.cache/nemo/models/models/llama_31_toy")
    parser.add_argument("--original-hf-path", type=str, default="models/llama_31_toy")
    parser.add_argument("--output-path", type=str, default="/tmp/output_hf")
    parser.add_argument("--add-model-name", action="store_true", default=False)
    parser.add_argument("--hf-target-class", type=str, default="AutoModelForCausalLM")
    parser.add_argument(
        "--allow-mismatch", nargs='+', default=[], help="List of parameter names to compare after casting to bfloat16"
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    kwargs = {}
    if args.add_model_name:
        kwargs = {
            'target_model_name': args.original_hf_path,
        }
    llm.export_ckpt(
        path=Path(args.nemo_path),
        target='hf',
        output_path=Path(args.output_path),
        overwrite=True,
        **kwargs,
    )

    hf_target_class = getattr(transformers, args.hf_target_class)
    original_hf = hf_target_class.from_pretrained(args.original_hf_path, trust_remote_code=True)
    converted_hf = hf_target_class.from_pretrained(args.output_path, trust_remote_code=True)

    for (name1, parameter1), (name2, parameter2) in zip(
        converted_hf.named_parameters(), original_hf.named_parameters()
    ):
        if any(k in name1 for k in args.allow_mismatch):
            param1_cmp = parameter1.bfloat16()
            param2_cmp = parameter2.bfloat16()
        else:
            param1_cmp = parameter1
            param2_cmp = parameter2
        assert torch.all(
            torch.isclose(param1_cmp, param2_cmp, atol=1e-3)
        ).item(), f'Parameter weight do not match for {name1}'

    print('All weights matched.')
