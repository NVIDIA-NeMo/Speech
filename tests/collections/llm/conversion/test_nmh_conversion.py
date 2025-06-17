# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
from nemo.collections.llm.gpt.model.ssm import HFNemotronHExporter, HFNemotronHImporter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversion_type", type=str, required=True)
    parser.add_argument("--source_ckpt", type=str, required=True)
    parser.add_argument("--target_ckpt", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    nmh_config = llm.NemotronHConfig4B()
    if args.conversion_type == "NEMO2_TO_HF":

        exporter = HFNemotronHExporter(args.source_ckpt, model_config=nmh_config)
        exporter.apply(args.target_ckpt)

    elif args.conversion_type == "HF_TO_NEMO2":

        exporter = HFNemotronHImporter(args.source_ckpt)
        exporter.apply(args.target_ckpt)

    else:
        raise ValueError(f"Invalid conversion type: {args.conversion_type}")
