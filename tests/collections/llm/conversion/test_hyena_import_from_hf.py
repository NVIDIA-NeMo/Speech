# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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
from nemo.collections.llm.gpt.model.hyena import HuggingFaceSavannaHyenaImporter


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-model", type=str, default="/home/TestData/nemo2_ckpt/HyenaConfig1B/savanna_evo2_1b_base.pt"
    )
    parser.add_argument("--output-path", type=str, default="/home/TestData/nemo2_ckpt/HyenaConfig1B/nemo2_hyena_1b")
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    evo2_config = llm.Hyena1bConfig()
    exporter = HuggingFaceSavannaHyenaImporter(args.hf_model, model_config=evo2_config)

    exporter.apply(args.output_path)
