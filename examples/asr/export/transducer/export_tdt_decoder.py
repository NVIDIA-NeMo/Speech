# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Package prediction/joint weights and a complete TDT decoder as a Triton model."""

import shutil
from argparse import ArgumentParser
from pathlib import Path

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.tdt_decoder_artifact import build_tdt_decoder, save_tdt_decoder


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Input TDT .nemo checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="New Triton model directory.")
    parser.add_argument("--max-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.max_batch_size < 1:
        parser.error("max-batch-size must be positive.")
    if args.output.exists():
        raise FileExistsError(args.output)
    model = ASRModel.restore_from(str(args.model), map_location="cpu").float().eval()
    durations = model.decoding.durations
    if not durations:
        raise ValueError("The checkpoint must be a TDT model.")
    max_symbols = model.cfg.decoding.get("greedy", {}).get("max_symbols", 10)
    if max_symbols is None:
        raise ValueError("Serving requires a finite greedy.max_symbols limit in the checkpoint.")
    decoder = build_tdt_decoder(
        model.decoder, model.joint, model.decoding.blank_id, list(durations), max_symbols, use_cuda_graphs=False
    )
    version = args.output / "1"
    version.mkdir(parents=True)
    save_tdt_decoder(decoder, version / "decoder.pt")
    shutil.copyfile(Path(__file__).with_name("tdt_triton_model.py"), version / "model.py")
    channels = model.joint.enc.in_features
    config = f'''backend: "python"
max_batch_size: {args.max_batch_size}
input [
  {{ name: "ENCODER_OUTPUT" data_type: TYPE_FP32 dims: [ {channels}, -1 ] }},
  {{ name: "ENCODED_LENGTH" data_type: TYPE_INT64 dims: [ 1 ] }}
]
output [
  {{ name: "TOKENS" data_type: TYPE_INT64 dims: [ -1 ] }},
  {{ name: "TOKEN_LENGTH" data_type: TYPE_INT64 dims: [ 1 ] }},
  {{ name: "TIMESTAMPS" data_type: TYPE_INT64 dims: [ -1 ] }},
  {{ name: "TOKEN_DURATIONS" data_type: TYPE_INT64 dims: [ -1 ] }}
]
instance_group [ {{ kind: KIND_GPU count: 1 }} ]
dynamic_batching {{}}
parameters {{ key: "FORCE_CPU_ONLY_INPUT_TENSORS" value: {{ string_value: "no" }} }}
parameters {{ key: "use_cuda_graphs" value: {{ string_value: "true" }} }}
'''
    (args.output / "config.pbtxt").write_text(config)


if __name__ == "__main__":
    main()
