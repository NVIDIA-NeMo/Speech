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

"""Build an offline ASR encoder engine from ONNX using NeMo's TensorRT builder.

This is the ONNX route; ``build_torch_trt_encoder.py`` converts the Transformer encoder directly with Torch-TensorRT.
The inference guide ("Offline TensorRT encoder" in ``docs/source/asr/inference.rst``) says when to use which route.
It also names the TensorRT versions each route needs.
"""

from argparse import ArgumentParser
from pathlib import Path

import torch

from nemo.utils.trt_utils import build_engine


def main() -> None:
    """Build one engine with a dynamic batch/time profile, leaving ONNX weights unchanged."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--features", type=int, required=True, help="Number of input acoustic features.")
    parser.add_argument("--min-frames", type=int, default=8, help="Minimum frames before subsampling.")
    parser.add_argument("--opt-frames", type=int, default=1000)
    parser.add_argument("--max-frames", type=int, default=4000)
    parser.add_argument("--opt-batch", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument(
        "--fp16", action="store_true", help="Allow FP16 internal kernels; validate recognition accuracy."
    )
    parser.add_argument("--tf32", action="store_true", help="Allow TF32 internal kernels.")
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--timing-cache", type=Path)
    args = parser.parse_args()
    if not 0 < args.min_frames <= args.opt_frames <= args.max_frames:
        parser.error("Require 0 < min-frames <= opt-frames <= max-frames.")
    if not 1 <= args.opt_batch <= args.max_batch or args.features <= 0 or args.workspace_gib <= 0:
        parser.error("Require 1 <= opt-batch <= max-batch and positive features/workspace.")
    if args.engine.exists():
        raise FileExistsError(args.engine)
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    build_engine(
        onnx_path=str(args.onnx),
        output_path=str(args.engine),
        fp16=args.fp16,
        tf32=args.tf32,
        input_profile={
            "audio_signal": [
                (1, args.features, args.min_frames),
                (args.opt_batch, args.features, args.opt_frames),
                (args.max_batch, args.features, args.max_frames),
            ],
            "length": [(1,), (args.opt_batch,), (args.max_batch,)],
        },
        timing_cache=str(args.timing_cache) if args.timing_cache else None,
        workspace_size=int(args.workspace_gib * 1024**3),
    )


if __name__ == "__main__":
    main()
