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

"""Build a single-step TensorRT engine from a NeMo RNNT/TDT decoder-joint ONNX export."""

from argparse import ArgumentParser
from pathlib import Path

import onnx


def decoder_joint_profile(onnx_path: Path, opt_batch: int, max_batch: int) -> dict[str, list[tuple[int, ...]]]:
    """Keep one acoustic frame and one label per call, with dynamic batch and explicit recurrent states."""
    if not 1 <= opt_batch <= max_batch:
        raise ValueError("Require 1 <= opt_batch <= max_batch.")
    model = onnx.load(str(onnx_path), load_external_data=False)
    axes = {
        "encoder_outputs": (0, 3),
        "targets": (0, 2),
        "target_length": (0, 1),
        "input_states_1": (1, 3),
        "input_states_2": (1, 3),
    }
    inputs = {value.name: value for value in model.graph.input}
    if set(inputs) != set(axes):
        raise ValueError(f"Expected NeMo decoder-joint inputs {set(axes)}, got {set(inputs)}.")
    profiles = {}
    for name, (batch_axis, rank) in axes.items():
        dims = inputs[name].type.tensor_type.shape.dim
        if len(dims) != rank:
            raise ValueError(f"Unexpected rank for {name}: {len(dims)}")
        shapes = []
        for batch in (1, opt_batch, max_batch):
            shape = []
            for axis, dim in enumerate(dims):
                if axis == batch_axis:
                    size = batch
                elif (name == "encoder_outputs" and axis == 2) or (name == "targets" and axis == 1):
                    size = 1
                elif dim.dim_value > 0:
                    size = dim.dim_value
                else:
                    raise ValueError(f"Expected a static non-batch dimension: {name}[{axis}]")
                if dim.dim_value > 0 and dim.dim_value != size:
                    raise ValueError(f"Requested shape conflicts with static dimension: {name}[{axis}]")
                shape.append(size)
            shapes.append(tuple(shape))
        profiles[name] = shapes
    return profiles


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--opt-batch", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument(
        "--fp16", action="store_true", help="Allow FP16 kernels; validate logits and recurrent states."
    )
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    parser.add_argument("--timing-cache", type=Path)
    args = parser.parse_args()
    if args.workspace_gib <= 0:
        parser.error("workspace-gib must be positive.")
    profile = decoder_joint_profile(args.onnx, args.opt_batch, args.max_batch)
    if args.engine.exists():
        raise FileExistsError(args.engine)
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    from nemo.utils.trt_utils import build_engine

    build_engine(
        onnx_path=str(args.onnx),
        output_path=str(args.engine),
        fp16=args.fp16,
        tf32=args.tf32,
        input_profile=profile,
        timing_cache=str(args.timing_cache) if args.timing_cache else None,
        workspace_size=int(args.workspace_gib * 1024**3),
    )


if __name__ == "__main__":
    main()
