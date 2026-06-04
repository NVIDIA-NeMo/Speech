#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Stage 2/2: build a TensorRT engine from the codec-decoder ONNX.

Consumes the ONNX produced by ``export_codec_decoder_onnx.py`` and runs
``trtexec`` to build an engine with a dynamic ``batch`` (and optionally dynamic
``frames``) shape profile.

The input tensor is ``audio_codes`` with shape ``(batch, frames, num_codebooks)``.
``num_codebooks`` is read from the ONNX graph; ``batch``/``frames`` come from the
profile flags.

Example:
    python examples/tts/easymagpie_vllm_omni/export_codec_decoder_trt.py \\
        --onnx-path codec/codec_decoder.onnx \\
        --trt-path  codec/codec_decoder.plan \\
        --batch-profile 1 8 32 \\
        --frames-profile 30 30 30 --fp16

Notes
-----
* The frame axis is usually static (export with a fixed ``--frames`` and use the
  same value for min/opt/max). A dynamic frame axis works too if the ONNX was
  exported with ``frames`` dynamic.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import onnx


def _infer_num_quantizers(onnx_path):
    model = onnx.load(str(onnx_path))
    for inp in model.graph.input:
        if inp.name != "audio_codes":
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) >= 3 and dims[2].dim_value > 0:
            return int(dims[2].dim_value)
    raise RuntimeError(
        f"could not infer num_quantizers from {onnx_path} (audio_codes dim 2 is not a static positive integer)"
    )


def convert_to_trt(onnx_path, trt_path, trtexec_bin, nq, batch_prof, frames_prof, fp32):
    exe = shutil.which(trtexec_bin) if "/" not in trtexec_bin else trtexec_bin
    if exe is None:
        raise FileNotFoundError(f"trtexec not found: {trtexec_bin}")
    trt_path.parent.mkdir(parents=True, exist_ok=True)

    def s(b, f):
        return f"{b}x{f}x{nq}"

    cmd = [
        exe,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        f"--minShapes=audio_codes:{s(batch_prof[0], frames_prof[0])}",
        f"--optShapes=audio_codes:{s(batch_prof[1], frames_prof[1])}",
        f"--maxShapes=audio_codes:{s(batch_prof[2], frames_prof[2])}",
    ]
    if not fp32:
        cmd.append("--fp16")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"TensorRT engine saved to {trt_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Build a TensorRT engine from the codec-decoder ONNX")
    p.add_argument("--onnx-path", required=True)
    p.add_argument("--trt-path", required=True)
    p.add_argument("--trtexec-bin", default="/usr/src/tensorrt/bin/trtexec")
    p.add_argument("--batch-profile", nargs=3, type=int, default=[1, 8, 32], metavar=("MIN", "OPT", "MAX"))
    p.add_argument("--frames-profile", nargs=3, type=int, default=[15, 15, 15], metavar=("MIN", "OPT", "MAX"))
    p.add_argument("--fp32", action="store_true", help="Build pure FP32 engine (default: FP16).")
    return p.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_path)
    trt_path = Path(args.trt_path)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    nq = _infer_num_quantizers(onnx_path)
    print(f"num_quantizers={nq} (from {onnx_path})")

    convert_to_trt(
        onnx_path,
        trt_path,
        args.trtexec_bin,
        nq=nq,
        batch_prof=tuple(args.batch_profile),
        frames_prof=tuple(args.frames_profile),
        fp32=args.fp32,
    )


if __name__ == "__main__":
    main()
