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
"""Uniformly average the weights of several exported (merged) checkpoints.

Weight averaging across neighbouring checkpoints of one run is a cheap way to
recover some of the generalization a converged-and-overfitting run has lost: the
checkpoints sit in the same basin, so their mean is a valid model and tends to
land in a flatter part of it.

Average *merged* checkpoints only. Averaging LoRA factors is not the same
operation -- mean(B)@mean(A) != mean(B@A) -- so adapters have to be folded into
the base weights first.

Integer and boolean buffers are taken from the first checkpoint rather than
averaged; they are counters and masks, not parameters.

The average is built under ``<dst>.unfinished`` and renamed into place only after the weights and every config file
are written, so an interrupted run never leaves something that looks complete. An existing destination is an error,
unless ``--reuse`` is given and its ``provenance.json`` lists exactly the same source checkpoints (by digest).
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprint import checkpoint_digest, differences  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True, help="two or more merged checkpoint dirs")
    p.add_argument("--dst", required=True)
    p.add_argument("--reuse", action="store_true", help="keep an existing destination built from the same sources")
    return p.parse_args()


def main():
    args = parse_args()
    from safetensors.torch import load_file, save_file

    if len(args.src) < 2:
        raise SystemExit("need at least two checkpoints to average")
    dst = Path(args.dst)
    staging = dst.parent / (dst.name + ".unfinished")
    provenance = {"sources": [checkpoint_digest(s) for s in args.src]}
    if dst.exists():
        prov = dst / "provenance.json"
        if not args.reuse:
            raise SystemExit(f"{dst} already exists; delete it or pass --reuse")
        if not (dst / "model.safetensors").is_file() or not prov.is_file():
            raise SystemExit(f"{dst} exists but is not a complete average; delete it first")
        diff = differences(json.loads(prov.read_text()), provenance)
        if diff:
            raise SystemExit(f"{dst} was averaged from different checkpoints; delete it or use another --dst")
        print(f"[avg] {dst} exists with matching provenance, reusing")
        return
    if staging.exists():
        raise SystemExit(f"{staging} exists (an interrupted average?); delete it first")
    staging.mkdir(parents=True)

    acc = None
    for i, src in enumerate(args.src):
        state = load_file(str(Path(src) / "model.safetensors"))
        if acc is None:
            acc = {k: (v.float() if v.is_floating_point() else v.clone()) for k, v in state.items()}
            dtypes = {k: v.dtype for k, v in state.items()}
            continue
        if set(state) != set(acc):
            raise SystemExit(f"{src} has a different tensor set than {args.src[0]}")
        for k, v in state.items():
            if v.is_floating_point():
                acc[k] += v.float()
    n = len(args.src)
    out = {}
    for k, v in acc.items():
        out[k] = (v / n).to(dtypes[k]) if v.is_floating_point() else v
    save_file(out, str(staging / "model.safetensors"), metadata={"format": "pt"})

    first = Path(args.src[0])
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (first / name).is_file():
            shutil.copy2(first / name, staging / name)
    if (first / "llm_backbone").is_dir():
        shutil.copytree(first / "llm_backbone", staging / "llm_backbone")
    (staging / "provenance.json").write_text(json.dumps(provenance, indent=1))
    os.replace(staging, dst)
    print(f"[avg] averaged {n} checkpoints -> {dst}")


if __name__ == "__main__":
    main()
