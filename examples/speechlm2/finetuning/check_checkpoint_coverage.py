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
"""Preflight: does the model this config builds actually accept every checkpoint tensor?

This is the cheapest guard against the failure mode that cost the most time in
this project: NeMo assembles a model whose module layout differs from the one
that produced the checkpoint, the loader is non-strict, and training or
evaluation proceeds with randomly initialized submodules and no warning.

The check builds the model on CPU and compares its `state_dict()` keys and
shapes against the safetensors header, which is read without loading any
checkpoint tensor data.

Exit code 0 means every checkpoint tensor has a home and every model parameter
is covered.
"""
import argparse
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="exported checkpoint directory")
    p.add_argument("--config", default=None, help="training YAML; defaults to the checkpoint's own config.json")
    p.add_argument("--ignore-suffix", default="_extra_state", help="tensor name suffix to ignore")
    return p.parse_args()


def safetensors_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Read tensor names and shapes from the header without mapping the payload."""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}


def group(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", name)


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    weights = ckpt / "model.safetensors"
    if not weights.is_file():
        sys.exit(f"missing {weights}")

    ckpt_shapes = {k: shape for k, shape in safetensors_shapes(weights).items() if not k.endswith(args.ignore_suffix)}
    ckpt_keys = set(ckpt_shapes)

    from omegaconf import OmegaConf

    from nemo.collections.speechlm2.models import SALMAutomodel

    if args.config:
        cfg = OmegaConf.to_container(OmegaConf.load(args.config).model, resolve=True)
    else:
        cfg = json.loads((ckpt / "config.json").read_text())
    # Build the architecture only: no pretrained child weights, no checkpoint restore.
    cfg["pretrained_weights"] = False
    cfg["init_from_checkpoint"] = None
    cfg["init_configure_model"] = True

    # The model is built for real on CPU rather than on the meta device: the
    # perception encoder copies tensors during construction, which meta cannot
    # service ("Cannot copy out of meta tensor; no data!"). bf16 keeps the
    # large LLM at ~2 bytes/parameter of host RAM.
    cfg["torch_dtype"] = "bfloat16"
    model = SALMAutomodel(cfg)
    model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items() if not k.endswith(args.ignore_suffix)}
    model_keys = set(model_shapes)
    del model

    missing_in_model = sorted(ckpt_keys - model_keys)  # checkpoint tensors that would be dropped
    # Freshly created adapter weights have no counterpart in a base checkpoint by
    # construction, so they are reported separately rather than as a mismatch.
    is_adapter = lambda k: ".lora_A." in k or ".lora_B." in k  # noqa: E731
    new_adapters = sorted(k for k in model_keys - ckpt_keys if is_adapter(k))
    missing_in_ckpt = sorted(k for k in model_keys - ckpt_keys if not is_adapter(k))

    shape_mismatches = sorted(k for k in ckpt_keys & model_keys if ckpt_shapes[k] != model_shapes[k])

    print(f"checkpoint tensors: {len(ckpt_keys)}")
    print(f"model parameters:   {len(model_keys)}")
    print(f"matched:            {len(ckpt_keys & model_keys) - len(shape_mismatches)}")
    if new_adapters:
        print(f"new LoRA adapters:  {len(new_adapters)} (expected — these are created by the recipe)")

    ok = True
    if missing_in_model:
        ok = False
        print(f"\nERROR: {len(missing_in_model)} checkpoint tensors have no matching model parameter.")
        print("These weights would be silently discarded. Grouped by shape of the name:")
        for g, c in Counter(group(k) for k in missing_in_model).most_common(20):
            print(f"  {c:5d}  {g}")
    if missing_in_ckpt:
        ok = False
        print(f"\nERROR: {len(missing_in_ckpt)} model parameters are absent from the checkpoint.")
        print("These modules would keep their random initialization. Grouped:")
        for g, c in Counter(group(k) for k in missing_in_ckpt).most_common(20):
            print(f"  {c:5d}  {g}")

    if shape_mismatches:
        ok = False
        print(f"\nERROR: {len(shape_mismatches)} checkpoint tensors have incompatible shapes.")
        for k in shape_mismatches[:20]:
            print(f"  {k}: checkpoint {ckpt_shapes[k]}, model {model_shapes[k]}")
        if len(shape_mismatches) > 20:
            print(f"  ... {len(shape_mismatches) - 20} more shape mismatches")

    if ok:
        print("\nOK: the configured model and the checkpoint agree on every tensor.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
