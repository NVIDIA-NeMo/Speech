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
"""Fold LoRA adapters into the base weights of an exported SpeechLM checkpoint.

`to_hf.py` exports the model's state dict verbatim, so a LoRA fine-tune ships
`lora_A`/`lora_B` tensors beside the frozen base weights. Whether those get
applied at inference then depends on the serving stack: vLLM's SpeechLM plugin
merges them only on backbones whose backend implements `preprocess_llm_weights`,
and it has to guess the scaling from the config.

Merging at export time removes that dependency. The output is an ordinary
checkpoint with no adapter tensors, numerically identical to the fine-tuned
model, that any loader handles correctly.

    W_effective = W_base + (alpha / dim) * (B @ A)

The merge runs in float32 and casts back to the base tensor's dtype.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

_LORA_RE = re.compile(r"(.+)\.lora_(A|B)(?:\.default)?\.weight$")
# TransformerEngine parks FP8 bookkeeping beside every linear it wraps, adapters
# included. Those tensors carry no weights, but leaving them behind would point
# at submodules that no longer exist after the merge.
_LORA_EXTRA_RE = re.compile(r"\.lora_(A|B)(?:\.default)?\._extra_state$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="exported checkpoint directory")
    p.add_argument("--dst", required=True, help="output directory for the merged checkpoint")
    p.add_argument("--scaling", type=float, default=None, help="override alpha/dim")
    return p.parse_args()


def main():
    args = parse_args()
    from safetensors.torch import load_file, save_file

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    config = json.loads((src / "config.json").read_text())
    scaling = args.scaling
    if scaling is None:
        lora_cfg = config.get("lora") or {}
        rank = lora_cfg.get("dim", lora_cfg.get("r"))
        alpha = lora_cfg.get("alpha", lora_cfg.get("lora_alpha"))
        if rank is None or alpha is None:
            raise SystemExit(f"config.json has no usable lora rank/alpha (keys: {sorted(lora_cfg)}); pass --scaling")
        scaling = float(alpha) / float(rank)
    print(f"[merge] scaling = {scaling}")

    state = load_file(str(src / "model.safetensors"))
    base, lora_a, lora_b = {}, {}, {}
    dropped_extra = 0
    for name, tensor in state.items():
        if _LORA_EXTRA_RE.search(name):
            dropped_extra += 1
            continue
        m = _LORA_RE.match(name)
        if m is None:
            base[name] = tensor
            continue
        key = m.group(1) + ".weight"
        (lora_a if m.group(2) == "A" else lora_b)[key] = tensor

    if not lora_a:
        raise SystemExit("no LoRA tensors found; the checkpoint is already merged or was trained without LoRA")

    merged = 0
    for key in sorted(set(lora_a) & set(lora_b)):
        if key not in base:
            raise SystemExit(f"adapter {key} has no base weight to merge into")
        w = base[key]
        delta = scaling * (lora_b[key].float() @ lora_a[key].float())
        if delta.shape != w.shape:
            raise SystemExit(f"shape mismatch merging {key}: base {tuple(w.shape)} vs delta {tuple(delta.shape)}")
        base[key] = (w.float() + delta).to(w.dtype)
        merged += 1

    unmatched = set(lora_a) ^ set(lora_b)
    if unmatched:
        raise SystemExit(f"unpaired LoRA tensors: {sorted(unmatched)[:5]}")

    print(
        f"[merge] merged {merged} adapter pairs; dropped {len(lora_a) + len(lora_b)} adapter "
        f"tensors and {dropped_extra} adapter _extra_state entries"
    )
    save_file(base, str(dst / "model.safetensors"), metadata={"format": "pt"})

    # The merged model has no adapters; leaving the block in would make a loader
    # that honours it apply the delta a second time.
    config.pop("lora", None)
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    if (src / "llm_backbone").is_dir():
        shutil.copytree(src / "llm_backbone", dst / "llm_backbone", dirs_exist_ok=True)
    print(f"[merge] wrote {dst}")


if __name__ == "__main__":
    main()
