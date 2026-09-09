# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Convert a StreamingSTTModelAutomodel checkpoint to the HF-backend layout.

Motivation: inference runs through ``StreamingSTTModel`` (the HuggingFace-backed
class), so there is a single, well-tested streaming/decoding implementation. Only
the LLM backbone differs, and ``transformers`` and ``nemo_automodel`` ship separate
Nemotron implementations whose weight layouts disagree.

The two disagree in exactly four ways (derived by diffing our checkpoint against a
meta-device ``NemotronHForCausalLM.state_dict()``; 353 of 401 LLM tensors already
match by name AND shape):

  1. experts.gate_and_up_projs -> experts.up_proj, transposed
  2. experts.down_projs        -> experts.down_proj, transposed
     Automodel contracts as ``x @ W`` so stores (E, in, out); transformers uses
     ``F.linear(x, W)`` so stores (E, out, in). Despite the name there is no gate
     projection -- this model's activation is relu2, and intermediate_size == 1856
     matches the single projection.
  3. model.embed_tokens.weight -> embed_tokens.weight, hoisted to the top level,
     because StreamingSTTModel.__init__ moves the embedding out of the LLM.
  4. model.norm.weight -> model.norm_f.weight

LoRA gets merged into the base weights (W += (alpha/dim) * B @ A, matching
``nemo_automodel._peft.lora``: ``scale = alpha / dim`` and
``lora_B(lora_A(x) * scale)``) and the adapter tensors are then dropped. This is
required, not cosmetic: transformers has no LoRA submodules, so unmerged adapter
keys would be discarded as "unexpected" and the fine-tuning would vanish silently.
Checkpoints without LoRA convert fine -- the merge step is simply skipped.

Memory: tensors are streamed one at a time and the safetensors header is computed
up front from shapes, so peak usage is a few tensors rather than the whole model.

Usage:
    python examples/speechlm2/automodel_ckpt_to_hf_backend.py <src_hf_ckpt> <dst_dir>
"""

import json
import re
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ST_DTYPE = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}

# Automodel-only config entries; StreamingSTTModel ignores them with a warning, and
# `lora` in particular must go: the adapters are merged into the weights here, so
# leaving it would make the HF model build adapters that no checkpoint tensor fills.
AUTOMODEL_ONLY_KEYS = (
    "automodel_backend",
    "lora",
    "moe_metrics",
    "aux_loss_coeff",
    "train_gate",
    "init_configure_model",
)

LORA_RE = re.compile(r"^(?P<base>.*)\.lora_(?P<ab>A|B)\.weight$")


def target_key(key: str) -> tuple[str, bool]:
    """Map one source key to its HF-backend name. Returns (new_key, needs_transpose)."""
    if key == "llm.model.embed_tokens.weight":
        return "embed_tokens.weight", False  # hoisted out of the LLM by StreamingSTTModel
    if key == "llm.model.norm.weight":
        return "llm.model.norm_f.weight", False
    if key.endswith(".experts.gate_and_up_projs"):
        return key[: -len("gate_and_up_projs")] + "up_proj", True
    if key.endswith(".experts.down_projs"):
        return key[: -len("down_projs")] + "down_proj", True
    return key, False


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    st_in = src / "model.safetensors"

    cfg = json.loads((src / "config.json").read_text())
    lora_cfg = cfg.get("lora") or {}
    scale = (lora_cfg["alpha"] / lora_cfg["dim"]) if lora_cfg else None

    with safe_open(str(st_in), framework="pt") as f:
        src_keys = list(f.keys())
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in src_keys}

    # Group LoRA pairs by the base weight they modify.
    lora_pairs: dict[str, dict[str, str]] = {}
    for k in src_keys:
        m = LORA_RE.match(k)
        if m:
            lora_pairs.setdefault(m.group("base") + ".weight", {})[m.group("ab")] = k
    incomplete = {b: v for b, v in lora_pairs.items() if set(v) != {"A", "B"}}
    if incomplete:
        raise RuntimeError(f"LoRA pairs missing a side (cannot merge): {list(incomplete)[:5]}")
    if lora_pairs and scale is None:
        raise RuntimeError(
            "checkpoint has lora_* tensors but config.json has no 'lora' section, " "so the merge scale is unknown"
        )
    if lora_pairs:
        print(f"LoRA: {len(lora_pairs)} modules to merge (scale={scale})")
    else:
        print("LoRA: none present, skipping merge")

    # Keys we emit: drop TE extra state and the adapters we are folding in.
    emit = [k for k in src_keys if "_extra_state" not in k and not LORA_RE.match(k)]

    header, offset, plan = {}, 0, []
    for k in emit:
        new_k, transpose = target_key(k)
        shape = list(shapes[k])
        if transpose:
            shape[-2], shape[-1] = shape[-1], shape[-2]
        nbytes = 1
        for d in shape:
            nbytes *= d
        nbytes *= torch.bfloat16.itemsize
        header[new_k] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
        plan.append((k, new_k, transpose))
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)

    dst.mkdir(parents=True, exist_ok=True)
    print(f"writing {len(plan)} tensors ({offset / 2**30:.1f} GiB) -> {dst / 'model.safetensors'}")

    merged = 0
    with safe_open(str(st_in), framework="pt") as f, open(dst / "model.safetensors", "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for i, (k, new_k, transpose) in enumerate(plan):
            t = f.get_tensor(k)
            if k in lora_pairs:
                a = f.get_tensor(lora_pairs[k]["A"]).to(torch.float32)  # (dim, in)
                b = f.get_tensor(lora_pairs[k]["B"]).to(torch.float32)  # (out, dim)
                t = (t.to(torch.float32) + scale * (b @ a)).to(torch.bfloat16)
                merged += 1
            if transpose:
                t = t.transpose(-2, -1)
            if t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            out.write(t.contiguous().flatten().view(torch.uint8).numpy().tobytes())
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(plan)}", flush=True)

    if merged != len(lora_pairs):
        raise RuntimeError(f"merged {merged} LoRA modules but expected {len(lora_pairs)}")
    print(f"merged {merged} LoRA modules into base weights")

    # config.json for the HF backend.
    out_cfg = {k: v for k, v in cfg.items() if k not in AUTOMODEL_ONLY_KEYS}
    out_cfg["use_nemo_automodel"] = False
    out_cfg["load_llm_weights"] = False  # this checkpoint supplies them
    out_cfg["load_asr_weights"] = True  # perception architecture comes from the ASR .nemo
    (dst / "config.json").write_text(json.dumps(out_cfg, indent=2))
    print(f"wrote {dst / 'config.json'}")


if __name__ == "__main__":
    main()
