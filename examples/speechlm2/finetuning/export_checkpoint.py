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
"""Turn a training checkpoint into a merged HuggingFace directory that vLLM can serve.

    python export_checkpoint.py --exp-dir <exp> --step 420 --base-checkpoint <ckpt> --out <merged_dir> [--delete-ckpt]

1. symlinks ``step=420.ckpt`` (or ``step=420-last.ckpt``) to an ``=``-free name
   (Hydra cannot parse ``=`` inside an override value),
2. runs ``examples/speechlm2/to_hf.py`` with the run's resolved ``exp_config.yaml``,
3. if the run trained LoRA adapters: asserts that ``lora_B.weight`` is non-zero (it initializes to exactly
   zero, so a zero ``lora_B.weight`` means the trained weights never made it into the export), then merges
   ``W + (alpha/dim) * B @ A`` into the base weights (merge_lora_checkpoint.py, which also rejects
   unpaired or mis-shaped adapters and all-zero paired deltas), so no serving stack can silently drop them.
   A run without LoRA (``--lora-targets ""``) keeps the ordinary full-rank export,
4. restores ``pretrained_asr`` from the base checkpoint's config (vLLM refuses a config without it),
5. writes ``provenance.json`` (the run's training fingerprint, the step and the base checkpoint digest).

The output is built under ``<out>.unfinished`` and renamed into place only when complete. An existing complete
output is reused only if its provenance matches; a partial or mismatched one is an error.
Holds the same GPU lock as run_training.py.
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# A Jupyter kernel exports MPLBACKEND=module://matplotlib_inline...; child processes that import matplotlib (NeMo does,
# via torchmetrics) fail on it when that backend is not installed in their environment.
os.environ.pop("MPLBACKEND", None)

HERE = Path(__file__).resolve().parent
NEMO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from fingerprint import FingerprintMismatch, checkpoint_digest, differences  # noqa: E402
from merge_lora_checkpoint import LORA_WEIGHT_RE  # noqa: E402


def export_provenance(exp_dir: Path, step: int, base_checkpoint: Path) -> dict:
    fp = exp_dir / "fingerprint.json"
    return {
        "training": json.loads(fp.read_text()) if fp.exists() else None,
        "step": step,
        "base_checkpoint": checkpoint_digest(base_checkpoint),
    }


def check_existing(out: Path, expected: dict) -> bool:
    """True if `out` is a complete export with matching provenance; raise if it is partial or different."""
    if not out.exists():
        return False
    if not (out / "model.safetensors").is_file() or not (out / "provenance.json").is_file():
        raise SystemExit(f"{out} exists but is not a complete export (partial or older output); delete it first")
    diff = differences(json.loads((out / "provenance.json").read_text()), expected)
    if diff:
        raise FingerprintMismatch(f"{out} was exported from different inputs (differs in: {', '.join(diff)})")
    return True


def lora_keys(path: Path) -> list:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        return [k for k in f.keys() if LORA_WEIGHT_RE.match(k)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-dir", required=True, type=Path)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--base-checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--delete-ckpt", action="store_true", help="remove the 66 GB training checkpoint afterwards")
    a = p.parse_args()
    provenance = export_provenance(a.exp_dir, a.step, a.base_checkpoint)
    if check_existing(a.out, provenance):
        print(f"[export] {a.out} exists with matching provenance, skipping")
        return
    staging = a.out.parent / (a.out.name + ".unfinished")
    shutil.rmtree(staging, ignore_errors=True)
    ckdir = a.exp_dir / "checkpoints"
    src = next((ckdir / n for n in (f"step={a.step}.ckpt", f"step={a.step}-last.ckpt") if (ckdir / n).exists()), None)
    if src is None:
        sys.exit(f"no checkpoint for step {a.step} in {ckdir}")
    link = ckdir / f"export_step_{a.step}.ckpt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src.name)

    work = Path(os.environ.get("SALM_FT_WORK", "salm_ft_work")).absolute()
    work.mkdir(parents=True, exist_ok=True)
    with open(work / ".train.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = a.out.parent / (a.out.name + "_raw")
        subprocess.run(
            [
                sys.executable,
                "examples/speechlm2/to_hf.py",
                "class_path=nemo.collections.speechlm2.models.SALMAutomodel",
                f"ckpt_path={link}",
                f"ckpt_config={a.exp_dir / 'exp_config.yaml'}",
                f"output_dir={raw}",
            ],
            cwd=NEMO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        fcntl.flock(lock, fcntl.LOCK_UN)

    from safetensors import safe_open

    keys = lora_keys(raw / "model.safetensors")
    raw_cfg = json.loads((raw / "config.json").read_text())
    if keys:
        if not raw_cfg.get("lora"):
            sys.exit("the export has LoRA tensors but its config has no `lora` block: malformed adapter state")
        with safe_open(str(raw / "model.safetensors"), "pt") as f:
            b = [
                t for t in (f.get_tensor(k) for k in keys if LORA_WEIGHT_RE.match(k).group(2) == "B") if t.numel() > 0
            ]
        if not b:
            sys.exit("the export has lora_A tensors but no non-empty lora_B: malformed adapter state")
        m = max(t.abs().max().item() for t in b)
        print(f"[export] {len(b)} lora_B tensors, max |B| = {m:.3e}")
        if m == 0:
            sys.exit("lora_B is all zero: the export did not load the trained checkpoint")
        subprocess.run(
            [sys.executable, str(HERE / "merge_lora_checkpoint.py"), "--src", str(raw), "--dst", str(staging)],
            check=True,
        )
        shutil.rmtree(raw)
    else:
        if raw_cfg.get("lora"):
            sys.exit("the config declares LoRA adapters but the export has no LoRA tensors: malformed adapter state")
        print("[export] no LoRA adapters: keeping the full-rank export")
        raw.rename(staging)
    base_cfg = json.loads((a.base_checkpoint / "config.json").read_text())
    cfg_path = staging / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["pretrained_asr"] = base_cfg["pretrained_asr"]
    cfg_path.write_text(json.dumps(cfg, indent=2))
    (staging / "provenance.json").write_text(json.dumps(provenance, indent=1, sort_keys=True))
    if a.out.exists():
        sys.exit(f"{a.out} appeared while exporting; leaving the new export at {staging}")
    os.replace(staging, a.out)
    link.unlink()
    if a.delete_ckpt:  # both the checkpoint and its `-last` twin, if any (each ~66 GB)
        for name in (f"step={a.step}.ckpt", f"step={a.step}-last.ckpt"):
            shutil.rmtree(ckdir / name, ignore_errors=True)
    print(f"[export] wrote {a.out}")


if __name__ == "__main__":
    main()
