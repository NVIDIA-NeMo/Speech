# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run fingerprints: never reuse a cached result, or resume training, across a changed input.

Training runs, exports, averages and evaluations write a fingerprint next to their outputs: digests of the base
checkpoint, the manifests, the resolved config or decoder arguments, and the code revision. A later run with the same
output name reuses the output only if the fingerprint matches exactly, and otherwise refuses with the list of
differences. Without this, reusing a work directory after changing the base checkpoint or the recipe silently shows
stale metrics or resumes incompatible weights.

Large weight files are digested by sampling (size, safetensors header, and evenly spaced 1 MiB chunks), which catches
any realistic change of checkpoint at a fraction of the cost of hashing tens of GB.
"""
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

CHUNK = 1 << 20
SAMPLES = 16
FULL_HASH_LIMIT = 64 << 20
HERE = Path(__file__).resolve().parent
NEMO_ROOT = HERE.parents[2]


class FingerprintMismatch(RuntimeError):
    pass


def file_digest(path) -> str:
    """sha256 of a file; files larger than FULL_HASH_LIMIT are sampled (size + header + SAMPLES chunks)."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        if size <= FULL_HASH_LIMIT:
            for block in iter(lambda: f.read(CHUNK), b""):
                h.update(block)
            return h.hexdigest()
        if path.suffix == ".safetensors":
            (n,) = struct.unpack("<Q", f.read(8))
            h.update(f.read(n))  # tensor names, dtypes, shapes and offsets
        for i in range(SAMPLES):
            f.seek(max(0, (size - CHUNK) * i // (SAMPLES - 1)))
            h.update(f.read(CHUNK))
    return "sampled:" + h.hexdigest()


def checkpoint_digest(ckpt_dir) -> str:
    """Digest of a HuggingFace-format checkpoint directory: configs and tokenizer in full, weights sampled."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt_dir}")
    h = hashlib.sha256()
    for p in sorted(ckpt_dir.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and p.name not in ("fingerprint.json", "provenance.json"):
            h.update(str(p.relative_to(ckpt_dir)).encode())
            h.update(file_digest(p).encode())
    return h.hexdigest()


def code_revision() -> str:
    """Git commit of the NeMo checkout, plus a digest of uncommitted changes to the code these tools run."""
    try:
        head = subprocess.run(
            ["git", "-C", str(NEMO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", str(NEMO_ROOT), "diff", "HEAD", "--", "examples/speechlm2", "nemo/collections/speechlm2"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return head + ("+dirty:" + hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else "")
    except (OSError, subprocess.CalledProcessError):
        h = hashlib.sha256()
        for p in sorted(HERE.glob("*.py")):
            h.update(p.read_bytes())
        return "nogit:" + h.hexdigest()


def config_digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def differences(stored: dict, expected: dict) -> list:
    return sorted(k for k in set(stored) | set(expected) if stored.get(k) != expected.get(k))


def check_or_write(path, expected: dict, what: str) -> bool:
    """Compare `expected` with the fingerprint stored at `path`.

    Returns True if a matching fingerprint exists (the output may be reused) and False if none exists (it is written
    now). Raises FingerprintMismatch if a different fingerprint exists.
    """
    path = Path(path)
    if path.exists():
        stored = json.loads(path.read_text())
        diff = differences(stored, expected)
        if diff:
            raise FingerprintMismatch(
                f"{what} was produced from different inputs (differs in: {', '.join(diff)}). Use a new output name or "
                f"work directory, or delete {path.parent} to recompute it."
            )
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(expected, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return False
