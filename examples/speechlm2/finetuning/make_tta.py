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
"""Write speed-perturbed copies of an evaluation set for test-time augmentation.

Decoding the same model on slightly resampled audio yields hypotheses whose
errors are less correlated than those of two checkpoints from one training run,
which is what makes them useful to a voting combiner. Speed perturbation is the
mildest transform that reliably moves the encoder's frame alignment without
changing what was said.

Each row is read exactly as a decoder would read it: the ``offset``/``duration`` segment of its audio file, not the
whole file. The written copy holds only that segment, so the output row has its duration rescaled and no offset.
Durations in the manifest are rescaled to match, so token-budget accounting and RTFx stay correct.
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-manifest", required=True)
    p.add_argument("--factor", type=float, required=True, help="e.g. 0.95 = slower, 1.05 = faster")
    return p.parse_args()


def read_segment(row):
    """The audio of a manifest row: the `offset`/`duration` segment if given, else the whole file (mono float32)."""
    import soundfile as sf

    offset = float(row.get("offset") or 0.0)
    duration = row.get("duration")
    with sf.SoundFile(row["audio_filepath"]) as fh:
        sr, total = fh.samplerate, fh.frames
        start = int(round(offset * sr))
        frames = total - start if duration is None else int(round(float(duration) * sr))
        if offset < 0 or start >= total or frames <= 0 or start + frames > total + 1:
            raise ValueError(
                f"{row['audio_filepath']}: segment offset={offset} duration={duration} is outside the file "
                f"({total / sr:.3f} s)"
            )
        fh.seek(start)
        x = fh.read(frames=min(frames, total - start), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def main():
    args = parse_args()
    import numpy as np
    import soundfile as sf

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.manifest) as fi, open(args.out_manifest, "w") as fo:
        for i, line in enumerate(fi):
            if not line.strip():
                continue
            row = json.loads(line)
            x, sr = read_segment(row)
            # Resample by linear interpolation: playing the same samples at a
            # different rate changes both speed and pitch, which is exactly the
            # classic 3-way speed perturbation used in ASR training.
            m = max(1, int(round(len(x) / args.factor)))
            y = np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x).astype("float32")
            path = out_dir / f"tta_{i:06d}.flac"
            sf.write(str(path), y, sr, format="FLAC")
            r = dict(row)
            r["audio_filepath"] = str(path)
            r["duration"] = round(len(y) / sr, 3)
            r.pop("offset", None)  # the copy holds only the segment
            fo.write(json.dumps(r) + "\n")
            n += 1
    print(f"[tta x{args.factor}] {n} utts -> {args.out_manifest}")


if __name__ == "__main__":
    main()
