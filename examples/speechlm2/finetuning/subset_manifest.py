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
"""Draw a random (optionally class-stratified) subset of a training manifest, for data-efficiency studies.

    python subset_manifest.py --in train_ft.json --out train_n10pc.json --per-class 10          # 10 examples per label
    python subset_manifest.py --in train_ft.json --out train_10pct.json --fraction 0.1           # 10% of the rows
    python subset_manifest.py --in train_ft.json --out train_300.json --count 300                # 300 rows

With --per-class, the class is the manifest's `text` field (classification targets); otherwise rows are sampled
uniformly. The output is shuffled, so it is safe for Lhotse's bounded shuffle buffer.
"""
import argparse
import collections
import json
import random


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--per-class", type=int)
    g.add_argument("--fraction", type=float)
    g.add_argument("--count", type=int)
    p.add_argument("--label-field", default="text")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.per_class is not None and a.per_class <= 0:
        p.error("--per-class must be a positive integer")
    if a.count is not None and a.count <= 0:
        p.error("--count must be a positive integer")
    if a.fraction is not None and not 0 < a.fraction <= 1:
        p.error("--fraction must be in (0, 1]")
    rows = [json.loads(line) for line in open(a.inp) if line.strip()]
    rng = random.Random(a.seed)
    if a.per_class is not None:
        by = collections.defaultdict(list)
        for r in rows:
            by[r[a.label_field]].append(r)
        out = [r for v in by.values() for r in rng.sample(v, min(a.per_class, len(v)))]
    else:
        n = a.count if a.count is not None else max(1, round(a.fraction * len(rows)))
        out = rng.sample(rows, min(n, len(rows)))
    rng.shuffle(out)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(float(r.get("duration", 0)) for r in out) / 3600
    print(f"wrote {a.out}: {len(out)} of {len(rows)} rows, {hours:.2f} h")


if __name__ == "__main__":
    main()
