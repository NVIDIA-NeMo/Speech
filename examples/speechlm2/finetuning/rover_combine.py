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
"""Combine hypotheses from several ASR systems by voting (ROVER).

Weight averaging only works for checkpoints from the same run, which sit in the
same loss basin. Independent runs do not, so their *outputs* are combined
instead: align the hypotheses to each other and take a per-position majority.

Uncorrelated errors cancel — if two of three systems get a word right, the vote
recovers it — so the combination can beat every input system. That only holds
when the systems are genuinely diverse; combining near-identical systems does
nothing.

Alignment is incremental: hypothesis 1 becomes the initial word transition
network, each further hypothesis is aligned to it with Levenshtein; matched
positions accumulate votes, and insertions are voted on per gap between base
words. Ties fall back to the first (best) system, so the result never
degenerates below it by accident.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--hyps",
        nargs="+",
        required=True,
        help="vllm_task_eval.py outputs for the same manifest, best system first (used for tie-breaks)",
    )
    p.add_argument("--out", required=True, help="combined jsonl; its metric summary goes to <out>.summary.json")
    p.add_argument(
        "--normalizer",
        default="simple",
        choices=["simple", "english", "basic", "none"],
        help="applied before voting and scoring (same names as task_metrics.py)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="comma-separated vote weight per system, same order as --hyps. "
        "Unweighted voting lets several weak systems outvote a strong one; "
        "weighting by quality keeps the strong system's word unless the others "
        "agree decisively against it.",
    )
    return p.parse_args()


def align(ref, hyp):
    """Levenshtein backtrace: list of (ref_idx|None, hyp_idx|None) pairs."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def _pick(counter, base_choice):
    """Highest vote mass; ties go to the base system's choice, then to the lexicographically smallest option."""
    best = max(counter.values())
    tied = [k for k, v in counter.items() if v == best]
    return base_choice if base_choice in tied else min(tied)


def rover(hyps, weights=None):
    """hyps: list of token lists, best first. weights: per-system vote mass.

    Every word of the base (first) hypothesis is a slot; each other system votes for the word it aligns there, or
    for deleting it. Every gap between slots (and before the first / after the last) is also voted on: each system
    votes for the word sequence it inserts there, or for no insertion (the base always votes for none). A majority
    insertion is kept; a tie keeps the base system's choice, so the result is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(hyps)
    base = hyps[0]
    slots = [Counter({w: weights[0]}) for w in base]  # votes aligned to the base words
    gaps = [Counter({(): weights[0]}) for _ in range(len(base) + 1)]  # votes on insertions before slot g
    for w_sys, h in zip(weights[1:], hyps[1:]):
        inserted = [[] for _ in range(len(base) + 1)]
        gap = 0
        for ri, hi in align(base, h):
            if ri is None:
                inserted[gap].append(h[hi])
                continue
            slots[ri][h[hi] if hi is not None else ""] += w_sys  # "" = this system deleted the word
            gap = ri + 1
        for g, words in enumerate(inserted):
            gaps[g][tuple(words)] += w_sys
    out = list(_pick(gaps[0], ()))
    for i, c in enumerate(slots):
        w = _pick(c, base[i])
        if w:
            out.append(w)
        out.extend(_pick(gaps[i + 1], ()))
    return out


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from task_metrics import score_wer, wer_normalizers

    systems = []
    for path in args.hyps:
        with open(path) as f:
            systems.append([json.loads(line) for line in f])
    n = len(systems[0])
    if any(len(s) != n for s in systems):
        raise SystemExit(f"hypothesis files differ in length: {[len(s) for s in systems]}")
    for s in systems[1:]:
        if [r["text"] for r in s] != [r["text"] for r in systems[0]]:
            raise SystemExit("hypothesis files are not aligned row by row (different references)")
    norm = wer_normalizers()[args.normalizer]
    weights = None
    if args.weights:
        weights = [float(x) for x in args.weights.split(",")]
        if len(weights) != len(systems):
            raise SystemExit(f"--weights has {len(weights)} entries for {len(systems)} systems")

    rows = []
    for i in range(n):
        cand = [norm(s[i]["pred_text"]).split() for s in systems]
        rows.append({**systems[0][i], "pred_text": " ".join(rover(cand, weights))})
    with open(args.out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    res = score_wer(rows, normalizer=args.normalizer)
    res.update(systems=args.hyps, weights=weights, num_rows=n)
    res["inputs"] = {p: score_wer(s, normalizer=args.normalizer)["score"] for p, s in zip(args.hyps, systems)}
    Path(f"{args.out}.summary.json").write_text(json.dumps(res, indent=1))
    print(f"[rover] {len(systems)} systems, {n} rows -> WER {100 * res['score']:.2f}%")
    for p, w in res["inputs"].items():
        print(f"[rover]   input {Path(p).name:<40} {100 * w:.2f}%")


if __name__ == "__main__":
    main()
