#!/usr/bin/env python3
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

"""
Analyze predicted word latency vs ground-truth word alignments.

For each cut found in both manifests, align the predicted word sequence
(`pred_alignments`) against the ground-truth word sequence (`alignments`)
using a Levenshtein-style word matcher (same as WER scoring). For every
matched (text-equal) word pair, compute:

    latency = pred_end_time - gt_end_time

This is the model's per-word commit-latency: how long after the word
actually ended did the model emit it as text. Positive = late commit
(typical for streaming with chunk-aligned emits); negative = early commit
(possible if the aux/LM head fires before the word ends).

Usage:
    python scripts/speechlm2/analyze_word_latency.py \\
        --pred /path/to/pred_generations.jsonl \\
        --gt   /path/to/dev_clean_cleaned-aligned.json

    # Also write per-word records to a CSV for downstream analysis:
    python scripts/speechlm2/analyze_word_latency.py \\
        --pred pred.jsonl --gt gt.json --out-csv latencies.csv

Both manifests are JSONL with one record per line:
- Pred: requires `id` (or `audio_filepath`) and `pred_alignments`
  (list of {text, start_time, end_time}).
- GT:   requires `audio_filepath` (or `id`) and `alignments`
  (list of {text, start_time, end_time}).
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------------- #


def _cut_id(record: dict) -> Optional[str]:
    """Resolve a stable cut identifier from a manifest record."""
    if "id" in record and record["id"]:
        return str(record["id"])
    if "audio_filepath" in record and record["audio_filepath"]:
        return Path(record["audio_filepath"]).stem
    return None


def load_manifest(path: str) -> dict[str, dict]:
    """Load a JSONL manifest, keyed by cut id (audio file stem)."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = _cut_id(d)
            if cid is None:
                continue
            out[cid] = d
    return out


# --------------------------------------------------------------------------- #
# Word-level alignment
# --------------------------------------------------------------------------- #

_NORMALIZE_RE = re.compile(r"[^a-z0-9']")


def normalize_word(w: str) -> str:
    """Lowercase + strip non-alphanumeric (keep apostrophes) for matching."""
    return _NORMALIZE_RE.sub("", w.lower())


def align_words(
    pred_words: list[dict],
    gt_words: list[dict],
    case_insensitive: bool = True,
) -> list[tuple[int, int]]:
    """Match pred ↔ GT word sequences via difflib SequenceMatcher.

    Returns list of (pred_idx, gt_idx) for matched (text-equal) word pairs.
    Substitutions, insertions, and deletions are skipped — only correctly
    transcribed words contribute to the latency analysis.
    """
    if case_insensitive:
        pred_keys = [normalize_word(w["text"]) for w in pred_words]
        gt_keys = [normalize_word(w["text"]) for w in gt_words]
    else:
        pred_keys = [w["text"] for w in pred_words]
        gt_keys = [w["text"] for w in gt_words]
    sm = difflib.SequenceMatcher(a=pred_keys, b=gt_keys, autojunk=False)
    matches: list[tuple[int, int]] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                matches.append((i, j))
    return matches


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": sum(values) / n,
        "median": s[n // 2],
        "stdev": statistics.stdev(values) if n > 1 else 0.0,
        "p10": s[n // 10],
        "p25": s[n // 4],
        "p75": s[(3 * n) // 4],
        "p90": s[(9 * n) // 10],
        "p99": s[min(n - 1, (99 * n) // 100)],
        "min": s[0],
        "max": s[-1],
    }


def print_summary(stats: dict[str, float], label: str = "Latency") -> None:
    n = stats.get("n", 0)
    if n == 0:
        print(f"{label}: no matched pairs")
        return
    print(f"=== {label} (pred_end_time - gt_end_time) ===")
    print(f"  matched pairs: {n}")
    print(f"  mean:          {stats['mean']:+.3f} s   ({stats['mean'] * 1000:+.1f} ms)")
    print(f"  median:        {stats['median']:+.3f} s   ({stats['median'] * 1000:+.1f} ms)")
    print(f"  stdev:         {stats['stdev']:.3f} s")
    print(f"  p10 / p25:     {stats['p10']:+.3f}  /  {stats['p25']:+.3f} s")
    print(f"  p75 / p90:     {stats['p75']:+.3f}  /  {stats['p90']:+.3f} s")
    print(f"  p99:           {stats['p99']:+.3f} s")
    print(f"  min / max:     {stats['min']:+.3f}  /  {stats['max']:+.3f} s")


def print_histogram(values: list[float], bin_size_s: float = 0.08, window_s: float = 0.8) -> None:
    """Print a fixed-width histogram of latencies, binned at frame granularity."""
    if not values:
        return
    bins: Counter[float] = Counter()
    for v in values:
        b = round(v / bin_size_s) * bin_size_s
        bins[b] += 1
    n_total = sum(bins.values())
    bar_max = max(bins.values()) if bins else 1
    print(f"=== Latency histogram ({int(bin_size_s * 1000)}ms bins, ±{int(window_s * 1000)}ms window) ===")
    n_steps = int(round(window_s / bin_size_s))
    for k in range(-n_steps, n_steps + 1):
        b = round(k * bin_size_s, 4)
        n = bins.get(b, 0)
        bar = "█" * max(0, int(60 * n / bar_max))
        pct = 100 * n / n_total
        print(f"  {b:+.2f}s: {n:>6} ({pct:>5.1f}%) {bar}")
    tail_neg = sum(n for b, n in bins.items() if b < -window_s - bin_size_s / 2)
    tail_pos = sum(n for b, n in bins.items() if b > window_s + bin_size_s / 2)
    if tail_neg or tail_pos:
        print(f"  ... ({tail_neg} more < -{window_s:.2f}s, " f"{tail_pos} more > +{window_s:.2f}s)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Word-level commit latency analysis: pred vs GT alignments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pred", required=True, help="Pred manifest JSONL (with pred_alignments).")
    ap.add_argument("--gt", required=True, help="GT manifest JSONL (with alignments).")
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to write per-word records (id, pred_text, gt_text, pred_end, gt_end, latency).",
    )
    ap.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Disable case-insensitive normalization for word matching.",
    )
    ap.add_argument(
        "--bin-size",
        type=float,
        default=0.08,
        help="Histogram bin size in seconds (default: 0.08s = one 80ms frame).",
    )
    ap.add_argument(
        "--hist-window",
        type=float,
        default=0.8,
        help="Histogram half-width in seconds (default: 0.8s; tail counts shown for values beyond).",
    )
    ap.add_argument(
        "--max-abs-latency",
        type=float,
        default=None,
        help=(
            "Filter out matched pairs with |latency| > this many seconds before "
            "computing statistics. Useful for excluding word-alignment outliers "
            "(e.g. the SequenceMatcher pairing the same repeated word with the "
            "wrong GT occurrence). Default: no filtering."
        ),
    )
    args = ap.parse_args()

    pred = load_manifest(args.pred)
    gt = load_manifest(args.gt)

    print(f"Loaded {len(pred)} pred entries, {len(gt)} GT entries")
    common_ids = sorted(set(pred.keys()) & set(gt.keys()))
    print(f"Common IDs: {len(common_ids)}")
    print(f"Pred-only IDs: {len(set(pred.keys()) - set(gt.keys()))}")
    print(f"GT-only IDs:   {len(set(gt.keys()) - set(pred.keys()))}")
    print()

    all_latencies: list[float] = []
    per_word_rows: list[dict] = []
    per_cut_means: list[float] = []
    n_pred_missing = 0
    n_gt_missing = 0
    n_no_match = 0
    n_pred_words_total = 0
    n_gt_words_total = 0
    n_matched_total = 0

    for cid in common_ids:
        pred_align = pred[cid].get("pred_alignments")
        gt_align = gt[cid].get("alignments")
        if not pred_align:
            n_pred_missing += 1
            continue
        if not gt_align:
            n_gt_missing += 1
            continue
        n_pred_words_total += len(pred_align)
        n_gt_words_total += len(gt_align)
        matches = align_words(pred_align, gt_align, case_insensitive=not args.case_sensitive)
        if not matches:
            n_no_match += 1
            continue
        n_matched_total += len(matches)
        cut_lats: list[float] = []
        for pi, gi in matches:
            lat = float(pred_align[pi]["end_time"]) - float(gt_align[gi]["end_time"])
            cut_lats.append(lat)
            per_word_rows.append(
                {
                    "id": cid,
                    "pred_text": pred_align[pi]["text"],
                    "gt_text": gt_align[gi]["text"],
                    "pred_start": float(pred_align[pi]["start_time"]),
                    "pred_end": float(pred_align[pi]["end_time"]),
                    "gt_start": float(gt_align[gi]["start_time"]),
                    "gt_end": float(gt_align[gi]["end_time"]),
                    "latency": lat,
                }
            )
        all_latencies.extend(cut_lats)
        per_cut_means.append(sum(cut_lats) / len(cut_lats))

    # Coverage diagnostics.
    print("=== Coverage ===")
    print(f"  cuts skipped (no pred_alignments): {n_pred_missing}")
    print(f"  cuts skipped (no GT alignments):   {n_gt_missing}")
    print(f"  cuts skipped (no word match):      {n_no_match}")
    print(f"  total pred words: {n_pred_words_total}")
    print(f"  total GT words:   {n_gt_words_total}")
    print(
        f"  matched pairs:    {n_matched_total} "
        f"({n_matched_total / max(1, n_pred_words_total):.1%} of pred, "
        f"{n_matched_total / max(1, n_gt_words_total):.1%} of GT)"
    )

    # Optional outlier filter: drop matched pairs where the latency magnitude
    # exceeds the threshold (almost always misalignments where the same word
    # appears multiple times in the utterance and SequenceMatcher paired the
    # wrong instances). Recompute per-cut means from the filtered set so they
    # also exclude outliers.
    if args.max_abs_latency is not None:
        thr = args.max_abs_latency
        kept_rows = [r for r in per_word_rows if abs(r["latency"]) <= thr]
        n_filtered = len(per_word_rows) - len(kept_rows)
        all_latencies = [r["latency"] for r in kept_rows]
        per_word_rows = kept_rows
        # Recompute per-cut means using only kept rows.
        per_cut_buckets: dict[str, list[float]] = {}
        for r in kept_rows:
            per_cut_buckets.setdefault(r["id"], []).append(r["latency"])
        per_cut_means = [sum(v) / len(v) for v in per_cut_buckets.values() if v]
        print(
            f"  outlier filter |lat| > {thr:.2f}s: dropped {n_filtered} pairs "
            f"({n_filtered / max(1, n_matched_total):.1%} of matched)"
        )
    print()

    print_summary(summarize(all_latencies), label="Word commit latency")
    print()
    print_histogram(all_latencies, bin_size_s=args.bin_size, window_s=args.hist_window)
    print()
    # Per-cut mean latency (averages out fluctuations within a single utterance).
    print_summary(summarize(per_cut_means), label="Per-cut MEAN latency")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "pred_text",
                    "gt_text",
                    "pred_start",
                    "pred_end",
                    "gt_start",
                    "gt_end",
                    "latency",
                ],
            )
            writer.writeheader()
            writer.writerows(per_word_rows)
        print(f"\nPer-word records written to: {out_path}")


if __name__ == "__main__":
    main()
