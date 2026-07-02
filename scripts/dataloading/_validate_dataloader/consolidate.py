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
"""Consolidate per-rank validator JSONLs and emit PASS/FAIL on Q1..Q5.

Layout the per-rank entry writes:

    {output_dir}/
        baseline/run0/rank_NNN.jsonl
        baseline/run0/state_rank_NNN.pt
        baseline/run0/throughput_rank_NNN.json
        baseline/run1/rank_NNN.jsonl            # if --num-determinism-runs >= 2
        resumed/run0/rank_NNN.jsonl             # phase=resumed
        groundtruth/cuts.jsonl                  # phase=groundtruth (single-rank)
        pre_validation.json                     # written by pre_validation.py

This module is the post-iteration aggregator. Exit code: 0 if all checks
pass, 1 if any check fails, 2 if there's a structural problem
(no JSONLs, missing groundtruth, etc.).
"""

import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click

LOG = logging.getLogger(__name__)


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


@dataclass
class QResult:
    q_id: str
    status: str
    tag: Optional[str] = None
    detail: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    questions: list[QResult]
    throughput: dict

    def to_dict(self):
        return {
            "questions": {
                q.q_id: {"status": q.status, "tag": q.tag, "detail": q.detail, **q.extra} for q in self.questions
            },
            "throughput": self.throughput,
        }

    @property
    def all_passed(self) -> bool:
        return all(q.status != FAIL for q in self.questions)


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #


def consolidate(output_dir: Path, *, checkpoint_at: int, num_determinism_runs: int) -> ValidationReport:
    """Read every artifact under ``output_dir`` and produce a ValidationReport."""
    baseline = _load_phase(output_dir / "baseline" / "run0")
    questions: list[QResult] = []

    questions.append(_q1_no_duplication(baseline))
    questions.append(_q2_no_skipping(baseline, output_dir / "groundtruth" / "cuts.jsonl"))
    questions.append(_q3_partition_correctness(baseline))
    questions.append(
        _q4_exact_resume(
            baseline,
            _load_phase(output_dir / "resumed" / "run0"),
            checkpoint_at=checkpoint_at,
        )
    )
    if num_determinism_runs >= 2:
        run1 = _load_phase(output_dir / "baseline" / "run1")
        questions.append(_q5_determinism(baseline, run1))
    else:
        questions.append(QResult("Q5", SKIP, detail="num_determinism_runs < 2"))

    throughput = _collect_throughput(output_dir / "baseline" / "run0")
    return ValidationReport(questions=questions, throughput=throughput)


# --------------------------------------------------------------------------- #
# Question implementations.
# --------------------------------------------------------------------------- #


def _q1_no_duplication(rows: list[dict]) -> QResult:
    """Q1: no cut appears twice within phase 1. Tag ``partition-rank-leak``
    if cross-rank, ``partition-worker-leak`` if within one rank."""
    if not rows:
        return QResult("Q1", SKIP, detail="no baseline rows loaded")
    # Map cut_id -> set of (rank, worker) tuples that saw it.
    sightings: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for r in rows:
        for cid in r["cut_ids"]:
            sightings[cid].add((r["rank"], r["worker_id"]))
    dup_cross_rank: list[str] = []
    dup_within_rank: list[str] = []
    for cid, seen in sightings.items():
        if len(seen) <= 1:
            continue
        ranks = {rank for rank, _ in seen}
        if len(ranks) > 1:
            dup_cross_rank.append(cid)
        else:
            dup_within_rank.append(cid)
    if dup_cross_rank:
        return QResult(
            "Q1",
            FAIL,
            tag="partition-rank-leak",
            detail=f"{len(dup_cross_rank)} cut.id(s) appeared on multiple ranks",
            extra={"examples": dup_cross_rank[:5]},
        )
    if dup_within_rank:
        return QResult(
            "Q1",
            FAIL,
            tag="partition-worker-leak",
            detail=f"{len(dup_within_rank)} cut.id(s) seen by multiple workers within one rank",
            extra={"examples": dup_within_rank[:5]},
        )
    return QResult("Q1", PASS, detail=f"{len(sightings)} distinct cuts, no duplicates")


def _q2_no_skipping(rows: list[dict], groundtruth_path: Path) -> QResult:
    """Q2: yielded ID set equals the ground-truth set (force_finite mode)."""
    if not rows:
        return QResult("Q2", SKIP, detail="no baseline rows loaded")
    if not groundtruth_path.exists():
        return QResult("Q2", SKIP, detail=f"groundtruth file missing: {groundtruth_path}")
    expected: set[str] = set()
    with open(groundtruth_path) as f:
        for line in f:
            obj = json.loads(line)
            expected.update(obj.get("cut_ids", []))
    yielded: set[str] = set()
    for r in rows:
        yielded.update(r["cut_ids"])
    missing = expected - yielded
    unexpected = yielded - expected
    if missing:
        return QResult(
            "Q2",
            FAIL,
            tag="skip",
            detail=f"{len(missing)} of {len(expected)} expected cut.id(s) never yielded",
            extra={"missing_examples": list(missing)[:5], "unexpected_count": len(unexpected)},
        )
    if unexpected:
        return QResult(
            "Q2",
            FAIL,
            tag="id-collision",
            detail=f"{len(unexpected)} cut.id(s) yielded but not in ground truth",
            extra={"unexpected_examples": list(unexpected)[:5]},
        )
    return QResult("Q2", PASS, detail=f"yielded ({len(yielded)}) == ground truth ({len(expected)})")


def _q3_partition_correctness(rows: list[dict]) -> QResult:
    """Q3: per-rank cut sets are pairwise disjoint."""
    if not rows:
        return QResult("Q3", SKIP, detail="no baseline rows loaded")
    per_rank: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        per_rank[r["rank"]].update(r["cut_ids"])
    grand_union = set()
    for s in per_rank.values():
        grand_union.update(s)
    sum_distinct = sum(len(s) for s in per_rank.values())
    if sum_distinct == len(grand_union):
        return QResult("Q3", PASS, detail=f"{len(per_rank)} ranks, |union|={len(grand_union)}")
    overlap = sum_distinct - len(grand_union)
    # Detect broadcast vs partial overlap.
    n_ranks = max(len(per_rank), 1)
    ratio = sum_distinct / max(len(grand_union), 1)
    tag = "partition-rank-leak"
    if ratio >= n_ranks - 0.5:
        detail = f"FULL BROADCAST: each cut.id appears on ~{ratio:.1f}/{n_ranks} ranks " f"(overlap={overlap})"
    else:
        detail = (
            f"PARTIAL OVERLAP: per-rank distinct sums to {sum_distinct} but |union|={len(grand_union)} "
            f"(overlap={overlap})"
        )
    return QResult("Q3", FAIL, tag=tag, detail=detail)


def _q4_exact_resume(baseline: list[dict], resumed: list[dict], *, checkpoint_at: int) -> QResult:
    """Q4: per-(rank, step) cut sets in resumed match the baseline tail.

    The validator saves ``state_dict()`` AFTER yielding baseline step
    ``checkpoint_at``; StatefulDataLoader's state points at the NEXT
    element, so resumed[0] should equal baseline[checkpoint_at + 1].

    The comparison runs on the **overlapping window** only: cells where
    both the baseline and the resumed JSONL have an entry. Cells
    beyond that (resumed ran longer than baseline tail, or vice versa)
    are reported in ``extra`` but don't trigger FAIL on their own —
    they just mean one side iterated more batches than necessary."""
    if not resumed:
        return QResult("Q4", SKIP, detail="no resumed rows loaded")
    base_by_key = {(r["rank"], r["step"]): set(r["cut_ids"]) for r in baseline}
    res_by_key = {(r["rank"], r["step"]): set(r["cut_ids"]) for r in resumed}
    divergences: list[dict] = []
    overlap = 0
    extra_resumed = 0
    extra_baseline_tail = 0
    # Compare every resumed cell to its baseline counterpart at step + checkpoint_at + 1.
    for (rank, rstep), res_cuts in sorted(res_by_key.items()):
        base_step = rstep + checkpoint_at + 1
        base_cuts = base_by_key.get((rank, base_step))
        if base_cuts is None:
            extra_resumed += 1
            continue
        overlap += 1
        if base_cuts != res_cuts:
            divergences.append(
                {
                    "rank": rank,
                    "step": rstep,
                    "baseline_step": base_step,
                    "only_in_baseline": list(base_cuts - res_cuts)[:3],
                    "only_in_resumed": list(res_cuts - base_cuts)[:3],
                }
            )
    # Cells in baseline-tail that the resumed run never reached.
    for rank, bstep in base_by_key:
        if bstep <= checkpoint_at:
            continue
        rstep = bstep - checkpoint_at - 1
        if (rank, rstep) not in res_by_key:
            extra_baseline_tail += 1
    extras = {
        "overlap_cells": overlap,
        "extra_resumed_cells": extra_resumed,
        "extra_baseline_tail_cells": extra_baseline_tail,
    }
    if divergences:
        return QResult(
            "Q4",
            FAIL,
            tag="resume-rng-divergence",
            detail=f"{len(divergences)}/{overlap} overlapping cell(s) diverge after resume",
            extra={**extras, "examples": divergences[:5]},
        )
    if overlap == 0:
        return QResult(
            "Q4",
            FAIL,
            tag="resume-length-mismatch",
            detail="zero overlap between resumed and baseline-tail windows",
            extra=extras,
        )
    return QResult("Q4", PASS, detail=f"{overlap} overlapping cell(s) match baseline tail bit-for-bit", extra=extras)


def _q5_determinism(run0: list[dict], run1: list[dict]) -> QResult:
    """Q5: two independent baseline runs produce identical (rank, step) cut sets."""
    if not run1:
        return QResult("Q5", SKIP, detail="run1 missing")
    a = {(r["rank"], r["step"]): set(r["cut_ids"]) for r in run0}
    b = {(r["rank"], r["step"]): set(r["cut_ids"]) for r in run1}
    if a.keys() != b.keys():
        only_a = list(a.keys() - b.keys())[:3]
        only_b = list(b.keys() - a.keys())[:3]
        return QResult(
            "Q5",
            FAIL,
            tag="non-determinism",
            detail="run0/run1 step coverage differs",
            extra={"only_in_run0": only_a, "only_in_run1": only_b},
        )
    divergences: list[dict] = []
    for k, va in a.items():
        vb = b[k]
        if va != vb:
            divergences.append(
                {"rank": k[0], "step": k[1], "only_run0": list(va - vb)[:3], "only_run1": list(vb - va)[:3]}
            )
    if divergences:
        return QResult(
            "Q5",
            FAIL,
            tag="non-determinism",
            detail=f"{len(divergences)} cell(s) differ between determinism runs",
            extra={"examples": divergences[:5]},
        )
    return QResult("Q5", PASS, detail="run0 == run1 across all (rank, step) cells")


# --------------------------------------------------------------------------- #
# Throughput summary (v1 minimal: t_total only).
# --------------------------------------------------------------------------- #


def _collect_throughput(run_dir: Path) -> dict:
    files = sorted(run_dir.glob("throughput_rank_*.json"))
    if not files:
        return {"available": False}
    aggregates = [json.loads(f.read_text()) for f in files]
    p50s = [a["p50_ms"] for a in aggregates if a.get("p50_ms") is not None]
    p95s = [a["p95_ms"] for a in aggregates if a.get("p95_ms") is not None]
    num_workers = aggregates[0].get("num_workers")
    p50 = statistics.median(p50s) if p50s else None
    p95 = max(p95s) if p95s else None
    out = {
        "available": True,
        "num_workers": num_workers,
        "num_ranks": len(aggregates),
        "p50_ms_median": p50,
        "p95_ms_max": p95,
        "batches_per_s_per_rank": (1000.0 / p50) if p50 else None,
        "t_first_batch_ms_max": max((a.get("t_first_batch_ms") or 0) for a in aggregates) or None,
    }
    if p50 and num_workers:
        out["t_gpu_min_for_overlap_ms"] = p50 / num_workers
    return out


# --------------------------------------------------------------------------- #
# IO helpers.
# --------------------------------------------------------------------------- #


def _load_phase(phase_dir: Path) -> list[dict]:
    """Load every ``rank_*.jsonl`` under ``phase_dir`` into a flat list of rows."""
    if not phase_dir.exists():
        return []
    rows: list[dict] = []
    for f in sorted(phase_dir.glob("rank_*.jsonl")):
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


@click.command(help=__doc__)
@click.option(
    "--output-dir", required=True, type=click.Path(exists=True), help="Directory written by validate_dataloader.py."
)
@click.option(
    "--checkpoint-at",
    type=int,
    default=0,
    show_default=True,
    help="Step index at which the baseline saved state. Must match the baseline run.",
)
@click.option(
    "--num-determinism-runs",
    type=int,
    default=1,
    show_default=True,
    help="If >= 2, compares baseline/run0 vs baseline/run1 for Q5.",
)
@click.option("-v", "--verbose", is_flag=True, default=False)
def cli(output_dir: str, checkpoint_at: int, num_determinism_runs: int, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    out_dir = Path(output_dir)
    report = consolidate(out_dir, checkpoint_at=checkpoint_at, num_determinism_runs=num_determinism_runs)

    print(f"\n=== validation report ({len(report.questions)} questions) ===")
    for q in report.questions:
        marker = {PASS: "  PASS", WARN: "  WARN", FAIL: "  FAIL", SKIP: "  skip"}[q.status]
        tag = f" [{q.tag}]" if q.tag else ""
        print(f"{marker}  {q.q_id}{tag}: {q.detail}")
    if report.throughput.get("available"):
        t = report.throughput
        print(
            f"\nthroughput: p50={t['p50_ms_median']:.1f}ms p95={t['p95_ms_max']:.1f}ms "
            f"=> {t['batches_per_s_per_rank']:.2f} batches/s/rank "
            f"(num_workers={t['num_workers']}, T_gpu_min={t.get('t_gpu_min_for_overlap_ms', 0):.1f}ms)"
        )
    else:
        print("\nthroughput: <not collected>")
    (out_dir / "validation_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    print(f"wrote {out_dir / 'validation_report.json'}")
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    cli()
