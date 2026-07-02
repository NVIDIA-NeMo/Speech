#!/usr/bin/env python
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
"""Analyze resumable Lhotse dataloader progress stored in a checkpoint.

This tool answers two operational questions for indexed/resumable
training runs:

* how far each leaf dataset in the blend has advanced, expressed as total
  utilization (for example ``70%`` or ``1389%``), completed epochs, and current
  in-progress epoch percentage;
* how the observed consumed-item share compares with the desired blend weight,
  which surfaces datasets that were over- or under-sampled by the checkpoint.

Expected inputs
---------------
Use ``--checkpoint`` for a checkpoint file, a checkpoint directory, or an
``eval-step-N`` directory. For FSDP/DCP checkpoints, the script first looks for
metadata-only ``meta.pt`` files and expects NeMo's per-rank
``train_dataloader_per_rank`` payload. Use ``--allow-full-ckpt-load`` only when
metadata is unavailable and loading a non-meta checkpoint is acceptable.

Pass ``--config`` when available so the script can resolve the training blend,
recover desired blend weights, dataset names, and count indexed examples from
``.idx`` sidecars. Use ``--indexes-root`` when the sidecars live in a mirrored
index tree instead of next to the manifests/tars. ``--state-json`` is a debugging
escape hatch for analyzing an already-extracted payload without importing torch.

Outputs
-------
By default the script prints a Markdown table. ``--output-dir`` writes
``summary.json``, ``summary.md``, and ``summary.csv``; the JSON also includes the
raw leaf progress states and resolved dataset specs for follow-up debugging.

When to use it
--------------
Run it after a resumable training checkpoint is produced, before continuing a
suspicious chain, or during a dataloader postmortem when blend utilization looks
wrong. It is read-only: it never modifies checkpoints, indexes, or configs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


try:
    import yaml
except ImportError as exc:  # pragma: no cover - startup guard
    raise SystemExit("PyYAML is required to parse training/blend configs.") from exc


BRACE_RANGE_PATTERN = re.compile(r"\{(-?\d+)\.\.(-?\d+)(?:\.\.(-?\d+))?\}")
EVAL_STEP_PATTERN = re.compile(r"eval-step-(\d+)$")
STATEFUL_KEY = "train_dataloader_per_rank"
POSITION_KEYS = {"position", "shard_id", "num_shards"}
INDEX_DIR_CACHE: dict[str, dict[str, int] | None] = {}


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    source_index: int
    name: str
    desired_weight: float | None = None
    raw_weight: float | None = None
    hours: float | None = None
    kind: str | None = None
    source_path: str | None = None
    total_items: int | None = None
    missing_index_paths: list[str] = field(default_factory=list)


@dataclass
class LeafProgress:
    source_index: int
    rank: int | None
    worker: str | None
    state_type: str
    epoch: int
    position: int
    shard_id: int | None
    num_shards: int | None
    total_len: int | None
    state_path: str


@dataclass
class SummaryRow:
    source_index: int
    dataset: str
    state_type: str
    desired_weight: float | None
    observed_weight: float | None
    drift_abs: float | None
    drift_ratio: float | None
    utilization_pct: float | None
    completed_epochs: int | None
    current_epoch_pct: float | None
    consumed_items: int | None
    total_items: int | None
    partitions_seen: int
    min_epoch: int | None
    max_epoch: int | None
    min_position: int | None
    max_position: int | None
    missing_total: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_dataset_specs(
    config: dict[str, Any] | None,
    *,
    config_path: Path | None,
    indexes_root: str | None,
    data_blend_dir: str | None,
) -> list[DatasetSpec]:
    """Resolve training blend leaves into ordered dataset specs.

    The checkpoint records source progress by leaf order, not by dataset name.
    This function walks ``data.train_ds.input_cfg`` with the same nested blend
    references and temperature-normalized weights used by the recipe, then
    counts examples from indexed sidecars when available.
    """
    if not config:
        return []
    train_ds = _get_path(config, "data.train_ds")
    if not isinstance(train_ds, dict):
        return []
    if data_blend_dir is None:
        raw_dir = config.get("data_blend_dir")
        if isinstance(raw_dir, str):
            data_blend_dir = raw_dir
    current_dir = config_path.parent if config_path is not None else None
    temps = _temperature_list(train_ds)
    leaves: list[dict[str, Any]] = []

    def recurse(
        node: Any,
        cumulative_weight: float,
        level: int,
        cur_dir: Path | None,
        inherited: dict[str, Any],
    ) -> None:
        node, next_dir = _load_ref_if_yaml(node, data_blend_dir=data_blend_dir, current_dir=cur_dir)
        if isinstance(node, dict):
            merged = dict(inherited)
            for key, value in node.items():
                if key not in ("input_cfg", "weight"):
                    merged.setdefault(key, value)
            if "input_cfg" in node:
                child = node["input_cfg"]
                recurse(child, cumulative_weight, level, next_dir, merged)
            else:
                leaf = dict(merged)
                leaf.update(node)
                leaf["_desired_weight"] = cumulative_weight
                leaf["_raw_weight"] = _safe_float(node.get("weight"))
                leaves.append(leaf)
            return
        if isinstance(node, list):
            weights = [_safe_float(item.get("weight")) if isinstance(item, dict) else None for item in node]
            if all(w is not None for w in weights):
                temperature = temps[level] if temps and level < len(temps) else 1.0
                local_weights = _normalize_weight_vector([float(w) for w in weights], temperature=temperature)
            else:
                local_weights = [1.0 / len(node)] * len(node)
            for item, local_weight in zip(node, local_weights):
                item_weight = cumulative_weight * local_weight
                recurse(item, item_weight, level + 1, next_dir, inherited)
            return
        if _looks_like_yaml_ref(node):
            loaded, loaded_dir = _load_ref_if_yaml(node, data_blend_dir=data_blend_dir, current_dir=cur_dir)
            recurse(loaded, cumulative_weight, level, loaded_dir, inherited)

    recurse(train_ds.get("input_cfg"), 1.0, 0, current_dir, {})

    specs: list[DatasetSpec] = []
    for idx, leaf in enumerate(leaves):
        path_groups = _source_path_groups_for_item(leaf)
        paths = path_groups[0] if path_groups else []
        total_items = None
        missing: list[str] = []
        for group in path_groups:
            group_total, group_missing = _count_indexed_items(group, indexes_root)
            if group_total is not None:
                total_items = group_total
                paths = group
                missing = group_missing
                break
            missing.extend(group_missing[:20])
        source_path = paths[0] if paths else None
        specs.append(
            DatasetSpec(
                source_index=idx,
                name=_dataset_name(leaf, source_path, idx),
                desired_weight=_safe_float(leaf.get("_desired_weight")),
                raw_weight=_safe_float(leaf.get("_raw_weight")),
                hours=_safe_float(leaf.get("hours")),
                kind=str(leaf.get("type")) if leaf.get("type") is not None else None,
                source_path=source_path,
                total_items=total_items,
                missing_index_paths=missing[:20],
            )
        )
    return specs


def extract_progress(payload: Any) -> tuple[list[LeafProgress], list[str]]:
    """Extract per-leaf dataloader progress from a loaded checkpoint payload.

    The preferred layout is NeMo's ``train_dataloader_per_rank`` list, but the
    scanner also handles raw nested ``sampler_state`` payloads for debugging and
    compatibility with partially extracted state dumps.
    """
    notes: list[str] = []
    progress: list[LeafProgress] = []
    stateful_payloads = _find_stateful_payloads(payload)
    if stateful_payloads:
        notes.append(f"found {len(stateful_payloads)} {STATEFUL_KEY!r} payload(s)")
        for state_path, per_rank in stateful_payloads:
            for idx, entry in enumerate(per_rank):
                if not isinstance(entry, dict):
                    continue
                rank = entry.get("dp_rank", idx)
                rank = rank if isinstance(rank, int) else idx
                inner_state = entry.get("state", entry)
                for sampler_path, sampler_state in _find_sampler_states(inner_state, f"{state_path}[{idx}].state"):
                    worker = _worker_from_path(sampler_path)
                    progress.extend(
                        _collect_leaves_from_sampler(sampler_state, rank=rank, worker=worker, path=sampler_path)
                    )
    else:
        notes.append(f"no {STATEFUL_KEY!r} payload found; scanning for raw sampler_state entries")
        for sampler_path, sampler_state in _find_sampler_states(payload):
            worker = _worker_from_path(sampler_path)
            progress.extend(_collect_leaves_from_sampler(sampler_state, rank=None, worker=worker, path=sampler_path))
    progress, removed = _deduplicate_progress(progress)
    if removed:
        notes.append(f"deduplicated {removed} duplicate leaf progress state(s)")
    return progress, notes


def summarize(progress: list[LeafProgress], specs: list[DatasetSpec]) -> list[SummaryRow]:
    """Combine checkpoint progress and dataset specs into report rows.

    Consumed examples are aggregated across ranks/workers, converted to dataset
    utilization percentages when totals are known, and normalized into observed
    blend weights for drift reporting.
    """
    spec_by_index = {spec.source_index: spec for spec in specs}
    grouped: dict[int, list[LeafProgress]] = {}
    for leaf in progress:
        grouped.setdefault(leaf.source_index, []).append(leaf)
    consumed_by_index: dict[int, int | None] = {}
    total_observed_consumed = 0
    for source_index, leaves in grouped.items():
        spec = spec_by_index.get(source_index)
        total_len = next((leaf.total_len for leaf in leaves if leaf.total_len is not None), None)
        if total_len is None and spec is not None:
            total_len = spec.total_items
        values = [_consumed_items(leaf, total_len) for leaf in leaves]
        consumed = sum(v for v in values if v is not None) if all(v is not None for v in values) else None
        consumed_by_index[source_index] = consumed
        if consumed is not None:
            total_observed_consumed += consumed

    rows: list[SummaryRow] = []
    for source_index in sorted(grouped):
        leaves = grouped[source_index]
        spec = spec_by_index.get(source_index)
        total_len = next((leaf.total_len for leaf in leaves if leaf.total_len is not None), None)
        if total_len is None and spec is not None:
            total_len = spec.total_items
        consumed = consumed_by_index[source_index]
        utilization = (100.0 * consumed / total_len) if consumed is not None and total_len else None
        observed = (consumed / total_observed_consumed) if consumed is not None and total_observed_consumed else None
        desired = spec.desired_weight if spec is not None else None
        drift_abs = observed - desired if observed is not None and desired is not None else None
        drift_ratio = observed / desired if observed is not None and desired not in (None, 0) else None
        completed_epochs = math.floor(utilization / 100.0) if utilization is not None else None
        current_epoch_pct = (
            utilization - completed_epochs * 100.0
            if utilization is not None and completed_epochs is not None
            else None
        )
        notes = []
        if spec is None:
            notes.append("no matching config source")
        elif spec.total_items is None and total_len is None:
            notes.append("missing total; provide --indexes-root or a config with indexed sidecars")
        elif spec.missing_index_paths:
            notes.append(f"{len(spec.missing_index_paths)} missing index path(s)")
        rows.append(
            SummaryRow(
                source_index=source_index,
                dataset=spec.name if spec is not None else f"source-{source_index}",
                state_type="/".join(sorted({leaf.state_type for leaf in leaves})),
                desired_weight=desired,
                observed_weight=observed,
                drift_abs=drift_abs,
                drift_ratio=drift_ratio,
                utilization_pct=utilization,
                completed_epochs=completed_epochs,
                current_epoch_pct=current_epoch_pct,
                consumed_items=consumed,
                total_items=total_len,
                partitions_seen=len(leaves),
                min_epoch=min(leaf.epoch for leaf in leaves) if leaves else None,
                max_epoch=max(leaf.epoch for leaf in leaves) if leaves else None,
                min_position=min(leaf.position for leaf in leaves) if leaves else None,
                max_position=max(leaf.position for leaf in leaves) if leaves else None,
                missing_total=total_len is None,
                notes="; ".join(notes),
            )
        )
    return rows


def load_checkpoint_payload(path: Path, *, allow_full_load: bool, max_full_load_mb: int) -> tuple[Any, Path]:
    """Load the smallest checkpoint payload that contains dataloader state.

    Metadata files are preferred. Full checkpoint files are skipped unless
    explicitly allowed and under the configured size cap.
    """
    errors = []
    for candidate in _checkpoint_metadata_candidates(path):
        if not candidate.is_file():
            continue
        if candidate.name != "meta.pt" and not allow_full_load:
            size_mb = candidate.stat().st_size / (1024 * 1024)
            if size_mb > max_full_load_mb:
                errors.append(f"skipped large non-meta checkpoint {candidate} ({size_mb:.1f} MiB)")
                continue
        try:
            payload = _torch_load(candidate)
        except Exception as exc:  # pragma: no cover - depends on checkpoint format
            errors.append(f"{candidate}: {exc}")
            continue
        progress, _ = extract_progress(payload)
        if progress:
            return payload, candidate
        errors.append(f"{candidate}: loaded but no dataloader progress state found")
    detail = "\n".join(errors[-10:])
    raise RuntimeError(f"Could not find dataloader state for {path}.\n{detail}")


def load_config(path: Path | None, checkpoint: Path) -> tuple[dict[str, Any] | None, Path | None, list[str]]:
    """Load an explicit or nearby training config used to annotate the report."""
    notes = []
    candidates = [path] if path is not None else _auto_config_candidates(checkpoint)
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        try:
            if candidate.suffix == ".json":
                data = _load_json(candidate)
            else:
                data = _load_yaml(candidate)
        except Exception as exc:
            notes.append(f"failed to load config candidate {candidate}: {exc}")
            continue
        if isinstance(data, dict):
            if _get_path(data, "data.train_ds") is not None:
                return data, candidate, notes
            notes.append(f"skipped config candidate {candidate}: no data.train_ds")
    if path is None:
        notes.append("no config found; pass --config for desired weights and index totals")
    else:
        notes.append(f"config not found or invalid: {path}")
    return None, None, notes


def markdown_table(rows: list[SummaryRow]) -> str:
    """Render summary rows as a compact Markdown table for logs/stdout."""
    headers = [
        "idx",
        "dataset",
        "utilization",
        "epochs",
        "desired_w",
        "observed_w",
        "drift",
        "items",
        "total",
        "parts",
        "notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        epoch_text = ""
        if row.completed_epochs is not None and row.current_epoch_pct is not None:
            epoch_text = f"{row.completed_epochs} + {row.current_epoch_pct:.2f}%"
        values = [
            str(row.source_index),
            row.dataset.replace("|", "\\|"),
            _fmt_pct(row.utilization_pct),
            epoch_text,
            _fmt_float(row.desired_weight),
            _fmt_float(row.observed_weight),
            _fmt_float(row.drift_abs),
            "" if row.consumed_items is None else str(row.consumed_items),
            "" if row.total_items is None else str(row.total_items),
            str(row.partitions_seen),
            row.notes.replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(summary: dict[str, Any], rows: list[SummaryRow], args: argparse.Namespace) -> None:
    """Write requested JSON, Markdown, and CSV artifacts."""
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.output_json) if args.output_json else (output_dir / "summary.json" if output_dir else None)
    md_path = Path(args.output_md) if args.output_md else (output_dir / "summary.md" if output_dir else None)
    csv_path = Path(args.output_csv) if args.output_csv else (output_dir / "summary.csv" if output_dir else None)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        body = [
            f"# Resumable Dataloader Checkpoint Analysis",
            "",
            f"- checkpoint: `{summary['checkpoint_input']}`",
            f"- metadata_loaded: `{summary.get('checkpoint_metadata_loaded')}`",
            f"- config: `{summary.get('config_path') or ''}`",
            f"- generated_at: `{summary['generated_at']}`",
            "",
            markdown_table(rows),
        ]
        md_path.write_text("\n".join(body), encoding="utf-8")
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else ["source_index"])
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for local or cluster-submitted analysis runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", "--checkpoint-path", dest="checkpoint", help="Checkpoint file/dir or eval-step dir."
    )
    parser.add_argument("--state-json", help="JSON payload to analyze instead of loading a torch checkpoint.")
    parser.add_argument("--config", help="Training YAML/JSON for desired weights and source index totals.")
    parser.add_argument("--data-blend-dir", help="Override ${data_blend_dir} while resolving nested blend YAMLs.")
    parser.add_argument("--indexes-root", help="Root containing mirrored .idx sidecars, e.g. /tmp/idx.")
    parser.add_argument("--output-dir", help="Directory for summary.json/summary.md/summary.csv.")
    parser.add_argument("--output-json", help="Explicit JSON output path.")
    parser.add_argument("--output-md", help="Explicit Markdown output path.")
    parser.add_argument("--output-csv", help="Explicit CSV output path.")
    parser.add_argument("--allow-full-ckpt-load", action="store_true", help="Allow loading non-meta checkpoint files.")
    parser.add_argument("--max-full-load-mb", type=int, default=512, help="Safety cap for non-meta checkpoint files.")
    parser.add_argument("--print-table", action="store_true", help="Print Markdown table to stdout.")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for loading inputs, computing rows, and writing outputs."""
    args = parse_args()
    if not args.checkpoint and not args.state_json:
        raise SystemExit("Pass --checkpoint or --state-json.")

    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else Path(args.state_json).expanduser()
    loaded_path: Path | None = None
    if args.state_json:
        payload = _load_json(Path(args.state_json).expanduser())
        loaded_path = Path(args.state_json).expanduser()
    else:
        payload, loaded_path = load_checkpoint_payload(
            checkpoint,
            allow_full_load=args.allow_full_ckpt_load,
            max_full_load_mb=args.max_full_load_mb,
        )

    progress, notes = extract_progress(payload)
    config_path = Path(args.config).expanduser() if args.config else None
    config, loaded_config_path, config_notes = load_config(config_path, checkpoint)
    notes.extend(config_notes)
    specs = collect_dataset_specs(
        config,
        config_path=loaded_config_path,
        indexes_root=args.indexes_root,
        data_blend_dir=args.data_blend_dir,
    )
    if specs and len(specs) != len({leaf.source_index for leaf in progress}):
        notes.append(
            f"config source count ({len(specs)}) differs from checkpoint source count "
            f"({len({leaf.source_index for leaf in progress})}); mapping is by source order only"
        )
    rows = summarize(progress, specs)
    summary = {
        "checkpoint_input": str(checkpoint),
        "checkpoint_metadata_loaded": str(loaded_path) if loaded_path else None,
        "config_path": str(loaded_config_path) if loaded_config_path else None,
        "indexes_root": args.indexes_root,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notes": notes,
        "num_leaf_progress_states": len(progress),
        "num_summary_rows": len(rows),
        "rows": [asdict(row) for row in rows],
        "leaf_progress": [asdict(leaf) for leaf in progress],
        "dataset_specs": [asdict(spec) for spec in specs],
    }
    write_outputs(summary, rows, args)
    if args.print_table or not (args.output_dir or args.output_json or args.output_md or args.output_csv):
        sys.stdout.write(markdown_table(rows))
        if notes:
            sys.stdout.write("\nNotes:\n")
            for note in notes:
                sys.stdout.write(f"- {note}\n")
    return 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_path(data: Any, dotted: str) -> Any:
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _normalize_weight_vector(weights: list[float], temperature: float = 1.0) -> list[float]:
    if not weights:
        return []
    scaled = [w**temperature for w in weights]
    total = sum(scaled)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in scaled]


def _strip_url_scheme(path: str) -> str:
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://(.+)$", path)
    return match.group(1) if match else path.lstrip("/")


def _index_path_for(data_path: str, indexes_root: str | None) -> Path | None:
    if not indexes_root:
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", data_path):
            return None
        return Path(data_path + ".idx")
    return Path(indexes_root) / (_strip_url_scheme(data_path) + ".idx")


def _indexed_file_size(idx_path: Path) -> int | None:
    parent = str(idx_path.parent)
    entries = INDEX_DIR_CACHE.get(parent)
    if entries is None and parent not in INDEX_DIR_CACHE:
        try:
            entries = {entry.name: entry.stat().st_size for entry in os.scandir(idx_path.parent) if entry.is_file()}
        except FileNotFoundError:
            INDEX_DIR_CACHE[parent] = None
            return None
        INDEX_DIR_CACHE[parent] = entries
    if entries is None:
        return None
    return entries.get(idx_path.name)


def _fallback_brace_expand(path: str) -> list[str]:
    match = BRACE_RANGE_PATTERN.search(path)
    if not match:
        return [path]
    start_text, end_text, step_text = match.group(1), match.group(2), match.group(3)
    start, end = int(start_text), int(end_text)
    step = int(step_text) if step_text is not None else (1 if start <= end else -1)
    if step == 0:
        return [path]
    if start < end and step < 0:
        return [path]
    if start > end and step > 0:
        return [path]
    width = max(len(start_text.lstrip("-")), len(end_text.lstrip("-")))
    stop = end + (1 if step > 0 else -1)
    expanded = []
    for idx in range(start, stop, step):
        sign = "-" if idx < 0 else ""
        repl = f"{sign}{abs(idx):0{width}d}"
        expanded.extend(_fallback_brace_expand(path[: match.start()] + repl + path[match.end() :]))
    return expanded


def _expand_op_path(path: str) -> list[str]:
    # Match NeMo expand_sharded_filepaths(): _OP_/_CL_ are aliases for brace ranges.
    sharded = path
    for brace_open in ("(", "[", "<", "_OP_"):
        sharded = sharded.replace(brace_open, "{")
    for brace_close in (")", "]", ">", "_CL_"):
        sharded = sharded.replace(brace_close, "}")
    try:
        import braceexpand

        return list(braceexpand.braceexpand(sharded, escape=False))
    except ImportError:
        return _fallback_brace_expand(sharded)


def _flatten_path_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
                out.append(item[0])
            elif isinstance(item, dict):
                out.extend(_flatten_path_values(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_flatten_path_values(item))
        return out
    return []


def _count_indexed_items(paths: list[str], indexes_root: str | None) -> tuple[int | None, list[str]]:
    total = 0
    missing: list[str] = []
    any_count = False
    for path in paths:
        for expanded in _expand_op_path(path):
            idx_path = _index_path_for(expanded, indexes_root)
            if idx_path is None:
                missing.append(f"{expanded}.idx")
                continue
            size = _indexed_file_size(idx_path)
            if size is None:
                missing.append(str(idx_path))
                continue
            if size < 8 or size % 8 != 0:
                missing.append(str(idx_path))
                continue
            total += size // 8 - 1
            any_count = True
    return (total if any_count else None), missing


def _resolve_ref(ref: str, *, data_blend_dir: str | None, current_dir: Path | None) -> Path:
    text = ref
    if data_blend_dir:
        text = text.replace("${data_blend_dir}", data_blend_dir)
    text = os.path.expandvars(text)
    path = Path(text)
    if path.is_absolute():
        return path
    if current_dir is not None:
        return current_dir / path
    return Path.cwd() / path


def _looks_like_yaml_ref(value: Any) -> bool:
    return isinstance(value, str) and value.endswith((".yaml", ".yml"))


def _load_ref_if_yaml(value: Any, *, data_blend_dir: str | None, current_dir: Path | None) -> tuple[Any, Path | None]:
    if not _looks_like_yaml_ref(value):
        return value, current_dir
    path = _resolve_ref(value, data_blend_dir=data_blend_dir, current_dir=current_dir)
    return _load_yaml(path), path.parent


def _source_path_groups_for_item(item: dict[str, Any]) -> list[list[str]]:
    groups: list[list[str]] = []
    keys = [
        "manifest_filepath",
        "cuts_path",
        "source_paths",
        "source_path",
        "shar_path",
        "tarred_audio_filepaths",
        "tarred_audio_filepath",
    ]
    kind = str(item.get("type", ""))
    if "nemo_tarred" in kind and (item.get("tarred_audio_filepaths") or item.get("tarred_audio_filepath")):
        keys = [
            "tarred_audio_filepaths",
            "tarred_audio_filepath",
            "manifest_filepath",
            "cuts_path",
            "source_paths",
            "source_path",
            "shar_path",
        ]
    for key in keys:
        paths = _flatten_path_values(item.get(key))
        if paths:
            groups.append(paths)
    return groups


def _source_paths_for_item(item: dict[str, Any]) -> list[str]:
    groups = _source_path_groups_for_item(item)
    return groups[0] if groups else []


def _dataset_name(item: dict[str, Any], source_path: str | None, fallback_index: int) -> str:
    pieces = []
    for key in ("corpus", "language", "dataset", "name", "type"):
        value = item.get(key)
        if value is not None and not isinstance(value, (dict, list)):
            pieces.append(str(value))
    if pieces:
        return "/".join(pieces)
    if source_path:
        return source_path
    return f"source-{fallback_index}"


def _temperature_list(train_ds: dict[str, Any]) -> list[float] | None:
    value = train_ds.get("reweight_temperature")
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)] * 16
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return None


def _iter_children(obj: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield f"{path}.{key}", value
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield f"{path}[{idx}]", value


def _find_stateful_payloads(obj: Any, path: str = "$") -> list[tuple[str, list[Any]]]:
    found: list[tuple[str, list[Any]]] = []
    if isinstance(obj, dict):
        value = obj.get(STATEFUL_KEY)
        if isinstance(value, list):
            found.append((f"{path}.{STATEFUL_KEY}", value))
        for child_path, child in _iter_children(obj, path):
            found.extend(_find_stateful_payloads(child, child_path))
    elif isinstance(obj, list):
        for child_path, child in _iter_children(obj, path):
            found.extend(_find_stateful_payloads(child, child_path))
    return found


def _find_sampler_states(obj: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(obj, dict):
        sampler_state = obj.get("sampler_state")
        if isinstance(sampler_state, dict):
            found.append((f"{path}.sampler_state", sampler_state))
        elif "cuts_state" in obj and "diagnostics" in obj:
            found.append((path, obj))
        for child_path, child in _iter_children(obj, path):
            if child is sampler_state:
                continue
            found.extend(_find_sampler_states(child, child_path))
    elif isinstance(obj, list):
        for child_path, child in _iter_children(obj, path):
            found.extend(_find_sampler_states(child, child_path))
    return found


def _worker_from_path(path: str) -> str | None:
    match = re.search(r"worker[_-]?(\d+)", path)
    if match:
        return match.group(1)
    return None


def _state_total_len(state: dict[str, Any]) -> int | None:
    range_state = state.get("range")
    if isinstance(range_state, dict):
        n = range_state.get("n")
        if isinstance(n, int) and n >= 0:
            return n
    n = state.get("total_len") or state.get("_total_len") or state.get("n")
    return int(n) if isinstance(n, int) and n >= 0 else None


def _leaf_from_state(
    source_index: int,
    rank: int | None,
    worker: str | None,
    path: str,
    node_type: str,
    state: dict[str, Any],
) -> LeafProgress | None:
    if not POSITION_KEYS.issubset(state.keys()):
        return None
    position = state.get("position")
    if not isinstance(position, int):
        return None
    epoch = state.get("epoch", 0)
    if not isinstance(epoch, int):
        epoch = 0
    shard_id = state.get("shard_id")
    num_shards = state.get("num_shards")
    return LeafProgress(
        source_index=source_index,
        rank=rank,
        worker=worker,
        state_type=node_type,
        epoch=epoch,
        position=position,
        shard_id=shard_id if isinstance(shard_id, int) else None,
        num_shards=num_shards if isinstance(num_shards, int) else None,
        total_len=_state_total_len(state),
        state_path=path,
    )


def _collect_leaf_states(tree: Any, *, rank: int | None, worker: str | None, path: str = "$") -> list[LeafProgress]:
    leaves: list[LeafProgress] = []

    def walk(node: Any, node_path: str) -> None:
        if isinstance(node, dict):
            node_type = str(node.get("_type", "state"))
            state = node.get("_state")
            if isinstance(state, dict):
                leaf = _leaf_from_state(len(leaves), rank, worker, f"{node_path}._state", node_type, state)
                if leaf is not None:
                    leaves.append(leaf)
                    return
                for key in ("source", "sources"):
                    if key in state:
                        walk(state[key], f"{node_path}._state.{key}")
            leaf = _leaf_from_state(len(leaves), rank, worker, node_path, node_type, node)
            if leaf is not None:
                leaves.append(leaf)
                return
            for child_path, child in _iter_children(node, node_path):
                walk(child, child_path)
        elif isinstance(node, list):
            for child_path, child in _iter_children(node, node_path):
                walk(child, child_path)

    walk(tree, path)
    return leaves


def _collect_leaves_from_sampler(
    sampler_state: dict[str, Any], *, rank: int | None, worker: str | None, path: str
) -> list[LeafProgress]:
    leaves: list[LeafProgress] = []
    nested = sampler_state.get("samplers") or sampler_state.get("bucket_samplers")
    if isinstance(nested, list):
        for idx, sub in enumerate(nested):
            if isinstance(sub, dict):
                leaves.extend(
                    _collect_leaves_from_sampler(
                        sub,
                        rank=rank,
                        worker=worker,
                        path=f"{path}.samplers[{idx}]",
                    )
                )
        if leaves:
            for idx, leaf in enumerate(leaves):
                leaf.source_index = idx
            return leaves
    cuts_state = sampler_state.get("cuts_state")
    if cuts_state is not None:
        leaves = _collect_leaf_states(cuts_state, rank=rank, worker=worker, path=f"{path}.cuts_state")
        for idx, leaf in enumerate(leaves):
            leaf.source_index = idx
    return leaves


def _deduplicate_progress(progress: list[LeafProgress]) -> tuple[list[LeafProgress], int]:
    deduped: list[LeafProgress] = []
    seen: set[tuple[Any, ...]] = set()
    for leaf in progress:
        key = (leaf.rank, leaf.worker, leaf.state_path, leaf.shard_id, leaf.num_shards)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(leaf)
    return deduped, len(progress) - len(deduped)


def _shard_len(total_len: int, shard_id: int | None, num_shards: int | None) -> int | None:
    if shard_id is None or num_shards is None or num_shards <= 0:
        return None
    if total_len <= shard_id:
        return 0
    return (total_len - shard_id + num_shards - 1) // num_shards


def _consumed_items(leaf: LeafProgress, total_len: int | None) -> int | None:
    total = leaf.total_len if leaf.total_len is not None else total_len
    if total is None:
        if leaf.epoch == 0:
            return leaf.position
        return None
    shard_len = _shard_len(total, leaf.shard_id, leaf.num_shards)
    if shard_len is None:
        return leaf.position if leaf.epoch == 0 else None
    return leaf.epoch * shard_len + leaf.position


def _eval_step_candidates(path: Path) -> list[Path]:
    match = EVAL_STEP_PATTERN.fullmatch(path.name)
    if not match:
        return []
    step = match.group(1)
    ckpt_dir = path.parent / "checkpoints"
    return [
        ckpt_dir / f"step={step}.ckpt",
        ckpt_dir / f"step={step}-last.ckpt",
        ckpt_dir / f"step-{step}.ckpt",
        ckpt_dir / f"step-{step}-last.ckpt",
    ]


def _checkpoint_metadata_candidates(path: Path) -> list[Path]:
    candidates: list[Path] = []
    if path.is_dir():
        candidates.extend(_eval_step_candidates(path))
        candidates.extend([path / "meta.pt", path / "checkpoint" / "meta.pt"])
        for child in sorted(path.glob("*.ckpt")):
            candidates.append(child)
        for child in sorted(path.glob("**/meta.pt")):
            candidates.append(child)
    else:
        candidates.append(path)
    expanded: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            expanded.extend([candidate / "meta.pt", candidate / "checkpoint" / "meta.pt"])
        expanded.append(candidate)
    deduped: list[Path] = []
    seen = set()
    for candidate in expanded:
        text = str(candidate)
        if text not in seen:
            seen.add(text)
            deduped.append(candidate)
    return deduped


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _auto_config_candidates(checkpoint: Path) -> list[Path]:
    candidates = []
    roots = [checkpoint]
    if checkpoint.is_file():
        roots.append(checkpoint.parent)
    if checkpoint.is_dir():
        roots.extend([checkpoint.parent, checkpoint.parent.parent])
    for root in roots:
        if root.is_file():
            continue
        candidates.extend(
            [
                root / "exp_config.yaml",
                root / "config.yaml",
                root / "hparams.yaml",
                root / "config.json",
            ]
        )
    deduped = []
    seen = set()
    for candidate in candidates:
        text = str(candidate)
        if text not in seen:
            seen.add(text)
            deduped.append(candidate)
    return deduped


def _fmt_pct(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}%"


def _fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
