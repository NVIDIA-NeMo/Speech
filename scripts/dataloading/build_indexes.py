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
Build O(1)-restore index sidecars for an arbitrary NeMo Lhotse ``input_cfg``.

Walks a NeMo dataloading config (``input_cfg`` YAML, including nested ``group``
entries and per-entry YAML references), discovers every JSONL/tar file an
indexed dataloader will need, and creates the corresponding ``.idx`` sidecars
next to each data file.

Two tar layouts are dispatched correctly:

* NeMo tarred audio (one regular member per sample, name-keyed) — uses
  ``nemo.collections.common.data.lhotse.indexed_adapters.create_tar_index``
  which records one offset per *basename group*.
* WebDataset/Shar tars (json + payload pairs) — uses
  ``lhotse.indexing.create_tar_index`` which records one offset per *member
  pair*.

Local files and remote URIs are both supported via lhotse's ``open_best``
(which routes to ``smart_open`` / AIStore SDK when available). The ``.idx`` is
written next to its source path, so the storage backend must accept writes at
that location — for read-only object stores, materialize the data locally
first or pre-build indexes at upload time.

Examples::

    # Build indexes for everything referenced by an input_cfg.yaml.
    python scripts/dataloading/build_indexes.py path/to/input_cfg.yaml

    # Multiple configs at once.
    python scripts/dataloading/build_indexes.py train.yaml validation.yaml

    # Show what would be built without writing anything.
    python scripts/dataloading/build_indexes.py --dry-run path/to/input_cfg.yaml

    # Rebuild even when an .idx already exists; parallelize across 16 workers.
    python scripts/dataloading/build_indexes.py --force --workers 16 path/to/input_cfg.yaml
"""

import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
from lhotse.indexing import index_file_path
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index as create_nemo_tar_index
from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths

# --------------------------------------------------------------------------- #
# Tar layout taxonomy.
# --------------------------------------------------------------------------- #
# NEMO_TAR  — one regular member per sample, indexed by basename. Used by
#             nemo / nemo_tarred / multimodal_conversation / share_gpt audio
#             tars (read via IndexedTarMemberReader).
# WDS_TAR   — WebDataset-style: each sample is a pair of consecutive members
#             (e.g. {N}.json + {N}.<audio>). Used by lhotse_shar tars and
#             share_gpt_webdataset tars (read via IndexedTarSampleReader).
NEMO_TAR = "nemo_tar"
WDS_TAR = "wds_tar"
JSONL = "jsonl"


@dataclass(frozen=True)
class IndexJob:
    path: str
    kind: str  # one of {JSONL, NEMO_TAR, WDS_TAR}
    indexes_root: Optional[str] = None

    def idx_path(self):
        return index_file_path(self.path, self.indexes_root)


# --------------------------------------------------------------------------- #
# Path discovery.
# --------------------------------------------------------------------------- #


def _as_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, (list, tuple, ListConfig)):
        return list(val)
    return [val]


def _flatten_path_spec(spec) -> list[str]:
    """
    NeMo's manifest_filepath / tarred_audio_filepaths accept several layouts:
      str, list[str], list[list[str]], list[tuple[str, weight]], ...
    Flatten any of those into a list of plain string paths.
    """
    out: list[str] = []
    for item in _as_list(spec):
        if isinstance(item, (str, Path)):
            out.append(str(item))
        elif isinstance(item, (list, tuple, ListConfig)):
            # [path] or [path, weight] or [[path], [path], ...]
            head = item[0]
            if isinstance(head, (str, Path)):
                out.append(str(head))
            else:
                out.extend(_flatten_path_spec(item))
    return out


def _expand_jsonl(spec) -> list[str]:
    return [p for raw in _flatten_path_spec(spec) for p in expand_sharded_filepaths(raw)]


def _expand_tars(spec) -> list[str]:
    return [p for raw in _flatten_path_spec(spec) for p in expand_sharded_filepaths(raw)]


def _resolve_input_cfg(val) -> ListConfig | None:
    """``input_cfg`` may be inline or a path to a YAML file. Materialize it."""
    if isinstance(val, (list, ListConfig)):
        return val
    if isinstance(val, (str, Path)):
        return OmegaConf.load(str(val))
    return None


# Types that don't read any data themselves — they delegate to
# ``read_cutset_from_config(config)`` and accept *any* underlying source's keys
# (``cuts_path``, ``shar_path``, ``manifest_filepath`` [+ ``tarred_audio_filepaths``],
# nested ``input_cfg``, …). Treat them as transparent passthroughs.
_TRANSFORM_TYPES = frozenset(
    {
        "lhotse_as_conversation",
        "sqa_as_conversation",
        "s2s_as_conversation",
        "s2s_duplex_overlap_as_s2s_duplex",
        "s2s_duplex_reverse_role",
        "lhotse_magpietts_data_as_continuation",
        "nemo_tarred_to_duplex",
    }
)

# Types that index nothing on their own.
_NO_INDEX_TYPES = frozenset({"txt", "txt_pair", "parquet", "multi_speaker_simulator"})


def _discover_keys(entry, jobs: list[IndexJob], indexes_root: Optional[str]) -> None:
    """
    Key-based dispatch: emit IndexJobs based on which underlying-source keys
    are present, regardless of ``type``. Used for transform types that
    delegate to ``read_cutset_from_config``, and as the inner step for
    concrete types that name them directly. Per-entry ``indexes_root``
    overrides the inherited value when set.
    """
    indexes_root = entry.get("indexes_root", indexes_root)
    if (cuts_path := entry.get("cuts_path")) is not None:
        for p in _expand_jsonl(cuts_path):
            jobs.append(IndexJob(p, JSONL, indexes_root))
    if (shar_path := entry.get("shar_path")) is not None:
        _discover_shar(shar_path, jobs, indexes_root)
    if (mfp := entry.get("manifest_filepath")) is not None:
        for p in _expand_jsonl(mfp):
            jobs.append(IndexJob(p, JSONL, indexes_root))
        for p in _expand_tars(entry.get("tarred_audio_filepaths")):
            jobs.append(IndexJob(p, NEMO_TAR, indexes_root))
    if (paths := entry.get("paths")) is not None:
        _discover_paths(paths, jobs, indexes_root)
    if (sub := _resolve_input_cfg(entry.get("input_cfg"))) is not None:
        discover(sub, jobs, indexes_root)


def _discover_paths(paths, jobs: list[IndexJob], indexes_root: Optional[str]) -> None:
    for p in _expand_jsonl(paths):
        path = Path(p)
        if path.is_dir():
            for tar_path in sorted(path.rglob("*.tar")):
                jobs.append(IndexJob(str(tar_path), NEMO_TAR, indexes_root))
        elif path.suffix == ".tar":
            jobs.append(IndexJob(p, NEMO_TAR, indexes_root))
        else:
            jobs.append(IndexJob(p, JSONL, indexes_root))


def _discover_share_gpt_webdataset(data_dir, jobs: list[IndexJob], indexes_root: Optional[str]) -> None:
    """
    Match NeMoMultimodalConversationShareGPTWebdatasetAdapter shard discovery.

    The adapter reads ``wids-meta.json`` when present; otherwise it recursively
    scans ``data_dir`` for tar shards. Energon exports commonly place shards
    under nested directories such as ``0/sharded_manifests/shard-0.tar``, so a
    non-recursive glob silently misses every runtime-required tar index.
    """
    if data_dir is None:
        return

    for raw in _flatten_path_spec(data_dir):
        root = Path(raw)
        meta_path = root / "wids-meta.json"
        if meta_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
            for shard in meta.get("shardlist", []):
                url = shard.get("url") if isinstance(shard, dict) else None
                if url:
                    jobs.append(IndexJob(str(root / url), WDS_TAR, indexes_root))
        elif root.is_dir():
            for tar_path in sorted(root.rglob("*.tar")):
                jobs.append(IndexJob(str(tar_path), WDS_TAR, indexes_root))

        # Preserve the previous behavior for optional root-level sidecar
        # manifests without recursively indexing unrelated metadata files.
        if root.is_dir():
            for jsonl_path in sorted(root.glob("*.jsonl")):
                jobs.append(IndexJob(str(jsonl_path), JSONL, indexes_root))


def discover(entry, jobs: list[IndexJob], indexes_root: Optional[str] = None) -> None:
    """Walk one entry of an ``input_cfg`` and append every required IndexJob."""
    if isinstance(entry, (list, ListConfig)):
        for sub in entry:
            discover(sub, jobs, indexes_root)
        return
    if not isinstance(entry, (dict, DictConfig)):
        return

    # Per-entry override: a nested entry can carry its own ``indexes_root``.
    indexes_root = entry.get("indexes_root", indexes_root)

    typ = entry.get("type")
    if typ is None:
        # Top-level wrapper (``input_cfg: [...]``) — recurse into every value.
        for v in entry.values():
            discover(v, jobs, indexes_root)
        return

    if typ in _NO_INDEX_TYPES:
        return

    if typ == "group" or typ in _TRANSFORM_TYPES:
        # Group and transform passthroughs: dispatch by keys.
        _discover_keys(entry, jobs, indexes_root)
        return

    if typ in ("nemo", "nemo_tarred", "multimodal_conversation", "share_gpt"):
        for p in _expand_jsonl(entry.get("manifest_filepath")):
            jobs.append(IndexJob(p, JSONL, indexes_root))
        for p in _expand_tars(entry.get("tarred_audio_filepaths")):
            jobs.append(IndexJob(p, NEMO_TAR, indexes_root))
        return

    if typ == "share_gpt_webdataset":
        # Layout: data_dir/wids-meta.json or recursive **/*.tar.
        _discover_share_gpt_webdataset(entry.get("data_dir"), jobs, indexes_root)
        return

    if typ == "lhotse":
        if (cuts_path := entry.get("cuts_path")) is not None:
            for p in _expand_jsonl(cuts_path):
                jobs.append(IndexJob(p, JSONL, indexes_root))
        if (shar_path := entry.get("shar_path")) is not None:
            _discover_shar(shar_path, jobs, indexes_root)
        return

    if typ == "lhotse_shar":
        _discover_shar(entry.get("shar_path"), jobs, indexes_root)
        return

    if typ in ("txt_jsonl", "nemotron_text_converation"):
        _discover_paths(entry.get("paths"), jobs, indexes_root)
        return

    # Unknown type — nothing to do.
    return


def _discover_shar(shar_path, jobs: list[IndexJob], indexes_root: Optional[str]) -> None:
    """Index every uncompressed JSONL/tar shard inside one or more Shar dirs."""
    if shar_path is None:
        return
    if isinstance(shar_path, (str, Path)):
        candidates = [shar_path]
    elif isinstance(shar_path, (list, ListConfig)):
        candidates = []
        for item in shar_path:
            if isinstance(item, (str, Path)):
                candidates.append(item)
            elif isinstance(item, (list, tuple, ListConfig)) and item:
                candidates.append(item[0])  # [path, weight] form
    elif isinstance(shar_path, (dict, DictConfig)):
        # {field: [shard, ...]} layout — index every shard in every field.
        for v in shar_path.values():
            for raw in _flatten_path_spec(v):
                for p in expand_sharded_filepaths(raw):
                    if p.endswith(".jsonl"):
                        jobs.append(IndexJob(p, JSONL, indexes_root))
                    elif p.endswith(".tar"):
                        jobs.append(IndexJob(p, WDS_TAR, indexes_root))
        return
    else:
        return

    for d in candidates:
        d = Path(str(d))
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix == ".jsonl":
                jobs.append(IndexJob(str(p), JSONL, indexes_root))
            elif p.suffix == ".tar":
                jobs.append(IndexJob(str(p), WDS_TAR, indexes_root))


# --------------------------------------------------------------------------- #
# Index builders.
# --------------------------------------------------------------------------- #


def _build_one(job: IndexJob) -> tuple[IndexJob, str]:
    """Run the right indexer for *job*. Returns (job, status)."""
    from lhotse.indexing import create_jsonl_index
    from lhotse.indexing import create_tar_index as create_wds_tar_index

    idx = job.idx_path()
    # Ensure the parent directory exists for mirrored layouts.
    idx_parent = Path(idx).parent
    if not str(idx).startswith(("ais://", "s3://", "http://", "https://", "gs://")):
        idx_parent.mkdir(parents=True, exist_ok=True)

    if job.kind == JSONL:
        create_jsonl_index(job.path, output_path=idx)
    elif job.kind == WDS_TAR:
        create_wds_tar_index(job.path, output_path=idx)
    elif job.kind == NEMO_TAR:
        # NeMo's create_tar_index has a (tar_path, idx_path) signature.
        create_nemo_tar_index(job.path, idx)
    else:
        raise ValueError(f"Unknown index kind: {job.kind!r}")
    return job, "built"


def _is_indexed(job: IndexJob) -> bool:
    """True if a non-empty .idx already exists locally."""
    p = Path(job.idx_path())
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


@click.command(context_settings={"show_default": True})
@click.argument("input_cfgs", type=click.Path(exists=True, dir_okay=False), nargs=-1, required=True)
@click.option("--force", is_flag=True, help="Rebuild .idx files even if they already exist.")
@click.option("--workers", type=int, default=4, help="Number of parallel index builders.")
@click.option("--dry-run", is_flag=True, help="List the jobs without writing anything.")
@click.option(
    "--executor",
    type=click.Choice(["process", "thread"]),
    default="process",
    help=(
        "Worker pool kind. ``process`` (default) gives true CPU-level parallelism by "
        "running each indexer in its own interpreter — required for tar indexing where "
        "tarfile.next() and the read-and-discard for data members hold the GIL and "
        "would otherwise serialize all workers onto one core. ``thread`` is useful for "
        "debugging or when indexing only JSONLs over a slow network."
    ),
)
@click.option(
    "--indexes-root",
    type=str,
    default=None,
    help=(
        "Write .idx sidecars to a mirror under this root (preserving the data files' "
        "directory structure) instead of next to each data file. CLI value overrides "
        "any 'indexes_root' present in the YAML."
    ),
)
def main(
    input_cfgs: tuple[str, ...],
    force: bool,
    workers: int,
    dry_run: bool,
    executor: str,
    indexes_root: Optional[str],
):
    """
    Build .idx sidecars for every JSONL/tar referenced by INPUT_CFGS.

    INPUT_CFGS are NeMo Lhotse dataloading configs (``input_cfg`` YAML).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    jobs: list[IndexJob] = []
    for cfg_path in input_cfgs:
        cfg = OmegaConf.load(cfg_path)
        discover(cfg, jobs, indexes_root=indexes_root)

    # Deduplicate while preserving order.
    seen: set[tuple[str, str, Optional[str]]] = set()
    unique: list[IndexJob] = []
    for j in jobs:
        key = (j.path, j.kind, j.indexes_root)
        if key not in seen:
            seen.add(key)
            unique.append(j)

    todo = unique if force else [j for j in unique if not _is_indexed(j)]
    skipped = len(unique) - len(todo)

    logging.info("Discovered %d files (%d already indexed, %d to build).", len(unique), skipped, len(todo))

    if dry_run or not todo:
        for j in todo:
            logging.info("  [%s] %s -> %s", j.kind, j.path, j.idx_path())
        return

    # Per-file success logging is suppressed: building 80k-400k indexes would
    # otherwise emit one log line per file, swamping the SLURM stdout buffer.
    # Failures are still logged inline; success only emits a periodic
    # "<built>/<total> processed" heartbeat (~every 5% of total or 5000 files,
    # whichever is smaller) plus a final summary.
    failures: list[tuple[IndexJob, Exception]] = []
    total = len(todo)
    log_every = max(1, min(5000, total // 20))
    pool_cls = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    with pool_cls(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(_build_one, j): j for j in todo}
        done = 0
        for fut in as_completed(futures):
            done += 1
            j = futures[fut]
            try:
                _, _status = fut.result()
            except Exception as e:  # surface worker failures but let interrupts/system exits propagate
                failures.append((j, e))
                logging.error("  [FAIL] %s %s: %s", j.kind, j.path, e)
                continue
            if done % log_every == 0 or done == total:
                logging.info(
                    "  built %d/%d (%.1f%%)  failures=%d",
                    done,
                    total,
                    100.0 * done / total,
                    len(failures),
                )

    if failures:
        logging.error("\n%d index build(s) failed:", len(failures))
        for j, e in failures:
            logging.error("  %s (%s): %s", j.path, j.kind, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
