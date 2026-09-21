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

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
from lhotse.indexing import index_file_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from scripts.dataloading._sharegpt_route_cli import ensure_sharegpt_route

from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index as create_nemo_tar_index
from nemo.collections.common.data.lhotse.indexed_adapters import (
    create_wds_v2_tar_index,
    validate_wds_v2_tar_index,
    wds_v2_index_path,
    wds_v2_metadata_path,
)
from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths
from nemo.collections.common.data.lhotse.wds_catalog import discover_webdataset_shards

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
WDS_TAR_V2 = "wds_tar_v2"
JSONL = "jsonl"
SHAREGPT_ROUTE = "sharegpt_route"
INDEX_KINDS = (JSONL, NEMO_TAR, WDS_TAR, WDS_TAR_V2, SHAREGPT_ROUTE)


@dataclass(frozen=True)
class IndexJob:
    path: str
    kind: str  # one of INDEX_KINDS
    indexes_root: Optional[str] = None
    manifest_paths: tuple[str, ...] = ()
    tar_paths: tuple[str, ...] = ()
    manifest_specs: tuple[str, ...] = ()
    audio_prefix_map: Optional[dict[str, str]] = None
    audio_placeholders: tuple[str, ...] = ()

    def idx_path(self):
        if self.kind == SHAREGPT_ROUTE:
            route_path = Path(self.path)
            if route_path.is_absolute():
                return route_path
            if self.indexes_root is None:
                raise ValueError("relative tar_routing_filepath requires indexes_root")
            if str(self.indexes_root).startswith(("ais://", "s3://")):
                raise ValueError("ShareGPT routes require a local indexes_root")
            return Path(self.indexes_root) / route_path
        if self.kind == WDS_TAR_V2:
            return wds_v2_index_path(self.path, self.indexes_root)
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


def _expand_jsonl_with_specs(spec) -> tuple[list[str], list[str]]:
    paths = []
    specs = []
    for raw in _flatten_path_spec(spec):
        expanded = list(expand_sharded_filepaths(raw))
        paths.extend(expanded)
        specs.extend([raw] * len(expanded))
    return paths, specs


def _expand_tars(spec) -> list[str]:
    return [p for raw in _flatten_path_spec(spec) for p in expand_sharded_filepaths(raw)]


def _resolve_relative_input_cfg_paths(config, containing_dir: Path) -> None:
    """Resolve nested YAML references relative to the file that declares them."""
    if isinstance(config, (list, ListConfig)):
        for item in config:
            _resolve_relative_input_cfg_paths(item, containing_dir)
        return
    if not isinstance(config, (dict, DictConfig)):
        return

    nested = config.get("input_cfg")
    if isinstance(nested, (str, Path)):
        nested_path = str(nested)
        if "://" not in nested_path and not Path(nested_path).is_absolute():
            config["input_cfg"] = str(containing_dir / nested_path)
    else:
        _resolve_relative_input_cfg_paths(nested, containing_dir)


def _load_input_cfg(path: str | Path, data_blend_dir: str | Path | None = None):
    config = OmegaConf.load(str(path))
    if data_blend_dir is not None:
        root = OmegaConf.create({"data_blend_dir": str(data_blend_dir)})
        root.input_cfg = config
        config = root.input_cfg
    _resolve_relative_input_cfg_paths(config, Path(path).parent)
    return config


def _resolve_input_cfg(val, data_blend_dir: str | Path | None = None) -> ListConfig | None:
    """``input_cfg`` may be inline or a path to a YAML file. Materialize it."""
    if isinstance(val, (list, ListConfig)):
        return val
    if isinstance(val, (str, Path)):
        return _load_input_cfg(val, data_blend_dir)
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


def _discover_keys(
    entry,
    jobs: list[IndexJob],
    indexes_root: Optional[str],
    data_blend_dir: str | Path | None = None,
) -> None:
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
    if (sub := _resolve_input_cfg(entry.get("input_cfg"), data_blend_dir)) is not None:
        discover(sub, jobs, indexes_root, data_blend_dir=data_blend_dir)


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


def _discover_share_gpt_webdataset(
    data_dir,
    jobs: list[IndexJob],
    indexes_root: Optional[str],
    *,
    index_version: int,
) -> None:
    """
    Match NeMoMultimodalConversationShareGPTWebdatasetAdapter shard discovery.

    WDS v2 requires bounded discovery through ``wids-meta.json`` or
    ``.nv-meta/split.yaml``. Legacy v1 retains its recursive fallback.
    """
    if data_dir is None:
        return

    if index_version not in (1, 2):
        raise ValueError(f"wds_sample_index_version must be 1 or 2, got {index_version!r}")
    kind = WDS_TAR_V2 if index_version == 2 else WDS_TAR

    for raw in _flatten_path_spec(data_dir):
        root = Path(raw)
        for tar_path in discover_webdataset_shards(raw, require_catalog=index_version == 2):
            jobs.append(IndexJob(tar_path, kind, indexes_root))

        # Preserve the previous behavior for optional root-level sidecar
        # manifests without recursively indexing unrelated metadata files.
        if root.is_dir():
            for jsonl_path in sorted(root.glob("*.jsonl")):
                jobs.append(IndexJob(str(jsonl_path), JSONL, indexes_root))


def discover(
    entry,
    jobs: list[IndexJob],
    indexes_root: Optional[str] = None,
    *,
    data_blend_dir: str | Path | None = None,
) -> None:
    """Walk one entry of an ``input_cfg`` and append every required IndexJob."""
    if isinstance(entry, (list, ListConfig)):
        for sub in entry:
            discover(sub, jobs, indexes_root, data_blend_dir=data_blend_dir)
        return
    if not isinstance(entry, (dict, DictConfig)):
        return

    # Per-entry override: a nested entry can carry its own ``indexes_root``.
    indexes_root = entry.get("indexes_root", indexes_root)

    typ = entry.get("type")
    if typ is None:
        # Top-level wrapper (``input_cfg: [...]``) — recurse into every value.
        for v in entry.values():
            discover(v, jobs, indexes_root, data_blend_dir=data_blend_dir)
        return

    if typ in _NO_INDEX_TYPES:
        return

    if typ == "group" or typ in _TRANSFORM_TYPES:
        # Group and transform passthroughs: dispatch by keys.
        _discover_keys(entry, jobs, indexes_root, data_blend_dir)
        return

    if typ in ("nemo", "nemo_tarred", "multimodal_conversation", "share_gpt"):
        raw_manifests = entry.get("manifest_filepath")
        raw_tars = entry.get("tarred_audio_filepaths")
        collection_mode = typ == "share_gpt" and entry.get("tar_lookup_mode") == "collection"
        if collection_mode and raw_manifests is None:
            raise ValueError("ShareGPT collection mode requires manifest_filepath")
        if collection_mode and raw_tars is None:
            raise ValueError("ShareGPT collection mode requires tarred_audio_filepaths")
        for p in _expand_jsonl(raw_manifests):
            jobs.append(IndexJob(p, JSONL, indexes_root))
        for p in _expand_tars(raw_tars):
            jobs.append(IndexJob(p, NEMO_TAR, indexes_root))
        if collection_mode:
            route_path = entry.get("tar_routing_filepath")
            legacy_route_path = entry.get("tar_routing_index")
            if route_path and legacy_route_path and str(route_path) != str(legacy_route_path):
                raise ValueError("tar_routing_filepath and tar_routing_index disagree")
            route_path = route_path or legacy_route_path
            if not isinstance(route_path, (str, Path)) or not str(route_path).endswith(".sgroute"):
                raise ValueError("ShareGPT collection mode requires tar_routing_filepath with the .sgroute suffix")
            manifests, manifest_specs = _expand_jsonl_with_specs(raw_manifests)
            tars = _expand_tars(raw_tars)
            jobs.append(
                IndexJob(
                    str(route_path),
                    SHAREGPT_ROUTE,
                    indexes_root,
                    manifest_paths=tuple(manifests),
                    tar_paths=tuple(tars),
                    manifest_specs=tuple(manifest_specs),
                    audio_prefix_map=dict(entry.get("audio_path_prefix_map") or {}),
                    audio_placeholders=tuple(entry.get("audio_placeholders") or ()),
                )
            )

        return
    if typ == "share_gpt_webdataset":
        # Layout: data_dir/wids-meta.json or data_dir/.nv-meta/split.yaml.
        if entry.get("data_dir") is None and int(entry.get("wds_sample_index_version", 1)) == 2:
            raise ValueError("WDS v2 requires share_gpt_webdataset.data_dir")
        _discover_share_gpt_webdataset(
            entry.get("data_dir"),
            jobs,
            indexes_root,
            index_version=int(entry.get("wds_sample_index_version", 1)),
        )
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

    if typ in ("txt_jsonl", "nemotron_text_converation", "materialized_sft_messages"):
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


def _remove_sidecars_for_rebuild(job: IndexJob) -> None:
    idx_path = Path(job.idx_path())
    idx_path.unlink(missing_ok=True)
    if job.kind == WDS_TAR_V2:
        wds_v2_metadata_path(idx_path).unlink(missing_ok=True)


def _build_one(job: IndexJob, *, force: bool = False) -> tuple[IndexJob, str]:
    """Run the right indexer for *job*. Returns (job, status)."""
    from lhotse.indexing import create_jsonl_index
    from lhotse.indexing import create_tar_index as create_wds_tar_index

    idx = job.idx_path()
    # Ensure the parent directory exists for mirrored layouts.
    idx_parent = Path(idx).parent
    if not str(idx).startswith(("ais://", "s3://", "http://", "https://", "gs://")):
        idx_parent.mkdir(parents=True, exist_ok=True)

    if force:
        _remove_sidecars_for_rebuild(job)

    if job.kind == JSONL:
        create_jsonl_index(job.path, output_path=idx)
    elif job.kind == WDS_TAR:
        create_wds_tar_index(job.path, output_path=idx)
    elif job.kind == NEMO_TAR:
        # NeMo's create_tar_index has a (tar_path, idx_path) signature.
        create_nemo_tar_index(job.path, idx)
    elif job.kind == WDS_TAR_V2:
        create_wds_v2_tar_index(job.path, idx_path=idx)
    elif job.kind == SHAREGPT_ROUTE:
        ensure_sharegpt_route(
            idx,
            manifest_paths=job.manifest_paths,
            tar_paths=job.tar_paths,
            manifest_specs=job.manifest_specs,
            indexes_root=job.indexes_root,
            audio_prefix_map=job.audio_prefix_map,
            audio_placeholders=job.audio_placeholders,
            build_if_missing=True,
        )
    else:
        raise ValueError(f"Unknown index kind: {job.kind!r}")
    return job, "built"


def _source_size(path: str) -> int:
    source_path = path
    if source_path.startswith("s3://"):
        from lhotse.audio.source import resolve_s3_to_local_mirror

        source_path = resolve_s3_to_local_mirror(source_path)
    if source_path.startswith(("ais://", "s3://")):
        from lhotse.ais import AISRangeReader

        with AISRangeReader(source_path) as source:
            return int(source.size)
    return os.path.getsize(source_path)


def _validate_legacy_sidecar(job: IndexJob) -> None:
    from lhotse.indexing import read_index

    idx_path = Path(job.idx_path())
    offsets = read_index(idx_path)
    if offsets.shape[0] < 1:
        raise ValueError(f"Index contains no source-size sentinel: {idx_path}")
    if offsets.shape[0] > 1 and (offsets[1:] < offsets[:-1]).any():
        raise ValueError(f"Index offsets are not monotonic: {idx_path}")
    source_size = _source_size(job.path)
    if int(offsets[-1]) != source_size:
        raise ValueError(f"Index sentinel mismatch for {job.path}: index={int(offsets[-1])}, source={source_size}")

    if not job.path.startswith(("ais://", "s3://")):
        source_stat = Path(job.path).stat()
        if source_stat.st_mtime_ns > idx_path.stat().st_mtime_ns:
            raise ValueError(f"Indexed source is newer than sidecar: {job.path}")


def _is_indexed(job: IndexJob) -> bool:
    """Return true only after validating sidecar layout and source identity."""
    try:
        if job.kind == SHAREGPT_ROUTE:
            ensure_sharegpt_route(
                job.idx_path(),
                manifest_paths=job.manifest_paths,
                tar_paths=job.tar_paths,
                manifest_specs=job.manifest_specs,
                indexes_root=job.indexes_root,
                audio_prefix_map=job.audio_prefix_map,
                audio_placeholders=job.audio_placeholders,
                build_if_missing=False,
            )
        elif job.kind == WDS_TAR_V2:
            validate_wds_v2_tar_index(job.path, idx_path=job.idx_path())
        else:
            _validate_legacy_sidecar(job)
        return True
    except (OSError, ValueError):
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
    "--kind",
    "kinds",
    type=click.Choice(INDEX_KINDS),
    multiple=True,
    help="Restrict building to an index kind. Repeat to select multiple kinds.",
)
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
@click.option(
    "--data-blend-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Resolve ${data_blend_dir} in nested input_cfg references.",
)
def main(
    input_cfgs: tuple[str, ...],
    force: bool,
    workers: int,
    dry_run: bool,
    kinds: tuple[str, ...],
    executor: str,
    indexes_root: Optional[str],
    data_blend_dir: Optional[str],
):
    """
    Build .idx sidecars for every JSONL/tar referenced by INPUT_CFGS.

    INPUT_CFGS are NeMo Lhotse dataloading configs (``input_cfg`` YAML).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    jobs: list[IndexJob] = []
    for cfg_path in input_cfgs:
        cfg = _load_input_cfg(cfg_path, data_blend_dir)
        discover(
            cfg,
            jobs,
            indexes_root=indexes_root,
            data_blend_dir=data_blend_dir,
        )

    # Deduplicate while preserving order.
    seen: set[tuple[str, str, Optional[str]]] = set()
    unique: list[IndexJob] = []
    for j in jobs:
        key = (j.path, j.kind, j.indexes_root)
        if key not in seen:
            seen.add(key)
            unique.append(j)

    if kinds:
        selected = frozenset(kinds)
        unique = [job for job in unique if job.kind in selected]

    all_todo = unique if force else [job for job in unique if not _is_indexed(job)]
    route_todo = [job for job in all_todo if job.kind == SHAREGPT_ROUTE]
    todo = [job for job in all_todo if job.kind != SHAREGPT_ROUTE]
    skipped = len(unique) - len(all_todo)

    logging.info("Discovered %d files (%d already indexed, %d to build).", len(unique), skipped, len(all_todo))

    if dry_run or not all_todo:
        for j in all_todo:
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
        futures = {ex.submit(_build_one, j, force=force): j for j in todo}
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

    if not failures:
        for job in route_todo:
            try:
                _build_one(job, force=force)
            except Exception as error:
                failures.append((job, error))
                logging.error("  [FAIL] %s %s: %s", job.kind, job.path, error)

    if failures:
        logging.error("\n%d index build(s) failed:", len(failures))
        for j, e in failures:
            logging.error("  %s (%s): %s", j.path, j.kind, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
