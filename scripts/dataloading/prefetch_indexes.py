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
Copy existing ``.idx`` sidecars from their source locations into a local
mirrored ``indexes_root``.

Use this when your data lives on shared storage (NFS, S3, AIStore) and you
want a local-disk copy of the indexes for fast random access during
training, without ever touching the data files themselves.

The script walks an arbitrary NeMo Lhotse ``input_cfg`` YAML (same machinery
as ``build_indexes.py``), enumerates every ``.idx`` file the dataloader will
need, and downloads each into ``<indexes_root>/<rel-path>.idx`` preserving
the data files' directory structure. Source paths are read via lhotse's
``open_best``, which routes ``ais://``, ``s3://``, ``http(s)://``, and local
paths to the correct backend.

Examples::

    # Local data, mirror indexes onto a fast local SSD.
    python scripts/dataloading/prefetch_indexes.py \\
        --indexes-root /scratch/idx \\
        path/to/input_cfg.yaml

    # Indexes live next to data on AIStore; pull them down.
    AIS_ENDPOINT=http://aistore.example.com \\
        python scripts/dataloading/prefetch_indexes.py \\
            --indexes-root /scratch/idx \\
            path/to/input_cfg.yaml

    # Skip files that are already in the mirror; re-run safely.
    python scripts/dataloading/prefetch_indexes.py \\
        --indexes-root /scratch/idx --workers 16 train.yaml validation.yaml

After prefetch, point your training config at the mirror via the top-level
``indexes_root: /scratch/idx`` option (no per-source changes required).
"""

import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Optional

import click
from lhotse.indexing import index_file_path
from omegaconf import OmegaConf

# Reuse the discovery + IndexJob machinery from build_indexes.py.
sys.path.insert(0, str(Path(__file__).parent))
from build_indexes import IndexJob, discover  # type: ignore[import-not-found]


def _copy_idx(src: str, dst: str) -> None:
    """Copy a single ``.idx`` from *src* (local or URL) to *dst* (local).

    Uses lhotse's ``open_best`` so URL schemes are routed to the right
    backend (smart_open / AIStore SDK).
    """
    from lhotse.serialization import open_best

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    # Stage to a sibling tmp file then rename, so partial writes never
    # leave a half-baked .idx in place.
    tmp = f"{dst}.tmp.{Path(dst).name}.partial"
    try:
        with open_best(src, "rb") as src_f, open(tmp, "wb") as dst_f:
            shutil.copyfileobj(src_f, dst_f, length=8 * 1024 * 1024)
        Path(tmp).replace(dst)
    finally:
        # Clean up if rename never happened (exception path).
        with suppress(FileNotFoundError):
            Path(tmp).unlink()


def _is_present(local_idx: str) -> bool:
    p = Path(local_idx)
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


@click.command(context_settings={"show_default": True})
@click.argument("input_cfgs", type=click.Path(exists=True, dir_okay=False), nargs=-1, required=True)
@click.option(
    "--indexes-root",
    type=str,
    required=True,
    help="Local directory the .idx mirror is written to. The data files' directory structure is preserved underneath.",
)
@click.option(
    "--source-indexes-root",
    type=str,
    default=None,
    help=(
        "If the source ``.idx`` files do not live next to the data (e.g. they "
        "are themselves under another mirror — possibly remote), set this to "
        "that root. Defaults to ``None`` meaning sidecars are read from "
        "next to each data file."
    ),
)
@click.option("--force", is_flag=True, help="Re-download even when a non-empty mirrored .idx already exists.")
@click.option("--workers", type=int, default=8, help="Number of parallel copies.")
@click.option("--dry-run", is_flag=True, help="List the (src, dst) pairs without copying anything.")
def main(
    input_cfgs: tuple[str, ...],
    indexes_root: str,
    source_indexes_root: Optional[str],
    force: bool,
    workers: int,
    dry_run: bool,
):
    """
    Prefetch .idx sidecars referenced by INPUT_CFGS into a local mirror.

    INPUT_CFGS are NeMo Lhotse dataloading configs (``input_cfg`` YAML).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    jobs: list[IndexJob] = []
    for cfg_path in input_cfgs:
        cfg = OmegaConf.load(cfg_path)
        # Walk with no inherited indexes_root — we want the *natural* data paths,
        # then we compute (source, destination) idx paths ourselves below.
        discover(cfg, jobs, indexes_root=None)

    # Deduplicate by (data_path, kind).
    seen: set[tuple[str, str]] = set()
    unique: list[IndexJob] = []
    for j in jobs:
        key = (j.path, j.kind)
        if key not in seen:
            seen.add(key)
            unique.append(j)

    pairs: list[tuple[str, str]] = []
    for j in unique:
        src = index_file_path(j.path, source_indexes_root)
        dst = index_file_path(j.path, indexes_root)
        pairs.append((src, dst))

    todo = pairs if force else [(s, d) for (s, d) in pairs if not _is_present(d)]
    skipped = len(pairs) - len(todo)
    logging.info(
        "Discovered %d sidecars (%d already present locally, %d to copy).",
        len(pairs),
        skipped,
        len(todo),
    )

    if dry_run or not todo:
        for s, d in todo:
            logging.info("  %s  ->  %s", s, d)
        return

    # Per-file success logging is suppressed (80k-400k sidecars would swamp
    # stdout); failures are still logged inline, success emits a periodic
    # progress heartbeat plus a final summary.
    failures: list[tuple[str, str, Exception]] = []
    total = len(todo)
    log_every = max(1, min(5000, total // 20))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(_copy_idx, s, d): (s, d) for (s, d) in todo}
        done = 0
        for fut in as_completed(futures):
            done += 1
            s, d = futures[fut]
            try:
                fut.result()
            except Exception as e:
                failures.append((s, d, e))
                logging.error("  [FAIL] %s  ->  %s: %s", s, d, e)
                continue
            if done % log_every == 0 or done == total:
                logging.info(
                    "  copied %d/%d (%.1f%%)  failures=%d",
                    done,
                    total,
                    100.0 * done / total,
                    len(failures),
                )

    if failures:
        logging.error("\n%d copy operation(s) failed:", len(failures))
        for s, d, e in failures:
            logging.error("  %s -> %s: %s", s, d, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
