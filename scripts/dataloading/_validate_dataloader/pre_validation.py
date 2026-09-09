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
"""Static pre-validation checks for a train_ds-shaped Lhotse dataloader config.

Run as either a function (``run_pre_validation(cfg)``) or a CLI
(``python pre_validation.py --config ... --output-dir ...``). All checks
operate on the resolved OmegaConf node — no iteration, no GPUs, no
SLURM. Intended runtime: < 5 s on a typical SALM ``train_ds`` config.

The output is a structured report (``pre_validation.json``) listing each
check's ``PASS``/``WARN``/``FAIL`` status. Exit code is ``0`` iff no
``FAIL`` checks remain after applying ``--ignore-fail`` overrides.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import click
from omegaconf import DictConfig, ListConfig, OmegaConf

LOG = logging.getLogger(__name__)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

_NON_INT_SEED_VALUES = {"randomized", "trng", "trng_initial", None}


@dataclass
class CheckResult:
    check_id: str
    severity: str  # FAIL or WARN — the worst this check is permitted to emit
    status: str  # PASS | WARN | FAIL | SKIP
    detail: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class PreValidationReport:
    checks: list[CheckResult]
    summary: dict

    def to_dict(self):
        return {
            "checks": {
                c.check_id: {"status": c.status, "severity": c.severity, "detail": c.detail, **c.extra}
                for c in self.checks
            },
            "summary": self.summary,
        }

    @property
    def all_passed(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #


def run_pre_validation(cfg: DictConfig, *, ignore_fail: Iterable[str] = ()) -> PreValidationReport:
    """Run every registered check against ``cfg`` (a train_ds-shaped node).

    Set ``ignore_fail`` to a list of check IDs to downgrade their ``FAIL``
    outcome to ``WARN``. Always run every check — never short-circuit —
    so the user sees the full picture.
    """
    ignore = set(ignore_fail)
    results: list[CheckResult] = []
    for check_id, severity, fn in _REGISTRY:
        try:
            status, detail, extra = fn(cfg)
        except Exception as e:  # pragma: no cover — safety net
            status, detail, extra = FAIL, f"check raised {type(e).__name__}: {e}", {}
        if check_id in ignore and status == FAIL:
            status = WARN
            detail = f"(downgraded to WARN via --ignore-fail) {detail}"
        results.append(CheckResult(check_id, severity, status, detail, extra))
    summary = {
        "total": len(results),
        "pass": sum(1 for r in results if r.status == PASS),
        "warn": sum(1 for r in results if r.status == WARN),
        "fail": sum(1 for r in results if r.status == FAIL),
        "skip": sum(1 for r in results if r.status == SKIP),
    }
    return PreValidationReport(checks=results, summary=summary)


# --------------------------------------------------------------------------- #
# Individual checks. Each returns (status, detail, extra_fields).
# --------------------------------------------------------------------------- #


def _check_seed_int(cfg: DictConfig):
    seed = cfg.get("seed", None)
    if isinstance(seed, int):
        return PASS, f"seed={seed}", {}
    if seed in _NON_INT_SEED_VALUES:
        return (
            FAIL,
            (
                f"train_ds.seed is {seed!r}; must be an integer for reproducibility across "
                "launches and determinism re-runs."
            ),
            {},
        )
    return FAIL, f"train_ds.seed={seed!r} (type={type(seed).__name__}); must be int", {}


def _check_shard_seed_int(cfg: DictConfig):
    shard_seed = cfg.get("shard_seed", None)
    if isinstance(shard_seed, int):
        return PASS, f"shard_seed={shard_seed}", {}
    return (
        FAIL,
        (
            f"train_ds.shard_seed={shard_seed!r}; must be an integer. "
            "LazyIteratorMultiplexer raises under multi-shard + 'randomized'."
        ),
        {},
    )


def _check_stateful_on(cfg: DictConfig):
    if cfg.get("use_stateful_dataloader", False) is True:
        return PASS, "", {}
    return (
        FAIL,
        ("use_stateful_dataloader is not True; resumability validation requires the " "StatefulDataLoader path."),
        {},
    )


def _check_indexed_implies_root(cfg: DictConfig):
    indexed = cfg.get("indexed", False)
    indexes_root = cfg.get("indexes_root", None)
    if not indexed:
        return SKIP, "train_ds.indexed != True; check not applicable", {}
    if indexes_root in (None, "", "null"):
        return (
            FAIL,
            (
                "train_ds.indexed=True but indexes_root is unset. Without indexes_root, "
                "LazyIndexedSharIterator falls back to looking next to (typically remote) "
                "data files."
            ),
            {},
        )
    return PASS, f"indexes_root={indexes_root}", {}


def _check_indexes_root_exists(cfg: DictConfig):
    indexes_root = cfg.get("indexes_root", None)
    if not indexes_root:
        return SKIP, "indexes_root unset; check not applicable", {}
    p = Path(indexes_root)
    if p.exists():
        return PASS, f"{indexes_root} exists", {}
    # Locally on a developer laptop the path is typically cluster-specific; downgrade to WARN.
    return (
        WARN,
        (
            f"indexes_root={indexes_root!r} does not exist on this host. "
            "Expected on cluster; downgraded to WARN locally."
        ),
        {},
    )


def _check_idx_files_present(cfg: DictConfig):
    indexes_root = cfg.get("indexes_root", None)
    if not indexes_root or not Path(indexes_root).exists():
        return SKIP, "indexes_root not present locally; cluster-side check only", {}
    try:
        from lhotse.indexing import index_exists, index_file_path
    except ImportError as e:
        return WARN, f"lhotse.indexing import failed: {e}", {}
    try:
        from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths
    except ImportError:
        expand_sharded_filepaths = None

    leaves = _collect_leaf_paths(cfg)
    if not leaves:
        return WARN, "no leaf data paths found under input_cfg", {}

    # Expand ``_OP_N..M_CL_`` shard patterns; sample 2 shards per leaf so
    # we cover every source without doing thousands of stat()s.
    expanded: list[str] = []
    for raw in leaves:
        if expand_sharded_filepaths is not None:
            try:
                shards = expand_sharded_filepaths(raw)
            except Exception:
                shards = [raw]
        else:
            shards = [raw]
        expanded.extend(shards[:2])

    missing: list[str] = []
    truncated: list[str] = []
    for shard_path in expanded[:64]:  # global cap, just in case
        idx_path = str(index_file_path(shard_path, indexes_root=indexes_root))
        if not Path(idx_path).exists():
            missing.append(idx_path)
        elif not index_exists(shard_path, idx_path):
            truncated.append(idx_path)
    if missing or truncated:
        detail = f"{len(missing)} missing, {len(truncated)} truncated of {len(expanded[:64])} sampled"
        return FAIL, detail, {"missing": missing[:5], "truncated": truncated[:5]}
    return PASS, f"sampled {len(expanded[:64])} .idx files across {len(leaves)} leaves; all valid", {}


def _check_constant_time_leaves(cfg: DictConfig):
    """The user's note: O(1) state-dict restore requires constant-time leaves
    in BOTH map-style (force_map_dataset=True) and iterable-style. So this
    check fires whenever use_stateful_dataloader is on, regardless of
    force_map_dataset. Implemented statically: every leaf type must be
    one that admits indexed mode, AND the indexed flag must propagate
    (top-level ``indexed: true`` OR per-leaf override)."""
    stateful = cfg.get("use_stateful_dataloader", False) is True
    top_indexed = cfg.get("indexed", False) is True
    non_indexable: list[dict] = []
    streaming: list[dict] = []
    for leaf in _iter_leaf_nodes(cfg):
        typ = leaf.get("type")
        if typ in _STREAMING_ONLY_TYPES:
            non_indexable.append({"type": typ, "corpus": leaf.get("corpus")})
            continue
        leaf_indexed = leaf.get("indexed", top_indexed) is True
        if not leaf_indexed:
            streaming.append({"type": typ, "corpus": leaf.get("corpus")})
    severity_status = FAIL if stateful else WARN
    if non_indexable or streaming:
        n = len(non_indexable) + len(streaming)
        detail = (
            f"{n} leaf source(s) lack constant-time access "
            f"({len(non_indexable)} non-indexable type, {len(streaming)} streaming-mode). "
            "Resume falls back to O(N) replay; with force_map_dataset=False they also leak "
            "across ranks."
        )
        return (
            severity_status,
            detail,
            {
                "non_indexable": non_indexable[:5],
                "streaming": streaming[:5],
            },
        )
    return PASS, "all leaf sources admit constant-time access", {}


def _check_mux_weights_sum(cfg: DictConfig):
    """A multiplexer in NeMo configs is any list of dicts where each entry
    carries a ``weight`` key. Validate that weights are positive finite floats."""
    bad: list[dict] = []
    for path, mux_entries in _iter_mux_groups(cfg):
        total = 0.0
        for i, e in enumerate(mux_entries):
            w = e.get("weight")
            if not isinstance(w, (int, float)) or w <= 0 or not _isfinite(w):
                bad.append({"path": f"{path}[{i}]", "weight": w, "type": e.get("type")})
            else:
                total += float(w)
        if total <= 0:
            bad.append({"path": path, "weights_sum": total})
    if bad:
        return FAIL, f"{len(bad)} bad weight(s) found", {"examples": bad[:5]}
    return PASS, "all mux weights sum to finite positive", {}


def _check_mux_seed_not_randomized(cfg: DictConfig):
    if cfg.get("force_map_dataset", True) is not False:
        return SKIP, "force_map_dataset != False; check not applicable", {}
    shard_seed = cfg.get("shard_seed")
    if isinstance(shard_seed, int):
        return PASS, f"shard_seed={shard_seed}", {}
    return (
        FAIL,
        (
            f"force_map_dataset=False but shard_seed={shard_seed!r}. "
            "LazyIteratorMultiplexer raises ValueError under multi-shard with "
            "shard_seed='randomized'."
        ),
        {},
    )


def _check_slice_length_vs_indexed(cfg: DictConfig):
    if not cfg.get("indexed", False):
        return SKIP, "train_ds.indexed != True; check not applicable", {}
    offenders: list[dict] = []
    for leaf in _iter_leaf_nodes(cfg):
        if leaf.get("slice_length") is not None:
            offenders.append({"type": leaf.get("type"), "corpus": leaf.get("corpus")})
    if offenders:
        return (
            FAIL,
            (
                f"{len(offenders)} source(s) set slice_length with indexed=True. "
                "Lhotse rejects: \"'slice_length' is not supported with indexed=True\"."
            ),
            {"examples": offenders[:5]},
        )
    return PASS, "", {}


def _check_cut_map_fns_vs_indexed(cfg: DictConfig):
    if not cfg.get("indexed", False):
        return SKIP, "train_ds.indexed != True; check not applicable", {}
    offenders: list[dict] = []
    for leaf in _iter_leaf_nodes(cfg):
        if leaf.get("cut_map_fns"):
            offenders.append({"type": leaf.get("type"), "corpus": leaf.get("corpus")})
    if offenders:
        return (
            FAIL,
            (
                f"{len(offenders)} source(s) set cut_map_fns with indexed=True. "
                "Lhotse rejects: \"'cut_map_fns' is not supported with indexed=True\"."
            ),
            {"examples": offenders[:5]},
        )
    return PASS, "", {}


def _check_lambda_in_pipeline(cfg: DictConfig):
    """Heuristic: scan the YAML-resolved config for strings containing
    '<lambda>' or 'lambda '. Real lambdas in YAML can't round-trip but
    some configs use ``_target_: somemodule:somefn`` strings — we look
    for the textual hint."""
    blob = OmegaConf.to_yaml(cfg, resolve=False)
    hits: list[str] = []
    for line in blob.splitlines():
        if "<lambda>" in line or "lambda:" in line or " lambda " in line:
            hits.append(line.strip())
    if hits:
        return WARN, f"{len(hits)} possible lambda reference(s) in config", {"examples": hits[:5]}
    return PASS, "no lambda references found", {}


def _check_bucketer_buffer(cfg: DictConfig):
    if not cfg.get("use_bucketing", False):
        return SKIP, "use_bucketing != True; check not applicable", {}
    n_buckets = cfg.get("num_buckets", 0)
    buffer_size = cfg.get("bucket_buffer_size", 0)
    if not n_buckets or not buffer_size:
        return WARN, f"num_buckets={n_buckets}, bucket_buffer_size={buffer_size}", {}
    ratio = buffer_size / max(n_buckets, 1)
    if ratio < 10:
        return (
            WARN,
            (
                f"bucket_buffer_size={buffer_size} is < 10×num_buckets ({n_buckets}). "
                "Low buffers can cause BucketsDontHaveEnoughData mid-run."
            ),
            {"ratio": ratio},
        )
    return PASS, f"bucket_buffer_size={buffer_size}, num_buckets={n_buckets}, ratio={ratio:.1f}", {}


def _check_multi_config_flags(cfg: DictConfig):
    """multi_config = True means input_cfg is a list of per-sub-config blocks.
    Top-level ``indexed`` / ``indexes_root`` only flow into sub-configs via
    the ``overwriting_opts`` list at NeMo_resumable/.../dataloader.py:455-473.
    The 2026-05 fixes added them to that list; we verify the runtime
    config doesn't paper over a missing entry per-sub-config."""
    if not cfg.get("multi_config", False):
        return SKIP, "multi_config != True; check not applicable", {}
    # Static structural check: at least one sub-config must carry the
    # indexed/indexes_root flags (or the top-level must have them so they propagate).
    top_indexed = cfg.get("indexed")
    top_root = cfg.get("indexes_root")
    if top_indexed is not None and top_root is not None:
        return PASS, "top-level indexed+indexes_root will propagate via overwriting_opts", {}
    sub_cfgs = cfg.get("input_cfg") or []
    if not isinstance(sub_cfgs, (list, ListConfig)):
        return WARN, "multi_config=True but input_cfg is not a list", {}
    missing = [
        i
        for i, sc in enumerate(sub_cfgs)
        if isinstance(sc, (dict, DictConfig)) and (sc.get("indexed") is None or sc.get("indexes_root") is None)
    ]
    if missing:
        return (
            FAIL,
            (
                f"multi_config=True; {len(missing)} sub-config(s) missing indexed/indexes_root "
                "and top-level doesn't supply both."
            ),
            {"indices": missing[:5]},
        )
    return PASS, "every sub-config sets indexed and indexes_root", {}


def _check_text_fields(cfg: DictConfig):
    """Best-effort: only run if at least one nemo_tarred leaf is reachable
    locally. Verifying ``text_field`` requires reading a manifest line."""
    # In v1 we just verify the field name is one of the well-known
    # candidates. Manifest-line inspection requires network/cluster access.
    valid = {"text", "answer", "transcript", "text_pnc", "text_normalized"}
    suspicious: list[dict] = []
    tf = cfg.get("text_field")
    if tf is not None and tf not in valid:
        suspicious.append({"path": "train_ds.text_field", "value": tf})
    for leaf in _iter_leaf_nodes(cfg):
        if leaf.get("type") == "nemo_tarred":
            tf = leaf.get("text_field")
            if tf is not None and tf not in valid:
                suspicious.append({"corpus": leaf.get("corpus"), "value": tf})
    if suspicious:
        return (
            WARN,
            f"{len(suspicious)} unusual text_field value(s); verify against shard 0",
            {"examples": suspicious[:5], "known_valid": sorted(valid)},
        )
    return PASS, "text_field values match known-valid set", {}


def _check_world_size_divides_workers(cfg: DictConfig):
    """Heuristic only — we don't yet know the runtime ``num-ranks``; emit
    INFO showing how many shards each leaf has so the user can eyeball it."""
    counts: list[dict] = []
    for leaf in _iter_leaf_nodes(cfg):
        n = _count_shards(leaf)
        if n is not None:
            counts.append({"corpus": leaf.get("corpus") or leaf.get("type"), "shards": n})
    if not counts:
        return SKIP, "no leaf-shard counts derivable from config", {}
    min_shards = min(c["shards"] for c in counts)
    if min_shards < 8:  # arbitrary "small enough to worry about" heuristic
        return (
            WARN,
            f"smallest source has only {min_shards} shards; verify (num_ranks × num_workers) ≤ this",
            {"counts": counts[:10]},
        )
    return PASS, f"smallest source has {min_shards} shards", {"counts": counts[:10]}


# --------------------------------------------------------------------------- #
# Check registry. Order = output order.
# --------------------------------------------------------------------------- #


_REGISTRY: list[tuple[str, str, Callable[[DictConfig], tuple[str, str, dict]]]] = [
    ("seed-int", FAIL, _check_seed_int),
    ("shard-seed-int", FAIL, _check_shard_seed_int),
    ("stateful-on", FAIL, _check_stateful_on),
    ("indexed-implies-root", FAIL, _check_indexed_implies_root),
    ("indexes-root-exists", FAIL, _check_indexes_root_exists),
    ("idx-files-present", FAIL, _check_idx_files_present),
    ("constant-time-leaves", FAIL, _check_constant_time_leaves),
    ("mux-weights-sum", FAIL, _check_mux_weights_sum),
    ("mux-seed-not-randomized", FAIL, _check_mux_seed_not_randomized),
    ("slice-length-vs-indexed", FAIL, _check_slice_length_vs_indexed),
    ("cut-map-fns-vs-indexed", FAIL, _check_cut_map_fns_vs_indexed),
    ("lambda-in-pipeline", WARN, _check_lambda_in_pipeline),
    ("bucketer-buffer", WARN, _check_bucketer_buffer),
    ("multi-config-flags", FAIL, _check_multi_config_flags),
    ("text-fields", WARN, _check_text_fields),
    ("world-size-divides-workers", WARN, _check_world_size_divides_workers),
]


# --------------------------------------------------------------------------- #
# Static topology helpers.
# --------------------------------------------------------------------------- #


# Types that read indexable underlying data.
_LEAF_TYPES = frozenset(
    {
        "lhotse_shar",
        "nemo",
        "nemo_tarred",
        "multimodal_conversation",
        "share_gpt",
    }
)

# Types that don't admit constant-time access at all.
_STREAMING_ONLY_TYPES = frozenset(
    {
        "txt",
        "txt_pair",
        "parquet",
        "multi_speaker_simulator",
    }
)

# Transparent passthrough types — recurse into input_cfg.
_TRANSFORM_TYPES = frozenset(
    {
        "lhotse_as_conversation",
        "sqa_as_conversation",
        "s2s_as_conversation",
        "s2s_duplex_overlap_as_s2s_duplex",
        "s2s_duplex_reverse_role",
        "lhotse_magpietts_data_as_continuation",
        "nemo_tarred_to_duplex",
        "group",
    }
)


def _iter_leaf_nodes(cfg: DictConfig) -> Iterable[DictConfig]:
    """Yield each leaf-source dict reachable from ``cfg.input_cfg``."""
    yield from _walk(cfg.get("input_cfg"))


def _walk(node: Any) -> Iterable[DictConfig]:
    if node is None:
        return
    if isinstance(node, (list, ListConfig)):
        for sub in node:
            yield from _walk(sub)
        return
    if isinstance(node, str):
        # input_cfg reference to another YAML file — try to load it.
        loaded = _try_load_yaml(node)
        if loaded is not None:
            yield from _walk(loaded)
        return
    if not isinstance(node, (dict, DictConfig)):
        return
    typ = node.get("type")
    if typ in _LEAF_TYPES or typ in _STREAMING_ONLY_TYPES:
        yield node
        return
    if typ in _TRANSFORM_TYPES or typ is None:
        if "input_cfg" in node:
            yield from _walk(node["input_cfg"])
        return
    # Unknown type — yield it so it's at least counted, but caller
    # should be defensive about its keys.
    yield node


def _iter_mux_groups(cfg: DictConfig) -> Iterable[tuple[str, list]]:
    """Yield ``(path, list-of-entries)`` for each input_cfg list whose entries
    carry a ``weight`` field (= an implicit multiplexer)."""
    yield from _walk_mux(cfg.get("input_cfg"), path="train_ds.input_cfg")


def _walk_mux(node: Any, path: str) -> Iterable[tuple[str, list]]:
    if node is None:
        return
    if isinstance(node, (list, ListConfig)):
        entries = [e for e in node if isinstance(e, (dict, DictConfig))]
        weighted = [e for e in entries if "weight" in e]
        if weighted and len(weighted) > 1:
            yield path, list(weighted)
        for i, sub in enumerate(node):
            yield from _walk_mux(sub, path=f"{path}[{i}]")
        return
    if isinstance(node, str):
        loaded = _try_load_yaml(node)
        if loaded is not None:
            yield from _walk_mux(loaded, path=f"{path}<{Path(node).name}>")
        return
    if isinstance(node, (dict, DictConfig)) and "input_cfg" in node:
        yield from _walk_mux(node["input_cfg"], path=f"{path}.input_cfg")


def _collect_leaf_paths(cfg: DictConfig) -> list[str]:
    """Flat list of every shard path referenced from leaf sources, in YAML order."""
    out: list[str] = []
    for leaf in _iter_leaf_nodes(cfg):
        for path in _leaf_to_paths(leaf):
            out.append(path)
    return out


def _leaf_to_paths(leaf: DictConfig) -> list[str]:
    """Resolve the shar/manifest paths inside ``leaf`` into flat strings."""
    paths: list[str] = []
    if shar := leaf.get("shar_path"):
        if isinstance(shar, (dict, DictConfig)):
            for key in ("cuts", "recording"):
                v = shar.get(key)
                if isinstance(v, str):
                    paths.append(v)
        elif isinstance(shar, str):
            paths.append(shar)
    if mfp := leaf.get("manifest_filepath"):
        paths.extend(_flatten_str(mfp))
    if taf := leaf.get("tarred_audio_filepaths"):
        paths.extend(_flatten_str(taf))
    if cuts := leaf.get("cuts_path"):
        paths.extend(_flatten_str(cuts))
    return paths


def _count_shards(leaf: DictConfig) -> Optional[int]:
    """Best-effort shard count from a leaf's ``_OP_N..M_CL_`` patterns."""
    import re

    paths = _leaf_to_paths(leaf)
    if not paths:
        return None
    rx = re.compile(r"_OP_(\d+)\.\.(\d+)_CL_")
    total = 0
    for p in paths:
        m = rx.search(str(p))
        if m:
            total += int(m.group(2)) - int(m.group(1)) + 1
    return total or None


def _flatten_str(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, ListConfig)):
        out: list[str] = []
        for item in v:
            out.extend(_flatten_str(item))
        return out
    return []


def _try_load_yaml(path: str) -> Optional[Any]:
    if not path or not isinstance(path, str):
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return OmegaConf.load(str(p))
    except Exception as e:
        LOG.debug("failed to load %s: %s", path, e)
        return None


def _isfinite(x: float) -> bool:
    import math

    return math.isfinite(x)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


@click.command(help=__doc__)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Training YAML containing data.train_ds.",
)
@click.option(
    "--data-blend-dir", default=None, help="Substituted into ${data_blend_dir} in the config (optional locally)."
)
@click.option("--section", default="train_ds", show_default=True, help="Which data.* section to validate.")
@click.option("--output-dir", default=None, type=click.Path(), help="Write pre_validation.json under this directory.")
@click.option(
    "--ignore-fail",
    multiple=True,
    default=(),
    help="Repeatable: check IDs whose FAIL outcome should be downgraded to WARN.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose logs.")
def cli(
    config_path: str,
    data_blend_dir: Optional[str],
    section: str,
    output_dir: Optional[str],
    ignore_fail: tuple,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = OmegaConf.load(config_path)
    if data_blend_dir is not None:
        cfg.data_blend_dir = data_blend_dir
    OmegaConf.resolve(cfg)
    section_cfg = cfg.data[section]
    report = run_pre_validation(section_cfg, ignore_fail=ignore_fail)

    # Pretty-print to stdout.
    print(f"\n=== pre-validation ({len(report.checks)} checks) ===")
    for c in report.checks:
        marker = {PASS: "  PASS", WARN: "  WARN", FAIL: "  FAIL", SKIP: "  skip"}[c.status]
        print(f"{marker}  [{c.check_id}] {c.detail}")
    print(f"\nsummary: {report.summary}")
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pre_validation.json").write_text(json.dumps(report.to_dict(), indent=2))
        print(f"wrote {out / 'pre_validation.json'}")
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    cli()
