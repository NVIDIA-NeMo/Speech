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
"""Validate a Lhotse + indexed dataloader config end-to-end.

Per-rank entry point launched under torchrun. Builds the production Lhotse
dataloader with either a no-op ``CutIdDataset`` (fast mode) or the production
``SALMDataset`` audio/tokenization/collation path (full mode), and dumps only
per-batch IDs and counters. Phase-aware:

* ``baseline`` — iterate ``--steps`` batches from a fresh dataloader; at
  ``--checkpoint-at`` save ``dl.state_dict()`` to ``state_rank_NNN.pt``.
* ``resumed``  — load the saved state and iterate the rest; downstream
  consolidation diffs the post-checkpoint window against the baseline tail.
* ``groundtruth`` — single-rank, single-worker enumeration of every cut
  the configured input_cfg yields under force_finite + metadata_only.

Launch as a step in a multi-phase pipeline; downstream aggregator is
``_validate_dataloader/consolidate.py``.

Example::

    torchrun --standalone --nnodes=1 --nproc-per-node=4 \\
        scripts/dataloading/validate_dataloader.py \\
        --config 0909-en-only-id2.yaml \\
        --data-blend-dir /lustre/.../data_blends/ord \\
        --output-dir validation_out \\
        --phase baseline --run-idx 0 \\
        --steps 200 --checkpoint-at 100
"""

import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import click
import torch
import torch.utils.data
from omegaconf import OmegaConf

# Local helpers — same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _validate_dataloader.config_inject import (  # noqa: E402
    inject_groundtruth_flags,
    inject_missing_manifest_policy,
    inject_validator_flags,
)
from _validate_dataloader.cut_id_dataset import _validation_identity  # noqa: E402
from _validate_dataloader.full_mode import build_tokenizer as _build_tokenizer  # noqa: E402
from _validate_dataloader.full_mode import build_validation_dataset as _build_validation_dataset
from _validate_dataloader.full_mode import validate_full_batch as _validate_full_batch
from _validate_dataloader.full_stats import (  # noqa: E402
    FullValidationStats,
    configured_audio_path_resolution_modes,
    full_summary_failure_guard,
)

LOG = logging.getLogger(__name__)


PHASE_BASELINE = "baseline"
PHASE_RESUMED = "resumed"
PHASE_GROUNDTRUTH = "groundtruth"


def _ensure_validation_process_group(*, rank: int, world_size: int) -> bool:
    """Initialize the CPU process group required by per-rank state gathering."""
    if world_size <= 1:
        return False
    if not torch.distributed.is_available():
        raise RuntimeError("Distributed validation requires torch.distributed support.")
    if torch.distributed.is_initialized():
        actual_rank = torch.distributed.get_rank()
        actual_world_size = torch.distributed.get_world_size()
        if (actual_rank, actual_world_size) != (rank, world_size):
            raise RuntimeError(
                "Distributed validation process-group mismatch: "
                f"environment=(rank={rank}, world_size={world_size}) "
                f"process_group=(rank={actual_rank}, world_size={actual_world_size})."
            )
        return False

    LOG.info("initializing gloo process group for rank=%d world_size=%d", rank, world_size)
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return True


def _finish_validation_process_group(*, initialized_here: bool) -> None:
    if initialized_here and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


@click.command(help=__doc__)
@click.option(
    "--input-cfg",
    "input_cfg_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Replace data.<section>.input_cfg with this generated leaf YAML mapping/list.",
)
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--data-blend-dir",
    default=None,
    help="Substituted into ${data_blend_dir} in the config.",
)
@click.option("--section", default="train_ds", show_default=True)
@click.option("--output-dir", required=True, type=click.Path())
@click.option(
    "--phase",
    type=click.Choice([PHASE_BASELINE, PHASE_RESUMED, PHASE_GROUNDTRUTH]),
    required=True,
)
@click.option(
    "--run-idx",
    type=int,
    default=0,
    show_default=True,
    help="Which determinism re-run this is. Only used with --phase=baseline.",
)
@click.option(
    "--steps",
    type=int,
    default=200,
    show_default=True,
    help="Batches to iterate. Ignored in groundtruth phase (iterates until exhaustion).",
)
@click.option(
    "--checkpoint-at",
    type=int,
    default=-1,
    show_default=True,
    help="Step index at which to save state in baseline phase. -1 = don't save.",
)
@click.option(
    "--state-dir",
    default=None,
    type=click.Path(),
    help="In --phase=resumed: directory containing state_rank_NNN.pt files.",
)
@click.option("--force-finite/--no-force-finite", default=True, show_default=True)
@click.option("--metadata-only/--no-metadata-only", default=True, show_default=True)
@click.option(
    "--num-workers-override",
    type=int,
    default=None,
    help="Override config.{section}.num_workers.",
)
@click.option(
    "--skip-missing-manifest-entries/--fail-on-missing-manifest-entries",
    default=None,
    help=(
        "Override the missing-manifest-entry policy on the complete input graph. "
        "When omitted, preserve the recipe setting."
    ),
)
@click.option(
    "--mode",
    type=click.Choice(["fast", "full"]),
    default="fast",
    show_default=True,
    help="fast: CutIdDataset metadata path. full: production SALMDataset payload materialization.",
)
@click.option("-v", "--verbose", is_flag=True, default=False)
def cli(
    config_path: str,
    data_blend_dir: Optional[str],
    input_cfg_path: Optional[str],
    section: str,
    output_dir: str,
    phase: str,
    run_idx: int,
    steps: int,
    checkpoint_at: int,
    state_dir: Optional[str],
    force_finite: bool,
    metadata_only: bool,
    num_workers_override: Optional[int],
    skip_missing_manifest_entries: Optional[bool],
    mode: str,
    verbose: bool,
) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=f"[rank{rank}/{world_size} %(asctime)s %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if phase == PHASE_GROUNDTRUTH and world_size != 1:
        raise click.ClickException(f"--phase=groundtruth requires nproc-per-node=1 (got world_size={world_size})")
    initialized_process_group = _ensure_validation_process_group(rank=rank, world_size=world_size)

    cfg = OmegaConf.load(config_path)
    if data_blend_dir is not None:
        cfg.data_blend_dir = data_blend_dir
    if input_cfg_path is not None:
        input_cfg_override = OmegaConf.load(input_cfg_path)
        if not (OmegaConf.is_dict(input_cfg_override) or OmegaConf.is_list(input_cfg_override)):
            raise click.ClickException("--input-cfg must contain a YAML mapping or list")
        cfg.data[section].input_cfg = input_cfg_override
        LOG.info("override data.%s.input_cfg from %s", section, input_cfg_path)
    OmegaConf.resolve(cfg)
    section_cfg = cfg.data[section]

    if mode == "full" and metadata_only:
        metadata_only = False
        LOG.info("full mode: forced metadata_only=False to materialize production payloads")

    inject_validator_flags(
        section_cfg, force_finite=force_finite, metadata_only=metadata_only
    )
    if skip_missing_manifest_entries is not None:
        inject_missing_manifest_policy(
            section_cfg,
            skip_missing_manifest_entries=skip_missing_manifest_entries,
        )
    if num_workers_override is not None:
        LOG.info(
            "override num_workers: %s -> %s",
            section_cfg.get("num_workers"),
            num_workers_override,
        )
        section_cfg.num_workers = num_workers_override
    # Groundtruth needs num_workers=0 so the single-process iteration enumerates everything.
    if phase == PHASE_GROUNDTRUTH:
        inject_groundtruth_flags(section_cfg)
        LOG.info(
            "groundtruth: forced num_workers=0, use_stateful_dataloader=False, "
            "force_iterable_dataset=False, force_map_dataset=True"
        )

    # Defer import until env vars and config injections are in place.
    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    tokenizer = _build_tokenizer(cfg, section_cfg, required=mode == "full")
    dataset = _build_validation_dataset(cfg, tokenizer, mode=mode, section=section)
    dataloader = get_lhotse_dataloader_from_config(
        config=section_cfg,
        global_rank=rank,
        world_size=world_size,
        dataset=dataset,
        tokenizer=tokenizer,
    )

    if phase == PHASE_RESUMED:
        _load_state(dataloader, state_dir=state_dir, rank=rank)

    out_dir = Path(output_dir)
    phase_dir = _phase_dir(out_dir, phase, run_idx)
    phase_dir.mkdir(parents=True, exist_ok=True)

    if phase == PHASE_GROUNDTRUTH:
        out_path = phase_dir / "cuts.jsonl"
    else:
        out_path = phase_dir / f"rank_{rank:03d}.jsonl"

    full_stats = None
    full_summary_path = None
    if mode == "full":
        audio_tag = cfg.get("model", {}).get("audio_locator_tag")
        placeholder_id = tokenizer.token_to_id(audio_tag)
        if not isinstance(placeholder_id, int) or placeholder_id < 0:
            raise click.ClickException(
                f"full validation could not resolve audio placeholder token ID for {audio_tag!r}"
            )
        full_stats = FullValidationStats(
            requested_batches=None if phase == PHASE_GROUNDTRUTH else steps,
            audio_placeholder_token_id=placeholder_id,
            audio_path_resolution_modes=configured_audio_path_resolution_modes(section_cfg),
        )
        full_summary_path = phase_dir / f"full_summary_rank_{rank:03d}.json"

        LOG.info(
            "full validator structured summary -> %s",
            full_summary_path,
        )

    LOG.info(
        "phase=%s run_idx=%d steps=%d checkpoint_at=%d -> %s",
        phase,
        run_idx,
        steps,
        checkpoint_at,
        out_path,
    )

    t_total_samples: list[float] = []
    completed_steps = 0
    t_first_batch_ms: Optional[float] = None
    iter_t0 = time.monotonic_ns()
    with (
        full_summary_failure_guard(
            full_stats,
            full_summary_path,
            phase=phase,
            rank=rank,
            world_size=world_size,
        ),
        open(out_path, "w") as fout,
    ):
        for step, batch in enumerate(dataloader):
            if mode == "full":
                _validate_full_batch(batch, step=step)
            t_step_end = time.monotonic_ns()
            if step == 0:
                t_first_batch_ms = (t_step_end - iter_t0) / 1e6
            t_total_ms = (t_step_end - iter_t0) / 1e6
            iter_t0 = t_step_end
            if full_stats is not None:
                full_stats.observe_batch(batch, latency_ms=t_total_ms)
            completed_steps = step + 1

            if phase != PHASE_GROUNDTRUTH and step > 0:
                t_total_samples.append(t_total_ms)

            cut_ids, worker_id = _extract_cuts(batch)
            semantic_cut_ids = _extract_semantic_cut_ids(batch, len(cut_ids))
            source_groups, source_ids = _extract_source_labels(batch, len(cut_ids))
            declared_durations = _extract_numeric_list(
                batch, "declared_duration_seconds", len(cut_ids), float
            )
            sampled_num_tokens = _extract_numeric_list(
                batch, "sampled_num_tokens", len(cut_ids), int
            )
            row = {
                "step": step,
                "rank": rank,
                "world_size": world_size,
                "worker_id": worker_id,
                "cut_ids": cut_ids,
                "batch_size": len(cut_ids),
                "t_total_ms": round(t_total_ms, 3),
                "t_first_batch_ms": round(t_first_batch_ms, 3) if step == 0 else None,
            }
            if semantic_cut_ids:
                row["semantic_cut_ids"] = semantic_cut_ids
            if any(source_groups):
                row["source_groups"] = source_groups
            if any(source_ids):
                row["source_ids"] = source_ids
            if declared_durations is not None:
                row["declared_duration_seconds"] = declared_durations
            if sampled_num_tokens is not None:
                row["sampled_num_tokens"] = sampled_num_tokens
            fout.write(json.dumps(row) + "\n")

            if step % 50 == 0:
                LOG.info(
                    "step=%d cuts=%d t_total=%.1fms (first cut: %s)",
                    step,
                    len(cut_ids),
                    t_total_ms,
                    cut_ids[0] if cut_ids else "<empty>",
                )

            if phase == PHASE_BASELINE and step == checkpoint_at:
                state_path = phase_dir / f"state_rank_{rank:03d}.pt"
                LOG.info("saving state_dict at step=%d -> %s", step, state_path)
                torch.save(dataloader.state_dict(), state_path)

            if phase != PHASE_GROUNDTRUTH and step + 1 >= steps:
                break

    if mode == "full" and phase != PHASE_GROUNDTRUTH and completed_steps < steps:
        error = click.ClickException(f"full validation materialized {completed_steps}/{steps} requested batches")
        full_stats.record_failure(step=completed_steps, error=error)
        full_stats.write(
            full_summary_path,
            phase=phase,
            rank=rank,
            world_size=world_size,
            status="failed",
        )
        raise error

    if full_stats is not None:
        full_stats.write(
            full_summary_path,
            phase=phase,
            rank=rank,
            world_size=world_size,
            status="passed",
        )
    if phase == PHASE_BASELINE and run_idx == 0:
        _write_throughput_summary(
            phase_dir / f"throughput_rank_{rank:03d}.json",
            t_total_samples=t_total_samples,
            t_first_batch_ms=t_first_batch_ms,
            num_workers=section_cfg.get("num_workers", 0),
        )

    LOG.info("DONE")
    _finish_validation_process_group(initialized_here=initialized_process_group)


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def _phase_dir(output_dir: Path, phase: str, run_idx: int) -> Path:
    if phase == PHASE_GROUNDTRUTH:
        return output_dir / phase
    return output_dir / phase / f"run{run_idx}"


def _extract_cuts(batch) -> tuple[list[str], int]:
    """``CutIdDataset.__getitem__`` returns validation graph identities.

    The JSON schema retains the historical ``cut_ids`` field name. Indexed
    fast- and full-mode values identify graph positions; semantic IDs are
    emitted in ``semantic_cut_ids``.

    The default collate stacks across the batch (which is always a single
    item under Lhotse's bucketing sampler), so we get back lists wrapped
    in length-1 outer lists. Handle both shapes defensively."""
    if isinstance(batch, dict):
        conversations = batch.get("conversations")
        if conversations is not None:
            return (
                [_validation_identity(conversation) for conversation in conversations],
                0,
            )

        cuts = batch.get("cut_ids", [])
        worker = batch.get("worker_id", 0)
        # Default collate wraps strings in lists; unwrap one level if needed.
        if cuts and isinstance(cuts[0], list):
            cuts = [c for sub in cuts for c in sub]
        if isinstance(worker, list):
            worker = int(worker[0]) if worker else 0
        elif isinstance(worker, torch.Tensor):
            worker = int(worker.item())
        return [str(c) for c in cuts], int(worker)
    # Fallback: unknown shape.
    return [], -1


def _extract_semantic_cut_ids(batch, expected_count: int) -> list[str]:
    """Extract optional source-provided IDs retained for diagnostics only."""
    if not isinstance(batch, dict):
        return []
    conversations = batch.get("conversations")
    if conversations is not None:
        values = [conversation.id for conversation in conversations]
    elif "semantic_cut_ids" in batch:
        values = batch.get("semantic_cut_ids") or []
    else:
        return []
    if values and isinstance(values[0], list):
        values = [value for nested in values for value in nested]
    normalized = [str(value) for value in values]
    if len(normalized) != expected_count:
        raise click.ClickException(
            "validator semantic ID cardinality mismatch: " f"expected={expected_count} actual={len(normalized)}"
        )
    return normalized


def _extract_source_labels(batch, expected_count: int) -> tuple[list[str], list[str]]:
    """Extract content-free source labels used by validation reports."""
    if not isinstance(batch, dict) or "cut_ids" not in batch:
        return [""] * expected_count, [""] * expected_count

    def normalize(values) -> list[str]:
        values = values or []
        if values and isinstance(values[0], list):
            values = [value for nested in values for value in nested]
        normalized = [str(value) for value in values]
        if len(normalized) != expected_count:
            raise click.ClickException(
                "validator source-label cardinality mismatch: " f"expected={expected_count} actual={len(normalized)}"
            )
        return normalized

    return normalize(batch.get("source_groups")), normalize(batch.get("source_ids"))


def _extract_numeric_list(batch, key: str, expected_count: int, cast):
    """Extract a content-free per-example numeric vector from fast-mode batches."""
    if not isinstance(batch, dict) or key not in batch:
        return None
    values = batch[key]
    if torch.is_tensor(values):
        values = values.detach().cpu().flatten().tolist()
    elif isinstance(values, tuple):
        values = list(values)
    if isinstance(values, list) and values and isinstance(values[0], (list, tuple)):
        values = [value for nested in values for value in nested]
    if not isinstance(values, list):
        values = [values]
    normalized = [cast(value) for value in values]
    if len(normalized) != expected_count:
        raise click.ClickException(
            f"validator {key} cardinality mismatch: "
            f"expected={expected_count} actual={len(normalized)}"
        )
    return normalized


def _load_state(dataloader, *, state_dir: Optional[str], rank: int) -> None:
    if state_dir is None:
        raise click.ClickException("--state-dir is required for --phase=resumed")
    state_path = Path(state_dir) / f"state_rank_{rank:03d}.pt"
    if not state_path.exists():
        raise click.ClickException(f"state file missing: {state_path}")
    LOG.info("loading state_dict from %s", state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    dataloader.load_state_dict(state)


def _write_throughput_summary(
    out_path: Path,
    *,
    t_total_samples: list[float],
    t_first_batch_ms: Optional[float],
    num_workers: int,
) -> None:
    if not t_total_samples:
        out_path.write_text(
            json.dumps(
                {
                    "p50_ms": None,
                    "p95_ms": None,
                    "mean_ms": None,
                    "count": 0,
                    "t_first_batch_ms": t_first_batch_ms,
                    "num_workers": num_workers,
                },
                indent=2,
            )
        )
        return
    samples = sorted(t_total_samples)
    p50 = statistics.median(samples)
    p95 = samples[int(0.95 * (len(samples) - 1))]
    mean = statistics.fmean(samples)
    out_path.write_text(
        json.dumps(
            {
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "mean_ms": round(mean, 3),
                "count": len(samples),
                "t_first_batch_ms": round(t_first_batch_ms, 3) if t_first_batch_ms else None,
                "num_workers": int(num_workers),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cli()
