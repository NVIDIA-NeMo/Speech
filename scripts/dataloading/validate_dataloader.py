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
"""Validate a Lhotse + indexed dataloader config end-to-end.

Per-rank entry point launched under torchrun. Builds the **exact** dataloader
the SALM training builds (via ``get_lhotse_dataloader_from_config``) on top
of a no-op ``CutIdDataset`` and dumps per-batch cut.id JSONL. Phase-aware:

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
from _validate_dataloader.config_inject import inject_validator_flags  # noqa: E402
from _validate_dataloader.cut_id_dataset import CutIdDataset  # noqa: E402

LOG = logging.getLogger(__name__)


PHASE_BASELINE = "baseline"
PHASE_RESUMED = "resumed"
PHASE_GROUNDTRUTH = "groundtruth"


@click.command(help=__doc__)
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--data-blend-dir", default=None, help="Substituted into ${data_blend_dir} in the config.")
@click.option("--section", default="train_ds", show_default=True)
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--phase", type=click.Choice([PHASE_BASELINE, PHASE_RESUMED, PHASE_GROUNDTRUTH]), required=True)
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
@click.option("--num-workers-override", type=int, default=None, help="Override config.{section}.num_workers.")
@click.option(
    "--mode",
    type=click.Choice(["fast", "full"]),
    default="fast",
    show_default=True,
    help="fast: CutIdDataset (default). full: stub-only in v1, raises.",
)
@click.option("-v", "--verbose", is_flag=True, default=False)
def cli(
    config_path: str,
    data_blend_dir: Optional[str],
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
    mode: str,
    verbose: bool,
) -> None:
    if mode == "full":
        raise click.ClickException("--mode=full is not implemented in v1; use --mode=fast.")

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

    cfg = OmegaConf.load(config_path)
    if data_blend_dir is not None:
        cfg.data_blend_dir = data_blend_dir
    OmegaConf.resolve(cfg)
    section_cfg = cfg.data[section]

    inject_validator_flags(section_cfg, force_finite=force_finite, metadata_only=metadata_only)
    if num_workers_override is not None:
        LOG.info("override num_workers: %s -> %s", section_cfg.get("num_workers"), num_workers_override)
        section_cfg.num_workers = num_workers_override
    # Groundtruth needs num_workers=0 so the single-process iteration enumerates everything.
    if phase == PHASE_GROUNDTRUTH:
        section_cfg.num_workers = 0
        section_cfg.use_stateful_dataloader = False
        section_cfg.force_map_dataset = True
        LOG.info("groundtruth: forced num_workers=0, use_stateful_dataloader=False, force_map_dataset=True")

    # Defer import until env vars and config injections are in place.
    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    tokenizer = _build_tokenizer_if_needed(cfg, section_cfg)
    dataset = CutIdDataset()
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

    LOG.info("phase=%s run_idx=%d steps=%d checkpoint_at=%d -> %s", phase, run_idx, steps, checkpoint_at, out_path)

    t_total_samples: list[float] = []
    t_first_batch_ms: Optional[float] = None
    iter_t0 = time.monotonic_ns()
    with open(out_path, "w") as fout:
        for step, batch in enumerate(dataloader):
            t_step_end = time.monotonic_ns()
            if step == 0:
                t_first_batch_ms = (t_step_end - iter_t0) / 1e6
            t_total_ms = (t_step_end - iter_t0) / 1e6
            iter_t0 = t_step_end

            if phase != PHASE_GROUNDTRUTH and step > 0:
                t_total_samples.append(t_total_ms)

            cut_ids, worker_id = _extract_cuts(batch)
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

    if phase == PHASE_BASELINE and run_idx == 0:
        _write_throughput_summary(
            phase_dir / f"throughput_rank_{rank:03d}.json",
            t_total_samples=t_total_samples,
            t_first_batch_ms=t_first_batch_ms,
            num_workers=section_cfg.get("num_workers", 0),
        )

    LOG.info("DONE")


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def _phase_dir(output_dir: Path, phase: str, run_idx: int) -> Path:
    if phase == PHASE_GROUNDTRUTH:
        return output_dir / phase
    return output_dir / phase / f"run{run_idx}"


def _extract_cuts(batch) -> tuple[list[str], int]:
    """``CutIdDataset.__getitem__`` returns ``{"cut_ids": [...], "worker_id": W}``.
    The default collate stacks across the batch (which is always a single
    item under Lhotse's bucketing sampler), so we get back lists wrapped
    in length-1 outer lists. Handle both shapes defensively."""
    if isinstance(batch, dict):
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


def _build_tokenizer_if_needed(full_cfg, section_cfg):
    """Bucketer length measurement under ``use_multimodal_sampling=True`` requires
    a tokenizer. Mirror SALM's construction (``salm.py:66``) so token counts
    match production. Returns ``None`` when the config doesn't ask for it."""
    if not section_cfg.get("use_multimodal_sampling", False):
        return None
    pretrained_llm = full_cfg.get("model", {}).get("pretrained_llm")
    if not pretrained_llm:
        raise click.ClickException(
            "use_multimodal_sampling=True requires model.pretrained_llm in the config to load a tokenizer."
        )
    from nemo.collections.common.tokenizers import AutoTokenizer

    trust_remote_code = bool(full_cfg.get("model", {}).get("trust_remote_code", False))
    LOG.info("loading tokenizer for %s (trust_remote_code=%s)", pretrained_llm, trust_remote_code)
    tokenizer = AutoTokenizer(pretrained_llm, use_fast=True, trust_remote_code=trust_remote_code)
    audio_tag = full_cfg.get("model", {}).get("audio_locator_tag")
    if audio_tag:
        tokenizer.add_special_tokens({"additional_special_tokens": [audio_tag]})
    return tokenizer


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
    out_path: Path, *, t_total_samples: list[float], t_first_batch_ms: Optional[float], num_workers: int
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
