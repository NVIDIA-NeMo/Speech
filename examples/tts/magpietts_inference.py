# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
TTS Inference and Evaluation Script.

Supports both encoder-decoder MagpieTTS and decoder-only EasyMagpieTTS models
with:
- Automatic MoE detection and FLOPs calculation
- Comprehensive evaluation metrics (RTF, FLOPs, CER, SSIM, etc.)

This script provides a clean CLI for running TTS inference with optional
evaluation. Model-specific behaviour (dataset creation, inference loop, CLI
arguments) is handled by separate runner classes so there is no scattered
if/else branching.

Example usage:
    # MagpieTTS inference (encoder-decoder, default)
    python examples/tts/magpietts_inference.py \\
        --model_type magpie \\
        --nemo_files /path/to/model.nemo \\
        --datasets_json_path /path/to/evalset_config.json \\
        --out_dir /path/to/output \\
        --codecmodel_path /path/to/codec.nemo

    # EasyMagpieTTS inference (decoder-only)
    python examples/tts/magpietts_inference.py \\
        --model_type easy_magpie \\
        --nemo_files /path/to/model.nemo \\
        --datasets_json_path /path/to/evalset_config.json \\
        --out_dir /path/to/output \\
        --codecmodel_path /path/to/codec.nemo

    # With evaluation
    python examples/tts/magpietts_inference.py \\
        --model_type magpie \\
        --hparams_files /path/to/hparams.yaml \\
        --checkpoint_files /path/to/model.ckpt \\
        --datasets_json_path /path/to/evalset_config.json \\
        --out_dir /path/to/output \\
        --codecmodel_path /path/to/codec.nemo \\
        --run_evaluation \\
        --num_repeats 3
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import time
from dataclasses import fields
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.models.easy_magpietts_inference import EasyModelInferenceParameters
from nemo.collections.tts.models.magpietts import ModelInferenceParameters
from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import load_evalset_config
from nemo.collections.tts.modules.magpietts_inference.evaluation import (
    DEFAULT_VIOLIN_METRICS,
    EvaluationConfig,
    compute_mean_with_confidence_interval,
    evaluate_generated_audio_dir,
)
from nemo.collections.tts.modules.magpietts_inference.inference import (
    BaseInferenceConfig,
    BaseInferenceRunner,
    EasyMagpieInferenceConfig,
    EasyMagpieInferenceRunner,
    EasyMagpieMultiturnUserAudioInferenceConfig,
    EasyMagpieMultiturnUserAudioInferenceRunner,
    MagpieInferenceConfig,
    MagpieInferenceRunner,
)
from nemo.collections.tts.modules.magpietts_inference.utils import (
    ModelLoadConfig,
    get_experiment_name_from_checkpoint_path,
    load_easy_magpie_model,
    load_magpie_model,
    log_model_architecture_summary,
)
from nemo.collections.tts.modules.magpietts_inference.visualization import create_combined_box_plot, create_violin_plot
from nemo.collections.tts.modules.magpietts_modules import EOSDetectionMethod
from nemo.utils import logging


def parse_layer_list(layer_str: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated list of layer indices."""
    if layer_str is None:
        return None
    return [int(l.strip()) for l in layer_str.split(",")]


def write_csv_header_if_needed(csv_path: str, header: str) -> None:
    """Write CSV header if file doesn't exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(header + "\n")


def append_metrics_to_csv(csv_path: str, checkpoint_name: str, dataset: str, metrics: dict) -> None:
    """Append metrics to a CSV file."""
    values = [
        checkpoint_name,
        dataset,
        metrics.get('cer_filewise_avg', ''),
        metrics.get('wer_filewise_avg', ''),
        metrics.get('cer_cumulative', ''),
        metrics.get('wer_cumulative', ''),
        metrics.get('ssim_pred_gt_avg', ''),
        metrics.get('ssim_pred_context_avg', ''),
        metrics.get('ssim_gt_context_avg', ''),
        metrics.get('ssim_pred_gt_avg_alternate', ''),
        metrics.get('ssim_pred_context_avg_alternate', ''),
        metrics.get('ssim_gt_context_avg_alternate', ''),
        metrics.get('esim_pred_gt_avg', ''),
        metrics.get('ems_pred_gt_avg', ''),
        metrics.get('cer_gt_audio_cumulative', ''),
        metrics.get('wer_gt_audio_cumulative', ''),
        metrics.get('utmosv2_avg', ''),
        metrics.get('total_gen_audio_seconds', ''),
        metrics.get('frechet_codec_distance', ''),
        metrics.get('eou_cutoff_rate', ''),
        metrics.get('eou_silence_rate', ''),
        metrics.get('eou_noise_rate', ''),
        metrics.get('eou_error_rate', ''),
    ]
    with open(csv_path, "a") as f:
        f.write(",".join(str(v) for v in values) + "\n")
    logging.info(f"Metrics appended to: {csv_path}")


def _mean_finite(values: list):
    vals = []
    for value in values:
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value):
            vals.append(value)
    return None if not vals else float(np.mean(vals))


def _enrich_filewise_metrics_with_manifest(filewise_metrics: list, manifest_path: str) -> list:
    """Attach multiturn manifest metadata back to evaluator filewise rows.

    evaluate_generated_audio_dir() returns one filewise row per generated turn,
    but the filtered row does not preserve source_sample_idx/turn_id. The
    generated multiturn manifest has the same order as predicted_audio_*.wav, so
    we merge by list index before grouping.
    """
    if manifest_path is None or not os.path.exists(manifest_path):
        logging.warning(f"Could not enrich multiturn filewise metrics; manifest missing: {manifest_path}")
        return filewise_metrics

    manifest_records = read_manifest(manifest_path)
    if len(manifest_records) != len(filewise_metrics):
        logging.warning(
            "Could not safely enrich multiturn filewise metrics; "
            f"manifest rows={len(manifest_records)} filewise rows={len(filewise_metrics)} "
            f"manifest_path={manifest_path}"
        )
        return filewise_metrics

    enriched = []
    for row, record in zip(filewise_metrics, manifest_records):
        new_row = dict(row)
        for key in [
            "source_sample_idx",
            "turn_id",
            "speaker",
            "rank",
            "rank_local_idx",
            "audio_filepath",
            "context_audio_filepath",
        ]:
            if key in record and key not in new_row:
                new_row[key] = record[key]
        enriched.append(new_row)

    return enriched


def _group_multiturn_filewise_metrics_by_sample(filewise_metrics: list) -> list:
    """Group turn-level multiturn metrics into one old-style row per sample.

    Each grouped row keeps turn-by-turn CER/WER/SSIM/UTMOS/text/audio lists plus
    averaged sample-level values. Rows are sorted by averaged CER descending so
    the worst conversations/samples appear first.
    """
    grouped = {}

    for row_idx, row in enumerate(filewise_metrics):
        source_sample_idx = row.get("source_sample_idx", None)
        if source_sample_idx is None:
            source_sample_idx = row.get("speaker", None)
        if source_sample_idx is None:
            source_sample_idx = row_idx

        key = str(source_sample_idx)
        if key not in grouped:
            grouped[key] = {
                "source_sample_idx": source_sample_idx,
                "speaker": row.get("speaker", source_sample_idx),
                "rank": row.get("rank", None),
                "target_audio_path": row.get("gt_audio_filepath", row.get("audio_filepath", "")),
                "context_audio_path": row.get("context_audio_filepath", ""),
                "turn_rows": [],
            }
        grouped[key]["turn_rows"].append(row)

    grouped_rows = []
    for _, group in grouped.items():
        turns = group["turn_rows"]

        def turn_sort_key(r):
            try:
                return int(r.get("turn_id", 0))
            except Exception:
                return 0

        turns = sorted(turns, key=turn_sort_key)

        cer_turns = [r.get("cer") for r in turns]
        wer_turns = [r.get("wer") for r in turns]
        pred_context_ssim_turns = [r.get("pred_context_ssim") for r in turns]
        pred_gt_ssim_turns = [r.get("pred_gt_ssim") for r in turns]
        gt_context_ssim_turns = [r.get("gt_context_ssim") for r in turns]
        pred_gt_esim_turns = [r.get("pred_gt_esim") for r in turns]
        pred_gt_ems_turns = [r.get("pred_gt_ems") for r in turns]
        utmosv2_turns = [r.get("utmosv2") for r in turns]
        eou_type_turns = [r.get("eou_type") for r in turns]
        eou_trailing_duration_turns = [r.get("eou_trailing_duration") for r in turns]
        eou_trail_rms_ratio_turns = [r.get("eou_trail_rms_ratio") for r in turns]

        grouped_rows.append(
            {
                "source_sample_idx": group["source_sample_idx"],
                "speaker": group["speaker"],
                "rank": group["rank"],
                "num_turns": len(turns),
                # Sample-level averages over all turns.
                "cer": _mean_finite(cer_turns),
                "wer": _mean_finite(wer_turns),
                "pred_context_ssim": _mean_finite(pred_context_ssim_turns),
                "pred_gt_ssim": _mean_finite(pred_gt_ssim_turns),
                "gt_context_ssim": _mean_finite(gt_context_ssim_turns),
                "pred_gt_esim": _mean_finite(pred_gt_esim_turns),
                "pred_gt_ems": _mean_finite(pred_gt_ems_turns),
                "utmosv2": _mean_finite(utmosv2_turns),
                "eou_trailing_duration": _mean_finite(eou_trailing_duration_turns),
                "eou_trail_rms_ratio": _mean_finite(eou_trail_rms_ratio_turns),
                # Turn-by-turn values, old-script style.
                "turn_ids": [r.get("turn_id", i) for i, r in enumerate(turns)],
                "cer_turns": cer_turns,
                "wer_turns": wer_turns,
                "pred_context_ssim_turns": pred_context_ssim_turns,
                "pred_gt_ssim_turns": pred_gt_ssim_turns,
                "gt_context_ssim_turns": gt_context_ssim_turns,
                "pred_gt_esim_turns": pred_gt_esim_turns,
                "pred_gt_ems_turns": pred_gt_ems_turns,
                "utmosv2_turns": utmosv2_turns,
                "eou_type_turns": eou_type_turns,
                "eou_trailing_duration_turns": eou_trailing_duration_turns,
                "eou_trail_rms_ratio_turns": eou_trail_rms_ratio_turns,
                "reference_text": [r.get("gt_text", "") for r in turns],
                "asr_hyp": [r.get("pred_text", "") for r in turns],
                "pred_audio_paths": [r.get("pred_audio_filepath", "") for r in turns],
                "target_audio_path": group["target_audio_path"],
                "context_audio_path": group["context_audio_path"],
                "turn_metrics": turns,
            }
        )

    grouped_rows.sort(
        key=lambda r: (
            r.get("cer") is not None,
            float(r["cer"]) if r.get("cer") is not None else -1.0,
        ),
        reverse=True,
    )
    return grouped_rows


def _write_grouped_multiturn_filewise_metrics_csv(csv_path: str, grouped_rows: list) -> None:
    fieldnames = [
        "source_sample_idx",
        "speaker",
        "rank",
        "num_turns",
        "cer",
        "wer",
        "pred_context_ssim",
        "pred_gt_ssim",
        "gt_context_ssim",
        "pred_gt_esim",
        "pred_gt_ems",
        "utmosv2",
        "eou_trailing_duration",
        "eou_trail_rms_ratio",
        "turn_ids",
        "cer_turns",
        "wer_turns",
        "pred_context_ssim_turns",
        "pred_gt_ssim_turns",
        "gt_context_ssim_turns",
        "pred_gt_esim_turns",
        "pred_gt_ems_turns",
        "utmosv2_turns",
        "eou_type_turns",
        "eou_trailing_duration_turns",
        "eou_trail_rms_ratio_turns",
        "target_audio_path",
        "context_audio_path",
        "pred_audio_paths",
        "reference_text",
        "asr_hyp",
    ]

    def csv_value(value):
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        if value is None:
            value = ""
        value = str(value).replace('"', '""')
        if "," in value or "\n" in value or "[" in value or "{" in value:
            value = f'"{value}"'
        return value

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in grouped_rows:
            f.write(",".join(csv_value(row.get(k, "")) for k in fieldnames) + "\n")


def _save_grouped_multiturn_filewise_metrics(
    eval_dir: str,
    dataset: str,
    repeat_idx: int,
    filewise_metrics: list,
    manifest_path: str,
) -> None:
    enriched_filewise = _enrich_filewise_metrics_with_manifest(filewise_metrics, manifest_path)
    grouped_rows = _group_multiturn_filewise_metrics_by_sample(enriched_filewise)

    json_path = os.path.join(eval_dir, f"{dataset}_grouped_filewise_metrics_{repeat_idx}.json")
    csv_path = os.path.join(eval_dir, f"{dataset}_grouped_filewise_metrics_{repeat_idx}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(grouped_rows, f, indent=4, ensure_ascii=False)

    _write_grouped_multiturn_filewise_metrics_csv(csv_path, grouped_rows)

    logging.info(f"Saved grouped multiturn filewise metrics JSON to: {json_path}")
    logging.info(f"Saved grouped multiturn filewise metrics CSV to: {csv_path}")


def create_formatted_metrics_mean_ci(metrics_mean_ci: dict) -> dict:
    """Create formatted metrics mean CI."""
    for k, v in metrics_mean_ci.items():
        if isinstance(v, list):
            mean, ci = float(v[0]), float(v[1])
            logging.info(f"Metric {k}: {mean:.4f} ± {ci:.4f}")
            metrics_mean_ci[k] = f"{mean:.4f} ± {ci:.4f}"
    return metrics_mean_ci


def filter_datasets(
    dataset_meta_info: dict,
    datasets: Optional[List[str]],
) -> List[str]:
    """Select datasets from the dataset meta info."""
    if datasets is None:
        # Dataset filtering not specified, return all datasets.
        return list(dataset_meta_info.keys())
    else:
        datasets = datasets.split(",")
        # Check if requested datasets are valid.
        for dataset in datasets:
            if dataset not in dataset_meta_info:
                raise ValueError(f"Dataset {dataset} not found in dataset meta info")
        # Return all requested datasets.
        return datasets


def _runner_eval_manifest_and_audio_dir(runner: BaseInferenceRunner, default_manifest: str, default_audio_dir: str):
    """Return evaluation manifest/audio dir produced by the runner, if any."""
    eval_manifest = getattr(runner, "evaluation_manifest_path", None) or default_manifest
    eval_audio_dir = getattr(runner, "evaluation_audio_dir", None) or default_audio_dir
    return eval_manifest, eval_audio_dir


def _get_torchrun_rank_info() -> Tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from torchrun/SLURM env vars.

    We intentionally do not initialize torch.distributed here. The inference
    script only needs env-based sharding, while NeMo evaluation models can run
    without distributed collectives.
    """
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    return rank, world_size, local_rank


def _configure_cuda_for_rank() -> Tuple[int, int, int]:
    rank, world_size, local_rank = _get_torchrun_rank_info()
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            torch.cuda.set_device(local_rank % device_count)
            logging.info(
                f"Using CUDA device {local_rank % device_count}; "
                f"rank={rank}, local_rank={local_rank}, world_size={world_size}"
            )
    return rank, world_size, local_rank


def _wait_for_multiturn_rank_manifests(repeat_audio_dir: str, world_size: int, timeout_sec: int = 7200) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        missing = []
        for rank in range(world_size):
            path = os.path.join(
                repeat_audio_dir,
                f"rank_{rank:04d}",
                f"multiturn_user_audio_turn_manifest_rank{rank:04d}.jsonl",
            )
            if not os.path.exists(path):
                missing.append(path)
        if not missing:
            return
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for multiturn rank manifests: {missing}")


def _path_is_under_dir(path: str, directory: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(directory)]) == os.path.realpath(directory)
    except ValueError:
        return False


def _move_or_copy_rank_output(src: str, dst: str, required: bool = False, rank_dir: str = None) -> None:
    if src is None or not os.path.exists(src):
        if os.path.lexists(dst):
            os.remove(dst)
        if required:
            raise FileNotFoundError(f"Missing required merge source: {src} -> {dst}")
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.lexists(dst):
        os.remove(dst)

    # Move files produced inside rank_XXXX/ into the merged evaluation directory.
    # If a manifest unexpectedly points outside rank_XXXX/ (for example an
    # absolute dataset path), copy instead so original input data is not moved.
    if rank_dir is not None and not _path_is_under_dir(src, rank_dir):
        shutil.copyfile(src, dst)
    else:
        shutil.move(src, dst)


def _merge_multiturn_rank_outputs(repeat_audio_dir: str, world_size: int, save_predicted_codes: bool) -> str:
    """Merge rank-local multiturn outputs into one EasyMagpie-compatible dir.

    Each rank writes local files named predicted_audio_0.wav, target_audio_0.wav,
    context_audio_0.wav, predicted_codes_0.pt, ... inside rank_XXXX/. This
    function moves rank-local artifacts to contiguous global indices in
    repeat_audio_dir/ and writes a merged turn-level manifest.
    """
    # clean previous merged files
    for pattern in [
        "predicted_audio_*.wav",
        "target_audio_*.wav",
        "context_audio_*.wav",
        "predicted_codes_*.pt",
    ]:
        for path in Path(repeat_audio_dir).glob(pattern):
            if path.is_symlink() or path.exists():
                path.unlink(missing_ok=True)

    merged_records = []
    global_idx = 0

    for rank in range(world_size):
        rank_dir = os.path.join(repeat_audio_dir, f"rank_{rank:04d}")
        rank_manifest = os.path.join(rank_dir, f"multiturn_user_audio_turn_manifest_rank{rank:04d}.jsonl")
        if not os.path.exists(rank_manifest):
            raise FileNotFoundError(f"Missing rank manifest: {rank_manifest}")

        with open(rank_manifest, "r", encoding="utf-8") as f:
            rank_records = [json.loads(line) for line in f if line.strip()]

        for local_idx, record in enumerate(rank_records):
            pred_src = os.path.join(rank_dir, f"predicted_audio_{local_idx}.wav")
            pred_dst = os.path.join(repeat_audio_dir, f"predicted_audio_{global_idx}.wav")
            _move_or_copy_rank_output(pred_src, pred_dst, required=True, rank_dir=rank_dir)

            if save_predicted_codes:
                code_src = os.path.join(rank_dir, f"predicted_codes_{local_idx}.pt")
                code_dst = os.path.join(repeat_audio_dir, f"predicted_codes_{global_idx}.pt")
                _move_or_copy_rank_output(code_src, code_dst, required=False, rank_dir=rank_dir)

            target_src = os.path.join(rank_dir, record.get("audio_filepath", f"target_audio_{local_idx}.wav"))
            target_dst = os.path.join(repeat_audio_dir, f"target_audio_{global_idx}.wav")
            _move_or_copy_rank_output(target_src, target_dst, required=True, rank_dir=rank_dir)

            context_src = os.path.join(
                rank_dir,
                record.get("context_audio_filepath", f"context_audio_{local_idx}.wav"),
            )
            context_dst = os.path.join(repeat_audio_dir, f"context_audio_{global_idx}.wav")
            _move_or_copy_rank_output(context_src, context_dst, required=True, rank_dir=rank_dir)

            merged = dict(record)
            merged["audio_filepath"] = f"target_audio_{global_idx}.wav"
            merged["context_audio_filepath"] = f"context_audio_{global_idx}.wav"
            merged["rank"] = rank
            merged["rank_local_idx"] = local_idx
            merged_records.append(merged)
            global_idx += 1

    merged_manifest = os.path.join(repeat_audio_dir, "multiturn_user_audio_turn_manifest.jsonl")
    with open(merged_manifest, "w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(f"Merged {len(merged_records)} multiturn turn records into {merged_manifest}")
    return merged_manifest


def run_inference_and_evaluation(
    runner: BaseInferenceRunner,
    checkpoint_name: str,
    inference_config: BaseInferenceConfig,
    eval_config: EvaluationConfig,
    dataset_meta_info: dict,
    datasets: List[str],
    out_dir: str,
    flops_per_component: dict,
    moe_info: str,
    num_repeats: int = 1,
    confidence_level: float = 0.95,
    violin_plot_metrics: Optional[List[str]] = None,
    clean_up_disk: bool = False,
    skip_evaluation: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """Run inference and optional evaluation on specified datasets.

    This function is model-type agnostic -- it delegates dataset creation
    and batch inference to the provided ``runner``.

    Args:
        runner: Concrete inference runner (MagpieInferenceRunner or EasyMagpieInferenceRunner).
        checkpoint_name: Human-readable checkpoint identifier for output naming.
        inference_config: Configuration for inference.
        eval_config: Configuration for evaluation.
        dataset_meta_info: Dictionary containing dataset metadata.
        datasets: List of dataset names to process.
        out_dir: Output directory for results.
        flops_per_component: FLOPs info dict from log_model_architecture_summary.
        moe_info: MoE identifier string from log_model_architecture_summary.
        num_repeats: Number of times to repeat inference (for CI estimation).
        confidence_level: Confidence level for CI calculation.
        violin_plot_metrics: Metrics to include in violin plots.
        clean_up_disk: Whether to clean up output directory after completion.
        skip_evaluation: Whether to skip evaluation (inference only mode).

    Returns:
        Tuple of (mean CER across datasets, mean SSIM across datasets).
    """
    if violin_plot_metrics is None:
        violin_plot_metrics = list(DEFAULT_VIOLIN_METRICS)

    # Remove UTMOSv2 from plots if disabled
    if not eval_config.with_utmosv2 and 'utmosv2' in violin_plot_metrics:
        violin_plot_metrics.remove('utmosv2')

    rank, world_size, _ = _get_torchrun_rank_info()
    is_distributed = world_size > 1
    is_multiturn_user_audio = getattr(runner, "produces_turn_level_evaluation", False)

    if hasattr(runner, "set_distributed_context"):
        runner.set_distributed_context(rank=rank, world_size=world_size)

    # Build full checkpoint identifier (include MoE info if present)
    full_checkpoint_name = (
        f"{checkpoint_name}_{moe_info}{inference_config.build_identifier()}_SV_{eval_config.sv_model}"
    )

    # Tracking metrics across datasets
    ssim_per_dataset = []
    cer_per_dataset = []
    all_datasets_filewise_metrics = {}

    # CSV headers
    csv_header = (
        "checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,"
        "wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,"
        "ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,"
        "ssim_gt_context_avg_alternate,esim_pred_gt_avg,ems_pred_gt_avg,"
        "cer_gt_audio_cumulative,wer_gt_audio_cumulative,"
        "utmosv2_avg,total_gen_audio_seconds,frechet_codec_distance,"
        "eou_cutoff_rate,eou_silence_rate,eou_noise_rate,eou_error_rate"
    )

    for dataset in datasets:
        logging.info(f"Processing dataset: {dataset}")

        meta = dataset_meta_info[dataset]
        manifest_records = read_manifest(meta['manifest_path'])
        language = meta.get('whisper_language', 'en')

        # Prepare dataset metadata (remove evaluation-specific keys)
        dataset_meta_for_dl = copy.deepcopy(meta)
        for key in ["whisper_language", "load_cached_codes_if_available"]:
            dataset_meta_for_dl.pop(key, None)

        # Setup output directories
        eval_dir = os.path.join(out_dir, f"{full_checkpoint_name}_{dataset}")
        audio_dir = os.path.join(eval_dir, "audio")
        os.makedirs(eval_dir, exist_ok=True)

        # Setup CSV files
        per_run_csv = os.path.join(eval_dir, "all_experiment_metrics.csv")
        if rank == 0:
            write_csv_header_if_needed(per_run_csv, csv_header)

        metrics_all_repeats = []
        filewise_metrics_all_repeats = []

        for repeat_idx in range(num_repeats):
            repeat_log_msg = f"Repeat {repeat_idx + 1}/{num_repeats} for dataset {dataset}"
            if is_distributed:
                repeat_log_msg += f", rank {rank}/{world_size}"
            logging.info(repeat_log_msg)

            repeat_audio_dir = os.path.join(audio_dir, f"repeat_{repeat_idx}")
            os.makedirs(repeat_audio_dir, exist_ok=True)

            # Create dataset and run inference
            test_dataset = runner.create_dataset({dataset: dataset_meta_for_dl})

            if len(test_dataset) != len(manifest_records):
                raise ValueError(
                    f"Dataset length mismatch: {len(test_dataset)} vs {len(manifest_records)} manifest records"
                )

            if is_distributed and not is_multiturn_user_audio:
                raise RuntimeError(
                    "torchrun multi-GPU sharding is currently implemented for "
                    "--easy_magpie_inference_mode multiturn_user_audio only. "
                    "Use the existing single-process path for single_turn/magpie, or add a "
                    "rank-safe merge path for those runners."
                )

            inference_output_dir = repeat_audio_dir
            if is_distributed and is_multiturn_user_audio:
                inference_output_dir = os.path.join(repeat_audio_dir, f"rank_{rank:04d}")

            rtf_metrics_list, _, codec_file_paths = runner.run_inference_on_dataset(
                dataset=test_dataset,
                output_dir=inference_output_dir,
                manifest_records=manifest_records,
                audio_base_dir=meta['audio_dir'],
                save_cross_attention_maps=True,
                save_context_audio=(repeat_idx == 0),  # Only save context audio once
                save_predicted_codes=eval_config.with_fcd,  # Code files are only needed for FCD computation
            )

            # Compute mean RTF metrics
            mean_rtf = runner.compute_mean_rtf_metrics(rtf_metrics_list)

            # Add FLOPs metrics per component
            for component_name, component_flops in flops_per_component.items():
                for key, value in component_flops.items():
                    mean_rtf[f"{component_name}_{key}"] = value
                logging.info(f"{component_name} FLOPs per token: {component_flops['total_flops_per_token']:,}")

            rtf_metrics_filename = f"{dataset}_rtf_metrics_{repeat_idx}.json"
            if is_distributed:
                rtf_metrics_filename = f"{dataset}_rtf_metrics_{repeat_idx}_rank{rank:04d}.json"
            with open(os.path.join(eval_dir, rtf_metrics_filename), "w") as f:
                json.dump(mean_rtf, f, indent=4)

            if skip_evaluation:
                logging.info("Skipping evaluation as requested.")
                continue

            # Run evaluation
            if is_distributed and is_multiturn_user_audio:
                if rank != 0:
                    # Non-zero ranks only generate. Rank 0 waits and evaluates merged outputs.
                    continue

                _wait_for_multiturn_rank_manifests(repeat_audio_dir, world_size)
                merged_manifest_path = _merge_multiturn_rank_outputs(
                    repeat_audio_dir=repeat_audio_dir,
                    world_size=world_size,
                    save_predicted_codes=eval_config.with_fcd,
                )
                eval_manifest_path = merged_manifest_path
                eval_audio_dir = repeat_audio_dir
            else:
                eval_manifest_path, eval_audio_dir = _runner_eval_manifest_and_audio_dir(
                    runner,
                    default_manifest=meta['manifest_path'],
                    default_audio_dir=meta['audio_dir'],
                )

            eval_config_for_dataset = EvaluationConfig(
                sv_model=eval_config.sv_model,
                asr_model_name=eval_config.asr_model_name,
                eou_model_name=eval_config.eou_model_name,
                language=language,
                with_utmosv2=eval_config.with_utmosv2,
                with_fcd=eval_config.with_fcd,
                codec_model_path=eval_config.codec_model_path,
                with_emotion_metrics=eval_config.with_emotion_metrics,
                emotion_model_size=eval_config.emotion_model_size,
                emotion_embedding_type=eval_config.emotion_embedding_type,
                emotion_cache_dir=eval_config.emotion_cache_dir,
                strip_text_annotations_for_metrics=eval_config.strip_text_annotations_for_metrics,
                device=eval_config.device,
                asr_batch_size=eval_config.asr_batch_size,
                eou_batch_size=eval_config.eou_batch_size,
            )

            metrics, filewise_metrics = evaluate_generated_audio_dir(
                manifest_path=eval_manifest_path,
                audio_dir=eval_audio_dir,
                generated_audio_dir=repeat_audio_dir,
                config=eval_config_for_dataset,
            )

            metrics_all_repeats.append(metrics)
            filewise_metrics_all_repeats.extend(filewise_metrics)

            # Save metrics
            with open(os.path.join(eval_dir, f"{dataset}_metrics_{repeat_idx}.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            sorted_filewise = sorted(filewise_metrics, key=lambda x: x.get('cer', 0), reverse=True)
            with open(os.path.join(eval_dir, f"{dataset}_filewise_metrics_{repeat_idx}.json"), "w") as f:
                json.dump(sorted_filewise, f, indent=4)

            if is_multiturn_user_audio:
                _save_grouped_multiturn_filewise_metrics(
                    eval_dir=eval_dir,
                    dataset=dataset,
                    repeat_idx=repeat_idx,
                    filewise_metrics=filewise_metrics,
                    manifest_path=eval_manifest_path,
                )

            # Append to per-run CSV
            append_metrics_to_csv(per_run_csv, full_checkpoint_name, dataset, metrics)

            # Create violin plot for this repeat
            violin_path = Path(eval_dir) / f"{dataset}_violin_{repeat_idx}.png"
            create_violin_plot(filewise_metrics, violin_plot_metrics, violin_path)

            # Delete temporary predicted codes files
            if is_distributed and is_multiturn_user_audio:
                for codec_file_path in Path(repeat_audio_dir).glob("predicted_codes_*.pt"):
                    if os.path.exists(codec_file_path):
                        os.remove(codec_file_path)
            else:
                for codec_file_path in codec_file_paths:
                    os.remove(codec_file_path)

        if rank != 0:
            continue

        if skip_evaluation or not metrics_all_repeats:
            continue

        # Store for combined plot
        all_datasets_filewise_metrics[dataset] = filewise_metrics_all_repeats

        # Compute mean with confidence interval across repeats
        metrics_mean_ci = compute_mean_with_confidence_interval(
            metrics_all_repeats,
            confidence=confidence_level,
        )

        formatted_metrics_mean_ci = create_formatted_metrics_mean_ci(metrics_mean_ci)

        # Write to aggregated CSV
        ci_csv = os.path.join(out_dir, "all_experiment_metrics_with_ci.csv")
        write_csv_header_if_needed(ci_csv, csv_header)
        append_metrics_to_csv(ci_csv, full_checkpoint_name, dataset, formatted_metrics_mean_ci)

        # Track per-dataset means
        ssim_values = [m['ssim_pred_context_avg'] for m in metrics_all_repeats]
        cer_values = [m['cer_cumulative'] for m in metrics_all_repeats]
        ssim_per_dataset.append(np.mean(ssim_values))
        cer_per_dataset.append(np.mean(cer_values))

    # Create combined plot if we have multiple datasets
    if rank == 0 and len(all_datasets_filewise_metrics) > 1:
        combined_plot_path = os.path.join(out_dir, f"{full_checkpoint_name}_combined_violin_plot.png")
        create_combined_box_plot(all_datasets_filewise_metrics, violin_plot_metrics, combined_plot_path)

    # Clean up if requested
    if rank == 0 and clean_up_disk:
        logging.info(f"Cleaning up output directory: {out_dir}")
        shutil.rmtree(out_dir)

    # Return averaged metrics
    if rank == 0 and ssim_per_dataset and cer_per_dataset:
        return np.mean(cer_per_dataset), np.mean(ssim_per_dataset)
    return None, None


def _get_shared_inference_param_names() -> set:
    """Return the field names shared by ModelInferenceParameters and EasyModelInferenceParameters."""
    magpie_fields = {f.name for f in fields(ModelInferenceParameters)}
    easy_fields = {f.name for f in fields(EasyModelInferenceParameters)}
    return magpie_fields & easy_fields


def _add_inference_param_fields(
    group: argparse._ArgumentGroup,
    param_cls: type,
    skip_fields: Optional[set] = None,
    only_fields: Optional[set] = None,
) -> None:
    """Auto-generate argparse arguments from fields of a dataclass.

    Args:
        group: The argparse argument group to add arguments to.
        param_cls: The dataclass whose fields to add.
        skip_fields: Field names to skip (already added by another group).
        only_fields: If provided, only add fields whose names are in this set.
    """
    if skip_fields is None:
        skip_fields = set()
    for f in fields(param_cls):
        if f.name in skip_fields:
            continue
        if only_fields is not None and f.name not in only_fields:
            continue
        extra_args: dict = {"type": f.type}
        if f.type == bool:
            extra_args = {"action": "store_true"}
        if f.name in ("estimate_alignment_from_layers", "apply_prior_to_layers"):
            extra_args = {
                "help": "Must be a comma separate string. Not enclosed in brackets",
                "type": str,
            }
        elif f.name == "eos_detection_method":
            extra_args["choices"] = [m.value for m in EOSDetectionMethod]
        group.add_argument(f"--{f.name}", **extra_args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all model types."""

    parser.add_argument(
        '--model_type',
        type=str,
        default='magpie',
        choices=['magpie', 'easy_magpie'],
        help='Model type: "magpie" for encoder-decoder MagpieTTSModel, '
        '"easy_magpie" for decoder-only EasyMagpieTTSInferenceModel',
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Attempts to make results deterministic to the best that can be done. Used for testing',
    )

    # Model loading
    model_group = parser.add_argument_group('Model Loading')
    model_group.add_argument(
        '--hparams_files',
        type=str,
        default=None,
        help='Comma-separated paths to hparams.yaml files (use with --checkpoint_files)',
    )
    model_group.add_argument(
        '--checkpoint_files',
        type=str,
        default=None,
        help='Comma-separated paths to .ckpt files (use with --hparams_files)',
    )
    model_group.add_argument(
        '--nemo_files',
        type=str,
        default=None,
        help='Comma-separated paths to .nemo files (alternative to hparams + checkpoint)',
    )
    model_group.add_argument(
        '--codecmodel_path',
        type=str,
        required=True,
        help='Path to the audio codec model',
    )
    model_group.add_argument(
        '--hparams_file_from_wandb',
        action='store_true',
        help='Set if hparams file was exported from wandb',
    )
    model_group.add_argument(
        '--legacy_codebooks',
        action='store_true',
        help='Use legacy codebook indices (for old checkpoints)',
    )
    model_group.add_argument(
        '--legacy_text_conditioning',
        action='store_true',
        help='Use legacy text conditioning (for old checkpoints)',
    )

    # Dataset and output
    data_group = parser.add_argument_group('Dataset and Output')
    data_group.add_argument(
        '--datasets_json_path',
        type=str,
        required=True,
        default=None,
        help='Path to dataset configuration JSON file',
    )
    data_group.add_argument(
        '--datasets_base_path',
        type=Path,
        default=None,
        help='Optional base path that paths in the "datasets_json_path" file are relative to',
    )
    data_group.add_argument(
        '--datasets',
        type=str,
        default=None,
        help='Comma-separated list of dataset names to process',
    )
    data_group.add_argument('--out_dir', type=str, required=True, help='Output directory')
    data_group.add_argument('--log_exp_name', action='store_true')
    data_group.add_argument('--clean_up_disk', action='store_true')

    # Common inference parameters
    infer_group = parser.add_argument_group('Common Inference Parameters')
    infer_group.add_argument('--batch_size', type=int, default=32)
    infer_group.add_argument('--use_cfg', action='store_true', help='Enable classifier-free guidance')
    infer_group.add_argument('--use_local_transformer', action='store_true')

    # Model inference parameters shared by both MagpieTTS and EasyMagpieTTS
    shared_param_names = _get_shared_inference_param_names()
    _add_inference_param_fields(infer_group, ModelInferenceParameters, only_fields=shared_param_names)

    # Evaluation
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--run_evaluation', action='store_true', help='Run evaluation after inference')
    eval_group.add_argument('--sv_model', type=str, default="titanet", choices=["titanet", "wavlm"])
    eval_group.add_argument('--asr_model_name', type=str, default="nvidia/parakeet-tdt-1.1b")
    eval_group.add_argument(
        '--eou_model_name',
        type=str,
        default="facebook/wav2vec2-base-960h",
        help=(
            'Hugging Face model id or local path to the EoU wav2vec2 model directory. '
            'For offline use, download the model locally and pass the directory path here.'
        ),
    )
    eval_group.add_argument('--num_repeats', type=int, default=1)
    eval_group.add_argument('--confidence_level', type=float, default=0.95)
    eval_group.add_argument('--disable_utmosv2', action='store_true')
    eval_group.add_argument('--with_emotion_metrics', action='store_true')
    eval_group.add_argument(
        '--strip_text_annotations_for_metrics',
        action='store_true',
        help='Strip bracket/tag/control annotations from reference and ASR hypothesis text while computing text metrics.',
    )
    eval_group.add_argument('--emotion_model_size', type=str, default="small", choices=["small", "large"])
    eval_group.add_argument(
        '--emotion_embedding_type',
        type=str,
        default="score_vector",
        choices=["head_concat", "head_mean", "score_vector"],
    )
    eval_group.add_argument('--emotion_cache_dir', type=str, default=None)
    eval_group.add_argument(
        '--violin_plot_metrics',
        type=str,
        nargs='*',
        default=['cer', 'pred_context_ssim', 'utmosv2'],
    )
    eval_group.add_argument('--disable_fcd', action='store_true')
    eval_group.add_argument("--asr_batch_size", type=int, default=32)
    eval_group.add_argument("--eou_batch_size", type=int, default=32)

    # Quality targets
    target_group = parser.add_argument_group('Quality Targets')
    target_group.add_argument('--cer_target', type=float, default=None)
    target_group.add_argument('--ssim_target', type=float, default=None)


def seed_all(seed: int):
    """
    Attempts to make script deterministic
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _add_magpie_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to encoder-decoder MagpieTTSModel."""
    group = parser.add_argument_group('MagpieTTS-specific Parameters')

    # MagpieTTS-specific model inference parameters (attention prior, EOS, etc.)
    shared_param_names = _get_shared_inference_param_names()
    _add_inference_param_fields(group, ModelInferenceParameters, skip_fields=shared_param_names)

    group.add_argument('--maskgit_n_steps', type=int, default=3)
    group.add_argument('--maskgit_noise_scale', type=float, default=0.0)
    group.add_argument('--maskgit_fixed_schedule', type=int, nargs='+', default=None)
    group.add_argument(
        '--maskgit_sampling_type',
        default=None,
        choices=["default", "causal", "purity_causal", "purity_default"],
    )


def _add_easy_magpie_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to decoder-only EasyMagpieTTSInferenceModel."""
    group = parser.add_argument_group('EasyMagpieTTS-specific Parameters')
    group.add_argument(
        '--easy_magpie_inference_mode',
        type=str,
        default='single_turn',
        choices=['single_turn', 'multiturn_user_audio'],
    )
    group.add_argument('--max_eval_turns', type=int, default=6)
    group.add_argument('--no_save_debug_multiturn_audio', action='store_true')
    group.add_argument(
        '--phoneme_input_type',
        type=str,
        default='gt',
        choices=['gt', 'predicted'],
        help='Source of phoneme input for decoder-only model',
    )
    group.add_argument(
        '--phoneme_sampling_method',
        type=str,
        default='argmax',
        choices=['argmax', 'multinomial', 'greedy'],
        help='Sampling method for phoneme prediction',
    )
    group.add_argument('--dropout_text_input', action='store_true', help='Force dropout on text input')
    group.add_argument(
        '--phoneme_tokenizer_path',
        type=str,
        default=None,
        help='Override path to the phoneme tokenizer file (overrides the path stored in the checkpoint config)',
    )
    group.add_argument(
        '--disable_cas_for_context_text',
        action='store_true',
        help='Skip CAS embeddings for context text when loading legacy EasyMagpieTTS models',
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser with all argument groups."""
    parser = argparse.ArgumentParser(
        description='TTS Inference and Evaluation (MagpieTTS & EasyMagpieTTS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _add_common_args(parser)
    _add_magpie_args(parser)
    _add_easy_magpie_args(parser)
    return parser


def _build_inference_params_from_args(param_cls: type, args):
    """Extract inference parameters from parsed CLI args for the given dataclass."""
    params = {}
    for f in fields(param_cls):
        arg_val = vars(args).get(f.name)
        if arg_val is not None:
            if f.name in ("estimate_alignment_from_layers", "apply_prior_to_layers"):
                params[f.name] = parse_layer_list(arg_val)
            else:
                params[f.name] = arg_val
    return param_cls.from_dict(params)


def _build_magpie_config(args) -> MagpieInferenceConfig:
    return MagpieInferenceConfig(
        model_inference_parameters=_build_inference_params_from_args(ModelInferenceParameters, args),
        batch_size=args.batch_size,
        use_cfg=args.use_cfg,
        apply_attention_prior=args.apply_attention_prior,
        use_local_transformer=args.use_local_transformer,
        maskgit_n_steps=args.maskgit_n_steps,
        maskgit_noise_scale=args.maskgit_noise_scale,
        maskgit_fixed_schedule=args.maskgit_fixed_schedule,
        maskgit_sampling_type=args.maskgit_sampling_type,
    )


def _build_easy_magpie_config(args) -> EasyMagpieInferenceConfig:
    cfg_cls = (
        EasyMagpieMultiturnUserAudioInferenceConfig
        if args.easy_magpie_inference_mode == 'multiturn_user_audio'
        else EasyMagpieInferenceConfig
    )
    kwargs = dict(
        model_inference_parameters=_build_inference_params_from_args(EasyModelInferenceParameters, args),
        batch_size=args.batch_size,
        use_cfg=args.use_cfg,
        use_local_transformer=args.use_local_transformer,
        phoneme_input_type=args.phoneme_input_type,
        phoneme_sampling_method=args.phoneme_sampling_method,
        dropout_text_input=args.dropout_text_input,
    )
    if cfg_cls is EasyMagpieMultiturnUserAudioInferenceConfig:
        kwargs.update(
            max_eval_turns=args.max_eval_turns,
            save_debug_multiturn_audio=not args.no_save_debug_multiturn_audio,
        )
    return cfg_cls(**kwargs)


def _select_runner_cls(args):
    if args.model_type == 'magpie':
        if args.easy_magpie_inference_mode != 'single_turn':
            raise ValueError('--easy_magpie_inference_mode is only supported with --model_type easy_magpie')
        return MagpieInferenceRunner
    if args.easy_magpie_inference_mode == 'multiturn_user_audio':
        return EasyMagpieMultiturnUserAudioInferenceRunner
    return EasyMagpieInferenceRunner


def main(argv=None):
    """Entry point for TTS inference and evaluation."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    if args.model_type == 'easy_magpie' and args.easy_magpie_inference_mode == 'multiturn_user_audio':
        _configure_cuda_for_rank()
        if args.batch_size > 1:
            parser.error("--easy_magpie_inference_mode multiturn_user_audio requires --batch_size 1.")

    if args.deterministic:
        seed_all(seed=9)

    dataset_meta_info = load_evalset_config(
        config_path=args.datasets_json_path, dataset_base_path=args.datasets_base_path
    )
    datasets = filter_datasets(dataset_meta_info, args.datasets)
    logging.info(f"Loaded {len(datasets)} datasets: {', '.join(datasets)}")

    # Validate model loading args
    has_checkpoint_mode = (
        args.hparams_files is not None
        and args.checkpoint_files is not None
        and args.hparams_files != "null"
        and args.checkpoint_files != "null"
    )
    has_nemo_mode = args.nemo_files is not None and args.nemo_files != "null"

    if not has_checkpoint_mode and not has_nemo_mode:
        parser.error("You must provide either:\n 1. --hparams_files and --checkpoint_files\n 2. --nemo_files")

    # Select model loader and config builder based on --model_type
    is_easy_magpie = args.model_type == 'easy_magpie'
    load_fn = load_easy_magpie_model if is_easy_magpie else load_magpie_model
    inference_config = _build_easy_magpie_config(args) if is_easy_magpie else _build_magpie_config(args)
    runner_cls = _select_runner_cls(args)

    eval_config = EvaluationConfig(
        sv_model=args.sv_model,
        asr_model_name=args.asr_model_name,
        eou_model_name=args.eou_model_name,
        with_utmosv2=not args.disable_utmosv2,
        with_fcd=not args.disable_fcd,
        codec_model_path=args.codecmodel_path if not args.disable_fcd else None,
        with_emotion_metrics=args.with_emotion_metrics,
        emotion_model_size=args.emotion_model_size,
        emotion_embedding_type=args.emotion_embedding_type,
        emotion_cache_dir=args.emotion_cache_dir,
        strip_text_annotations_for_metrics=args.strip_text_annotations_for_metrics,
        asr_batch_size=args.asr_batch_size,
        eou_batch_size=args.eou_batch_size,
    )

    cer, ssim = None, None

    # Iterate over model files (checkpoint or nemo)
    if has_checkpoint_mode:
        hparam_files = args.hparams_files.split(",")
        checkpoint_files = args.checkpoint_files.split(",")

        if len(hparam_files) != len(checkpoint_files):
            parser.error("Number of hparams_files must match number of checkpoint_files")

        for hparams_file, checkpoint_file in zip(hparam_files, checkpoint_files):
            logging.info(f"Processing checkpoint: {checkpoint_file}")

            model_config = ModelLoadConfig(
                hparams_file=hparams_file,
                checkpoint_file=checkpoint_file,
                codecmodel_path=args.codecmodel_path,
                legacy_codebooks=args.legacy_codebooks,
                legacy_text_conditioning=args.legacy_text_conditioning,
                hparams_from_wandb=args.hparams_file_from_wandb,
                phoneme_tokenizer_path=getattr(args, 'phoneme_tokenizer_path', None),
                disable_cas_for_context_text=args.disable_cas_for_context_text,
            )

            # Load model
            model, checkpoint_name = load_fn(model_config)
            # Log architecture summary and get MoE info + FLOPs metrics
            moe_info, flops_per_component = log_model_architecture_summary(model)

            # Add experiment name prefix if requested
            if args.log_exp_name and model_config.checkpoint_file:
                exp_name = get_experiment_name_from_checkpoint_path(model_config.checkpoint_file)
                checkpoint_name = f"{exp_name}__{checkpoint_name}"

            # Create inference runner
            runner = runner_cls(model, inference_config)

            cer, ssim = run_inference_and_evaluation(
                runner=runner,
                checkpoint_name=checkpoint_name,
                inference_config=inference_config,
                eval_config=eval_config,
                dataset_meta_info=dataset_meta_info,
                datasets=datasets,
                out_dir=args.out_dir,
                flops_per_component=flops_per_component,
                moe_info=moe_info,
                num_repeats=args.num_repeats,
                confidence_level=args.confidence_level,
                violin_plot_metrics=args.violin_plot_metrics,
                clean_up_disk=args.clean_up_disk,
                skip_evaluation=not args.run_evaluation,
            )

    else:  # nemo mode
        for nemo_file in args.nemo_files.split(","):
            logging.info(f"Processing NeMo file: {nemo_file}")

            model_config = ModelLoadConfig(
                nemo_file=nemo_file,
                codecmodel_path=args.codecmodel_path,
                legacy_codebooks=args.legacy_codebooks,
                legacy_text_conditioning=args.legacy_text_conditioning,
                phoneme_tokenizer_path=getattr(args, 'phoneme_tokenizer_path', None),
                disable_cas_for_context_text=args.disable_cas_for_context_text,
            )

            # Load model
            model, checkpoint_name = load_fn(model_config)
            # Log architecture summary and get MoE info + FLOPs metrics
            moe_info, flops_per_component = log_model_architecture_summary(model)

            # Create inference runner
            runner = runner_cls(model, inference_config)

            cer, ssim = run_inference_and_evaluation(
                runner=runner,
                checkpoint_name=checkpoint_name,
                inference_config=inference_config,
                eval_config=eval_config,
                dataset_meta_info=dataset_meta_info,
                datasets=datasets,
                out_dir=args.out_dir,
                flops_per_component=flops_per_component,
                moe_info=moe_info,
                num_repeats=args.num_repeats,
                confidence_level=args.confidence_level,
                violin_plot_metrics=args.violin_plot_metrics,
                clean_up_disk=args.clean_up_disk,
                skip_evaluation=not args.run_evaluation,
            )

    # Check quality targets
    if cer is not None and args.cer_target is not None:
        if cer > args.cer_target:
            raise ValueError(f"CER {cer:.4f} exceeds target {args.cer_target:.4f}")
        logging.info(f"CER {cer:.4f} meets target {args.cer_target:.4f}")

    if ssim is not None and args.ssim_target is not None:
        if ssim < args.ssim_target:
            raise ValueError(f"SSIM {ssim:.4f} below target {args.ssim_target:.4f}")
        logging.info(f"SSIM {ssim:.4f} meets target {args.ssim_target:.4f}")

    logging.info("Inference and evaluation completed successfully.")


if __name__ == '__main__':
    main()
