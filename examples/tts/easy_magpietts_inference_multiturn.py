#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Multi-GPU EasyMagpieTTS / NemotronTTS multiturn inference evaluation.

Key behavior:
  - Uses torchrun env vars RANK, LOCAL_RANK, WORLD_SIZE for sharding/GPU assignment.
  - Does NOT initialize torch.distributed. This avoids NeMo ASR doing distributed
    collectives during metric computation.
  - Generation runs first for all assigned samples.
  - ASR and speaker-similarity models are loaded only after generation is done and the TTS/codec model
    has been deleted from GPU memory.
  - ASR and speaker-similarity models are loaded sequentially: ASR first, then released; speaker-similarity second.
  - Supports multiturn-user-audio and regular single-turn inference; metrics are turn/file based.
    Final filewise outputs are grouped back to one row per original sample, with
    lists for asr_hyp/reference_text/cer_turns/wer_turns/ssim_turns.
  - Uses DistributedSampler with explicit rank/world_size. A few repeated samples
    may appear when len(dataset) is not divisible by world_size. Filewise final
    metrics deduplicate sampler-padding repeats by (run_id, dataset_index,
    turn_id), then group turns into one row per sample with metric lists, while
    preserving --num_eval_runs repetitions.
  - --sort_by_text_token_count sorts samples by total text-token count before
    sharding to improve GPU load balance.
  - Saves audio in out_dir/audios/.
  - Saves metrics in out_dir/.

Recommended single-node torchrun:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --nproc_per_node=8 easy_magpietts_inference_multiturn_multigpu_postgen_metrics.py ...

Recommended single-node srun wrapper:
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --container-image=... \
    bash -lc 'torchrun --standalone --nproc_per_node=8 easy_magpietts_inference_multiturn_multigpu_postgen_metrics.py ...'
"""

import argparse
import csv
import json
import math
import os
import socket
import shutil
import time
from collections import Counter
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Iterable, List, Tuple

import librosa
import soundfile as sf
import torch
from omegaconf import open_dict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.asr.metrics.wer import word_error_rate, word_error_rate_detail
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.modules.magpietts_modules import CodecHelper
from nemo.collections.tts.parts.utils.tts_dataset_utils import normalize_volume
from nemo.utils import logging
from whisper_normalizer.english import EnglishTextNormalizer

try:
    import nemo.collections.asr as nemo_asr
except Exception:
    nemo_asr = None

try:
    from nemo.collections.asr.models import ASRModel
except Exception:
    ASRModel = None

try:
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
except Exception:
    Wav2Vec2FeatureExtractor = None
    WavLMForXVector = None

try:
    from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import (
        compute_utmosv2_scores,
        extract_embedding,
    )
except Exception:
    compute_utmosv2_scores = None
    extract_embedding = None

try:
    from nemo.collections.tts.metrics.eou_classifier import EoUClassifier, EoUType
except Exception:
    EoUClassifier = None
    EoUType = None

try:
    from nemo.collections.tts.modules.magpietts_inference.evaluation import DEFAULT_VIOLIN_METRICS
except Exception:
    DEFAULT_VIOLIN_METRICS = ['cer', 'pred_context_ssim', 'utmosv2']

try:
    from nemo.collections.tts.modules.magpietts_inference.visualization import create_violin_plot
except Exception:
    create_violin_plot = None

try:
    from nemo.collections.tts.metrics.frechet_codec_distance import FrechetCodecDistance
except Exception:
    FrechetCodecDistance = None



torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


# -----------------------------
# Rank / file helpers
# -----------------------------


def get_rank_info() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    distributed = world_size > 1
    return distributed, rank, local_rank, world_size


def get_visible_device_index(local_rank: int) -> int:
    if not torch.cuda.is_available():
        return -1
    ndev = torch.cuda.device_count()
    if ndev <= 0:
        return -1
    return local_rank % ndev


def setup_distributed():
    """
    Do not initialize torch.distributed.

    We only need RANK/LOCAL_RANK/WORLD_SIZE for rank assignment and dataset
    sharding. Initializing a process group can cause NeMo ASR to run distributed
    collectives during transcribe(), which may hang when ranks have different
    audio lengths or workloads.
    """
    distributed, rank, local_rank, world_size = get_rank_info()
    device_index = get_visible_device_index(local_rank)

    if torch.cuda.is_available() and device_index >= 0:
        torch.cuda.set_device(device_index)

    return distributed, rank, local_rank, world_size, device_index


def cleanup_distributed():
    return


def all_rank_print(rank: int, msg: str):
    print(f"[rank={rank}] {msg}", flush=True)


def rank0_print(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


def get_audio_out_dir(args) -> str:
    return os.path.join(args.out_dir, "audios")


def get_generated_turn_audio_dir(args) -> str:
    return os.path.join(get_audio_out_dir(args), "metric_turns")


def get_context_metric_audio_dir(args) -> str:
    return os.path.join(get_audio_out_dir(args), "metric_context")


def get_predicted_codes_dir(args) -> str:
    return os.path.join(get_audio_out_dir(args), "predicted_codes")


def write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp_path, path)


def write_text_atomic(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)

def write_csv_header_if_needed(csv_path: str, header: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")


def append_metrics_to_csv(csv_path: str, checkpoint_name: str, dataset: str, metrics: Dict[str, Any]) -> None:
    """Append metrics using the same column order as MagpieTTS inference/eval."""
    csv_header = (
        "checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,"
        "wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,"
        "ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,"
        "ssim_gt_context_avg_alternate,cer_gt_audio_cumulative,wer_gt_audio_cumulative,"
        "utmosv2_avg,total_gen_audio_seconds,frechet_codec_distance,"
        "eou_cutoff_rate,eou_silence_rate,eou_noise_rate,eou_error_rate"
    )
    write_csv_header_if_needed(csv_path, csv_header)

    values = [
        checkpoint_name,
        dataset,
        metrics.get("cer_filewise_avg", ""),
        metrics.get("wer_filewise_avg", ""),
        metrics.get("cer_cumulative", ""),
        metrics.get("wer_cumulative", ""),
        metrics.get("ssim_pred_gt_avg", ""),
        metrics.get("ssim_pred_context_avg", ""),
        metrics.get("ssim_gt_context_avg", ""),
        metrics.get("ssim_pred_gt_avg_alternate", ""),
        metrics.get("ssim_pred_context_avg_alternate", ""),
        metrics.get("ssim_gt_context_avg_alternate", ""),
        metrics.get("cer_gt_audio_cumulative", ""),
        metrics.get("wer_gt_audio_cumulative", ""),
        metrics.get("utmosv2_avg", ""),
        metrics.get("total_gen_audio_seconds", ""),
        metrics.get("frechet_codec_distance", ""),
        metrics.get("eou_cutoff_rate", ""),
        metrics.get("eou_silence_rate", ""),
        metrics.get("eou_noise_rate", ""),
        metrics.get("eou_error_rate", ""),
    ]

    def clean_csv_value(v):
        if v is None:
            return ""
        if isinstance(v, float) and not math.isfinite(v):
            return "nan"
        return str(v).replace(",", " ")

    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(",".join(clean_csv_value(v) for v in values) + "\n")
    logging.info(f"Metrics appended to: {csv_path}")


def get_checkpoint_name(args) -> str:
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if checkpoint_path:
        stem = os.path.basename(checkpoint_path)
        if stem.endswith(".nemo"):
            stem = stem[:-5]
        return stem
    return "checkpoint"


def get_dataset_name(args) -> str:
    out_name = os.path.basename(os.path.normpath(args.out_dir))
    if out_name:
        return out_name
    dataset_path = getattr(args, "datasets_json_path", None)
    return os.path.splitext(os.path.basename(dataset_path))[0] if dataset_path else "dataset"


def create_violin_plot_if_available(metrics: List[Dict[str, Any]], metric_keys: List[str], output_path: str):
    if create_violin_plot is None:
        logging.warning(
            "create_violin_plot is unavailable; skipping violin plot. "
            "Make sure nemo.collections.tts.modules.magpietts_inference.visualization is importable."
        )
        return

    if not metrics:
        logging.warning(f"No metrics available for violin plot: {output_path}")
        return

    available_keys = []
    for key in metric_keys:
        for row in metrics:
            value = row.get(key, None)
            if value is None:
                continue
            try:
                value = float(value)
            except Exception:
                continue
            if math.isfinite(value):
                available_keys.append(key)
                break

    if not available_keys:
        logging.warning(f"No finite requested plot metrics available for violin plot: {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_violin_plot(metrics, available_keys, output_path)


def _copy_or_link(src: str, dst: str):
    if src is None or not src or not os.path.exists(src):
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    except Exception:
        shutil.copyfile(src, dst)
    return dst


def write_easymagpie_generated_audio_dir(args, sample_rows: List[Dict[str, Any]]):
    """Write EasyMagpie/MagpieTTS-style generated audio/code files.

    This creates files named predicted_audio_*.wav and predicted_codes_*.pt,
    plus target/context audio files, so downstream EasyMagpie demo/report tools
    can consume this output directory.
    """
    generated_audio_dir = os.path.join(args.out_dir, "easy_magpie_generated_audio")
    os.makedirs(generated_audio_dir, exist_ok=True)

    manifest_rows = []
    filewise_rows = []

    rows = sorted(sample_rows, key=lambda r: (int(r.get("run_id", 0)), int(r.get("dataset_index", -1))))

    for item_idx, row in enumerate(rows):
        pred_src = row.get("sample_pred_audio_path") or (
            row.get("pred_audio_paths", [None])[0] if isinstance(row.get("pred_audio_paths"), list) else None
        )
        code_src = row.get("sample_predicted_codes_path") or (
            row.get("predicted_codes_paths", [None])[0] if isinstance(row.get("predicted_codes_paths"), list) else None
        )
        target_src = _resolve_audio_path(row.get("target_audio_path"), args.audio_dir)
        context_src = row.get("context_audio_path")

        pred_dst = os.path.join(generated_audio_dir, f"predicted_audio_{item_idx}.wav")
        code_dst = os.path.join(generated_audio_dir, f"predicted_codes_{item_idx}.pt")
        target_dst = os.path.join(generated_audio_dir, f"target_audio_{item_idx}.wav")
        context_dst = os.path.join(generated_audio_dir, f"context_audio_{item_idx}.wav")

        _copy_or_link(pred_src, pred_dst)
        _copy_or_link(code_src, code_dst)
        _copy_or_link(target_src, target_dst)
        _copy_or_link(context_src, context_dst)

        reference_text = row.get("reference_text", "")
        if isinstance(reference_text, list):
            manifest_text = " ".join(str(x) for x in reference_text)
        else:
            manifest_text = str(reference_text)

        manifest_rows.append(
            {
                "audio_filepath": f"target_audio_{item_idx}.wav",
                "context_audio_filepath": f"context_audio_{item_idx}.wav",
                "text": manifest_text,
                "speaker": row.get("dataset_index", item_idx),
                "original_dataset_index": row.get("dataset_index"),
                "run_id": row.get("run_id", 0),
            }
        )

        metric_row = dict(row)
        metric_row.update(
            {
                "easy_magpie_item_idx": item_idx,
                "gt_audio_filepath": target_dst if os.path.exists(target_dst) else target_src,
                "pred_audio_filepath": pred_dst if os.path.exists(pred_dst) else pred_src,
                "context_audio_filepath": context_dst if os.path.exists(context_dst) else context_src,
                "predicted_codes_path": code_dst if os.path.exists(code_dst) else code_src,
            }
        )
        filewise_rows.append(metric_row)

    manifest_path = os.path.join(args.out_dir, "easy_magpie_generated_manifest.jsonl")
    filewise_path = os.path.join(args.out_dir, "easy_magpie_generated_filewise_metrics.json")
    write_jsonl(manifest_path, manifest_rows)
    write_json(filewise_path, {"filewise_metrics": filewise_rows})

    logging.info(f"Saved EasyMagpie-style generated audio dir to: {generated_audio_dir}")
    logging.info(f"Saved EasyMagpie-style generated manifest to: {manifest_path}")
    logging.info(f"Saved EasyMagpie-style generated filewise metrics to: {filewise_path}")

    return {
        "generated_audio_dir": generated_audio_dir,
        "manifest_path": manifest_path,
        "filewise_metrics_path": filewise_path,
    }


def save_easymagpie_style_eval_outputs(args, sample_rows: List[Dict[str, Any]], filewise_summary: Dict[str, Any]):
    """Save CSV, plots, and generated-audio artifacts following EasyMagpie conventions."""
    easy_magpie_artifacts = write_easymagpie_generated_audio_dir(args, sample_rows)
    filewise_summary["easy_magpie_generated_audio_dir"] = easy_magpie_artifacts["generated_audio_dir"]
    filewise_summary["easy_magpie_generated_manifest"] = easy_magpie_artifacts["manifest_path"]

    checkpoint_name = get_checkpoint_name(args)
    dataset_name = get_dataset_name(args)

    per_run_csv = os.path.join(args.out_dir, "all_experiment_metrics.csv")
    append_metrics_to_csv(per_run_csv, checkpoint_name, dataset_name, filewise_summary)

    # Keep this alias because EasyMagpie aggregation scripts often look for the CI CSV.
    ci_csv = os.path.join(args.out_dir, "all_experiment_metrics_with_ci.csv")
    append_metrics_to_csv(ci_csv, checkpoint_name, dataset_name, filewise_summary)

    if not args.save_plots:
        return

    violin_metrics = list(args.violin_plot_metrics)
    if args.disable_utmosv2 and "utmosv2" in violin_metrics:
        violin_metrics.remove("utmosv2")

    plot_dir = os.path.join(args.out_dir, "plots")
    create_violin_plot_if_available(
        sample_rows,
        violin_metrics,
        os.path.join(plot_dir, f"{dataset_name}_violin.png"),
    )

    # Also write in eval_dir root with the same style used by MagpieTTS:
    # f"{dataset}_violin_{repeat_idx}.png". Here the merged final output is repeat 0.
    create_violin_plot_if_available(
        sample_rows,
        violin_metrics,
        os.path.join(args.out_dir, f"{dataset_name}_violin_0.png"),
    )


def wait_for_files(paths: List[str], timeout_sec: float = 7200.0, poll_sec: float = 5.0):
    start = time.time()
    while True:
        missing = [p for p in paths if not os.path.exists(p)]
        if not missing:
            return
        if time.time() - start > timeout_sec:
            raise TimeoutError("Timed out waiting for files:\n" + "\n".join(missing))
        time.sleep(poll_sec)


def wait_for_rank_metric_files(args, world_size: int):
    paths = [os.path.join(args.out_dir, f"metrics_rank{r:04d}.json") for r in range(world_size)]
    wait_for_files(paths)


def wait_for_rank_filewise_metric_files(args, world_size: int):
    paths = [os.path.join(args.out_dir, f"filewise_metrics_rank{r:04d}.jsonl") for r in range(world_size)]
    wait_for_files(paths)


def scalarize_metric_value(v: Any):
    if torch.is_tensor(v):
        if v.numel() == 1:
            return float(v.detach().cpu().item())
        return v.detach().cpu().tolist()
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return float(v.item())
    except Exception:
        pass
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def metric_dict_to_jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): scalarize_metric_value(v) for k, v in d.items()}


def safe_metric_scalar(metric_dict: Dict[str, Any], preferred_keys: List[str]):
    for key in preferred_keys:
        if key in metric_dict:
            value = metric_dict[key]
            if torch.is_tensor(value):
                return float(value.detach().cpu().item())
            return float(value)
    return None


def get_first_metric(metrics: Dict[str, Any], names: List[str], default=None):
    for name in names:
        if name in metrics:
            return metrics[name]
    return default



def format_final_metric_text(final_metrics: Dict[str, Any]) -> str:
    intelligibility = final_metrics.get("intelligibility", {})
    speaker_similarity = final_metrics.get("speaker_similarity", {})

    cer = get_first_metric(intelligibility, ["cer", "cer_dataset", "cer_cumulative"])
    wer = get_first_metric(intelligibility, ["wer", "wer_dataset", "wer_cumulative"])
    ssim_value = get_first_metric(
        speaker_similarity,
        ["ssim", "ssim_dataset", "ssim_pred_context_avg", "pred_context_ssim"],
    )

    def fmt(x):
        if x is None:
            return "nan"
        try:
            return f"{float(x):.10f}"
        except Exception:
            return str(x)

    return f"Average CER: {fmt(cer)}\nAverage WER: {fmt(wer)}\nSSIM: {fmt(ssim_value)}\n"



def format_filewise_final_metric_text(filewise_summary: Dict[str, Any]) -> str:
    def fmt(x):
        if x is None:
            return "nan"
        try:
            return f"{float(x):.10f}"
        except Exception:
            return str(x)

    ordered_keys = [
        ("cer", "CER filewise avg"),
        ("wer", "WER filewise avg"),
        ("cer_cumulative", "CER cumulative"),
        ("wer_cumulative", "WER cumulative"),
        ("ssim", "SSIM"),
        ("ssim_pred_gt_avg", "SSIM pred/GT avg"),
        ("ssim_pred_context_avg", "SSIM pred/context avg"),
        ("ssim_gt_context_avg", "SSIM GT/context avg"),
        ("ssim_pred_gt_avg_alternate", "SSIM pred/GT avg alternate"),
        ("ssim_pred_context_avg_alternate", "SSIM pred/context avg alternate"),
        ("ssim_gt_context_avg_alternate", "SSIM GT/context avg alternate"),
        ("cer_gt_audio_cumulative", "CER GT-audio cumulative"),
        ("wer_gt_audio_cumulative", "WER GT-audio cumulative"),
        ("utmosv2_avg", "UTMOSv2 avg"),
        ("total_gen_audio_seconds", "Total generated audio seconds"),
        ("frechet_codec_distance", "Frechet codec distance"),
        ("eou_cutoff_rate", "EOU cutoff rate"),
        ("eou_silence_rate", "EOU silence rate"),
        ("eou_noise_rate", "EOU noise rate"),
        ("eou_error_rate", "EOU error rate"),
    ]

    lines = [
        f"Average CER: {fmt(filewise_summary.get('cer'))}",
        f"Average WER: {fmt(filewise_summary.get('wer'))}",
        f"SSIM: {fmt(filewise_summary.get('ssim'))}",
    ]

    for key, label in ordered_keys:
        if key in {"cer", "wer", "ssim"}:
            continue
        if key in filewise_summary:
            lines.append(f"{label}: {fmt(filewise_summary.get(key))}")

    return "\n".join(lines) + "\n"



def write_filewise_csv(path: str, rows: List[Dict[str, Any]]):
    """Write sample-level filewise metrics.

    Several fields are lists (turn_ids, reference_text, asr_hyp, cer_turns,
    etc.), so they are JSON-encoded inside CSV cells.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    fieldnames = [
        "run_id",
        "dataset_index",
        "rank",
        "num_turns",
        "cer",
        "wer",
        "ssim",
        "pred_gt_ssim",
        "pred_context_ssim",
        "gt_context_ssim",
        "pred_gt_ssim_alternate",
        "pred_context_ssim_alternate",
        "gt_context_ssim_alternate",
        "utmosv2",
        "eou_error",
        "turn_ids",
        "cer_turns",
        "wer_turns",
        "ssim_turns",
        "pred_gt_ssim_turns",
        "pred_context_ssim_turns",
        "gt_context_ssim_turns",
        "pred_gt_ssim_alternate_turns",
        "pred_context_ssim_alternate_turns",
        "gt_context_ssim_alternate_turns",
        "utmosv2_turns",
        "eou_type_turns",
        "eou_trailing_duration_turns",
        "eou_trail_rms_ratio_turns",
        "pred_audio_seconds_turns",
        "target_audio_path",
        "context_audio_path",
        "pred_audio_paths",
        "predicted_codes_paths",
        "sample_pred_audio_path",
        "sample_predicted_codes_path",
        "reference_text",
        "asr_hyp",
    ]

    def csv_value(v):
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return v

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, None)) for k in fieldnames})

    os.replace(tmp_path, path)



def write_turnwise_csv(path: str, rows: List[Dict[str, Any]]):
    """Write merged turn-level filewise metrics sorted by CER."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    fieldnames = [
        "run_id",
        "dataset_index",
        "turn_id",
        "rank",
        "cer",
        "wer",
        "ssim",
        "pred_gt_ssim",
        "pred_context_ssim",
        "gt_context_ssim",
        "pred_gt_ssim_alternate",
        "pred_context_ssim_alternate",
        "gt_context_ssim_alternate",
        "utmosv2",
        "eou_type",
        "eou_trailing_duration",
        "eou_trail_rms_ratio",
        "pred_audio_seconds",
        "target_audio_path",
        "context_audio_path",
        "pred_audio_path",
        "predicted_codes_path",
        "sample_pred_audio_path",
        "sample_predicted_codes_path",
        "reference_text",
        "asr_hyp",
    ]

    def csv_value(v):
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return v

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, None)) for k in fieldnames})

    os.replace(tmp_path, path)


# -----------------------------
# Dataset helpers
# -----------------------------


def _combined_audio_name(first_audio_filepath: str, paths: List[str]) -> str:
    base_names = [os.path.splitext(os.path.basename(p))[0] for p in paths if p]
    ext = os.path.splitext(paths[-1])[1] if paths and paths[-1] else ""
    combined_name = "_".join(base_names) + ext
    dir_name = os.path.dirname(first_audio_filepath)
    return os.path.join(dir_name, combined_name) if dir_name else combined_name


class EvalJSONLDataset(Dataset):
    def __init__(self, file_path: str, emulate_multiturn_num_turns: int = 1):
        self.samples = []
        raw_samples = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    sample["__dataset_index__"] = len(raw_samples)
                    raw_samples.append(sample)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_idx}: {e}")

        if emulate_multiturn_num_turns <= 1:
            self.samples = raw_samples
            return

        single_turn_by_speaker = {}
        for sample in raw_samples:
            if isinstance(sample["text"], list):
                self.samples.append(sample)
            else:
                speaker = sample.get("speaker", "unknown")
                single_turn_by_speaker.setdefault(speaker, []).append(sample)

        synthetic_index = len(raw_samples)
        for _, speaker_samples in single_turn_by_speaker.items():
            buffer_texts, buffer_paths = [], []
            first_sample_meta = None

            for sample in speaker_samples:
                if not buffer_texts:
                    first_sample_meta = dict(sample)

                buffer_texts.append(sample["text"])
                buffer_paths.append(sample.get("audio_filepath", ""))

                if len(buffer_texts) == emulate_multiturn_num_turns:
                    first_sample_meta["text"] = buffer_texts
                    first_sample_meta["audio_filepath"] = _combined_audio_name(
                        first_sample_meta.get("audio_filepath", ""),
                        buffer_paths,
                    )
                    first_sample_meta["__dataset_index__"] = synthetic_index
                    synthetic_index += 1
                    self.samples.append(first_sample_meta)
                    buffer_texts, buffer_paths, first_sample_meta = [], [], None

            if buffer_texts and first_sample_meta is not None:
                first_sample_meta["text"] = buffer_texts
                first_sample_meta["audio_filepath"] = _combined_audio_name(
                    first_sample_meta.get("audio_filepath", ""),
                    buffer_paths,
                )
                first_sample_meta["__dataset_index__"] = synthetic_index
                synthetic_index += 1
                self.samples.append(first_sample_meta)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _sample_text_segments_for_count(sample: Dict[str, Any], max_eval_turns=None) -> List[str]:
    text_data = sample.get("text", "")
    if isinstance(text_data, list):
        segments = text_data
        if max_eval_turns is not None:
            segments = segments[: int(max_eval_turns)]
        return [str(x) for x in segments]
    return [str(text_data)]


def estimate_text_token_count(sample: Dict[str, Any], model, max_eval_turns=None) -> int:
    main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
    total = 0
    for segment in _sample_text_segments_for_count(sample, max_eval_turns=max_eval_turns):
        total += len(model.tokenizer.encode(segment, tokenizer_name=main_tokenizer_name)) + 1
    return int(total)


class SortedByTextTokenCountDataset(Dataset):
    def __init__(self, dataset: Dataset, model, max_eval_turns=None, descending: bool = True):
        self.dataset = dataset
        scored = []
        for i in range(len(dataset)):
            sample = dict(dataset[i])
            token_count = estimate_text_token_count(sample, model=model, max_eval_turns=max_eval_turns)
            sample["__text_token_count__"] = int(token_count)
            scored.append((token_count, i, sample))

        scored.sort(key=lambda x: (x[0], -x[1]), reverse=bool(descending))
        self.indices = [i for _, i, _ in scored]
        self.token_counts = {i: int(tok) for tok, i, _ in scored}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        original_idx = self.indices[local_idx]
        sample = dict(self.dataset[original_idx])
        sample["__text_token_count__"] = self.token_counts[original_idx]
        return sample


# -----------------------------
# Audio / collate helpers
# -----------------------------


def _resolve_audio_path(path, root_path):
    if path is None:
        return None
    if root_path is not None and not os.path.isabs(path):
        return os.path.join(root_path, path)
    return path


def _load_audio(path, sample_rate, normalize=True, use_librosa=False):
    if path is None or not os.path.exists(path):
        return torch.zeros(1, dtype=torch.float32)

    if use_librosa:
        wav, sr = librosa.load(path, sr=sample_rate, mono=True)
        if normalize:
            wav = normalize_volume(wav)
        return torch.as_tensor(wav, dtype=torch.float32)

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if normalize:
        wav = normalize_volume(wav)

    wav = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0)
    return resample(wav, sr, sample_rate).squeeze(0)




def collate_and_tokenize_custom(
    batch,
    model,
    sample_rate=22050,
    root_path=None,
    normalize_audio_volume=True,
    use_librosa=False,
    max_eval_turns=None,
    inference_mode="auto",
):
    """Collate for either multiturn-user-audio or regular single-turn inference.

    Mode selection:
      - multiturn_user_audio: turn-based multiturn user-audio prefill with user_audio_file_path.
      - single_turn: regular batched TTS, no user-speech/silence prefill.
      - auto: multiturn_user_audio when samples look multiturn/user-conditioned; otherwise
        single_turn. This keeps old LibriTTS commands working with batch_size=32.
    """
    main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]

    if max_eval_turns is not None:
        max_eval_turns = int(max_eval_turns)
        if max_eval_turns <= 0:
            raise ValueError("--max_eval_turns must be > 0 when provided.")
        truncated_batch = []
        for sample in batch:
            sample = dict(sample)
            if isinstance(sample["text"], list):
                sample["text"] = sample["text"][:max_eval_turns]
                if isinstance(sample.get("user_audio_file_path"), list):
                    sample["user_audio_file_path"] = sample["user_audio_file_path"][:max_eval_turns]
            truncated_batch.append(sample)
        batch = truncated_batch

    def looks_multiturn_user_audio(sample):
        return isinstance(sample.get("text"), list) or bool(sample.get("user_audio_file_path", None))

    if inference_mode == "multiturn_user_audio":
        is_multiturn_user_audio = True
    elif inference_mode == "single_turn":
        is_multiturn_user_audio = False
    elif inference_mode == "auto":
        is_multiturn_user_audio = any(looks_multiturn_user_audio(sample) for sample in batch)
    else:
        raise ValueError(f"Unknown inference_mode={inference_mode}")

    out_dict = {
        "multiturn_user_audio": bool(is_multiturn_user_audio),
        "dataset_indices": [int(s.get("__dataset_index__", -1)) for s in batch],
    }

    if is_multiturn_user_audio:
        max_turns = 1
        for sample in batch:
            if isinstance(sample["text"], list):
                max_turns = max(max_turns, len(sample["text"]))

        raw_turn_texts = []
        for sample in batch:
            if isinstance(sample["text"], list):
                raw_turn_texts.append([str(x) for x in sample["text"]])
            else:
                raw_turn_texts.append([str(sample["text"])])

        batched_turns = []
        batched_turn_lens = []
        valid_turn_masks = []

        for turn_id in range(max_turns):
            turn_tokens = []
            turn_lens = []
            turn_valid = []
            for sample in batch:
                text_data = sample["text"]
                if isinstance(text_data, list):
                    if turn_id < len(text_data):
                        seg_ids = model.tokenizer.encode(text_data[turn_id], tokenizer_name=main_tokenizer_name) + [
                            model.eos_id
                        ]
                        turn_tokens.append(torch.as_tensor(seg_ids, dtype=torch.long))
                        turn_lens.append(len(seg_ids))
                        turn_valid.append(True)
                    else:
                        turn_tokens.append(torch.as_tensor([model.pad_id], dtype=torch.long))
                        turn_lens.append(1)
                        turn_valid.append(False)
                else:
                    if turn_id == 0:
                        seg_ids = model.tokenizer.encode(text_data, tokenizer_name=main_tokenizer_name) + [model.eos_id]
                        turn_tokens.append(torch.as_tensor(seg_ids, dtype=torch.long))
                        turn_lens.append(len(seg_ids))
                        turn_valid.append(True)
                    else:
                        turn_tokens.append(torch.as_tensor([model.pad_id], dtype=torch.long))
                        turn_lens.append(1)
                        turn_valid.append(False)

            batched_turns.append(pad_sequence(turn_tokens, batch_first=True, padding_value=model.pad_id))
            batched_turn_lens.append(torch.tensor(turn_lens, dtype=torch.long))
            valid_turn_masks.append(torch.tensor(turn_valid, dtype=torch.bool))

        user_audio_by_turn = [[] for _ in range(max_turns)]
        user_audio_lens_by_turn = [[] for _ in range(max_turns)]

    else:
        # Single-turn regular inference: one text segment per sample, batched.
        raw_turn_texts = []
        single_turn_tokens = []
        single_turn_lens = []
        for sample in batch:
            text_data = sample["text"]
            if isinstance(text_data, list):
                text = " ".join(str(x) for x in text_data)
            else:
                text = str(text_data)
            raw_turn_texts.append([text])
            seg_ids = model.tokenizer.encode(text, tokenizer_name=main_tokenizer_name) + [model.eos_id]
            single_turn_tokens.append(torch.as_tensor(seg_ids, dtype=torch.long))
            single_turn_lens.append(len(seg_ids))

        out_dict["input_ids"] = pad_sequence(single_turn_tokens, batch_first=True, padding_value=model.pad_id)
        out_dict["input_lengths"] = torch.tensor(single_turn_lens, dtype=torch.long)
        user_audio_by_turn = []
        user_audio_lens_by_turn = []

    audio_list = []
    audio_lengths = []

    for i, sample in enumerate(batch):
        context_path = _resolve_audio_path(sample.get("context_audio_filepath"), root_path)
        context_wav = _load_audio(context_path, sample_rate, normalize=normalize_audio_volume, use_librosa=use_librosa)
        audio_list.append(context_wav)
        audio_lengths.append(len(context_wav))

        if is_multiturn_user_audio:
            user_audio_paths = sample.get("user_audio_file_path", None)
            for turn_id in range(len(user_audio_by_turn)):
                has_valid_text_turn = (
                    isinstance(sample["text"], list) and turn_id < len(sample["text"])
                ) or ((not isinstance(sample["text"], list)) and turn_id == 0)

                if (
                    isinstance(user_audio_paths, list)
                    and turn_id < len(user_audio_paths)
                    and user_audio_paths[turn_id]
                    and has_valid_text_turn
                ):
                    user_path = _resolve_audio_path(user_audio_paths[turn_id], root_path)
                    user_wav = _load_audio(
                        user_path,
                        sample_rate=sample_rate,
                        normalize=normalize_audio_volume,
                        use_librosa=use_librosa,
                    )
                else:
                    user_wav = torch.zeros(int(2 * sample_rate), dtype=torch.float32)

                user_audio_by_turn[turn_id].append(user_wav)
                user_audio_lens_by_turn[turn_id].append(len(user_wav))

    max_audio_len = max(audio_lengths)
    batch_size = len(audio_lengths)
    padded_audio = torch.zeros((batch_size, max_audio_len), dtype=torch.float32)
    for i, wav in enumerate(audio_list):
        padded_audio[i, : len(wav)] = wav

    if is_multiturn_user_audio:
        padded_user_audio_turns = []
        padded_user_audio_turn_lens = []
        for turn_id in range(len(user_audio_by_turn)):
            turn_lens = user_audio_lens_by_turn[turn_id]
            max_turn_audio_len = max(turn_lens)
            padded_turn_audio = torch.zeros((batch_size, max_turn_audio_len), dtype=torch.float32)
            for i, wav in enumerate(user_audio_by_turn[turn_id]):
                padded_turn_audio[i, : len(wav)] = wav
            padded_user_audio_turns.append(padded_turn_audio)
            padded_user_audio_turn_lens.append(torch.tensor(turn_lens, dtype=torch.long))

        out_dict["batched_turns"] = batched_turns
        out_dict["batched_turn_lens"] = batched_turn_lens
        out_dict["valid_turn_masks"] = valid_turn_masks
        out_dict["user_audio_turns"] = padded_user_audio_turns
        out_dict["user_audio_turns_lens"] = padded_user_audio_turn_lens

    out_dict["context_audio"] = padded_audio
    out_dict["context_audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)
    out_dict["target_audio_paths"] = [s["audio_filepath"] for s in batch]
    out_dict["raw_text"] = [" ".join(x) for x in raw_turn_texts]
    out_dict["raw_turn_texts"] = raw_turn_texts

    return out_dict


# -----------------------------
# Model / generation
# -----------------------------


def attach_dtype_counter(model):
    handles = []
    stats = {}
    examples = {}

    def is_leaf(module):
        return len(list(module.children())) == 0

    def get_dtype(x):
        if torch.is_tensor(x):
            return str(x.dtype)
        if isinstance(x, (list, tuple)):
            for t in x:
                if torch.is_tensor(t):
                    return str(t.dtype)
        return "other"

    def get_module_group(name):
        return name.split(".")[0] if "." in name else name

    def hook_fn(name):
        def fn(module, inputs, outputs):
            dtype = get_dtype(outputs)
            if dtype not in ["torch.float16", "torch.bfloat16", "torch.float32"]:
                dtype = "other"
            group = get_module_group(name)
            if group not in stats:
                stats[group] = {
                    "torch.float16": 0,
                    "torch.bfloat16": 0,
                    "torch.float32": 0,
                    "other": 0,
                }
                examples[group] = {
                    "torch.float16": [],
                    "torch.bfloat16": [],
                    "torch.float32": [],
                    "other": [],
                }
            stats[group][dtype] += 1
            if len(examples[group][dtype]) < 3:
                examples[group][dtype].append(module.__class__.__name__)

        return fn

    for name, module in model.named_modules():
        if is_leaf(module):
            handles.append(module.register_forward_hook(hook_fn(name)))
    return handles, stats, examples


def report_dtype_stats(handles, stats, examples, rank=0):
    for h in handles:
        h.remove()
    logging.info(f"[rank={rank}] === DTYPE USAGE PER MODULE ===")
    for group, group_stats in stats.items():
        total = sum(group_stats.values())
        if total == 0:
            continue
        logging.info(f"[rank={rank}] --- {group} ---")
        for dtype, count in group_stats.items():
            if count > 0:
                logging.info(f"[rank={rank}] {dtype}: {count} ({100 * count / total:.2f}%)")
    logging.info(f"[rank={rank}] === DTYPE EXAMPLES ===")
    for group, group_examples in examples.items():
        for dtype, mods in group_examples.items():
            if mods:
                logging.info(f"[rank={rank}] {group} {dtype}: {mods}")


def build_model_and_codec(args, target_device, target_dtype):
    model_cfg = EasyMagpieTTSInferenceModel.restore_from(args.checkpoint_path, return_config=True)

    with open_dict(model_cfg):
        model_cfg.target = "nemo.collections.tts.models.easy_magpietts_inference.EasyMagpieTTSInferenceModel"
        model_cfg.codecmodel_path = args.codec_model_path
        model_cfg.train_ds = None
        model_cfg.validation_ds = None
        model_cfg.run_val_inference = False
        model_cfg.use_utmos = False
        model_cfg.use_meta_init_for_decoder = True

        if args.phoneme_tokenizer_path and getattr(model_cfg, "phoneme_tokenizer", None) is not None:
            model_cfg.phoneme_tokenizer.tokenizer_path = args.phoneme_tokenizer_path

    model = EasyMagpieTTSInferenceModel.restore_from(
        args.checkpoint_path,
        override_config_path=model_cfg,
        map_location=torch.device("cpu"),
    )
    model.use_kv_cache_for_inference = True
    model.to(dtype=target_dtype)
    model.eval().to(target_device)

    model.input_samples_per_frame = int(model.codec_model_samples_per_frame * model.frame_stacking_factor)
    model.target_samples_per_frame = model.input_samples_per_frame / (model.sample_rate / model.output_sample_rate)

    codec_model = AudioCodecModel.restore_from(args.codec_model_path, strict=False, map_location=torch.device("cpu"))
    if hasattr(codec_model, "discriminator"):
        del codec_model.discriminator
    codec_model.freeze()
    codec_model = codec_model.to(target_device).eval()

    codec_converter = None
    if getattr(model, "_codec_converter", None) is not None:
        vq_new = deepcopy(model._codec_converter.vector_quantizer_new).to(target_device).eval()
        codec_converter = VectorQuantizerIndexConverter(
            vector_quantizer_original=codec_model.vector_quantizer,
            vector_quantizer_new=vq_new,
        ).to(target_device).eval()

    model._codec_helper = CodecHelper(codec_model=codec_model, codec_converter=codec_converter)
    model._generate_codec_silence_buffer()

    return model


def prepare_inputs_for_device(inputs, model, args, target_dtype, speaker_wav=None):
    B = inputs["context_audio"].size(0)
    device = model.device

    inputs["context_audio"] = inputs["context_audio"].to(device, dtype=target_dtype)
    inputs["context_audio_lengths"] = inputs["context_audio_lengths"].to(device)

    if args.user_custom_speaker_reference and speaker_wav is not None:
        inputs["context_audio"] = speaker_wav.repeat(B, 1).detach()
        inputs["context_audio_lengths"] = torch.full((B,), speaker_wav.size(-1), dtype=torch.long, device=device)

    if "user_audio_turns" in inputs:
        inputs["user_audio_turns"] = [x.to(device, dtype=target_dtype) for x in inputs["user_audio_turns"]]
        inputs["user_audio_turns_lens"] = [x.to(device) for x in inputs["user_audio_turns_lens"]]

    return inputs




def run_single_turn_generation(model, inputs, args):
    """Regular batched single-turn EasyMagpieTTS generation.

    This path does not prefill with user speech or synthetic silence. It is for
    classic single-turn datasets such as LibriTTS and supports batch_size > 1.
    """
    B = inputs["context_audio"].size(0)
    device = model.device

    with torch.inference_mode():
        wav = inputs["context_audio"]
        wav_len = inputs["context_audio_lengths"]
        codes, codes_lens = model._codec_helper.audio_to_codes(wav, wav_len)

        use_lang = bool(getattr(model, "add_language_to_context_text", False))
        ctx_text = f"[{args.language.upper()}]" if use_lang else "[NO TEXT CONTEXT]"
        ctx_text_ids = model.tokenizer.encode(ctx_text, tokenizer_name=model.text_conditioning_tokenizer_name)
        ctx_toks = torch.tensor([ctx_text_ids], dtype=torch.long, device=device).expand(B, -1)
        ctx_toks_lens = torch.tensor([len(ctx_text_ids)] * B, dtype=torch.long, device=device)

        state = model.streaming_init(
            context_audio_codes=codes,
            context_audio_codes_lens=codes_lens,
            context_text_tokens=ctx_toks,
            context_text_tokens_lens=ctx_toks_lens,
            use_cfg=args.use_cfg,
            cfg_scale=args.cfg_scale,
            use_local_transformer=True,
            temperature=args.temperature,
            topk=args.topk,
            phoneme_input_type="pred",
            phoneme_sampling_method="argmax",
            use_inference_mode=True,
        )

        text = inputs["input_ids"].to(device)
        text_lens = inputs["input_lengths"].to(device)

        turn_offsets = torch.zeros(B, dtype=torch.long, device=device)
        turn_steps = 0

        while not state.finished.all() and turn_steps < args.max_tts_steps:
            turn_steps += 1
            relative_positions = state.text_tokens_seen - turn_offsets
            positions = relative_positions.clamp(min=0, max=text.size(1) - 1)
            current_tokens = text[torch.arange(B, device=device), positions]
            exhausted = relative_positions >= text_lens
            current_tokens = torch.where(
                exhausted,
                torch.full_like(current_tokens, model.eos_id),
                current_tokens,
            )
            state, _, _ = model.streaming_step(
                state=state,
                text_tokens=current_tokens,
                use_inference_mode=True,
            )

        generated_codes = None
        if getattr(state, "all_predictions", None):
            try:
                generated_codes = torch.cat(state.all_predictions, dim=-1).detach()
            except Exception:
                generated_codes = None

        finalize_output = model.streaming_finalize(state, use_inference_mode=True)

    # For single turn, there is one generated segment per sample and no
    # multiturn frame alignment needed.
    return finalize_output, [], 0, generated_codes

def run_generation(model, inputs, args, codec_sil_codes):
    """Run either multiturn-user-audio or regular single-turn generation."""
    if not inputs.get("multiturn_user_audio", False):
        return run_single_turn_generation(model, inputs, args)

    B = inputs["context_audio"].size(0)
    if B != 1:
        raise RuntimeError("Multiturn user-audio inference requires --batch_size=1 per process.")

    device = model.device
    multiturn_turn_frame_ranges = []
    multiturn_decode_start_frame = 0

    with torch.inference_mode():
        wav = inputs["context_audio"]
        wav_len = inputs["context_audio_lengths"]
        codes, codes_lens = model._codec_helper.audio_to_codes(wav, wav_len)

        use_lang = bool(getattr(model, "add_language_to_context_text", False))
        ctx_text = f"[{args.language.upper()}]" if use_lang else "[NO TEXT CONTEXT]"
        ctx_text_ids = model.tokenizer.encode(ctx_text, tokenizer_name=model.text_conditioning_tokenizer_name)
        ctx_toks = torch.tensor([ctx_text_ids], dtype=torch.long, device=device).expand(B, -1)
        ctx_toks_lens = torch.tensor([len(ctx_text_ids)] * B, dtype=torch.long, device=device)

        state = model.streaming_init(
            context_audio_codes=codes,
            context_audio_codes_lens=codes_lens,
            context_text_tokens=ctx_toks,
            context_text_tokens_lens=ctx_toks_lens,
            use_cfg=args.use_cfg,
            cfg_scale=args.cfg_scale,
            use_local_transformer=True,
            temperature=args.temperature,
            topk=args.topk,
            phoneme_input_type="pred",
            phoneme_sampling_method="argmax",
            use_inference_mode=True,
        )

        batched_turns = inputs["batched_turns"]
        batched_turn_lens = inputs["batched_turn_lens"]
        valid_turn_masks = inputs["valid_turn_masks"]

        for turn_id in range(len(batched_turns)):
            turn_text = batched_turns[turn_id].to(device)
            turn_lens = batched_turn_lens[turn_id].to(device)
            valid_mask = valid_turn_masks[turn_id].to(device)
            if not bool(valid_mask[0].item()):
                continue

            state.finished.zero_()
            state.text_finished.zero_()
            state.audio_prediction_end_idx.fill_(-1)
            if hasattr(state, "turn_text_tokens_seen"):
                state.turn_text_tokens_seen.zero_()
            if hasattr(state, "phoneme_steps"):
                state.phoneme_steps.zero_()
            if hasattr(state, "phoneme_stream_ended"):
                state.phoneme_stream_ended.zero_()
            if hasattr(state, "phoneme_eos_detected"):
                state.phoneme_eos_detected.zero_()
            state.last_phoneme_tokens = None

            if not model.cfg.get("condition_on_user_speech", False):
                user_audio = inputs["user_audio_turns"][turn_id]
                user_audio_prefill_steps = int(round(user_audio.size(-1) / model.input_samples_per_frame))
                user_audio_prefill_seconds = user_audio_prefill_steps * model.input_samples_per_frame / model.sample_rate
                user_audio_prefill_tokens = torch.full((1, user_audio_prefill_steps), model.pad_id, dtype=torch.long, device=device)
                user_audio_channel_embedding = None
            else:
                user_audio = inputs["user_audio_turns"][turn_id]
                user_audio_lens = inputs["user_audio_turns_lens"][turn_id]
                user_audio_codes, user_audio_codes_lens = model._codec_helper.audio_to_codes(user_audio, user_audio_lens)

                if model._codec_converter is not None:
                    user_audio_codes = model._codec_converter.convert_original_to_new(
                        audio_tokens=user_audio_codes,
                        audio_lens=user_audio_codes_lens,
                    ).long()

                user_audio_codes, user_audio_codes_lens = model.stack_codes(
                    user_audio_codes,
                    user_audio_codes_lens,
                    model.audio_bos_id,
                    model.audio_eos_id,
                    model.frame_stacking_factor,
                    model.num_audio_codebooks,
                )
                user_audio_embedded = model.embed_audio_tokens(user_audio_codes)

                boundary_trim = model.cfg.get("user_audio_boundary_trim", 4)
                boundary_trim = 0 if boundary_trim is None else int(boundary_trim)
                if boundary_trim == 0:
                    real_start = 0
                    real_end = int(user_audio_codes_lens[0].item())
                else:
                    turn_len_with_special = int(user_audio_codes_lens[0].item())
                    real_start = 1
                    real_end = max(real_start, turn_len_with_special - 1)

                user_audio_embedded = user_audio_embedded[:, real_start:real_end]
                copy_len = user_audio_embedded.size(1)
                if boundary_trim > 0:
                    trim = min(boundary_trim, copy_len // 2)
                    if trim > 0:
                        user_audio_embedded[:, :trim] = 0.0
                        user_audio_embedded[:, copy_len - trim :] = 0.0

                bos_user_pad = torch.zeros(
                    user_audio_embedded.size(0),
                    1,
                    user_audio_embedded.size(2),
                    device=user_audio_embedded.device,
                    dtype=user_audio_embedded.dtype,
                )
                user_audio_embedded = torch.cat([bos_user_pad, user_audio_embedded], dim=1)
                user_audio_prefill_steps = user_audio_embedded.size(1)
                user_audio_prefill_tokens = torch.full((B, user_audio_prefill_steps), model.pad_id, dtype=torch.long, device=device)
                user_audio_channel_embedding = user_audio_embedded
                user_audio_prefill_seconds = user_audio_prefill_steps * model.input_samples_per_frame / model.sample_rate

            delay_tokens = int(state.config.training_mode.streaming_speech_delay)
            delay_tokens = min(delay_tokens, int(turn_lens[0].item()), user_audio_prefill_steps)

            warmup_tokens = turn_text[:, :delay_tokens]
            turn_text = turn_text[:, delay_tokens:]
            turn_lens = torch.clamp(turn_lens - delay_tokens, min=0)

            if user_audio_channel_embedding is not None and delay_tokens > 0:
                warmup_user_audio = user_audio_channel_embedding[:, -delay_tokens:]
                user_audio_channel_embedding = user_audio_channel_embedding[:, :-delay_tokens]
                user_audio_prefill_tokens = user_audio_prefill_tokens[:, :-delay_tokens]
            else:
                warmup_user_audio = None

            if user_audio_prefill_tokens.size(1) > 0:
                state = model.streaming_prefill_profile(
                    state=state,
                    text_tokens=user_audio_prefill_tokens,
                    use_inference_mode=True,
                    user_audio_channel_embedding=user_audio_channel_embedding,
                )

            for i in range(delay_tokens):
                user_step_emb = warmup_user_audio[:, i] if warmup_user_audio is not None else None
                state.finished.zero_()
                state, _, _ = model.streaming_step(
                    state=state,
                    text_tokens=warmup_tokens[:, i],
                    user_audio_channel_embedding=user_step_emb,
                    prefill_like_step=not bool(model.cfg.get("agent_mask_include_transition_prefix", False)),
                    prefill_like_is_last_step=(i == delay_tokens - 1),
                    use_inference_mode=True,
                )

            logging.info(f"[multiturn_user_audio] turn={turn_id} prefilled {user_audio_prefill_steps} steps ({user_audio_prefill_seconds:.2f}s)")

            turn_start_frame = sum(p.size(-1) for p in state.all_predictions)
            if turn_id == 0:
                state.audio_prediction_start_idx.fill_(turn_start_frame)
                multiturn_decode_start_frame = turn_start_frame

            turn_offset = state.text_tokens_seen.clone()
            turn_steps = 0
            saw_audio = False
            turn_ended_with_audio_eos = False

            while turn_steps < args.max_tts_steps:
                turn_steps += 1
                state.finished.zero_()
                relative_position = state.text_tokens_seen - turn_offset
                text_exhausted = relative_position >= turn_lens

                if turn_text.size(1) == 0:
                    current_tokens = torch.full((B,), model.eos_id, dtype=torch.long, device=device)
                else:
                    position = relative_position.clamp(min=0, max=turn_text.size(1) - 1)
                    current_tokens = turn_text[torch.arange(B, device=device), position]
                    current_tokens = torch.where(
                        text_exhausted,
                        torch.full_like(current_tokens, model.eos_id),
                        current_tokens,
                    )

                state, audio_codes, _ = model.streaming_step(
                    state=state,
                    text_tokens=current_tokens,
                    use_inference_mode=True,
                )

                if audio_codes is not None and not saw_audio:
                    saw_audio = True

                if bool(text_exhausted[0].item()) and bool(state.finished[0].item()):
                    turn_ended_with_audio_eos = True
                    break

            state.audio_prediction_end_idx.fill_(-1)
            state.finished.zero_()
            turn_end_frame = sum(p.size(-1) for p in state.all_predictions)
            multiturn_turn_frame_ranges.append((turn_id, turn_start_frame, turn_end_frame))
            logging.info(
                f"[multiturn_user_audio] turn={turn_id} steps={turn_steps} "
                f"saw_audio={saw_audio} ended_with_audio_eos={turn_ended_with_audio_eos}"
            )

        bos_id = getattr(model, "audio_bos_id", -1)
        eos_id = getattr(model, "audio_eos_id", -1)
        speaking_id = getattr(model, "audio_user_speaking_id", -1)
        speaking_end_id = getattr(model, "audio_user_speaking_end_id", -1)
        sil_injection = codec_sil_codes.view(1, -1, 1)

        for step_idx in range(len(state.all_predictions)):
            pred = state.all_predictions[step_idx]
            mask = (pred == bos_id) | (pred == eos_id) | (pred == speaking_id) | (pred == speaking_end_id)
            frame_mask = mask.any(dim=1, keepdim=True)
            if frame_mask.any():
                state.all_predictions[step_idx] = torch.where(frame_mask, sil_injection.expand_as(pred), pred)

        state.audio_prediction_end_idx.fill_(-1)

        generated_codes = None
        if getattr(state, "all_predictions", None):
            try:
                generated_codes = torch.cat(state.all_predictions, dim=-1).detach()
            except Exception:
                generated_codes = None

        finalize_output = model.streaming_finalize(state, use_inference_mode=True)

    return finalize_output, multiturn_turn_frame_ranges, multiturn_decode_start_frame, generated_codes


def load_speaker_wav_if_needed(args, model, target_dtype):
    if args.user_custom_speaker_reference and args.inference_speaker_reference:
        return _load_audio(
            args.inference_speaker_reference,
            model.sample_rate,
            normalize=args.normalize_volume,
            use_librosa=args.use_librosa,
        ).unsqueeze(0).to(model.device, dtype=target_dtype)

    return None


# -----------------------------
# Save generation outputs and metric manifests
# -----------------------------


def write_audio_1d(path: str, wav: torch.Tensor, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav_np = wav.detach().cpu().float().numpy()
    sf.write(path, wav_np, samplerate=sr)


def build_metric_item(
    run_id: int,
    rank: int,
    dataset_index: int,
    turn_id: int,
    target_audio_path: str,
    reference_text: str,
    pred_audio_path: str,
    context_audio_path: str,
    pred_audio_samples: int,
    context_audio_samples: int,
    output_sample_rate: int,
    context_sample_rate: int,
    predicted_codes_path: str = None,
    sample_pred_audio_path: str = None,
    sample_predicted_codes_path: str = None,
):
    return {
        "run_id": int(run_id),
        "rank": int(rank),
        "dataset_index": int(dataset_index),
        "turn_id": int(turn_id),
        "target_audio_path": target_audio_path,
        "reference_text": reference_text,
        "pred_audio_path": pred_audio_path,
        "context_audio_path": context_audio_path,
        "pred_audio_samples": int(pred_audio_samples),
        "context_audio_samples": int(context_audio_samples),
        "pred_audio_seconds": float(pred_audio_samples / output_sample_rate),
        "context_audio_seconds": float(context_audio_samples / context_sample_rate),
        "output_sample_rate": int(output_sample_rate),
        "context_sample_rate": int(context_sample_rate),
        "predicted_codes_path": predicted_codes_path,
        "sample_pred_audio_path": sample_pred_audio_path or pred_audio_path,
        "sample_predicted_codes_path": sample_predicted_codes_path or predicted_codes_path,
    }


def save_generated_code_slice(generated_codes, batch_idx: int, start_frame: int, end_frame: int, path: str):
    """Save predicted codec codes as [num_codebooks, T] for MagpieTTS FCD."""
    if generated_codes is None:
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        T = int(generated_codes.size(-1))
        start_frame = max(0, min(int(start_frame), T))
        end_frame = max(start_frame, min(int(end_frame), T))
        if end_frame <= start_frame:
            return None
        codes = generated_codes[batch_idx, :, start_frame:end_frame].detach().cpu().long()
        if codes.numel() == 0:
            return None
        torch.save(codes, path)
        return path
    except Exception as e:
        logging.warning(f"Could not save predicted codes to {path}: {repr(e)}")
        return None


def save_generation_outputs_and_build_metric_items(
    model,
    inputs,
    finalize_output,
    multiturn_turn_frame_ranges,
    multiturn_decode_start_frame,
    generated_codes,
    args,
    rank: int,
    run_id: int,
):
    device = model.device
    B = inputs["context_audio"].size(0)

    with fp32_precision():
        audio_f32 = finalize_output.audio.float()
        audio_len = finalize_output.audio_len.int()

        # Use model-reported generated audio lengths for both supported modes.
        audio_len = torch.clamp(audio_len, max=audio_f32.size(1))

        audio_out_dir = get_audio_out_dir(args)
        metric_turn_dir = get_generated_turn_audio_dir(args)
        metric_context_dir = get_context_metric_audio_dir(args)
        predicted_codes_dir = get_predicted_codes_dir(args)
        os.makedirs(audio_out_dir, exist_ok=True)
        os.makedirs(metric_turn_dir, exist_ok=True)
        os.makedirs(metric_context_dir, exist_ok=True)
        os.makedirs(predicted_codes_dir, exist_ok=True)

        audio_f32_cpu = audio_f32.detach().cpu()
        audio_len_cpu = audio_len.detach().cpu()
        metric_items = []

        for i in range(B):
            target_path = inputs["target_audio_paths"][i]
            base_name = os.path.basename(target_path)
            stem, ext = os.path.splitext(base_name)
            if not ext:
                ext = ".wav"

            dataset_idx = int(inputs.get("dataset_indices", [-1] * B)[i])
            safe_stem = (
                f"run{run_id:02d}_idx{dataset_idx:08d}_{stem}"
                if dataset_idx >= 0
                else f"run{run_id:02d}_rank{rank}_{stem}"
            )

            context_len = int(inputs["context_audio_lengths"][i].detach().cpu().item())
            context_wav = inputs["context_audio"][i, :context_len].detach().cpu().float()
            context_metric_path = os.path.join(metric_context_dir, f"{safe_stem}_context.wav")
            write_audio_1d(context_metric_path, context_wav, model.sample_rate)

            if inputs.get("multiturn_user_audio", False):
                full_len = int(audio_len_cpu[i].item())
                full_wav_t = audio_f32_cpu[i, :full_len].float()
                full_out_path = os.path.join(audio_out_dir, f"{safe_stem}{ext}")

                full_codes_path = os.path.join(predicted_codes_dir, f"{safe_stem}_sample.pt")
                sample_predicted_codes_path = save_generated_code_slice(
                    generated_codes,
                    i,
                    multiturn_decode_start_frame,
                    generated_codes.size(-1) if generated_codes is not None else multiturn_decode_start_frame,
                    full_codes_path,
                )

                samples_per_prediction_frame = model.codec_model_samples_per_frame / (
                    model.sample_rate / model.output_sample_rate
                )

                aligned_agent = torch.zeros_like(full_wav_t)
                raw_turn_texts = inputs.get("raw_turn_texts", [[] for _ in range(B)])

                for turn_id, start_frame, end_frame in multiturn_turn_frame_ranges:
                    rel_start_frame = start_frame - multiturn_decode_start_frame
                    rel_end_frame = end_frame - multiturn_decode_start_frame

                    start_sample = int(round(rel_start_frame * samples_per_prediction_frame))
                    end_sample = int(round(rel_end_frame * samples_per_prediction_frame))

                    start_sample = max(0, min(start_sample, full_len))
                    end_sample = max(start_sample, min(end_sample, full_len))

                    aligned_agent[start_sample:end_sample] = full_wav_t[start_sample:end_sample]

                    turn_wav = aligned_agent[start_sample:end_sample].float()
                    turn_out_path = os.path.join(audio_out_dir, f"{safe_stem}_turn_{turn_id}{ext}")
                    write_audio_1d(turn_out_path, turn_wav, model.output_sample_rate)

                    metric_turn_path = os.path.join(metric_turn_dir, f"{safe_stem}_turn_{turn_id}.wav")
                    write_audio_1d(metric_turn_path, turn_wav, model.output_sample_rate)

                    turn_codes_path = os.path.join(predicted_codes_dir, f"{safe_stem}_turn_{turn_id}.pt")
                    predicted_codes_path = save_generated_code_slice(
                        generated_codes,
                        i,
                        start_frame,
                        end_frame,
                        turn_codes_path,
                    )

                    if turn_id < len(raw_turn_texts[i]):
                        metric_items.append(
                            build_metric_item(
                                run_id=run_id,
                                rank=rank,
                                dataset_index=dataset_idx,
                                turn_id=turn_id,
                                target_audio_path=target_path,
                                reference_text=str(raw_turn_texts[i][turn_id]),
                                pred_audio_path=metric_turn_path,
                                context_audio_path=context_metric_path,
                                pred_audio_samples=int(turn_wav.numel()),
                                context_audio_samples=int(context_wav.numel()),
                                output_sample_rate=model.output_sample_rate,
                                context_sample_rate=model.sample_rate,
                                predicted_codes_path=predicted_codes_path,
                                sample_pred_audio_path=full_out_path,
                                sample_predicted_codes_path=sample_predicted_codes_path,
                            )
                        )

                write_audio_1d(full_out_path, aligned_agent, model.output_sample_rate)

                if "user_audio_turns" in inputs:
                    user_segments = []

                    first_user_len_in = int(inputs["user_audio_turns_lens"][0][i].item())
                    first_user_delay_out = int(round(first_user_len_in * model.output_sample_rate / model.sample_rate))

                    for turn_id, start_frame, _ in multiturn_turn_frame_ranges:
                        if turn_id >= len(inputs["user_audio_turns"]):
                            continue

                        turn_audio = inputs["user_audio_turns"][turn_id][i].detach().cpu().float()
                        turn_audio_len = int(inputs["user_audio_turns_lens"][turn_id][i].item())
                        turn_audio = turn_audio[:turn_audio_len]

                        turn_audio_out = resample(
                            turn_audio.unsqueeze(0),
                            model.sample_rate,
                            model.output_sample_rate,
                        ).squeeze(0)

                        if turn_id == 0:
                            user_start_sample = 0
                        else:
                            prev_turn_end_frame = multiturn_turn_frame_ranges[turn_id - 1][2]
                            rel_prev_end_frame = prev_turn_end_frame - multiturn_decode_start_frame
                            user_start_sample = first_user_delay_out + int(
                                round(rel_prev_end_frame * samples_per_prediction_frame)
                            )

                        user_segments.append((user_start_sample, turn_audio_out.detach().cpu().float()))

                    total_user_len = 0
                    for s, wav_seg in user_segments:
                        total_user_len = max(total_user_len, s + wav_seg.numel())

                    user_ch = torch.zeros(total_user_len)
                    for s, wav_seg in user_segments:
                        e = s + wav_seg.numel()
                        user_ch[s:e] += wav_seg

                    agent_ch = torch.cat([torch.zeros(first_user_delay_out, dtype=aligned_agent.dtype), aligned_agent])

                    stereo_len = max(user_ch.numel(), agent_ch.numel())
                    user_pad = torch.zeros(stereo_len)
                    agent_pad = torch.zeros(stereo_len)

                    user_pad[: user_ch.numel()] = user_ch
                    agent_pad[: agent_ch.numel()] = agent_ch

                    stereo = torch.stack([user_pad, agent_pad], dim=1).numpy()
                    aligned_path = os.path.join(audio_out_dir, f"{safe_stem}_user_agent_aligned{ext}")
                    sf.write(aligned_path, stereo, samplerate=model.output_sample_rate)

            else:
                full_len = int(audio_len_cpu[i].item())
                wav = audio_f32_cpu[i, :full_len].float()
                out_path = os.path.join(audio_out_dir, f"{safe_stem}{ext}")
                write_audio_1d(out_path, wav, model.output_sample_rate)

                metric_turn_path = os.path.join(metric_turn_dir, f"{safe_stem}_turn_0.wav")
                write_audio_1d(metric_turn_path, wav, model.output_sample_rate)

                codes_path = os.path.join(predicted_codes_dir, f"{safe_stem}_turn_0.pt")
                predicted_codes_path = save_generated_code_slice(
                    generated_codes,
                    i,
                    0,
                    generated_codes.size(-1) if generated_codes is not None else 0,
                    codes_path,
                )

                metric_items.append(
                    build_metric_item(
                        run_id=run_id,
                        rank=rank,
                        dataset_index=dataset_idx,
                        turn_id=0,
                        target_audio_path=target_path,
                        reference_text=str(inputs["raw_text"][i]),
                        pred_audio_path=metric_turn_path,
                        context_audio_path=context_metric_path,
                        pred_audio_samples=int(wav.numel()),
                        context_audio_samples=int(context_wav.numel()),
                        output_sample_rate=model.output_sample_rate,
                        context_sample_rate=model.sample_rate,
                        predicted_codes_path=predicted_codes_path,
                        sample_pred_audio_path=out_path,
                        sample_predicted_codes_path=predicted_codes_path,
                    )
                )

    return metric_items


# -----------------------------
# Metrics after generation
# -----------------------------


def torch_rms_norm(wav: torch.Tensor, db_level: float = -27.0) -> torch.Tensor:
    denom = torch.sum(wav**2)
    if denom <= 0:
        return wav
    r = 10 ** (db_level / 20)
    a = torch.sqrt((wav.size(-1) * (r**2)) / denom)
    return wav * a


def _load_audio_for_metric(path: str, sample_rate: int):
    wav = _load_audio(path, sample_rate=sample_rate, normalize=False, use_librosa=False)
    if wav.numel() == 0:
        wav = torch.zeros(1, dtype=torch.float32)
    return wav.float()


def _pad_audio_1d_list(wavs: List[torch.Tensor], device, dtype=torch.float32):
    if len(wavs) == 0:
        return torch.zeros((0, 1), device=device, dtype=dtype), torch.zeros((0,), device=device, dtype=torch.long)

    lens = torch.tensor([max(1, int(w.numel())) for w in wavs], device=device, dtype=torch.long)
    max_len = int(lens.max().item())
    out = torch.zeros((len(wavs), max_len), device=device, dtype=dtype)

    for i, w in enumerate(wavs):
        w = w.to(device=device, dtype=dtype).flatten()
        if w.numel() == 0:
            continue
        out[i, : w.numel()] = w

    return out, lens


def chunk_list(xs: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(xs), chunk_size):
        yield xs[start : start + chunk_size]


def _metric_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_metric_batch_audio(batch_items: List[Dict[str, Any]], args):
    pred_wavs = []
    context_wavs = []

    for item in batch_items:
        pred = _load_audio_for_metric(item["pred_audio_path"], sample_rate=int(item["output_sample_rate"]))
        context = _load_audio_for_metric(item["context_audio_path"], sample_rate=int(item["context_sample_rate"]))

        if args.max_metric_audio_sec is not None:
            max_pred_len = int(float(args.max_metric_audio_sec) * int(item["output_sample_rate"]))
            pred = pred[: max(1, max_pred_len)]

        pred_wavs.append(pred)
        context_wavs.append(context)

    device = _metric_device()
    pred_audio, pred_lens = _pad_audio_1d_list(pred_wavs, device=device)
    context_audio, context_lens = _pad_audio_1d_list(context_wavs, device=device)
    output_sample_rate = int(batch_items[0]["output_sample_rate"])
    context_sample_rate = int(batch_items[0]["context_sample_rate"])

    return pred_audio, pred_lens, context_audio, context_lens, output_sample_rate, context_sample_rate



def _nan():
    return float("nan")


def finite_avg(values):
    finite_values = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            finite_values.append(value)
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _safe_word_error_detail(hyp_text: str, ref_text: str, use_cer: bool):
    ref_text = "" if ref_text is None else str(ref_text).strip()
    hyp_text = "" if hyp_text is None else str(hyp_text).strip()
    if ref_text == "":
        return None
    try:
        detailed = word_error_rate_detail(hypotheses=[hyp_text], references=[ref_text], use_cer=use_cer)
        value = float(detailed[0])
        if not math.isfinite(value):
            return None
        return detailed
    except Exception:
        return None


def _safe_detail_value(detailed):
    if detailed is None:
        return None
    try:
        value = float(detailed[0])
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _load_speaker_eval_models(args, device: str):
    """Load speaker verification models in the same style as MagpieTTS evaluation utils."""
    models = {
        "feature_extractor": None,
        "sv_model": None,
        "sv_model_alternate": None,
    }

    if nemo_asr is None or extract_embedding is None:
        logging.warning("Speaker metric dependencies are unavailable; speaker similarity metrics will be NaN.")
        return models

    try:
        if args.sv_model_type == "wavlm":
            if Wav2Vec2FeatureExtractor is None or WavLMForXVector is None:
                raise RuntimeError("transformers WavLM dependencies are unavailable")
            models["feature_extractor"] = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
            models["sv_model"] = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
        else:
            models["sv_model"] = (
                nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large").to(device).eval()
            )

        logging.info("Loading alternate speaker model `titanet_small`.")
        with logging.temp_verbosity(logging.ERROR):
            models["sv_model_alternate"] = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name="titanet_small"
            )
        models["sv_model_alternate"] = models["sv_model_alternate"].to(device).eval()
    except Exception as e:
        logging.warning(f"Could not load speaker evaluation models: {repr(e)}")
        models = {"feature_extractor": None, "sv_model": None, "sv_model_alternate": None}

    return models


def _compute_speaker_similarity_rows(args, rows: List[Dict[str, Any]]):
    """Populate pred/GT/context speaker similarity metrics per turn."""
    if args.disable_speaker_metrics:
        for row in rows:
            row["pred_gt_ssim"] = _nan()
            row["pred_context_ssim"] = _nan()
            row["gt_context_ssim"] = _nan()
            row["pred_gt_ssim_alternate"] = _nan()
            row["pred_context_ssim_alternate"] = _nan()
            row["gt_context_ssim_alternate"] = _nan()
            row["ssim"] = _nan()
        return

    device = _metric_device()
    models = _load_speaker_eval_models(args, device=device)
    sv_model = models.get("sv_model")
    sv_model_alt = models.get("sv_model_alternate")
    extractor = models.get("feature_extractor")

    if sv_model is None or sv_model_alt is None or extract_embedding is None:
        for row in rows:
            row["pred_gt_ssim"] = _nan()
            row["pred_context_ssim"] = _nan()
            row["gt_context_ssim"] = _nan()
            row["pred_gt_ssim_alternate"] = _nan()
            row["pred_context_ssim_alternate"] = _nan()
            row["gt_context_ssim_alternate"] = _nan()
            row["ssim"] = _nan()
        return

    emb_cache = {}
    emb_alt_cache = {}

    def get_emb(path: str, alternate: bool = False):
        if path is None or not path or not os.path.exists(path):
            return None
        cache = emb_alt_cache if alternate else emb_cache
        if path in cache:
            return cache[path]
        model = sv_model_alt if alternate else sv_model
        sv_type = "titanet" if alternate else args.sv_model_type
        try:
            with torch.inference_mode():
                emb = extract_embedding(
                    model=model,
                    extractor=extractor,
                    audio_path=path,
                    device=device,
                    sv_model_type=sv_type,
                )
            cache[path] = emb
            return emb
        except Exception as e:
            logging.warning(f"Could not extract speaker embedding for {path}: {repr(e)}")
            cache[path] = None
            return None

    def cosine(a, b):
        if a is None or b is None:
            return _nan()
        try:
            return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        except Exception:
            return _nan()

    for row in rows:
        pred_path = row.get("pred_audio_path")
        gt_path = row.get("target_audio_path")
        context_path = row.get("context_audio_path")

        pred = get_emb(pred_path, alternate=False)
        gt = get_emb(gt_path, alternate=False)
        context = get_emb(context_path, alternate=False)

        pred_alt = get_emb(pred_path, alternate=True)
        gt_alt = get_emb(gt_path, alternate=True)
        context_alt = get_emb(context_path, alternate=True)

        row["pred_gt_ssim"] = cosine(pred, gt)
        row["pred_context_ssim"] = cosine(pred, context)
        row["gt_context_ssim"] = cosine(gt, context)
        row["pred_gt_ssim_alternate"] = cosine(pred_alt, gt_alt)
        row["pred_context_ssim_alternate"] = cosine(pred_alt, context_alt)
        row["gt_context_ssim_alternate"] = cosine(gt_alt, context_alt)
        row["ssim"] = row["pred_context_ssim"]

    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compute_utmos_rows(args, rows: List[Dict[str, Any]], rank: int):
    if args.disable_utmosv2:
        for row in rows:
            row["utmosv2"] = _nan()
        return

    if compute_utmosv2_scores is None:
        logging.warning("UTMOSv2 utility is unavailable; setting utmosv2 to NaN.")
        for row in rows:
            row["utmosv2"] = _nan()
        return

    try:
        # All predicted metric turns are written into the same directory.
        audio_dir = get_generated_turn_audio_dir(args)
        scores = compute_utmosv2_scores(audio_dir, _metric_device())
        for row in rows:
            row["utmosv2"] = scores.get(os.path.normpath(row.get("pred_audio_path", "")), _nan())
    except Exception as e:
        all_rank_print(rank, f"UTMOSv2 computation failed; setting utmosv2 to NaN: {repr(e)}")
        for row in rows:
            row["utmosv2"] = _nan()


def _compute_eou_rows(args, rows: List[Dict[str, Any]], rank: int):
    if args.disable_eou or args.language != "en" or EoUClassifier is None:
        for row in rows:
            row["eou_type"] = None
            row["eou_trailing_duration"] = _nan()
            row["eou_trail_rms_ratio"] = _nan()
        return

    try:
        kwargs = {"device": _metric_device()}
        if args.eou_model_name:
            kwargs["model_name"] = args.eou_model_name
        classifier = EoUClassifier(**kwargs)
        items = [(row.get("pred_audio_path"), row.get("reference_text", "")) for row in rows]

        results = []
        batch_size = max(1, int(args.eou_batch_size))
        for start in range(0, len(items), batch_size):
            results.extend(classifier.classify_batch(items[start : start + batch_size]))

        for row, result in zip(rows, results):
            row["eou_type"] = result.eou_type.value
            row["eou_trailing_duration"] = result.trailing_duration
            row["eou_trail_rms_ratio"] = result.trail_rms_ratio
    except Exception as e:
        all_rank_print(rank, f"EOU computation failed; setting EOU metrics to NaN: {repr(e)}")
        for row in rows:
            row["eou_type"] = None
            row["eou_trailing_duration"] = _nan()
            row["eou_trail_rms_ratio"] = _nan()


def compute_magpie_style_global_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the same metric keys used by MagpieTTS evaluate_generated_audio."""
    n = len(rows)
    if n == 0:
        return {
            "cer_filewise_avg": None,
            "wer_filewise_avg": None,
            "cer_cumulative": None,
            "wer_cumulative": None,
            "ssim_pred_gt_avg": None,
            "ssim_pred_context_avg": None,
            "ssim_gt_context_avg": None,
            "ssim_pred_gt_avg_alternate": None,
            "ssim_pred_context_avg_alternate": None,
            "ssim_gt_context_avg_alternate": None,
            "cer_gt_audio_cumulative": _nan(),
            "wer_gt_audio_cumulative": _nan(),
            "utmosv2_avg": None,
            "total_gen_audio_seconds": 0.0,
            "frechet_codec_distance": _nan(),
            "eou_cutoff_rate": _nan(),
            "eou_silence_rate": _nan(),
            "eou_noise_rate": _nan(),
            "eou_error_rate": _nan(),
        }

    pred_texts = [str(r.get("pred_text", r.get("asr_hyp", ""))) for r in rows if r.get("gt_text", r.get("reference_text", ""))]
    gt_texts = [str(r.get("gt_text", r.get("reference_text", ""))) for r in rows if r.get("gt_text", r.get("reference_text", ""))]

    out = {}
    out["cer_filewise_avg"] = finite_avg([r.get("cer") for r in rows])
    out["wer_filewise_avg"] = finite_avg([r.get("wer") for r in rows])

    if pred_texts and gt_texts:
        try:
            out["cer_cumulative"] = float(word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=True)[0])
        except Exception:
            out["cer_cumulative"] = None
        try:
            out["wer_cumulative"] = float(word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=False)[0])
        except Exception:
            out["wer_cumulative"] = None
    else:
        out["cer_cumulative"] = None
        out["wer_cumulative"] = None

    out["ssim_pred_gt_avg"] = finite_avg([r.get("pred_gt_ssim") for r in rows])
    out["ssim_pred_context_avg"] = finite_avg([r.get("pred_context_ssim") for r in rows])
    out["ssim_gt_context_avg"] = finite_avg([r.get("gt_context_ssim") for r in rows])
    out["ssim_pred_gt_avg_alternate"] = finite_avg([r.get("pred_gt_ssim_alternate") for r in rows])
    out["ssim_pred_context_avg_alternate"] = finite_avg([r.get("pred_context_ssim_alternate") for r in rows])
    out["ssim_gt_context_avg_alternate"] = finite_avg([r.get("gt_context_ssim_alternate") for r in rows])

    gt_audio_texts = [r.get("gt_audio_text") for r in rows]
    if gt_audio_texts and all(x is not None for x in gt_audio_texts):
        try:
            out["cer_gt_audio_cumulative"] = float(
                word_error_rate_detail(hypotheses=gt_audio_texts, references=gt_texts, use_cer=True)[0]
            )
        except Exception:
            out["cer_gt_audio_cumulative"] = _nan()
        try:
            out["wer_gt_audio_cumulative"] = float(
                word_error_rate_detail(hypotheses=gt_audio_texts, references=gt_texts, use_cer=False)[0]
            )
        except Exception:
            out["wer_gt_audio_cumulative"] = _nan()
    else:
        out["cer_gt_audio_cumulative"] = _nan()
        out["wer_gt_audio_cumulative"] = _nan()

    out["utmosv2_avg"] = finite_avg([r.get("utmosv2") for r in rows])
    out["total_gen_audio_seconds"] = sum(float(r.get("total_gen_audio_seconds", r.get("pred_audio_seconds", 0.0)) or 0.0) for r in rows)
    out["frechet_codec_distance"] = _nan()

    eou_types = [r.get("eou_type") for r in rows]
    if eou_types and eou_types[0] is not None:
        counts = Counter(eou_types)
        if EoUType is not None:
            labels = list(EoUType.error_types())
            good_label = EoUType.GOOD
        else:
            labels = ["cutoff", "silence", "noise"]
            good_label = "good"

        for label in labels:
            out[f"eou_{label}_rate"] = counts.get(label, 0) / n
        out["eou_error_rate"] = 1.0 - counts.get(good_label, 0) / n
    else:
        out["eou_cutoff_rate"] = _nan()
        out["eou_silence_rate"] = _nan()
        out["eou_noise_rate"] = _nan()
        out["eou_error_rate"] = _nan()

    return out


def _load_asr_model_for_metrics(args, rank: int):
    """Load ASR directly, matching the EasyMagpie/MagpieTTS evaluation style."""
    asr_cls = ASRModel
    if asr_cls is None and nemo_asr is not None:
        asr_cls = getattr(getattr(nemo_asr, "models", None), "ASRModel", None)
    if asr_cls is None:
        raise RuntimeError("NeMo ASRModel is unavailable, cannot load ASR model.")

    all_rank_print(rank, f"loading ASR model after generation: {args.asr_model_name}")
    with fp32_precision():
        asr_model = asr_cls.from_pretrained(model_name=args.asr_model_name)
        asr_model = asr_model.to(_metric_device()).eval()

    return asr_model


def _asr_transcribe_audio_batch(asr_model, audio: torch.Tensor, audio_lens: torch.Tensor, batch_size: int):
    audio_list = [a[: int(alen.item())].detach().cpu() for a, alen in zip(audio, audio_lens)]
    with fp32_precision(), torch.inference_mode():
        hyps = asr_model.transcribe(audio_list, batch_size=batch_size, verbose=False)

    out = []
    for hyp in hyps:
        if hasattr(hyp, "text"):
            out.append(str(hyp.text))
        else:
            out.append(str(hyp))
    return out



def compute_metrics_after_generation(args, rank: int, world_size: int, metric_items: List[Dict[str, Any]]):
    """
    Compute metrics after generation without the speechlm2 metric wrappers.

    This follows the MagpieTTS/EasyMagpieTTS evaluation style more closely:
      - ASR is loaded directly from args.asr_model_name and used for transcription.
      - CER/WER are computed from ASR hypotheses with word_error_rate_detail.
      - Speaker similarity is computed with the MagpieTTS embedding helper and
        reported as SSIM, especially pred_context_ssim / ssim.
    """
    metric_start = time.time()

    if len(metric_items) == 0:
        return {
            "rank": int(rank),
            "world_size": int(world_size),
            "num_processed": 0,
            "num_metric_items": 0,
            "metric_elapsed_sec": 0.0,
            "intelligibility": {},
            "speaker_similarity": {},
            "magpie_style_metrics": {},
        }, []

    normalizer = EnglishTextNormalizer()
    normalizer.ignore_patterns = r"$^"
    filewise_rows = []

    # ASR pass, directly using ASRModel as in MagpieTTS evaluation.
    asr_model = _load_asr_model_for_metrics(args, rank=rank)

    for batch_items in chunk_list(metric_items, args.metric_batch_size):
        pred_audio, pred_lens, _, _, output_sr, _ = _load_metric_batch_audio(batch_items, args)

        with fp32_precision():
            pred_16k = resample(pred_audio, output_sr, 16000)
            pred_16k_lens = (pred_lens / output_sr * 16000).to(torch.long)

        asr_hyps = _asr_transcribe_audio_batch(
            asr_model=asr_model,
            audio=pred_16k,
            audio_lens=pred_16k_lens,
            batch_size=len(batch_items),
        )

        for item, hyp in zip(batch_items, asr_hyps):
            ref_norm = normalizer(str(item["reference_text"])).strip()
            hyp_norm = normalizer(str(hyp)).strip()

            detailed_cer = _safe_word_error_detail(hyp_norm, ref_norm, use_cer=True)
            detailed_wer = _safe_word_error_detail(hyp_norm, ref_norm, use_cer=False)
            cer = _safe_detail_value(detailed_cer)
            wer = _safe_detail_value(detailed_wer)

            row = dict(item)
            row["asr_hyp"] = hyp
            row["pred_text"] = hyp_norm
            row["gt_text"] = ref_norm
            row["detailed_cer"] = detailed_cer
            row["detailed_wer"] = detailed_wer
            row["cer"] = cer
            row["wer"] = wer
            row["ssim"] = _nan()
            row["gt_audio_text"] = None
            row["utmosv2"] = _nan()
            row["eou_type"] = None
            row["eou_trailing_duration"] = _nan()
            row["eou_trail_rms_ratio"] = _nan()
            row["pred_gt_ssim"] = _nan()
            row["pred_context_ssim"] = _nan()
            row["gt_context_ssim"] = _nan()
            row["pred_gt_ssim_alternate"] = _nan()
            row["pred_context_ssim_alternate"] = _nan()
            row["gt_context_ssim_alternate"] = _nan()
            row["total_gen_audio_seconds"] = float(row.get("pred_audio_seconds", 0.0) or 0.0)
            filewise_rows.append(row)

    del asr_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Speaker similarity pass. This is the standardized "SSIM" used by the
    # MagpieTTS evaluation scripts: pred_context_ssim is the main speaker
    # similarity against the conditioning/context audio.
    _compute_speaker_similarity_rows(args, filewise_rows)
    for row in filewise_rows:
        row["ssim"] = row.get("pred_context_ssim", _nan())

    _compute_utmos_rows(args, filewise_rows, rank=rank)
    _compute_eou_rows(args, filewise_rows, rank=rank)

    magpie_style_metrics = compute_magpie_style_global_metrics(filewise_rows)

    cer_wer = {
        "cer": magpie_style_metrics.get("cer_cumulative"),
        "wer": magpie_style_metrics.get("wer_cumulative"),
        "cer_dataset": magpie_style_metrics.get("cer_cumulative"),
        "wer_dataset": magpie_style_metrics.get("wer_cumulative"),
        "cer_filewise_avg": magpie_style_metrics.get("cer_filewise_avg"),
        "wer_filewise_avg": magpie_style_metrics.get("wer_filewise_avg"),
    }

    speaker_similarity = {
        "ssim": magpie_style_metrics.get("ssim_pred_context_avg"),
        "ssim_dataset": magpie_style_metrics.get("ssim_pred_context_avg"),
        "ssim_pred_gt_avg": magpie_style_metrics.get("ssim_pred_gt_avg"),
        "ssim_pred_context_avg": magpie_style_metrics.get("ssim_pred_context_avg"),
        "ssim_gt_context_avg": magpie_style_metrics.get("ssim_gt_context_avg"),
        "ssim_pred_gt_avg_alternate": magpie_style_metrics.get("ssim_pred_gt_avg_alternate"),
        "ssim_pred_context_avg_alternate": magpie_style_metrics.get("ssim_pred_context_avg_alternate"),
        "ssim_gt_context_avg_alternate": magpie_style_metrics.get("ssim_gt_context_avg_alternate"),
    }

    metric_elapsed = time.time() - metric_start

    rank_metrics = {
        "rank": int(rank),
        "world_size": int(world_size),
        "num_processed": len({(x["run_id"], x["dataset_index"]) for x in metric_items}),
        "num_metric_items": int(len(metric_items)),
        "metric_elapsed_sec": float(metric_elapsed),
        "intelligibility": cer_wer,
        "speaker_similarity": speaker_similarity,
        "magpie_style_metrics": magpie_style_metrics,
    }

    return rank_metrics, filewise_rows


# -----------------------------
# Merge helpers
# -----------------------------


def compute_and_save_rank_metrics_file(args, rank_metrics: Dict[str, Any], rank: int):
    rank_path = os.path.join(args.out_dir, f"metrics_rank{rank:04d}.json")
    write_json(rank_path, rank_metrics)
    return rank_metrics


def merge_metrics_on_rank0(args, rank, world_size):
    if rank != 0:
        return None

    rank_metric_files = [os.path.join(args.out_dir, f"metrics_rank{r:04d}.json") for r in range(world_size)]

    rank_metrics = []
    for path in rank_metric_files:
        if not os.path.exists(path):
            logging.warning(f"Missing rank metric file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            rank_metrics.append(json.load(f))

    total_n = sum(int(m.get("num_metric_items", m.get("num_processed", 0))) for m in rank_metrics)

    def weighted_average(section: str):
        keys = set()
        for m in rank_metrics:
            keys.update(m.get(section, {}).keys())

        out = {}
        for k in sorted(keys):
            numerator = 0.0
            denominator = 0

            for m in rank_metrics:
                n = int(m.get("num_metric_items", m.get("num_processed", 0)))
                if n <= 0:
                    continue

                value = m.get(section, {}).get(k, None)
                if value is None or isinstance(value, str):
                    continue

                try:
                    value = float(value)
                except Exception:
                    continue

                numerator += value * n
                denominator += n

            if denominator > 0:
                out[k] = numerator / denominator

        return out

    final_metrics = {
        "world_size": int(world_size),
        "num_metric_items": int(total_n),
        "aggregation": "sum(rank_metric * rank_num_metric_items) / total_num_metric_items",
        "intelligibility": weighted_average("intelligibility"),
        "speaker_similarity": weighted_average("speaker_similarity"),
        "ranks": rank_metrics,
    }

    final_json_path = os.path.join(args.out_dir, "metrics_final.json")
    final_txt_path = os.path.join(args.out_dir, "metrics_final.txt")

    write_json(final_json_path, final_metrics)

    final_text = format_final_metric_text(final_metrics)
    write_text_atomic(final_txt_path, final_text)

    print("\n--- Final Evaluation Metrics ---", flush=True)
    print(final_text, flush=True)

    logging.info(f"Final metrics JSON saved to: {final_json_path}")
    logging.info(f"Final metrics TXT saved to: {final_txt_path}")
    logging.info(json.dumps(final_metrics, indent=2, sort_keys=True))

    return final_metrics


def _cer_sort_value(row: Dict[str, Any]) -> float:
    """Return finite CER for sorting; missing/non-finite values go last."""
    value = row.get("cer", None)
    if value is None:
        return float("-inf")
    try:
        value = float(value)
    except Exception:
        return float("-inf")
    if not math.isfinite(value):
        return float("-inf")
    return value


def merge_filewise_metrics_on_rank0(args, rank: int, world_size: int):
    """Merge per-turn rank metric rows and write global CER-sorted outputs.

    Writes:
      - filewise_metrics_turns_sorted_by_cer.jsonl/csv:
          one row per turn, merged across ranks, sorted by turn CER.
      - filewise_metrics_global_sorted_by_cer.jsonl/csv:
          compatibility alias for the same turn-level global output.
      - filewise_metrics_sorted_by_cer.jsonl/csv:
          one row per original sample, with turn metric lists, sorted by
          sample-average CER.
    """
    if rank != 0 or not args.save_filewise_metrics:
        return []

    turn_rows = []

    for r in range(world_size):
        path = os.path.join(args.out_dir, f"filewise_metrics_rank{r:04d}.jsonl")
        if not os.path.exists(path):
            logging.warning(f"Missing filewise metrics file: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    turn_rows.append(json.loads(line))

    # Deduplicate DistributedSampler padding repeats, but preserve --num_eval_runs.
    deduped_turns = {}
    for row in turn_rows:
        run_id = int(row.get("run_id", 0))
        idx = int(row.get("dataset_index", -1))
        turn_id = int(row.get("turn_id", 0))
        key = (run_id, idx, turn_id)
        if key not in deduped_turns:
            deduped_turns[key] = row

    turn_rows = list(deduped_turns.values())

    # Global turn-level output sorted by CER descending.
    turn_rows_sorted = sorted(
        turn_rows,
        key=lambda x: (
            x.get("cer") is not None,
            _cer_sort_value(x),
        ),
        reverse=True,
    )

    turn_jsonl_path = os.path.join(args.out_dir, "filewise_metrics_turns_sorted_by_cer.jsonl")
    turn_csv_path = os.path.join(args.out_dir, "filewise_metrics_turns_sorted_by_cer.csv")
    turn_json_path = os.path.join(args.out_dir, "filewise_metrics_turns_sorted_by_cer.json")
    write_jsonl(turn_jsonl_path, turn_rows_sorted)
    write_json(turn_json_path, {"filewise_metrics": turn_rows_sorted})
    write_turnwise_csv(turn_csv_path, turn_rows_sorted)

    # Compatibility alias using the "global" name.
    global_jsonl_path = os.path.join(args.out_dir, "filewise_metrics_global_sorted_by_cer.jsonl")
    global_csv_path = os.path.join(args.out_dir, "filewise_metrics_global_sorted_by_cer.csv")
    write_jsonl(global_jsonl_path, turn_rows_sorted)
    write_turnwise_csv(global_csv_path, turn_rows_sorted)

    turn_global_metrics = compute_magpie_style_global_metrics(turn_rows_sorted)
    turn_global_metrics_path = os.path.join(args.out_dir, "metrics_final_turn_global.json")
    write_json(
        turn_global_metrics_path,
        {
            "aggregation": "magpie_style_global_metrics_over_turn_rows",
            **turn_global_metrics,
        },
    )

    logging.info(f"Saved global turn-level filewise metrics JSONL to: {turn_jsonl_path}")
    logging.info(f"Saved global turn-level filewise metrics JSON to: {turn_json_path}")
    logging.info(f"Saved global turn-level filewise metrics CSV to: {turn_csv_path}")
    logging.info(f"Saved global filewise compatibility JSONL to: {global_jsonl_path}")
    logging.info(f"Saved global filewise compatibility CSV to: {global_csv_path}")
    logging.info(f"Saved turn global metrics JSON to: {turn_global_metrics_path}")

    # Group turn rows into one row per original file/sample.
    grouped = {}
    for row in turn_rows:
        run_id = int(row.get("run_id", 0))
        idx = int(row.get("dataset_index", -1))
        key = (run_id, idx)

        if key not in grouped:
            grouped[key] = {
                "run_id": run_id,
                "dataset_index": idx,
                "rank": int(row.get("rank", -1)),
                "target_audio_path": row.get("target_audio_path", ""),
                "context_audio_path": row.get("context_audio_path", ""),
                "turn_rows": [],
            }

        grouped[key]["turn_rows"].append(row)

    def avg(vals):
        finite_vals = []
        for x in vals:
            if x is None:
                continue
            try:
                x = float(x)
            except Exception:
                continue
            if math.isfinite(x):
                finite_vals.append(x)
        return None if not finite_vals else sum(finite_vals) / len(finite_vals)

    sample_rows = []
    for _, group in grouped.items():
        turns = sorted(group["turn_rows"], key=lambda x: int(x.get("turn_id", 0)))

        cer_turns = [r.get("cer") for r in turns]
        wer_turns = [r.get("wer") for r in turns]
        ssim_turns = [r.get("ssim") for r in turns]

        pred_gt_ssim_turns = [r.get("pred_gt_ssim") for r in turns]
        pred_context_ssim_turns = [r.get("pred_context_ssim") for r in turns]
        gt_context_ssim_turns = [r.get("gt_context_ssim") for r in turns]
        pred_gt_ssim_alternate_turns = [r.get("pred_gt_ssim_alternate") for r in turns]
        pred_context_ssim_alternate_turns = [r.get("pred_context_ssim_alternate") for r in turns]
        gt_context_ssim_alternate_turns = [r.get("gt_context_ssim_alternate") for r in turns]
        utmosv2_turns = [r.get("utmosv2") for r in turns]
        eou_type_turns = [r.get("eou_type") for r in turns]
        eou_trailing_duration_turns = [r.get("eou_trailing_duration") for r in turns]
        eou_trail_rms_ratio_turns = [r.get("eou_trail_rms_ratio") for r in turns]

        sample_row = {
            "run_id": group["run_id"],
            "dataset_index": group["dataset_index"],
            "rank": group["rank"],
            "num_turns": len(turns),
            "turn_ids": [int(r.get("turn_id", 0)) for r in turns],
            "target_audio_path": group["target_audio_path"],
            "context_audio_path": group["context_audio_path"],
            "pred_audio_paths": [r.get("pred_audio_path", "") for r in turns],
            "predicted_codes_paths": [r.get("predicted_codes_path") for r in turns],
            "sample_pred_audio_path": turns[0].get("sample_pred_audio_path", turns[0].get("pred_audio_path", "")),
            "sample_predicted_codes_path": turns[0].get(
                "sample_predicted_codes_path",
                turns[0].get("predicted_codes_path"),
            ),
            "pred_audio_seconds_turns": [r.get("pred_audio_seconds") for r in turns],
            "reference_text": [r.get("reference_text", "") for r in turns],
            "asr_hyp": [r.get("asr_hyp", "") for r in turns],
            "gt_text": [r.get("gt_text", "") for r in turns],
            "pred_text": [r.get("pred_text", "") for r in turns],
            "cer_turns": cer_turns,
            "wer_turns": wer_turns,
            "ssim_turns": ssim_turns,
            "pred_gt_ssim_turns": pred_gt_ssim_turns,
            "pred_context_ssim_turns": pred_context_ssim_turns,
            "gt_context_ssim_turns": gt_context_ssim_turns,
            "pred_gt_ssim_alternate_turns": pred_gt_ssim_alternate_turns,
            "pred_context_ssim_alternate_turns": pred_context_ssim_alternate_turns,
            "gt_context_ssim_alternate_turns": gt_context_ssim_alternate_turns,
            "utmosv2_turns": utmosv2_turns,
            "eou_type_turns": eou_type_turns,
            "eou_trailing_duration_turns": eou_trailing_duration_turns,
            "eou_trail_rms_ratio_turns": eou_trail_rms_ratio_turns,
            "cer": avg(cer_turns),
            "wer": avg(wer_turns),
            "ssim": avg(ssim_turns),
            "pred_gt_ssim": avg(pred_gt_ssim_turns),
            "pred_context_ssim": avg(pred_context_ssim_turns),
            "gt_context_ssim": avg(gt_context_ssim_turns),
            "pred_gt_ssim_alternate": avg(pred_gt_ssim_alternate_turns),
            "pred_context_ssim_alternate": avg(pred_context_ssim_alternate_turns),
            "gt_context_ssim_alternate": avg(gt_context_ssim_alternate_turns),
            "utmosv2": avg(utmosv2_turns),
            "eou_error": None if not eou_type_turns or eou_type_turns[0] is None else float(
                sum(1 for x in eou_type_turns if str(x).lower() != "good") / len(eou_type_turns)
            ),
            "total_gen_audio_seconds": sum(float(r.get("total_gen_audio_seconds", r.get("pred_audio_seconds", 0.0)) or 0.0) for r in turns),
        }

        sample_rows.append(sample_row)

    # Sample-level output sorted by average CER descending.
    sample_rows.sort(
        key=lambda x: (
            x.get("cer") is not None,
            _cer_sort_value(x),
        ),
        reverse=True,
    )

    jsonl_path = os.path.join(args.out_dir, "filewise_metrics_sorted_by_cer.jsonl")
    json_path = os.path.join(args.out_dir, "filewise_metrics_sorted_by_cer.json")
    csv_path = os.path.join(args.out_dir, "filewise_metrics_sorted_by_cer.csv")

    write_jsonl(jsonl_path, sample_rows)
    write_json(json_path, {"filewise_metrics": sample_rows})
    write_filewise_csv(csv_path, sample_rows)

    logging.info(f"Saved sample-level filewise metrics JSONL to: {jsonl_path}")
    logging.info(f"Saved sample-level filewise metrics JSON to: {json_path}")
    logging.info(f"Saved sample-level filewise metrics CSV to: {csv_path}")

    topk = min(int(args.filewise_metrics_topk_log), len(sample_rows))
    if topk > 0:
        logging.info(f"Top {topk} worst CER samples:")
        for row in sample_rows[:topk]:
            logging.info(
                "run_id=%s dataset_index=%s num_turns=%s cer=%s wer=%s ssim=%s path=%s"
                % (
                    row.get("run_id"),
                    row.get("dataset_index"),
                    row.get("num_turns"),
                    row.get("cer"),
                    row.get("wer"),
                    row.get("ssim"),
                    row.get("target_audio_path"),
                )
            )

    topk_turns = min(int(args.filewise_metrics_topk_log), len(turn_rows_sorted))
    if topk_turns > 0:
        logging.info(f"Top {topk_turns} worst CER turns:")
        for row in turn_rows_sorted[:topk_turns]:
            logging.info(
                "run_id=%s dataset_index=%s turn_id=%s cer=%s wer=%s ssim=%s path=%s text=%s"
                % (
                    row.get("run_id"),
                    row.get("dataset_index"),
                    row.get("turn_id"),
                    row.get("cer"),
                    row.get("wer"),
                    row.get("ssim"),
                    row.get("pred_audio_path"),
                    row.get("reference_text"),
                )
            )

    return sample_rows

def compute_frechet_codec_distance_from_sample_rows(args, rows: List[Dict[str, Any]]):
    """Compute FCD in the same spirit as MagpieTTS: GT audio vs predicted codec codes."""
    if args.disable_fcd:
        return _nan()
    if FrechetCodecDistance is None:
        logging.warning("FrechetCodecDistance is unavailable; setting frechet_codec_distance to NaN.")
        return _nan()

    gt_paths = []
    code_paths = []
    seen = set()

    for row in rows:
        key = (int(row.get("run_id", 0)), int(row.get("dataset_index", -1)))
        if key in seen:
            continue
        seen.add(key)

        gt_path = _resolve_audio_path(row.get("target_audio_path"), args.audio_dir)
        code_path = row.get("sample_predicted_codes_path") or row.get("predicted_codes_path")
        if gt_path and code_path and os.path.exists(gt_path) and os.path.exists(code_path):
            gt_paths.append(gt_path)
            code_paths.append(code_path)

    if not gt_paths:
        logging.warning("No valid GT-audio/predicted-code pairs found for FCD; setting FCD to NaN.")
        return _nan()

    device = _metric_device()
    try:
        fcd_metric = FrechetCodecDistance(codec_name=args.codec_model_path).to(device)
        for gt_path, code_path in zip(gt_paths, code_paths):
            fcd_metric.update_from_audio_file(gt_path, True)
            predicted_codes = torch.load(code_path, map_location="cpu").unsqueeze(0).to(device)
            predicted_codes_lens = torch.tensor([predicted_codes.size(-1)], dtype=torch.int, device=device)
            fcd_metric.update(predicted_codes, predicted_codes_lens, False)

        fcd = fcd_metric.compute().detach().cpu().item()
        fcd_metric.reset()
        return float(fcd)
    except Exception as e:
        logging.warning(f"Frechet Codec Distance computation failed: {repr(e)}")
        return _nan()


def compute_aggregates_from_filewise_rows(rows: List[Dict[str, Any]]):
    """Aggregate over sample-level rows using the MagpieTTS evaluation metric set."""
    if len(rows) == 0:
        out = compute_magpie_style_global_metrics([])
        out["cer"] = None
        out["wer"] = None
        out["ssim"] = None
        out["num_samples"] = 0
        return out

    def avg_key(key):
        return finite_avg([r.get(key) for r in rows])

    out = {
        "cer": avg_key("cer"),
        "wer": avg_key("wer"),
        "ssim": avg_key("ssim"),
        "num_samples": len(rows),
        "cer_filewise_avg": avg_key("cer"),
        "wer_filewise_avg": avg_key("wer"),
        "ssim_pred_gt_avg": avg_key("pred_gt_ssim"),
        "ssim_pred_context_avg": avg_key("pred_context_ssim"),
        "ssim_gt_context_avg": avg_key("gt_context_ssim"),
        "ssim_pred_gt_avg_alternate": avg_key("pred_gt_ssim_alternate"),
        "ssim_pred_context_avg_alternate": avg_key("pred_context_ssim_alternate"),
        "ssim_gt_context_avg_alternate": avg_key("gt_context_ssim_alternate"),
        "utmosv2_avg": avg_key("utmosv2"),
        "total_gen_audio_seconds": sum(float(r.get("total_gen_audio_seconds", 0.0) or 0.0) for r in rows),
        "frechet_codec_distance": _nan(),
    }

    # Sample rows contain lists, so cumulative CER/WER are computed by flattening
    # the normalized turn text lists.
    pred_texts = []
    gt_texts = []
    for row in rows:
        preds = row.get("pred_text", row.get("asr_hyp", []))
        refs = row.get("gt_text", row.get("reference_text", []))
        if not isinstance(preds, list):
            preds = [preds]
        if not isinstance(refs, list):
            refs = [refs]
        for pred, ref in zip(preds, refs):
            ref = "" if ref is None else str(ref).strip()
            pred = "" if pred is None else str(pred).strip()
            if ref:
                pred_texts.append(pred)
                gt_texts.append(ref)

    if pred_texts and gt_texts:
        try:
            out["cer_cumulative"] = float(word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=True)[0])
        except Exception:
            out["cer_cumulative"] = None
        try:
            out["wer_cumulative"] = float(word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=False)[0])
        except Exception:
            out["wer_cumulative"] = None
    else:
        out["cer_cumulative"] = None
        out["wer_cumulative"] = None

    out["cer_gt_audio_cumulative"] = _nan()
    out["wer_gt_audio_cumulative"] = _nan()

    eou_types = []
    for row in rows:
        values = row.get("eou_type_turns", [])
        if isinstance(values, list):
            eou_types.extend(values)
    eou_types = [x for x in eou_types if x is not None]

    if eou_types:
        counts = Counter(eou_types)
        n = len(eou_types)
        if EoUType is not None:
            labels = list(EoUType.error_types())
            good_label = EoUType.GOOD
        else:
            labels = ["cutoff", "silence", "noise"]
            good_label = "good"
        for label in labels:
            out[f"eou_{label}_rate"] = counts.get(label, 0) / n
        out["eou_error_rate"] = 1.0 - counts.get(good_label, 0) / n
    else:
        out["eou_cutoff_rate"] = _nan()
        out["eou_silence_rate"] = _nan()
        out["eou_noise_rate"] = _nan()
        out["eou_error_rate"] = _nan()

    return out

def save_filewise_final_summary(args, filewise_rows: List[Dict[str, Any]]):
    filewise_summary = compute_aggregates_from_filewise_rows(filewise_rows)
    filewise_summary["frechet_codec_distance"] = compute_frechet_codec_distance_from_sample_rows(args, filewise_rows)
    save_easymagpie_style_eval_outputs(args, filewise_rows, filewise_summary)

    obj = {
        "aggregation": "mean_over_sample_metrics_each_sample_contains_turn_metric_lists",
        **filewise_summary,
    }

    path = os.path.join(args.out_dir, "metrics_final_filewise_average.json")
    write_json(path, obj)

    sample_metrics_final_path = os.path.join(args.out_dir, "metrics_final_sample_average.json")
    write_json(sample_metrics_final_path, obj)

    final_txt_path = os.path.join(args.out_dir, "metrics_final.txt")
    final_text = format_filewise_final_metric_text(filewise_summary)
    write_text_atomic(final_txt_path, final_text)

    print("\n--- Final Sample-Averaged Evaluation Metrics ---", flush=True)
    print(final_text, flush=True)

    logging.info(f"Filewise averaged final metrics saved to: {path}")
    logging.info(f"Sample averaged metrics_final JSON saved to: {sample_metrics_final_path}")
    logging.info(f"Final metrics TXT saved to: {final_txt_path}")

    return obj


# -----------------------------
# Args / main
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="EasyMagpieTTS Multi-GPU Inference Evaluation")

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--codec_model_path", type=str, required=True)
    parser.add_argument("--datasets_json_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--phoneme_tokenizer_path", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--inference_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--debug_dtype", action="store_true")
    parser.add_argument("--debug_gpu_assignment", action="store_true")
    parser.add_argument("--use_librosa", action="store_true")

    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--emulate_multiturn",
        action="store_true",
        help=(
            "Group scalar single-turn JSONL rows by speaker into synthetic multiturn samples. "
            "This replaces the older --num_turns behavior."
        ),
    )
    parser.add_argument(
        "--emulate_multiturn_num_turns",
        type=int,
        default=1,
        help="Number of scalar single-turn rows to group when --emulate_multiturn is enabled.",
    )
    parser.add_argument("--max_eval_turns", type=int, default=6)

    parser.add_argument(
        "--inference_mode",
        type=str,
        default="auto",
        choices=["auto", "multiturn_user_audio", "single_turn"],
        help=(
            "auto selects multiturn_user_audio for samples with list text or user_audio_file_path, "
            "and single_turn for classic scalar-text datasets such as LibriTTS. "
            "single_turn does not prefill with user/silence audio and supports batch_size > 1."
        ),
    )

    parser.add_argument("--user_custom_speaker_reference", action="store_true")
    parser.add_argument("--inference_speaker_reference", type=str, default=None)
    parser.add_argument("--language", type=str, default="en")

    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=2.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--max_tts_steps", type=int, default=2000)
    parser.add_argument("--normalize_volume", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True)

    parser.add_argument(
        "--save_filewise_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save filewise metrics. Enabled by default. Use --no-save_filewise_metrics to disable.",
    )
    parser.add_argument(
        "--filewise_metrics_topk_log",
        type=int,
        default=20,
        help="Number of worst CER samples to print on rank 0.",
    )
    parser.add_argument(
        "--num_eval_runs",
        type=int,
        default=1,
        help="Repeat the full eval set N times. Repetitions are preserved in final filewise average.",
    )
    parser.add_argument(
        "--sort_by_text_token_count",
        action="store_true",
        help="Sort eval samples by total text token count before distributed sharding for better load balancing.",
    )
    parser.add_argument(
        "--metric_batch_size",
        type=int,
        default=8,
        help="Batch size used for post-generation ASR/SSIM metric computation.",
    )
    parser.add_argument(
        "--max_metric_audio_sec",
        type=float,
        default=120.0,
        help="Clamp generated audio length used for ASR/SSIM metrics to avoid metric OOM/hangs.",
    )
    parser.add_argument(
        "--asr_model_name",
        type=str,
        default="nvidia/parakeet-tdt-1.1b",
        help="Pretrained ASR model used for CER/WER, matching the EasyMagpie/MagpieTTS eval default.",
    )
    parser.add_argument(
        "--sv_model_type",
        type=str,
        default="titanet",
        choices=["titanet", "wavlm"],
        help="Speaker verification model type for MagpieTTS-style SSIM metrics.",
    )
    parser.add_argument(
        "--disable_speaker_metrics",
        action="store_true",
        help="Disable pred/GT/context speaker similarity metrics.",
    )
    parser.add_argument(
        "--disable_utmosv2",
        action="store_true",
        help="Disable UTMOSv2. By default UTMOSv2 is computed when the dependency is available.",
    )
    parser.add_argument(
        "--disable_eou",
        action="store_true",
        help="Disable end-of-utterance classification metrics.",
    )
    parser.add_argument(
        "--disable_fcd",
        action="store_true",
        help="Disable Frechet Codec Distance. By default FCD is computed from saved predicted codec codes.",
    )
    parser.add_argument(
        "--eou_model_name",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face model id or local path for the EOU classifier.",
    )
    parser.add_argument(
        "--eou_batch_size",
        type=int,
        default=32,
        help="Batch size for EOU classification.",
    )

    parser.add_argument(
        "--save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save EasyMagpie/MagpieTTS-style violin plots. Enabled by default.",
    )
    parser.add_argument(
        "--violin_plot_metrics",
        type=str,
        nargs="*",
        default=list(DEFAULT_VIOLIN_METRICS),
        help="Metrics to include in violin plots.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(get_audio_out_dir(args), exist_ok=True)
    os.makedirs(get_generated_turn_audio_dir(args), exist_ok=True)
    os.makedirs(get_context_metric_audio_dir(args), exist_ok=True)
    os.makedirs(get_predicted_codes_dir(args), exist_ok=True)

    distributed, rank, local_rank, world_size, device_index = setup_distributed()

    if args.inference_mode == "multiturn_user_audio" and args.batch_size != 1:
        raise RuntimeError(
            "--inference_mode multiturn_user_audio requires --batch_size=1 per process. "
            "Use multiple GPUs/processes for parallelism instead of increasing batch_size."
        )

    if args.num_eval_runs <= 0:
        raise RuntimeError("--num_eval_runs must be >= 1.")

    if args.emulate_multiturn and args.emulate_multiturn_num_turns <= 1:
        raise RuntimeError("--emulate_multiturn_num_turns must be > 1 when --emulate_multiturn is enabled.")

    target_device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() and device_index >= 0 else "cpu")
    target_dtype = getattr(torch, args.inference_dtype)
    torch.set_default_dtype(target_dtype)

    hostname = socket.gethostname()
    cuda_name = torch.cuda.get_device_name(target_device) if torch.cuda.is_available() and device_index >= 0 else "cpu"

    all_rank_print(
        rank,
        f"host={hostname} local_rank={local_rank} world_size={world_size} "
        f"device={target_device} device_name={cuda_name}",
    )

    model = build_model_and_codec(args, target_device, target_dtype)
    codec_sil_codes = model.codec_sil_codes

    if args.debug_dtype:
        handles, stats, examples = attach_dtype_counter(model)
    else:
        handles = stats = examples = None

    emulate_multiturn_num_turns = args.emulate_multiturn_num_turns if args.emulate_multiturn else 1
    full_eval_dataset = EvalJSONLDataset(
        args.datasets_json_path,
        emulate_multiturn_num_turns=emulate_multiturn_num_turns,
    )
    # debug
    # full_eval_dataset.samples = full_eval_dataset.samples[:7]

    if args.sort_by_text_token_count:
        full_eval_dataset = SortedByTextTokenCountDataset(
            full_eval_dataset,
            model=model,
            max_eval_turns=args.max_eval_turns,
            descending=True,
        )

    collate_fn = partial(
        collate_and_tokenize_custom,
        model=model,
        sample_rate=model.sample_rate,
        root_path=args.audio_dir,
        normalize_audio_volume=args.normalize_volume,
        use_librosa=args.use_librosa,
        max_eval_turns=args.max_eval_turns,
        inference_mode=args.inference_mode,
    )

    speaker_wav = load_speaker_wav_if_needed(args, model, target_dtype)

    generation_start = time.time()
    all_metric_items = []
    total_batches = 0
    total_generated_samples = 0

    for run_id in range(args.num_eval_runs):
        if distributed:
            sampler = DistributedSampler(
                full_eval_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            sampler.set_epoch(run_id)
        else:
            sampler = SequentialSampler(full_eval_dataset)

        if args.debug_gpu_assignment:
            try:
                assigned_indices = list(iter(sampler))
                assigned_dataset_indices = [
                    int(full_eval_dataset[i].get("__dataset_index__", -1)) for i in assigned_indices
                ]
                all_rank_print(
                    rank,
                    f"run_id={run_id} assigned {len(assigned_dataset_indices)} / {len(full_eval_dataset)} "
                    f"samples to gpu={local_rank}: dataset_indices={assigned_dataset_indices}",
                )
            except Exception as e:
                all_rank_print(rank, f"Could not print assigned indices: {repr(e)}")

        dataloader = DataLoader(
            dataset=full_eval_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        for batch_id, inputs in enumerate(dataloader):
            total_batches += 1
            batch_indices = inputs.get("dataset_indices", [])
            total_generated_samples += len(batch_indices)

            if args.debug_gpu_assignment:
                all_rank_print(
                    rank,
                    f"run_id={run_id} gpu={local_rank} batch_id={batch_id} "
                    f"dataset_indices={batch_indices} text_token_counts={inputs.get('text_token_counts', [])} "
                    f"target_paths={inputs.get('target_audio_paths', [])}",
                )

            inputs = prepare_inputs_for_device(inputs, model, args, target_dtype, speaker_wav=speaker_wav)

            finalize_output, multiturn_turn_frame_ranges, multiturn_decode_start_frame, generated_codes = run_generation(
                model=model,
                inputs=inputs,
                args=args,
                codec_sil_codes=codec_sil_codes,
            )

            metric_items = save_generation_outputs_and_build_metric_items(
                model=model,
                inputs=inputs,
                finalize_output=finalize_output,
                multiturn_turn_frame_ranges=multiturn_turn_frame_ranges,
                multiturn_decode_start_frame=multiturn_decode_start_frame,
                generated_codes=generated_codes,
                args=args,
                rank=rank,
                run_id=run_id,
            )
            all_metric_items.extend(metric_items)

            if args.debug_dtype and batch_id == 0 and run_id == 0:
                report_dtype_stats(handles, stats, examples, rank=rank)

    generation_elapsed = time.time() - generation_start

    # Save pre-metric manifest for debugging and restartability.
    metric_manifest_path = os.path.join(args.out_dir, f"metric_items_rank{rank:04d}.jsonl")
    write_jsonl(metric_manifest_path, all_metric_items)

    # Free TTS/codec model memory before loading ASR and speaker encoder metrics.
    del model
    if speaker_wav is not None:
        del speaker_wav
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_rank_print(
        rank,
        f"generation done: batches={total_batches} generated_samples_with_sampler_padding={total_generated_samples} "
        f"metric_items={len(all_metric_items)} elapsed_sec={generation_elapsed:.2f}. "
        "Loading ASR/SSIM metrics now.",
    )

    rank_metrics, rank_filewise_rows = compute_metrics_after_generation(
        args=args,
        rank=rank,
        world_size=world_size,
        metric_items=all_metric_items,
    )
    rank_metrics["generation_elapsed_sec"] = float(generation_elapsed)
    rank_metrics["num_generated_samples_with_sampler_padding"] = int(total_generated_samples)

    rank_metrics = compute_and_save_rank_metrics_file(args, rank_metrics, rank)
    all_rank_print(rank, f"saved rank metrics: {json.dumps(rank_metrics, sort_keys=True)}")

    if args.save_filewise_metrics:
        rank_filewise_rows.sort(
            key=lambda x: (
                x.get("cer") is not None,
                float(x.get("cer")) if x.get("cer") is not None else -1.0,
            ),
            reverse=True,
        )

        rank_filewise_path = os.path.join(args.out_dir, f"filewise_metrics_rank{rank:04d}.jsonl")
        write_jsonl(rank_filewise_path, rank_filewise_rows)
        all_rank_print(rank, f"saved filewise metrics: {rank_filewise_path}")

    if rank == 0:
        wait_for_rank_metric_files(args, world_size)

    merge_metrics_on_rank0(args, rank, world_size)

    if args.save_filewise_metrics:
        if rank == 0:
            wait_for_rank_filewise_metric_files(args, world_size)

        filewise_rows = merge_filewise_metrics_on_rank0(args, rank, world_size)

        if rank == 0:
            save_filewise_final_summary(args, filewise_rows)

    cleanup_distributed()


if __name__ == "__main__":
    main()
