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
Offline evaluation script for StreamingSTTModel.

Usage::

    python streaming_stt_generate.py \
        pretrained_name=nvidia/streaming-stt-v1 \
        inputs=/data/test.jsonl \
        batch_size=32

    # Simulate streaming (chunk-by-chunk with blanks):
    python streaming_stt_generate.py \
        pretrained_name=nvidia/streaming-stt-v1 \
        inputs=/data/test.jsonl \
        simulate_streaming=true

The model's ``generate()`` method returns ``list[str]`` directly.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Optional

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import GenerationConfig
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.common.data.lhotse.dataloader import pad_extra_duration
from nemo.collections.speechlm2.models import StreamingSTTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


def _load_alignments_from_manifest(manifest_path: str) -> dict[str, list]:
    """Load word-level alignments from a NeMo-style manifest.

    Each line is expected to have ``audio_filepath`` and ``alignments`` keys
    where alignments is a list of {text, start_time, end_time}.
    Returns dict mapping cut_id (audio file stem) → alignment list.
    """
    out: dict[str, list] = {}
    p = Path(manifest_path)
    if not p.exists():
        return out
    try:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                if "alignments" in d:
                    stem = Path(d.get("audio_filepath", "")).stem
                    if stem:
                        out[stem] = d["alignments"]
    except Exception as e:
        logging.warning(f"Failed to load alignments from {manifest_path}: {e}")
    return out


def _gt_emit_frames(alignments: list, frame_length_in_secs: float, num_delay_frames: int = 0) -> list[int]:
    """For each word in the alignment, compute the frame index at which
    the model is expected to emit (word_end_frame + num_delay_frames)."""
    return [int(math.ceil(w["end_time"] / frame_length_in_secs)) + num_delay_frames for w in alignments]


def _boundary_match(gt_frames: list[int], pred_frames: list[int], tolerance: int) -> tuple[int, int, list[int]]:
    """Greedy nearest-match between GT and predicted emit frames.

    Returns (n_gt_recalled, n_pred_matched, signed_offsets).
    Recall = matched / len(gt), Precision = matched / len(pred).
    Each GT is matched to at most one pred (the closest within tolerance).
    signed_offsets is a list of (pred_frame - gt_frame) for each matched
    pair — negative = aux fired EARLY (before GT), positive = aux fired LATE.
    """
    if not gt_frames or not pred_frames:
        return 0, 0, []
    gt_used = [False] * len(gt_frames)
    offsets: list[int] = []
    for p in pred_frames:
        best_i, best_d = -1, 10**9
        for i, g in enumerate(gt_frames):
            if gt_used[i]:
                continue
            d = abs(p - g)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_d <= tolerance:
            gt_used[best_i] = True
            offsets.append(p - gt_frames[best_i])
    return sum(gt_used), len(offsets), offsets


class ToAudio(torch.utils.data.Dataset):
    """Minimal dataset that loads audio from a CutSet."""

    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


@dataclass
class StreamingSTTGenerationConfig:
    """
    A proxy class for GenerationConfig so that we can use OmegaConf with hydra overrides.
    All parameters will be passed to GenerationConfig.
    """

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


@dataclass
class StreamingSTTEvalConfig:
    pretrained_name: str = ""
    inputs: str = ""
    batch_size: int = 64
    num_workers: int = 4
    max_new_tokens: int = 64
    system_prompt: str = "Transcribe the audio into text."
    output_manifest: Optional[str] = "streaming_stt_generations.jsonl"
    verbose: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_normalizer: Optional[str] = "english"  # "english", "basic", or "none"
    use_offline_embs: bool = False
    seed: Optional[int] = None  # Set for deterministic results
    pad_extra_duration: Optional[float] = 0.0
    use_state_machine_inference: bool = (
        False  # recommended turned off for chunk_size > 0, no effect for chunk_size <= 0
    )
    dynamic_min_chunk_size: int = 0  # dynamic chunking: min frames before allowing generation
    dynamic_max_chunk_size: Optional[int] = None  # dynamic chunking: max frames before forcing generation
    # When True, the aux chunk-boundary classifier (built during training with
    # use_chunk_classifier=True) drives the boundary decision at inference.
    # When False (default), the LM head's signal drives the decision.
    use_chunk_classifier_at_inference: bool = False
    # Probability threshold for the boundary decision.
    #   - When use_chunk_classifier_at_inference=True: threshold on the aux
    #     head's sigmoid output. None → 0.5 default.
    #   - When False: threshold on p(user_footer_first_id) from the LM head.
    #     None → fall back to argmax (legacy behavior).
    emit_threshold: Optional[float] = None
    # Defer the actual FOOTER transition by K LISTENING frames after the
    # boundary classifier decides to emit. The stream keeps listening for K
    # more frames, then transitions. Compensates for an early-firing aux
    # head (peak shifted left of GT) and gives GENERATING K more frames of
    # audio context in the KV cache. K=0 (default) = emit immediately on
    # decision (legacy behavior). +K frames adds K * frame_length_in_secs
    # of latency per emit.
    emit_delay_frames: int = 0
    # num_delay_frames used during training. Boundary GT emit position is
    # computed as ceil(end_time / frame_length) + num_delay_frames so it
    # matches where the dataset supervised the model to fire. Mismatched
    # values produce a constant offset bias in the histogram (each +1 in
    # this knob shifts all matched offsets by −1 in the same direction).
    boundary_num_delay_frames: int = 0
    # DIAGNOSTIC: when True, the model never emits during streaming — it
    # stays in LISTENING for the entire audio so the per-frame log captures
    # a continuous aux_p_emit trace across all audio frames (no gaps from
    # GENERATING phases). Transcript output is unusable (single force-emit
    # at audio end). Combine with debug_log_audio_frames=true.
    disable_emit_for_debug: bool = False
    # When True, dump per-LISTENING-frame diagnostics (LM head top-5, prob of
    # user_footer_first / blank, aux head sigmoid, decision taken) to a
    # sibling JSONL alongside output_manifest. Slows inference; use on
    # small eval sets when debugging boundary-decision behavior.
    debug_log_audio_frames: bool = False
    generation_config: StreamingSTTGenerationConfig = field(default_factory=StreamingSTTGenerationConfig)


@hydra_runner(config_name="StreamingSTTEvalConfig", schema=StreamingSTTEvalConfig)
def main(cfg: StreamingSTTEvalConfig):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.seed is not None:
        logging.warning(f"Setting random seed to {cfg.seed}, this will slow down the inference")
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        logging.warning("Random seed not set, results will not be deterministic")

    model = StreamingSTTModel.from_pretrained(cfg.pretrained_name)
    model = model.eval().to(getattr(torch, cfg.dtype)).to(cfg.device)

    cuts = guess_parse_cutset(cfg.inputs)
    # Resample to model's expected sample rate if needed.
    sample_cut = next(iter(cuts))
    if sample_cut.sampling_rate != model.sampling_rate:
        logging.info(f"Resampling cuts from {sample_cut.sampling_rate} to {model.sampling_rate} Hz")
        cuts = CutSet.from_cuts(c.resample(model.sampling_rate) for c in cuts)
    cuts = cuts.sort_by_duration()
    cuts = cuts.map(partial(pad_extra_duration, extra_duration=cfg.pad_extra_duration))
    sampler = lhotse.dataset.DynamicCutSampler(cuts, max_cuts=cfg.batch_size)
    num_batches = math.ceil(len(cuts) / cfg.batch_size)
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=sampler,
        num_workers=cfg.num_workers,
        batch_size=None,
    )

    _normalizer_key = cfg.use_normalizer.lower() if isinstance(cfg.use_normalizer, str) else cfg.use_normalizer
    normalizer = {"english": EnglishTextNormalizer(), "basic": BasicTextNormalizer()}.get(_normalizer_key, lambda x: x)

    refs = []
    hyps = []
    input_durations = []
    infer_durations = []

    # Optional per-frame debug log file (one record per LISTENING frame per
    # cut, keyed by cut id). Only opened when debug_log_audio_frames=True.
    debug_log_writer = None
    boundary_alignments: dict[str, list] = {}
    # Corpus-level boundary stats (only populated when frame debug is on and
    # the input manifest carries word alignments).
    # Tolerances (in 80ms frames) for boundary recall/precision. The curve
    # ±0 → ±5 shows how quickly the aux head's "near miss" recall saturates;
    # the gap from ±5 to 1.0 is the genuine-miss tail (no fire anywhere near
    # the GT boundary). Stop at ±5 (=400ms) — beyond that, matches start
    # being coincidental rather than informative about boundary precision.
    BOUNDARY_TOLERANCES = (0, 1, 2, 3, 5)
    BOUNDARY_MAX_TOL = max(BOUNDARY_TOLERANCES)
    boundary_total_gt = 0
    boundary_total_pred = 0
    boundary_total_recalled = {k: 0 for k in BOUNDARY_TOLERANCES}
    boundary_total_matched = {k: 0 for k in BOUNDARY_TOLERANCES}
    # Split by direction: aux fired EARLY (offset < 0) vs LATE (offset > 0).
    boundary_recalled_early = {k: 0 for k in BOUNDARY_TOLERANCES if k > 0}
    boundary_recalled_late = {k: 0 for k in BOUNDARY_TOLERANCES if k > 0}
    boundary_matched_early = {k: 0 for k in BOUNDARY_TOLERANCES if k > 0}
    boundary_matched_late = {k: 0 for k in BOUNDARY_TOLERANCES if k > 0}
    # Full signed-offset histogram for matched pairs (at the max tolerance).
    boundary_offset_hist: dict[int, int] = {}
    # Aux-head confidence diagnostics. These answer "is the model confident
    # at the GT position, or only at the (earlier) fire position?"
    # - aux_p_at_fire: aux sigmoid at the frame where model committed
    # - aux_p_at_gt: aux sigmoid at the GT-supervised frame, IF available.
    #   Only available when model didn't already exit LISTENING before GT
    #   (i.e., fire offset >= 0 — late or exact). This sample is biased
    #   toward "non-early" pairs, but still useful as a sanity check.
    # - aux_p_curve_by_offset: for each GT frame, sample aux_p at offsets
    #   −5..+5. Tells us where the confidence peaks around boundaries.
    boundary_aux_p_at_fire: list[float] = []
    boundary_aux_p_at_gt: list[float] = []
    boundary_aux_p_curve: dict[int, list[float]] = {o: [] for o in range(-5, 6)}
    if cfg.debug_log_audio_frames and cfg.output_manifest is not None:
        manifest_path = Path(cfg.output_manifest)
        debug_log_path = manifest_path.with_name(
            manifest_path.stem.replace("_generations", "") + "_audio_frame_log.jsonl"
        )
        debug_log_writer = SequentialJsonlWriter(str(debug_log_path))
        logging.info(f"Audio frame debug log → {debug_log_path}")
        # Load GT word alignments from the input manifest, if present. Used to
        # compute boundary recall/precision against the aux head's emit
        # decisions during real (auto-regressive) streaming.
        boundary_alignments = _load_alignments_from_manifest(cfg.inputs)
        if boundary_alignments:
            logging.info(
                f"Loaded GT word alignments for {len(boundary_alignments)} cuts — "
                "will compute boundary recall/precision at tolerances "
                f"{BOUNDARY_TOLERANCES} frames."
            )
        else:
            logging.info(
                "No alignments found in manifest — boundary debug metrics will be skipped. "
                "Add 'alignments' field (list of {text,start_time,end_time}) to enable."
            )

    for batch_idx, batch in tqdm(enumerate(dloader), total=num_batches):
        ts = perf_counter()
        cfg.generation_config.max_new_tokens = cfg.max_new_tokens
        generation_config = GenerationConfig(**OmegaConf.to_container(cfg.generation_config))
        batch_debug_logs: Optional[list] = [] if cfg.debug_log_audio_frames else None
        batch_hyps_raw = model.generate(
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            system_prompt=cfg.system_prompt,
            max_new_tokens=cfg.max_new_tokens,
            generation_config=generation_config,
            use_offline_embs=cfg.use_offline_embs,
            use_state_machine_inference=cfg.use_state_machine_inference,
            dynamic_min_chunk_size=cfg.dynamic_min_chunk_size,
            dynamic_max_chunk_size=cfg.dynamic_max_chunk_size,
            use_chunk_classifier_at_inference=cfg.use_chunk_classifier_at_inference,
            emit_threshold=cfg.emit_threshold,
            emit_delay_frames=cfg.emit_delay_frames,
            debug_logs=batch_debug_logs,
            disable_emit_for_debug=cfg.disable_emit_for_debug,
        )
        batch_infer_duration = perf_counter() - ts

        # Write per-frame debug records keyed by cut id, and compute per-cut
        # boundary metrics against GT word alignments (when available).
        if debug_log_writer is not None and batch_debug_logs is not None:
            for cut, frames in zip(batch["cuts"], batch_debug_logs):
                record = {"id": cut.id, "duration": cut.duration, "frames": frames}

                # Boundary recall/precision vs GT word alignments.
                cut_alignments = boundary_alignments.get(cut.id) or boundary_alignments.get(Path(cut.id).stem)
                if cut_alignments:
                    gt_frames = _gt_emit_frames(
                        cut_alignments,
                        model.core_cfg.frame_length_in_secs,
                        num_delay_frames=cfg.boundary_num_delay_frames,
                    )
                    # All decision types that represent an actual model-emit
                    # transition. "emit_pending" is the deferred-decision
                    # intermediate frame and does NOT count as a fire.
                    EMIT_FIRED = {
                        "emit_model",
                        "emit_model_delayed",
                        "emit_model_delayed_at_end",
                        "emit_forced_audio_end",
                        "emit_forced_audio_end_below_min",
                        "emit_forced_max",
                        "emit_forced_chunk_size",
                    }
                    pred_frames = [fr["total_frame_idx"] for fr in frames if fr.get("decision") in EMIT_FIRED]
                    cut_boundary = {
                        "n_gt": len(gt_frames),
                        "n_pred": len(pred_frames),
                    }
                    boundary_total_gt += len(gt_frames)
                    boundary_total_pred += len(pred_frames)

                    # Run matching once at the max tolerance, derive all
                    # per-tolerance counts (recall/precision and the
                    # early/late split) from the resulting signed offsets.
                    _, _, offsets_max = _boundary_match(gt_frames, pred_frames, BOUNDARY_MAX_TOL)
                    cut_boundary["mean_signed_offset"] = sum(offsets_max) / len(offsets_max) if offsets_max else 0.0
                    cut_boundary["n_matched_max"] = len(offsets_max)
                    for o in offsets_max:
                        boundary_offset_hist[o] = boundary_offset_hist.get(o, 0) + 1

                    # --- Aux-head confidence diagnostics ---
                    # Per-cut lookup: total_frame_idx → aux_p_emit (only
                    # populated for LISTENING frames; model exits LISTENING
                    # after emit, so frames past an early fire have no entry).
                    prob_by_idx = {
                        fr["total_frame_idx"]: fr.get("aux_p_emit")
                        for fr in frames
                        if fr.get("aux_p_emit") is not None
                    }
                    # Re-run matching to recover the (fire, gt) pairs themselves
                    # so we can sample confidence at each.
                    gt_used_2 = [False] * len(gt_frames)
                    matched_pairs: list[tuple[int, int]] = []
                    for p_frame in pred_frames:
                        best_i, best_d = -1, 10**9
                        for i, g in enumerate(gt_frames):
                            if gt_used_2[i]:
                                continue
                            d = abs(p_frame - g)
                            if d < best_d:
                                best_d = d
                                best_i = i
                        if best_i >= 0 and best_d <= BOUNDARY_MAX_TOL:
                            gt_used_2[best_i] = True
                            matched_pairs.append((p_frame, gt_frames[best_i]))
                    for fire_frame, gt_frame in matched_pairs:
                        p_fire = prob_by_idx.get(fire_frame)
                        if p_fire is not None:
                            boundary_aux_p_at_fire.append(p_fire)
                        p_gt = prob_by_idx.get(gt_frame)
                        if p_gt is not None:
                            boundary_aux_p_at_gt.append(p_gt)
                    # Confidence curve around each GT: sample aux_p at GT+offset
                    # for offset in [-5, +5]. Some samples will be missing (if
                    # the model exited LISTENING before reaching that frame).
                    for gt_frame in gt_frames:
                        for offset in range(-5, 6):
                            p = prob_by_idx.get(gt_frame + offset)
                            if p is not None:
                                boundary_aux_p_curve[offset].append(p)

                    for k in BOUNDARY_TOLERANCES:
                        n_within = sum(1 for o in offsets_max if abs(o) <= k)
                        cut_boundary[f"recalled_t{k}"] = n_within
                        cut_boundary[f"matched_t{k}"] = n_within
                        boundary_total_recalled[k] += n_within
                        boundary_total_matched[k] += n_within
                        if k > 0:
                            n_early = sum(1 for o in offsets_max if -k <= o < 0)
                            n_late = sum(1 for o in offsets_max if 0 < o <= k)
                            cut_boundary[f"early_t{k}"] = n_early
                            cut_boundary[f"late_t{k}"] = n_late
                            boundary_recalled_early[k] += n_early
                            boundary_matched_early[k] += n_early
                            boundary_recalled_late[k] += n_late
                            boundary_matched_late[k] += n_late
                    record["boundary"] = cut_boundary
                debug_log_writer.write(record)

        batch_duration = sum(c.duration for c in batch["cuts"])
        batch_refs = [normalizer(cut.supervisions[0].text) for cut in batch["cuts"]]
        batch_hyps = [normalizer(h.strip()) for h in batch_hyps_raw]

        if cfg.verbose:
            batch_wer, _, nins, ndel, nsub = word_error_rate_detail(batch_hyps, batch_refs)
            batch_rtfx = batch_duration / batch_infer_duration
            logging.info("--------------------------------")
            logging.info(
                f"Batch {batch_idx}: "
                f"WER={batch_wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}] "
                f"RTFx={batch_rtfx:.1f}"
            )
            for ref, hyp in zip(batch_refs, batch_hyps):
                logging.info(f"\n[REF]\t`{ref}`\n[HYP]\t`{hyp}`\n")
            logging.info("--------------------------------")

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)
        input_durations.append(batch_duration)
        infer_durations.append(batch_infer_duration)

    if debug_log_writer is not None:
        debug_log_writer.close()

    # --- Corpus-level boundary recall / precision (vs GT word alignments) ---
    boundary_summary = None
    if boundary_total_gt > 0 or boundary_total_pred > 0:
        logging.info("--- Aux-head boundary metrics (vs GT word alignments) ---")
        logging.info(f"  GT emit boundaries: {boundary_total_gt}, model emit fires: {boundary_total_pred}")
        boundary_summary = {"gt_total": boundary_total_gt, "pred_total": boundary_total_pred}

        # Exact-frame fires (signed offset 0). Doubles as a sanity anchor for
        # the split below — the +0 bucket is neither early nor late.
        n_exact = boundary_offset_hist.get(0, 0)
        boundary_summary["n_exact"] = n_exact

        for k in BOUNDARY_TOLERANCES:
            recalled = boundary_total_recalled[k]
            matched = boundary_total_matched[k]
            recall = recalled / max(1, boundary_total_gt)
            precision = matched / max(1, boundary_total_pred)
            f1 = 2 * recall * precision / max(1e-9, recall + precision)
            logging.info(
                f"  ±{k} frames: recall={recall:.3f}  precision={precision:.3f}  F1={f1:.3f}  "
                f"(recalled={recalled}/{boundary_total_gt}, matched={matched}/{boundary_total_pred})"
            )
            boundary_summary[f"recall_t{k}"] = recall
            boundary_summary[f"precision_t{k}"] = precision
            boundary_summary[f"f1_t{k}"] = f1
            if k > 0:
                n_early = boundary_recalled_early[k]
                n_late = boundary_recalled_late[k]
                # Recall split (denominator = total GT)
                recall_early = n_early / max(1, boundary_total_gt)
                recall_late = n_late / max(1, boundary_total_gt)
                # Precision split (denominator = total pred fires)
                prec_early = boundary_matched_early[k] / max(1, boundary_total_pred)
                prec_late = boundary_matched_late[k] / max(1, boundary_total_pred)
                logging.info(
                    f"    split: EARLY [-{k},-1] recall={recall_early:.3f} precision={prec_early:.3f} "
                    f"({n_early}), LATE [+1,+{k}] recall={recall_late:.3f} precision={prec_late:.3f} "
                    f"({n_late})"
                )
                boundary_summary[f"recall_early_t{k}"] = recall_early
                boundary_summary[f"recall_late_t{k}"] = recall_late
                boundary_summary[f"precision_early_t{k}"] = prec_early
                boundary_summary[f"precision_late_t{k}"] = prec_late

        # Aux-head confidence at fire vs GT positions (only available when
        # model actually emitted — empty in disable_emit_for_debug mode).
        if boundary_aux_p_at_fire or boundary_aux_p_at_gt:
            mean_p_fire = (
                sum(boundary_aux_p_at_fire) / len(boundary_aux_p_at_fire) if boundary_aux_p_at_fire else float("nan")
            )
            mean_p_gt = sum(boundary_aux_p_at_gt) / len(boundary_aux_p_at_gt) if boundary_aux_p_at_gt else float("nan")
            logging.info(
                f"  Aux sigmoid: mean@FIRE={mean_p_fire:.3f} (n={len(boundary_aux_p_at_fire)})  "
                f"mean@GT={mean_p_gt:.3f} (n={len(boundary_aux_p_at_gt)})  "
                "(GT sample biased toward non-early fires; "
                "disable_emit_for_debug=true gives unbiased trace)"
            )
            boundary_summary["aux_p_at_fire_mean"] = mean_p_fire
            boundary_summary["aux_p_at_gt_mean"] = mean_p_gt

        # Confidence curve around GT — populated whenever the per-frame log
        # has aux_p_emit samples at GT±k positions. This is independent of
        # whether the model actually emitted; in disable_emit_for_debug mode
        # it's the *unbiased* view of where confidence peaks around boundaries.
        if any(len(v) > 0 for v in boundary_aux_p_curve.values()):
            curve_str = "  Aux sigmoid curve around GT (offset:mean(n)): "
            for off in range(-5, 6):
                vals = boundary_aux_p_curve.get(off, [])
                if vals:
                    curve_str += f"{off:+d}:{sum(vals)/len(vals):.3f}({len(vals)})  "
                else:
                    curve_str += f"{off:+d}:-  "
            logging.info(curve_str)
            for off, vals in boundary_aux_p_curve.items():
                if vals:
                    boundary_summary[f"aux_p_curve_t{off:+d}_mean"] = sum(vals) / len(vals)
                    boundary_summary[f"aux_p_curve_t{off:+d}_n"] = len(vals)

        # Mean signed offset (+ = late, − = early) — overall bias indicator.
        if boundary_offset_hist:
            total_off = sum(o * n for o, n in boundary_offset_hist.items())
            n_off = sum(boundary_offset_hist.values())
            mean_off = total_off / n_off
            boundary_summary["mean_signed_offset"] = mean_off
            logging.info(
                f"  Mean signed offset of matched fires: {mean_off:+.2f} frames  "
                f"(over {n_off} matches within ±{BOUNDARY_MAX_TOL})"
            )
            # Histogram of all matched-pair offsets in [-MAX_TOL, +MAX_TOL].
            hist_str = "  Offset histogram (matched fires, pred-gt): "
            for o in range(-BOUNDARY_MAX_TOL, BOUNDARY_MAX_TOL + 1):
                hist_str += f"{o:+d}:{boundary_offset_hist.get(o, 0)}  "
            logging.info(hist_str)

    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    rtfx = sum(input_durations) / sum(infer_durations)
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    logging.info(f"RTFx: {rtfx:.1f}")

    if cfg.output_manifest is not None:
        log_file = Path(cfg.output_manifest).parent / "log.txt"
        with open(log_file, "a") as f:
            f.write(f"======{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}======\n")
            f.write(f"Input: {cfg.inputs}\n")
            f.write(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]\n")
            f.write(f"RTFx: {rtfx:.1f}\n")
            if boundary_summary is not None:
                f.write(f"Aux boundary (GT={boundary_summary['gt_total']}, pred={boundary_summary['pred_total']}): ")
                for k in BOUNDARY_TOLERANCES:
                    f.write(
                        f"±{k}f recall={boundary_summary[f'recall_t{k}']:.3f} "
                        f"precision={boundary_summary[f'precision_t{k}']:.3f} "
                        f"F1={boundary_summary[f'f1_t{k}']:.3f}  "
                    )
                f.write("\n")
            f.write(f"=============================================\n\n")
        with SequentialJsonlWriter(cfg.output_manifest) as writer:
            for cut, ref, hyp in zip(cuts, refs, hyps):
                wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=False)
                writer.write(
                    {
                        "id": cut.id,
                        "duration": cut.duration,
                        "text": ref,
                        "pred_text": hyp,
                        "wer": wer,
                        "ins": nins,
                        "del": ndel,
                        "sub": nsub,
                    }
                )


if __name__ == "__main__":
    main()
