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

    # Long-form: segment each recording into <=30s windows, decode segments in
    # batches of 16, and concatenate transcripts per recording:
    python streaming_stt_generate.py \
        pretrained_name=nvidia/streaming-stt-v1 \
        inputs=/data/longform.jsonl \
        max_segment_duration=30 \
        max_concurrent_segments=16

The model's ``generate()`` method returns ``list[str]`` directly.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Callable, NamedTuple, Optional

import lhotse.dataset
import torch
from kaldialign import edit_distance
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import GenerationConfig

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.asr.parts.utils.sot_speaker_alignment import remove_speaker_tags
from nemo.collections.asr.parts.utils.text_normalizers import build_normalizer
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.common.data.lhotse.dataloader import pad_extra_duration
from nemo.collections.speechlm2.models import StreamingSTTModel
from nemo.collections.speechlm2.parts.metrics import CpWER, CpWERScoringConfig, CpWERSessionResult, resolve_normalizer
from nemo.collections.speechlm2.parts.metrics.cpwer_report import cpwer_metrics_dict, format_cpwer_report
from nemo.core.config import hydra_runner
from nemo.utils import logging


class ReducedRecord(NamedTuple):
    """One recording's results, after segments (if any) are regrouped.

    Replaces the 7-tuple this used to be so that `meta` can ride along: the cut object is dropped
    at construction time, and `cut.custom` is unreachable from anywhere downstream.
    """

    id: str
    duration: float
    ref_raw: str  # cut.supervisions[0].text, verbatim -- tags intact
    hyp_raw: str  # model output, verbatim -- tags intact
    alignments: Optional[list]
    content_scores: Optional[list]
    annotated: Optional[str]
    meta: dict  # {'custom': <the input manifest row>}, or {} when the cut carries none


class ToAudio(torch.utils.data.Dataset):
    """Minimal dataset that loads audio from a CutSet."""

    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


def _oracle_targets(cfg, batch: dict, device) -> dict:
    """Resolve oracle speaker targets for one batch, or refuse if they were asked for and absent.

    `oracle_spk_targets` used to degrade silently: `ToAudio` yields only audio, so `spk_targets`
    was never on a batch, the flag did nothing, and the run still stamped `oracle_spk_targets: true`
    into the manifest. A four-arm eval comparing oracle against the embedded diarizer therefore
    produced two pairs of byte-identical arms -- 0 of 600 hypotheses differing -- which reads as
    "the diarizer is already as good as oracle" rather than as "the targets were never applied".

    Args:
        cfg: the eval config; only ``oracle_spk_targets`` is consulted.
        batch: one dataloader batch.
        device: where to move the targets.

    Returns:
        dict: ``{"spk_targets": Tensor}`` when oracle targets are requested and present, else ``{}``.

    Raises:
        ValueError: when oracle targets are requested but the batch carries none.
    """
    if not cfg.oracle_spk_targets:
        return {}
    targets = batch.get("spk_targets")
    if targets is None:
        raise ValueError(
            "oracle_spk_targets=true, but this batch carries no `spk_targets`. This script's "
            "dataloader (`ToAudio`) loads audio only, so RTTM-derived targets are never collated "
            "-- the flag would otherwise be silently ignored while the embedded diarizer ran and "
            "the manifest still recorded `oracle_spk_targets: true`. Feeding oracle targets needs "
            "a dataloader that emits them (see `StreamingSTTDataset`, which builds them when its "
            "multi-speaker config is enabled). Until then, set oracle_spk_targets=false."
        )
    return {"spk_targets": targets.to(device, non_blocking=True)}


def compute_segment_spans(cut, max_segment_duration: float, method: str = "fixed") -> list[tuple[float, float]]:
    """Compute (offset, duration) spans (seconds) tiling one recording for segmentation.

    Each returned span is at most ``max_segment_duration`` seconds. This is the single
    extension point for segmentation strategy: to add Silero VAD later, add a
    ``method == "silero_vad"`` branch that loads the cut's audio, runs the VAD, and
    returns speech-based spans (merging/splitting so each stays <= max_segment_duration).
    The rest of the pipeline is agnostic to how spans are produced.

    Args:
        cut: a lhotse cut representing a whole recording.
        max_segment_duration: maximum segment length in seconds.
        method: "fixed" = even contiguous windows (no external dependencies).
    """
    if method == "fixed":
        total = cut.duration
        n = max(1, math.ceil(total / max_segment_duration))
        seg_len = total / n  # even split so every window is <= max_segment_duration, no tiny tail
        spans = []
        for i in range(n):
            offset = i * seg_len
            duration = (total - offset) if i == n - 1 else seg_len
            spans.append((offset, duration))
        return spans
    raise ValueError(
        f"Unknown seg_method={method!r}. Supported: 'fixed'. "
        f"To add VAD, implement a new branch in compute_segment_spans() that returns "
        f"(offset, duration) spans each <= max_segment_duration."
    )


def segment_cutset(cuts, max_segment_duration: float, method: str = "fixed"):
    """Split each recording into <= max_segment_duration windows for long-form inference.

    Returns:
        (segments, seg_meta):
          - ``segments``: a CutSet of truncated cuts, each tagged with
            ``parent_id`` / ``seg_index`` / ``seg_start`` custom attrs and a stable
            id ``f"{parent_id}__seg{i:04d}"``.
          - ``seg_meta``: dict mapping parent recording id -> ordered list of
            ``(seg_index, seg_id, seg_start_secs)``. This is the source of truth for
            regrouping (robust to padding, which preserves ids but may drop attrs).
    """
    segments = []
    seg_meta: dict[str, list[tuple[int, str, float]]] = {}
    for cut in cuts:
        entries = []
        for i, (offset, duration) in enumerate(compute_segment_spans(cut, max_segment_duration, method=method)):
            seg_id = f"{cut.id}__seg{i:04d}"
            seg = cut.truncate(offset=offset, duration=duration, preserve_id=False).with_id(seg_id)
            seg.parent_id = cut.id
            seg.seg_index = i
            seg.seg_start = offset
            segments.append(seg)
            entries.append((i, seg_id, offset))
        seg_meta[cut.id] = entries
    return CutSet.from_cuts(segments), seg_meta


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
class StreamingSTTEvalConfig(CpWERScoringConfig):
    """Inference settings, composed with the scoring settings shared with the offline scorer.

    `CpWERScoringConfig` is a MIXIN, not a metric-specific base: an eval config *has* scoring
    settings rather than being a kind of cpWER config, and a second metric must be able to
    contribute its own fields beside these. Its fields -- `compute_cpwer`, `use_normalizer`,
    `normalizer_language`, the ten cpWER axes, `cpwer_placement`, `cpwer_max_speakers`,
    `cpwer_report_notag_ceiling`, `subset_field` -- are defined once there so the two entry points
    cannot drift. Inherited fields sort first in `--cfg job`, which is the one cosmetic cost.
    """

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
    use_offline_embs: bool = False
    seed: Optional[int] = None  # Set for deterministic results
    pad_extra_duration: Optional[float] = 0.0
    use_state_machine_inference: bool = (
        False  # recommended turned off for chunk_size > 0, no effect for chunk_size <= 0
    )
    dynamic_min_chunk_size: int = 0  # dynamic chunking: min frames before allowing generation
    dynamic_max_chunk_size: Optional[int] = None  # dynamic chunking: max frames before forcing generation
    # Fixed-chunk size (frames) to run inference at. None → use the model config
    # chunk_size (the longest value when the model was trained with a list of sizes).
    chunk_size_override: Optional[int] = None
    # Probability threshold for the boundary decision.
    #   - When use_chunk_classifier_at_inference=True: threshold on the aux
    #     head's sigmoid output. None → 0.5 default.
    #   - When False: threshold on p(user_footer_first_id) from the LM head.
    #     None → fall back to argmax (legacy behavior).
    emit_threshold: Optional[float] = None
    # When True, dump per-LISTENING-frame diagnostics (LM head top-5, prob of
    # user_footer_first / blank, aux head sigmoid, decision taken) to a
    # sibling JSONL alongside output_manifest. Slows inference; use on
    # small eval sets when debugging boundary-decision behavior.
    emit_delay_frames: int = 0
    # K-frame grouping for dynamic-chunking read/write decisions. When None,
    # falls back to model.core_cfg.dynamic_chunk_step (= the value the model
    # was trained with). Override for ablations only.
    dynamic_chunk_step: Optional[int] = None
    disable_emit_for_debug: bool = False
    debug_log_audio_frames: bool = False
    # NOTE: ``use_chunk_classifier_at_inference`` is removed — the aux head is
    # used automatically when the model was trained with ``use_chunk_classifier=True``.
    # When True, save per-word alignments alongside pred_text in the output
    # manifest. Each predicted word inherits the start_time / end_time of the
    # audio chunk it was generated from. Format matches the GT manifest:
    #   [{"text": "...", "start_time": float_s, "end_time": float_s}, ...]
    save_alignments: bool = True
    # --- multi-speaker (SOT) scoring ---
    # Score cpWER during inference. `false` still writes the full manifest -- raw fields, `_run`,
    # WER, everything -- it only skips the cpWER block, which `streaming_stt_score.py` can then
    # produce offline from that manifest. Useful when the GPU box should not also be deciding how
    # scoring works.
    score_inline: bool = True
    # Every cpWER setting -- `compute_cpwer`, the ten axes, `cpwer_placement`, `cpwer_max_speakers`,
    # `cpwer_report_notag_ceiling`, `subset_field` -- is inherited from CpWERScoringConfig, so this
    # script and the offline scorer cannot disagree about what they mean.
    # Feed ORACLE RTTM speaker targets to the encoder at inference instead of letting a
    # ParallelExpertEncoder run its own streaming diarizer. Separates "does the speaker kernel
    # help" from "is the streaming diarizer good enough" -- a weak infusion result is otherwise
    # ambiguous between the two. Needs spk_targets on the batch; runs on either decoder.
    oracle_spk_targets: bool = False
    # --- Long-form segmentation -------------------------------------------
    # When ``max_segment_duration`` is set (> 0), each input recording is split into
    # windows of at most this many seconds; inference runs per segment and the
    # segment transcripts are concatenated back into one hypothesis per
    # recording. This keeps every decode within the model's training length,
    # which is essential for long-form audio far longer than the model saw in
    # training (otherwise the decoder goes out-of-distribution and truncates).
    max_segment_duration: Optional[float] = None
    # How many segments to batch together per generate() call. 0 -> use batch_size.
    # Because segments are bounded by max_segment_duration, padding is bounded too, so
    # this can be pushed high to speed up inference within GPU-memory limits.
    max_concurrent_segments: int = 0
    # Segmentation strategy. "fixed" = even contiguous windows (no deps).
    # Extensible: add "silero_vad" by implementing its spans in
    # ``compute_segment_spans`` — the rest of the pipeline is method-agnostic.
    seg_method: str = "fixed"
    generation_config: StreamingSTTGenerationConfig = field(default_factory=StreamingSTTGenerationConfig)


@hydra_runner(config_name="StreamingSTTEvalConfig", schema=StreamingSTTEvalConfig)
def main(cfg: StreamingSTTEvalConfig):
    if cfg.max_new_tokens is not None or cfg.max_new_tokens > 0:
        cfg.generation_config.max_new_tokens = cfg.max_new_tokens
        logging.warning(f"Setting generation_config.max_new_tokens to {cfg.max_new_tokens}")
        logging.warning(
            f"Using `max_new_tokens` is deprecated, please use `generation_config.max_new_tokens` instead."
        )

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
        # Not a warning: greedy decoding on a fixed machine at fixed settings is already bit-exact,
        # pinned by tests. `seed` is for reproducing across MACHINES -- and note it does not
        # reproduce an unseeded run on this one: it enables deterministic algorithms, which
        # disables the fused subsampling kernel, which rounds differently. It also costs throughput.
        logging.info("Random seed not set; runs are still reproducible at identical settings.")

    model = StreamingSTTModel.from_pretrained(cfg.pretrained_name)
    model = model.eval().to(getattr(torch, cfg.dtype)).to(cfg.device)

    cuts = guess_parse_cutset(cfg.inputs)
    # Resample to model's expected sample rate if needed.
    sample_cut = next(iter(cuts))
    if sample_cut.sampling_rate != model.sampling_rate:
        logging.info(f"Resampling cuts from {sample_cut.sampling_rate} to {model.sampling_rate} Hz")
        cuts = CutSet.from_cuts(c.resample(model.sampling_rate) for c in cuts)

    # Long-form segmentation: split each recording into <= max_segment_duration
    # windows, decode each independently, then concatenate transcripts per
    # recording afterwards (reduction step below). ``ref_cuts`` holds the full
    # recordings (source of references + output order); ``seg_meta`` maps each
    # recording id -> its ordered segments; ``seg_start_by_id`` gives per-segment
    # start offsets (used to globalize per-word alignment timestamps).
    seg_mode = cfg.max_segment_duration is not None and cfg.max_segment_duration > 0
    # Gated, not deleted. Segmented inference may now WRITE a manifest, since scoring can happen
    # later; what is still impossible is scoring cpWER over it, here or offline. The scorer raises
    # the same message from `_run.seg_mode`.
    if seg_mode and cfg.compute_cpwer and cfg.score_inline:
        raise ValueError(
            "cpWER requires globally consistent speaker indices, but <spk:N> is arrival-ordered "
            "WITHIN each decode window -- segments are decoded independently, so <spk:0> in one "
            "segment is generally a different person than in the next, and a session-global "
            "permutation cannot undo a per-segment relabeling. Score cpWER per cut "
            "(max_segment_duration=0), or add cross-segment speaker stitching."
        )
    ref_cuts = cuts
    seg_meta: dict[str, list[tuple[int, str, float]]] = {}
    seg_start_by_id: dict[str, float] = {}
    if seg_mode:
        ref_cuts = list(cuts)  # materialize full recordings (re-iterated for refs/order)
        cuts, seg_meta = segment_cutset(ref_cuts, cfg.max_segment_duration, method=cfg.seg_method)
        seg_start_by_id = {sid: off for entries in seg_meta.values() for (_, sid, off) in entries}
        max_cuts = cfg.max_concurrent_segments or cfg.batch_size
        logging.info(
            f"Segmentation ON (method={cfg.seg_method}, max_segment_duration={cfg.max_segment_duration}s): "
            f"{len(ref_cuts)} recordings -> {len(cuts)} segments; max_concurrent_segments={max_cuts}"
        )
    else:
        max_cuts = cfg.batch_size

    cuts = cuts.sort_by_duration()
    cuts = cuts.map(partial(pad_extra_duration, extra_duration=cfg.pad_extra_duration))
    sampler = lhotse.dataset.DynamicCutSampler(cuts, max_cuts=max_cuts)
    num_batches = math.ceil(len(cuts) / max_cuts)
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=sampler,
        num_workers=cfg.num_workers,
        batch_size=None,
    )

    normalizer = build_normalizer(cfg.use_normalizer, cfg.normalizer_language)
    logging.info(f"Using normalizer {cfg.use_normalizer!r} (language={cfg.normalizer_language})")

    input_durations = []
    infer_durations = []
    # Results keyed by cut id (a "cut" is a segment in seg_mode, else a whole
    # recording). Keying by id decouples accumulation from batch/sort order and
    # lets us regroup segments per recording afterwards. Hypotheses are stored
    # RAW (un-normalized); normalization happens once in the reduction step.
    hyp_by_id: dict[str, str] = {}
    align_by_id: dict[str, list[dict]] = {}
    content_score_by_id: dict[str, list[float]] = {}
    annotated_by_id: dict[str, str] = {}
    content_score_mode: Optional[str] = None

    # Optional per-frame debug log file (one record per LISTENING frame per
    # cut, keyed by cut id). Only opened when debug_log_audio_frames=True.
    debug_log_writer = None
    if cfg.debug_log_audio_frames and cfg.output_manifest is not None:
        manifest_path = Path(cfg.output_manifest)
        debug_log_path = manifest_path.with_name(
            manifest_path.stem.replace("_generations", "") + "_audio_frame_log.jsonl"
        )
        debug_log_writer = SequentialJsonlWriter(str(debug_log_path))
        logging.info(f"Audio frame debug log → {debug_log_path}")

    for batch_idx, batch in tqdm(enumerate(dloader), total=num_batches):
        ts = perf_counter()
        generation_config = GenerationConfig(**OmegaConf.to_container(cfg.generation_config))
        result = model.generate(
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            system_prompt=cfg.system_prompt,
            max_new_tokens=cfg.max_new_tokens,
            generation_config=generation_config,
            use_offline_embs=cfg.use_offline_embs,
            use_state_machine_inference=cfg.use_state_machine_inference,
            dynamic_min_chunk_size=cfg.dynamic_min_chunk_size,
            dynamic_max_chunk_size=cfg.dynamic_max_chunk_size,
            emit_threshold=cfg.emit_threshold,
            emit_delay_frames=cfg.emit_delay_frames,
            dynamic_chunk_step=cfg.dynamic_chunk_step,
            disable_emit_for_debug=cfg.disable_emit_for_debug,
            chunk_size_override=cfg.chunk_size_override,
            return_alignments=cfg.save_alignments,
            return_debug_logs=cfg.debug_log_audio_frames,
            **_oracle_targets(cfg, batch, model.device),
        )
        batch_infer_duration = perf_counter() - ts

        # Write per-frame debug records keyed by cut id.
        if debug_log_writer is not None and result.debug_logs is not None:
            for cut, frames in zip(batch["cuts"], result.debug_logs):
                debug_log_writer.write({"id": cut.id, "duration": cut.duration, "frames": frames})

        if result.content_score_mode is not None:
            content_score_mode = result.content_score_mode

        # Accumulate raw per-cut results keyed by id (segment id in seg_mode).
        for i, cut in enumerate(batch["cuts"]):
            hyp_by_id[cut.id] = result.texts[i].strip()
            if cfg.save_alignments and result.pred_alignments is not None:
                al = result.pred_alignments[i]
                if seg_mode:
                    # Shift per-word timestamps from segment-local to recording-global.
                    off = seg_start_by_id.get(cut.id, 0.0)
                    al = [{**w, "start_time": w["start_time"] + off, "end_time": w["end_time"] + off} for w in al]
                align_by_id[cut.id] = al
            if result.content_scores is not None and i < len(result.content_scores):
                content_score_by_id[cut.id] = result.content_scores[i]
            if result.pred_text_annotated is not None and i < len(result.pred_text_annotated):
                annotated_by_id[cut.id] = result.pred_text_annotated[i]

        batch_duration = sum(c.duration for c in batch["cuts"])
        if cfg.verbose:
            batch_hyps = [normalizer(h.strip()) for h in result.texts]
            batch_rtfx = batch_duration / batch_infer_duration
            logging.info("--------------------------------")
            if seg_mode:
                # No meaningful per-segment reference; log hyps only (corpus WER printed at the end).
                logging.info(f"Batch {batch_idx}: {len(batch_hyps)} segments RTFx={batch_rtfx:.1f}")
                for cut, hyp in zip(batch["cuts"], batch_hyps):
                    logging.info(f"\n[SEG {cut.id}]\t`{hyp}`\n")
            else:
                # Strip tags on BOTH sides: batch_hyps are raw and still carry <spk:N>, while the
                # normalizer removes them from the reference -- comparing the two directly inflates
                # this progress WER.
                batch_refs = [normalizer(remove_speaker_tags(cut.supervisions[0].text)) for cut in batch["cuts"]]
                batch_wer, _, nins, ndel, nsub = word_error_rate_detail(
                    [normalizer(remove_speaker_tags(h)) for h in batch_hyps], batch_refs
                )
                logging.info(
                    f"Batch {batch_idx}: "
                    f"WER={batch_wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}] "
                    f"RTFx={batch_rtfx:.1f}"
                )
                for ref, hyp in zip(batch_refs, batch_hyps):
                    logging.info(f"\n[REF]\t`{ref}`\n[HYP]\t`{hyp}`\n")
            logging.info("--------------------------------")

        input_durations.append(batch_duration)
        infer_durations.append(batch_infer_duration)

    if debug_log_writer is not None:
        debug_log_writer.close()

    # --- Reduce per-cut results to one ReducedRecord per recording ---
    # In seg_mode: concatenate each recording's segment hypotheses (ordered by
    # seg_index) and its offset-adjusted alignments; the reference is the full
    # recording supervision. In normal mode: one recording == one cut.
    # `meta` is captured from the cut HERE, because the cut object is not reachable
    # downstream -- everything after this point sees only ReducedRecord.
    reduced: list[ReducedRecord] = []
    if seg_mode:
        for rec in ref_cuts:
            segs = sorted(seg_meta.get(rec.id, []), key=lambda t: t[0])
            hyp_raw = " ".join(hyp_by_id.get(sid, "") for _, sid, _ in segs).strip()
            alignments = None
            if cfg.save_alignments:
                alignments = [w for _, sid, _ in segs for w in align_by_id.get(sid, [])]
            reduced.append(
                ReducedRecord(
                    id=rec.id,
                    duration=rec.duration,
                    ref_raw=rec.supervisions[0].text,
                    hyp_raw=hyp_raw,
                    alignments=alignments,
                    content_scores=None,
                    annotated=None,
                    meta=_cut_meta(rec),
                )
            )
    else:
        for cut in cuts:
            reduced.append(
                ReducedRecord(
                    id=cut.id,
                    duration=cut.duration,
                    ref_raw=cut.supervisions[0].text,
                    hyp_raw=hyp_by_id.get(cut.id, ""),
                    alignments=align_by_id.get(cut.id) if cfg.save_alignments else None,
                    content_scores=content_score_by_id.get(cut.id),
                    annotated=annotated_by_id.get(cut.id),
                    meta=_cut_meta(cut),
                )
            )

    # Strip speaker tags BEFORE normalizing for the speaker-agnostic WER. Without this, a pure
    # speaker swap with word-identical content scores as word errors when use_normalizer=none.
    refs = [normalizer(remove_speaker_tags(r.ref_raw)) for r in reduced]
    hyps = [normalizer(remove_speaker_tags(r.hyp_raw)) for r in reduced]
    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)

    cpwer_results = cpwer_metrics = None
    cpwer_report = ""
    if cfg.compute_cpwer and cfg.score_inline:
        # cpWER gets its OWN normalizer. Reusing the WER one above silently ignored
        # `cpwer_normalizer`, scoring with `use_normalizer` while the axis stamp still reported the
        # requested family -- a wrong number rather than an error. They coincide only at the
        # inherit-from-WER default.
        cpwer_normalizer = build_normalizer(resolve_normalizer(cfg), cfg.normalizer_language)
        logging.info(f"Using cpWER normalizer {resolve_normalizer(cfg)!r}")
        # Solve each session ONCE and feed the result to both the per-row dump and the accumulator.
        # Scoring per session and then calling update() re-solved every session, four Hungarian
        # solves per session with the ceiling on.
        cpwer_metric = CpWER.from_config(cfg, normalizer=cpwer_normalizer)
        cpwer_results = {}
        for rec in reduced:
            cpwer_results[rec.id] = cpwer_metric.score_session(rec.ref_raw, rec.hyp_raw)
        cpwer_metric.update("corpus", [r.ref_raw for r in reduced], [r.hyp_raw for r in reduced])
        corpus_summary = cpwer_metric.compute()

        # Per-subset buckets, keyed off the input manifest row carried in `meta`.
        subset_metric = CpWER.from_config(cfg, normalizer=cpwer_normalizer)
        subsets_seen = set()
        for rec in reduced:
            subset = (rec.meta.get("custom") or {}).get(cfg.subset_field) or rec.meta.get(cfg.subset_field)
            if subset:
                subset_metric.update(str(subset), [rec.ref_raw], [rec.hyp_raw])
                subsets_seen.add(str(subset))
        subset_summary = subset_metric.compute()
        subsets = {
            name: {k: v for k, v in subset_summary.items() if k.endswith(f"_{name}")} for name in sorted(subsets_seen)
        }
        cpwer_metrics = cpwer_metrics_dict(corpus_summary, subsets, cfg)
        # The same renderer the offline scorer uses, so the two blocks cannot drift.
        cpwer_report = format_cpwer_report(cpwer_metrics, cfg, wer=wer)

    rtfx = sum(input_durations) / sum(infer_durations)
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    # RTFx lives only here: it measures inference and cannot be recomputed offline, which also makes
    # it the visual tell between a block written by this script and one written by the scorer.
    logging.info(f"RTFx: {rtfx:.1f}")
    if cpwer_report:
        # One record per line, not one multi-line record: only the first line of a multi-line
        # message carries the `[NeMo I ...]` prefix, and run scripts grep the log for that prefix
        # plus a keyword. A single record would silently drop every line but the first.
        for line in cpwer_report.splitlines():
            logging.info(line)

    if cfg.output_manifest is not None:
        log_file = Path(cfg.output_manifest).parent / "log.txt"
        with open(log_file, "a") as f:
            f.write(f"======{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}======\n")
            f.write(f"Input: {cfg.inputs}\n")
            if seg_mode:
                f.write(f"Segmentation: method={cfg.seg_method} max_segment_duration={cfg.max_segment_duration}s\n")
            f.write(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]\n")
            f.write(f"RTFx: {rtfx:.1f}\n")
            if cpwer_report:
                f.write(cpwer_report + "\n")
            f.write(f"=============================================\n\n")
        run_block = _build_run_block(cfg, seg_mode=seg_mode)
        with SequentialJsonlWriter(cfg.output_manifest) as writer:
            for rec in reduced:
                cpwer_result = cpwer_results.get(rec.id) if cpwer_results is not None else None
                writer.write(_build_record(rec, normalizer, run_block, cpwer_result, content_score_mode))
        if cpwer_metrics is not None:
            metrics_path = Path(cfg.output_manifest).with_suffix(".metrics.json")
            with open(metrics_path, "w") as handle:
                json.dump(cpwer_metrics, handle, indent=1, sort_keys=True)
                handle.write("\n")
            logging.info(f"Wrote {metrics_path}")


def _cut_meta(cut) -> dict:
    """The input manifest row for this cut, nested whole under ``custom``.

    Nested rather than flattened onto the record, because `cut.custom` carries both ``text`` (the
    RAW reference) and ``duration`` (the UNPADDED one) under names the record already uses for
    different values -- flattening would silently overwrite the normalized ``text`` every consumer
    reads, with no error. Nesting also keeps the record's own schema a function of the code rather
    than of whatever the input manifest happened to contain, so a test can assert it.

    Copied as-is, with no allowlist: whatever the input carried is carried through, so re-scoring
    never has to go back to the input manifest for a field nobody anticipated.

    Args:
        cut: a lhotse cut. ``cut.custom`` may be ``None`` (a MonoCut built without one) or ``{}``
            (a MixedCut whose tracks carry none); both are tolerated.

    Returns:
        dict: ``{"custom": <a copy of the input manifest row>}``, or ``{}`` when the cut carries
            nothing -- an absent ``custom`` key and an empty one are different things to a reader,
            so the empty case writes no key at all. Spread into the record by the caller.
    """
    custom = getattr(cut, "custom", None) or {}
    if not custom:
        return {}
    return {"custom": dict(custom)}


def _wer_counts(ref: str, hyp: str) -> dict:
    """The five integer WER counts for one row, from a single edit-distance call.

    Args:
        ref: the reference, already normalized and tag-stripped -- i.e. the same string written to
            the record's ``text``, so the counts describe exactly what ``wer`` describes.
        hyp: the hypothesis, under the same treatment (the record's ``pred_text``).

    Returns:
        dict: ``wer_errors``, ``wer_ref_words``, ``wer_ins``, ``wer_del``, ``wer_sub``, all ``int``,
            with ``wer_errors == wer_ins + wer_del + wer_sub``. Well-defined on an empty reference
            (``wer_ref_words == 0`` and every error is an insertion) -- no division and no sentinel,
            unlike the ``wer`` rate beside them, which is ``inf`` in that case.
    """
    ref_words, hyp_words = ref.split(), hyp.split()
    d = edit_distance(ref_words, hyp_words)
    return {
        "wer_errors": int(d["total"]),
        "wer_ref_words": len(ref_words),
        "wer_ins": int(d["ins"]),
        "wer_del": int(d["del"]),
        "wer_sub": int(d["sub"]),
    }


def _head_sha() -> Optional[str]:
    """Short sha of the working tree, or ``None`` when not in a checkout.

    `nemo.package_info.__version__` is stamped at install time, so on an editable install it goes
    stale the moment a commit lands -- it identifies the release, not the code that ran. This is the
    live counterpart; both are recorded.

    Returns:
        Optional[str]: the abbreviated sha of ``HEAD``, or ``None`` when git is unavailable, the
            call fails, or the file is not inside a checkout (e.g. an installed wheel). Never
            raises -- provenance must not be able to fail a run.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _build_run_block(cfg: StreamingSTTEvalConfig, *, seg_mode: bool) -> dict:
    """Per-row provenance: what produced this manifest, and what a scorer must refuse on.

    Written on EVERY row rather than to a sidecar or a header line. A sidecar is lost the moment two
    arms are `cat`ed together, and a header breaks strict one-record-per-line readers -- and
    surviving exactly that concatenation is the whole point, since `placement` and `seg_mode` are
    what let an offline scorer refuse work it cannot score correctly.

    Args:
        cfg: the resolved run config; every value recorded is read from it, so the block cannot
            drift from what actually ran.
        seg_mode: whether long-form segmentation was active. Keyword-only because it is derived
            (``max_segment_duration`` set and positive), not a config field, and a positional bool
            beside ``cfg`` would be unreadable at the call site.

    Returns:
        dict: one flat block, identical for every row of a run. Groups: writer provenance
            (``build``, ``head_sha``); what a scorer must refuse on (``placement``, ``seg_mode``,
            ``max_segment_duration``, ``seg_method``); what it needs to re-derive ``text`` /
            ``pred_text`` (``inference_normalizer``, ``normalizer_language``); and the run knobs
            that change the hypothesis, so two manifests are only comparable where these agree.
    """
    from nemo.package_info import __version__ as nemo_version

    return {
        "build": nemo_version,
        "head_sha": _head_sha(),
        # --- what a scorer must refuse on ---
        "placement": cfg.cpwer_placement,
        "seg_mode": seg_mode,
        "max_segment_duration": cfg.max_segment_duration,
        "seg_method": cfg.seg_method,
        # --- needed to re-derive `text` / `pred_text` ---
        "inference_normalizer": cfg.use_normalizer,
        "normalizer_language": cfg.normalizer_language,
        # --- run knobs that change the hypothesis, so two manifests are only comparable if equal ---
        "pretrained_name": cfg.pretrained_name,
        "inputs": cfg.inputs,
        "seed": cfg.seed,
        "pad_extra_duration": cfg.pad_extra_duration,
        "max_new_tokens": cfg.max_new_tokens,
        "system_prompt": cfg.system_prompt,
        # The single biggest confound for a multi-speaker number: oracle RTTM targets versus the
        # streaming diarizer. A manifest that does not record it cannot be compared to another.
        "oracle_spk_targets": cfg.oracle_spk_targets,
        "use_state_machine_inference": cfg.use_state_machine_inference,
        "chunk_size_override": cfg.chunk_size_override,
        "emit_threshold": cfg.emit_threshold,
    }


def _build_record(
    rec: ReducedRecord,
    normalizer: Callable[[str], str],
    run_block: dict,
    cpwer_result: Optional[CpWERSessionResult],
    content_score_mode: Optional[str],
) -> dict:
    """Assemble one output-manifest row.

    Split out of the writer loop so the schema is testable without a GPU or a model: the unit tests
    assert the key inventory, the raw round-trip and the ``_run`` block against this function
    directly.

    Args:
        rec: the reduced record for one recording, carrying the verbatim reference and hypothesis.
        normalizer: the text normalizer to apply. Called AFTER speaker tags are stripped, which is
            what makes ``text`` / ``pred_text`` lossy and ``text_raw`` / ``pred_text_raw``
            necessary.
        run_block: the shared ``_run`` provenance block; built once per run, not per row.
        cpwer_result: this recording's cpWER scores, or ``None`` when cpWER did not run
            (``compute_cpwer=false``, or the id was not scored). When ``None`` the row carries no
            ``cpwer*`` keys at all rather than nulls.
        content_score_mode: run-level label for ``content_scores``; ignored when the record has
            none.

    Returns:
        dict: the row, ready to serialize. Always carries the identity, both text pairs, the WER
            rates and counts, and ``_run``; carries ``custom``, the ``cpwer*`` block,
            ``pred_alignments``, ``content_scores`` and ``pred_text_annotated`` only when the
            corresponding input is present, so an absent key means absent data rather than null.
    """
    rec_id, ref_raw, hyp_raw = rec.id, rec.ref_raw, rec.hyp_raw
    ref = normalizer(remove_speaker_tags(ref_raw))
    hyp = normalizer(remove_speaker_tags(hyp_raw))
    uwer, _, unins, undel, unsub = word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=False)
    record = {
        "id": rec_id,
        "duration": rec.duration,
        "text": ref,
        "pred_text": hyp,
        # RAW, verbatim: tags intact, no normalizer, no tag stripping. `text` and
        # `pred_text` above are lossy -- every Whisper-style normalizer deletes
        # `<spk:N>` as a bracket span -- so these two are the only fields that make
        # offline re-scoring under a different normalizer or parser possible.
        "text_raw": ref_raw,
        "pred_text_raw": hyp_raw,
        "wer": uwer,
        "ins": unins,
        "del": undel,
        "sub": unsub,
        # Integer counts alongside the rates above. Naming follows the row's existing convention:
        # a bare name is a RATE (`wer`, `ins`), a prefixed one is a COUNT (`wer_errors`, `wer_ins`).
        # Counts are not derivable from the rates -- `word_error_rate_detail` returns rates and a
        # word total only, and `wer` is inf on an empty reference -- so they come from one direct
        # edit_distance call.
        **_wer_counts(ref, hyp),
        # Provenance, on every row including runs where cpWER never ran: `placement`
        # and `seg_mode` are what let an offline scorer REFUSE invalid work.
        "_run": run_block,
        # The input manifest row, nested whole. Per-subset bucketing reads its key out of here
        # (`custom.subset_for_metrics` by default). See _cut_meta.
        **rec.meta,
    }
    if cpwer_result is not None:
        record.update(
            {
                "cpwer": cpwer_result.cpwer,
                "cpwer_errors": cpwer_result.errors,
                "cpwer_ref_words": cpwer_result.ref_words,
                "cpwer_ins": cpwer_result.ins,
                "cpwer_del": cpwer_result.dels,
                "cpwer_sub": cpwer_result.subs,
                "num_ref_speakers": cpwer_result.num_ref_speakers,
                "num_hyp_speakers": cpwer_result.num_hyp_speakers,
                "ref_by_speaker": cpwer_result.ref_by_speaker,
                # permuted into REFERENCE order by the assignment -- sorting the two
                # speaker dicts independently would misalign them whenever the key sets
                # differ (hyp {0,2} vs ref {0,1}).
                "hyp_by_speaker": cpwer_result.hyp_in_ref_order,
                "cpwer_assignment": cpwer_result.assignment,
                "notag_ceiling": cpwer_result.notag_ceiling,
            }
        )
    # Per-word predicted timestamps (same schema as the GT manifest's
    # `alignments` field: text / start_time / end_time, in seconds).
    # In seg_mode these are already shifted to recording-global time.
    if rec.alignments is not None:
        record["pred_alignments"] = rec.alignments
    if rec.content_scores is not None:
        record["content_scores"] = rec.content_scores
        record["content_score_mode"] = content_score_mode
    if rec.annotated is not None:
        record["pred_text_annotated"] = rec.annotated
    return record


if __name__ == "__main__":
    main()
