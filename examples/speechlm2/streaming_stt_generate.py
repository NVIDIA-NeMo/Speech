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
from nemo.collections.asr.parts.utils.hf_asr_normalizer import get_hf_normalizer
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.common.data.lhotse.dataloader import pad_extra_duration
from nemo.collections.speechlm2.models import StreamingSTTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


class ToAudio(torch.utils.data.Dataset):
    """Minimal dataset that loads audio from a CutSet."""

    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


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
    # "english"/"basic": Whisper's normalizer (whisper_normalizer package).
    # "hf": the Open ASR Leaderboard normalizer -- a fork of Whisper's that additionally collapses
    #       acronyms ("b b c" -> "bbc"), normalizes names, and rewrites compounds ("wi fi" -> "wifi").
    #       Required to reproduce leaderboard WER; pair with `normalizer_language`.
    # "none": no normalization.
    use_normalizer: Optional[str] = "english"  # "english", "basic", "hf", or "none"
    normalizer_language: str = "en"  # only used by use_normalizer="hf"; non-"en" needs `num2words`
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
        logging.warning("Random seed not set, results will not be deterministic")

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

    _normalizer_key = cfg.use_normalizer.lower() if isinstance(cfg.use_normalizer, str) else cfg.use_normalizer
    if _normalizer_key == "english":
        normalizer = EnglishTextNormalizer()
    elif _normalizer_key == "basic":
        normalizer = BasicTextNormalizer()
    elif _normalizer_key == "hf":
        normalizer = get_hf_normalizer(cfg.normalizer_language)
        logging.info(f"Using Open ASR Leaderboard normalizer (language={cfg.normalizer_language})")
    else:
        normalizer = lambda x: x  # noqa: E731

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
                batch_refs = [normalizer(cut.supervisions[0].text) for cut in batch["cuts"]]
                batch_wer, _, nins, ndel, nsub = word_error_rate_detail(batch_hyps, batch_refs)
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

    # --- Reduce per-cut results to one (ref, hyp, ...) per recording ---
    # In seg_mode: concatenate each recording's segment hypotheses (ordered by
    # seg_index) and its offset-adjusted alignments; the reference is the full
    # recording supervision. In normal mode: one recording == one cut.
    # Each entry: (id, duration, ref, hyp, alignments|None, content_scores|None, annotated|None).
    reduced: list[tuple] = []
    if seg_mode:
        for rec in ref_cuts:
            segs = sorted(seg_meta.get(rec.id, []), key=lambda t: t[0])
            hyp_raw = " ".join(hyp_by_id.get(sid, "") for _, sid, _ in segs).strip()
            alignments = None
            if cfg.save_alignments:
                alignments = [w for _, sid, _ in segs for w in align_by_id.get(sid, [])]
            reduced.append(
                (
                    rec.id,
                    rec.duration,
                    normalizer(rec.supervisions[0].text),
                    normalizer(hyp_raw),
                    alignments,
                    None,
                    None,
                )
            )
    else:
        for cut in cuts:
            reduced.append(
                (
                    cut.id,
                    cut.duration,
                    normalizer(cut.supervisions[0].text),
                    normalizer(hyp_by_id.get(cut.id, "")),
                    align_by_id.get(cut.id) if cfg.save_alignments else None,
                    content_score_by_id.get(cut.id),
                    annotated_by_id.get(cut.id),
                )
            )

    refs = [r[2] for r in reduced]
    hyps = [r[3] for r in reduced]
    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    rtfx = sum(input_durations) / sum(infer_durations)
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    logging.info(f"RTFx: {rtfx:.1f}")

    if cfg.output_manifest is not None:
        log_file = Path(cfg.output_manifest).parent / "log.txt"
        with open(log_file, "a") as f:
            f.write(f"======{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}======\n")
            f.write(f"Input: {cfg.inputs}\n")
            if seg_mode:
                f.write(f"Segmentation: method={cfg.seg_method} max_segment_duration={cfg.max_segment_duration}s\n")
            f.write(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]\n")
            f.write(f"RTFx: {rtfx:.1f}\n")
            f.write(f"=============================================\n\n")
        with SequentialJsonlWriter(cfg.output_manifest) as writer:
            for rec_id, duration, ref, hyp, alignments, content_scores, annotated in reduced:
                uwer, _, unins, undel, unsub = word_error_rate_detail(
                    hypotheses=[hyp], references=[ref], use_cer=False
                )
                record = {
                    "id": rec_id,
                    "duration": duration,
                    "text": ref,
                    "pred_text": hyp,
                    "wer": uwer,
                    "ins": unins,
                    "del": undel,
                    "sub": unsub,
                }
                # Per-word predicted timestamps (same schema as the GT manifest's
                # `alignments` field: text / start_time / end_time, in seconds).
                # In seg_mode these are already shifted to recording-global time.
                if alignments is not None:
                    record["pred_alignments"] = alignments
                if content_scores is not None:
                    record["content_scores"] = content_scores
                    record["content_score_mode"] = content_score_mode
                if annotated is not None:
                    record["pred_text_annotated"] = annotated
                writer.write(record)


if __name__ == "__main__":
    main()
