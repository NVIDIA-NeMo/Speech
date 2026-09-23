#!/usr/bin/env python3
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
Add word-level alignments to NeMo manifests using QwenForcedAligner.

Handles three things the naive path gets wrong on multi-speaker (nemoSOT) manifests:

* **``offset``** -- entries are windows into long recordings, so only ``[offset, offset+duration)``
  is read and aligned. Reading the whole file aligns the wrong audio entirely.
* **SOT speaker tags** -- ``<spk:N>`` is stripped before alignment (the aligner would otherwise
  mangle it into a phantom word ``spk0``) and re-attached as a parallel ``speaker_ids`` list.
* **per-entry language** -- taken from the manifest's language field rather than one global value.

``--mode per-speaker`` exists but is **not recommended** -- it measured consistently *worse* than
the default. See "Negative result" below.

Emitted ``alignments`` are in **cut-local seconds** (0 .. duration), which is what
``get_word_alignments_for_batch`` expects: the Lhotse adapter pops ``offset`` before the cut is
built, so absolute times are unrecoverable downstream.

Usage:
    # Single manifest
    python scripts/speechlm2/align_manifest.py \
        --input /path/to/manifest.json \
        --batch-size 8

    # Multiple manifests (comma-separated)
    python scripts/speechlm2/align_manifest.py \
        --input /path/to/train.json,/path/to/dev.json,/path/to/test.json \
        --batch-size 8

Reads each line of the input manifest (JSON-lines with ``audio_filepath``,
``text``, ``duration``), runs forced alignment in batches, and writes a new
manifest with an ``-aligned`` suffix containing an additional ``alignments``
field per utterance:

    {"audio_filepath": "...", "text": "...", "duration": ...,
     "alignments": [{"text": "hello", "start_time": 0.12, "end_time": 0.36}, ...],
     "speaker_ids": [0, 0, 1, ...]}     # SOT manifests only, parallel to `alignments`

Negative result -- ``--mode per-speaker``
-----------------------------------------
The aligner's monotonicity pass collapses runs of words to a single instant, most often at speaker
changes in overlapped speech, because SOT text serialises overlapping speech into a monotone
sequence. The obvious fix is to align each speaker separately against only their own RTTM
segments -- within one speaker, speech really is monotone.

**It does not work.** Measured on 40 multi-speaker debug cuts (1711 words), zero-duration rate:

===========================================  =======  =====  =====  =====  =====
mode                                         overall  1 spk  2 spk  3 spk  4 spk
===========================================  =======  =====  =====  =====  =====
monotonic (default)                           14.1%    6.7%  10.6%  19.7%  28.5%
per-speaker, segments concatenated             40.2%   36.0%  41.1%  44.1%  30.6%
per-speaker, segments masked                   37.1%   35.1%  34.0%  43.2%  36.1%
per-speaker, masked + 0.3 s segment padding    33.8%   31.4%  30.8%  39.5%  34.0%
per-speaker, masked + 1.0 s segment padding    30.8%   29.6%  31.2%  31.2%  29.9%
===========================================  =======  =====  =====  =====  =====

The **1-speaker column is the diagnostic**: those cuts have no overlap, so per-speaker mode should
be a no-op there, yet it regressed 6.7% -> ~30% in every variant. The damage comes from modifying
the audio at all, not from the overlap logic. RTTM speech covers only ~0.79 of the timeline across
~7 segments per cut, so masking replaces a fifth of the audio with digital silence and clips word
edges at every boundary; concatenation additionally deletes the natural pauses. This aligner is
more sensitive to audio discontinuity than to non-monotonic text. Padding helps monotonically
(37.1 -> 33.8 -> 30.8) but converges to ~2x worse than the baseline.

The overlap hypothesis was not wrong -- at 4 speakers, padded per-speaker (29.9%) reaches parity
with monotonic (28.5%) -- it is just swamped by the artifact cost.

Use the default mode plus ``--min-word-step`` (``spread_collapsed_words``), which halves the
zero-duration rate without touching the audio. The mode is retained because a **different aligner**,
one that accepts per-segment time constraints instead of requiring audio surgery, could make this
approach work; Qwen's takes only ``(audio, text)``, which is what forced the surgery.
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from typing import Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

from nemo.collections.asr.parts.utils.sot_speaker_alignment import (
    _SPEAKER_TOKEN_SPLIT_PATTERN,
    SPEAKER_TOKEN_PATTERN,
    has_speaker_tokens,
    parse_speaker_tokens,
    strip_speaker_tags,
)
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# The aligner takes language *names*; NeMo manifests carry ISO codes.
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "ar": "Arabic",
    "th": "Thai",
}


def read_manifest(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_audio(audio_path: str, offset: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
    """Load ``[offset, offset+duration)`` of an audio file as 16 kHz mono float32.

    Slicing at read time (rather than loading the file and trimming) is what makes the emitted
    alignments cut-local, and keeps long-recording manifests cheap to process.
    """
    info = sf.info(audio_path)
    start_frame = int(round(offset * info.samplerate)) if offset else 0
    n_frames = int(round(duration * info.samplerate)) if duration else -1
    audio, sr = sf.read(audio_path, dtype="float32", start=start_frame, frames=n_frames)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def get_output_path(input_path: str) -> str:
    """Derive output path by adding '-aligned' suffix before the extension."""
    p = Path(input_path).absolute()
    return str(p.with_name(f"{p.stem}-aligned{p.suffix}"))


def read_rttm_segments(rttm_path: str, offset: float, duration: float) -> list[tuple[float, float, str]]:
    """Return ``(start, end, speaker)`` RTTM segments clipped to ``[offset, offset+duration)``, cut-local.

    Segments are sorted by start time, which is what makes the arrival-order speaker index below
    agree with the ``<spk:N>`` convention used by these manifests.
    """
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8 or float(parts[4]) == 0:
                continue
            start, dur, speaker = float(parts[3]), float(parts[4]), parts[7]
            end = start + dur
            if end <= offset or start >= offset + duration:
                continue
            segments.append((max(start, offset) - offset, min(end, offset + duration) - offset, speaker))
    segments.sort()
    return segments


def arrival_order_speakers(segments: list[tuple[float, float, str]]) -> list[str]:
    """Speaker labels in order of first appearance — the ``<spk:N>`` index convention."""
    order = []
    for _, _, speaker in segments:
        if speaker not in order:
            order.append(speaker)
    return order


def words_by_speaker(text: str) -> dict[int, list[str]]:
    """Split SOT text into ``{speaker_index: [words in order]}``."""
    per_speaker: dict[int, list[str]] = {}
    current = None
    for part in _SPEAKER_TOKEN_SPLIT_PATTERN.split(text):
        match = SPEAKER_TOKEN_PATTERN.fullmatch(part)
        if match:
            current = int(match.group(1))
            continue
        if current is None:
            continue
        per_speaker.setdefault(current, []).extend(part.split())
    return per_speaker


def pad_and_merge(segments: list[tuple[float, float]], pad: float, limit: float) -> list[tuple[float, float]]:
    """Widen each segment by ``pad`` seconds on both sides and merge overlaps.

    RTTM boundaries are approximate, so masking on the raw spans clips word onsets and offsets --
    with ~7 segments per cut that is ~14 damaged word edges. Padding restores them.
    """
    if not segments:
        return []
    widened = sorted((max(a - pad, 0.0), min(b + pad, limit)) for a, b in segments)
    merged = [list(widened[0])]
    for a, b in widened[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def mask_speaker_audio(audio: np.ndarray, segments: list[tuple[float, float]]) -> np.ndarray:
    """Zero every sample outside ``segments``, keeping the original timeline.

    Masking rather than concatenating is the lesser of two evils: splicing a speaker's segments
    together also removes the natural pauses between them (single-speaker cuts regressed
    6.7% -> 36.0% zero-duration words), while masking at least keeps every timestamp cut-local by
    construction. Both are worse than not touching the audio -- see the module docstring's
    "Negative result" section before using this path.
    """
    masked = np.zeros_like(audio)
    for seg_start, seg_end in segments:
        i0 = max(int(round(seg_start * SAMPLE_RATE)), 0)
        i1 = min(int(round(seg_end * SAMPLE_RATE)), len(audio))
        if i1 > i0:
            masked[i0:i1] = audio[i0:i1]
    return masked


def spread_collapsed_words(alignments: list[dict], duration: float, min_step: float, eps: float = 1e-9) -> int:
    """Give zero-duration words distinct timestamps, in place. Returns how many were adjusted.

    The aligner's monotonicity pass collapses runs of words to a single instant, most often at
    speaker changes in overlapped speech (measured: 9% of words mid-turn, 32% at a speaker change).
    Downstream that is not cosmetic: ``get_llm_messages_for_sample`` assigns a word to a chunk via
    ``ceil(end_time / frame_length)``, so a collapsed run is emitted in ONE chunk instead of spread
    across the turn -- distorting the streaming emission schedule exactly where multi-talker
    behaviour lives.

    Each collapsed run is spread over the gap up to the next word's start (or the cut end), giving
    each word up to ``min_step`` seconds -- one encoder frame by default, which is the granularity
    the chunker actually resolves. Words are never pushed past the following word, so monotonicity
    is preserved.
    """
    n_adjusted = 0
    i = 0
    while i < len(alignments):
        if alignments[i]["end_time"] - alignments[i]["start_time"] > eps:
            i += 1
            continue
        run_start = alignments[i]["start_time"]
        j = i
        while j < len(alignments) and alignments[j]["end_time"] - alignments[j]["start_time"] <= eps:
            j += 1
        next_start = alignments[j]["start_time"] if j < len(alignments) else duration
        count = j - i
        step = min((next_start - run_start) / count, min_step) if count and next_start > run_start else 0.0
        if step > 0:
            for m in range(count):
                alignments[i + m]["start_time"] = round(run_start + m * step, 3)
                alignments[i + m]["end_time"] = round(run_start + (m + 1) * step, 3)
            n_adjusted += count
        i = j
    return n_adjusted


def _prepare_entry(entry: dict, input_path: str, cfg: argparse.Namespace) -> Optional[list[dict]]:
    """Build the alignment job(s) for one manifest entry, or None if it cannot be aligned.

    Monotonic mode yields one job for the whole cut. Per-speaker mode yields one job per speaker,
    each covering only that speaker's RTTM segments — within a single speaker, speech really is
    monotone, so the aligner's monotonicity pass has nothing to fight and words stop collapsing to
    zero duration at speaker changes.
    """
    text = entry.get(cfg.text_field, "")
    if not text:
        return None

    offset = float(entry.get(cfg.offset_field) or 0.0)
    duration = entry.get(cfg.duration_field)
    audio_path = get_full_path(entry[cfg.audio_field], manifest_file=input_path)
    audio = load_audio(audio_path, offset, duration)

    lang_code = str(entry.get(cfg.lang_field) or "").lower()
    language = LANGUAGE_NAMES.get(lang_code, cfg.language)

    if cfg.mode == "monotonic" or not has_speaker_tokens(text):
        align_text, speaker_ids = strip_speaker_tags(text) if has_speaker_tokens(text) else (text, None)
        if not align_text:
            return None
        return [
            {"audio": audio, "text": align_text, "speaker": None, "speaker_ids": speaker_ids, "language": language}
        ]

    rttm_path = entry.get(cfg.rttm_field)
    if not rttm_path:
        raise ValueError(f"--mode per-speaker requires '{cfg.rttm_field}' on every entry")
    rttm_path = get_full_path(rttm_path, manifest_file=input_path)
    segments = read_rttm_segments(rttm_path, offset, float(duration))
    if not segments:
        raise ValueError("no RTTM segments overlap the cut window")
    arrival = arrival_order_speakers(segments)

    jobs = []
    for speaker_idx, words in sorted(words_by_speaker(text).items()):
        if speaker_idx >= len(arrival):
            raise ValueError(f"<spk:{speaker_idx}> has no RTTM counterpart (only {len(arrival)} speakers in window)")
        label = arrival[speaker_idx]
        spans = [(a, b) for a, b, spk in segments if spk == label]
        spans = pad_and_merge(spans, cfg.segment_pad, float(duration))
        if not spans or not words:
            raise ValueError(f"<spk:{speaker_idx}> has no usable audio or words")
        jobs.append(
            {
                "audio": mask_speaker_audio(audio, spans),
                "text": " ".join(words),
                "speaker": speaker_idx,
                "speaker_ids": None,
                "language": language,
            }
        )
    return jobs


def _reassemble(entry: dict, jobs: list[dict], aligned: list[list], cfg: argparse.Namespace) -> Optional[dict]:
    """Turn per-job aligner output into cut-local ``alignments`` + ``speaker_ids`` in SOT order."""
    if jobs[0]["speaker"] is None:  # monotonic mode: one job, already cut-local and in text order
        return {"alignments": [asdict(a) for a in aligned[0]], "speaker_ids": jobs[0]["speaker_ids"]}

    # Per-speaker mode: map each speaker's words back to cut-local time, then replay the SOT order.
    # Masking preserves the timeline, so aligner output is already cut-local.
    queues: dict[int, list[dict]] = {}
    for job, words in zip(jobs, aligned):
        queues[job["speaker"]] = [
            {"text": w.text, "start_time": round(w.start_time, 3), "end_time": round(w.end_time, 3)} for w in words
        ]

    speaker_ids = parse_speaker_tokens(entry[cfg.text_field])
    alignments, cursor = [], {k: 0 for k in queues}
    for spk in speaker_ids:
        q = queues.get(spk)
        if q is None or cursor[spk] >= len(q):
            return None  # aligner returned fewer words than the text has for this speaker
        alignments.append(q[cursor[spk]])
        cursor[spk] += 1
    return {"alignments": alignments, "speaker_ids": speaker_ids}


def align_manifest(
    input_path: str,
    output_path: str,
    aligner: QwenForcedAligner,
    cfg: argparse.Namespace,
    manifest_label: str = "",
):
    """Align a single manifest file and write the output."""
    log.info("%sProcessing: %s -> %s", manifest_label, input_path, output_path)
    entries = read_manifest(input_path)

    # Flatten to jobs: monotonic mode gives one per entry, per-speaker mode one per speaker.
    jobs_by_entry: dict[int, list[dict]] = {}
    n_dropped = 0
    for i, entry in enumerate(entries):
        try:
            jobs = _prepare_entry(entry, input_path, cfg)
        except Exception as e:  # noqa: BLE001 - one bad entry must not kill the run
            log.warning("Entry %d: preparation failed (%s: %s)", i, type(e).__name__, e)
            jobs = None
        if jobs is None:
            n_dropped += 1
        else:
            jobs_by_entry[i] = jobs

    flat = [(i, j) for i, jobs in jobs_by_entry.items() for j in range(len(jobs))]
    by_language: dict[str, list[tuple[int, int]]] = {}
    for key in flat:
        by_language.setdefault(jobs_by_entry[key[0]][key[1]]["language"], []).append(key)

    results: dict[tuple[int, int], list] = {}
    failed_entries: set[int] = set()
    total_batches = sum((len(v) + cfg.batch_size - 1) // cfg.batch_size for v in by_language.values())
    pbar = tqdm(total=total_batches, desc=f"{manifest_label}{Path(input_path).name}", unit="batch")
    original_language = aligner.language
    try:
        for language, keys in by_language.items():
            aligner.language = language
            for batch_start in range(0, len(keys), cfg.batch_size):
                batch = keys[batch_start : batch_start + cfg.batch_size]
                try:
                    out = aligner.align_numpy(
                        [jobs_by_entry[i][j]["audio"] for i, j in batch],
                        [jobs_by_entry[i][j]["text"] for i, j in batch],
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "Alignment failed for a %s batch (%s: %s); dropping it.", language, type(e).__name__, e
                    )
                    failed_entries.update(i for i, _ in batch)
                    pbar.update(1)
                    continue
                for (i, j), words in zip(batch, out):
                    results[(i, j)] = words
                pbar.update(1)
                pbar.set_postfix(done=len(results), dropped=n_dropped)
    finally:
        aligner.language = original_language
        pbar.close()

    n_written = 0
    n_nonmono = 0
    n_spread = 0
    with open(output_path, "w") as out_f:
        for i, entry in enumerate(entries):
            jobs = jobs_by_entry.get(i)
            if jobs is None or i in failed_entries or any((i, j) not in results for j in range(len(jobs))):
                continue
            merged = _reassemble(entry, jobs, [results[(i, j)] for j in range(len(jobs))], cfg)
            if merged is None or not merged["alignments"]:
                log.warning("Entry %d: reassembly failed (word-count mismatch); dropping.", i)
                continue
            aligns = merged["alignments"]
            if cfg.min_word_step > 0:
                n_spread += spread_collapsed_words(aligns, float(entry[cfg.duration_field]), cfg.min_word_step)
            n_nonmono += sum(
                1 for k in range(len(aligns) - 1) if aligns[k]["start_time"] > aligns[k + 1]["start_time"] + 1e-6
            )
            out_entry = dict(entry)
            out_entry["alignments"] = aligns
            if merged["speaker_ids"] is not None:
                out_entry["speaker_ids"] = merged["speaker_ids"]
            out_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
            n_written += 1

    drop_rate = (len(entries) - n_written) / max(len(entries), 1)
    log.info(
        "%sDone. Written: %d, Dropped: %d, Total: %d (drop rate %.2f%%); "
        "non-monotonic transitions: %d; collapsed words spread: %d",
        manifest_label,
        n_written,
        len(entries) - n_written,
        len(entries),
        100 * drop_rate,
        n_nonmono,
        n_spread,
    )
    if drop_rate > cfg.max_drop_rate:
        raise RuntimeError(
            f"{input_path}: drop rate {drop_rate:.2%} exceeds --max-drop-rate {cfg.max_drop_rate:.2%}. "
            "A silent aligner regression would otherwise quietly shrink the corpus."
        )


def main():
    parser = argparse.ArgumentParser(description="Add word-level alignments to NeMo manifests.")
    parser.add_argument(
        "--input",
        required=True,
        help="Comma-separated paths to input NeMo manifests (JSON-lines).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Comma-separated output paths (one per input). Defaults to <input-stem>-aligned.json.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B", help="Pretrained aligner model.")
    parser.add_argument(
        "--language",
        default="English",
        help="Fallback language name, used when an entry's language field is missing or unmapped.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for alignment.")
    parser.add_argument("--device", default="cuda", help="Device for the aligner model.")
    parser.add_argument("--text-field", default="text", help="Manifest field holding the transcript.")
    parser.add_argument("--audio-field", default="audio_filepath", help="Manifest field holding the audio path.")
    parser.add_argument(
        "--offset-field",
        default="offset",
        help="Manifest field holding the start offset in seconds (entries are windows into long recordings).",
    )
    parser.add_argument("--duration-field", default="duration", help="Manifest field holding the duration.")
    parser.add_argument("--rttm-field", default="rttm_filepath", help="Manifest field holding the RTTM path.")
    parser.add_argument(
        "--min-word-step",
        type=float,
        default=0.08,
        help="Spread runs of zero-duration words so consecutive words differ by up to this many "
        "seconds (default 0.08 = one encoder frame). Set 0 to disable.",
    )
    parser.add_argument(
        "--segment-pad",
        type=float,
        default=0.2,
        help="Seconds to widen each RTTM segment by in per-speaker mode, so masking does not clip word edges.",
    )
    parser.add_argument(
        "--mode",
        choices=("monotonic", "per-speaker"),
        default="monotonic",
        help="monotonic: align the whole cut at once (default, no RTTM needed). per-speaker: align each "
        "speaker against only their own masked RTTM segments. NOT RECOMMENDED -- measured ~2x WORSE "
        "than monotonic (14.1%% -> 30.8%% zero-duration words at best); see the module docstring. "
        "Retained for a future aligner that supports per-segment time constraints.",
    )
    parser.add_argument("--lang-field", default="source_lang", help="Manifest field holding the ISO language code.")
    parser.add_argument(
        "--max-drop-rate",
        type=float,
        default=0.02,
        help="Fail the run if more than this fraction of entries is dropped (default 2%%).",
    )
    args = parser.parse_args()

    input_paths = [p.strip() for p in args.input.split(",")]
    if args.output is not None:
        output_paths = [p.strip() for p in args.output.split(",")]
        if len(output_paths) != len(input_paths):
            parser.error(
                f"Number of --output paths ({len(output_paths)}) must match "
                f"number of --input paths ({len(input_paths)})."
            )
    else:
        output_paths = [get_output_path(p) for p in input_paths]

    log.info("Loading aligner: %s", args.model)
    aligner = QwenForcedAligner(
        pretrained_model=args.model,
        language=args.language,
        device=args.device,
    )

    n_manifests = len(input_paths)
    for mi, (input_path, output_path) in enumerate(zip(input_paths, output_paths), 1):
        label = f"[{mi}/{n_manifests}] "
        align_manifest(input_path, output_path, aligner, args, manifest_label=label)

    log.info("All %d manifest(s) processed.", n_manifests)


if __name__ == "__main__":
    main()
