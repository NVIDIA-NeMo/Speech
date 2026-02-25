# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Instrumented streaming evaluation script for StreamingSALM timestamp accuracy.

Processes audio in single-frame chunks (chunk_size=0.08s) for precise
per-token emission frame attribution.  Computes emission delay, jitter,
and word-boundary F1 against QFA forced-alignment ground truth.

Usage::

    python streaming_salm_timestamp_eval.py \
        pretrained_name=models/baseline_hf \
        inputs=/data/librispeech/lhotse/librispeech_cuts_lower_test-clean.jsonl.gz \
        latency=5 \
        output_manifest=results/eval/timestamps_K5.jsonl
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.speechlm2.models import StreamingSALM
from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder
from nemo.core.config import hydra_runner
from nemo.utils import logging


MIMI_FRAME_SHIFT = MimiEncoder.FRAME_SHIFT  # 0.08 seconds
MIMI_FRAME_SAMPLES = int(MimiEncoder.SAMPLE_RATE * MIMI_FRAME_SHIFT)  # 1920


@dataclass
class TimestampEvalConfig:
    pretrained_name: str = ""
    inputs: str = ""
    latency: int = 5
    output_manifest: Optional[str] = "streaming_salm_timestamps.jsonl"
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_normalizer: Optional[str] = "english"
    # Tolerances for boundary F1 (in milliseconds)
    tolerances_ms: list[int] = field(default_factory=lambda: [50, 100, 200])


@hydra_runner(config_name="TimestampEvalConfig", schema=TimestampEvalConfig)
def main(cfg: TimestampEvalConfig):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    model = StreamingSALM.from_pretrained(cfg.pretrained_name)
    model = model.eval().to(getattr(torch, cfg.dtype)).to(cfg.device)
    device = model.device

    cuts = guess_parse_cutset(cfg.inputs)
    normalizer = EnglishTextNormalizer() if cfg.use_normalizer == "english" else (lambda x: x)

    all_emission_delays = []  # in frames
    all_word_results = []  # per-utterance word-level results
    refs = []
    hyps = []

    for cut_idx, cut in enumerate(cuts):
        ref_text = normalizer(cut.supervisions[0].text)

        # Load and resample audio to 24 kHz
        audio_path = cut.recording.sources[0].source
        audio, sr = torchaudio.load(audio_path)
        if sr != MimiEncoder.SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, MimiEncoder.SAMPLE_RATE)
        audio = audio[0]  # mono

        # Run QFA forced alignment to get ground-truth word timestamps
        gt_word_times = _run_forced_alignment(model, audio, ref_text, device, cfg.dtype)

        # Run instrumented single-frame streaming inference
        emission_log, emitted_tokens = _instrumented_streaming(
            model, audio, cfg.latency, device
        )

        hyp_text = normalizer(model.tokenizer.ids_to_text(emitted_tokens).strip())
        refs.append(ref_text)
        hyps.append(hyp_text)

        # Compute per-word emission delays
        if gt_word_times and emission_log:
            word_delays = _compute_word_delays(
                model.tokenizer, emission_log, gt_word_times, ref_text
            )
            for wd in word_delays:
                all_emission_delays.append(wd["delay_frames"])
            all_word_results.append({
                "id": cut.id,
                "duration": cut.duration,
                "text": ref_text,
                "pred_text": hyp_text,
                "word_delays": word_delays,
            })
        else:
            all_word_results.append({
                "id": cut.id,
                "duration": cut.duration,
                "text": ref_text,
                "pred_text": hyp_text,
                "word_delays": [],
            })

        if (cut_idx + 1) % 50 == 0:
            logging.info(f"Processed {cut_idx + 1} utterances...")

    # Compute WER
    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")

    # Compute emission delay statistics
    if all_emission_delays:
        delays_ms = np.array(all_emission_delays) * (MIMI_FRAME_SHIFT * 1000)
        theoretical_min_ms = cfg.latency * MIMI_FRAME_SHIFT * 1000
        excess_ms = delays_ms - theoretical_min_ms

        logging.info(f"\n=== Emission Delay Statistics (K={cfg.latency}) ===")
        logging.info(f"Theoretical minimum delay: {theoretical_min_ms:.0f} ms")
        logging.info(f"Mean emission delay: {np.mean(delays_ms):.1f} ms")
        logging.info(f"Median emission delay: {np.median(delays_ms):.1f} ms")
        logging.info(f"P90 emission delay: {np.percentile(delays_ms, 90):.1f} ms")
        logging.info(f"P95 emission delay: {np.percentile(delays_ms, 95):.1f} ms")
        logging.info(f"Emission jitter (std of excess delay): {np.std(excess_ms):.1f} ms")

        # Compute boundary F1 at each tolerance
        logging.info(f"\n=== Word Boundary F1 ===")
        for tol_ms in cfg.tolerances_ms:
            precision, recall, f1 = _compute_boundary_f1(all_word_results, tol_ms)
            logging.info(
                f"Tolerance={tol_ms}ms: P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
            )

    # Write output manifest
    if cfg.output_manifest:
        with open(cfg.output_manifest, "w") as f:
            for entry in all_word_results:
                f.write(json.dumps(entry) + "\n")
        logging.info(f"Wrote {len(all_word_results)} entries to {cfg.output_manifest}")


def _run_forced_alignment(
    model: StreamingSALM,
    audio: torch.Tensor,
    text: str,
    device: torch.device,
    dtype: str,
) -> list[dict]:
    """
    Run QFA forced alignment to get ground-truth word start times.

    Returns list of {"word": str, "start_time": float, "end_time": float}.
    """
    if not hasattr(model, "forced_aligner") or model.forced_aligner is None:
        return []

    try:
        audio_tensor = audio.unsqueeze(0).to(device, dtype=getattr(torch, dtype))
        audio_lens = torch.tensor([audio.shape[0]], device=device)

        with torch.inference_mode():
            alignment = model.forced_aligner.align(audio_tensor, audio_lens, [text])

        word_times = []
        if alignment and len(alignment) > 0:
            for seg in alignment[0]:
                word_times.append({
                    "word": seg["word"] if isinstance(seg, dict) else seg.word,
                    "start_time": seg["start"] if isinstance(seg, dict) else seg.start,
                    "end_time": seg["end"] if isinstance(seg, dict) else seg.end,
                })
        return word_times
    except Exception as e:
        logging.warning(f"Forced alignment failed: {e}")
        return []


def _instrumented_streaming(
    model: StreamingSALM,
    audio: torch.Tensor,
    latency: int,
    device: torch.device,
) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Run streaming inference one Mimi frame at a time, recording emission frame index.

    Returns:
        emission_log: list of (token_id, emission_frame_idx) tuples
        emitted_tokens: flat list of all emitted token IDs
    """
    emission_log = []
    all_tokens = []

    audio_tensor = audio.to(device)
    total_samples = audio_tensor.shape[0]
    n_frames = total_samples // MIMI_FRAME_SAMPLES

    state = None
    frame_idx = 0

    with torch.inference_mode():
        for f in range(n_frames):
            start = f * MIMI_FRAME_SAMPLES
            end = start + MIMI_FRAME_SAMPLES
            chunk_audio = audio_tensor[start:end].unsqueeze(0)
            chunk_lens = torch.tensor([MIMI_FRAME_SAMPLES], device=device)

            # Encode single frame with Mimi
            codes, code_lens = model.mimi.encode(chunk_audio, chunk_lens)

            # Generate streaming
            emitted, state = model.generate_streaming(codes, state, latency=latency)

            for tok in emitted[0]:
                emission_log.append((tok, frame_idx))
                all_tokens.append(tok)

            frame_idx += 1

        # Handle residual audio (pad to full frame)
        residual_samples = total_samples - n_frames * MIMI_FRAME_SAMPLES
        if residual_samples > 0:
            residual = audio_tensor[n_frames * MIMI_FRAME_SAMPLES:]
            padded = torch.nn.functional.pad(
                residual, (0, MIMI_FRAME_SAMPLES - residual_samples)
            ).unsqueeze(0)
            chunk_lens = torch.tensor([residual_samples], device=device)
            codes, code_lens = model.mimi.encode(padded, chunk_lens)
            emitted, state = model.generate_streaming(codes, state, latency=latency)
            for tok in emitted[0]:
                emission_log.append((tok, frame_idx))
                all_tokens.append(tok)
            frame_idx += 1

        # Flush latency buffer
        if state is not None:
            emitted, state = model.generate_streaming(None, state, latency=latency)
            for tok in emitted[0]:
                emission_log.append((tok, frame_idx))  # flushed at end
                all_tokens.append(tok)

    return emission_log, all_tokens


def _compute_word_delays(
    tokenizer,
    emission_log: list[tuple[int, int]],
    gt_word_times: list[dict],
    ref_text: str,
) -> list[dict]:
    """
    Map emitted tokens back to words and compute per-word emission delay.

    Returns list of dicts with word, ref_frame, emission_frame, delay_frames, delay_ms.
    """
    # Decode emitted tokens to text, tracking token-to-word mapping
    word_delays = []

    # Simple word-based alignment: decode tokens and match to reference words
    token_ids = [tok for tok, _ in emission_log]
    token_frames = [frame for _, frame in emission_log]

    if not token_ids or not gt_word_times:
        return word_delays

    # Decode each token individually to map to words
    decoded_tokens = []
    for tid in token_ids:
        try:
            text = tokenizer.ids_to_text([tid])
            decoded_tokens.append(text)
        except Exception:
            decoded_tokens.append("")

    # Build word spans: group consecutive tokens into words
    # A new word starts when decoded text starts with a space or is the first token
    word_start_indices = [0]
    accumulated = decoded_tokens[0]
    for i in range(1, len(decoded_tokens)):
        t = decoded_tokens[i]
        if t.startswith(" ") or t.startswith("▁"):
            word_start_indices.append(i)
        accumulated += t

    # For each word in ground truth, find the matching emitted word's first token
    ref_words = ref_text.lower().split()
    emitted_words_frames = []
    for ws_idx in word_start_indices:
        emitted_words_frames.append(token_frames[ws_idx])

    # Match ground truth words to emitted words (by position)
    n_match = min(len(gt_word_times), len(emitted_words_frames))
    for i in range(n_match):
        gt = gt_word_times[i]
        ref_frame = round(gt["start_time"] / MIMI_FRAME_SHIFT)
        emission_frame = emitted_words_frames[i]
        delay_frames = emission_frame - ref_frame
        delay_ms = delay_frames * MIMI_FRAME_SHIFT * 1000

        word_delays.append({
            "word": gt["word"],
            "ref_frame": ref_frame,
            "emission_frame": emission_frame,
            "delay_frames": delay_frames,
            "delay_ms": delay_ms,
            "ref_start_time": gt["start_time"],
            "emission_time": emission_frame * MIMI_FRAME_SHIFT,
        })

    return word_delays


def _compute_boundary_f1(
    all_word_results: list[dict],
    tolerance_ms: float,
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for word boundary detection at given tolerance.
    """
    tolerance_s = tolerance_ms / 1000.0
    total_ref = 0
    total_hyp = 0
    total_matched = 0

    for entry in all_word_results:
        word_delays = entry.get("word_delays", [])
        if not word_delays:
            continue

        ref_times = [wd["ref_start_time"] for wd in word_delays]
        hyp_times = [wd["emission_time"] for wd in word_delays]

        total_ref += len(ref_times)
        total_hyp += len(hyp_times)

        # Greedy matching: for each ref time, find closest hyp time within tolerance
        used_hyp = set()
        for rt in ref_times:
            best_idx = None
            best_dist = float("inf")
            for j, ht in enumerate(hyp_times):
                if j in used_hyp:
                    continue
                dist = abs(ht - rt)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            if best_idx is not None and best_dist <= tolerance_s:
                total_matched += 1
                used_hyp.add(best_idx)

    precision = total_matched / total_hyp if total_hyp > 0 else 0.0
    recall = total_matched / total_ref if total_ref > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


if __name__ == "__main__":
    main()
