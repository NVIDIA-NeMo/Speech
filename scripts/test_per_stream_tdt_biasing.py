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

"""Test per-stream keyphrase boosting for Parakeet TDT beam search (malsd_batch).

This script verifies two properties of per-stream biasing:

1. **Effectiveness**: Boosting with a known phrase changes the decoded transcript
   (ideally toward the boosted phrase). We compare no-biasing baseline vs.
   ground-truth-boosted transcripts.

2. **Per-stream independence**: Biasing applied to stream A does not affect
   stream B's output in the same batch. We run the same audio in two settings:
   - A batch [A, B] where A is boosted with phrase_X and B with phrase_Y
   - Each stream decoded alone with the same phrase
   The outputs must be identical, proving no cross-stream leakage.

Usage:
    # Set GPU
    export CUDA_VISIBLE_DEVICES=0

    # With a manifest (JSON lines: {"audio_filepath": "...", "text": "..."})
    python scripts/test_per_stream_tdt_biasing.py \
        --manifest /path/to/manifest.json \
        --model nvidia/parakeet-tdt-0.6b-v2

    # Or with explicit audio files and phrases
    python scripts/test_per_stream_tdt_biasing.py \
        --audio file1.wav file2.wav \
        --phrases "phrase one" "phrase two" \
        --model nvidia/parakeet-tdt-0.6b-v2

    # With word-level timestamps
    python scripts/test_per_stream_tdt_biasing.py \
        --manifest /path/to/manifest.json \
        --timestamps

    # With inference speed benchmark
    python scripts/test_per_stream_tdt_biasing.py \
        --manifest /path/to/manifest.json \
        --benchmark --warmup 3 --runs 10
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from Levenshtein import opcodes as lev_opcodes

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from nemo.collections.asr.parts.submodules.tdt_beam_decoding import BeamBatchedTDTInfer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs


# ---------------------------------------------------------------------------
# Helpers (following the pattern from demo/demo_per_stream_biasing.py)
# ---------------------------------------------------------------------------


def load_audio(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    import librosa

    audio, _ = librosa.load(file_path, sr=target_sr)
    return torch.tensor(audio, dtype=torch.float32)


def make_preprocessor_deterministic(model: ASRModel):
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0


def encode_audio(
    audio_paths: list[str],
    model: ASRModel,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the encoder on a batch of audio files and return (encoded, encoded_len)."""
    make_preprocessor_deterministic(model)
    model.eval()
    all_inputs, all_lengths = [], []
    for path in audio_paths:
        audio = load_audio(path)
        all_inputs.append(audio)
        all_lengths.append(torch.tensor(audio.shape[0], dtype=torch.int64))
    input_batch = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True).to(
        device=device, dtype=torch.float32
    )
    length_batch = torch.stack(all_lengths).to(device)
    with torch.no_grad(), torch.inference_mode():
        encoded, encoded_len = model(input_signal=input_batch, input_signal_length=length_batch)
    return encoded, encoded_len


def make_beam_decoder(
    model: ASRModel,
    beam_size: int = 4,
    enable_per_stream_biasing: bool = False,
) -> BeamBatchedTDTInfer:
    """Create a standalone BeamBatchedTDTInfer decoder.

    Unlike ``model.change_decoding_strategy``, this constructs the decoder with
    ``model.decoder`` and ``model.joint`` which are already on the correct device,
    so all internal submodules (including ``biasing_multi_model``) inherit the
    right device placement.
    """
    model_cfg = model.to_config_dict()
    durations = list(model_cfg["model_defaults"]["tdt_durations"])
    vocab_size = model.tokenizer.vocab_size
    decoder = BeamBatchedTDTInfer(
        decoder_model=model.decoder,
        joint_model=model.joint,
        durations=durations,
        blank_index=vocab_size,
        beam_size=beam_size,
        score_norm=True,
        return_best_hypothesis=True,
        allow_cuda_graphs=False,
        enable_per_stream_biasing=enable_per_stream_biasing,
    )
    return decoder


def decode_with_beam(
    decoder: BeamBatchedTDTInfer,
    tokenizer,
    encoder_output: torch.Tensor,
    encoded_lengths: torch.Tensor,
    multi_biasing_ids: Optional[torch.Tensor] = None,
) -> list[str]:
    """Decode encoder output with a BeamBatchedTDTInfer decoder."""
    with torch.no_grad(), torch.inference_mode():
        hyps = decoder(
            encoder_output=encoder_output,
            encoded_lengths=encoded_lengths,
            multi_biasing_ids=multi_biasing_ids,
        )[0]
    texts = [
        h.text if h.text is not None else tokenizer.ids_to_text(h.y_sequence.tolist()) for h in hyps
    ]
    return texts


def decode_with_beam_hypotheses(
    decoder: BeamBatchedTDTInfer,
    tokenizer,
    encoder_output: torch.Tensor,
    encoded_lengths: torch.Tensor,
    multi_biasing_ids: Optional[torch.Tensor] = None,
) -> list[Hypothesis]:
    """Decode and return full Hypothesis objects (with token-level timestamps)."""
    with torch.no_grad(), torch.inference_mode():
        hyps = decoder(
            encoder_output=encoder_output,
            encoded_lengths=encoded_lengths,
            multi_biasing_ids=multi_biasing_ids,
        )[0]
    for h in hyps:
        if h.text is None:
            h.text = tokenizer.ids_to_text(h.y_sequence.tolist())
    return hyps


def compute_word_timestamps(
    model: ASRModel,
    hyps: list[Hypothesis],
) -> list[Hypothesis]:
    """Convert token-level timestamps on hypotheses to word-level timestamps in seconds.

    Uses model.decoding.compute_rnnt_timestamps() which dispatches to the TDT
    variant (_compute_offsets_tdt) for TDT models, then process_timestamp_outputs()
    to convert frame indices to seconds.
    """
    subsampling_factor = model.encoder.subsampling_factor
    window_stride = model.cfg["preprocessor"]["window_stride"]
    for hyp in hyps:
        model.decoding.compute_rnnt_timestamps(hyp, timestamp_type="word")
    return process_timestamp_outputs(hyps, subsampling_factor, window_stride)


def format_word_timestamps(hyp: Hypothesis) -> str:
    """Format word-level timestamps as a readable string."""
    if not isinstance(hyp.timestamp, dict) or "word" not in hyp.timestamp:
        return "  (no word timestamps available)"
    lines = []
    for w in hyp.timestamp["word"]:
        start = w.get("start", 0.0)
        end = w.get("end", 0.0)
        word = w.get("word", "")
        lines.append(f"    {start:7.3f} - {end:7.3f}  {word}")
    return "\n".join(lines)


def register_per_stream_biasing(
    decoder: BeamBatchedTDTInfer,
    tokenizer,
    batch_phrases: list[str | list[str] | None],
    device: torch.device,
    alpha: float = 2.0,
) -> tuple[torch.Tensor, list[BiasingRequestItemConfig | None]]:
    """Register a different boosting tree for each batch element.

    Args:
        batch_phrases: one entry per batch element. Each entry can be:
            - None or empty: no biasing for this stream
            - str: a single phrase to boost
            - list[str]: multiple phrases to boost (all registered in one trie)

    Returns:
        multi_biasing_ids: [B] tensor of model IDs (-1 = no biasing)
        biasing_requests: list of BiasingRequestItemConfig for cleanup
    """
    biasing_multi_model = decoder.decoding_computer.biasing_multi_model
    assert biasing_multi_model is not None, "enable_per_stream_biasing must be True on the decoder"

    batch_size = len(batch_phrases)
    multi_biasing_ids = torch.full([batch_size], fill_value=-1, dtype=torch.long, device=device)
    biasing_requests: list[BiasingRequestItemConfig | None] = []

    for batch_idx, phrases in enumerate(batch_phrases):
        if not phrases:
            biasing_requests.append(None)
            continue
        # Normalize to a list of phrase strings
        if isinstance(phrases, str):
            phrase_list = [phrases]
        else:
            phrase_list = list(phrases)
        request = BiasingRequestItemConfig(
            boosting_model_cfg=BoostingTreeModelConfig(key_phrases_list=phrase_list),
            boosting_model_alpha=alpha,
        )
        request.add_to_multi_model(
            tokenizer=tokenizer,
            biasing_multi_model=biasing_multi_model,
        )
        if request.multi_model_id is not None:
            multi_biasing_ids[batch_idx] = request.multi_model_id
        biasing_requests.append(request)

    return multi_biasing_ids, biasing_requests


def cleanup_biasing(decoder: BeamBatchedTDTInfer, biasing_requests: list[BiasingRequestItemConfig | None]):
    """Remove all registered biasing models from the multi-model."""
    biasing_multi_model = decoder.decoding_computer.biasing_multi_model
    with torch.inference_mode():
        for request in biasing_requests:
            if request is not None and request.multi_model_id is not None:
                biasing_multi_model.remove_model(request.multi_model_id)
                request.multi_model_id = None


# ---------------------------------------------------------------------------
# Alignment display
# ---------------------------------------------------------------------------


def _tokenize_words(text: str) -> list[str]:
    """Simple whitespace tokenizer (lowercased) for word-level alignment."""
    return text.lower().split()


def format_alignment(ref: str, hyp: str) -> str:
    """Format a word-level alignment between reference and hypothesis using Levenshtein opcodes.

    Each line shows the edit operation (equal/replace/insert/delete) with aligned words.
    """
    ref_words = _tokenize_words(ref)
    hyp_words = _tokenize_words(hyp)
    ops = lev_opcodes(ref_words, hyp_words)

    lines = []
    for tag, i1, i2, j1, j2 in ops:
        ref_slice = " ".join(ref_words[i1:i2]) or "—"
        hyp_slice = " ".join(hyp_words[j1:j2]) or "—"
        symbol = {"equal": "=", "replace": "~", "insert": "+", "delete": "-"}[tag]
        lines.append(f"    {symbol} {tag:8} ref: {ref_slice!s:40s} hyp: {hyp_slice}")
    return "\n".join(lines)


def print_aligned_example(
    idx: int,
    audio_path: str,
    reference: str,
    baseline: str,
    boosted: str,
    trie_phrases: list[str],
):
    """Print a single example with ref/baseline/boosted and word-level alignments."""
    print(f"  [{idx}] {Path(audio_path).name}")
    print(f"      Reference:  {reference}")
    print(f"      Baseline:   {baseline}")
    print(f"      Boosted:    {boosted}")
    print(f"      Trie words: {trie_phrases}")
    print(f"      Alignment (ref vs baseline):")
    print(format_alignment(reference, baseline))
    print(f"      Alignment (ref vs boosted):")
    print(format_alignment(reference, boosted))
    print()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str


def test_effectiveness(
    model: ASRModel,
    audio_paths: list[str],
    references: list[str],
    boosting_phrases: list[str],
    device: torch.device,
    boosting_alpha: float = 2.0,
    beam_size: int = 4,
) -> list[TestResult]:
    """Test 1: Biasing with keyphrase phrases changes the transcript.

    For each audio file, compare:
      - Baseline (no biasing)
      - Boosted (biasing with the keyphrase(s) from the dataset)

    The test passes if at least one sample shows a difference, proving the
    boosting mechanism actually affects decoding.
    """
    results = []
    print("\n=== Test 1: Effectiveness ===")
    print(f"Testing {len(audio_paths)} audio files\n")

    tokenizer = model.tokenizer

    # Encode all audio once (encoder output is independent of decoding strategy)
    encoder_output, encoded_lengths = encode_audio(audio_paths, model, device)

    # Baseline: beam search without biasing
    baseline_decoder = make_beam_decoder(model, beam_size=beam_size, enable_per_stream_biasing=False)
    baseline_texts = decode_with_beam(baseline_decoder, tokenizer, encoder_output, encoded_lengths)

    # Boosted: per-stream biasing with keyphrase phrases
    boosted_decoder = make_beam_decoder(model, beam_size=beam_size, enable_per_stream_biasing=True)
    multi_biasing_ids, biasing_requests = register_per_stream_biasing(
        boosted_decoder, tokenizer, boosting_phrases, device, alpha=boosting_alpha,
    )
    boosted_texts = decode_with_beam(
        boosted_decoder, tokenizer, encoder_output, encoded_lengths, multi_biasing_ids,
    )
    cleanup_biasing(boosted_decoder, biasing_requests)

    any_changed = False
    for i, (base, boost, ref_text) in enumerate(zip(baseline_texts, boosted_texts, references)):
        changed = base != boost
        if changed:
            any_changed = True
        trie_phrases = boosting_phrases[i] if boosting_phrases[i] else []
        print_aligned_example(i, audio_paths[i], ref_text, base, boost, trie_phrases)

    if any_changed:
        results.append(TestResult("effectiveness_any_changed", True, "At least one transcript changed with biasing"))
    else:
        results.append(
            TestResult(
                "effectiveness_any_changed",
                False,
                "No transcript changed with biasing — boosting may not be effective",
            )
        )

    # Also check WER
    from nemo.collections.asr.metrics.wer import word_error_rate

    wer_baseline = word_error_rate(hypotheses=baseline_texts, references=references)
    wer_boosted = word_error_rate(hypotheses=boosted_texts, references=references)
    print(f"  WER baseline:  {wer_baseline:.4f}")
    print(f"  WER boosted:   {wer_boosted:.4f}")
    print(f"  WER improved:   {wer_boosted < wer_baseline}")
    print()

    results.append(
        TestResult(
            "effectiveness_wer",
            wer_boosted <= wer_baseline,
            f"WER baseline={wer_baseline:.4f}, boosted={wer_boosted:.4f}",
        )
    )

    return results


def test_independence(
    model: ASRModel,
    audio_paths: list[str],
    references: list[str],
    boosting_phrases: list[str],
    device: torch.device,
    boosting_alpha: float = 2.0,
    beam_size: int = 4,
) -> list[TestResult]:
    """Test 2: Per-stream biasing is independent across batch elements.

    For a batch [A, B] where A is boosted with phrase_A and B with phrase_B:
      - Decode the batch together
      - Decode A alone with phrase_A
      - Decode B alone with phrase_B
      - Verify A_batch == A_alone and B_batch == B_alone

    This proves no cross-stream biasing leakage.
    """
    results = []
    print("\n=== Test 2: Per-Stream Independence ===")

    if len(audio_paths) < 2:
        results.append(TestResult("independence", False, "Need at least 2 audio files for independence test"))
        return results

    tokenizer = model.tokenizer
    ap_a, ap_b = audio_paths[0], audio_paths[1]
    ref_a, ref_b = references[0], references[1]
    boost_a, boost_b = boosting_phrases[0], boosting_phrases[1]

    print(f"  Stream A: {Path(ap_a).name} (boost: '{boost_a}')")
    print(f"  Stream B: {Path(ap_b).name} (boost: '{boost_b}')")
    print()

    # --- Batch decode: A and B together, each with its own biasing ---
    encoded_batch, encoded_batch_len = encode_audio([ap_a, ap_b], model, device)

    batch_decoder = make_beam_decoder(model, beam_size=beam_size, enable_per_stream_biasing=True)
    multi_biasing_ids, biasing_requests = register_per_stream_biasing(
        batch_decoder, tokenizer, [boost_a, boost_b], device, alpha=boosting_alpha,
    )
    batch_texts = decode_with_beam(
        batch_decoder, tokenizer, encoded_batch, encoded_batch_len, multi_biasing_ids,
    )
    cleanup_biasing(batch_decoder, biasing_requests)

    batch_a, batch_b = batch_texts[0], batch_texts[1]
    print(f"  Batch decode:")
    print(f"    A (batch): {batch_a}")
    print(f"    B (batch): {batch_b}")
    print(f"    A trie: {boost_a}")
    print(f"    B trie: {boost_b}")
    print(f"    Alignment (A ref vs A batch):")
    print(format_alignment(ref_a, batch_a))
    print(f"    Alignment (B ref vs B batch):")
    print(format_alignment(ref_b, batch_b))
    print()

    # --- Solo decode: A alone with boost_a ---
    encoded_a, encoded_a_len = encode_audio([ap_a], model, device)
    solo_decoder_a = make_beam_decoder(model, beam_size=beam_size, enable_per_stream_biasing=True)
    multi_ids_a, reqs_a = register_per_stream_biasing(
        solo_decoder_a, tokenizer, [boost_a], device, alpha=boosting_alpha,
    )
    solo_a = decode_with_beam(solo_decoder_a, tokenizer, encoded_a, encoded_a_len, multi_ids_a)[0]
    cleanup_biasing(solo_decoder_a, reqs_a)
    print(f"  Solo decode:")
    print(f"    A (solo):  {solo_a}")
    print(f"    A trie:    {boost_a}")
    print(f"    Alignment (A ref vs A solo):")
    print(format_alignment(ref_a, solo_a))
    print()

    # --- Solo decode: B alone with boost_b ---
    encoded_b, encoded_b_len = encode_audio([ap_b], model, device)
    solo_decoder_b = make_beam_decoder(model, beam_size=beam_size, enable_per_stream_biasing=True)
    multi_ids_b, reqs_b = register_per_stream_biasing(
        solo_decoder_b, tokenizer, [boost_b], device, alpha=boosting_alpha,
    )
    solo_b = decode_with_beam(solo_decoder_b, tokenizer, encoded_b, encoded_b_len, multi_ids_b)[0]
    cleanup_biasing(solo_decoder_b, reqs_b)
    print(f"    B (solo):  {solo_b}")
    print(f"    B trie:    {boost_b}")
    print(f"    Alignment (B ref vs B solo):")
    print(format_alignment(ref_b, solo_b))
    print()

    # --- Compare ---
    a_match = batch_a == solo_a
    b_match = batch_b == solo_b
    print(f"  A batch == A solo: {a_match}")
    print(f"  B batch == B solo: {b_match}")
    print()

    if a_match and b_match:
        results.append(TestResult("independence", True, "Per-stream biasing is independent (no cross-stream leakage)"))
    else:
        if not a_match:
            print(f"  MISMATCH A: batch='{batch_a}' vs solo='{solo_a}'")
        if not b_match:
            print(f"  MISMATCH B: batch='{batch_b}' vs solo='{solo_b}'")
        results.append(
            TestResult("independence", False, "Cross-stream biasing leakage detected — batch and solo outputs differ")
        )

    # --- Additional test: biasing A with a different phrase shouldn't change B ---
    wrong_phrase = "supercalifragilisticexpialidocious"
    cross_decoder = make_beam_decoder(model, beam_size=beam_size, enable_per_stream_biasing=True)
    multi_ids_cross, reqs_cross = register_per_stream_biasing(
        cross_decoder, tokenizer, [wrong_phrase, boost_b], device, alpha=boosting_alpha,
    )
    cross_texts = decode_with_beam(
        cross_decoder, tokenizer, encoded_batch, encoded_batch_len, multi_ids_cross,
    )
    cleanup_biasing(cross_decoder, reqs_cross)

    b_still_same = cross_texts[1] == batch_b
    print(f"  Cross-contamination check:")
    print(f"    A (wrong boost): {cross_texts[0]}")
    print(f"    B (correct boost): {cross_texts[1]}")
    print(f"    A trie: {wrong_phrase}")
    print(f"    B trie: {boost_b}")
    print(f"    Alignment (A wrong ref vs A wrong boost):")
    print(format_alignment(wrong_phrase, cross_texts[0]))
    print(f"    Alignment (B ref vs B correct boost):")
    print(format_alignment(ref_b, cross_texts[1]))
    print(f"    B unaffected by A's wrong boost: {b_still_same}")
    print()

    if b_still_same:
        results.append(
            TestResult("independence_cross_contamination", True, "B's output unaffected by A's wrong biasing")
        )
    else:
        results.append(
            TestResult(
                "independence_cross_contamination",
                False,
                f"B's output changed when A's biasing changed: '{batch_b}' -> '{cross_texts[1]}'",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    name: str
    mean_time_s: float
    std_time_s: float
    rtfx: float
    audio_duration_s: float


def get_audio_durations(audio_paths: list[str]) -> list[float]:
    """Get audio durations in seconds."""
    import librosa

    durations = []
    for p in audio_paths:
        dur = librosa.get_duration(path=p)
        durations.append(dur)
    return durations


def time_decode(
    decoder: BeamBatchedTDTInfer,
    tokenizer,
    encoder_output: torch.Tensor,
    encoded_lengths: torch.Tensor,
    multi_biasing_ids: Optional[torch.Tensor],
    warmup: int,
    runs: int,
    model: Optional[ASRModel] = None,
) -> tuple[float, float]:
    """Time the decode step with warmup. Returns (mean_s, std_s).

    If ``model`` is provided, word-level timestamp computation is included in
    the timed region (decode + compute_rnnt_timestamps + process_timestamp_outputs).
    """
    do_timestamps = model is not None

    def run_once():
        with torch.no_grad(), torch.inference_mode():
            hyps = decoder(
                encoder_output=encoder_output,
                encoded_lengths=encoded_lengths,
                multi_biasing_ids=multi_biasing_ids,
            )[0]
        if do_timestamps:
            for h in hyps:
                if h.text is None:
                    h.text = tokenizer.ids_to_text(h.y_sequence.tolist())
            compute_word_timestamps(model, hyps)
        return hyps

    # Warmup
    for _ in range(warmup):
        _ = run_once()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = run_once()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    import statistics

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def benchmark_biasing(
    model: ASRModel,
    audio_paths: list[str],
    boosting_phrases: list[str | list[str] | None],
    device: torch.device,
    boosting_alpha: float = 2.0,
    beam_size: int = 4,
    warmup: int = 3,
    runs: int = 10,
) -> list[BenchResult]:
    """Benchmark inference speed: baseline (no biasing) vs. per-stream biasing.

    Times the decode step only (encoder output is computed once and reused).
    Reports RTFx = audio_duration / decode_time (higher = faster).
    """
    print("\n=== Benchmark: Inference Speed Impact ===")
    print(f"  Warmup: {warmup} iters, Measured: {runs} iters")
    print()

    tokenizer = model.tokenizer
    encoder_output, encoded_lengths = encode_audio(audio_paths, model, device)
    audio_durations = get_audio_durations(audio_paths)
    total_audio_s = sum(audio_durations)

    results = []

    # Baseline: no biasing
    baseline_decoder = make_beam_decoder(model, beam_size=beam_size, enable_per_stream_biasing=False)
    mean_base, std_base = time_decode(
        baseline_decoder, tokenizer, encoder_output, encoded_lengths, None, warmup, runs
    )
    rtfx_base = total_audio_s / mean_base
    results.append(BenchResult("baseline_no_biasing", mean_base, std_base, rtfx_base, total_audio_s))
    print(f"  Baseline (no biasing):")
    print(f"    decode time: {mean_base * 1000:.1f} ± {std_base * 1000:.1f} ms")
    print(f"    RTFx:        {rtfx_base:.2f}x")
    print()

    # With per-stream biasing
    biased_decoder = make_beam_decoder(model, beam_size=beam_size, enable_per_stream_biasing=True)
    multi_biasing_ids, biasing_requests = register_per_stream_biasing(
        biased_decoder, tokenizer, boosting_phrases, device, alpha=boosting_alpha,
    )
    mean_bias, std_bias = time_decode(
        biased_decoder, tokenizer, encoder_output, encoded_lengths, multi_biasing_ids, warmup, runs
    )
    rtfx_bias = total_audio_s / mean_bias
    results.append(BenchResult("with_per_stream_biasing", mean_bias, std_bias, rtfx_bias, total_audio_s))
    cleanup_biasing(biased_decoder, biasing_requests)

    print(f"  With per-stream biasing:")
    print(f"    decode time: {mean_bias * 1000:.1f} ± {std_bias * 1000:.1f} ms")
    print(f"    RTFx:        {rtfx_bias:.2f}x")
    print()

    # With per-stream biasing + word-level timestamps
    biased_decoder_ts = make_beam_decoder(model, beam_size=beam_size, enable_per_stream_biasing=True)
    multi_biasing_ids_ts, biasing_requests_ts = register_per_stream_biasing(
        biased_decoder_ts, tokenizer, boosting_phrases, device, alpha=boosting_alpha,
    )
    mean_ts, std_ts = time_decode(
        biased_decoder_ts, tokenizer, encoder_output, encoded_lengths, multi_biasing_ids_ts,
        warmup, runs, model=model,
    )
    rtfx_ts = total_audio_s / mean_ts
    results.append(BenchResult("biasing_with_timestamps", mean_ts, std_ts, rtfx_ts, total_audio_s))
    cleanup_biasing(biased_decoder_ts, biasing_requests_ts)

    print(f"  With per-stream biasing + timestamps:")
    print(f"    decode time: {mean_ts * 1000:.1f} ± {std_ts * 1000:.1f} ms")
    print(f"    RTFx:        {rtfx_ts:.2f}x")
    print()

    overhead_pct = (mean_bias - mean_base) / mean_base * 100
    print(f"  Overhead from biasing:           {overhead_pct:+.1f}% ({(mean_bias - mean_base) * 1000:.1f} ms)")
    ts_overhead_pct = (mean_ts - mean_bias) / mean_bias * 100
    print(f"  Overhead from timestamps:         {ts_overhead_pct:+.1f}% ({(mean_ts - mean_bias) * 1000:.1f} ms)")
    total_overhead_pct = (mean_ts - mean_base) / mean_base * 100
    print(f"  Total overhead (biasing + ts):    {total_overhead_pct:+.1f}% ({(mean_ts - mean_base) * 1000:.1f} ms)")
    print(f"  RTFx ratio (biasing):     {rtfx_bias / rtfx_base:.3f}x")
    print(f"  RTFx ratio (biasing+ts):  {rtfx_ts / rtfx_base:.3f}x")
    print()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Test per-stream keyphrase boosting for Parakeet TDT beam search")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="Pretrained model name or path to .nemo file",
    )
    parser.add_argument("--manifest", type=str, help="Path to manifest JSON file (one JSON per line)")
    parser.add_argument("--audio", nargs="+", help="Audio file paths")
    parser.add_argument("--phrases", nargs="+", help="Key phrases to boost (one per audio file)")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for malsd_batch decoding")
    parser.add_argument("--boosting_alpha", type=float, default=2.0, help="Boosting model alpha weight")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect, uses CUDA if available)",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run inference speed benchmark")
    parser.add_argument("--timestamps", action="store_true", help="Print word-level timestamps")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations for benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Measured iterations for benchmark")
    return parser.parse_args()


def load_manifest(manifest_path: str) -> list[dict]:
    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    args = parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model}")
    model = ASRModel.from_pretrained(model_name=args.model, map_location="cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded. Type: {type(model).__name__}")

    # Verify the beam decoder can be created with per-stream biasing
    test_decoder = make_beam_decoder(model, beam_size=args.beam_size, enable_per_stream_biasing=True)
    assert test_decoder.decoding_computer.biasing_multi_model is not None, \
        "biasing_multi_model is None — per-stream biasing not wired"
    print(f"  biasing_multi_model: {test_decoder.decoding_computer.biasing_multi_model}")
    print()

    # Load data
    if args.manifest:
        records = load_manifest(args.manifest)
        audio_paths = [r["audio_filepath"] for r in records]
        references = [r["text"] for r in records]
        # Use medical_terms column for boosting if available, otherwise fall back to full reference
        boosting_phrases = [r.get("medical_terms", r["text"]) for r in records]
    elif args.audio and args.phrases:
        audio_paths = args.audio
        references = args.phrases
        boosting_phrases = args.phrases
    else:
        print("Error: provide --manifest or both --audio and --phrases")
        sys.exit(1)

    assert len(audio_paths) == len(references), "Number of audio files must match number of phrases"
    print(f"Loaded {len(audio_paths)} audio files")
    for i, (ap, ref, bp) in enumerate(zip(audio_paths, references, boosting_phrases)):
        print(f"  [{i}] {ap}")
        print(f"      ref:  '{ref[:80]}{'...' if len(ref) > 80 else ''}'")
        print(f"      boost: '{bp}'")
    print()

    # Run tests
    all_results: list[TestResult] = []

    all_results.extend(
        test_effectiveness(model, audio_paths, references, boosting_phrases, device,
                           boosting_alpha=args.boosting_alpha,
                           beam_size=args.beam_size)
    )
    all_results.extend(
        test_independence(model, audio_paths, references, boosting_phrases, device,
                          boosting_alpha=args.boosting_alpha,
                          beam_size=args.beam_size)
    )

    # Word-level timestamps
    if args.timestamps:
        print("\n=== Word-Level Timestamps ===")
        tokenizer = model.tokenizer
        encoder_output, encoded_lengths = encode_audio(audio_paths, model, device)

        # Baseline timestamps
        baseline_decoder = make_beam_decoder(model, beam_size=args.beam_size, enable_per_stream_biasing=False)
        base_hyps = decode_with_beam_hypotheses(
            baseline_decoder, tokenizer, encoder_output, encoded_lengths,
        )
        base_hyps = compute_word_timestamps(model, base_hyps)

        # Boosted timestamps
        boosted_decoder = make_beam_decoder(model, beam_size=args.beam_size, enable_per_stream_biasing=True)
        multi_biasing_ids, biasing_requests = register_per_stream_biasing(
            boosted_decoder, tokenizer, boosting_phrases, device, alpha=args.boosting_alpha,
        )
        boost_hyps = decode_with_beam_hypotheses(
            boosted_decoder, tokenizer, encoder_output, encoded_lengths, multi_biasing_ids,
        )
        boost_hyps = compute_word_timestamps(model, boost_hyps)
        cleanup_biasing(boosted_decoder, biasing_requests)

        for i in range(len(audio_paths)):
            print(f"\n  [{i}] {Path(audio_paths[i]).name}")
            print(f"      Baseline transcript: {base_hyps[i].text}")
            print(f"      Word timestamps (baseline):")
            print(format_word_timestamps(base_hyps[i]))
            print(f"      Boosted transcript: {boost_hyps[i].text}")
            print(f"      Word timestamps (boosted):")
            print(format_word_timestamps(boost_hyps[i]))

    # Benchmark
    if args.benchmark:
        bench_results = benchmark_biasing(
            model, audio_paths, boosting_phrases, device,
            boosting_alpha=args.boosting_alpha,
            beam_size=args.beam_size,
            warmup=args.warmup,
            runs=args.runs,
        )
        for r in bench_results:
            print(f"  {r.name}: {r.mean_time_s * 1000:.1f} ± {r.std_time_s * 1000:.1f} ms, RTFx: {r.rtfx:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.detail}")
    print(f"\n{passed}/{total} tests passed")
    if passed == total:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
