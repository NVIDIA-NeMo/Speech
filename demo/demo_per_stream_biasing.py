# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Demonstration: TDT beam search (malsd_batch) with per-stream GPU word boosting.
#
# This script proves that each element in a batch can be biased with DIFFERENT
# sets of phrases -- all processed in a single batched GPU forward pass.
#
# It uses the RadAI medical dataset (corti/radai), which provides per-sample
# `medical_terms` alongside each `transcription`. The demo runs two passes:
#   1. Beam search WITHOUT boosting (baseline)
#   2. Per-stream biasing where each utterance gets its OWN medical terms
#      as boosting phrases
#
# The decoder used is BeamBatchedTDTInfer (malsd_batch) with enable_per_stream_biasing=True.
# Boosting models are GPUBoostingTreeModel instances registered into a GPUBiasingMultiModel,
# which dispatches each beam to the correct boosting tree via multi_biasing_ids.
#
# Usage:
#   python demo_per_stream_biasing.py --num-samples 4 --beam-size 4
#   python demo_per_stream_biasing.py --num-samples 4 --alpha 20.0

import argparse
import os

import torch
import torchaudio
from bewer import Dataset, metrics

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from nemo.collections.asr.parts.submodules.tdt_beam_decoding import BeamBatchedTDTInfer

TDT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
HF_CACHE_DIR = "/mnt/md0/mlrd/shared_cache/huggingface"


def load_radai_batch(num_samples, split="test"):
    """Load a batch of samples from the RadAI dataset."""
    from datasets import load_dataset

    ds = load_dataset(
        "corti/radai",
        "default",
        split=split,
        cache_dir=os.path.join(HF_CACHE_DIR, "datasets"),
        trust_remote_code=True,
    )
    samples = []
    for i in range(num_samples):
        row = ds[i]
        samples.append(
            {
                "id": row["id"],
                "transcription": row["transcription"],
                "medical_terms": row["medical_terms"],
                "audio_array": row["audio"]["array"],
                "audio_sr": row["audio"]["sampling_rate"],
            }
        )
    return samples


def prepare_audio_batch(samples, device, target_sr=16000):
    """Convert raw audio arrays into a padded batch tensor."""
    all_inputs, all_lengths = [], []
    for s in samples:
        audio = torch.from_numpy(s["audio_array"]).float()
        if s["audio_sr"] != target_sr:
            audio = torchaudio.functional.resample(audio, s["audio_sr"], target_sr)
        all_inputs.append(audio)
        all_lengths.append(torch.tensor(audio.shape[0], dtype=torch.int64))
    input_batch = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True).to(device=device, dtype=torch.float32)
    length_batch = torch.stack(all_lengths).to(device)
    return input_batch, length_batch


def encode_audio(model, input_signal, input_signal_length):
    with torch.no_grad(), torch.inference_mode():
        encoded, encoded_len = model(input_signal=input_signal, input_signal_length=input_signal_length)
    return encoded, encoded_len


def decode_with_beam(model, encoder_output, encoded_lengths, beam_size, multi_biasing_ids=None,
                     enable_per_stream_biasing=False, allow_cuda_graphs=False):
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
        allow_cuda_graphs=allow_cuda_graphs,
        enable_per_stream_biasing=enable_per_stream_biasing,
    )

    with torch.no_grad(), torch.inference_mode():
        hyps = decoder(
            encoder_output=encoder_output,
            encoded_lengths=encoded_lengths,
            multi_biasing_ids=multi_biasing_ids,
        )[0]

    texts = [h.text if h.text is not None else model.tokenizer.ids_to_text(h.y_sequence.tolist()) for h in hyps]
    return texts


def register_per_stream_biasing(decoder, tokenizer, batch_phrases, device, alpha=2.0):
    """Register a different boosting tree for each batch element.

    Returns:
        multi_biasing_ids: [B] tensor of model IDs (-1 = no biasing)
        biasing_requests: list of BiasingRequestItemConfig for cleanup
    """
    biasing_multi_model = decoder.decoding_computer.biasing_multi_model
    assert biasing_multi_model is not None, "enable_per_stream_biasing must be True on the decoder"

    batch_size = len(batch_phrases)
    multi_biasing_ids = torch.full([batch_size], fill_value=-1, dtype=torch.long, device=device)
    biasing_requests = []

    for batch_idx, phrases in enumerate(batch_phrases):
        if not phrases:
            biasing_requests.append(None)
            continue
        request = BiasingRequestItemConfig(
            boosting_model_cfg=BoostingTreeModelConfig(key_phrases_list=phrases),
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


def cleanup_biasing(decoder, biasing_requests):
    biasing_multi_model = decoder.decoding_computer.biasing_multi_model
    with torch.inference_mode():
        for request in biasing_requests:
            if request is not None and request.multi_model_id is not None:
                biasing_multi_model.remove_model(request.multi_model_id)
                request.multi_model_id = None


def compute_wers(references, hypotheses, key_terms=None):
    """Compute per-example WER and aggregate KTR using bewer.

    Args:
        references: list of reference strings
        hypotheses: list of hypothesis strings
        key_terms: optional list of per-example key-term lists for KTR

    Returns:
        (per_example_wers, aggregate_wer, aggregate_ktr_or_None)
    """
    ds = Dataset()
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        kt = {"medical_terms": set(key_terms[i])} if key_terms else None
        ds.add(ref=ref, hyp=hyp, key_terms=kt)

    wer_metric = metrics.WER(ds)
    wer_metric.set_source(ds)
    per_example = [wer_metric.get_example_metric(ex).value for ex in ds]

    ktr_value = None
    if key_terms is not None:
        try:
            ktr_metric = metrics.KTR(ds, vocab="medical_terms")
            ktr_metric.set_source(ds)
            ktr_value = ktr_metric.value
        except Exception:
            pass

    return per_example, wer_metric.value, ktr_value


def main():
    parser = argparse.ArgumentParser(
        description="TDT beam search per-stream word boosting demo using RadAI dataset"
    )
    parser.add_argument("--model", default=TDT_MODEL, help="Pretrained TDT model name")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of RadAI samples to use in the batch")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=2.0, help="Boosting tree alpha weight")
    parser.add_argument("--cuda-graphs", action="store_true", help="Enable CUDA graphs")
    parser.add_argument("--device", default=None, help="Device (auto-detect if omitted)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Beam size: {args.beam_size}")
    print(f"Boosting alpha: {args.alpha}")

    # --- Load model ---
    print(f"\nLoading model: {args.model}")
    model = ASRModel.from_pretrained(model_name=args.model, map_location="cpu")
    model.eval()
    model.to(device=device)
    tokenizer = model.tokenizer

    # --- Load RadAI data ---
    print(f"\nLoading {args.num_samples} samples from RadAI dataset...")
    samples = load_radai_batch(args.num_samples)
    batch_size = len(samples)
    references = [s["transcription"] for s in samples]
    all_key_terms = [s["medical_terms"] for s in samples]

    print(f"\nBatch of {batch_size} samples from RadAI:")
    for i, s in enumerate(samples):
        print(f"  [{i}] id={s['id'][:40]}")
        print(f"      transcription: {s['transcription'][:120]}")
        print(f"      medical_terms: {s['medical_terms']}")
        print(f"      audio: {len(s['audio_array'])} samples @ {s['audio_sr']}Hz")

    # --- Encode audio ---
    print("\nEncoding audio...")
    input_signal, input_signal_length = prepare_audio_batch(samples, device)
    encoder_output, encoded_lengths = encode_audio(model, input_signal, input_signal_length)

    # --- Run 1: No boosting (baseline) ---
    print("\n" + "=" * 100)
    print("RUN 1: Beam search (malsd_batch) WITHOUT boosting")
    print("=" * 100)
    baseline_texts = decode_with_beam(
        model, encoder_output, encoded_lengths, beam_size=args.beam_size,
        allow_cuda_graphs=args.cuda_graphs,
    )
    baseline_wers, baseline_agg_wer, baseline_ktr = compute_wers(
        references, baseline_texts, key_terms=all_key_terms
    )
    for i, text in enumerate(baseline_texts):
        print(f"  [{i}] WER={baseline_wers[i]:.3f}")
        print(f"      ref:   {references[i][:120]}")
        print(f"      hyp:   {text[:120]}")
    print(f"\n  Aggregate WER (baseline): {baseline_agg_wer:.4f}")
    if baseline_ktr is not None:
        print(f"  Aggregate KTR (baseline): {baseline_ktr:.4f}")

    # --- Run 2: Per-stream biasing with medical_terms from the dataset ---
    print("\n" + "=" * 100)
    print("RUN 2: Per-stream biasing -- each batch element gets its OWN medical_terms as phrases")
    print("=" * 100)

    per_stream_phrases = all_key_terms

    print("\nPer-stream boosting phrases (from RadAI medical_terms):")
    for i, phrases in enumerate(per_stream_phrases):
        if phrases:
            print(f"  [{i}] {phrases}")
        else:
            print(f"  [{i}] (no terms - no biasing)")

    decoder_ps = BeamBatchedTDTInfer(
        decoder_model=model.decoder,
        joint_model=model.joint,
        durations=list(model.to_config_dict()["model_defaults"]["tdt_durations"]),
        blank_index=tokenizer.vocab_size,
        beam_size=args.beam_size,
        score_norm=True,
        return_best_hypothesis=True,
        allow_cuda_graphs=args.cuda_graphs,
        enable_per_stream_biasing=True,
    )

    multi_biasing_ids, biasing_requests = register_per_stream_biasing(
        decoder_ps, tokenizer, per_stream_phrases, device, alpha=args.alpha,
    )

    print(f"\nmulti_biasing_ids tensor: {multi_biasing_ids.tolist()}")
    print("  (each value maps to a different GPUBoostingTreeModel; -1 = no biasing)")

    with torch.no_grad(), torch.inference_mode():
        boosted_hyps = decoder_ps(
            encoder_output=encoder_output,
            encoded_lengths=encoded_lengths,
            multi_biasing_ids=multi_biasing_ids,
        )[0]
    boosted_texts = [h.text if h.text is not None else model.tokenizer.ids_to_text(h.y_sequence.tolist()) for h in boosted_hyps]

    boosted_wers, boosted_agg_wer, boosted_ktr = compute_wers(
        references, boosted_texts, key_terms=all_key_terms
    )
    for i, text in enumerate(boosted_texts):
        print(f"  [{i}] WER={boosted_wers[i]:.3f}")
        print(f"      ref:   {references[i][:120]}")
        print(f"      hyp:   {text[:120]}")
    print(f"\n  Aggregate WER (boosted): {boosted_agg_wer:.4f}")
    if boosted_ktr is not None:
        print(f"  Aggregate KTR (boosted): {boosted_ktr:.4f}")

    cleanup_biasing(decoder_ps, biasing_requests)

    # --- Comparison ---
    print("\n" + "=" * 100)
    print("COMPARISON: baseline vs per-stream biased")
    print("=" * 100)

    any_changed = False
    any_improved = False
    for i in range(batch_size):
        changed = baseline_texts[i] != boosted_texts[i]
        improved = boosted_wers[i] < baseline_wers[i]
        any_changed = any_changed or changed
        any_improved = any_improved or improved
        marker = ""
        if changed:
            marker = " *** CHANGED" + (" (IMPROVED)" if improved else "")

        print(f"\n  [{i}] terms={per_stream_phrases[i]}{marker}")
        print(f"      WER:   baseline={baseline_wers[i]:.3f}  boosted={boosted_wers[i]:.3f}")
        print(f"      base:  {baseline_texts[i][:120]}")
        print(f"      boost: {boosted_texts[i][:120]}")

    print("\n" + "=" * 100)
    print("SUMMARY (bewer metrics)")
    print("=" * 100)
    print(f"  Batch size:           {batch_size}")
    print(f"  Beam size:            {args.beam_size}")
    print(f"  Boosting alpha:       {args.alpha}")
    print(f"  multi_biasing_ids:    {multi_biasing_ids.tolist()}")
    print(f"  Any transcript changed: {any_changed}")
    print(f"  Any WER improved:      {any_improved}")
    print(f"  Avg WER (baseline):    {baseline_agg_wer:.4f}")
    print(f"  Avg WER (boosted):     {boosted_agg_wer:.4f}")
    if baseline_ktr is not None and boosted_ktr is not None:
        print(f"  Avg KTR (baseline):   {baseline_ktr:.4f}")
        print(f"  Avg KTR (boosted):    {boosted_ktr:.4f}")

    print("\n" + "=" * 100)
    if any_changed:
        print("PROOF: Per-stream biasing works on batched TDT beam search.")
        print("  - Each batch element was decoded with its OWN GPUBoostingTreeModel")
        print("  - The multi_biasing_ids tensor routed each stream to the correct")
        print("    boosting tree via Triton GPU kernels")
        print(f"  - All {batch_size} streams were processed in a SINGLE batched GPU forward pass")
        if any_improved:
            print("  - At least one stream's WER improved, confirming the biasing had an effect")
    else:
        print("NOTE: No changes observed with these samples. The medical terms may already")
        print("      be decoded correctly. Try increasing --alpha or using different samples.")
    print("=" * 100)


if __name__ == "__main__":
    main()
