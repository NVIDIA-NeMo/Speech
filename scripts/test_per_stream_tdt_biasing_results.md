# Per-Stream Keyphrase Biasing for Parakeet TDT

## Overview

Per-stream keyphrase biasing via beam search is available for Parakeet TDT models on the `corticph/NeMo` branch (PR #3). The feature flows through `BeamBatchedTDTInfer` -> `ModifiedALSDBatchedTDTComputer` -> `GPUBiasingMultiModel`, controlled by the config flag `enable_per_stream_biasing`.

This script (`scripts/test_per_stream_tdt_biasing.py`) verifies:

1. **Effectiveness** -- boosting with domain-specific phrases improves WER.
2. **Per-stream independence** -- each batch element's biasing is independent; no cross-stream leakage.
3. **Cross-contamination** -- changing stream A's biasing does not affect stream B's output.
4. **Inference speed impact** -- measures decode-only overhead from biasing and timestamp computation.

## Test Data

Six audio samples from the `corti/radai` dataset (medical dictation). Each sample has a `medical_terms` column providing domain-specific phrases for the boosting trie.

**Model:** `nvidia/parakeet-tdt-0.6b-v2`

## Configuration

| Parameter | Value |
|---|---|
| Beam size | 4 |
| Boosting alpha | 2.0 |
| `unk_score` | 0.0 (default) |
| `enable_per_stream_biasing` | True |

The `unk_score` default of 0.0 is critical. Setting it to -100 (as seen in some examples) penalizes non-boosted tokens so heavily that the decoder emits only boosted phrases, destroying surrounding context.

## Results

### Effectiveness

| Metric | Baseline | Boosted |
|---|---|---|
| WER | 0.5833 | 0.3796 |

All 6 transcripts changed with biasing. Boosting corrected targeted medical terms (e.g., "hypronized" -> "heparinized", "bow gas" -> "bowel gas", "C45" -> "C four five") while preserving surrounding context.

### Per-Stream Independence

Batch decode of `[A, B]` (each with its own boosting trie) matches solo decode for both streams. Changing A's biasing to an unrelated phrase ("supercalifragilisticexpialidocious") does not affect B's output. No cross-stream leakage.

### Inference Speed

Decode-only timing (encoder output precomputed and reused), 3 warmup + 10 measured iterations, with CUDA synchronization.

| Config | Decode time | RTFx | Overhead |
|---|---|---|---|
| Baseline (no biasing) | 333.5 +/- 17.1 ms | 127.20x | -- |
| Per-stream biasing | 406.8 +/- 0.5 ms | 104.29x | +22.0% |
| Biasing + word timestamps | 417.2 +/- 0.6 ms | 101.67x | +25.1% |

- Biasing adds **~22%** decode overhead (~73 ms).
- Word-level timestamp computation adds only **~2.6%** on top of biasing (~11 ms).
- Even with biasing + timestamps, RTFx remains above 100x.

## Key Implementation Notes

- The script creates **standalone** `BeamBatchedTDTInfer` decoders (passing `model.decoder`/`model.joint` which are already on GPU) rather than using `model.change_decoding_strategy()`. The latter constructs `biasing_multi_model` on CPU after `model.to(device)`, causing CUDA illegal memory access in the Triton kernel.
- The standard `transcribe()` path raises `NotImplementedError` for `partial_hypotheses` with beam search. The script calls `decoder(...)` directly with `multi_biasing_ids`.
- Word-level timestamps use `model.decoding.compute_rnnt_timestamps()` (dispatches to `_compute_offsets_tdt` for TDT) + `process_timestamp_outputs()` to convert token-level frames to seconds.

## Usage

```bash
# Full test suite with benchmark and timestamps
python scripts/test_per_stream_tdt_biasing.py \
    --manifest manifest.json \
    --model nvidia/parakeet-tdt-0.6b-v2 \
    --boosting_alpha 2.0 \
    --timestamps --benchmark --warmup 3 --runs 10

# Tests only (no benchmark)
python scripts/test_per_stream_tdt_biasing.py \
    --manifest manifest.json \
    --model nvidia/parakeet-tdt-0.6b-v2
```

The manifest JSONL must have `audio_filepath`, `text`, and `medical_terms` fields. The `medical_terms` field is a list of phrases used for per-stream boosting.
