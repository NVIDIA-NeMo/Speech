# NeMo Streaming ASR Evaluation with Config Support

## Overview

The `run_nemo_streaming_eval.py` script now supports loading configuration from YAML files to match the parameters used in NeMo's standard streaming inference examples.

## Usage

### Option 1: Use Config File (Recommended)

Use a config file that matches the official NeMo streaming configs:

```bash
python run_nemo_streaming_eval.py \
    --config conf/asr_streaming_inference/cache_aware_rnnt.yaml \
    --model_path=/path/to/model.nemo \
    --dataset=librispeech \
    --split=test.clean \
    --batch_size=16
```

Or use the example config:

```bash
python run_nemo_streaming_eval.py \
    --config conf/streaming_eval_example.yaml \
    --model_path=/path/to/model.nemo \
    --dataset=librispeech \
    --split=test.clean \
    --batch_size=16
```

### Option 2: Command-line Arguments Only

Continue using command-line arguments as before:

```bash
python run_nemo_streaming_eval.py \
    --model_path=/path/to/model.nemo \
    --dataset=librispeech \
    --split=test.clean \
    --chunk_size=32 \
    --left_chunks=2 \
    --batch_size=16
```

## Config Parameters

When using a config file, the following parameters are extracted:

### From `asr` section:
- `device_id`: GPU device number (overrides `--cuda`)

### From `streaming` section:
- `att_context_size`: [total_chunks, lookahead_chunks] - Used to calculate `left_chunks`
  - Example: `[70, 13]` means total 70 chunks with 13 lookahead chunks
  - `left_chunks = att_context_size[0] // chunk_size`
- `chunk_size_in_secs`: Audio chunk duration in seconds
  - If specified, calculates `chunk_size` automatically
  - Example: `0.08` for 80ms chunks (common for FastConformer)
- `sample_rate`: Audio sample rate for calculations

## Priority Order

Command-line arguments take priority over config file values:

1. Explicitly provided CLI args (e.g., `--chunk_size=64`)
2. Config file values (e.g., `streaming.chunk_size_in_secs`)
3. Model defaults (from `model.encoder.streaming_cfg`)

## Examples

### Use official cache-aware RNNT config:
```bash
python run_nemo_streaming_eval.py \
    --config examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml \
    --model_path=stt_en_fastconformer_hybrid_large_streaming_multi \
    --dataset=librispeech \
    --split=test.clean
```

### Override config values with CLI args:
```bash
python run_nemo_streaming_eval.py \
    --config conf/streaming_eval_example.yaml \
    --model_path=/path/to/model.nemo \
    --dataset=librispeech \
    --chunk_size=64 \
    --left_chunks=4
```

### Evaluate with different attention contexts:

**Standard latency (att_context_size: [70, 13])**:
```bash
# Modify config to use att_context_size: [70, 13]
python run_nemo_streaming_eval.py --config conf/streaming_standard.yaml ...
```

**Lower latency (att_context_size: [70, 6])**:
```bash
# Modify config to use att_context_size: [70, 6]
python run_nemo_streaming_eval.py --config conf/streaming_low_latency.yaml ...
```

## Config Alignment

The config file format matches the official NeMo streaming inference configs:
- `examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml`
- `examples/asr/conf/asr_streaming_inference/cache_aware_ctc.yaml`

This ensures consistency between evaluation and production inference settings.
