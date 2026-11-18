# Quantizing SpeechLM checkpoints for vLLM (FP8 and NVFP4)

These scripts quantize a NeMo SpeechLM checkpoint whose language model is a NemotronH decoder (hybrid Mamba and
MoE), for serving with vLLM and the NeMo SpeechLM plugin. The result is one checkpoint whose `config.json` carries a
ModelOpt `quantization_config`; vLLM builds its quantized layers from it, and the plugin builds the FP8 audio
encoder from the same config.

The documentation page `docs/source/speechlm2/quantization.rst` covers the same recipe with background on how the
pipeline works, validation advice and limitations.

## What the recipes quantize

| Part | `fp8` | `nvfp4` |
|---|---|---|
| Decoder: routed and shared MoE experts, attention projections | FP8 | NVFP4 |
| Decoder: Mamba `in_proj` and `out_proj` | FP8 | FP8 |
| Decoder: `lm_head`, Mamba `conv1d`, MoE routers | BF16 | BF16 |
| KV cache | FP8 with calibrated scales | FP8 at serving time (`--kv-cache-dtype fp8`) |
| Audio encoder: `attn.w_qkv`, `attn.out_proj`, `ffn.net.0`, `ffn.net.3` of each layer | FP8 | FP8 |
| MTP draft head (its router stays BF16) | FP8 | FP8 |
| CTC timestamp head | BF16 | BF16 |

FP8 layers have a per-tensor weight scale and a static per-tensor input scale. NVFP4 layers use ModelOpt's NVFP4
format: 4-bit values in blocks of 16 with an FP8 scale per block and a per-tensor scale. Mamba's `A_log`, `D` and
`dt_bias` keep the FP32 values of the source checkpoint.

Expect FP8 to match BF16 within run-to-run noise. NVFP4 trades a small accuracy cost, larger for some languages
than for English, for more throughput. Measure both on your own test sets against BF16 served by the same NeMo build.

## Requirements

- A GPU with at least 80 GB of memory to quantize the decoder, which loads in BF16. Serving FP8 needs SM89 or newer;
  serving NVFP4 needs Blackwell (SM100 or SM120).
- A ModelOpt build that includes NVIDIA/Model-Optimizer#2508 (tested with main at `54d44161e`). Earlier builds write
  module names that vLLM cannot resolve, and `assemble_checkpoint.py` refuses their exports.
- `transformers` with NemotronH support, `torch`, `safetensors` and `datasets`.
- The default FP8 calibration reads the gated `nvidia/Nemotron-Post-Training-Dataset-v2`. Log in to Hugging Face
  with access to it, or pass `--calib-dataset cnn_dailymail`.
- vLLM 0.32 and NeMo with the SpeechLM plugin's quantized serving (NVIDIA-NeMo/Speech#16341), for calibration and
  serving. Serving the CTC timestamp head also needs NVIDIA-NeMo/Speech#16344; without it, assemble with
  `--skip-ctc-head`.
- Working space on top of the source checkpoint. Keeping the BF16 decoder export, the quantized decoder and the
  assembled checkpoint at the same time takes about 2 times the size of the BF16 checkpoint for FP8 and 1.5 times for
  NVFP4. Step 3 rewrites every decoder shard that holds Mamba parameters (for the decoders we tested, all of them), so
  `--hardlink` saves little. The BF16 decoder export can go after step 2, and the quantized decoder after step 3.
  Step 5 hard-links unchanged files into `$OUT`; if `$OUT` is on another filesystem than `$WORK`, it copies them
  instead, so budget one more copy of the final checkpoint.

## Steps

```bash
SRC=/models/speechlm-bf16        # BF16 SpeechLM checkpoint
WORK=/work/quant
RECIPE=nvfp4                     # or fp8
OUT=/models/speechlm-$RECIPE
```

1. Export the decoder as a Hugging Face NemotronH checkpoint, the input ModelOpt needs:
   ```bash
   python export_llm_to_hf.py --checkpoint $SRC --output $WORK/llm_bf16
   ```
2. Quantize the decoder with ModelOpt (128 calibration samples of up to 512 tokens):
   ```bash
   python quantize_decoder.py --llm $WORK/llm_bf16 --recipe $RECIPE --output $WORK/llm_$RECIPE
   ```
3. Assemble a SpeechLM checkpoint with the quantized decoder. The audio encoder and the MTP head are still BF16,
   and vLLM can already serve this checkpoint:
   ```bash
   python assemble_checkpoint.py --source $SRC --quantized-llm $WORK/llm_$RECIPE --output $WORK/ckpt_bf16_heads --hardlink
   ```
4. Calibrate the audio encoder and the MTP head by serving the step 3 checkpoint with vLLM and transcribing held-out
   audio with MTP speculative decoding on. Add `--kv-cache-dtype fp8` for NVFP4:
   ```bash
   VLLM_PLUGINS=nemo_speechlm VLLM_WORKER_MULTIPROC_METHOD=spawn \
   python calibrate_activations.py --checkpoint $WORK/ckpt_bf16_heads --manifest calib.jsonl --output $WORK/ranges.json
   ```
   `calib.jsonl` holds one `{"audio_filepath": ...}` per line. Use a few hundred held-out clips that resemble your
   traffic, never your test sets.
5. Write the FP8 audio encoder and the FP8 MTP head into the final checkpoint (unchanged files are hard-linked, or
   copied across filesystems):
   ```bash
   python quantize_encoder_mtp_fp8.py --checkpoint $WORK/ckpt_bf16_heads --activation-ranges $WORK/ranges.json --output $OUT
   ```

Step 2 needs ModelOpt and `transformers`, and step 4 needs vLLM and NeMo. Steps 1 and 5 need only `torch` and
`safetensors`, and step 3 only Python. A script replaces an existing output only when it offers `--overwrite` and
you pass it, and refuses an output that is, contains or lies inside one of its inputs.

## Serve

```bash
VLLM_PLUGINS=nemo_speechlm VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve $OUT --trust-remote-code --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Use `--kv-cache-dtype fp8` for NVFP4 checkpoints; FP8 checkpoints carry calibrated KV cache scales and serve with
`auto`. For word timestamps, add
`--middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware --trust-request-mm-kwargs`.
Requests opt in through `mm_processor_kwargs`, which vLLM rejects without `--trust-request-mm-kwargs`; set it only
when clients are trusted. On SM12x GPUs with cuDNN older than 9.23.1, set
`VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`; the FlashInfer FP8 GEMM can crash partway through a run
there.

The server log should show `FP8 audio encoder: N Linears built from quantization_config` and, for NVFP4,
`Using fp8 data type to store kv cache`.

## Checks built into the scripts

- Copied tensors (perception weights, MTP and CTC heads, FP32 Mamba parameters) are verified against the source by
  SHA-256.
- `assemble_checkpoint.py` refuses ModelOpt exports with unresolved module names.
- `quantize_encoder_mtp_fp8.py` requires a calibrated range for every encoder Linear it quantizes, and fails unless
  every FP8 weight has scalar FP32 scales and every perception Linear is FP8 exactly when the config says so.
- Unit tests, from the repository root: `python -m pytest --noconftest tests/collections/speechlm2/test_quantization_scripts.py`.
  `--noconftest` skips the repository-wide test setup, which downloads test data these tests do not need.

The encoder targets assume the Transformer ASR encoder these SpeechLMs use; another encoder needs its own target list
in `calibrate_activations.py` and `quantize_encoder_mtp_fp8.py`. Calibration always runs MTP speculative decoding, so
the checkpoint needs an MTP draft head (`llm.mtp.*` tensors) that vLLM can load.
