---
name: quantize-speech-llm
description: Quantize a SALM speech-LLM checkpoint with a NemotronH decoder to FP8 or NVFP4 for vLLM serving with the NeMo SpeechLM plugin, including the audio encoder and the MTP draft head, and verify the result. Use to produce a quantized checkpoint, check one before sharing it, debug a quantized checkpoint that loads but scores or drafts badly, or choose between FP8 and NVFP4.
---

# Quantizing a SALM speech LLM for vLLM

Produce an FP8 or NVFP4 checkpoint that vLLM serves with the NeMo SpeechLM plugin, then prove it is as good as
claimed. This file is the procedure and its checks. The commands, requirements and serving flags are in
`scripts/speechlm2/quantization/README.md`; the scripts named below live in that directory.

## The governing risk: silent failure

A quantized checkpoint usually loads and answers even when part of it is wrong. vLLM skips its missing-weight check
for quantized models, and a module the quantization config does not describe simply runs in another format.

| Symptom | Actual cause | Caught by |
|---|---|---|
| MTP draft acceptance near zero, WER looks fine | The checkpoint has no MTP head: an LLM-only export dropped `llm.mtp.*` | `assemble_checkpoint.py` copies the head; the plugin refuses MTP without it; Step 6 |
| NVFP4 runs far slower than expected | ModelOpt without NVIDIA/Model-Optimizer#2508 wrote unresolvable module names, so vLLM falls back to another kernel path | `assemble_checkpoint.py` refuses such exports |
| NVFP4 KV cache silently BF16 | The NVFP4 export declares no KV cache scheme, and `--kv-cache-dtype auto` keeps BF16 | Serve with `--kv-cache-dtype fp8`; Step 6 log check |
| Audio encoder silently BF16 | Step 5 skipped, or a plugin build without NVIDIA-NeMo/Speech#16341 | Step 6 log line `FP8 audio encoder: N Linears` |
| FP8 activation scales differ from a reference build | Different calibration data or padding side (`hf_ptq.py` pads left) | `quantize_decoder.py` loads calibration data as `hf_ptq.py` does; reproduction check below |
| Mamba numerics differ slightly from the BF16 release | ModelOpt exports the FP32 `A_log`, `D` and `dt_bias` rounded to BF16 | `assemble_checkpoint.py` restores the FP32 values |
| Quantized model looks worse or better than it is | Baseline served by another NeMo build, prompt or decoding setup, or a single noisy run | Step 7 |

The one loud failure: a plugin build without NVIDIA-NeMo/Speech#16344 stops at startup with "no module or parameter
named 'ctc_timestamp'". Assemble with `--skip-ctc-head` for such builds, or serve with #16344.

## Step 1: Plan

1. **Choose the recipe.** `fp8` keeps accuracy at BF16 level and serves on SM89 or newer. `nvfp4` gives more
   throughput, needs Blackwell, and costs some accuracy, more on some languages than on English.
2. **Check the requirements** in the README: a ModelOpt build with #2508, a GPU with at least 80 GB to quantize,
   access to the gated calibration dataset for the `fp8` default, vLLM 0.28 and NeMo with #16341 (and #16344 for
   word timestamps), and the disk space.
3. **Pick calibration audio**: a few hundred held-out clips that resemble the traffic. Never use test sets.
4. **Fix the baseline before measuring anything**: the BF16 checkpoint served by the same NeMo build, with the same
   prompts, decoding settings and test sets the quantized model will get.

## Step 2: Quantize the decoder

Run `export_llm_to_hf.py`, then `quantize_decoder.py --recipe fp8|nvfp4`. The quantized export's `config.json` must
hold a `quantization_config`. `quantize_decoder.py` reports how many quantizers calibration never reached; it fills
them with the largest range seen, but a large count means the calibration texts did not reach many MoE experts.

## Step 3: Assemble

Run `assemble_checkpoint.py --hardlink`. Its output must report the FP32 Mamba parameters it restored (three per
Mamba layer) and the MTP and CTC head tensors it copied. This checkpoint already serves: decoder quantized, encoder
and MTP head BF16.

## Step 4: Calibrate the encoder and the MTP head

Run `calibrate_activations.py` on the Step 3 checkpoint, with `--kv-cache-dtype fp8` for NVFP4. The draft head reads
the target model's hidden states, so calibrate the checkpoint you will serve, not the BF16 one. It must report
ranges for every encoder target and for the MTP layers; no MTP ranges means speculative decoding did not run.

## Step 5: Write the FP8 encoder and MTP head

Run `quantize_encoder_mtp_fp8.py`. It refuses encoder Linears without a calibrated range and checks that every FP8
weight has scalar scales and every perception Linear matches the config.

## Step 6: Serve and smoke-test

Serve with the flags in the README and check the startup log: `FP8 audio encoder: N Linears built from
quantization_config`, and for NVFP4 `Using fp8 data type to store kv cache`. Send a few real requests with MTP on and
compare the draft acceptance in the `SpecDecoding` log with the BF16 checkpoint's on the same audio. With #16344 and
the CTC middleware, check that timestamp requests come back with `error: null`.

## Step 7: Evaluate honestly

- Compare against the Step 1 baseline only; a different NeMo build, prompt or decoding setup can move WER more than
  quantization does.
- Report per-dataset and per-language deltas, not only averages: quantization costs concentrate in a few languages.
- Serve each checkpoint at least twice to learn the run-to-run noise of batched serving before calling a delta real;
  quantized checkpoints can vary more between runs than BF16.
- Measure throughput on otherwise idle GPUs; parallel jobs perturb it.

## Reproducing a reference build

`quantize_decoder.py` has rebuilt reference decoders bit for bit when the ModelOpt build, the calibration set, the
PyTorch and `transformers` versions and the GPU type all matched; a change in any of them can move the scales. Under
those conditions, compare the decoder tensors by hash to confirm a rebuild: weights and scales should be
bit-identical. A fresh calibration of the encoder and the MTP head changes their input scales slightly; compare those
by quality, not bytes.
