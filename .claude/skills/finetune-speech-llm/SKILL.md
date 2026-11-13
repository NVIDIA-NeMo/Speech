---
name: finetune-speech-llm
description: Fine-tune a SALM speech-LLM checkpoint (speech encoder + projection + LLM decoder) with NeMo speechlm2 on a new domain or task, and evaluate it with vLLM. Use to adapt a speech-to-text LLM to new audio, improve WER on a domain corpus, train speech translation, spoken QA or audio classification, or debug a fine-tune that shows no improvement. Covers planning and cost, manifests, config derivation, preflight, training, LoRA export and merge, evaluation, diagnosis, checkpoint averaging, and ensembling.
---

# Fine-tuning a SALM speech LLM

Adapt a SALM checkpoint (speech encoder + modality projection + LLM decoder) to a new domain or task on one GPU, then
measure the result honestly. This file is the procedure. For background and the reasoning behind each check, see:

- `examples/speechlm2/finetuning/docs/fine-tuning-guide.md`: concepts, judgment calls, pitfalls.
- `examples/speechlm2/finetuning/docs/results.md`: measured results, cost per task, data efficiency, prompting vs
  fine-tuning.
- `examples/speechlm2/finetuning/docs/medical-asr-synthetic-data.md`: a worked synthetic-data example.
- `examples/speechlm2/finetuning/README.md`: the tools used below (paths below are relative to that directory).
- `examples/speechlm2/finetuning/sdg/`: synthetic-data generation scripts (LLM text, spoken form, TTS).
- `tutorials/speechlm2/finetuning/`: end-to-end notebooks (new-language ASR, translation, QA, classification, probes).

## The governing risk: silent failure

Most of this stack fails **silently**. Loaders are non-strict, so a config that describes a different architecture
than the checkpoint builds a model with randomly initialized submodules and no warning. An inference bug can drop the
LoRA adapters, so a good fine-tune serves at base quality. Every check below guards one of these failures.

| Symptom | Actual cause | Caught by |
|---|---|---|
| Evaluation simply scores badly | Config describes a different encoder; hundreds of tensors dropped at load, encoder is random | Step 6 preflight and restore log |
| Fine-tune scores exactly like the base model | Serving stack never merged the LoRA adapters | Step 7 merge and `lora_B` check |
| Training is slow; memory looks fine | `use_packed_sequence_sampling` unset; batches ~80% short of `batch_tokens` | Step 5 |
| `'val_loss' was not in top k` at every checkpoint | Validation never ran; `save_top_k` retains nothing | Step 5 |
| Target WER improves, general WER collapses | Training references use a different surface style than the model emits | Step 4 audit, Step 8 guard |
| Every number is ~1 WER worse, consistently | A decode-time optimization is not lossless | Step 8 exactness check |

None of these raise. Never report a WER without having (a) passed the coverage preflight, (b) checked the restore
log, and (c) confirmed the fine-tuned output actually differs from the baseline's.

## Step 1: Plan first (before any GPU work)

Given a task and a target ("20% better", "beat system X"), write a one-page plan and show it before running anything.

1. **Task as a measurement.** Input audio, output text, metric, and the exact test set. Grep the references for
   scoring markers (`IGNORE_TIME_SEGMENT_IN_SCORING`, `<unk>`, `((…))`). Find the published scoring code: it defines
   the normalization, and a leaderboard number is only comparable under the same protocol. If a published system is
   open (e.g. Whisper), plan to re-score it yourself to confirm the protocol.
2. **Prompt first.** It is free. Try 3-5 prompts that name the language, domain and output format, on dev *and* test.
   The baseline is the test score of the prompt chosen on dev; report the best prompt on test only as a labeled
   **test-oracle** upper bound. Prompts alone can move WER by up to 20 points and QA F1 by up to 17. A speech LLM may
   answer in the wrong script: on Indian-accented English it may write Devanagari or Tamil until the prompt says
   "in English, in the Latin alphabet" (medical WER 30.8 → 23.8). `prompt_search.py` automates the sweep in
   ~20 minutes per task. It pays off when the failure is output *format* (pronunciation-score correlation
   0.18 → 0.36 on test); on label-choice tasks its 2-3 point dev gains do not transfer to test.
3. **Data inventory.** Take the first branch that applies:
   - *Labeled in-domain training data exists.* Search the HF hub by task, language and domain words; check license
     and gating. Confirm it is not a mirror of the test set by comparing text hashes and n-gram overlap.
   - *Only a test set exists* (common for benchmarks). Combine **(a)** real neighboring data (same acoustics, i.e.
     accent and channel, or same vocabulary) with **(b)** synthetic data: an LLM writes text in the target's styles
     (entities, dictation, conversation) and several TTS voices speak it. Keep real speech in the mix so the model
     does not learn TTS artifacts. Write the text, then a **spoken form** ("HbA1c" → "H B A one C"): TTS reads the
     spoken form, the **label stays the written text**. Run `check_contamination.py --test test.json --train ...`
     against the test set before training. **Dev must include real in-domain speech** (≥ 100 clips; record or label
     them if necessary). A synthetic dev set cannot validate synthetic training data from the same text source: a
     catalog-derived dev set can reward catalog spellings and say −49% while test gets worse. Never select on test.
   - *Only a few examples* (≤ 50). Use them as style seeds for the LLM and as a sanity dev set, never as the test set.
4. **Expected cost** (one GB300 at ~$9/GPU-h; details per task in `results.md`):

   | Item | GPU-h | $ |
   |---|---:|---:|
   | One recipe iteration (baseline decodes, 300-500 steps, 2-4 exports, dev + test decodes) | 1.1-2.3 | 10-20 |
   | Exploration, when the first recipe works | 1-1.8× one iteration | |
   | Exploration, hard benchmarks (strong base model, convention-bound references) | 8-11× | 100-200 |
   | Synthetic data: LLM text 50k lines ≈ 0.4 GPU-h; TTS ≈ 1 GPU-h per 30 h audio (Kokoro), 5× slower (Magpie) | 1-3 | 10-30 |
   | Full medical case study including data generation | ~14 | ~125 |

   Decoding is about half the GPU time, and every exported candidate adds an engine start-up. Keep 2-3 candidates,
   not 10.
5. **Expected gain and risk.**
   - New task or label space (classification, QA, detection, regression, new language): +40% to +380% relative.
   - Accuracy already above ~85%: judge by error-rate reduction; expect 20-60%.
   - Conversational ASR whose remaining errors are annotation conventions (e.g. EdAcc): 5-20%, and a large share of it can come from the prompt alone.
   - Target above the published state of the art: say so, and propose another benchmark or accept a likely miss.

   State a probability of reaching the target and what would change it.

## Step 2: Establish the environment

```bash
# Which CUDA device is the compute GPU? CUDA order != nvidia-smi order.
python -c "import torch; print([(i, torch.cuda.get_device_properties(i).name) for i in range(torch.cuda.device_count())])"
# Does a GPU Conv1d work? (catches a broken cuDNN install)
python -c "import torch; torch.nn.Conv1d(192,1536,3,padding=1).cuda()(torch.randn(2,192,100,device='cuda'))"
# Is anything else holding the GPU?
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

Use **separate environments** for training (NeMo) and serving (vLLM); they pin different torch versions. Pin
**vLLM 0.28.0** (0.29 is incompatible with the SpeechLM plugin's processor). Pin `nvidia-cudnn-cu13==9.21.0.82` in
both and re-pin after any install. Export `VLLM_WORKER_MULTIPROC_METHOD=spawn`. Install commands: guide section 2.

## Step 3: Read the checkpoint, don't assume

```bash
python -c "
import json,struct,re,collections
f=open('<ckpt>/model.safetensors','rb'); n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n))
c=collections.Counter(re.sub(r'\.\d+\.','.N.',k) for k in h if k!='__metadata__')
[print(v,k) for k,v in sorted(c.items())]
"
cat <ckpt>/config.json
```

Note the encoder type (a `pe_encoder_config` block means a parallel-expert encoder whose parameters live under
`perception.encoder.asr_encoder.*`), any speculative-decoding head block, the `prompt_format`, and which LLM modules
are `nn.Linear`. Grouped 3-D expert tensors cannot take LoRA.

## Step 4: Build manifests

NeMo JSON, one object per line, absolute `audio_filepath`, optional `offset`/`duration`. **The file must end in
`.json`**: `.jsonl` is parsed as Lhotse and fails confusingly.

**Audit the reference surface style first** (guide section 4). Print references beside the base model's
predictions (`vllm_task_eval.py --limit 32`). Fine-tuning teaches surface form (case, punctuation, verbalized digits)
as surely as vocabulary, and normalizers hide the mismatch from the target metric; the damage only shows on the
general-domain guard. Fold references to the model's casing before training. Punctuation that was never annotated
cannot be restored, so an unpunctuated corpus still pulls the model toward unpunctuated output; blend in a cased,
punctuated source if that matters.

Compare the training pool's distribution against the evaluation set's on any metadata available (site, channel,
speaker, recording condition). If they differ, cap each group's contribution (e.g. 20 h per group, high-confidence
rows only) rather than training on the raw pool.

**Data order is a silent killer.** HF parquet exports are often sorted by label. A shard prefix can miss most classes
(10 of 35 keywords), and Lhotse's 10k-row shuffle buffer cannot fix a larger sorted manifest (a language-ID run
collapsed to one language). Sample uniformly across shards, shuffle on disk (`shuffle_manifest.py`), and assert every
test label occurs in train. A near-zero training loss in the first steps is a data bug.

## Step 5: Derive the config from the checkpoint

Never start from `examples/speechlm2/conf/salm_automodel.yaml`; it describes a different model.

```bash
python make_ft_config.py --checkpoint <ckpt> --out conf/ft.yaml \
  --train-manifest train.json --val-manifest dev.json --exp-dir <exp> \
  --train-encoder --train-proj --lr 1e-4 --max-steps 2000 --warmup-steps 200 --limit-train-batches 250
```

Confirm in the generated YAML: `tokenizer_path` points at the checkpoint root; `pe_encoder_config` is present if the
checkpoint had it; the speculative-decoding head block is unchanged; `automodel_backend.dispatcher: torch`;
`train_gate: false`; `ep_size: 1`; `pretrained_asr` is a non-null string (vLLM refuses an export without it, though
training never reads it).

**Two batching settings decide whether the run is fast and whether it validates at all.** Count batches per epoch
with `epoch_batches.py <manifest.json> <batch_tokens> --tokenizer <ckpt>`; do not estimate by hand (hand estimates
are easily ~40% off, in either direction). Rough guides:

```
batches_per_epoch  >~  dataset_hours * 3600 / 0.08 / batch_tokens   # audio tokens only; a lower bound
peak_GB            ~=  65 + 0.95 * batch_tokens / 1000               # 256 GB card
```

- `use_packed_sequence_sampling: true` is **required**. Without it the sampler measures batches as
  `batch_size * longest_example`; `batch_tokens: 48000` produced batches of 8,431 real tokens with nothing logged.
  With it, `packing_efficiency` is 1.00 and `sequence_length == batch_tokens`.
- `use_bucketing: false`. Packing already removes the padding bucketing exists to avoid; keeping it only correlates
  the gradient.
- `max_tokens` is a **filter**, not a batch cap: it silently drops longer examples. Leave it unset.
- `limit_train_batches` and `val_check_interval` must be **strictly below** batches per epoch (≤ 0.8×), or validation
  never runs.
- Size `batch_tokens` for throughput on a large corpus (~120k saturates a 256 GB card at ~15.6k tok/s). Under ~20 h,
  size it for *steps* instead: aim at 15-25 optimizer steps per epoch.
- Size `batch_tokens` on **total** tokens for non-ASR tasks: passages and label lists can outweigh the audio.

Default freeze policy: freeze everything; re-open the ASR encoder branch, the modality projection, and LoRA
(r=32, α=64) on the backbone's attention, recurrent-mixer and shared linear layers. Leave any auxiliary branch
(e.g. diarization), routers and routed experts frozen. Freeze the encoder when the shift is in the output, not the
audio (translation from a language the model already hears); train it when the label lives in the acoustics (new
language, emotion).

## Step 6: Preflight, smoke train, train

`run_training.py --config conf/ft.yaml [--keep-steps ...]` runs all three stages below with the checks built in. If you
run them by hand, apply the same checks.

**Preflight (never skip).** `check_checkpoint_coverage.py --checkpoint <ckpt> --config conf/ft.yaml`. Require
`matched == checkpoint tensors`, and the only unmatched model parameters must be the new LoRA adapters. It needs a
free GPU (~66 GB) and cannot run on CPU or the meta device.

**Smoke train** (20 steps, `limit_val_batches=2`, no checkpointing) with `examples/speechlm2/salm_train.py`. In the
log verify:

- the encoder mount message (`Mounted ParallelExpertEncoder from model.pe_encoder_config` when applicable);
- `N / M layers are restored (N exact, 0 skipped)` with **no** dropped-tensor warning;
- loss is finite;
- steady-state step time (step 1 includes compilation) and peak memory.

If memory leaves headroom, raise `batch_tokens` before adding gradient accumulation. Then launch the real run with the
same config.

**After the first validation interval, verify validation ran**: `val_loss` must appear in TensorBoard. If it does
not, every checkpoint logs `'val_loss' was not in top k` (which reads like "worse", not "missing") and `save_top_k`
keeps only `-last`. The cause is `limit_train_batches` exceeding what the dataloader yields per epoch. Switching to
`check_val_every_n_epoch` does **not** fix it; lower `limit_train_batches`. `run_training.py` stops a run with no
`val_loss` by 1.6× the validation interval.

Expect the *first* checkpoint of any run to log `was not in top k`: the callback reads metrics before that step's
validation, so `step=N.ckpt` is ranked on the metric from step `N - interval`.

Redirect the log **inside** any container (`bash -c '... > log 2>&1'`). With the redirect outside, killing the
launching shell leaves the trainer writing to a dead pipe and a healthy run looks hung. Track progress from the run's
event file, and use `py-spy dump --pid` to tell "stuck" from "mid-step".

**Disk and GPU sharing.** Each checkpoint and each export is tens of GB: delete checkpoints once merged, keep the best
two exports, and check free disk before starting a run. Training (≤ ~110 GB) can share a 256 GB GPU with a vLLM engine
at 0.40 memory utilization (retry engine start-up). Never edit a script a running job is executing.

## Step 7: Export and merge

```bash
python export_checkpoint.py --exp-dir <exp> --step 1250 --base-checkpoint <ckpt> --out <merged>
```

It wraps `examples/speechlm2/to_hf.py` and `merge_lora_checkpoint.py`. If you do it by hand:

- Use the run's resolved `<exp>/exp_config.yaml`, not the input YAML.
- **Always merge LoRA**; never rely on the serving stack to apply adapters.
- Hydra cannot parse `=` inside an override value, so `ckpt_path=.../step=1250.ckpt` fails with
  `mismatched input '='`. Symlink the checkpoint to an `=`-free name first.
- Verify the export carries trained weights: `lora_B` initializes to exactly zero, so a non-zero `lora_B` proves the
  training checkpoint was loaded.
- Restore `pretrained_asr` from the base checkpoint's config if it is missing.

## Step 8: Evaluate

```bash
python vllm_task_eval.py --model <merged> --task asr --manifest dev.json librispeech.json \
  --out eval/dev.jsonl eval/libri.jsonl --spec-k 2 --max-num-seqs 64 --normalizer <n>
```

**Prove every decode-time optimization is exact before trusting the harness.** Decode a few hundred utterances with
it on and off and diff the *hypotheses*, not just the scores; do this once per optimization. Known trap (guide
section 9): on hybrid state-space backbones, speculative decoding with `spec_k >= 2` plus prefix caching is exact only
with `--mamba-cache-mode all` (the script's default); the `align` mode silently costs ~1 WER. Disabling prefix caching
also fixes it, at a throughput cost.

Non-negotiables:

- `enable_thinking=false`. With thinking on, the model writes reasoning instead of transcribing (WER far above 100%).
- The evaluation prompt equals the training `context` string (`vllm_task_eval.py` reads the row's `context` by
  default).
- Report which normalizer produced each number; one system spanned 73%-88% WER on one set depending on the choice.
- Score a **general-domain guard** (e.g. LibriSpeech test-clean) alongside the target. A fine-tune that improves the
  domain while destroying general performance has replaced the distribution, not adapted it.
- Guard with the prompts users will send: a model trained on one fixed prompt may answer question-like audio when
  given *no* prompt. Vary prompt phrasing in training if no-prompt use matters.
- Tune on dev; touch test once.
- **Select checkpoints by decoding, not by `val_loss` or `val_acc`.** Neither tracks WER: `val_loss` can rank a
  checkpoint best that loses by 5.8 WER to a later one, and `val_acc` can tie across four checkpoints. Monitor
  `val_loss` only because it does not tie (a tying metric turns `save_top_k` into "keep the last k"). Do not
  early-stop on it either; run the schedule, keep a spread of checkpoints including `-last`, and decode them.
- Decoded WER and general-domain damage are **not monotonic** in training time; "stop earlier to be safe" is not
  reliable. Score several checkpoints.
- On long-form ASR, cap output length by audio duration (`--max-tokens-per-sec`) for base and fine-tune alike:
  greedy repetition loops on a handful of clips can swing WER by 5-15 points.
- When a gain concentrates in a few rows, read those rows.

## Step 9: When the result plateaus, diagnose before tuning

Do not reach for the next hyperparameter. Decode the **training set** first:

| Train WER | Dev WER | Diagnosis | What can work |
|---|---|---|---|
| ≈ 0 | high | Memorization; generalization gap | New unseen audio, ensembling, distillation on unseen audio |
| high | high | Underfitting | More capacity, higher LR, longer training |

If train WER is near zero, **self-training, confidence filtering, curriculum pruning and distillation on the training
set are all no-ops**: they re-teach labels the model already fits.

Levers for memorization, cheapest first:

1. **Average checkpoints** within one run (`average_checkpoints.py`, merged checkpoints only).
2. **Do not add trainable capacity.** On a small corpus more trainable parameters made results monotonically worse;
   full fine-tuning was worst.
3. **Add genuinely unseen in-domain audio**, relabeled with your own model rather than with its own references, or you
   teach two conflicting annotation conventions.
4. **Ensemble** (Step 10) if inference cost is acceptable.
5. **Distill the ensemble** into one model, on unseen audio only.

### Benchmarks with their own transcription convention

- **Learn the convention from dev references, then teach the model to follow instructions about it.** Scorers that
  delete punctuation make joining decisions count ("Piopar-MF" is one word, "Piopar MF" two; "500mg" vs "500 mg").
  Render each training label in **several stated conventions** (numbers, units, company suffixes, hyphens,
  separators), state the convention in the prompt, and at test time state the benchmark's convention. This was worth
  ~1.3 WER on a medical benchmark (Eka Care). Prompting a model fine-tuned *without* this does nothing.
- **Check a text source's conventions before training on it.** A catalog that writes "Ltd" 176k times teaches "Ltd"
  when the benchmark writes "Limited".
- **Measure vocabulary coverage per entity class.** Split entity hits by whether the term was in your training
  vocabulary. A fine-tune can beat a specialist where its lexicon covers the term and lose where it does not; then the
  fix is a bigger lexicon, not more training. Count *unique* entries: a "196k-product" catalog had 7,469.
- **Compare like for like.** Run open competitors through your own pipeline; published numbers may not reproduce
  (13.4 WER measured vs 10.9 published). Report single-pass systems against single-pass systems: retrieval, second
  passes and post-processing are product features, not benchmark wins.
- **Drop segments carrying scoring markers** (e.g. `IGNORE_TIME_SEGMENT_IN_SCORING`) from train, dev and test. If
  kept, a fine-tune learns to emit the marker while the base model transcribes those passages as insertions, and a
  −0.8% change reads as −15.4%.

## Step 10: Ensemble when one model is not enough

Combining hypotheses needs no training and is often the largest remaining gain. Diversity matters more than member
quality.

```bash
python make_tta.py --manifest dev.json --factor 0.95 --out-dir <dir> --out-manifest dev_tta_0.95.json
python rover_combine.py --hyps sysA.jsonl sysB.jsonl sysC.jsonl --out combined.jsonl --normalizer <n>
```

Assume these until disproved:

- **Three or more systems**, or voting cannot break ties.
- **Independently seeded runs combine better than checkpoints of one run.**
- **Members 1-2 WER worse than your best still help.**
- **Unweighted voting beats weighting**: up-weighting the strongest member suppresses the disagreement that makes
  voting work.
- **Report the single-model number alongside.** An N-way ensemble costs N decodes; it is a benchmark result, not a
  deployment.

To make it deployable, label held-out in-domain audio with the ensemble and fine-tune one student on those labels
blended with your real references.

## Step 11: Report

Give baseline and fine-tuned scores on the same sets with the same normalizer and prompt; absolute and relative
change; the general-domain guard; the inference settings (vLLM version, speculative `k`, batch); and the cost. If the
fine-tune shows no change, suspect the inference path before concluding training failed.

## Beyond ASR: translation, QA, classification, detection

The same procedure works for speech translation, spoken QA, intent and emotion classification, keyword spotting,
sound and language ID (see the notebooks and `results.md`). What changes:

- **Per-example prompts.** Put the prompt in each manifest row's `context` field (`set_context.py`). A fixed prompt
  attached via `tags` *overwrites* every row's `context`. Evaluate with `vllm_task_eval.py --task
  {bleu,squad,cls,...}`; metrics come from `task_metrics.py`. `prepare_benchmarks.py` builds manifests for common
  public benchmarks.
- **The baseline prompt is chosen on dev,** like every other free parameter, and compared with the fine-tune on
  test. Also print the **test-oracle** (the best prompt on test): the prompt chosen on dev can lose on test when the
  label prior shifts, and the test-oracle shows by how much.
- **Never reuse results across changed inputs.** The tools fingerprint every output (base checkpoint, manifests,
  config or decoder settings, code revision) and refuse a mismatch; after changing the checkpoint or recipe, use a new
  `SALM_FT_WORK`. Pass `--reuse` to `vllm_task_eval.py` / `average_checkpoints.py` / `vllm_label_scores.py` only to
  skip work whose fingerprint matches.
- **Map classification outputs leniently** (synonyms, `_` vs space) and read the baseline's confusion matrix and raw
  outputs before trusting it: a class it "never predicts" can be a scorer bug.
- **Check labels and splits against the official protocol** (e.g. SLURP's `intent` field is inconsistent, use
  `scenario_action`; CREMA-D's HF split leaks speakers).
- **Detection: score EER from label log-odds** (`--score-labels NEG POS`). Generated answers can be degenerate while
  log-probabilities still rank well. Exact scores need an engine without prefix caching (both scorers disable it): on
  a hybrid backbone, cached prefixes corrupt `prompt_logprobs` silently. Check that scores agree with generated labels.
- **When fine-tuning erases knowledge, probe the frozen model.** Score each label exactly with `vllm_label_scores.py`
  and fit a logistic regression on training clips with `label_probe.py` (music genre: weight fine-tunes fell below
  the prompted base model, while the probe beat it).
- **Pick targets below published SOTA.** If your target over a strong prompted baseline would exceed the literature, the
  benchmark tests nothing.

## Tools

All in `examples/speechlm2/finetuning/` (see its README):

- Data: `prepare_benchmarks.py`, `set_context.py`, `shuffle_manifest.py`, `subset_manifest.py`,
  `check_contamination.py`; synthetic data in `sdg/`.
- Config and training: `make_ft_config.py`, `epoch_batches.py`, `check_checkpoint_coverage.py`, `run_training.py`.
- Export: `export_checkpoint.py`, `merge_lora_checkpoint.py`, `average_checkpoints.py`.
- Evaluation: `vllm_task_eval.py`, `task_metrics.py`, `prompt_search.py`, `vllm_label_scores.py`, `label_probe.py`.
- Ensembling: `make_tta.py`, `rover_combine.py`.
