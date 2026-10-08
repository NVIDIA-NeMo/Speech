# Fine-tuning a SALM speech LLM with NeMo and serving it with vLLM

This guide covers adapting a SALM checkpoint (speech encoder, modality projection, and LLM backbone) to a new
domain or task on a single GPU. Training uses NeMo Speech's `speechlm2` collection, and evaluation and serving use
vLLM. The guide gives the mechanics of the stack, the checks that catch its silent failures, and the judgment calls
that decide whether a fine-tune actually helps.

The most expensive mistake available here is not a bad hyperparameter. It is **trusting a number that was never
true**: the decode path was lossy, the baseline was quoted rather than measured, or the split you tuned on was the
split you reported. Much of the advice below exists to prevent that.

## 1. Who this is for

This guide is for practitioners who have a SALM checkpoint exported by `SALMAutomodel` and want to adapt it to a new
acoustic domain, vocabulary, language, or speech-to-text task (translation, spoken QA, classification, scoring). It
assumes you know PyTorch and Hydra configs. You don't need to know NeMo's internals.

How it fits with the rest of the material:

| Resource | Use it for |
|---|---|
| This guide | The reference: how the stack works, which settings matter, and how to judge results |
| [Tutorial notebooks](../../../../tutorials/speechlm2/finetuning/README.md) | Sixteen end-to-end recipes on public benchmarks. Start with `00_Overview_and_Setup` |
| [Tools](../README.md) | The scripts the notebooks call. Each step below names the one to use |
| [`results.md`](results.md) | Measured outcomes of the recipes, including where fine-tuning helps least |
| [`medical-asr-synthetic-data.md`](medical-asr-synthetic-data.md) | A case study on building a domain corpus from synthetic speech (tools in [`sdg/`](../sdg/)) |

Commands below run from `examples/speechlm2/finetuning/`. The tools, by stage (the [tools README](../README.md) has one
line per script):

| Stage | Tools |
|---|---|
| Data | [`prepare_benchmarks.py`](../prepare_benchmarks.py) (public benchmarks to 16 kHz audio and manifests, plus a LibriSpeech guard set), [`set_context.py`](../set_context.py) (prompts), [`shuffle_manifest.py`](../shuffle_manifest.py), [`subset_manifest.py`](../subset_manifest.py) (random or class-stratified subsets for data-efficiency studies), [`check_contamination.py`](../check_contamination.py), [`sdg/`](../sdg/) (synthetic text and speech) |
| Config | [`make_ft_config.py`](../make_ft_config.py) (training YAML from the checkpoint's `config.json`), [`epoch_batches.py`](../epoch_batches.py) (real batches per epoch) |
| Train | [`run_training.py`](../run_training.py) (preflight, smoke test, watchdog, pruning), [`check_checkpoint_coverage.py`](../check_checkpoint_coverage.py) |
| Export | [`export_checkpoint.py`](../export_checkpoint.py), [`merge_lora_checkpoint.py`](../merge_lora_checkpoint.py), [`average_checkpoints.py`](../average_checkpoints.py) |
| Evaluate | [`vllm_task_eval.py`](../vllm_task_eval.py), [`task_metrics.py`](../task_metrics.py), [`prompt_search.py`](../prompt_search.py), [`vllm_label_scores.py`](../vllm_label_scores.py), [`label_probe.py`](../label_probe.py) |
| Ensemble | [`make_tta.py`](../make_tta.py) (test-time augmentation), [`rover_combine.py`](../rover_combine.py) (ROVER voting) |

## 2. Environment

Training and inference need **two separate Python environments**:

| Environment | Contents | Runs |
|---|---|---|
| Training | NeMo Speech with `speechlm2` and the exact NeMo Automodel revision this repository pins | `salm_train.py`, `to_hf.py`, the config, export, and data tools |
| Inference | vLLM 0.28.0 plus this repository installed in editable mode | `vllm_task_eval.py`, `vllm_label_scores.py`, `prompt_search.py` |

**Use the repository's exact Automodel pin for training.** `pyproject.toml` pins `nemo_automodel` to a specific git
revision (`[tool.uv.sources]`); install the training environment from the lockfile
(`uv sync --locked --extra all --extra cu13`) or use a container built from this checkout. A container with an older
Automodel imports cleanly but fails at the first loss call, because the API changed between revisions. Check with
`python -c "import nemo_automodel, importlib.metadata as m; print(m.version('nemo_automodel'))"` against the pin.

vLLM 0.27 through 0.29 pin a newer torch than the training stack's TransformerEngine and flash-attn are compiled
against, so installing vLLM into the training environment breaks training. Set up the inference environment
separately:

```bash
uv venv --python 3.13 "$VLLM_VENV"
VIRTUAL_ENV="$VLLM_VENV" uv pip install vllm==0.28.0
# expose NeMo and its vLLM plugin entry point to the inference environment
cd /path/to/Speech
VIRTUAL_ENV="$VLLM_VENV" uv pip install -e ".[core,asr-only,common-only]" torch==2.13.0   # the torch vLLM 0.28 pins
VIRTUAL_ENV="$VLLM_VENV" uv pip install "peft>=0.18.1" kaldialign   # kaldialign: WER/MER scoring in task_metrics.py
```

`vllm_task_eval.py` checks that the metric's dependencies are importable *before* it loads the model, so a missing
scoring package fails in seconds rather than after a full decode.

**Pin vLLM to 0.28.0.** vLLM 0.29 invokes multimodal processors differently, and the SpeechLM plugin (which has no
HuggingFace `ProcessorMixin`) fails at startup with `TypeError: Invalid type of HuggingFace processor`. In the
training environment, `cross_entropy_backend: fused_linear` additionally requires `uv pip install cut-cross-entropy`.

**Match the pip cuDNN to the system cuDNN.** If the pip wheel (`nvidia-cudnn-cu13`) is an older minor version than
the system cuDNN, it lacks sublibraries the system copy provides, cuDNN mixes the two, and every GPU `Conv1d` fails
with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. Install the matching wheel (for example
`nvidia-cudnn-cu13==9.21.0.82` against a 9.21 system cuDNN) in **both** environments. Any later `uv pip install` can
pull the older wheel back in, so re-check after installing anything:

```bash
python -c "import torch; torch.nn.Conv1d(192,1536,3,padding=1).cuda()(torch.randn(2,192,100,device='cuda'))"
```

**Confirm which CUDA device is which.** CUDA's default ordering is not `nvidia-smi`'s; with a display card beside
the compute card, check before every run:

```bash
python -c "import torch; print([(i, torch.cuda.get_device_properties(i).name) for i in range(torch.cuda.device_count())])"
```

**Run one GPU job at a time unless you budget memory.** vLLM reserves `gpu_memory_utilization` of the whole device
for the engine's lifetime, so training or a second evaluation started beside it fails with an OOM whose traceback
points at the wrong thing. To decode while training on a large GPU, lower `VLLM_GPU_UTIL` (training needs up to
~110 GB; on a 256 GB GPU, 0.40 leaves room). `run_training.py` and `export_checkpoint.py` share a file lock so that
concurrent runs queue.

| Variable | Value | Why |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | the compute GPU's CUDA index | see above |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | otherwise the `EngineCore` subprocess reports `CUDA driver initialization failed` |
| `HF_HUB_OFFLINE` | `1` | keeps a local checkpoint from triggering hub lookups |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | reduces fragmentation with variable-length batches |
| `SALM_FT_WORK` | a work directory | where the tools write audio, manifests, and the GPU lock |

**Persist the JIT caches.** On a new GPU architecture, vLLM's first engine start can spend about 15 minutes
compiling kernels. It looks like a hang, but `ps` shows busy `cc1plus`/`ptxas` processes. Keep `~/.cache/flashinfer`
and `~/.cache/vllm` on persistent storage (mount them into the container) so you pay this cost once.

## 3. The model

A SALM (Speech-Augmented Language Model) is assembled from these components:

| Component | Role |
|---|---|
| Preprocessor | log-mel spectrogram, 16 kHz, 10 ms hop |
| Speech encoder (`perception.encoder`) | frame embeddings; with 8x subsampling, one audio token covers 80 ms |
| Projection (`perception.proj`) | maps encoder output into the LLM's embedding space |
| LLM backbone (`llm`) | decodes text from the interleaved text and audio embeddings |

The checkpoint's `config.json` is the authoritative description. Read it before anything else. Three properties
affect fine-tuning:

- **The encoder may be a parallel-expert encoder** described by an inline `pe_encoder_config`. It has an ASR branch
  (`perception.encoder.asr_encoder.*`) and a diarization branch (`perception.encoder.diarization_model.*`). NeMo
  builds this encoder only when the config asks for it. A config describing a plain `TransformerEncoder` builds a
  model whose parameter names match nothing in the checkpoint. Because the loaders are non-strict, **that failure is
  silent** (see [section 10](#10-verifying-you-didnt-break-anything)).
- **The backbone may be a hybrid (state-space plus attention) model.** This matters at serving time: its recurrent
  state needs exact rollback under speculative decoding (see [section 9](#9-evaluation-and-serving-correctness)).
- **Not every weight is an `nn.Linear`.** Some backbones store parts of their weights as grouped 3-D tensors. LoRA
  cannot target those. Check which modules your `target_modules` actually reach.

Checkpoints trained with multi-token-prediction (MTP) heads can serve those heads as a speculative-decoding draft.
Keep the `mtp` block of the config unchanged so that the fine-tuned checkpoint keeps this ability.

## 4. Data and prompts

### Manifest format

Use NeMo JSON manifests, one object per line:

```json
{"id": "utt-0001", "audio_filepath": "/abs/path/audio.flac", "offset": 0.11, "duration": 7.06, "text": "oscar kilo foxtrot charlie alpha you are leaving tma praha", "context": "Provide a verbatim transcript of the audio."}
```

`offset` and `duration` select a segment of a longer recording, so you don't have to pre-cut audio. `context` is the
prompt. [`prepare_benchmarks.py`](../prepare_benchmarks.py) writes this format for sixteen public benchmarks.

**Name the file `*.json`, not `*.jsonl`.** The loader dispatches on the extension (`.jsonl` means Lhotse), and a NeMo
manifest named `.jsonl` fails with `SupervisionSegment.__init__() got an unexpected keyword argument 'audio_filepath'`.
**Check sample rates:** some dataset mirrors declare the wrong rate, so decode with a reader that trusts the file
header (soundfile does).

### Wiring the prompt

The training prompt comes from the `context` field, which `lhotse_as_conversation` prepends as a user text turn
before the audio:

```yaml
input_cfg:
  - type: lhotse_as_conversation
    manifest_filepath: /path/to/train.json
    audio_locator_tag: "<|audio|>"
    tags:
      context: "Provide a verbatim transcript of the audio."
```

A fixed prompt under `tags` **overwrites** any per-row `context`. For per-example prompts (QA passages, label lists,
target languages), drop `tags` and write the prompt into the manifest with [`set_context.py`](../set_context.py)
(`make_ft_config.py --prompt manifest` does this). [`vllm_task_eval.py`](../vllm_task_eval.py) reads the same field,
so the evaluation prompt equals the training prompt by construction. **Never train and evaluate with different
prompt strings.**

Choose the prompt before training. Score several candidate prompts on the base model; the one that wins on dev is
your baseline. It is usually a good training prompt too, but any prompt that names the task and the output format
works: fine-tuning teaches the rest, as long as training and evaluation use the same string. [`prompt_search.py`](../prompt_search.py) automates this on dev. Naming the
target explicitly (the language, the label set, the output format) is often worth more than any training change.

### Audit the reference surface style

Fine-tuning teaches the *surface form* of the references as readily as their vocabulary. Before training, print
references beside the base model's predictions (decode 32 dev rows with `vllm_task_eval.py --limit 32` and compare
`text` with `pred_text`). Look for:

- **Case and punctuation.** A corpus with upper-case, unpunctuated references (AMI, for example) teaches a model
  that emits `No, or like.` to shout.
- **Verbalization.** A corpus with spelled-out digits (`THREE FIVE ZERO`) and expanded letters (`NOVEMBER`) penalizes
  a base model heavily for style alone, so its raw WER understates its recognition quality.

Nearly every normalizer case-folds both sides, so a style mismatch costs nothing on the target metric while it
rewrites the model's output behavior. The damage shows up only on a general-domain guard set. Folding references to a
single case is easy; restoring punctuation that was never annotated is not. If cased, punctuated output must survive,
blend in a cased, punctuated source.

### Match the training pool to the evaluation distribution

Compare the training pool with the evaluation set along every grouping you have: site, channel, speaker, recording
condition. A pool dominated by two sites can face an evaluation set dominated by two others. Apply a quality floor,
then cap the hours any one group contributes so that no single group consumes the budget. Don't expect gains on
groups you have no data for. Expect domain-level gains: phraseology, closed vocabularies, speaking rate, and channel
conventions.

### Shuffle, check contamination, and consider synthetic data

**Shuffle large manifests on disk** with [`shuffle_manifest.py`](../shuffle_manifest.py). Lhotse's shuffle buffer
holds 10,000 rows, so a larger label-sorted manifest trains in sorted stretches and the model collapses to whatever
it saw last. **Check contamination whenever you add a corpus**: public datasets overlap.
[`check_contamination.py`](../check_contamination.py) `--test test.json --train train.json` reports verbatim and
n-gram overlap with test references, and `--drop-out` writes a filtered copy.

When in-domain audio is scarce, the [`sdg/`](../sdg/) tools generate domain text with an open LLM, render how each
string is spoken aloud, and synthesize it with TTS while the label keeps the written form. See
[`medical-asr-synthetic-data.md`](medical-asr-synthetic-data.md) for an end-to-end case study.

## 5. Deriving the config

**Don't start from a generic example config** such as `examples/speechlm2/conf/salm_automodel.yaml`. It describes a
different model: a different prompt format, a plain encoder, a different projection size, MTP disabled, and
multi-GPU expert parallelism. Editing it into shape is how you end up with a silently mis-assembled model.

Start from the checkpoint's own `config.json` and replace only what a fine-tune must own:

```bash
python make_ft_config.py \
  --checkpoint /path/to/checkpoint \
  --out conf/my_finetune.yaml \
  --train-manifest manifests/train.json --val-manifest manifests/dev.json \
  --exp-dir /path/to/exp \
  --train-encoder --train-proj \
  --lr 1e-4 --max-steps 2000 --warmup-steps 200 \
  --batch-tokens 12000 --limit-train-batches 250
```

| Key | Value | Reason |
|---|---|---|
| `pretrained_weights` | `false` | child modules must not re-initialize from their original sources |
| `init_from_checkpoint` | checkpoint directory | where the trained weights come from |
| `pretrained_llm` | `<ckpt>/llm_backbone` | architecture config only |
| `tokenizer_path` | `<ckpt>` | **required**: the tokenizer files sit at the checkpoint root, not in `llm_backbone` |
| `automodel_backend.dispatcher` | `torch` | the exported dispatcher may need NVLink/NVSHMEM |
| `train_gate` | `false` | keep any routing gates frozen (see [section 6](#6-choosing-what-to-train)) |
| `freeze_params`, `prevent_freeze_params`, `lora`, `optimizer`, `lr_scheduler` | recipe-owned | the actual recipe |

Everything else (`perception`, `pe_encoder_config`, `mtp`, `prompt_format`, `audio_locator_tag`,
`packed_sequences`, `pretrained_asr`) is carried over verbatim. That is what keeps the fine-tuned checkpoint
compatible with export and vLLM.

## 6. Choosing what to train

`make_ft_config.py --train-encoder` derives the encoder's parameter names from the checkpoint config. A regular SALM
keeps its encoder at `perception.encoder.*`. A checkpoint with a parallel-expert encoder (`pe_encoder_path` or
`pe_encoder_config`) trains only its ASR branch (`perception.encoder.asr_encoder.*` and `asr_norm.*`) and keeps the
diarization branch frozen. `--tune-mode` controls only the LLM (`lora`, `llm-partial`, `full`); the perception
modules move only if you pass `--train-encoder` / `--train-proj`, whatever the mode. `prevent_freeze_params` also
re-enables parameters that adapter setup froze at module level.

`freeze_params` is a list of regexes matched against parameter names, and `prevent_freeze_params` overrides it. LoRA
parameters are re-opened automatically. The recommended starting point freezes everything and re-opens three things:

```yaml
freeze_params:
  - "^llm\\..+$"
  - "^perception\\..+$"
prevent_freeze_params:
  - "^perception\\.encoder\\..+$"     # the acoustic front end
  - "^perception\\.proj\\..+$"        # the modality projection
lora:
  dim: 32
  alpha: 64
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, up_proj, down_proj]
```

The encoder branch carries the acoustic adaptation and is cheap to train, the projection realigns encoder output
with the LLM embedding space, and LoRA carries lexical and phraseology adaptation without touching base weights.
Leave the **diarization branch** frozen (a fixed speaker-activity feature extractor whose objective isn't part of
this fine-tune) and keep **routing gates** frozen (`train_gate: false`): retraining routing on a narrow domain
re-specializes the experts behind it, irreversibly. Weights LoRA can't reach need full fine-tuning (see
[section 11](#capacity-less-is-usually-more)).

`make_ft_config.py` exposes these choices:

| Option | Effect |
|---|---|
| `--train-encoder`, `--train-proj` | unfreeze the ASR encoder branch and the projection |
| `--encoder-lr-scale` | give encoder parameters a larger step than the LoRA adapters; useful for acoustic shifts |
| `--lora-targets ""` | disable LoRA (encoder-only adaptation, for a pure acoustic shift) |
| `--tune-mode llm-partial` | also unfreeze the final `--unfreeze-llm-layers` LLM blocks |
| `--tune-mode full --optimizer adafactor` | full fine-tuning with a factored optimizer |
| `--llm-lr-scale` | LR multiplier for unfrozen full-rank LLM weights (0.1 to 0.3 is a reasonable range) |

Ablate along the two axes. Encoder-only (no LoRA) and LoRA-only (no `--train-encoder`) runs tell you whether your
domain shift is acoustic or lexical. When only the output language or format changes (translation, for example),
freezing the encoder is usually right.

## 7. Training

```bash
python run_training.py --config conf/my_finetune.yaml --keep-steps 240 420
```

`run_training.py` runs the coverage preflight, a 20-step smoke test that fails unless the restore log is clean, and
then `torchrun examples/speechlm2/salm_train.py` with a watchdog that stops the run if validation never happens.
`--keep-steps` deletes saved checkpoints you don't plan to evaluate as soon as they are complete. A checkpoint takes
about 2 bytes per parameter, which is tens of GB for a large LLM. The full log goes to `<exp_dir>/train.log`.

### Batching

Use **token-based multimodal sampling with packed-sequence accounting**:

```yaml
train_ds:
  batch_size: null
  use_multimodal_sampling: true
  measure_total_length: true
  batch_tokens: 120000
  use_packed_sequence_sampling: true   # required
  use_bucketing: false
```

`lhotse_as_conversation` yields conversation objects that expose `total_length` but no `duration`, so
duration-based batching fails with `AttributeError: No such attribute: duration`. One audio token is 0.08 s, so a
2.5 s utterance costs about 32 audio tokens plus its text tokens.

- **`use_packed_sequence_sampling: true` is not optional.** Without it, the sampler measures a batch the padded way
  (`batch_size * longest_example`) and declares it full long before its real token count approaches `batch_tokens`.
  Batches can hold less than 20% of the budget, with nothing logged. With the flag on, `packing_efficiency` is 1.00.
  Exact packed sampling also needs the `audio_token_estimator` block, which `make_ft_config.py` derives from the
  checkpoint's preprocessor and subsampling config.
- **Turn bucketing off.** THD packing already removes padding. Bucketing then only narrows the length distribution
  of each batch, which correlates the gradient for no memory saving.
- **`max_tokens` is a filter, not a batch cap.** It silently *drops* every example longer than its value. Leave it
  unset unless you mean to discard long utterances.
- **Leave `pretokenize` at its default.** `pretokenize: false` with multimodal sampling fails with
  `No such attribute: context_ids`.

### Sizing `batch_tokens`

With activation checkpointing and bf16, memory is linear in `batch_tokens`. For a large LLM on a single 256 GB GPU,
with LoRA plus the encoder and projection trainable:

    peak_GB ≈ 65 + 0.95 × (batch_tokens / 1000)

| `batch_tokens` | Step time | Tokens/s | Peak memory |
|---:|---:|---:|---:|
| 12,000 | 3.1 s | 2.9k | 81 GB |
| 48,000 | 4.3 s | 11.1k | 111 GB |
| **120,000** | 7.7 s | **15.6k** | ~179 GB |
| 144,000 | 9.8 s | 14.7k | 208 GB |
| 200,000 | — | — | ~255 GB (don't use) |

Throughput peaks around 120k tokens; going from 12k to 120k is a 5.4x speedup with no change to the model. Raise
`batch_tokens` before reaching for gradient accumulation. The first step takes over a minute and the second step is
also slow (compilation); timing reaches steady state by step 3.

**On a small corpus, size for steps instead.** A 6-hour corpus is about 300k tokens per epoch, so 120k tokens per batch
gives about 2.5 optimizer steps per epoch, and no learning rate can rescue a run of a few dozen steps. Below roughly
20 hours of audio, pick `batch_tokens` to yield 15 to 25 steps per epoch, and treat the memory ceiling as a limit, not
a goal.

### Validation and checkpoint cadence

Validation runs every `limit_train_batches` batches. If that exceeds the number of batches in an epoch, **validation
never runs**: no `val_*` metrics appear, and every checkpoint logs `'val_loss' was not in top k`. Hand estimates of
batches per epoch are easily 40% off, so count them with [`epoch_batches.py`](../epoch_batches.py)
`train.json <batch_tokens> --tokenizer <ckpt>` and set `limit_train_batches` strictly below the result. For very
small sets, switch to epoch cadence with `make_ft_config.py --val-every-n-epochs`.

**Gradient accumulation.** `limit_train_batches` and `val_check_interval` count dataloader batches, while checkpoint
cadence and the training logs count optimizer steps. With `--accumulate-grad-batches k`, `make_ft_config.py` sets
`every_n_train_steps = limit_train_batches / k` (and requires it to divide evenly), and the `run_training.py` watchdog
measures the validation interval in optimizer steps, so validation and checkpoints stay aligned.

`exp_manager` defaults `every_n_epochs` to 1, and Lightning rejects having both cadences set:

```yaml
checkpoint_callback_params:
  every_n_train_steps: 250
  every_n_epochs: 0        # required with every_n_train_steps
  save_top_k: -1           # keep everything; prune with --keep-steps
```

### Running long jobs

Launch long jobs detached on the host that runs them and log there; if a client session dies, a healthy run looks hung
and a dead one looks alive. This applies equally when a coding agent runs your jobs. In driver scripts, check that
each step produced its artifact rather than trusting its exit status, and skip completed steps so the script is
restartable. Judge progress from TensorBoard event files and correctness only from decoding.

`run_training.py` launches `torchrun --standalone`, which picks a free rendezvous port, so independent one-GPU jobs on
the same node don't collide.

### Provenance: never reuse a result across changed inputs

Every stage records a fingerprint next to its output: digests of the base checkpoint (configs and tokenizer in full,
weights sampled), the manifests, the resolved config or decoder and scoring settings, and the code revision.

| Stage | Fingerprint | On re-run |
|---|---|---|
| `run_training.py` | `<exp_dir>/fingerprint.json` | resumes only if it matches; refuses a mismatch, or checkpoints without a fingerprint |
| `export_checkpoint.py` | `<out>/provenance.json` | reuses a complete export with matching provenance; a partial or mismatched one is an error |
| `average_checkpoints.py --reuse` | `<dst>/provenance.json` | reuses only an average of the same sources |
| `vllm_task_eval.py --reuse` | `fingerprint` in `<out>.summary.json` | reuses a result only on an exact match; refuses otherwise |
| `vllm_label_scores.py --reuse` | `<out>.fingerprint.json` | same |

After changing the base checkpoint or the recipe, use a new `SALM_FT_WORK` (or new output names). The tools refuse to
mix old and new results instead of silently showing stale metrics.

## 8. Export and LoRA merge

Training writes Lightning checkpoints. vLLM needs a merged HuggingFace directory:

```bash
python export_checkpoint.py --exp-dir /path/to/exp --step 420 \
  --base-checkpoint /path/to/checkpoint --out /path/to/merged
```

It builds the output under `<out>.unfinished` and renames it into place only when complete. It symlinks
`step=420.ckpt` to a name without `=` (Hydra can't parse `=` in an override value), runs
`examples/speechlm2/to_hf.py` with the run's resolved `exp_config.yaml` (not your input YAML), and asserts that
`lora_B.weight` is non-zero (it initializes to zero, so zero means the trained weights never reached the export).
TransformerEngine `_extra_state` bookkeeping is excluded from this check. It also requires at least one non-zero
paired, scaled `B @ A` delta before writing the merged weights. It then merges
`W_effective = W_base + (alpha / dim) · (B @ A)` in float32, drops the adapter tensors and the `lora` config block so nothing can apply the update twice, and restores `pretrained_asr`, without which vLLM refuses the config. A
run trained without LoRA (`--lora-targets ""`) keeps the ordinary full-rank export. Adapter tensors without a `lora`
config block, a `lora` block without adapter tensors, or unpaired or mis-shaped adapters are rejected.

**Always merge before serving.** `to_hf.py` exports the state dict verbatim, so an unmerged LoRA fine-tune ships
`lora_A`/`lora_B` beside the frozen base weights, and whether they are applied depends on the serving stack. A
backend that doesn't merge, or that reads PEFT's `r`/`lora_alpha` while NeMo writes `dim`/`alpha`, serves the base
model at base quality without any warning. Merging at export removes that dependency.
[`merge_lora_checkpoint.py`](../merge_lora_checkpoint.py) `--src exported --dst merged` performs only the merge.

**Average merged checkpoints, never adapters.** [`average_checkpoints.py`](../average_checkpoints.py)
`--src a b c --dst avg` averages exported checkpoints from one run, building under `avg.unfinished` and renaming only
when the weights and every config file are written. Averaging LoRA factors is a different operation,
because `mean(B) @ mean(A) ≠ mean(B @ A)`.

## 9. Evaluation and serving correctness

Evaluate with vLLM. The PyTorch `salm_eval.py` path is roughly 150x slower on a large model (RTFx below 1, compared
with 60 to 150 in vLLM).

```bash
python vllm_task_eval.py --model /path/to/merged --task asr \
  --manifest manifests/dev.json manifests/test.json --out eval/dev.jsonl eval/test.jsonl \
  --spec-k 2 --mamba-cache-mode all --max-num-seqs 64 --gpu-memory-utilization 0.85
```

`--task` takes `asr`, `bleu`, `cls`, `squad`, `mer`, `pcc`, `eer`, or `multilabel`. Loading the engine once and
decoding several manifests saves the startup cost. Hypotheses are written as JSONL (`text`, `pred_text`), so
[`task_metrics.py`](../task_metrics.py) can re-score them offline, for example with a different normalizer, or with
`--by-duration` to break WER down by segment length.

### Disable thinking

This is the highest-impact setting. Reasoning-capable prompt formats (for example `nemotron3p5`) default to
`enable_thinking=True`, which ends the inference prefix with `<think>\n`. The model then writes a chain of thought
that ignores the audio, and WER goes well above 100%. With `enable_thinking=False` (the default in
`vllm_task_eval.py`), the prefix ends with an empty `<think></think>`, matching how training targets are written. The
rendered prompt should look exactly like this:

```
<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<your prompt> <|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>
```

### Speculative decoding with MTP heads

A checkpoint trained with multi-token-prediction heads can draft for itself:

```python
speculative_config = {"method": "mtp", "model": "<same checkpoint>", "num_speculative_tokens": 2}
```

Keep the `mtp` config block unchanged; vLLM requires `use_repeated_layer: true` for more than one prediction layer.

**Speculative decoding is not automatically exact.** Under greedy sampling, vLLM's rejection sampler always emits the
target model's argmax, so drafts should affect only speed. Output can change only if the target model's own state is
corrupted. On hybrid state-space backbones, the recurrent state must be rolled back to the exact rejection point.
With prefix caching on, vLLM's default cache mode (`align`) checkpoints that state too coarsely, and with two or
more speculative tokens the output degrades silently (about 1 WER point on ASR):

| Configuration | Exact? |
|---|---|
| `--spec-k 0` (no speculation) | yes |
| `--spec-k 1`, defaults | yes |
| `--spec-k 2`, defaults (prefix caching on, cache mode `align`) | **no** |
| `--spec-k 2 --mamba-cache-mode all` | yes, and keeps prefix caching and the speed-up |
| `--spec-k 2` with prefix caching disabled | yes |

`--mamba-cache-mode all` is required for exact speculative decoding on hybrid state-space backbones, and it is
`vllm_task_eval.py`'s default. For other backbones, or other optimizations (quantization, caching, batching), verify
the same way: decode a few hundred utterances with the optimization on and off, and compare the *outputs*, not just
the scores.

### Choose and report the normalizer

Normalization moves WER more than most modeling decisions. The same hypotheses on an air-traffic-control test set
score 73.2% WER with lowercasing and punctuation stripping (`simple`), 74.4% with Whisper's basic normalizer, 84.1%
with Whisper's English normalizer, and 87.9% with none. Whisper's English normalizer converts spelled-out numbers to
digits and expands abbreviations, which is a large intervention on text where digit strings are spoken one by one.
Pick the normalizer that matches your domain's conventions, use it for baseline and fine-tune alike, and state which
one produced every number you publish.

### Closed-set tasks: score labels, don't parse text

For classification and detection, [`vllm_label_scores.py`](../vllm_label_scores.py) returns the exact
log-probability of every label for every example (optionally over fixed windows). This yields a full score vector
for EER, calibration, window voting, and ensembling. When weight fine-tuning erodes knowledge the prompted model
already had, [`label_probe.py`](../label_probe.py) fits a linear probe on these scores from the frozen model, using
only training data, and selects regularization on dev. `vllm_task_eval.py --score-labels NEG POS` gives label
log-odds for binary detectors: the exact sequence log-probabilities of the two label continuations of the prompt,
read from `prompt_logprobs`. Don't score detectors from top-k first-token logprobs: a label missing from the top k
gets an arbitrary floor, which reorders scores and changes EER without any visible error.

Both scorers run vLLM with prefix caching and speculative decoding **off**. On a hybrid backbone, requests that share
a cached prompt-and-audio prefix can get wrong `prompt_logprobs`, while their generated text stays correct. With
caching on, an anti-spoofing detector scored 3.1% EER whose clean score was 0.6%, and a fifth of its exact scores
disagreed with its own generated answers. If you write your own scorer, turn caching off, or check that the scores
agree with the generated labels.

## 10. Verifying you didn't break anything

### Before training: checkpoint coverage

The costliest failure in this stack is silent: NeMo assembles a model whose layout differs from the checkpoint's,
the loaders are non-strict, and training proceeds from randomly initialized submodules.

```bash
python check_checkpoint_coverage.py --checkpoint /path/to/checkpoint --config conf/my_finetune.yaml
```

A correct config matches every checkpoint tensor by name and shape, and the only unmatched model parameters are
the new LoRA adapters. Shape mismatches are reported with the tensor name and both shapes and cause preflight to fail:

```
checkpoint tensors: N
model parameters:   N + K
matched:            N
new LoRA adapters:  K (expected — these are created by the recipe)
```

The check reads only the safetensors header, without loading checkpoint tensor data. It builds the configured model
on CPU in bfloat16, so it needs enough host RAM for the model (roughly two bytes per parameter).

### During training: the restore log

Look for `| > N / M layers are restored (N exact, 0 partial, 0 skipped ...)` and, for a parallel-expert encoder,
`Mounted ParallelExpertEncoder from model.pe_encoder_config`. There must be **no**
`have no matching parameter ... were dropped` warning. `run_training.py --smoke` checks all of this, plus a finite
loss, before the full run starts.

### After export: does the fine-tune differ from the base at all?

Decode a handful of utterances with the base and fine-tuned checkpoints and diff the outputs. "The fine-tune changed
nothing" is a plausible-looking result that is usually an inference-path bug (unmerged adapters, wrong checkpoint),
not a training failure.

### After training: the general-domain guard

Score every candidate on a general set as well as the target domain. Target-domain WER alone cannot distinguish
adaptation from replacement of the model's distribution. A fine-tune can reach a competitive domain WER while
general-English WER rises from about 3% to over 15%. LibriSpeech `test-clean` is a convenient guard:

```bash
python prepare_benchmarks.py librispeech
python vllm_task_eval.py --model /path/to/merged --task asr \
  --manifest "$SALM_FT_WORK/manifests/librispeech/test_clean.json" --out eval/ls.jsonl
```

## 11. Judgment calls

### Before training anything

- **Reproduce every baseline yourself.** Re-decode the base model and any published competitor under *your*
  protocol. Published numbers depend on normalizer, split, and decoding settings that are rarely stated in full. You
  can't claim to beat something you didn't measure.
- **Prove your decode path is lossless** before benchmarking through it (see
  [section 9](#speculative-decoding-with-mtp-heads)). An optimization that is mathematically exact as an algorithm
  is not necessarily exact in its implementation in your stack, on your architecture, with your flags. A lossy path
  shifts every absolute number in the same direction and hides the error.

### Diagnose before you choose a lever

Measure the decoded metric on your *training* set, not just on dev. That one number decides which levers can work:

| Train WER | Dev WER | Diagnosis | What can help |
|---|---|---|---|
| ≈ 0 | high | memorization; a generalization gap | new data, regularization, ensembling, distillation on unseen audio |
| high | high | underfitting | capacity, learning rate, longer training |
| ≈ dev | — | well matched | more data of the same kind |

On a small corpus, a large model reaches near-zero training error quickly. Once it has, **every technique that
learns from the training set does nothing**: self-training, confidence filtering, curriculum pruning, and
sequence-level distillation all reduce to training on labels the model already fits. Check before you build any of
them.

### Capacity: less is usually more

On a small in-domain corpus (around 10 hours), adaptation quality does not increase with the number of trainable
parameters. LoRA plus the encoder typically beats partial unfreezing, which beats full fine-tuning. Start there and
treat any capacity increase as a hypothesis to test.

**Adapt the encoder when the shift is acoustic.** If the domain *sounds* different (channel, noise, accent, speaking
style), unfreezing the speech encoder and giving it a *higher* learning rate than the LLM adapters is usually the
largest single gain available. Domain shift in ASR is usually acoustic first and lexical second.

**Full fine-tuning feasibility is an optimizer-memory question.** Weights, gradients, and optimizer state must all
fit. With bf16 weights taking W GB:

| Mode | Memory |
|---|---|
| LoRA + encoder (recommended) | about W plus activations |
| full fine-tune, AdamW (weights, gradients, two fp32 moments) | about 6W |
| full fine-tune, Adafactor (factored second moment) | about 2W |

A factored optimizer can bring full fine-tuning onto one GPU. Feasible is not the same as useful: on a few hours of
audio, a fully fine-tuned large model usually trains worse, and a factored optimizer's relative step size is not a
drop-in replacement for a hand-set learning rate. Use it to answer "can it fit?", not as a default.

**Average checkpoints within a run.** Averaging several checkpoints from the same run is one of the most reliable
free gains, because they share a loss basin. It doesn't transfer to independently seeded runs. Combine their
outputs instead.

### Data: more is not automatically better

**Blending an out-of-domain corpus into a small in-domain one usually hurts**, across mixing ratios and replay
fractions. There are two causes, and only one can be fixed with ratios:

1. **Distribution dilution:** gradient steps spent on audio you don't care about. Ratio, curriculum, or sequential
   ordering (auxiliary corpus first, then target) can fix this.
2. **Annotation-convention conflict:** the auxiliary corpus transcribes the same sounds by different rules, which
   teaches the model two contradictory output policies. No ratio fixes this.

**If you must use an auxiliary corpus, relabel it with your own model** instead of using its references. The
pseudo-labels arrive in your model's output convention, and the acoustic diversity, which is why you wanted the
corpus, is preserved. This can turn an auxiliary corpus that hurts into one that helps.

**Treat augmentation as a hypothesis.** Noise, channel simulation, and speed perturbation are standard when training
from scratch. On a small fine-tuning corpus against a strong model they frequently hurt. `make_ft_config.py --augment`
is off by default; measure before you adopt it.

### The last points: ensembling and distillation

When a single model plateaus on a generalization gap, combining *hypotheses* is usually the next real gain, and it
needs no further training.

- **Test-time augmentation is the cheapest source of diversity.** Decode at mild speed perturbations
  ([`make_tta.py`](../make_tta.py) `--factor 0.95` and `1.05`) and vote over the transcripts
  ([`rover_combine.py`](../rover_combine.py) `--hyps best.jsonl tta1.jsonl tta2.jsonl --out combined.jsonl`). Each
  perturbed decode is worse alone and still improves the combination.
- **Diversity beats member quality.** Two independently seeded runs combine far better than several checkpoints of
  one run, because their errors fall in different places. Weaker members still help, and up-weighting the best
  member suppresses the disagreement that makes voting work, so start unweighted.
- **Average within a run, vote across runs.** The two are complementary; use both.
- **An ensemble is a benchmark result, not a deployment.** An N-way ensemble costs N decodes. Always report the best
  single-model number beside it.
- **Distillation recovers part of the ensemble gain, and only on audio the student hasn't fit.** Label held-out
  in-domain audio with the ensemble and train on it; if the student already scores near zero there, it learns
  nothing.

### Selection and reporting discipline

- **Select on dev; touch test once.** State which split chose the checkpoint, the prompt, the ensemble size, and
  every other free parameter. A configuration chosen by looking at test produces a test number, not a result. This
  includes the base-model baseline: choose its prompt on dev, exactly like the fine-tuned checkpoint, and compare the
  two on test. Report the best prompt *on test* only as a labeled **test-oracle**: an upper bound for prompting.
- **Select checkpoints by decoding, not validation loss.** Validation loss and token accuracy track the decoded
  metric only loosely. If the monitored metric ties across checkpoints, "keep the best k" silently becomes "keep the
  last k".
- **The decoded metric is not monotonic in training time.** Score several checkpoints. A middle checkpoint often beats
  the last one, and "stop earlier to be safe" is not reliable either.
- **Always run the general-domain guard** (see [section 10](#after-training-the-general-domain-guard)).
- **Report inference settings with every number:** decoding flags, speculation depth, cache mode, batch settings,
  normalizer, and serving version.
- **Keep negative results.** A list of plausible moves that didn't work is often the most reusable artifact of a
  tuning effort. [`results.md`](results.md) records misses beside successes.

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| WER > 100%, hypotheses unrelated to the audio | thinking mode enabled | `enable_thinking=False` in the chat template |
| Output fluent but wrong, WER very high, no errors anywhere | encoder never loaded | carry `pe_encoder_config` over verbatim; run `check_checkpoint_coverage.py` |
| Fine-tune decodes identically to the base model | LoRA adapters never applied, or the wrong checkpoint was served | export with `export_checkpoint.py` (checks `lora_B`, merges); diff against the base |
| WER about 1 point worse with speculative decoding on | inexact state rollback on a hybrid backbone | `--mamba-cache-mode all`, or disable prefix caching |
| `SupervisionSegment.__init__() got an unexpected keyword argument 'audio_filepath'` | NeMo manifest named `.jsonl` | rename it to `.json` |
| `AttributeError: No such attribute: duration` | duration batching on multimodal conversations | `use_multimodal_sampling: true` and `batch_tokens` |
| `AttributeError: No such attribute: context_ids` | `pretokenize: false` with multimodal sampling | leave `pretokenize` at its default |
| `Unable to instantiate HuggingFace AUTOTOKENIZER` | `tokenizer_path` not set | point it at the checkpoint root |
| `Value error, NeMo SpeechLM config must declare pretrained_asr` | the config dropped `pretrained_asr` | copy the base value verbatim (`export_checkpoint.py` does this) |
| `ImportError: cut_cross_entropy is not installed` | `cross_entropy_backend: fused_linear` | `uv pip install cut-cross-entropy`, then re-check cuDNN |
| `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` | pip cuDNN older than system cuDNN | install the matching `nvidia-cudnn-cu13` in both environments |
| `MisconfigurationException: ... every_n_train_steps ... every_n_epochs ... mutually exclusive` | `exp_manager` default | `every_n_epochs: 0` |
| `'val_loss' was not in top k` at every checkpoint, no `val_*` metrics | `limit_train_batches` exceeds batches per epoch, so validation never runs | count with `epoch_batches.py` and set it lower, or use epoch cadence |
| Training slower than expected, `packing_efficiency` < 1 | `use_packed_sequence_sampling` unset | set it to `true` |
| Long utterances vanish from training | `max_tokens` set | it is a filter, not a batch cap; unset it |
| Model answers every input with one label | label-sorted manifest larger than the shuffle buffer | `shuffle_manifest.py` |
| Target metric improves, general-domain WER collapses | reference surface style differs from model output | audit style before training; blend in a cased, punctuated source |
| `OverrideParseException: mismatched input '='` | Hydra override value contains `=` | symlink the checkpoint to a name without `=` (`export_checkpoint.py` does this) |
| `to_hf.py`: `No such file or directory` on an existing checkpoint | the save is still in progress | wait until `<step>-unfinished` is gone |
| `TypeError: Invalid type of HuggingFace processor` | vLLM 0.29 or later | pin vLLM 0.28.0 |
| `RuntimeError: CUDA driver initialization failed` in `EngineCore` | fork start method | `VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| vLLM appears hung on its first start | one-time kernel JIT compilation (about 15 minutes) | wait; persist the flashinfer and vLLM caches |
| `incorrect regex pattern … set fix_mistral_regex=True` when loading the tokenizer | transformers assumes an old Mistral tokenizer because the SALM `config.json` has no `transformers_version` | harmless: the checkpoint's `tokenizer.json` already has the corrected pattern, and token IDs are identical with or without the flag |
| `CUDA out of memory` in a job that normally fits | another GPU job holds the device | run jobs one at a time; check `nvidia-smi --query-compute-apps` |
| OOM loading the checkpoint on a large GPU | wrong CUDA device selected | CUDA order differs from `nvidia-smi` order |
| Training log stops growing but the run is alive | the launching shell died and stdout is a dead pipe | launch detached on the host, log there; read the TensorBoard event file |
| `OSError: libavutil.so...` loading a HuggingFace audio dataset | torchcodec needs an ffmpeg runtime | `cast_column("audio", Audio(decode=False))` and decode with soundfile |
