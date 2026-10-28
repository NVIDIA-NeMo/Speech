---
name: train-speech-llm
description: Train a SALM speech LLM from a pretrained LLM and pretrained speech encoders with NeMo speechlm2 and the Automodel backend on a multi-node cluster. Use to set up, launch, monitor, continue, fork, or debug a large multitask training run (ASR, contextual ASR, translation, speech QA, multi-speaker, text retention), to design or change its data blend, or to find out why such a run regressed in quality or throughput. To adapt an already trained SALM checkpoint on a single GPU, use finetune-speech-llm instead.
---

# Training a speech LLM from pretrained components

This skill turns a pretrained LLM and pretrained speech encoders into one SALM model and trains it end to end. The run
uses a weighted multitask blend, tens of thousands of steps and tens of nodes. The procedure is written for a hybrid
Mamba-attention MoE LLM such as Nemotron 3 Nano, and most of it applies to any backbone.

- `examples/speechlm2/conf/salm_nemotron.yaml` is the annotated starting config. Start from it rather than from
  `salm_automodel.yaml`, whose portable defaults are slow at this scale.
- `examples/speechlm2/salm_train.py` trains the model, and `examples/speechlm2/to_hf.py` exports a checkpoint for
  vLLM.
- `.claude/skills/migrate-to-resumable-dataloader/` sets up indexed, resumable data loading, which this skill
  requires.
- `.claude/skills/debug-training-logs/` finds the root cause when a distributed job dies.

Two terms recur throughout. A **source** is one leaf entry of the data blend, and a **draw** is one example sampled
from a source.

## The governing risk: silent damage

A run of this size costs tens of thousands of GPU-hours. The failures that waste it rarely crash. They surface
thousands of steps later as an evaluation regression, and by then the only fix is a fork from an earlier checkpoint.
Every check in this skill guards against one of them:

| What you notice | Actual cause | Caught by |
|---|---|---|
| Insertions climb and WER drifts upward over thousands of steps | `precision: bf16-true` cast the LLM's FP32 tensors (Mamba, router) to BF16 | Step 4, dtype check |
| WER swings between neighboring checkpoints, with many empty outputs | A "non-speech" source contained real speech labeled with an empty answer, so the model learned to stop early | Step 2, audits |
| After a restart, examples are garbled or mismatched | A manifest changed after its index was built | Step 2, immutable data |
| A source contributes nothing, and no error appears | Wrong source `type` or a missing `text_field`, so every draw is dropped | Step 4, loss-token probe |
| Long examples never appear in training | `max_tokens` filters examples out; it does not cap them | Step 3 |
| A fork resumes from its parent's latest step instead of the requested one | The fork reused the parent's log directory, and `resume_if_exists` overrode `resume_from_checkpoint` | Step 5 |

None of these raises an error. Do not launch a run, or resume one after any change, without Steps 2 and 4.

## Step 1: Assemble the model

- **LLM.** Use a chat checkpoint and its matching `prompt_format` (`nemotron-nano-v3` for Nemotron 3 Nano). The
  formatter prepends an empty `<think></think>` to every training answer, so serve with `enable_thinking=false`.
- **Speech encoders.** `pe_encoder_path` mounts a parallel-expert encoder: an ASR encoder and a diarization encoder
  side by side, fused per frame, so one model handles both single- and multi-speaker transcription. It replaces the
  encoder of `pretrained_asr`, which still supplies the preprocessor. Use the same feature normalization
  (`asr_normalize_type`, `diar_normalize_type`) in training and serving: a mismatch moves WER and produces no other
  symptom. Without a speaker branch, drop `pe_encoder_path` and `data.multispeaker_cfg`.
- **Chunking and packing.** With `encoder_chunk_size_seconds: 30`, the encoders handle long recordings at the cost
  and memory of short ones. A transformer or parallel-expert encoder also accepts packed input
  (`packed_encoder_sequences`), which removes encoder padding. Mirror the encoder's front end and subsampling in
  `audio_token_estimator`.
- **Connector.** `IdentityConnector` passes encoder states unchanged to perception's single linear projection into the
  LLM's hidden size. Its `d_model` must equal the encoder's output width.
- **Multi-token prediction.** Train two MTP layers jointly with the backbone. They cost little during training and
  speed up serving through speculative decoding. Configure them only under `model.mtp`. The legacy top-level
  `compute_mtp` flag has no effect on training.
- **Checkpoints.** Download the HF checkpoints into the cluster's shared cache before the first job, so that hundreds
  of ranks do not download them at once.

## Step 2: Build the data blend

### Structure

Each entry in `input_cfg` has a type that decides how its rows become conversations:

- `lhotse_as_conversation` turns speech rows into `[system, context, audio] → answer`. Its `preference_sampling`
  option handles rows that carry several alternative targets: it draws one task per row according to fixed task
  weights. Set `skip_invalid_target: true` so that rows lacking the drawn target are skipped rather than trained
  with an empty answer. `preference_sampling` has no effect under any other type.
- `group` nests a blend whose data is already in conversation form, such as multi-speaker data or text-only chat.

`tags` overwrite fields of the same name (`context`, `system_prompt`) on every row of their source.

### Weights

**Weight sources by the answer tokens they contribute to the loss, not by hours or rows.** Sources differ by more
than ten times in supervised tokens per draw. A short contextual-ASR answer and a long multi-speaker transcript are
one draw each, so weights based on hours or rows misallocate the gradient.

1. Measure `a_i`, the mean answer tokens per draw, for every source. Measure through the real training path: indexed
   reader, preference sampling, prompt formatter and loss mask. Sample at least 1,000 rows per source and require
   zero errors.
2. Choose each source's target share `t_i` of all answer tokens. Temperature 0.5 is a good default:
   `t_i ∝ m_i^0.5`, where `m_i` is the source's total answer tokens. Give capability-retention data, such as
   text-only chat, a small explicit share instead.
3. Set each draw weight to `w_i ∝ t_i / a_i`, and normalize within every list of siblings.
4. Measure again whenever a source's content, prompt or task weights change. Weights are valid only for the blend
   revision they were measured on.

### Audits

Audit every source before it joins a blend:

- **Empty or very short targets.** Run an independent ASR model on a random sample drawn from many shards. Corpora
  labeled "non-speech" often contain intelligible speech. Trained with an empty answer, such an example supervises
  only EOS, and the model learns to stop before transcribing.
- **Schema.** Check that the expected `text_field` is present, that sample rates match, and that multi-speaker rows
  respect `num_speakers`.
- **Access.** From inside the training container and as the training user, read a sample from across the source, not
  only its first record. Every path prefix in the manifests must be mounted.

### Immutability

An index pack holds the precomputed indexes that let the dataloader resume a source exactly; a pack root is the
directory of packs for one blend. Freeze the manifests first, then build their packs into a new pack root for each
blend revision. Build into a temporary directory and rename it to its final path when complete. A weight-only change
can reuse existing packs. A content change needs a new blend revision, rebuilt packs, re-measured weights and a new
config file. Never rebuild a pack that a running job reads.

## Step 3: Configure

`salm_nemotron.yaml` encodes the settings below. They decide speed, stability and whether data goes missing silently.

| Setting | Value | Why |
|---|---|---|
| `trainer.precision` | `bf16-automodel` | Automodel owns parameter dtypes, so FP32 tensors stay FP32. Never use `bf16-true`. |
| `model.optimizer` | TE `FusedAdam` with FP32 master weights and BF16 moments | Halves optimizer-state memory and keeps update precision |
| `model.lr_scheduler` | Cosine decay after a linear warmup | Retune the peak rate for each backbone and global batch size |
| `data.train_ds.batch_tokens` | 65,536 per rank | Exact with `use_packed_sequence_sampling` and `measure_total_length` |
| `data.train_ds.max_tokens` | equal to `batch_tokens` | It is a filter. A lower value silently drops long examples. |
| `data.train_ds.use_bucketing` | `false` | Packing already removes padding, so bucketing only correlates batches |
| `data.train_ds.audio_token_estimator` | mirrors the encoder front end | Exact packing needs it to count audio tokens |
| `model.packed_sequences` | `true` | The LLM sees examples concatenated without padding. Set `packed_encoder_sequences` too when the encoder supports it. |
| `model.cross_entropy_backend` | `fused_linear` | Never materializes full-vocabulary logits |
| `model.automodel_backend` | TE attention and linear layers, grouped-GEMM experts, DeepEP dispatcher | Treat as fixed. Never fall back to `torch` kernels to debug. |
| `trainer.strategy` | 8-way expert parallelism, FSDP across all ranks, activation checkpointing in LLM and perception | Fits 80 GB GPUs at 65,536 tokens per rank |
| `model.train_gate` | `true` | Lets the MoE router adapt to speech. `aux_loss_coeff` applies only when above 0; a value of 0 keeps the checkpoint's coefficient rather than disabling the loss. |
| `data.train_ds.indexed`, `use_stateful_dataloader`, `index_pack_root` | on, with an immutable pack root | Restores the exact data order after every restart |
| `exp_manager.checkpoint_callback_params` | every 500 steps, `save_top_k: -1` | Keeps every checkpoint for selection by decoding |
| `exp_manager.max_time_per_run` | a few minutes under the job's wall-time limit | Leaves time for the final distributed checkpoint save |

Model-config keys that no code reads are ignored silently, and unknown data keys only log a warning. After adding or
renaming a key, find the code that reads it in the exact source tree you package for the job.

## Step 4: Preflight

1. **Loss-token probe.** Pass 50 to 100 rows from each source through the real dataset, formatter and collator, and
   print the decoded loss-bearing tokens. Every assistant turn must be supervised and nothing else may be. No source
   may produce EOS as its only target, unless it is true non-speech that passed the audit.
2. **Single-node smoke test.** Run 100 to 200 steps with the production container and code, scaled down to one node.
   At load, there must be no missing-key or dropped-tensor warnings. During training, require finite loss,
   `packing_efficiency` near 1.0 and peak memory with a few GiB of headroom. A single node does not predict
   multi-node step time.
3. **Dtype check.** Read the metadata of the first saved distributed checkpoint. Every tensor that the HF checkpoint
   stores in FP32 must still be FP32. Repeat the check after any change to precision settings or to Automodel.
4. **Restart check.** Stop the smoke run after a save and resume it. The log must show that the optimizer, scheduler,
   RNG and dataloader states were restored, and training must continue from the saved step.

Point `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` at shared storage and warm them once with the production
image. Otherwise every job spends minutes compiling kernels before its first step.

## Step 5: Launch

- **Chain jobs.** Submit the run as a chain of jobs, each within the wall-time limit. Each job saves a checkpoint when
  its timer expires. A job can still exit with an error after that save, for example from a crash during teardown,
  and an `afterok` dependency then cancels the rest of the chain. Chain with `afterany` and confirm each job's save.
  To stop the run, cancel every queued job; a forgotten `afterany` job keeps training.
- **Fork from a specific step.** `resume_if_exists` resumes from the `-last` checkpoint in the log directory and
  overrides `resume_from_checkpoint`. To fork from step N, give the fork a new `explicit_log_dir` and set
  `resume_from_checkpoint` to the parent's `step=N` checkpoint.
- **Exclude bad nodes.** Keep a list of nodes to exclude and re-check it before every launch. Typical offenders are a
  GPU that fails at startup, a thermally throttled GPU that doubles step time, and a faulty network link that triggers
  NCCL watchdog timeouts.
- **Crashes that repeat at the same step.** A crash that recurs at the same step on every attempt depends on that
  step's data or on the parallel layout. Inspect the batch, then try a different layout before retrying. A crash that
  does not recur only needs a resume.

## Step 6: Monitor

Take the step time, dataloader wait and memory of the first stable hour as the baseline. Expect the first hour after
each resume to run slower while caches warm up.

| Signal | Act when |
|---|---|
| Step time, rolling median | It stays above 1.5 times the baseline for more than 30 minutes. With healthy loss, look for a slow node. |
| Dataloader wait | It grows from tens of milliseconds toward a second. Suspect storage, I/O or too few workers. |
| Per-rank memory and utilization | One rank differs from the rest. Suspect a straggler. |
| Loss and MTP loss | A spike appears. Resume from the previous checkpoint and see whether it recurs at the same step; a spike that does not recur is noise. |

Read the current step from the rank-zero log. Logger dashboards can lag behind it, and epoch or batch counters do not
equal the optimizer step.

## Step 7: Evaluate and select checkpoints

**Accept a checkpoint before evaluating it.** Require two identical file listings taken a few minutes apart, no
temporary or zero-byte files, a passing dtype check, and at least one finite training step after the save. Evaluate
the numbered `step=N` checkpoint. The `-last` checkpoint is overwritten as training continues.

**Select checkpoints by decoding, never by loss.** Export with `to_hf.py`, serve with vLLM, and score the full
benchmark suite:

- Use MTP speculative decoding. With vLLM 0.32 or newer, serve with `--use-replayssm --mamba-backend flashinfer
  --mamba-cache-mode none --no-enable-prefix-caching`. The older `--mamba-cache-mode all` setting is no longer
  available.
- Serving pads the vocabulary, and its untrained rows can outscore every real token. The NeMo vLLM plugin masks them
  (`speechlm_output_vocab_size`); replicate that mask if you serve with another stack. Count `<|audio|>` tokens and
  empty outputs in every evaluation.
- Report insertions, deletions and empty outputs, not only WER. The silent failures above first show up in these
  counts.
- Score a text-only benchmark as well, to catch regressions in the LLM's text capability.
- Compare checkpoints across the whole suite. Neighboring checkpoints fluctuate, so a single benchmark cannot pick a
  winner.

A model that learned to stop early cannot be repaired at decode time. Forcing a first token recovers the affected
utterances but also breaks correct abstention on true non-speech. Remove the bad data and continue training; a later
RL stage that rewards complete transcripts can also correct the remaining early stops.

## Step 8: Change a run in flight

Fork from an accepted checkpoint into a new log directory and a new config revision. A full-state resume restores the
optimizer, scheduler and dataloader states along with the weights.

| Change | How | Never |
|---|---|---|
| Learning rate or schedule | New config, full-state resume | Edit a config that a queued job will read |
| Source weights | New weights on the same packs, full-state resume | Reweight without measuring answer tokens again |
| Add a source | Declare likely future sources from the start with weight `0.0`, then raise the weight | Insert a new entry into a running blend: the saved dataloader state no longer matches it |
| Remove a source | Set its weight to `0.0` | Delete its entry, which misaligns the saved dataloader state |
| Fix a source's content | New pack root, re-measured weights, new revision | Rebuild a pack in place |

Make one change per fork, name its parent, and compare the two at matched steps under an identical evaluation setup.
A change usually improves its target benchmarks while moving unrelated ones. Adopt it only after scoring the full
suite.
