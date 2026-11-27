---
name: train-dflash
description: Train or adapt a DFlash or DFlash2 speculative draft against a frozen SALM speech-LLM target with NeMo speechlm2. Use for online argmax or offline model-generated labels, pretrained draft adaptation, projection-only ablations, checkpoint resume/export, and acceptance evaluation. For fine-tuning the speech encoder or target LLM itself, use finetune-speech-llm instead.
---

# Train a DFlash speech draft

Use `examples/speechlm2/dflash_train.py` with `examples/speechlm2/conf/dflash.yaml`.
This trains a separate draft; the entire SALM target, including its audio encoder and modality projection, stays frozen.
The ordinary `salm_train.py` entry point does not enable draft training.

Read these repository sources as needed; paths below are relative to the repository root:

- `docs/source/speechlm2/dflash_training.rst`: supported configuration, labels, warm starts, and projection-only training.
- `examples/speechlm2/conf/dflash.yaml`: current draft defaults; inherits target/data/trainer settings from `salm_automodel.yaml`.
- `nemo/collections/speechlm2/parts/dflash.py`: audio expansion, feature capture, packing, losses, and export.
- `docs/source/speechlm2/vllm_dflash.rst`: serving setup and draft architecture registration.
- `docs/source/starthere/install.rst` and `pyproject.toml`: training environment and pinned Automodel dependency.

## 1. Establish the experiment

Reuse the user's specified target, draft, data, and compute allocation. Identify any missing checkpoint paths,
training budget, and evaluation set before launching work that depends on them. Record the Speech and Automodel
revisions, resolved config, target/draft paths, data blend, seed, and output directory with the job.

Keep three checkpoint roles separate:

| Setting | Role | State restored |
|---|---|---|
| `model.init_from_checkpoint` | Frozen audio-conditioned SALM target | Target weights |
| `dflash.init_from_pretrained` | Local, compatible HF draft export | Draft weights only; fresh optimizer/scheduler/steps |
| `exp_manager.resume_from_checkpoint` | Interrupted DFlash training run | Draft and training state, including optimizer/scheduler/steps |

Use a new `exp_manager.explicit_log_dir` for an independent experiment. The inherited recipe enables auto-resume;
disable `exp_manager.resume_if_exists` for a fresh run so old checkpoints cannot silently override a warm start.
For a continuation, verify the resumed global step and optimizer/scheduler state rather than treating an HF export
as a full training checkpoint. Keep the same frozen target and trainable parameter selection when resuming.

## 2. Match both architectures

Read the target's saved config and tokenizer metadata. Derive the `model` settings from that target's known working
SALM recipe, including encoder/adapter type, language backbone, tokenizer paths, prompt format, and audio token
estimator. Merely setting `model.init_from_checkpoint` does not replace the inherited model architecture.
Keep strict target initialization enabled; investigate missing parameters instead of disabling it to make a run start.

For a warm start, read the draft's `config.json` and match its hidden size, layer count, attention/MLP dimensions,
rotary settings, target-layer taps, mask token, and DFlash2 convolution/selector settings. Strict loading checks tensor
coverage and shapes, but cannot detect every semantic mismatch, such as different same-shaped layer taps.
Use a trainable floating-point draft export; the packed NVFP4 inference artifact cannot directly initialize this
BF16 training path. A null `dflash.init_from_pretrained` initializes a fresh draft, not pretrained weights.

The shipped recipe has six draft layers and target taps `[1, 5, 19, 29, 41, 51]`. These are zero-based target block
outputs captured before the final norm, not six copies of the final layer. Do not transfer these IDs to a different
backbone without checking its depth and the draft's feature contract.

Supported topology is target `tp_size=pp_size=cp_size=1`, with draft DP/FSDP and compatible target EP.
Set topology for the allocated GPUs; the inherited `ep_size=8` is not a one-GPU configuration.
Use a supported BF16 precision (`bf16-flash` or `bf16-automodel`), and leave `use_fused_linear_ce=false`.
Packed decoder inputs are supported; packed encoder execution is not enabled by this entry point.

## 3. Configure labels, adaptation, and data

Preserve explicit user choices. Otherwise start with the shipped `dflash.label_source=target_argmax`:
these are the frozen target's greedy next-token labels under the original teacher-forced text/audio context.
They are not generated rollouts or soft-logit distillation. Reference tokens and audio-conditioned features remain
the draft's input context. Validation uses the same label source. Use `ground_truth` to train on the supplied dataset
tokens, whether they are human references or model-generated labels (see the workflow below).

Current recipe defaults are block size **8**, `loss_decay_gamma=4.0`, `num_anchors=512`, and constant AdamW LR **6e-4**.
Block16 or gamma7 are experiment overrides, not repository defaults. Supply `dflash.optimizer` and
`dflash.lr_scheduler` explicitly when an adaptation schedule is wanted; the training guide has an example.
For an extended run, inspect the restored LR and scheduler horizon: increasing `trainer.max_steps` alone does not
redesign the learning-rate schedule. Specify whether a step budget is a total or additional steps.

- **Full-draft adaptation:** `dflash.projection_only=false` (default).
- **Projection-only ablation:** `dflash.projection_only=true` requires DFlash2 and a compatible pretrained draft.
  Only the draft feature projection `fc.weight` trains, not the SALM audio projection. Frozen draft layers still
  transmit gradients to `fc`. Compare against full-draft training with matched data, labels, schedule, and initialization.
- **Block8 to block16:** set `dflash.block_size=16` for a weights-only warm start while preserving the matching
  architecture and taps. Changing block size does not require changing weight shapes. Do not assume projection-only
  adaptation or inference beyond the trained block length will perform as well; measure them.

For packed training, configure `model.packed_sequences=true` and the matching Lhotse token-based sampler
(`use_multimodal_sampling`, `use_packed_sequence_sampling`, `measure_total_length`, `batch_tokens`, and no fixed
`batch_size`). Verify the audio token estimator matches the encoder and chunking. `num_anchors` is per row:
packed THD uses one row per rank; padded BSHD can allocate `batch_size * num_anchors` blocks per rank.
Reduce the token budget or anchor count when memory requires it rather than assuming packing removes draft-logit cost.

For long-audio adaptation, inspect duration filters and actual sampled batches before changing blend weights.
Dataset draw weights are not anchor fractions: long examples supply more eligible token positions.
If targeting a long-audio anchor fraction, define the duration threshold and estimate exposure with the actual
sampler, token budget, block size, and anchor cap. Include short-only batches in the denominator and keep evaluation
data separate. Record estimated exposure as an estimate, not a measured full-run fraction.

### Train DFlash2 from offline model-generated labels

Use this workflow when the user supplies generated transcripts or wants to generate training labels in advance.
It uses the existing dataset-label path; `ground_truth` names that code path and does not require human annotation.

1. Generate transcripts from the training audio, preferably with the same frozen target and prompt used for serving.
   Record the teacher checkpoint, prompt, and decoding settings. Retain sample IDs and exclude failed or truncated
   generations from the training-label artifact until they are repaired. Keep the original references separately.
2. Write a new training manifest with generated text as the supervised assistant response, preserving audio paths,
   offsets, durations, and prompt context. For ordinary `lhotse_as_conversation` ASR input this is the cut's
   `supervisions[0].text` (or NeMo ASR manifest `text` before conversion). For a conversation dataset, replace the
   intended assistant text turns while preserving user/audio turns. Inspect the formatted examples and loss mask;
   custom preference/task adapters can select a different response field.
3. Point the existing data loader at that artifact and select dataset labels. For example, merge this into the
   verified run config; the cuts must retain the prompt context used for generation:

   ```yaml
   dflash:
     variant: dflash2
     label_source: ground_truth
   data:
     train_ds:
       input_cfg:
         - type: lhotse_as_conversation
           cuts_path: /path/to/generated-train-cuts.jsonl.gz
           audio_locator_tag: ${model.audio_locator_tag}
   ```

4. Run the same smoke/resume/export checks as for reference-label training. For draft validation against this teacher,
   prepare a separate held-out generated-label set. Keep human evaluation references for WER and never train on them.

The target still computes audio-conditioned hidden features during training. Generated response tokens provide both
the teacher-forced context and supervision. This differs from `target_argmax`, which recomputes next-token labels
under whatever response context is in the dataset; it would override the supplied labels. No new label-source mode,
separate label-file argument, or cached hidden-state loader is required for offline text labels.

## 4. Smoke, then train

Create a run config such as `examples/speechlm2/conf/dflash_local.yaml` inheriting `dflash` and `_self_`, with the
verified target, draft, data, schedule, and output settings above. Resolve it before submission:

```bash
python examples/speechlm2/dflash_train.py --config-name=dflash_local --cfg job --resolve
```

Check for unresolved data paths and unintended inherited defaults. For a single allocated eight-GPU node, a short
smoke invocation is:

```bash
torchrun --standalone --nproc_per_node=8 examples/speechlm2/dflash_train.py \
  --config-name=dflash_local \
  trainer.devices=8 trainer.num_nodes=1 trainer.max_steps=25 \
  trainer.limit_train_batches=25 trainer.val_check_interval=25 trainer.limit_val_batches=2 \
  exp_manager.explicit_log_dir=/path/to/new-smoke-run exp_manager.resume_if_exists=false
```

Use a representative held-out validation subset. For multi-node runs, use the site's existing launcher/rendezvous
and set `trainer.num_nodes` to the allocation; do not use independent `--standalone` processes on each node.
Use the project's established source-staging and storage workflow rather than copying a source tree per worker.

Verify finite loss and gradients, the intended trainable parameters, advancing steps, memory, and steady-state step
time after compilation. Check `train/dflash_skipped_step` and its reason metrics: synchronized skips advance training
without learning from those batches. Confirm validation processes valid anchors; an all-skipped epoch logs
`val_acc=0` to satisfy checkpoint monitoring and is not evidence of useful validation.

For a new distributed recipe, include checkpoint save/restore and HF export reload in the smoke test. Verify the
resumed step and LR, and that all expected distributed checkpoint shards exist. Scale to the requested run after
these checks pass. Monitor progress from logs/checkpoints, not queue state alone. On repeated failure, diagnose the
first failing rank and data/storage access before retrying; retain the last known good checkpoint.

## 5. Export and measure acceptance

`SALMDFlashModule.on_train_end` writes a draft-only HF export to `dflash.output_dir`; a relative path resolves under
`trainer.log_dir`. An interrupted job may have a resumable checkpoint without that final export. Verify which step
an export represents and strictly reload it with the matching draft class before serving it with the frozen target.
Retain the resumable checkpoint when further training is planned.

Follow the serving guide for the installed vLLM/plugin pair. DFlash2 uses speculative `method="dflash"`; confirm the
DFlash2 architecture/selector is actually selected. Start with `num_speculative_tokens=block_size-1` for a matched
training/inference block. Smaller K is a separate inference evaluation; K proposals plus one visible anchor make
a runtime block of K+1 positions. Do not rewrite the export architecture to bypass serving compatibility checks.

Compare baseline and adapted drafts on identical sample IDs, target, prompts, decoding limits, precision, K, and
scoring. Include both long- and short-form data when changing the training duration mix. Validate greedy outputs
against target-only inference and report WER, throughput/latency, and acceptance; higher acceptance alone is not
proof of higher serving throughput or unchanged transcription quality.

For the convention that includes the target/bonus token, aggregate **MAL = 1 + accepted draft tokens / draft rounds**
from counters scoped to the measured requests. Report K and the counter definition; do not average per-job MAL
without weighting by draft rounds. For per-position acceptance, distinguish accepted-at-position / proposed-at-position
from conditional acceptance given that the earlier prefix survived. Training `accept_len` is teacher-forced and
does not substitute for serving MAL.

Record evaluated/expected sample counts, failures, truncations, concurrency, and export step with every result.
Retry failed samples under the same settings and preserve their IDs. Do not combine rerun counters with an original
run's failed partial-generation counters unless those partial contributions can be excluded; label any mixed-run
aggregate as provisional when exact attribution is unavailable.
