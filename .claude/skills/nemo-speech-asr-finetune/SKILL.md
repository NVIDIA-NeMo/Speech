---
name: nemo-speech-asr-finetune
description: Guide NeMo Speech users through ASR fine-tuning with container setup and Lhotse training.
---

# NeMo Speech ASR Fine-Tuning

Use this skill when a user wants to fine-tune a NeMo Speech ASR model, choose a checkpoint, adapt a tokenizer,
configure Lhotse dataloading, train, average checkpoints, or evaluate a fine-tuned ASR `.nemo` checkpoint.
Also use it for post-run refinement planning after fine-tuning.

Default posture:

- Use the NeMo container unless the user explicitly asks for local execution.
- Prefer Lhotse for train and validation dataloaders.
- Use `trainer.max_steps`, not `trainer.max_epochs`.
- Use `val_wer` as the checkpoint monitor for validation.
- By default, evaluate WER without capitalization and punctuation effects. Change that only when the user explicitly
  asks for raw/cased/punctuated scoring.
- Report final quality from standalone evaluation, not only in-training validation logs.

## Staged Workflow

Load only the reference file needed for the current stage:

1. Setup and checkpoint selection: read `references/setup-checkpoints.md`.
2. Data prep, transcript-style preflight, Lhotse, bucketing, validation dataloader, and blends: read
   `references/data-lhotse.md`.
3. Architecture detection, tokenizer changes, and AED/Canary multitask metrics: read
   `references/architecture-tokenizer-metrics.md`.
4. Training, checkpoint averaging, and evaluation: read `references/training-evaluation.md` and, when reporting WER,
   `references/evaluation-style-contract.md`.
5. Post-run refinement, error analysis, curriculum, and general-vs-domain evaluation: read
   `references/refinement-iteration.md`.

If the user explicitly asks for parallel/sub-agent work, split the work by these same stages. Keep each agent scoped to
one stage and have the main agent integrate the final command/config.

## Core Commands

Generic fine-tuning uses `examples/asr/speech_to_text_finetune.py`. For architecture-specific recipes, route to:

- CTC: `examples/asr/asr_ctc/speech_to_text_ctc_bpe.py`
- RNNT: `examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py`
- Hybrid RNNT/CTC or TDT/CTC: `examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py`
- AED/Canary: `examples/asr/speech_multitask/speech_to_text_aed.py`

Always check the current repo docs before giving version-sensitive claims:

- `README.md`
- `docs/source/asr/fine_tuning.rst`
- `docs/source/asr/datasets.rst`
- `docs/source/dataloaders.rst`
- `docs/source/asr/featured_models.rst`
- `docs/source/asr/asr_checkpoints.rst`
- `nemo/collections/common/data/lhotse/dataloader.py`

## Non-Negotiable Pitfalls

- When changing Lhotse batch modes, explicitly null conflicting options. For OOMptimizer profiles, set
  `batch_size=null`, `batch_duration=null`, and `quadratic_duration=null` when adding `bucket_batch_size`.
- Set `model.validation_ds.use_lhotse=true`, but prefer static validation `batch_size` with bucketing disabled.
- Do not use fused loss/WER or tune `fused_batch_size` for RNNT/TDT fine-tuning guidance from this skill.
- Run the first OOMptimizer pass with default CLI settings; lower `--memory-fraction` only after a real training OOM.
- Run preflight checks before long jobs: disk space, free GPUs, manifest validity, and duration/text sanity.
- Before any fine-tuning, audit transcript style within and across all fine-tuning/validation/test sources. Do not
  train on mixed casing, punctuation, inverse-text-normalization, or symbol conventions; choose and fix one target style
  first, and compare it with the original checkpoint's prediction style when applicable.
- For small domain adaptation, start with a lower LR than large-data fine-tuning; do not blindly use `1e-4`.
- Do not train a tokenizer on validation or test transcripts.
- Do not ignore silent Lhotse filtering from `min_duration`, `max_duration`, `min_tps`, and `max_tps`.
- Do not use `amp=true` for inference/evaluation; use `amp=false compute_dtype=bfloat16`.
- Unless the user asks otherwise, report the default WER with capitalization and punctuation removed, and record any raw
  WER separately when it helps diagnose transcript-style mismatch.
- For AED/Canary, configure `multitask_metrics_cfg` so ASR and translation/task-specific samples are evaluated with
  the right constrained metrics.
- If checkpoint averaging is used, evaluate the averaged checkpoint and keep it only if it beats the best individual
  checkpoint.
