# Stage 5: Refinement And Iteration

Use this stage after the first fine-tuning run has a standalone evaluation result. The goal is to decide what to change
next from evidence, not from guesswork.

## Evaluation Matrix

Always compare:

- Baseline pretrained model on the domain validation/test set.
- Fine-tuned model on the same domain validation/test set.

Add a general or out-of-domain guardrail set only when it matches the user's goal, such as preserving same-language
general ASR quality, broad production behavior, or a known existing benchmark. If the user is intentionally changing
language, task, script, domain, or deployment scope, do not assume a generic guardrail is meaningful; pick validation
sets that reflect the desired behavior.

Report standalone `speech_to_text_eval.py` WER for each row. In-training `val_wer` is only for checkpoint selection.
For small-domain adaptation, avoid optimizing only one tiny target set when the model must still work broadly; track the
smallest goal-relevant guardrail set that would reveal unacceptable regressions.

Example tracking table:

| Run | Checkpoint | Target WER | Optional Guardrail WER | Notes |
| --- | --- | --- | --- | --- |
| baseline | pretrained |  |  | no fine-tune |
| ft-001 | best single |  |  | first pass |
| ft-001-avg | averaged |  |  | keep only if better |

## Error Analysis Loop

1. Transcribe the held-out domain set with the current best checkpoint.
2. Compute per-utterance WER/CER and sort from worst to best.
3. Before treating raw WER as acoustic/model quality, rescore with the chosen transcript normalization and compare
   raw vs the default capitalization/punctuation-normalized WER. This is mandatory. A large gap usually means the
   Stage 2 transcript-style preflight or the evaluation-style contract is incomplete.
4. Categorize errors by actionable patterns: numbers, named entities, abbreviations, rare domain words, commands,
   readbacks, punctuation/capitalization, language/task tags, accents, noise, long utterances, and clipping/VAD issues.
5. Count errors by category and inspect representative audio. Separate model errors from label/audio defects.
6. Choose one or two interventions for the next run. Avoid changing LR, data mix, tokenizer, and decoding all at once.

Do not train on validation or test transcripts. If error analysis uses a public or user-visible test set to guide new
data generation, create a new blind holdout before claiming final quality.

## Intervention Choices

Prefer the least invasive intervention that matches the error pattern:

- Data issue: fix labels/audio, remove broken samples, adjust `min_duration`, `max_duration`, `min_tps`, or `max_tps`.
- Rare vocabulary or entities: add more real examples if available; otherwise add carefully reviewed synthetic examples.
- Overfitting or regression: lower LR, reduce `max_steps`, stop at an earlier checkpoint, or increase a generic/guardrail
  blend only when preserving that behavior is part of the user's objective.
- Domain underfitting: raise target-domain real-data weight, add targeted data, or run a lower-LR domain-focus phase.
- Decoding issue: compare decoder options, prompts, punctuation/capitalization settings, or CTC/RNNT head for hybrids.
- Tokenization issue: revisit tokenizer only when transcript language/domain coverage cannot be represented well by the
  existing tokenizer.

## Source-Weighted Rebalancing

Use source-weighted rebalancing when standalone evaluation shows that a few train sources, domains, or acoustic styles
are underfitting and those sources are important to the user's goal. Do not use it to chase a public test set without a
new blind holdout.

Recipe:

1. Ensure every sample has a stable source/domain tag, either in manifest metadata such as `source_dataset` or in
   Lhotse `input_cfg.tags`.
2. Run standalone evaluation and compute source-wise WER using the default evaluation-style contract.
3. Decide whether weights should reflect user priority, source-wise validation errors, dataset hours, or a blend of
   those signals.
4. Apply a floor so low-error sources are not removed entirely, and cap extreme weight jumps unless the user explicitly
   wants domain-only adaptation.
5. Write a new Lhotse `input_cfg` with normalized weights. Prefer changing weights over duplicating manifests.
6. Re-run duration-bin estimation and OOMptimizer, or at least verify the previous profile is still valid when the
   source union is unchanged.
7. Fine-tune from the current best checkpoint with a lower LR refinement phase, commonly `1e-5` or `5e-6`.
8. Compare the previous best, the rebalanced final checkpoint, the best individual checkpoint, and any averaged model
   with the same standalone scoring contract.

When using `estimate_data_weights.py`, validate the YAML shape expected by the current script before running it. If the
new weights were influenced by validation/test performance, record that in the ledger. For final claims, evaluate on a
holdout that did not influence the reweighting decision.

## Targeted Synthetic Data

Synthetic data is most useful when it fills a measured gap. Generate small, targeted batches for the worst categories
instead of flooding the run with generic synthetic audio.

Recommendations:

- Keep synthetic text TTS-friendly: expand symbols and ambiguous abbreviations when needed, and avoid text forms that a
  synthesizer will read incorrectly.
- Match target-domain acoustics only when the target deployment needs them; generic noise can hurt.
- Filter synthetic audio with ASR or manual spot checks before adding it to training.
- Add synthetic data as a separately weighted Lhotse input source so it can be ablated.
- For small-domain adaptation, keep real target-domain audio dominant unless standalone WER proves otherwise.

## Run Ledger

Use `assets/experiment-ledger-template.md` as the starting template for experiment notes.

Maintain a compact run ledger with:

- Data sources and blend weights.
- Transcript style policy, any manifest transform, and whether it differs from the original checkpoint output style.
- LR, `max_steps`, warmup, precision, and batch profile.
- Duration/token filters and number of examples filtered.
- Final `.nemo`, best individual checkpoint-derived artifact, averaged artifact, and the keep/drop decision.
- Raw WER/CER and default capitalization/punctuation-normalized WER/CER when both are informative.
- Target-set WER and any goal-relevant guardrail WER.
- GPU utilization evidence from training steps, not validation/checkpoint/export windows.
- Decision for the next run.

## Late-Stage Curriculum Pattern

Use curriculum-style staged fine-tuning only after simpler refinements stop improving standalone WER. Try data cleanup,
filter fixes, blend-weight changes, targeted data additions, decoding choices, and tokenizer decisions first.

For small-data domain adaptation, use short lower-LR phases instead of one long aggressive run:

1. Foundation phase: preserve required broad behavior with a mix of source and target-domain data when that is a goal.
2. Domain-focus phase: increase real target-domain and targeted data weight, lower LR.
3. Final refinement phase: lower LR again and evaluate carefully; this phase can regress.

Typical starting points:

- First small-domain phase: `model.optim.lr=3e-5`.
- Follow-up domain-focus phase: `model.optim.lr=1e-5`.
- Final refinement phase: `model.optim.lr=5e-6` or lower.

Use `trainer.max_steps` for each phase, checkpoint by `val_wer`, and run standalone evaluation after each phase. Keep
the best phase, not necessarily the last phase.
