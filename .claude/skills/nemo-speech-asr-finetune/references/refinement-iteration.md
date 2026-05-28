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
3. Categorize errors by actionable patterns: numbers, named entities, abbreviations, rare domain words, commands,
   readbacks, punctuation/capitalization, language/task tags, accents, noise, long utterances, and clipping/VAD issues.
4. Count errors by category and inspect representative audio. Separate model errors from label/audio defects.
5. Choose one or two interventions for the next run. Avoid changing LR, data mix, tokenizer, and decoding all at once.

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

Maintain a compact run ledger with:

- Data sources and blend weights.
- LR, `max_steps`, warmup, precision, and batch profile.
- Duration/token filters and number of examples filtered.
- Best checkpoint and standalone WER/CER.
- Target-set WER and any goal-relevant guardrail WER.
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
