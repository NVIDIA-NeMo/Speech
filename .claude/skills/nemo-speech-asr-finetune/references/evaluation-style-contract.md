# Evaluation Style Contract

Use this reference whenever reporting WER/CER for an ASR fine-tune.

## Default Metric

Unless the user explicitly requests a different metric, evaluate WER with capitalization and punctuation removed. This
default keeps model quality comparisons focused on lexical recognition instead of punctuation/casing style drift.

Still preserve raw prediction manifests. Raw WER is useful as a diagnostic, especially when the fine-tuning labels use
a different style than the original checkpoint, but it is not the default success metric for this skill.

Change the default only when the user asks for raw/cased/punctuated scoring. If the target product requires
punctuation, casing, ITN, or symbolic formatting to be part of the ASR metric, treat that as an explicit user
requirement and record the changed metric contract.

## Two-Step Evaluation

First transcribe with stable decoding, no AMP, and bfloat16 compute:

```bash
python examples/asr/speech_to_text_eval.py \
  model_path=/exp/asr-ft/checkpoints/best.nemo \
  dataset_manifest=/data/test.json \
  output_filename=/exp/asr-ft/test_predictions.json \
  batch_size=32 \
  amp=false \
  compute_dtype=bfloat16 \
  matmul_precision=high
```

For RNNT/TDT, enable CUDA graphs when supported:

```bash
python examples/asr/speech_to_text_eval.py \
  model_path=/exp/asr-ft/checkpoints/best.nemo \
  dataset_manifest=/data/test.json \
  output_filename=/exp/asr-ft/test_predictions.json \
  batch_size=32 \
  amp=false \
  compute_dtype=bfloat16 \
  matmul_precision=high \
  rnnt_decoding.strategy=greedy_batch \
  rnnt_decoding.greedy.use_cuda_graph_decoder=true
```

Then score the saved predictions with the default text-processing contract:

```bash
python examples/asr/speech_to_text_eval.py \
  dataset_manifest=/exp/asr-ft/test_predictions.json \
  only_score_manifest=true \
  text_processing.do_lowercase=true \
  text_processing.rm_punctuation=true \
  use_cer=false
```

The WER printed by the first transcription command is raw unless the same `text_processing` overrides were passed to
that command. Treat the score-only command above as the default metric.

If the user requested raw scoring, run score-only without `text_processing.do_lowercase` and
`text_processing.rm_punctuation`, and label the result as raw WER.

## Punctuation Coverage

`text_processing.rm_punctuation=true` removes the punctuation marks configured in
`text_processing.punctuation_marks`. If predictions contain punctuation or symbols outside that list, either:

- Override `text_processing.punctuation_marks` with the full set relevant to the target style.
- Or create a derived scored manifest where both `text` and `pred_text` are transformed by the same documented
  normalizer, then score that manifest.

Use the second option when the target style removes broad Unicode punctuation/symbol categories, normalizes whitespace,
or applies language-specific text normalization that NeMo's built-in punctuation helper does not express.

## Reporting

For each model variant, report:

- Default WER/CER from score-only evaluation with capitalization and punctuation removed.
- Raw WER/CER only when it helps explain style mismatch or the user requested it.
- The exact prediction manifest and text-processing settings.
- Whether the target references were raw labels or derived style-normalized labels.

If raw WER and default WER differ substantially, treat that as a transcript-style finding before changing training
hyperparameters. Do not claim an acoustic/model regression from raw WER alone when the default normalized WER improved.
