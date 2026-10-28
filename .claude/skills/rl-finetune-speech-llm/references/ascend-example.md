# Example: Mandarin–English transcription with ASCEND

This example asks a focused question: can GSPO turn the model's better sampled transcriptions into better greedy
transcriptions? It uses the public Nemotron Transcribe seed, 500 ASCEND training recordings and mixed error rate
(MER). Mandarin characters and English words each count as one scoring unit.

The original 20-update trial improved test MER from **19.469% to 18.674%**, with most of the gain in mixed speech.
English-only errors increased. The extended learning curve is still being evaluated; longer training is not yet a
measured recommendation.

## Set up the supported code

Follow [Execution](execution.md) for the native SpeechLM policy, actor environments and canonical exporter. Use
coordinated Speech and RL revisions containing both:

- `nemo.collections.speechlm2.parts.metrics.mer.mer_tokens` in Speech;
- the `reward_metric="mer"` branch in RL's `nemo_rl/environments/asr_reward.py`.

These example additions are local contributions pending publication. The general GSPO PR alone does not contain
MER support. The archived experiment used a matching v0.7.0 stack; it does not qualify arbitrary current mainlines
inside that image. No NeMo-Skills feature branch is needed for MER.

Set `$SPEECH`, `$RL`, `$SPEECHLM_GSPO_PYTHON` and `$SERVING_PYTHON` as in the execution guide. Place model, data and
outputs on storage visible to every worker. Run preparation on CPU; decoding/export use one eight-GPU node and
training uses four eight-GPU nodes.

```bash
export ASCEND_SCRIPTS="$SPEECH/.claude/skills/rl-finetune-speech-llm/scripts"
export ASCEND_DATA=/shared/data/ascend500
export ASCEND_RESULTS=/shared/results/ascend-gspo
export SPEECHLM_GSPO_MODEL=/shared/models/nemotron-transcribe
hf download nvidia/Nemotron-3.5-Transcribe-30B-A4B \
  --revision 471d3b578578edef8c1ed919190140ef7c1f6d8d \
  --local-dir "$SPEECHLM_GSPO_MODEL"
"$SERVING_PYTHON" "$ASCEND_SCRIPTS/prepare_ascend.py" \
  --output "$ASCEND_DATA" --train-samples 500 \
  --revision 737e9800ae31be9932ba8464c80366559bd28424
```

The helper selects the 500 lowest SHA-256 hashes of `42:<training-id>`, converts audio to mono 16-kHz WAV, and
preserves the official development/test membership (1,130/1,315 rows). It writes `train.parquet`, JSONL manifests,
a 32-row training pilot and `DATA_DONE.json` with membership, hashes and duration. The 500 recordings total about
0.44 hours. Use the same seed/data revisions to reproduce this subset.

This reproduction uses **“Transcribe the audio into text.”** with no system message and text before audio. It is the
historical experiment prompt, not the seed's established ASCEND benchmark prompt. For a new adaptation, start with
the [task-specific prompt](tasks-and-presets.md#prompt-defaults) and remeasure seed quality and training signal;
do not attribute a prompt-only gain to RL. The reference
goes to the reward scorer, never into the model prompt. MER reward is `1 − edits/reference_units`, unbounded below;
empty normalized references are rejected. It does not spell out numbers or translate mixed speech into one language.

## Measure signal and the seed

Use the prepared generation environment. `--pilot` adds eight temperature-0.8 responses to each greedy response;
these are offline probes, separate from the 64 online responses per training group.

```bash
export VLLM_PLUGINS=nemo_speechlm VLLM_USE_DEEP_GEMM=0
"$SERVING_PYTHON" "$ASCEND_SCRIPTS/evaluate_asr.py" \
  --model "$SPEECHLM_GSPO_MODEL" --manifest "$ASCEND_DATA/pilot32.jsonl" \
  --output "$ASCEND_RESULTS/pilot.jsonl" --metric mer --pilot --tensor-parallel-size 8
"$SERVING_PYTHON" "$ASCEND_SCRIPTS/evaluate_asr.py" \
  --model "$SPEECHLM_GSPO_MODEL" --manifest "$ASCEND_DATA/validation.jsonl" \
  --output "$ASCEND_RESULTS/seed-dev.jsonl" --metric mer --tensor-parallel-size 8
```

Inspect raw alternatives as well as reward spread. The original pilot had 14/32 nonconstant groups and 8/32 groups
with an alternative better than greedy. That justified a small trial; it did not predict its eventual gain. This
reference experiment trains on the deterministic 500-row pool and uses online nonconstant-group admission. It does
not claim the pool was selected by an additional offline mining pass. For a new mined variant, use only training
rows, preserve monolingual/mixed coverage and record the changed membership separately.

## Train a bounded comparison

Launch Ray using the native RL cluster workflow, then run the driver once inside the four-node allocation:

```bash
export SPEECHLM_GSPO_DATA="$ASCEND_DATA/train.parquet"
export SPEECHLM_GSPO_OUTPUT_DIR="$ASCEND_RESULTS/train-r1"
cd "$RL"
"$SPEECHLM_GSPO_PYTHON" examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml \
  grpo.max_num_steps=20 grpo.max_num_epochs=64 \
  checkpointing.save_period=10 checkpointing.keep_top_k=3
```

The base recipe supplies AdamW at `1e-6`, five-update warmup, 64 responses per recording and 20 admitted groups per
update. The update surface and FP32 policy storage stay unchanged; CTC and MTP remain frozen. MER is selected by
row metadata. The epoch allowance permits repeated use of the small pool; the step limit controls the run length.

Use development results to decide whether to extend. An authorized 100-update curve can set `grpo.max_num_steps=100`,
`checkpointing.save_period=20` and `checkpointing.keep_top_k=5`. The same output directory resumes complete native
optimizer/scheduler/data state; a fresh directory starts an independent run. Do not resume from a serving export
and label it an exact continuation. Record completed steps and unique recordings that actually supplied gradients.

## Export, select and evaluate

Use the [canonical export commands](execution.md#export-the-actual-trained-state) for each saved checkpoint.
Set `$EXPORT_OUTPUT` to the resulting full HF directory, then decode development:

```bash
"$SERVING_PYTHON" "$ASCEND_SCRIPTS/evaluate_asr.py" \
  --model "$EXPORT_OUTPUT" --manifest "$ASCEND_DATA/validation.jsonl" \
  --output "$ASCEND_RESULTS/candidate-dev.jsonl" --metric mer --tensor-parallel-size 8
```

Use a unique output path for each checkpoint. The decoder preserves the training prompt and uses greedy decoding,
stop token 11, context 62,464, output cap 17,278, TP8, no prefix caching and no speculative decoding. Compare models
under the same settings. Output contains IDs, raw predictions, error counts and finish reasons; summaries contain
manifest hashes and corpus totals.

Report Mandarin-only, English-only and mixed slices from the saved predictions, without decoding again:

```bash
python - "$ASCEND_DATA/validation.jsonl" "$ASCEND_RESULTS/candidate-dev.jsonl" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

manifest, predictions = [
    [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for path in sys.argv[1:]
]
languages = {row["id"]: row["language"] for row in manifest}
assert len(languages) == len(manifest) == len(predictions)
assert {row["id"] for row in predictions} == set(languages)
totals = defaultdict(lambda: [0, 0, 0])
for row in predictions:
    for key in ("all", languages[row["id"]]):
        values = totals[key]
        values[0] += 1
        values[1] += row["errors"]
        values[2] += row["reference_tokens"]
for language, (rows, errors, units) in sorted(totals.items()):
    print(f"{language}: {rows} recordings, MER {100 * errors / units:.3f}% ({errors}/{units})")
PY
```

Repeat the slice calculation for the seed predictions. Choose the checkpoint using development MER and the declared
retention constraints, then decode `test.jsonl` for that checkpoint and the seed and report the same slices.
Evaluate a fixed general-English retention manifest with `--metric wer`.
The existing project's test was inspected previously; its new scores are reused-benchmark results. A new customer
experiment should leave its test untouched until selection is final.

## What the original short run established

| Measurement | Public seed | Selected step20 |
|---|---:|---:|
| Development MER, 1,130 rows | 19.362% (2,404/12,416) | 18.895% (2,346/12,416) |
| Test MER, 1,315 rows | 19.469% (2,253/11,572) | 18.674% (2,161/11,572) |
| Mixed test errors | 1,438 | 1,343 |
| Mandarin-only test errors | 488 | 481 |
| English-only test errors | 327 | 337 |
| English retention WER, 100 rows | 1.429% (27/1,889) | 1.482% (28/1,889) |

The test gain was **4.08% relative**, not a broad claim about all English or Mandarin speech. Twenty updates used
400 retained groups from 240 distinct recordings. The English-retention difference was one edit, so report the
counts and uncertainty rather than treating that tiny slice as a full regression suite. Frozen CTC weights also do
not prove unchanged timestamps when the speech encoder is updated.

On development, mixed speech supplied 57 of the 58 fewer errors; Mandarin-only errors rose by 6 and English-only
errors fell by 7. A paired utterance bootstrap gave a 95% interval of approximately **−1.207 to +0.040 percentage
points** for the overall MER change (candidate minus seed). Development has only three speakers. The observed gain
therefore motivates the extension experiment but does not establish a reliable broad improvement.
