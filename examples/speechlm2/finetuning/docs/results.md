# Results: what fine-tuning a SALM checkpoint buys, and what it costs

This page summarizes the outcomes of the recipes in
[`tutorials/speechlm2/finetuning`](../../../../tutorials/speechlm2/finetuning/README.md). Each tutorial adapts one SALM
checkpoint (speech encoder + projection + LLM backbone, fine-tuned with LoRA on the LLM) to one benchmark. It measures
the base model with the prompt chosen on dev, trains, selects a checkpoint on dev, and decodes the test set once. All
numbers come from a single NVIDIA GB300 GPU. Costs assume $9 per GPU-hour, a typical on-demand price; spot capacity is
about half that.

## Sixteen tasks, one recipe

| Task | Benchmark | Metric | Base model (prompt chosen on dev) | Fine-tuned | Relative change |
|---|---|---|---:|---:|---:|
| ASR, new language | FLEURS Swahili | WER ↓ | 81.2 | 18.0 | −78% |
| Code-switching ASR | ASCEND (Mandarin–English) | MER ↓ | 18.1 | 11.0 | −40% |
| Accented conversational ASR | EdAcc | WER ↓ | 18.4 | 15.0 | −19% |
| Speech translation | CoVoST 2 En→Ca | BLEU ↑ | 15.2 | 27.8 | +83% |
| Spoken question answering | HeySQuAD | F1 ↑ | 43.3 | 84.4 | +95% |
| Intent classification | SLURP | accuracy ↑ | 50.6 | 87.9 | +74% |
| Speech emotion | CREMA-D | accuracy ↑ | 36.7 | 76.7 | +109% |
| Keyword spotting | Speech Commands v2 | accuracy ↑ | 94.8 | 97.4 | −51% errors |
| Environmental sound | ESC-50 | accuracy ↑ | 40.3 | 62.0 | +54% |
| Non-verbal vocalizations | VocalSound | accuracy ↑ | 76.7 | 91.3 | +19% |
| Music genre (linear probe, no training) | GTZAN | accuracy ↑ | 79.0 | 86.2 | +9% |
| Instrument family | NSynth | accuracy ↑ | 48.0 | 57.4 | +20% |
| Language identification | CommonLanguage (45 languages) | accuracy ↑ | 49.8 | 92.9 | +86% |
| Pronunciation assessment | speechocean762 | Pearson r ↑ | 0.217 | 0.678 | +213% |
| Anti-spoofing | ASVspoof 2019 LA | EER ↓ | 30.7 | 0.6 | −98% |
| Stuttering events | SEP-28k | macro-F1 ↑ | 13.0 | 62.0 | +377% |

Baselines use the prompt chosen on dev (out of two or three per task), scored on test, the same way the fine-tuned
checkpoint is selected. Each tutorial also prints the test-oracle (the best prompt on test) as an upper bound for
prompting alone.

**What to expect.**
- **New tasks gain the most.** New languages, translation, spoken QA, new label spaces over speech, scores and
  detectors move by 40–380% relative.
- **Non-speech audio starts higher.** The base model already knows a fair amount about sounds, vocalizations,
  instruments and music (40–79% before training), so those tasks gain 9–54% relative.
- **Near the ceiling, count errors instead.** Keyword spotting already starts at 95%, so it gains through error-rate
  reduction.
- **Conversational ASR in a well-covered language gains least.** On EdAcc, part of the −19% comes from the prompt:
  against the test-oracle prompt (16.2 WER) the gain is −8%. Many of the remaining errors are annotation conventions
  rather than recognition errors.
- **Variance.** Training is not bit-for-bit reproducible. On small test sets, expect ±1–6 points between runs; each
  tutorial states its tolerance.

## Cost per task

GPU-hours to reproduce one recipe end to end: baseline decodes, training (300–500 steps), exports, dev selection,
test. Costs were measured with a different checkpoint of the same model; treat them as estimates.

| Task family | GPU-hours (median, range) | Cost | Wall-clock |
|---|---:|---:|---:|
| Classification (8 tasks) | 1.2 (0.5–2.0) | $4–18 | ~1.2 h |
| ASR (3 tasks) | 1.5 (1.4–1.7) | $13–15 | ~1.6 h |
| Translation | 2.3 | $20 | ~2.3 h |
| QA, regression, detection (4 tasks) | 1.1–2.0 | $10–18 | 1.1–2.0 h |
| Medical ASR with synthetic data ([case study](medical-asr-synthetic-data.md)) | ~14 | ~$125 | about one day |

- **Evaluation is half the bill.** Decoding takes about half the GPU time and training the other half. Each exported
  candidate adds a few minutes of model-loading time, so keep 2–3 candidates, not 10.
- **Fixed costs dominate training.** Model loading, validation and checkpoint saving take 20–60% of a training run.
  Steps on a larger dataset cost about the same as steps on a small one.
- **Exploration costs more than reproduction.** Finding a recipe for a new task costs 1–2× one reproduction when the
  first recipe you try works. On hard benchmarks it can be 8–11×.

## How much data you need

Each row trains the same recipe for the same number of steps on a random subset of the training data
(class-stratified for classification) and reports test scores. These runs used a different checkpoint of the same
model, so the base and full-data numbers differ from the table above; compare within this table.

| Task | Base model | ~10 per class / 50 clips | 10% of the data | Full data |
|---|---:|---:|---:|---:|
| Emotion, 6 classes (accuracy) | 38.3 | 43.4 (60 clips, 2 min of audio) | 62.6 | 75.5 |
| Intent, 60 classes (accuracy) | 47.3 | 71.0 (596 clips) | 83.4 | 86.9 |
| New-language ASR, Swahili (WER ↓) | 73.7 | 40.8 (50 clips, 13 min) | 31.8 (1.3 h) | 20.8 (13.5 h) |
| Pronunciation scoring (Pearson r) | 0.18 | 0.68 (50 clips) | 0.65 | 0.62 |

- **Output-format tasks:** when the model mainly has to learn an output format or scale (scores, structured answers),
  about 50 examples are enough.
- **New label spaces:** 10 examples per class give a real gain, and a few hundred per class get within a few points of
  the full dataset.
- **New language:** 13 minutes of audio already captures more than half of the full-data gain.
- **Cost doesn't shrink with data:** compute is set by the step count, so each run costs about 1 GPU-hour whatever the
  dataset size.

## Prompting vs fine-tuning

Try prompts first; they're free. Name the language and script, the domain, and the exact output format.
`prompt_search.py` automates the search: an LLM proposes prompts from the model's dev errors (about 20 minutes per
task). The comparison below was measured with a different checkpoint of the same model; compare within the table.

| Task (test) | Best hand-written prompt | Automated prompt search | Fine-tuned |
|---|---:|---:|---:|
| Emotion (accuracy) | 38.3 | 37.2 | 75.5 |
| Intent (accuracy) | 47.3 | 48.2 | 86.9 |
| Pronunciation (Pearson r) | 0.18 | 0.36 | 0.62 |
| Medical ASR (WER ↓) | 30.8 → 23.8 (naming "English, Latin alphabet") | — | 12.1–12.8 |

- **Prompting fixes format errors.** It works when the base model gets the output format or script wrong: it doubled
  the pronunciation-score correlation, and stopped the model from writing accented English in Devanagari.
- **Prompting doesn't add knowledge.** On label-choice tasks, prompt gains on a small dev set often fail to transfer.
- **Fine-tuning wins.** It beats the best prompt on every task.
