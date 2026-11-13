# TTS Fine-Tuning Experiment Ledger

## Goal

- Task (task file, e.g. `references/new-voices-supported-language/README.md`):
- Voice(s), language (real / synthetic):
- Product use (typical inputs: long sentences, numbers, one-word replies):
- Release checkpoint:
- User's target, if any (e.g. CER no higher than X, not worse than the release):

## Setup

- NeMo Speech tree + commit (no git checkout: the installed `nemo-toolkit` version, `pip show nemo-toolkit`):
- Learning rates: two (default; `Parameter Group 0` and `1` in the log) or one 5e-6 (alternative; why):
- Container / image (default `nvcr.io/nvidia/nemo-speech:26.07.00`):
- Codec, phoneme tokenizer, text tokenizer paths:
- Model caches (`HF_HOME`, `NEMO_CACHE_DIR`; online or offline):

## Data

- Source recordings and hours (train / val / test):
- Short utterances (count, source):
- ASR data check (optional, off by default): run? if yes: ASR model, CER threshold, dropped count (listened to?)
- Audio format (mono / converted from stereo; clipped clips reported):
- Leading silence cut (clips) / trailing silence fix (clips):
- espeak-ng version (must be 1.52.0):
- Reference clips: training rule (most similar, ≥ 3 s; utterances ≥ 3 s below 0.6 dropped: count); val / test /
  evaluation clips per voice (chosen by the user, or the 5 proposed; paths):
- Text checks: initialisms / loanwords listed, typos corrected:
- IPA: espeak-ng voice, language-switch markers kept (count):
- Listening lists sent to the user and answers (or "default used": queued chain, no answer):
- Prepared data folder (copy of the user's audio and metadata; the originals are not changed):
- Manifests, Lhotse Shar, codec codes paths; decoded sample listened to (yes / no):

## Preflight

- GPU / CPU cores / host RAM / disk:
- Load log: restored tensors (expected count from the task file):
- Optimizer log (per-module LRs: `Parameter Group 0` with `model.optim.lr`, `Parameter Group 1` with the module's LR):

## Runs

| Run | Data | Encoder LR / rest LR | max_steps / val interval / patience | Stopped at | Kept checkpoints (val_loss) | Broken outputs / cumulative CER / SSIM per checkpoint | Chosen step | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | |


## Evaluation (standalone, CFG off, 3 repeats; means over the repeats; no pass / fail, the user judges)

- ASR per language (default or other; same for every model):
- Release as shipped evaluated on the same sets (always):

One row per voice (several voices in one model: every number is per voice).

| Model / voice | SSIM pred↔gt | SSIM gt↔context | Cumulative CER | CER real recordings | Regressions vs release (optional, count) | Broken outputs per set (runaway / empty / silent / far too long) | Hard CER / % repeat / % missing | One-word % failed (automatic / after listening) / % silent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Release as shipped (same reference clips) | (low by design) | | | | – | | | |
| Fine-tuned, <voice> | | | | | | | | |

- Number scoring (non-English hard set): converter used, CER with / without closest-reading matching:

- Scripts that write vowels as combining marks (e.g. Hindi): CER with combining marks kept (fine-tuned / real
  recordings; hard and one-word sets):
- Reference clip for serving, per voice (chosen by the user, or "not chosen"):

## Error analysis

- Worst items listened to and what they show (numbers, repeated / missing words, silent or late-starting outputs, ASR
  spelling variants, one reference clip worse than the others):

## Decisions and next steps

- Ship / iterate (what changes, one at a time):
