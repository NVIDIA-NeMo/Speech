# Task: New Voices (One or More) in a Supported Language

Goal: add one or more speaker voices to the NemotronTTS release in a language it speaks (the language table,
`references/new-voices-supported-language/data.md`), from synthetic or real recordings. Several voices are trained together into one model. Use
the values below for every language and run the whole checklist for each voice.

The release ships without a speaker encoder: given a reference clip of a new voice it does not clone it. Every new
voice therefore needs this task.

## Data (with `references/new-voices-supported-language/data.md`), per voice

- **At least ~1 h of clean speech per voice; the more, the better.** More data mainly improves long and difficult
  text (numbers, repeated words). With less, see the held-out sizes below.
- Hold out ~100 utterances for validation and ~100 for test per voice, on top of the ~1 h for training. With less
  data, hold out fewer (starting point, not yet validated: ~50 + ~50; ~25 + ~25 below ~30 min of recordings) and tell the user that the test
  numbers are then less precise. Voices that read the same sentences: hold out every voice's validation and test
  sentences from the training data of all voices (`references/new-voices-supported-language/data.md`).
- Difficult sentences like the product's inputs (long numbers, repeated words) help if the user has them; transcripts
  in spoken form (`references/new-voices-supported-language/data.md`).
- **Add 150–200 short utterances (1–2 words, ~3–4 min) per voice; strongly recommended.** Very short inputs ("Yes.",
  "Thanks.") are the weak spot of fine-tuned voices, and short utterances in the training data reduce it.
  - Real recordings: use their own short clips (1–2 words) or record a short list (yes / no / thanks /
    names / numbers / the product's typical replies).
  - Synthetic voices: the user generates them with the same voice.
  - Fewer than 150: use all of them, tell the user that one-word inputs may be weaker, and offer the one-word set (it shows this weak spot). None, and
    the user cannot add any: say that one-word inputs will be the weak spot, proceed, and offer the one-word set (it shows this weak spot).
- Reference clips: training = most similar other clip of the same speaker, ≥ 3 s; utterances of ≥ 3 s below 0.6
  similarity are dropped; validation / test / serving = the clip(s) the user chooses from 5 proposed typical clips or
  adds, else the 5 proposed clips (`references/new-voices-supported-language/data.md`).
- Training data: one input config with one Lhotse group holding all the voices of a language; each voice's share of
  the training steps is its hours ÷ the total hours. One group per voice (equal shares) only when the user asks for it
  (`references/new-voices-supported-language/data.md`).

## Method: joint fine-tuning, one run (with `references/new-voices-supported-language/training.md`)

Train the new speaker encoder and the rest of the model together from the release, with two learning rates
(default; `model.optim_param_group_rules`, `references/new-voices-supported-language/setup.md`). Launch with `--config-name=easy_magpietts_finetuning` and the
overrides of `references/new-voices-supported-language/training.md`, with one validation set holding every voice's validation lines (same number per voice).

| Setting | Value | Note |
| --- | --- | --- |
| Rest of the model LR | 5e-6 | pretrained weights: keep the update small |
| New speaker encoder LR | **5e-5** | randomly initialised |
| Learning rates | two (default) | alternative only when the NeMo version has no per-module LRs: one LR 5e-6 for the whole model (no `optim_param_group_rules` overrides); quality close to two LRs, one-word inputs slightly worse, similarity can be slightly lower |
| `train_shuffle_context_embedding_prob` | 0.1 (config default; keep) | |
| `use_speaker_encoder` | true | the release has none; a new one is created and randomly initialised |
| `cfg_unconditional_prob` | default | keep |
| Checkpoint selection | validation loss, early stopping after 5 checks, 2 lowest-loss checkpoints kept, then the rule of `references/new-voices-supported-language/training.md` | |
| `val_check_interval` | 50–500 steps: about one pass over the training data (total training hours × 3600 / `batch_duration`), rounded down to 50, 100, 200, 250 or 500 (these divide `trainer.limit_train_batches` = 1000, so the checks are evenly spaced); e.g. 50 up to ~2.7 h, 100 at ~2.8–5.5 h, 200 at ~5.6–6.9 h, 250 at ~7–13.8 h, 500 from ~13.9 h | larger intervals space the 2 kept checkpoints further apart and delay early stopping by 5 × the interval; must not exceed `trainer.limit_train_batches` (1000) |
| `max_steps` | 20000 (upper bound) | early stopping ends the run |

Expected load log: `994 / 1001 layers are restored (994 exact, 0 partial, 0 skipped due to incompatible shape)` (the 7 missing tensors are the
new encoder); the optimizer log shows `Parameter Group 0` with `lr: 5e-06` and `Parameter Group 1` with `lr: 5e-05`
(one-LR alternative: only `Parameter Group 0`).
One run takes roughly 20–40 min on one A100 for one voice at 1 h of data and about 1 h at 5 h; two voices about
30 min at 2 × 1 h and 1–1.5 h at 2 × 5 h (early stopping decides).

Voices in one model:
- A voice with much less data than the others gets a smaller share of the training: tell the user the
  shares before training; if that voice comes out clearly worse, offer to add data for it, to train it alone, or to give every voice the same
  share (one group per voice, `references/new-voices-supported-language/data.md`).
- At inference the voice is chosen only by its reference clip (there is no speaker id).

## Evaluation report (with `references/new-voices-supported-language/evaluation.md`), per voice

There are no pass / fail thresholds: report each number next to its references and let the user judge. Ask at the
start whether the user has a target (e.g. "CER no higher than X", "not worse than the release", a voice similarity
level) and compare with it if they do. Ask also whether to run the hard set and the one-word set (~40 / ~20 min per
model on one GPU, plus the release), and whether the user has their own texts for them (or recordings). Use the
user's texts as they are, with the number matching of `references/new-voices-supported-language/evaluation.md`; with recordings, evaluate them as
another test set. Otherwise build the sets as `references/new-voices-supported-language/evaluation.md` describes. Default: not run. Every number is
standalone evaluation, 3 repeats, CFG off, for each voice on its own sets and reference clips. Report each model's
number with the ± of the evaluator's CSV (`all_experiment_metrics_with_ci.csv`, a t-interval over the 3 repeats); for
a CER computed outside the evaluator (number matching, Hindi with combining marks), compute the same interval over
the 3 repeats.

| Report | Set | Numbers | References shown next to it |
| --- | --- | --- | --- |
| Voice similarity | ~100 test utterances × 3 repeats, the voice's reference clips | `ssim_pred_gt_avg` | `ssim_gt_context_avg` (how similar two real recordings of the voice are) |
| Intelligibility | same outputs | cumulative CER (`cer_cumulative`) | the real recordings' CER (`cer_gt_audio_cumulative`) and the release's CER on the same set |
| Broken outputs | every set that was run | one line: runaway, empty, silent and far too long outputs (count and %), from the wav files (`references/new-voices-supported-language/evaluation.md`, "Metrics") | 0 expected |
| Regressions (optional) | same outputs | test sentences the release reads correctly in every repeat but the fine-tuned voice misreads in at least one | the release's transcript of each |
| Hard sentences (optional) | ~300 items from ~200 sentences | CER, % repeated, % missing words | the release on the same set |
| One-word inputs (optional) | ~60 words × ≥ 5 items | cumulative CER (`cer_cumulative`) first, then % failed outputs (read wrongly or broken), also after listening | the release on the same set |

**Baseline: the release as shipped, always run** on every set that is reported, with the same reference clips, ASR
and settings (~25 min per 2 voices on the test set, one evaluation per GPU). It separates two causes that look the
same in a CER number: errors the release also makes on the same sentences (a limit of the base model, which more
fine-tuning data rarely fixes) and errors the fine-tuned voice makes where the release is right (a regression from
fine-tuning). The release has no speaker encoder, so its voice
similarity is low by design: show its CER only.

How to read the report (write one line of this next to each number):
- Voice similarity: close to the real recordings' value is good; expressive voices and short clips have low values
  on both sides.
- Intelligibility: the ASR also makes errors on the real recordings; a clone close to their CER and below the
  release is good. Cumulative CER = all character errors ÷ all reference characters over the set and repeats
  (`references/new-voices-supported-language/evaluation.md`, "Metrics"). Differences of a few tenths of a point between models are often within
  run-to-run noise; listen to the worst items before concluding. With Whisper, compare numbers only after the number
  matching of `references/new-voices-supported-language/evaluation.md`.
- Broken outputs (always counted; a few seconds on CPU): the ASR hears only the first 30 s of an output, so a runaway
  can hide behind a low CER; count them from the wav files (`references/new-voices-supported-language/evaluation.md`, "Metrics"). Listening to
  each one and looking for the cause is optional, when the user wants the analysis.
- Regressions (optional: when the user wants the analysis, or the CER is above the release's): list each one with the
  release's and the fine-tuned transcript, and the audio to listen to.
- Several voices: one row per voice; tell the user if one voice is clearly worse than the others.
- Hindi: report CER both as the evaluator computes it and with combining marks kept (`references/new-voices-supported-language/evaluation.md`).

## What to try

| What the report shows | What to try |
| --- | --- |
| Voice similarity far below the real recordings' value | other / longer (≥ 4 s) reference clips; check the training clips are the target speaker; more data |
| CER above the release, or a long regression list | listen to the regressions (ASR error or real mistake?); evaluate the other kept checkpoint; more data; run the optional ASR data check to find bad transcripts |
| Hard sentences worse than the release | more data; offer the user another reference clip (per-clip results, `references/new-voices-supported-language/evaluation.md`) |
| One-word inputs: many failed outputs | add / increase short utterances; if many outputs are silent, also cut long leading silence in the data (`references/new-voices-supported-language/data.md`) |
| Broken outputs | listen to them; check the training clips' trailing silence (≥ ~0.2 s, none over ~1 s) and short utterances; evaluate the other kept checkpoint; more data. Evaluation run times above assume no runaways: each one adds up to ~80 s of generation |
| One voice clearly worse than the others in a shared model | tell the user; offer a one-voice model for it as a comparison |
| Training stops very early | check `val_check_interval` and patience; confirm the two LR groups in the log |

Change one thing at a time; record it in the ledger.

More data later: retrain from the release with all the data and the same settings, keeping the validation / test sets
and the reference clips so the two models can be compared; continuing from the fine-tuned model is untested.

## Pitfalls

- Load log must show `994 / 1001 layers are restored (994 exact, 0 partial, 0 skipped due to incompatible shape)` and the optimizer
  `Parameter Group 0` and `Parameter Group 1` (only `Parameter Group 0` with the one-LR alternative); anything else means the wrong checkpoint, config or NeMo
  version.
- Do not change `ignore_phoneme_languages`: new voices use the checkpoint's language routing as is.

## Do not use

| Option | Why |
| --- | --- |
| Training only a new speaker encoder (rest frozen) | lower similarity and intelligibility, worst on real voices |
| Two-stage training (encoder only, then whole model) | no gain over joint fine-tuning, twice the work |
| One shared learning rate above 5e-6 (e.g. 2e-5) | worse on short and difficult inputs |
| CFG on at inference | empty outputs and higher CER |
| Mixing public multi-speaker data into the training set | no consistent gain for the new voices, slightly lower similarity |
