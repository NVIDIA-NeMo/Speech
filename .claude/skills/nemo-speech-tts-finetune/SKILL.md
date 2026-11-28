---
name: nemo-speech-tts-finetune
description: "Guide NeMo Speech users through fine-tuning NemotronTTS checkpoints, currently for new voices in a language the checkpoint speaks, from data preparation to an evaluated .nemo checkpoint. Use to prepare TTS fine-tuning data, train and select a checkpoint, export and evaluate it against the real recordings and the release, or diagnose a fine-tuned voice that sounds wrong."
license: Apache-2.0
---

# NeMo Speech TTS Fine-Tuning

Use this skill when a user wants to fine-tune a NemotronTTS checkpoint: prepare data, train, select
and export a checkpoint, evaluate it, and decide what to change next. Also use it to diagnose a fine-tuned model that
sounds wrong (mispronounced words, cut-off or overlong outputs, weak one-word replies).

## Choose the task first

Every task shares the stages below, but hyperparameters, data requirements and evaluation reports are task-specific and
live only in the task file. **Never carry values from one task to another without validation.**

| Task | Status | Task file |
| --- | --- | --- |
| New voices (one or more) in a language the checkpoint already supports | ✅ supported: the release's 12 languages (`references/new-voices-supported-language/data.md`, language table) | `references/new-voices-supported-language/README.md` |

Requests outside these tasks (e.g. a language the checkpoint does not speak) are not covered: say so, and do not
apply this task's values to them. Run the task file's whole evaluation checklist for every language.

Each task is self-contained in its own folder `references/<task>/`: the task file `README.md` (goal, data needs,
method, evaluation report, what to try, pitfalls) and one file per stage (`setup.md`, `data.md`, `training.md`,
`evaluation.md`). Use only the files of the chosen task; tasks share no reference files. A new task is added as one
row here plus its own folder, started as a copy of `references/new-voices-supported-language/` and then changed
freely, and, only if the task needs a different model or training setup, its own config in
`examples/tts/conf/magpietts/`. Changing one task's files never changes another task. When a fix applies to every
task (e.g. the container launcher in `setup.md`), apply it in each task folder and compare the folders' copies (`diff`).

## Defaults common to all tasks

These apply unless the task file says otherwise.

- **Scope: fine-tuning from prepared clips; no segmentation, alignment, transcription, recording or serving.**
  - Input: one audio clip per utterance with its transcript and the voice it belongs to. Data that mixes voices
    without saying which clip is whose goes back to the user to label; the agent may propose a grouping by speaker
    similarity for the user to confirm, and never trains on unlabelled mixed voices.
  - The agent does: the data checks and light fixes of `references/new-voices-supported-language/data.md` (format, silence, text written out, IPA,
    reference clips; listing ASR-check mismatches with suggested corrections for the user to confirm) and every stage
    after them.
  - The agent does not: cut long recordings into utterances, align text to audio, transcribe or re-transcribe audio,
    record or generate new audio (e.g. missing short utterances), or build serving pipelines.
  - For those, tell the user what is needed (e.g. long session recordings: cut them into utterances and align the text
    precisely, for example with NeMo Forced Aligner in `tools/nemo_forced_aligner`) and continue once the data is ready.

- Start from the checkpoint the user actually has (for NemotronTTS customers: the release, which ships **without a
  speaker encoder**); compare only against models the user can run.
- Training entrypoint `examples/tts/easy_magpietts.py --config-name=easy_magpietts_finetuning` (a standalone
  config with the released checkpoint's architecture and fine-tuning settings; TODO: the config is added with the
  model release) and per-run overrides;
  `trainer.max_steps`; 1 GPU per run by default (data under ~50 h), more GPUs optional.
- Data as Lhotse Shar with pre-computed codec codes.
- Checkpoint selection: early stopping on validation loss, 2 lowest-loss checkpoints kept, then the one without broken
  outputs and with the lowest standalone validation cumulative CER (`references/new-voices-supported-language/training.md`).
- Inference and evaluation with **CFG off**, temperature 0.7, top-k 80, predicted phonemes.
- Report final quality from standalone evaluation (`examples/tts/magpietts_inference.py`), never only from
  in-training validation metrics.
- Record every run in a copy of `assets/experiment-ledger-template.md`.

## Core commands and docs

| Step | Entry point |
| --- | --- |
| Lhotse Shar | `scripts/magpietts/create_lhotse_shar_from_nemo_manifest.py` |
| Codec codes | `scripts/magpietts/extend_lhotse_shards_with_audio_codes.py` |
| Training | `examples/tts/easy_magpietts.py --config-name=easy_magpietts_finetuning` |
| Evaluation | `examples/tts/magpietts_inference.py` |

Check the repository's own docs before version-sensitive claims: `docs/source/tts/easymagpietts-finetuning.rst` (the
recipe), `docs/source/starthere/install.rst`, `examples/tts/conf/magpietts/easy_magpietts_finetuning.yaml`.

## Instructions

Work in stages. Load only what the current stage needs: the stage's file in the task's folder, then the task file's
section for that stage. Paths below are for the new-voices task; another task has the same files in its own folder.

1. Setup (container, NeMo version, checkpoint, codec, tokenizers, model caches): `references/new-voices-supported-language/setup.md`.
2. Data (quality checks, manifests, IPA / text-channel routing, reference clips, Lhotse Shar, codec codes):
   `references/new-voices-supported-language/data.md`.
3. Training, checkpoint selection, export: `references/new-voices-supported-language/training.md`.
4. Evaluation (inference settings, metrics, hard / one-word sets): `references/new-voices-supported-language/evaluation.md`.
5. Task-specific choices, evaluation report and iteration: the task file.

A task may add its own stages (for example tokenizer screening or preference optimisation) as
`references/<task>/<stage>.md`, listed in its task file in the order they run.

If the user explicitly asks for parallel / sub-agent work, split by these stages; the main agent integrates the final
commands and the ledger.

## Examples

**User:** "I have 2 hours of recordings of our support agent's voice in English (wav + transcripts). Make the
NemotronTTS release speak in that voice."

**Agent:**
1. Task = new voices in a supported language, here one voice (`references/new-voices-supported-language/README.md`).
2. Setup: checks the container, the NeMo version (per-module LR support, `optim_param_group_rules`), the release checkpoint, codec and tokenizers.
3. Data: asks whether to run the optional ASR check of the transcripts,
   converts the audio to mono, cuts long leading silence and fixes trailing silence, holds out ~100 validation and
   ~100 test utterances (fewer when data is scarce), asks for 150–200 short utterances (or uses the short clips the
   user already has), builds the manifests with IPA and reference clips, proposes 5 fixed reference clips for the user
   to keep, swap or replace with their own, then Lhotse Shar and codec codes; decodes a sample for the user to listen
   to.
4. Training: one joint fine-tuning run with the task file's values; confirms the restored-tensor count and
   the two optimizer parameter groups in the log.
5. Selection and export: evaluates the 2 kept checkpoints on the validation set, exports the chosen one to `.nemo`.
6. Evaluation: test set × 3 repeats for the fine-tuned model and for the release as shipped; reports voice
   similarity, CER next to the real recordings and the release, and the count of broken outputs; if asked, the list
   of sentences the release reads correctly but the fine-tuned voice does not, and the hard and one-word sets; fills the ledger and says what
   to try for what the report shows. No pass / fail: the user judges, against their own target if they gave one.

**User:** "The fine-tuned voice says 'Yes.' and 'Thanks.' badly." **Agent:** checks whether the training data has
short utterances; if not, adds 150–200 of them and retrains (`references/new-voices-supported-language/README.md`, "What to try").

**User:** "Fine-tune it for Thai." **Agent:** Thai is not one of the checkpoint's languages; says this task is not
covered and does not apply this task's values.

## Assets

- `assets/experiment-ledger-template.md`.

## Non-Negotiable Pitfalls

Task-specific pitfalls live in the task file. Common to all tasks:

- Check the checkpoint load log: the number of restored tensors must match what the task file expects. Anything else
  means the wrong checkpoint or config.
- Per-module learning rates need a NeMo version that includes commit `d2a01114d` (`model.optim_param_group_rules`,
  PR #16426); older versions silently ignore it. Confirm the number of parameter groups in the log.
  Without that support the task file's one-learning-rate alternative applies: say so to the user.
- **CFG off** at inference and evaluation: `use_cfg=False` (the `do_tts` default is True). If validation inference is
  enabled (`model.run_val_inference=true`), also set `+model.inference_use_cfg_in_val=false` (its default is CFG on).
- IPA with **espeak-ng 1.52.0** and the voice of the language table only (Brazilian Portuguese: `pt-br`); other
  versions or voices produce phonemes the model never saw. Keep espeak-ng's output as written, including the
  `(en)…(<lang>)` language-switch markers it puts on words it reads in another language (`references/new-voices-supported-language/data.md`). Which languages take phonemes and which go through the text
  channel is set by the checkpoint's `ignore_phoneme_languages`.
- Audio mono (the data scripts read only the first channel). Clips end with ≥ ~0.2 s of silence (abrupt endings teach
  the model to cut off the last word); a leading silence longer than ~0.5 s is cut to ~0.2 s.
- Reference clip of a training utterance: the most similar other clip of the same speaker from the training split,
  ≥ 3 s, never the utterance itself; utterances of ≥ 3 s whose best clip is below 0.6 similarity are dropped.
- Save generated audio at `model.output_sample_rate`.
- Machines without internet: the install, the espeak-ng build and the model downloads need it; prepare an image and the
  caches first (`references/new-voices-supported-language/setup.md`).
- If the optional ASR data check is run, use the same ASR model as for the evaluation.
- CER for languages other than English uses Whisper by default and can be inaccurate: tell the user and ask once
  which ASR to use (`references/new-voices-supported-language/evaluation.md`, "ASR for CER"); use the same ASR for every model compared. Hindi: the
  evaluator's CER ignores vowel signs; also report CER with combining marks kept (`references/new-voices-supported-language/evaluation.md`).
- The evaluator's ASR hears at most the first 30 s of an output (Whisper), so an output that runs on to the length
  limit can still score a low CER: count runaway and silent outputs from the audio files on every set, the test set
  included (`references/new-voices-supported-language/evaluation.md`).
- Manifest fields `normalized_text` and `original_text` are not free extra fields: the evaluator uses them as the CER
  reference when present. Do not store other text in them.
- Keep your own files out of `exp_manager.explicit_log_dir`: training moves files already there into `run_0/`.
- Evaluator: `--batch_size 16` (8 if it runs out of memory or the GPU has less than 80 GB) and a new `--out_dir` per
  run (it deletes the generated audio in an existing one); one evaluation per GPU at a time (long outputs on the hard
  set need most of an 80 GB GPU).
- Preflight before long jobs: GPU, CPU / RAM, disk, manifests, a decoded codec sample listened to, model caches on
  persistent storage.
