# Stage 4: Evaluation

Evaluate the chosen `.nemo` with the standalone evaluator, never only with in-training validation metrics.

## Inference settings (also for serving)

```python
audio, audio_len = model.do_tts(
    transcript=text,
    context_audio_file_path=<the voice's reference clip chosen by the user>,
    context_text=f"[{language.upper()}]",  # e.g. "[EN]": required; training used it, not the default "[NO TEXT CONTEXT]"
    use_cfg=False,          # default is True: always pass False
    temperature=0.7, topk=80,
)  # gt_phoneme_text unset -> the model predicts the phonemes; context_audio_duration: keep the default 5 s
sf.write(path, audio[0, :audio_len[0]].cpu().numpy(), model.output_sample_rate)
```

- `context_text` carries the language tag, as in training and in the evaluator (which builds `[<LANG>]` from each
  manifest line's `language`); `do_tts` has no `language` argument and does not add it.
- `context_audio_duration` keeps its default (5 s, as the evaluator): `do_tts` takes that much of the clip (shorter
  clips are repeated), so pass the clip as it is.
- Write the input text in spoken form, as in the training transcripts (numbers, symbols and abbreviations written out,
  the same respellings): the model does not normalise text.
- The voice is chosen only by the reference clip (there is no speaker id); with several voices in one model, pass the
  voice's own clip.

## Standalone evaluator

Evalset JSON (one entry per set; keys outside the evaluator's list are rejected):

```json
{
  "<voice>_test": {
    "manifest_path": "<voice>/test.jsonl",
    "audio_dir": "<audio root the manifest paths are relative to>",
    "language": "en",
    "asr_model": {"name": "nvidia/parakeet-tdt-1.1b", "type": "nemo"},
    "strip_text_annotations_for_metrics": true
  }
}
```

`language` and `asr_model` are set per set (they override the command line); `strip_text_annotations_for_metrics`
removes text annotations before CER. Relative paths resolve against `--datasets_base_path`; with absolute paths in the
manifest use `"audio_dir": "/"` and `--datasets_base_path /`.

```bash
python examples/tts/magpietts_inference.py \
  --model_type easy_magpie --easy_magpie_inference_mode single_turn \
  --nemo_files <voice>.nemo \
  --codecmodel_path <codec>.nemo --phoneme_tokenizer_path <ipa_tokenizer>.json --tokenizer_name nemotron_nano_30b \
  --datasets_json_path <evalset>.json --datasets_base_path <root> --datasets <voice>_test \
  --phoneme_input_type predicted --phoneme_sampling_method argmax \
  --temperature 0.7 --topk 80 --use_local_transformer --max_decoder_steps 1000 \
  --run_evaluation --num_repeats 3 --disable_fcd --batch_size 16 \
  --out_dir <work>/eval/<voice>_test
```

- CFG off = no `--use_cfg` (the output folder name shows `Cfg_False`).
- `--batch_size 16`: with the default (32) the codec runs out of memory when several outputs reach the length limit;
  if it runs out of memory at 16, or the GPU has less than 80 GB, use 8 (GPUs below 80 GB: batch 8 or lower,
  untested).
- A new `--out_dir` per evaluation run: the evaluator deletes the generated audio (`predicted_audio*.wav`) already in
  an existing one, so two runs sharing a folder overwrite each other's audio.
- `--tokenizer_name` is the model's text tokenizer (`text_tokenizers` key in the config).
- Checkpoint mode: `--hparams_files` + `--checkpoint_files` instead of `--nemo_files` (`references/new-voices-supported-language/training.md`);
  several models: comma-separated lists.
- Outputs per model and set, in a folder named after the model and the inference settings (it contains `Cfg_False`):
  per repeat `<set>_metrics_<i>.json` (set-level metrics) and `<set>_filewise_metrics_<i>.json` (per utterance), plus
  a CSV with a t-interval over the repeats; generated audio in `audio/repeat_<i>/predicted_audio_<j>.wav` (j = manifest
  line). CER and SSIM values are fractions. The filewise JSON is sorted by CER, not by manifest line: match its items
  by the `<j>` in each `pred_audio_filepath`.
- Run time on one 80 GB GPU, per model, 3 repeats, batch 16: ~100-utterance test set ~10–15 min, ~300-item hard set
  ~30–40 min, ~300-item one-word set ~15–20 min. Run one evaluation per GPU at a time: memory grows with long outputs,
  and on the hard set one evaluation can reach ~55–60 GB. With several GPUs, run sets and models as separate jobs in
  parallel.

## Metrics

- **SSIM** = cosine similarity of `titanet_large` embeddings (another verification model gives a different scale).
- **CER** after the evaluator's `process_text_for_wer` normalisation, always **cumulative**: all character errors of
  all sentences (and repeats) ÷ all reference characters. The evaluator reports it as `cer_cumulative` for the clone
  and `cer_gt_audio_cumulative` for the real recordings; use these, on every set. Do not use `cer_filewise_avg` (the
  mean of per-sentence CERs): a garbled output can have a CER of 4 or 20 on its own sentence and then decides that
  mean alone. Broken outputs are counted separately (below).
- The evaluator picks the text normaliser from `language` (CER is per character, so Chinese, Japanese and Korean need
  no word segmentation). It keeps letters and digits only, so it drops combining marks.
- **Hindi** (and any script that writes vowels as combining marks): the dropped marks are the vowel signs, anusvara,
  virama and nukta, so the evaluator's CER misses vowel errors (दो / दे), and its saved `pred_text` / `gt_text` are
  already stripped. Also compute CER with the marks kept: transcribe the generated audio and the real test recordings
  with the same ASR (as for an ASR the evaluator cannot load, below), normalise both sides with `for_cer_keep_marks`,
  compute cumulative CER the same way and report it next
  to the evaluator's CER for the test, hard and one-word sets and for model comparisons. Spelling
  variants (nukta, anusvara / chandrabindu) then count as errors too: listen to the worst items.

  ```python
  import unicodedata
  def for_cer_keep_marks(text):  # the evaluator's normalisation, with combining marks kept
      text = unicodedata.normalize("NFC", text).lower().replace("-", " ")
      text = "".join(c for c in text if c.isspace() or c.isalnum() or unicodedata.category(c)[0] == "M")
      return " ".join(text.split())
  ```

- The automatic cut-off check (`eou_*`) runs for English only; in other languages a cut-off last word shows up as a
  missing word in the CER. The user can listen to the endings in the report's samples if they want.
- **Broken outputs, on every set (test set included):** the ASR hears only the first 30 s of an output (Whisper), so
  an output that runs on to the length limit can still score a low CER, and CER cannot tell a silent output from a
  wrongly read one. Count from the wav files and report on their own line: runaway (output ≥ 90% of the length limit,
  ≈ 72 s with `--max_decoder_steps 1000`), empty (no transcript), silent (peak amplitude below 0.01) and far too long:
  on sets with real recordings (validation, test), output > 2 × its real recording + 1 s; on sentence sets without
  them (hard sentences), output > 2 × (the voice's median seconds per character over its training utterances of ≥ 3 words × the
  item's characters) + 1 s; on the one-word set, output > 5 s. List the far-too-long items for listening: some are
  slow but correct readings (e.g. repeated words with pauses). The checkpoint choice uses the same definition
  (`references/new-voices-supported-language/training.md`). Many silent outputs point to long leading silence in the training
  data (`references/new-voices-supported-language/data.md`); runaways, to long trailing silence or missing short utterances.

## ASR for CER

| Language | Default ASR (evalset `asr_model`) |
| --- | --- |
| English (also Indian English) | `{"name": "nvidia/parakeet-tdt-1.1b", "type": "nemo"}` |
| Every other language | `{"name": "openai/whisper-large-v3", "type": "whisper"}` ⚠ |

⚠ **Other languages: tell the user, then ask once which ASR to use.** Whisper's CER can be inaccurate: its own error
rate on the language can be several times that of a dedicated ASR, it may write numbers as digits, and it can rank two
TTS models differently from a good ASR. With Whisper:
- the comparison with the real recordings and the release still holds (same ASR on all of them);
- do not decide between models on Whisper CER alone: listen to the worst items;
- numbers: Whisper writes them as digits, and the evaluator saves only the normalised transcript (digits run together,
  separators, `%`, currency signs and times dropped), so the numbers cannot be recovered from its output. For the hard
  set and model comparisons, transcribe the generated audio again with the same Whisper model (raw text) and score it
  with **closest-reading matching**:
  1. find each number in the raw text (digits with their separators and a unit or currency sign next to them);
  2. write out its correct readings in the language: the cardinal (e.g. `num2words(n, lang=...)`), digit by digit,
     in groups (phone numbers), the ordinal, and amounts with their unit ("twelve euros fifty", "twelve point five");
  3. keep the reading that gives the lowest CER against the reference sentence (numbers right to left, one at a time);
  4. normalise with `process_text_for_wer` and compute cumulative CER as the evaluator does.

  Use one converter for every model and set, and check it on a few hundred numbers (large numbers included) before
  scoring; report CER with and without the matching. Without it, a correctly read number can count as several errors.

Another ASR can be used for any language:
- a NeMo ASR model: put its HF name or a local `.nemo` path in `asr_model` (`"type": "nemo"`). If the `.nemo` file in
  an HF repo is not named after the repo, `from_pretrained` cannot find it: download the file and give its path. Gated
  HF repos need the user to accept the terms and an HF token in the job (`HF_TOKEN`). The user chooses the model (and
  is responsible for its license); a dedicated ASR can read clean speech worse than Whisper on some voices, so if
  the clone looks much worse only with it, listen before deciding;
- an ASR the evaluator cannot load: generate the audio with the evaluator as usual, then compute CER outside it on
  `audio/repeat_<i>/predicted_audio_<j>.wav` against the manifest text, with the same normaliser
  (`get_text_processor(language).process_text_for_wer`) and cumulative CER.

Use one ASR for every model, the baseline and every set of a comparison, and record it in the ledger.

## Report

What to report is in the task file (`references/new-voices-supported-language/README.md`); there are no pass / fail thresholds, the user judges.
Numbers use the mean over the 3 repeats (`<set>_metrics_<i>.json`, i = 0, 1, 2) and are per voice when one model has
several. Every task reports, from standalone evaluation with 3 repeats: SSIM (`ssim_pred_gt_avg`, with
`ssim_gt_context_avg` as the real-recording reference), test-set CER (cumulative, with `cer_gt_audio_cumulative`),
broken outputs (count only), and optionally the hard and one-word sets below, each next to the **release as shipped** evaluated on the
same sets with the same reference clips, ASR and settings (always run).

**Regressions** (optional: when the user wants the analysis, or the CER is above the release's): match the release's and the fine-tuned model's filewise items by the `<j>` in `pred_audio_filepath`;
list the sentences the release reads correctly in every repeat (CER 0 after number matching) and the fine-tuned model
misreads in at least one repeat, with both transcripts and the audio paths.

## Hard and one-word sets (the user's, or generated by the agent)

Build both manifests in the test manifest's format without `ipa` (evaluation predicts the phonemes), with the voice's
reference clips (the user's chosen clips, else the 5 proposed ones, `references/new-voices-supported-language/data.md`) taken in turn. These sets have no real recordings, but the evaluator still needs one per item:
set `"audio_filepath": "dummy_audio.wav"`, a real file (e.g. 1 s of silence) in the set's `audio_dir` (with
`"audio_dir": "/"`, give its absolute path), and any `duration`. Ignore the metrics that compare with the recording
(`*_gt_*`) for these sets; the evaluator still transcribes the dummy recording for every item, which costs time only.
Evaluate them with 3 repeats.

**Hard set** (finds skipped / repeated words on long, difficult text):
- ~85% sentences with long numbers, ~15% with repeated words ("wait wait wait", "Sam Sam Sam"); include names and long
  sentences.
- Write every sentence directly in spoken form, numbers written out as they are read in the language ("three thousand
  and twelve dollars"); then check that no digits or symbols remain. Where a number has several
  correct readings ("one hundred and five" / "one hundred five", "zero" / "oh"), use the language's most common one
  throughout the set: the ASR transcript is compared with this text, so a mix counts correct readings as errors.
- Reference clips: every sentence with 1 clip, and with 2 or more clips half of the sentences with a second one
  (~200 sentences → ~300 items), the clips taken in turn.
- Use the number system the product uses (e.g. lakh / crore in Indian English) and check how the ASR spells such words;
  count spelling variants of the same word as correct.

**One-word set**: ~60 single words with sentence punctuation ("Yes.", "Thanks.", "Never."), including the product's real
replies (ask the user for them; if they have none, propose a list and show it to them), each with every reference
clip (≥ 5 items per word: repeat the clip list if the user chose fewer clips). Words that are also in the training short utterances are allowed; mark them
in the results. The product's multi-word short replies ("You're all set.") can be added as a small extra set, reported
separately.

**Per-item results** (starting-point thresholds, not yet validated): align the normalised text and ASR transcript word by word (e.g.
`difflib.SequenceMatcher` on the word lists), after the number matching above (a number compound written as two words
is not an inserted word);
- repeated: ≥ 2 inserted words;
- missing: ≥ 2 deleted words;
- broken: empty, runaway, silent or far too long (above; one decoder step = 2 codec frames = 0.08 s, so `--max_decoder_steps 1000`
  ≈ 80 s);
- one-word set: report first the **cumulative CER** (`cer_cumulative`, mean over the 3 repeats, as on the other
  sets), then **% failed outputs** = read wrongly (normalised ASR transcript ≠ the word) or broken, over every item
  and repeat; report its parts too. The references are a few characters long, so a single runaway output (hundreds of
  transcribed characters) can dominate the cumulative CER of a repeat: say so next to it when the set has runaways,
  and read the comparison with the release from % failed. Before scoring, write the accepted spellings
  of each word (homophones, regional spellings, digits for numbers: "cheque" / "check", "seven" / "7"), or leave such
  words out of the set; write the word and its accepted spellings in the evaluator's normalised form
  (`process_text_for_wer`, which e.g. drops apostrophes: "you're" → "youre") and compare in that form, else correct
  readings count as wrong; Hindi: also with combining marks kept. ASR often mishears isolated words: list the outputs
  read wrongly with their transcript, ask the user to listen to them (or to ~20 of them) and mark each as an ASR error
  or a real misreading, and report the rate after listening next to the automatic one.

**Reference clip for serving** (when the user has not chosen one yet): report, per reference clip, CER and broken
outputs on each set that was run, SSIM vs the clip (`pred_context_ssim`) on the test and hard sets only (one-word
outputs are too short for a reliable SSIM), with a few generated samples per clip to listen to. The filewise JSON has
no per-item SSIM against the real recording, so `ssim_pred_gt_avg` cannot be split by clip. The user picks the clip (tone and
style matter as much as CER); record the choice, or "not chosen", in the ledger.
