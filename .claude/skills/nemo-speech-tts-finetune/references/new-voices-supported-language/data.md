# Stage 2: Data

Requirements of this task (how much data, short utterances, held-out sets) are in the task file
(`references/new-voices-supported-language/README.md`). Input: one clip per utterance with its transcript;
anything else (long recordings, untranscribed audio) goes back to the user first (SKILL.md, "Scope").

## Languages

The release speaks 12 languages. Each language reaches the model through one of two channels, fixed by the checkpoint's
`ignore_phoneme_languages`; keep the checkpoint's list.

| Language | `language` | Channel | espeak-ng voice for `ipa` |
| --- | --- | --- | --- |
| English (also Indian English) | `en` | phonemes | `en-us` |
| Hindi | `hi` | phonemes | `hi` |
| German, Spanish, French, Italian | `de`, `es`, `fr`, `it` | phonemes | `de`, `es`, `fr`, `it` |
| Portuguese (Brazil) | `pt` | phonemes | `pt-br` |
| Arabic | `ar` | phonemes | `ar` |
| Vietnamese, Chinese, Japanese, Korean | `vi`, `zh`, `ja`, `ko` | text | none (`ipa` not used) |

- Use the same `language` code in every manifest line (training, validation, test, evaluation sets), the `speaker`
  field, the training input config's `lang` tag and the evalset. The model sees it as the context tag `[<CODE>]`
  (e.g. `[PT]`), as in pretraining: use exactly the codes of this table (no region suffix such as `pt-BR`).
- The espeak-ng voices are the ones used to build the checkpoint's training data; do not substitute other voices.
  Portuguese: the `language` code is `pt` but the espeak-ng voice is `pt-br` (in espeak-ng 1.52.0, `pt` is European
  Portuguese, whose phonemes the release's tokenizer does not cover).
- Text-channel languages: the model reads the written text directly, so text normalisation (numbers, symbols,
  abbreviations written out) carries all the pronunciation.

## Audio and text checks (before building the manifest)

Never change the user's files. Tell the user which fixes apply and to how many clips or lines (mono conversion,
leading / trailing silence, text changes), and ask once where to write the prepared data (default `<work>/data/<voice>`;
about 0.2–0.35 GB per hour of audio). Copy the user's whole delivery there, audio and metadata (transcript lists, split
file), keeping their folder layout, and make every fix on this copy: audio fixes on the copied clips; text fixes
(items 0 and 5, and the corrections the user confirmed) in the copied metadata, whose audio paths point to the copied
clips. Keep the delivered text of a changed line in a field such as `delivered_text` (not `original_text`, which the
evaluator reads as the CER reference). Drops (ASR check, similarity rule) leave a line out of the manifests, not out of
this copy. Build the manifests from this copy, with `duration` and `context_audio_duration` measured on the copied
clips (the Lhotse Shar script cuts each clip to the manifest's duration, so a duration taken from the original file
would cut off the added silence); the manifests, Lhotse Shar and codec codes go to the same folder.

0. **Read the delivery:** strip a byte-order mark and `\r` line endings, skip a header row, collapse repeated spaces,
   and report transcripts without audio, audio without transcript and duplicate lines to the user. Identify each
   utterance by its full audio path, not its file name: separately delivered folders (e.g. short utterances) often
   reuse the same file names. If you derive internal ids from file names, check they are unique per voice.
   One voice: the whole delivery is that voice; no voice label is needed. Several voices: they may come as one folder
   per voice or as one set of files with a voice column in the metadata; either way, every clip must belong to a known
   voice. Split the lines by voice for every per-voice step (checks, held-out sets, reference clips, the
   `Speaker:<voice_name>` of the `speaker` field) and report each voice's hours of audio to the user. If the delivery
   mixes voices without labels, stop and ask the user to label the clips: unlabelled voices pair training clips with
   reference clips of another speaker, which lowers voice similarity without any error. You may propose a grouping by
   speaker similarity (embeddings as under "Reference ("context") clips") for the user to confirm.

1. **ASR check (optional, off by default):** ask the user whether to run it; skip it unless they ask for it. It finds
   transcripts that do not match the audio (missing or extra words, wrong file), which matters most for transcripts
   that were not checked against the audio. If run, transcribe with the evaluation ASR model of the language
   (`references/new-voices-supported-language/evaluation.md`) and compute CER the way the evaluator does, so data and evaluation use the same scale:

   ```python
   from nemo.collections.asr.metrics.wer import word_error_rate_detail
   from nemo.collections.tts.parts.utils.tts_dataset_utils import get_text_processor
   tp = get_text_processor(language)        # the evaluator's text normaliser for the language
   ref, hyp = tp.process_text_for_wer(text), tp.process_text_for_wer(asr_text)
   cer = word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=True)[0]
   ```

   Transcribe with the evaluator's own transcriber classes (same `name` / `type` as the evalset's `asr_model`):

   ```python
   from nemo.collections.tts.parts.utils.tts_dataset_utils import NemoTranscriber, WhisperTranscriber
   asr = NemoTranscriber(device="cuda", model_name="nvidia/parakeet-tdt-1.1b")   # or WhisperTranscriber(...)
   asr_texts = asr.transcribe(audio_paths=paths, batch_size=32, language=language)
   ```

   Run it on the original files. Many failures are spelling differences between the transcript and the ASR output
   (abbreviations, numerals, contractions: "Doctor" / "Dr.", "twenty first" / "21st", "I am" / "I'm"), not bad audio.

   - Training utterances with CER > 5%: drop, or review by hand. Keep the original transcript; do not "fix" it to
     match the ASR.
   - Short utterances (1–2 words): same bar, but ASR is unreliable on very short speech; listen to the dropped ones
     and keep those that are correct.
   - Validation / test utterances chosen by the user: keep them and report the failures to the user, who decides.
2. **Audio format:** mono; any sample rate (the Shar script stores the audio as it is; the codec-code script and the data loader
   resample it). The Shar script reads only the first
   channel: convert stereo files by averaging the two channels, after checking each channel for clipping (a clipped
   channel can look clean after averaging). Report clips with at least 3 consecutive samples at full scale
   (|x| ≥ 0.999; a single peak sample is normal for peak-normalised audio) to the user as likely clipped; replace or
   drop them if they sound distorted. Audio that was clipped and then turned down has flat runs below full scale:
   also report clips with ≥ 3 consecutive samples at the clip's own peak value when that peak is ≥ 0.9.
3. **Leading silence:** measured like the trailing silence below (before the first speech frame). If it is longer than
   ~0.5 s, cut it to ~0.2 s; keep shorter pauses. This matters most for short
   utterances: a one-word clip with a long lead-in is mostly silence.
4. **Trailing silence:** ≥ ~0.2 s at the end of every clip, all splits. Measure it as the time after the last speech
   frame: a 10 ms frame whose energy is within 40 dB of the clip's loudest frame and that belongs to a run of at least
   3 such frames (a lone click inside the padding is not speech). If it is shorter, fade out the last 5 ms of the
   clip and append 200 ms of zeros, keeping its sample rate and format (voices trained on abrupt endings learn to cut
   off the last word). Recordings often have almost no trailing silence, so this usually applies to most clips.
   A trailing silence longer than ~1 s (starting point, not yet validated): cut it to ~0.5 s.
5. **Texts:** numbers, symbols and abbreviations written out as spoken, in the language's own form (the model reads
   text as given); find leftovers by scanning for digits, symbols and all-caps tokens, and for abbreviations with a
   period that such a scan misses (Jr., St., Dr.: espeak-ng may read them letter by letter or as another word).
   Store text in Unicode NFC (composed characters, as most text and ASR output use), except Arabic: keep its
   diacritics in the order given, without NFC (NFC moves the shadda after the vowel, which changes the text the model
   reads).
   - Initialisms and all-caps words: check each one's IPA **inside its sentences**, because espeak-ng can read the same
     token differently alone and in context. Both directions occur: espeak-ng reads as a word one that the speaker
     spells (write the letters separately, e.g. `G P S`), or spells one that the speaker says as a word (write it
     lower-case, e.g. an ordinary word in capitals). List them for the user with espeak-ng's reading; use the same
     spelling at serving.
   - Invented words, brand and product names: check how espeak-ng reads each one (its IPA). If it is wrong, respell
     the word in the text so that espeak-ng reads it as the speaker says it, and use the same respelling at serving;
     or (not yet validated) correct its phonemes in the `ipa` field, which leaves the model to learn the reading from the
     recordings. Tell the user which words were changed. ASR does not know such words: their CER means little, listen.
   - Loanwords and words of another language (e.g. English words in a French or Hindi sentence): espeak-ng
     either reads them in the other language and marks them `(en)…(<lang>)` (keep, see "Check the IPA") or reads them
     with the language's own rules and no marker. Neither is an error; list them for the user to listen to.
   - Typos and words glued together ("theyare"): they change the IPA. List them; correct only what the user confirms.

Use all clean data the user has; more than the task file's minimum helps long and difficult text.

## Reference ("context") clips

Speaker similarity = cosine similarity of `titanet_large` embeddings, one embedding per clip:

```python
from nemo.collections.asr.models import EncDecSpeakerLabelModel
sv = EncDecSpeakerLabelModel.from_pretrained("titanet_large").eval().cuda()
emb = sv.get_embedding(path).squeeze()      # per clip; normalise, then dot products = cosine similarities
```

- **Training:** every utterance gets the **most similar** other clip of the same speaker from the training split
  (file duration ≥ 3 s, never the utterance itself), for every kind of voice. Utterances of 3 s or more whose most
  similar clip is below 0.6 similarity (the pretraining value) are dropped; shorter utterances and clips of 1–2 words
  (which the added trailing pause can push just over 3 s) need no threshold (short clips rarely reach it). Choose the
  references after all drops: a clip dropped by any check (e.g. this rule) is not a reference candidate; repeat until nothing changes. No upper duration limit: the loader cuts clips
  longer than the config's `context_duration_max` (10 s).
- **Validation, test, inference:** propose 5 fixed, typical clips: clips ≥ 3 s (ideally ≥ 4 s; speaker similarity is
  unreliable below ~2 s) with the highest mean similarity to the voice's other clips of ≥ 3 s, from any split. Ask the
  user once to listen to them and keep them, swap any, or add clips of their own if none fits the voice's style (check
  an added clip: the same speaker by similarity, ≥ 3 s; warn the user if it fails). **If the user chooses the reference
  clip(s), build the validation, test and evaluation manifests with only those clips**; use the 5 proposed clips when
  the user does not choose. Utterance i gets clip i mod (number of clips); the reference clip must differ from the
  utterance's own audio (if it is the same, use the next clip). Do not cut or pad the clips: the evaluator and `do_tts`
  handle the duration.

## Manifest

NeMo JSONL, one line per utterance, same fields in the training, validation and test manifests (validation computes
the training loss, so it needs `ipa` too); extra fields such as an id are allowed, except `normalized_text` and
`original_text`: when present, the evaluator uses them as the CER reference (and IPA generation reads
`normalized_text`), so use them only for the text the model reads. Do not name an extra field
`context_speaker_similarity` or `cer`: the fine-tuning config's data loader reads these two fields and silently drops
training cuts with `context_speaker_similarity` < 0.6 or `cer` > 0.03 (`min_context_speaker_similarity`, `max_cer`);
store your own similarity or ASR scores under other names.

| Field | Value |
| --- | --- |
| `audio_filepath`, `duration`, `text` | the utterance |
| `language` | code from the language table above |
| `speaker` | `"\| Language:<lang> Dataset:<name> Speaker:<voice_name> \|"` (the Shar script checks this format; `<voice_name>` = the name the user gives the voice, the same in every line of that voice; no spaces in `<name>` or `<voice_name>`: the loader splits the string on spaces) |
| `context_audio_filepath`, `context_audio_duration` | the reference clip |
| `ipa` | phoneme-channel languages: espeak-ng **1.52.0** phonemes with the voice from the language table; text-channel languages: omit |

IPA for one utterance: pass its text (`normalized_text` if present, else `text`) on stdin to
`espeak-ng -v <voice> --ipa -q` and collapse all whitespace in the output to single spaces. Other espeak-ng versions give
phonemes the model was not trained on; check `espeak-ng --version` (must report 1.52.0) before generating. The
output has no punctuation other than what espeak-ng writes itself (e.g. hyphens in French); that is expected and
matches the checkpoint's training data. Use the output as written: do not edit it.

Check the IPA before training, because a bad string trains silently:
- no empty strings;
- language-switch markers such as `(en)…(fr)`: espeak-ng writes them when it reads a word in another language.
  **Keep them** (loanwords, names, English words in Latin letters inside Hindi text): the checkpoint's training data
  kept espeak-ng's output unchanged, markers included; do not remove them, respell the words or drop the utterances.

## Lhotse Shar and codec codes

Training split only (validation and test stay JSONL manifests). Absolute paths in the manifest → `--audio-base-dir /`.

```bash
python scripts/magpietts/create_lhotse_shar_from_nemo_manifest.py \
  --manifest-path <voice>/train.jsonl --audio-base-dir / --output-dir <voice>/shar \
  --num-jobs <CPU cores> --processing-chunk-size 256 --audio-format flac --shuffle --shuffle-seed 42
python scripts/magpietts/extend_lhotse_shards_with_audio_codes.py \
  --cuts-dir <voice>/shar/cuts --target-audio-dir <voice>/shar/target_audio \
  --context-audio-dir <voice>/shar/context_audio --output-dir <voice>/shar \
  --codec-model-name codes_25fps --codec-model-path <codec>.nemo --codec-frame-rate 25 \
  --devices 1 --num-nodes 1 --batch-size 32
```

`--codec-frame-rate 25` must match the release's codec (the script's default is another codec's rate). `--shuffle`
also writes `train_shuffled.jsonl` next to the input manifest; it is a by-product, not a second training manifest.
Before training, decode a few codes back to audio and listen / compare with the originals:

```python
import glob, numpy as np, soundfile as sf, torch
from lhotse import CutSet
from nemo.collections.tts.models import AudioCodecModel
shar = "<voice>/shar"
fields = {"cuts": sorted(glob.glob(f"{shar}/cuts/cuts.*.jsonl.gz"))[:1],
          "target_codes": sorted(glob.glob(f"{shar}/codes_25fps/target_codes/codes.*.tar"))[:1],
          "target_audio": sorted(glob.glob(f"{shar}/target_audio/recording.*.tar"))[:1]}
codec = AudioCodecModel.restore_from("<codec>.nemo", map_location="cuda").eval()
for n, cut in zip(range(8), CutSet.from_shar(fields=fields)):
    codes = torch.from_numpy(cut.load_target_codes().astype(np.int64))[None].cuda()
    with torch.inference_mode():
        audio, alen = codec.decode(tokens=codes, tokens_len=torch.tensor([codes.shape[-1]], device="cuda"))
    sf.write(f"decoded_{n}.wav", audio[0, :alen[0]].float().cpu().numpy(), codec.output_sample_rate)
    sf.write(f"original_{n}.wav", cut.load_audio()[0], cut.sampling_rate)
```

The decoded length should match `cut.duration` within one codec frame (40 ms).

Compute: speaker embeddings, codec codes and the optional ASR check run on a GPU in minutes for ~5 h of audio (about
20–30 min with the ASR check). Without a GPU, speaker embeddings take a few minutes and codec codes ~15–20 min per
~5 h voice on 16 CPU cores (drop `.cuda()` and load with `map_location="cpu"` in the snippets); the ASR check needs a
GPU. Silence fixes, the text checks and IPA run on CPU. Listening checks are for the user: prepare short lists
(clipped clips, initialisms and loanwords with espeak-ng's reading, decoded samples next to
their originals, the proposed reference clips, and the dropped utterances if the ASR check ran) and ask the user to
listen before training. If the jobs run as a queued chain that cannot wait for the answer, stop the chain before
training, or record in the ledger that the default ("no problems") was used.

The training input config (`<voice>_input_cfg.yaml`) is one Lhotse Shar group (with several voices: one group for all
the voices of a language, see below); `NNNNNN` = number of shards
− 1 written with six digits (`ls <voice>/shar/cuts/cuts.*.jsonl.gz | wc -l` minus 1, e.g. `{000000..000012}` for 13
shards; the four tar folders must have the same count, and the number of cuts must equal the manifest's lines):

```yaml
- type: group
  weight: 1.0
  tags: {task: tts, lang: <language code>}
  input_cfg:
  - type: lhotse_shar
    weight: 1.0
    shar_path:
      cuts: <voice>/shar/cuts/cuts.{000000..NNNNNN}.jsonl.gz
      target_audio: <voice>/shar/target_audio/recording.{000000..NNNNNN}.tar
      context_audio: <voice>/shar/context_audio/recording.{000000..NNNNNN}.tar
      target_codes: <voice>/shar/codes_25fps/target_codes/codes.{000000..NNNNNN}.tar
      context_codes: <voice>/shar/codes_25fps/context_codes/codes.{000000..NNNNNN}.tar
    tags:
      tokenizer_names: [nemotron_nano_30b]
```

Several voices of one language: build each voice's lines as above (its own held-out sets, its own reference clips,
its own `speaker` field), then concatenate the voices' training lines into one training manifest, build one Lhotse Shar
folder from it and list it as one `- type: group` block. Lhotse then draws the training samples in
proportion to the audio, so each voice's share of the training steps is its hours ÷ the total hours: tell the user each
voice's share before training, and that a voice with a much smaller share than the others may come out weaker
(lower similarity, more errors on difficult text). Several languages: one Shar folder and one group per language,
each with weight 1.

If the user wants every voice to get the same share of the training steps: build one Lhotse Shar folder per voice
(from that voice's training lines) and list one `- type: group` block per voice, each with weight 1; the clips of a
smaller voice are then repeated more often. The validation set stays the merged one (`references/new-voices-supported-language/training.md`).

Voices that read the same sentences (e.g. one script recorded by two speakers): hold out every voice's validation and
test sentences from the training data of all voices; otherwise the model has already learned a test sentence through
another voice and the test numbers look better than they are. Sentences shared within the training data, or within the
validation / test sets, are fine. Compare sentences after lower-casing them and removing punctuation and extra spaces.
Choose each voice's validation and test sentences first among the sentences no other voice read, so that
few training lines are removed; if a voice has fewer such sentences than it needs, take the rest among the shared
sentences that the fewest other recordings share. Tell the user how many training lines and hours per voice the
removal costs.
If the user delivers their own training / validation / test split, keep it: do not re-split or move sentences; the
preference above applies only to splits you make. Still check it: count the validation and test sentences that appear
in the training data of any voice (the same voice included), tell the user, and by default remove those lines from the
training data (not from validation / test); report the lines and hours removed. After the removal, compare each
voice's training hours with the ~1 h of the task file; if a voice falls clearly below it, tell the user and offer to
keep those lines instead (the test numbers are then less trustworthy).
