# Medical ASR from a test-only benchmark, with synthetic data

This worked example adapts a SALM checkpoint to Indian medical speech when **no training data exists**: only a public
test set, and public text resources. The fine-tuned model outperforms both a medical-specialist open model and every
commercial system on the benchmark's published leaderboard.

## The problem

[Eka Care Medical ASR Evaluation](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset)
(English, MIT) has 3,619 recordings (8.4 h) of Indian doctors and medical students:

| Clip type | Clips | Example |
|---|---:|---|
| Isolated medical entities | 2,206 | "Jardiance 10Mg Tablet", "left inguinal ln", "Cipla Limited" |
| Dictated sentences | 1,303 | drug-information and case-report sentences |
| Doctor–patient conversation | 110 | including Hindi–English code-switching |

It's a **test set only**. Its references follow a pharmacy-catalog convention: digits, strengths joined to the number,
brand hyphens kept, company names in full. Scoring uses Eka's own toolkit,
[KARMA](https://github.com/eka-care/KARMA-OpenMedEvalKit), which lowercases and *removes* punctuation, so "Piopar-MF"
scores as one word and "Piopar MF" as two. It reports four metrics:

| Metric | What it measures |
|---|---|
| WER | word error rate |
| CER | character error rate |
| semWER | a semantic WER that forgives equivalent spellings |
| kwWER | keyword WER over the annotated medical entities |

The base model, prompted to transcribe, reaches **30.8% WER**. On isolated entities it often writes Indian-accented
English in Devanagari or Tamil script. Naming "English, in the Latin alphabet" in the prompt brings it to 23.8%.

## Approach

Three ideas carry the result:

1. **Synthetic speech for the domain.** Write medical text, convert each line to how it's said aloud, and synthesize
   it with several TTS voices. Mix in real speech with the right accent and the right domain.
2. **Convention-conditioned labels.** Label each synthetic clip in several written conventions, state the convention
   in the prompt, and at test time state the benchmark's convention. The model learns to *follow* a transcription
   convention instead of guessing one.
3. **Vocabulary coverage.** Public Indian medicine catalogs supply real brand names, salts and manufacturers, so the
   model hears the long tail of brand names it will be asked to spell.

## Data

| Source | Audio | Built with |
|---|---:|---|
| Medical text written by an open LLM (gpt-oss-120b): 51k lines over 18 specialties × 7 styles | 57 h | `sdg/generate_text.py` |
| Catalog text from the PharmaLens formulary (MIT): product names, salts, manufacturers, 12k prescriptions around real brands | 34 h | `sdg/catalog_text.py` |
| Lexicon text from the 1mg medicine list (MIT, 222k names): 50k names, one per brand family first, plus 12k prescriptions | 62 h | `sdg/lexicon_text.py` |
| Common Voice English, India/South Asia accent tag (CC0) | 29 h | `prepare_benchmarks.py cv_indian` |
| MultiMed English, real medical lectures and interviews (MIT); caption-style transcripts removed | 63 h | `prepare_benchmarks.py multimed_en` |

All synthetic text goes through `sdg/spoken_form.py`. An LLM writes how each line is spoken ("HbA1c" → "H B A one C",
"500mg" → "five hundred milligram"), and the TTS reads that spoken form while the label stays the written text. Voices
come from MagpieTTS (`sdg/tts_magpie.py`) and Kokoro-82M (`sdg/tts_kokoro.py`), including Hindi-accented voices, with
a few voices held out for a synthetic dev set.

`sdg/render_conventions.py` then turns each clip into two training rows:
- **Each row states a convention.** It picks one, renders the label in it, and names it in the prompt. The
  conventions vary:
  - numbers as digits or words;
  - units joined to drug strengths or spaced;
  - lab units always spaced;
  - "Private Limited" or "Pvt Ltd";
  - hyphens kept or dropped;
  - "/" or "by" between strengths;
  - casing and punctuation.
- **Some prompts carry a term list.** It holds the catalog name and near-miss distractors, with the correct term
  missing a quarter of the time, so the model learns to copy a spelling only when the audio supports it.

**Contamination.** `check_contamination.py` drops every training row that equals a test reference or shares an
8-gram with one. The catalogs naturally contain many of the benchmark's brand names; that's domain vocabulary, as it
would be in a hospital formulary. No test audio or test transcript is used.

## Recipe

Two models are trained from the same base checkpoint with the same settings:
- the speech encoder and projection trained;
- LoRA on the LLM;
- learning rate 1e-4, 60 warmup steps;
- 24k tokens per batch;
- 2,000 steps.

The models differ only in the data mix, and the final model is their uniform weight average (`average_checkpoints.py`):

| Model | Synthetic (convention-conditioned) | Common Voice Indian | MultiMed | Term-list share | Checkpoint |
|---|---|---:|---:|---:|---|
| A: catalog-focused | medical + catalog text, 60% | 15% | 25% | 50% | step 2,000 |
| B: lexicon-focused | medical + catalog + lexicon text, 55% | 20% | 25% | 15% | step 1,000 |

Each model's checkpoint was chosen on the tune split. The average is better than either model alone on tune: WER 10.57,
against 10.65 for A and 11.20 for B.

**Inference is a single pass with one fixed prompt** that states the benchmark's convention:

> Transcribe this Indian English medical audio verbatim in the Latin alphabet. Write numbers as digits. Attach units to
> drug strengths without a space (500mg, 5ml), but keep a space before lab units (6.35 mg/dL). Write company suffixes
> in full (Private Limited). Keep hyphens in brand names (Piopar-MF). Use a slash between combined strengths
> (15mg/500mg). Use normal capitalization. Use normal punctuation.

There's no retrieval, no second pass and no post-processing.

## Evaluation protocol

- **Selection.** The test set is split by speaker into a *tune* part (855 clips) and a *held-out* part (2,764 clips).
  The convention, the checkpoints and the average were all chosen on tune. Held-out was decoded once per final
  candidate.
- **Like-for-like comparison.** The open medical specialist
  [Parrotlet-a-en-5b](https://huggingface.co/ekacare/parrotlet-a-en-5b) and Whisper-large-v3 were run through the same
  scorer: Parrotlet with its own `transcribe()` code and default settings, Whisper with greedy decoding.
  - Run through this pipeline, Parrotlet scores 13.4% WER on the full test set, against 10.9% on the dataset card.
  - Whisper scores 14.75% here, against 15.7% published.
  - Published numbers therefore aren't directly comparable, so the head-to-head comparison uses only systems run here.

## Outcome

**Held-out speakers (2,764 clips), single pass, KARMA scorer, all systems run here:**

| System | WER | semWER | kwWER | Entity-clip WER | Entity semWER |
|---|---:|---:|---:|---:|---:|
| Base checkpoint, best prompt | 22.44 | 10.0 | 9.2 | 52.1 | 28.8 |
| Whisper-large-v3 | 14.92 | 8.2 | 8.0 | 38.8 | 21.9 |
| Parrotlet-a-en-5b (medical specialist) | 13.79 | 9.1 | 7.4 | 32.8 | 20.1 |
| **Fine-tuned (this recipe)** | **12.17** | **7.6** | **6.5** | **30.0** | **18.4** |

The fine-tuned model is best on every metric, including entity semWER (18.4 vs 20.1 for the medical specialist).

Exact matches of annotated entities, held-out:

| Entity type | Mentions | Parrotlet | Fine-tuned |
|---|---:|---:|---:|
| Clinical findings | 4,420 | 82.9% | **85.7%** |
| Generic drug names | 703 | 66.6% | **74.1%** |
| Advice | 303 | 61.7% | **68.0%** |
| Diagnostics | 797 | **74.0%** | 73.4% |
| Brand-name drugs | 1,209 | **38.3%** | 36.6% |
| All | 8,280 | 74.0% | **76.1%** |

**Full test set, against the published leaderboard.** These numbers come from this pipeline, which reads about one
point optimistic relative to published numbers (Whisper). A quarter of the set (the tune split) informed selection.

| System | WER | CER | semWER | kwWER |
|---|---:|---:|---:|---:|
| **Fine-tuned (this recipe)** | **11.8** | **4.7** | 7.4 | **6.5** |
| Gemini 2.5 Flash (published) | 14.8 | 5.5 | 7.2 | 6.8 |
| GPT-4o (published) | 16.1 | 9.7 | 11.6 | 11.7 |
| AWS Transcribe (published) | 18.3 | 7.4 | 11.1 | 12.2 |
| ElevenLabs Scribe v1 (published) | 18.6 | 8.7 | 10.2 | 9.3 |

General English is preserved: LibriSpeech test-clean WER moves from 1.54% to 1.69%.

## Limits

- **The brand-name long tail.** Brand-name drugs and diagnostics are the two entity types where the specialist still
  leads (38.3% vs 36.6% exact brand matches on held-out). Eka's brands are a long tail: the 20,000 most common brand
  families cover only 45% of mentions. One synthetic rendering per name across 100k families
  is too sparse to learn spellings. Closing this gap most likely needs real recordings of Indian doctors saying brand
  names.
- **Validate on real speech.** A dev set built from the same synthetic text can reward the wrong thing: catalog spelling
  conventions, or snapping unfamiliar words to known brands. Keep at least 100 real, labeled in-domain clips for
  selection.

## Cost

About **14 GPU-hours on one NVIDIA GB300 (~$125 at $9/GPU-hour)**, roughly one day of wall-clock time:

| Stage | GPU-hours |
|---|---:|
| Text generation and spoken forms (LLM) | ~1 |
| Speech synthesis (MagpieTTS + Kokoro), ~150 h of audio | ~5 |
| Two training runs, including exports | ~6.5 |
| Averaging and evaluation decodes | ~1.5 |

## Reproduce

```bash
FT=examples/speechlm2/finetuning
# 1. benchmark and real speech
python $FT/prepare_benchmarks.py eka
python $FT/prepare_benchmarks.py cv_indian --hours 30
python $FT/prepare_benchmarks.py multimed_en
# 2. synthetic text -> spoken form -> speech (repeat for catalog_text.py and lexicon_text.py)
python $FT/sdg/generate_text.py --out text.jsonl --rounds 15 --per-request 40
python $FT/sdg/spoken_form.py --texts text.jsonl --out spoken.jsonl --exclude $SALM_FT_WORK/manifests/eka/test.json
python $FT/sdg/tts_kokoro.py --texts spoken.jsonl --out-dir tts --manifest tts.jsonl
# 3. convention-conditioned rows, then drop anything overlapping the test set
python $FT/sdg/render_conventions.py --inputs tts.jsonl --catalog base_catalog.json --out conv.json
python $FT/check_contamination.py --test $SALM_FT_WORK/manifests/eka/test.json --train conv.json --drop-out conv_clean.json
# 4. train (repeat with the Model B blend), average, evaluate with the convention prompt
python $FT/make_ft_config.py --checkpoint $SALM_CHECKPOINT --prompt manifest --out a.yaml --exp-dir exp_a \
    --train-manifest "conv_clean.json:0.6,cv_train.json:0.15,mm_train.json:0.25" --val-manifest synth_dev.json \
    --train-encoder --train-proj --lr 1e-4 --warmup-steps 60 --max-steps 2000 --batch-tokens 24000 \
    --limit-train-batches 100 --save-top-k -1
python $FT/run_training.py --config a.yaml --preflight --smoke --keep-steps 1500
python $FT/average_checkpoints.py --src export_a export_b --dst final
python $FT/vllm_task_eval.py --model final --manifest $SALM_FT_WORK/manifests/eka/test.json --out hyps.jsonl \
    --task asr --max-tokens 1024 --max-tokens-per-sec 10 --prompt "<convention prompt above>"
KARMA_SRC=path/to/KARMA-OpenMedEvalKit python $FT/sdg/score_eka.py --hyps hyps.jsonl --manifest $SALM_FT_WORK/manifests/eka/test.json
```
