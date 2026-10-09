# Synthetic training data for speech fine-tuning

Scripts to build training data when only a test set exists. The worked example is
[medical ASR from a test-only benchmark](../docs/medical-asr-synthetic-data.md). The steps carry over to any domain:
write text, convert it to how it's spoken, synthesize it, and render convention-conditioned labels.

| Step | Script | Notes |
|---|---|---|
| domain text | `generate_text.py` | open LLM via vLLM (default `openai/gpt-oss-120b`); the domain is defined by the specialty, style and prompt lists at the top of the file, so edit them for your domain |
| catalog text | `catalog_text.py` | real product names, ingredients, manufacturers and prescriptions from a catalog JSON; holds out 5% of product families for dev |
| lexicon coverage | `lexicon_text.py` | names from a large CSV list, one per family first for breadth, plus LLM sentences around them |
| spoken form | `spoken_form.py` | how each line is said aloud ("HbA1c" → "H B A one C"); the TTS reads this, the label stays the written text; `--exclude` drops lines equal to test references |
| speech | `tts_kokoro.py`, `tts_magpie.py` | Kokoro-82M (fast, many voices) and MagpieTTS (batched); hold some voices out for a dev set |
| labels | `render_conventions.py` | each clip × several written conventions, stated in the prompt; optional near-miss term lists from a catalog |
| scoring | `score_eka.py` | the Eka Care benchmark's own KARMA metrics (WER/CER/semWER/kwWER) |

**Before training:**
- Run `../check_contamination.py` against your test set.
- Keep at least 100 real, labeled in-domain clips for checkpoint selection. A dev set made from the same synthetic text
  can reward the wrong thing.

**Requirements:**
- The text scripts need vLLM.
- `tts_magpie.py` runs in the NeMo environment.
- `tts_kokoro.py` needs `pip install kokoro "misaki[en]"`.
- `render_conventions.py` needs `num2words` and `rapidfuzz`.
- `score_eka.py` needs a checkout of [KARMA-OpenMedEvalKit](https://github.com/eka-care/KARMA-OpenMedEvalKit), passed
  in `$KARMA_SRC`.
