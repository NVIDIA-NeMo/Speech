# SALM Fine-tuning Tutorials
------------

One speech LLM, sixteen tasks it wasn't built for. Each notebook takes the same SALM checkpoint (speech encoder →
projection → LLM backbone) and adapts it to a new task on a public benchmark: new languages, translation, spoken QA,
classification of speech, music and sounds, pronunciation scoring, anti-spoofing. Every notebook follows the same arc:

1. **the problem**: what the base model does, with real examples of its mistakes;
2. **a fair baseline**: several prompts; the test score of the prompt chosen on dev is the number to beat;
3. **the recipe**: data preparation, a config derived from the checkpoint, training with safety checks;
4. **selection on dev**: export, merge LoRA, decode candidate checkpoints, sometimes average them;
5. **one test evaluation**, plus a LibriSpeech check that general English survived;
6. **takeaways**: the lessons and pitfalls that carry over to your own data.

Each notebook downloads its own data and checks its numbers against the expected results (`✅ / ❌` at the end).
The tools live in [`examples/speechlm2/finetuning/`](../../../examples/speechlm2/finetuning/).

------------

0) `00_Overview_and_Setup`: The model, the environment (training and vLLM serving), the pipeline shared by all
tutorials, and seven ways a fine-tune can fail silently. Ends with a hands-on look at the base model and its prompts.

1) `01_New_Language_ASR_Swahili`: Teach the model a language it doesn't speak (FLEURS Swahili, 81.2% → 18.0% WER).
Language drift, why naming the language in the prompt is worth 20 points, and why `val_loss` picks the wrong
checkpoint.

2) `02_Speech_Translation_English_Catalan`: English speech → Catalan text (CoVoST 2, 15.2 → 27.8 BLEU). The model
already translates, but Spanish leaks in. Freeze the encoder when only the output language changes.

3) `03_Spoken_Question_Answering`: Answer a spoken question from a text passage, or say "unanswerable" (HeySQuAD,
43.3 → 84.4 F1). A test set that is 76% unanswerable, and a gain that comes from learning when to abstain.

4) `04_Intent_Classification_SLURP`: Spoken smart-home requests → 60 intents (SLURP, 50.6% → 87.9%). The dataset's own
label column is inconsistent; rebuilding the official `scenario_action` labels.

5) `05_Speech_Emotion_CREMA-D`: Six emotions from how, not what, is said (CREMA-D, actor-disjoint, 36.7% → 76.7%).
Checking the scorer against raw outputs, and why the benchmark must close lexical shortcuts.

6) `06_Keyword_Spotting_Speech_Commands`: 35 keywords (Speech Commands v2, error 5.2% → 2.6%). Label-sorted shards
that collapse the model to one word, and measuring progress as error-rate reduction near the ceiling.

7) `07_Environmental_Sound_ESC-50`: 50 environmental sounds (ESC-50, 40.3% → 62.0%). Official folds, and counting
batches per epoch so validation actually runs.

8) `08_Vocal_Sounds_VocalSound`: Laughter, coughs, sighs, sneezes (VocalSound, 76.7% → 91.3%). Datasets that lie about
their sample rate.

9) `09_Music_Genre_GTZAN_Probe`: Music genre (GTZAN fault-filtered, 79.0% → 86.2%). Weight fine-tuning has
little room to help; a 600-parameter linear probe on the frozen model's label log-probabilities gains +9% without
changing a weight.

10) `10_Instrument_Family_NSynth`: The instrument family of a single note (NSynth, 48.0% → 57.4%). Sampling a big
dataset safely, and a mid-run checkpoint that beats the final one.

11) `11_Language_ID_CommonLanguage`: Which of 45 languages is spoken (CommonLanguage, 49.8% → 92.9%). A shuffle buffer
is not a shuffle.

12) `12_Code_Switching_ASR_ASCEND`: Mandarin–English code-switching (ASCEND, 18.1% → 11.0% MER). Prompting for
expected languages, and a gentler learning rate.

13) `13_Accented_Conversational_ASR_EdAcc`: Accented conversational English (EdAcc, 18.4% → 15.0% WER). Greedy
repetition loops, a reference that says "ignore me" (and a fine-tune that learns to say it), and a benchmark where the
usual levers run out.

14) `14_Pronunciation_Assessment_speechocean762`: Score non-native pronunciation 0-10 (speechocean762, Pearson r 0.22 →
0.68). Regression written as text.

15) `15_Anti-Spoofing_ASVspoof2019`: Real voice or synthetic? (ASVspoof 2019 LA, EER 30.7% → 0.57% on unseen attacks).
Scoring detectors with label log-odds instead of generated words.

16) `16_Stuttering_Events_SEP-28k`: Detect five stuttering events (SEP-28k, macro-F1 13.0% → 62.0%). A 3-speaker
training split, and teaching a transcriber to hear what it was trained to ignore.

------------

**Requirements.** A large GPU (the expected results were measured on a single NVIDIA GB300; training uses 70-160 GB of
GPU memory), this repository with `speechlm2`, a separate environment with vLLM 0.28.0 for decoding, and a SALM
checkpoint in HuggingFace format (exported by `SALMAutomodel`), set via the `SALM_CHECKPOINT`
environment variable. See `00_Overview_and_Setup` for details and the other environment variables.
