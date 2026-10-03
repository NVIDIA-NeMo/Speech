# Speech LLM (speechlm2) Tutorials
------------

Tutorials for speech-augmented language models (SALM) in the `speechlm2` collection.

1) `SpeechLM_With_NeMo_Automodel`: Train a SALM from a pretrained ASR encoder and a pretrained LLM with NeMo Automodel:
data preparation with Lhotse, training, checkpoint conversion to HuggingFace format, and evaluation.

2) `finetuning/`: **SALM fine-tuning series**. Sixteen notebooks, each adapting one SALM checkpoint to a new task on a
public benchmark (new-language ASR, speech translation, spoken QA, intent and emotion recognition, keyword spotting,
sound and music classification, language ID, code-switching and accented ASR, pronunciation assessment,
anti-spoofing, stuttering detection). Each has a fair baseline, one test evaluation and expected results to check
against. Start with [`finetuning/README.md`](finetuning/README.md).
