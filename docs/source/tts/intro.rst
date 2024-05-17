Text-to-Speech (TTS)
====================

Text-to-Speech (TTS) synthesis refers to a system that converts textual inputs into natural human speech. The synthesized speech is expected to sound intelligible and natural. With the resurgence of deep neural networks, TTS research has achieved tremendous progress. NeMo implementation focuses on the state-of-the-art neural TTS where both **cascaded** and **end-to-end** (upcoming) systems are included,

1. **Cascaded TTS** follows a three-stage process. *Text analysis stage* transliterates grapheme inputs into phonemes by either looking up in a canonical dictionary or using a grapheme-to-phoneme (G2P) conversion; *acoustic modeling stage* generates acoustic features from phoneme inputs or from a mixer of graphemes and phonemes. NeMo chooses mel-spectrograms to represent expressive acoustic features, so we would use the term in the context, mel-spectrogram generators or acoustic models, interchangeably; *vocoder stage* synthesizes waveform audios from acoustic features accordingly.
2. **End-to-End TTS** alternatively integrates the above three stages as a single model so that it directly synthesizes audios from graphemes/phonemes inputs without any intermediate processes.

We will illustrate details in the following sections.

.. toctree::
    :maxdepth: 2

    models
    datasets
    checkpoints
    configs
    g2p

.. include:: resources.rst
