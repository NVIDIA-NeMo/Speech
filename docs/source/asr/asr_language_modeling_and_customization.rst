.. _asr_language_modeling_and_customization:

#######################################
ASR Language Modeling and Customization
#######################################

NeMo supports decoding-time customization techniques such as *language modeling* and *word boosting*,
which improve transcription accuracy by incorporating external knowledge or domain-specific vocabulary—without retraining the model.

Language Modeling
-----------------

In NeMo two approaches of external language modeling are supported:

- **Language Model Fusion:** 
    Language model (LM) fusion integrates scores from an external statistical n-gram model into the ASR decoder.
    This helps guide decoding toward more likely word sequences based on text corpora.

    NeMo provides two approaches for language model shallow fusion with ASR systems:

    **1. NGPU-LM (Recommended for Production)**
        GPU-accelerated LM fusion for all major model types: CTC, RNN-T, TDT, and AED models.

        - Customization during both greedy and beam decoding.

        - Fast beam decoding for all major model types, offering only 20% RTFx difference between beam and greedy decoding.

        - Integration with NGPU-LM GPU-based ngram LM.

        For details, please refer to :ref:`ngpulm_ngram_modeling`

    **2. KenLM (Traditional CPU-based)**
        CPU-based LM fusion using the KenLM library.
        
        .. note::

            These approaches, especially beam decoding, can be extremely slow and are retained in the repository primarily for backward compatibility.
            If possible, we recommend using NGPU-LM for improved performance.

        For details, please refer to :ref:`ngram_modeling`

- **Neural Rescoring:** 
    When using the neural rescoring approach, a neural network is used to score candidates. A candidate is the text transcript predicted by the ASR model’s decoder. 
    The top K candidates produced by beam search decoding (with a beam width of K) are given to a neural language model for ranking.
    The language model assigns a score to each candidate, which is usually combined with the scores from beam search decoding to produce the final scores and rankings.

    For details, please refer to :ref:`neural_rescoring`.


Word Boosting
-------------

Word boosting increases the likelihood of specific words or phrases during decoding by applying a positive bias, helping the model better recognize names,
uncommon terms, and custom vocabulary.

- :ref:`word_boosting_gpupb` (preferred): GPU-accelerated phrase-boosting for CTC, RNN-T/TDT, and AED (Canary) models supporting greedy and beam search decoding.

- :ref:`word_boosting_flashlight`: Word-boosting method for CTC models with external n-gram LM.

- :ref:`word_boosting_ctcws`: Word-boosting method for hybrid (Transducer-CTC) models without LM.

For details, please refer to: :ref:`word_boosting`.


LM Training
-----------

NeMo provides tools for training n-gram language models that can be used for language model fusion or word-boosting.
For details, please refer to: :ref:`ngram-utils`.


.. toctree::
   :maxdepth: 1
   :hidden:

   asr_customization/ngpulm_language_modeling_and_customization
   asr_customization/neural_rescoring
   asr_customization/legacy_language_modeling_and_customization
   asr_customization/ngram_utils
   asr_customization/word_boosting