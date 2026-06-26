.. _asr_language_modeling_and_customization:

#######################################
ASR Language Modeling and Customization
#######################################

NeMo supports decoding-time customization techniques such as *language modeling* and *word boosting*,
which improve transcription accuracy by incorporating external knowledge or domain-specific vocabulary—without retraining the model.


Decoder Types
-------------

NeMo ASR models use different decoder architectures. The table below summarizes them:

.. list-table::
   :header-rows: 1

   * - Decoder
     - Type
     - Description
     - Models
   * - **CTC**
     - Non-autoregressive
     - Connectionist Temporal Classification. Fast inference, supports LM fusion and word boosting.
     - Parakeet-CTC, FastConformer-CTC
   * - **RNN-T**
     - Autoregressive
     - Recurrent Neural Network Transducer. Strong accuracy, streaming-friendly.
     - Parakeet-RNNT, FastConformer-Transducer
   * - **TDT**
     - Autoregressive
     - Token-and-Duration Transducer. Extends RNN-T with duration prediction for better timestamps.
     - Parakeet-TDT
   * - **AED**
     - Autoregressive
     - Attention Encoder-Decoder. Multi-task capable (ASR + AST), prompt-based language control.
     - Canary-1B, Canary-1B-V2, Canary-1B-Flash
   * - **Hybrid**
     - Both
     - Joint RNN-T + CTC training. Use either decoder at inference time.
     - FastConformer Hybrid models


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


CUDA Graphs
-----------

CUDA graphs accelerate decoding by capturing and replaying GPU operations, eliminating kernel launch overhead.
Support varies by decoder strategy:

.. list-table::
   :header-rows: 1

   * - Strategy
     - Config Parameter
     - Default
     - Notes
   * - ``greedy_batch`` (RNN-T, TDT)
     - ``use_cuda_graph_decoder``
     - ``true``
     - Requires ``loop_labels=True`` and ``blank_as_pad=True``
   * - ``maes_batch``, ``malsd_batch`` (beam)
     - ``allow_cuda_graphs``
     - ``true``
     - Batched beam search strategies
   * - Non-batched ``greedy`` / ``beam``
     - N/A
     - N/A
     - Not supported; standard decoding used

To disable CUDA graphs (e.g. for debugging or when preserving alignments with frame-looping):

**Via Python (at runtime):**

.. code-block:: python

    model.disable_cuda_graphs()

**Greedy decoding** — use ``use_cuda_graph_decoder=true/false``:

.. code-block:: bash

    python examples/asr/speech_to_text_eval.py \
       pretrained_name="nvidia/parakeet-rnnt-1.1b" \
       dataset_manifest=<dataset_manifest> \
       batch_size=32 \
       output_filename=decoded.jsonl \
       rnnt_decoding.strategy="greedy_batch" \
       rnnt_decoding.greedy.use_cuda_graph_decoder=true

**Beam decoding** — use ``allow_cuda_graphs=true/false``:

.. code-block:: bash

    python examples/asr/speech_to_text_eval.py \
       pretrained_name="nvidia/parakeet-rnnt-1.1b" \
       dataset_manifest=<dataset_manifest> \
       batch_size=32 \
       output_filename=decoded.jsonl \
       rnnt_decoding.strategy="malsd_batch" \
       rnnt_decoding.beam.max_symbols_per_step=10 \
       rnnt_decoding.beam.beam_size=12 \
       rnnt_decoding.beam.allow_cuda_graphs=true

When unsupported, NeMo falls back to standard decoding automatically.


Confidence Estimation
---------------------

NeMo supports per-frame, per-token, and per-word confidence scores during decoding.
Confidence estimation helps applications decide when to trust ASR output and when to request human review.

.. code-block:: yaml

  decoding:
    confidence_cfg:
      preserve_frame_confidence: false
      preserve_token_confidence: false
      preserve_word_confidence: false
      exclude_blank: true
      aggregation: "mean"       # mean, min, max, prod
      method_cfg:
        name: "entropy"         # max_prob or entropy
        entropy_type: "tsallis" # gibbs, tsallis, renyi
        alpha: 0.33
        entropy_norm: "exp"     # lin or exp

**Confidence methods:**

* ``max_prob``: Maximum token probability as confidence. Simple and fast.
* ``entropy``: Normalized entropy of the log-likelihood vector (default). Entropy types:

  - ``gibbs``: Standard Gibbs entropy
  - ``tsallis``: Tsallis entropy (default, recommended)
  - ``renyi``: Renyi entropy

**Aggregation** combines frame-level scores into token/word scores: ``mean``, ``min``, ``max``, or ``prod``.

For TDT models, set ``tdt_include_duration_confidence: true`` to include duration prediction confidence.


.. toctree::
   :maxdepth: 1
   :hidden:

   asr_customization/ngpulm_language_modeling_and_customization
   asr_customization/neural_rescoring
   asr_customization/legacy_language_modeling_and_customization
   asr_customization/ngram_utils
   asr_customization/word_boosting