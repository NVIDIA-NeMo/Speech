.. _asr-configs-dataset-configuration:

NeMo ASR Configuration Files
============================

This page covers ASR-specific configuration. For general NeMo setup (Experiment Manager, trainer), see :doc:`../core/core`.
Example configs: `examples/asr/conf <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/conf>`_.


.. _asr-configs-preprocessor-configuration:

Preprocessor Configuration
--------------------------

If you are loading audio files for your experiment, you will likely want to use a preprocessor to convert from the
raw audio signal to features (e.g. mel-spectrogram or MFCC). The ``preprocessor`` section of the config specifies the audio
preprocessor to be used via the ``_target_`` field, as well as any initialization parameters for that preprocessor.

An example of specifying a preprocessor is as follows:

.. code-block:: yaml

  model:
    ...
    preprocessor:
      # _target_ is the audio preprocessor module you want to use
      _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
      normalize: "per_feature"
      window_size: 0.02
      ...
      # Other parameters for the preprocessor

Refer to the :ref:`Audio Preprocessors <asr-audio-preprocessors>` API section for the preprocessor options, expected arguments,
and defaults.

.. _asr-configs-augmentation-configurations:

Augmentation Configurations
---------------------------

There are a few on-the-fly spectrogram augmentation options for NeMo ASR, which can be specified by the
configuration file using a ``spec_augment`` section.

For example, there are options for `Cutout <https://arxiv.org/abs/1708.04552>`_ and
`SpecAugment <https://arxiv.org/abs/1904.08779>`_ available via the ``SpectrogramAugmentation`` module.

The following example sets up both ``Cutout`` (via the ``rect_*`` parameters) and ``SpecAugment`` (via the ``freq_*``
and ``time_*`` parameters).

.. code-block:: yaml

  model:
    ...
    spec_augment:
      _target_: nemo.collections.asr.modules.SpectrogramAugmentation
      # Cutout parameters
      rect_masks: 5   # Number of rectangles to cut from any given spectrogram
      rect_freq: 50   # Max cut of size 50 along the frequency dimension
      rect_time: 120  # Max cut of size 120 along the time dimension
      # SpecAugment parameters
      freq_masks: 2   # Cut two frequency bands
      freq_width: 15  # ... of width 15 at maximum
      time_masks: 5    # Cut out 10 time bands
      time_width: 25  # ... of width 25 at maximum

You can use any combination of ``Cutout``, frequency/time ``SpecAugment``, or neither of them.

You can also add audio augmentation pipelines via an ``augmentor`` section in ``train_ds``.
Augmentors are applied on-the-fly to audio data in the data layer. The following example
adds white noise (probability 0.5, level between -50 dB and -10 dB) and room impulse response
augmentation (probability 0.3, from a manifest of impulse responses):

.. code-block:: yaml

  model:
    ...
    train_ds:
    ...
        augmentor:
            white_noise:
                prob: 0.5
                min_level: -50
                max_level: -10
            impulse:
                prob: 0.3
                manifest_path: /path/to/impulse_manifest.json

Refer to the :ref:`Audio Augmentors <asr-api-audio-augmentors>` API section for more details.


Metric Configurations
---------------------

NeMo ASR models supports WER and BLEU metric logging during training and validation. All metrics are based on the TorchMetrics backend, allowing for distributed training without additional code.

Word Error Rate (WER)
~~~~~~~~~~~~~~~~~~~~~

WER is the default metric for all ASR models and measures transcription accuracy at the word or character level.

.. code-block:: yaml

  model:
    use_cer: false                  # Set to true for Character Error Rate instead (default: false)
    log_prediction: true            # Whether to log a sample prediction during training (default: true)
    batch_dim_index: 0              # Index of batch dimension in prediction tensors output. Set to 1 for RNNT models.

BLEU Score
~~~~~~~~~~

BLEU score can be used for ASR models to evaluate translation quality. NeMo's BLEU implementation is based on SacreBLEU for standardized, reproducible scoring:

.. code-block:: yaml

  model:
    bleu_tokenizer: "13a"        # SacreBLEU tokenizer type (see below). (default: "13a")
    n_gram: 4                    # Maximum n-gram order for BLEU calculation. (default: 4)
    lowercase: false             # Whether to lowercase before computing BLEU. (default: False)
    weights: null                # Optional custom weights for n-gram orders. (default: null)
    smooth: false                # Whether to apply smoothing to BLEU calculation. (default: False)
    check_cuts_for_bleu_tokenizers: false  # Enable per-sample tokenizer selection. (See below for more details.) (default: False)
    log_prediction: true         # Whether to log sample predictions. (default: True)
    batch_dim_index: 0           # Index of batch dimension in prediction tensors output. Set to 1 for RNNT models. (default: 0)

BLEU score relies on TorchMetrics' SacreBLEU implementation and supports all SacreBLEU tokenization options. Valid strings may be passed to ``bleu_tokenizer`` parameter to configure base tokenizer behavior during BLEU calculation. Available options are:

* ``"13a"`` - Default WMT tokenizer (mteval-v13a script compatible)
* ``"none"`` - No tokenization applied
* ``"intl"`` - International tokenization (mteval-v14 script compatible)  
* ``"char"`` - Character-level tokenization (language-agnostic)
* ``"zh"`` - Chinese tokenization (separates Chinese characters, uses 13a for non-Chinese)
* ``"ja-mecab"`` - Japanese tokenization using MeCab morphological analyzer
* ``"ko-mecab"`` - Korean tokenization using MeCab-ko morphological analyzer
* ``"flores101"`` / ``"flores200"`` - SentencePiece models from Flores datasets

**Note** Due to their unique orthographies, it is highly recommended to use ``zh``, ``ja-mecab``, or ``ko-mecab`` tokenizers for Chinese, Japanese, and Korean target evaluations, respectively. For more information on SacreBLEU tokenizers, please refer to the `SacreBLEU documentation <https://github.com/mjpost/sacrebleu>`__.

**Dynamic Tokenizer Selection**

In multilingual training scenarios, it is somtimes desireable to configure the BLEU tokenizer per sample to avoid sub-optimal parsing (e.g. tokenizing Chinese characters as English words). This can be toggled with ``check_cuts_for_bleu_tokenizers: true``. When enabled with Lhotse dataloading, BLEU will check individual ``cuts`` in a batch's Lhotse ``CutSet`` for the ``bleu_tokenizer`` attribute. If found, the tokenizer will be used for that sample. If not, the default ``bleu_tokenizer`` from config will be used.

MultiTask Metrics
~~~~~~~~~~~~~~~~~

Multiple metrics can be configured simultaneously using a ``MultiTaskMetric`` config. This is done by specifying in the config each desired metric as a DictConfig entry with a custom key name and ``_target_`` path, along with desired properties. All properties specified within a metric config will be passed only to the metric class. All properties specified at the top level of the config will be inherited by all submetrics.

.. code-block:: yaml

  model:
    multitask_metrics_config:
      log_prediction: true
      metrics:
        wer:
          _target_: nemo.collections.asr.metrics.wer.WER
          use_cer: true
          constraint: ".task==transcribe"  # Only apply WER to transcription samples
        bleu:
          _target_: nemo.collections.asr.metrics.bleu.BLEU
          bleu_tokenizer: flores101
          lowercase: true
          check_cuts_for_bleu_tokenizers: true
          constraint: ".task==translate"   # Only apply BLEU to translation samples

**Metric Constraints**

Each metric within ``MultiTaskMetric`` can be configured with an optional boolean ``constraint`` pattern that filters batch samples before metric computation. This allows validation to be limited to only applicable samples in a batch (e.g. only apply WER to transcription samples, only apply BLEU to translation samples). Constraint patterns match against property keywords in the batch's Lhotse CutSet.

.. code-block:: yaml

  model:
    multitask_metrics_config:
      metrics:
        pnc_wer:
          _target_: nemo.collections.asr.metrics.wer.WER
          constraint: ".task==transcribe and .pnc==true"

        multilingual_bleu:
          _target_: nemo.collections.asr.metrics.bleu.BLEU
          constraint: "(.source_lang!=.target_lang) or .task==translate"

**Note:** MultiTaskMetric is currently only supported for AED multitask models.


Tokenizer Configurations
------------------------

Models using sub-word encoding require a ``tokenizer`` section:

.. code-block:: yaml

  model:
    tokenizer:
      dir: "<path to tokenizer>"
      type: "bpe"  # or "wpe"

For multilingual models, use ``AggregateTokenizer`` (``type: "agg"``), which combines multiple monolingual tokenizers
into one. Each sub-tokenizer is assigned a language id that must match the ``lang`` field in the manifest:

.. code-block:: yaml

  model:
    tokenizer:
      type: agg
      langs:
        en:
          dir: /path/to/en_tokenizer
          type: bpe
        es:
          dir: /path/to/es_tokenizer
          type: bpe


Transducer Configurations
-------------------------

CTC configs can be extended for Transducer training by adding prediction network, joint network, decoding, and loss sections.

.. code-block:: yaml

    model:
      model_defaults:
        enc_hidden: 256
        pred_hidden: 256
        joint_hidden: 256

      decoder:
        _target_: nemo.collections.asr.modules.RNNTDecoder
        blank_as_pad: true
        prednet:
          pred_hidden: ${model.model_defaults.pred_hidden}
          pred_rnn_layers: 1

      joint:
        _target_: nemo.collections.asr.modules.RNNTJoint
        log_softmax: null
        fuse_loss_wer: false
        fused_batch_size: 16
        jointnet:
          joint_hidden: ${model.model_defaults.joint_hidden}
          activation: "relu"

      decoding:
        strategy: "greedy_batch"  # greedy, greedy_batch, beam, tsd, alsd, maes
        greedy:
          max_symbols: 10
        beam:
          beam_size: 2
          score_norm: true

      loss:
        loss_name: "default"
        warprnnt_numba_kwargs:
          fastemit_lambda: 0.0

For large vocabularies, use ``SampledRNNTJoint`` with ``n_samples`` to reduce memory.
`FastEmit <https://arxiv.org/abs/2010.11148>`_ regularization controls transducer latency via ``fastemit_lambda``.

For decoding customization (confidence scores, CUDA graphs, language models, word boosting), see :doc:`ASR Language Modeling and Customization <./asr_language_modeling_and_customization>`.


InterCTC Config
---------------

All CTC-based models also support `InterCTC loss <https://arxiv.org/abs/2102.03216>`_. To use it, you need to specify
2 parameters as in example below

.. code-block:: yaml

   model:
      # ...
      interctc:
        loss_weights: [0.3]
        apply_at_layers: [8]

which can be used to reproduce the default setup from the paper (assuming the total number of layers is 18).
You can also specify multiple CTC losses from different layers, e.g., to get 2 losses from layers 3 and 8 with
weights 0.1 and 0.3, specify:

.. code-block:: yaml

   model:
      # ...
      interctc:
        loss_weights: [0.1, 0.3]
        apply_at_layers: [3, 8]

Note that the final-layer CTC loss weight is automatically computed to normalize
all weight to 1 (0.6 in the example above).


Stochastic Depth Config
-----------------------

`Stochastic Depth <https://arxiv.org/abs/2102.03216>`_ is a useful technique for regularizing ASR model training.
Currently it's only supported for :ref:`nemo.collections.asr.modules.ConformerEncoder <conformer-encoder-api>`. To
use it, specify the following parameters in the encoder config file to reproduce the default setup from the paper:

.. code-block:: yaml

   model:
      # ...
      encoder:
        # ...
        stochastic_depth_drop_prob: 0.3
        stochastic_depth_mode: linear  # linear or uniform
        stochastic_depth_start_layer: 1

See :ref:`documentation of ConformerEncoder <conformer-encoder-api>` for more details. Note that stochastic depth
is supported for both CTC and Transducer model variations (or any other kind of model/loss that's using
conformer as encoder).


.. _Hybrid-Transducer-CTC-Prompt_model__Config:

Hybrid-Transducer-CTC with Prompt Conditioning Configuration
------------------------------------------------------------

The :ref:`Hybrid-Transducer-CTC model with prompt conditioning <Hybrid-Transducer-CTC-Prompt_model>` 
(``EncDecHybridRNNTCTCBPEModelWithPrompt``) extends the base hybrid model to support prompt-based multilingual ASR/AST.

**Key Configuration Parameters:**

The model introduces several prompt-specific configuration parameters in the ``model_defaults`` section:

.. code-block:: yaml

  model:
    model_defaults:
      # Prompt Feature Configuration
      initialize_prompt_feature: true  # Enable prompt conditioning
      num_prompts: 128                 # Number of supported prompt categories
      prompt_dictionary: {             # Mapping from identifiers to prompt indices
        # Language prompts (0-99)
        'en-US': 0,
        'de-DE': 1,
        'fr-FR': 2,
        'es-ES': 3,
        # Task/domain prompts (100-127)
        'pnc': 100,                    # Punctuation mode
        'no_pnc': 101,                 # No punctuation mode
      }

**Dataset Configuration:**

The model requires training data with prompt annotations when using Lhotse datasets:

.. code-block:: yaml

  model:
    train_ds:
      use_lhotse: true
      initialize_prompt_feature: true
      prompt_field: "target_lang"     # Field name for prompt extraction
      prompt_dictionary: ${model.model_defaults.prompt_dictionary}
      num_prompts: ${model.model_defaults.num_prompts}
      
    validation_ds:
      use_lhotse: true
      initialize_prompt_feature: true
      prompt_field: "target_lang"
      prompt_dictionary: ${model.model_defaults.prompt_dictionary}
      num_prompts: ${model.model_defaults.num_prompts}

**Manifest Format:**

Training manifests should include prompt information:

.. code-block:: json

  {
    "audio_filepath": "/path/to/audio.wav",
    "text": "transcription text",
    "duration": 10.5,
    "target_lang": "en-US"
  }

**Example Configuration:**

A complete example configuration can be found at:
``<NeMo_git_root>/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe_prompt.yaml``

**Training Command:**

.. code-block:: bash

  python <NeMo_git_root>/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe_prompt.py \
      --config-path=<NeMo_git_root>/examples/asr/conf/fastconformer/hybrid_transducer_ctc/ \
      --config-name=fastconformer_hybrid_transducer_ctc_bpe_prompt.yaml \
      model.train_ds.manifest_filepath=<path_to_train_manifest> \
      model.validation_ds.manifest_filepath=<path_to_val_manifest> \
      model.tokenizer.dir=<path_to_tokenizer> \
      model.test_ds.manifest_filepath=<path_to_test_manifest>
