.. _asr-configs-dataset-configuration:
.. _asr-configs-preprocessor-configuration:
.. _asr-configs-augmentation-configurations:

NeMo ASR Configuration Files
============================

This page covers ASR-specific configuration. For general NeMo setup (Experiment Manager, trainer), see :doc:`../core/core`.
Example configs: `examples/asr/conf <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/conf>`_.


Metric Configurations
---------------------

NeMo ASR supports WER and BLEU metrics via TorchMetrics.

.. code-block:: yaml

  model:
    use_cer: false
    log_prediction: true

For BLEU (translation): set ``bleu_tokenizer`` (``13a``, ``none``, ``intl``, ``char``, ``zh``, ``ja-mecab``, ``ko-mecab``, ``flores101``, ``flores200``).

For multitask models, use ``multitask_metrics_config`` with per-metric constraints:

.. code-block:: yaml

  model:
    multitask_metrics_config:
      metrics:
        wer:
          _target_: nemo.collections.asr.metrics.wer.WER
          constraint: ".task==transcribe"
        bleu:
          _target_: nemo.collections.asr.metrics.bleu.BLEU
          constraint: ".task==translate"


Tokenizer Configurations
------------------------

Models using sub-word encoding require a ``tokenizer`` section:

.. code-block:: yaml

  model:
    tokenizer:
      dir: "<path to tokenizer>"
      type: "bpe"  # or "wpe"

For multilingual models, use aggregate tokenizers (``type: "agg"``) with per-language sub-tokenizers.


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
