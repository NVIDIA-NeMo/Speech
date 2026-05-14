Featured Models
===============

NeMo's ASR collection supports several model architectures. This page covers the key model families and their capabilities.
For pretrained checkpoints, see :doc:`All Checkpoints <./asr_checkpoints>`.
For config file details, see :doc:`Configuration Files <./configs>`.

Parakeet
~~~~~~~~

Parakeet is a family of ASR models with a :ref:`FastConformer Encoder <Fast-Conformer>` and CTC, RNN-T, or TDT decoders.

* `Parakeet-TDT-0.6B V3 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`__ — 25 languages, PnC, blazing fast
* `Parakeet-TDT-0.6B V2 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2>`__ — English-only, PnC, blazing fast
* `Parakeet-TDT/CTC-110M <https://huggingface.co/nvidia/parakeet-tdt_ctc-110m>`__ — Edge deployment
* `Nemotron-Speech-Streaming <https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b>`__ — Real-time streaming
* `Multitalker-Parakeet <https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1>`__ — Multi-speaker streaming


Canary
~~~~~~

Canary models are encoder-decoder models with a :ref:`FastConformer Encoder <Fast-Conformer>` and Transformer Decoder :cite:`asr-models-vaswani2017aayn`.
They support ASR in 25 EU languages, speech translation (AST), and punctuation/capitalization (PnC).

* `Canary-1B V2 <https://huggingface.co/nvidia/canary-1b-v2>`__ — Flagship: 25 languages, PnC, timestamps
* `Canary-Qwen-2.5B <https://huggingface.co/nvidia/canary-qwen-2.5b>`__ — English only, PnC, highest accuracy
* `Canary-1B Flash <https://huggingface.co/nvidia/canary-1b-flash>`__ / `180M Flash <https://huggingface.co/nvidia/canary-180m-flash>`__ — Optimized for speed

Canary supports chunked and `streaming inference <https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/Streaming_ASR_Pipelines.ipynb>`__.


.. _Conformer_model:

Conformer
---------

The Conformer :cite:`asr-models-gulati2020conformer` combines self-attention and convolution modules. NeMo supports CTC, Transducer, and HAT variants.

* **Conformer-CTC**: Non-autoregressive, uses :class:`~nemo.collections.asr.models.EncDecCTCModelBPE`
* **Conformer-Transducer**: Autoregressive, uses :class:`~nemo.collections.asr.models.EncDecRNNTBPEModel`
* **Conformer-HAT**: Separates labels and blank predictions for better external LM integration (`paper <https://arxiv.org/abs/2003.07705>`_)

.. _Conformer-CTC_model:
.. _Conformer-Transducer_model:
.. _Conformer-HAT_model:

Configs: ``examples/asr/conf/conformer/``

.. _Fast-Conformer:

Fast-Conformer
--------------

Fast Conformer has 8x depthwise convolutional subsampling and reduced kernel sizes, making it ~2.4x faster than standard Conformer with minimal quality loss.
Supports Longformer-style local attention for audio >1 hour.

Configs: ``examples/asr/conf/fastconformer/``

.. _cache-aware streaming conformer:

Cache-aware Streaming Conformer
-------------------------------

Streaming models trained with limited right context for real-time inference with caching to avoid duplicate computation. Supports three modes: fully causal, regular look-ahead, and chunk-aware look-ahead (recommended).

* `Tutorial notebook <https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb>`_
* Simulation script: ``examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py``
* Supports multiple look-aheads with ``att_context_size`` lists

Configs: ``examples/asr/conf/fastconformer/cache_aware_streaming/``


Multitalker Streaming
---------------------

Streaming multi-speaker ASR based on cache-aware FastConformer with speaker kernel injection :cite:`asr-models-wang25y_interspeech`. Deploys one model instance per speaker for robust transcription of overlapped speech.

* `Model card <https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1>`__
* `Tutorial <https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Streaming_Multitalker_ASR.ipynb>`_

.. _Hybrid-Transducer_CTC_model:

Hybrid-Transducer-CTC
----------------------

Models with both RNN-T and CTC decoders trained jointly. Switch at inference time via ``asr_model.change_decoding_strategy(decoder_type='ctc' or 'rnnt')``.

* :class:`~nemo.collections.asr.models.EncDecHybridRNNTCTCBPEModel` (BPE) / :class:`~nemo.collections.asr.models.EncDecHybridRNNTCTCModel` (char)
* Configs: ``examples/asr/conf/fastconformer/hybrid_transducer_ctc/``

.. _Hybrid-Transducer-CTC-Prompt_model:

**With Prompt Conditioning:** Extends Hybrid models with learnable prompt embeddings for multilingual/multi-domain ASR via :class:`~nemo.collections.asr.models.EncDecHybridRNNTCTCBPEModelWithPrompt`. Config: ``fastconformer_hybrid_transducer_ctc_bpe_prompt.yaml``


References
----------

.. bibliography:: asr_all.bib
    :style: plain
    :labelprefix: ASR-MODELS
    :keyprefix: asr-models-
