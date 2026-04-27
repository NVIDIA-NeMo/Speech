.. _asr-checkpoints-list:

=======================
ASR Model Checkpoints
=======================

This page lists all supported ASR model checkpoints released by NVIDIA NeMo.
Benchmark scores for each model can be found on its `HuggingFace model card <https://huggingface.co/nvidia>`__.

Glossary
--------

.. list-table::
   :header-rows: 1

   * - Term
     - Definition
   * - **ASR**
     - Automatic Speech Recognition — transcribing speech to text
   * - **AST**
     - Automatic Speech Translation — translating speech to text from one language to another
   * - **AED**
     - Attention Encoder-Decoder — autoregressive decoder using cross-attention (Canary family)
   * - **CTC**
     - Connectionist Temporal Classification — non-autoregressive decoder
   * - **RNN-T**
     - Recurrent Neural Network Transducer — autoregressive streaming-friendly decoder
   * - **TDT**
     - Token-and-Duration Transducer — extends RNN-T with duration prediction for faster inference
   * - **Hybrid**
     - Joint RNN-T + CTC model — both decoders trained together, either usable at inference
   * - **PnC**
     - Punctuation and Capitalization in the output
   * - **SALM**
     - Speech Augmented Language Model — combines a speech encoder with a large language model
   * - **Streaming**
     - Real-time / cache-aware inference capability
   * - **EU4**
     - English, German, Spanish, French
   * - **EU25**
     - English, German, Spanish, French, Italian, Polish, Portuguese, Dutch, Russian, Ukrainian, Belarusian, Croatian, Czech, Bulgarian, Danish, Estonian, Finnish, Greek, Hungarian, Latvian, Lithuanian, Maltese, Romanian, Slovak, Slovenian, Swedish

Canary Models (AED)
-------------------

Multi-task encoder-decoder models supporting ASR, AST, PnC, and timestamps across multiple languages.

.. list-table::
   :header-rows: 1

   * - Model
     - Decoder
     - Capabilities
     - Language
     - Size
   * - `canary-1b-v2 <https://huggingface.co/nvidia/canary-1b-v2>`__
     - AED
     - ASR, AST, PnC, timestamps
     - EU25
     - 1B
   * - `canary-qwen-2.5b <https://huggingface.co/nvidia/canary-qwen-2.5b>`__
     - SALM
     - ASR, AST, PnC, timestamps
     - EU25
     - 2.5B
   * - `canary-1b-flash <https://huggingface.co/nvidia/canary-1b-flash>`__
     - AED
     - ASR, AST, PnC, timestamps, fast
     - EU4
     - 1B
   * - `canary-180m-flash <https://huggingface.co/nvidia/canary-180m-flash>`__
     - AED
     - ASR, AST, PnC, timestamps, fast
     - EU4
     - 180M
   * - `canary-1b <https://huggingface.co/nvidia/canary-1b>`__
     - AED
     - ASR, AST, PnC
     - EU4
     - 1B


Parakeet Models
-----------------

High-accuracy ASR models built on the FastConformer encoder architecture.
Parakeet, Nemotron Speech, and the ``stt_*_fastconformer_*`` models below all share the same underlying FastConformer encoder;
the different names reflect release branding, not architectural differences.

.. list-table::
   :header-rows: 1

   * - Model
     - Decoder
     - Capabilities
     - Language
     - Size
   * - `parakeet-tdt-0.6b-v3 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`__
     - TDT
     - ASR, PnC, timestamps
     - English
     - 0.6B
   * - `parakeet-tdt-0.6b-v2 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2>`__
     - TDT
     - ASR, PnC, timestamps
     - English
     - 0.6B
   * - `parakeet-tdt-1.1b <https://huggingface.co/nvidia/parakeet-tdt-1.1b>`__
     - TDT
     - ASR, timestamps
     - English
     - 1.1B
   * - `parakeet-tdt_ctc-1.1b <https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b>`__
     - Hybrid TDT+CTC
     - ASR, timestamps
     - English
     - 1.1B
   * - `parakeet-tdt_ctc-0.6b-ja <https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja>`__
     - Hybrid TDT+CTC
     - ASR, timestamps
     - Japanese
     - 0.6B
   * - `parakeet-tdt_ctc-110m <https://huggingface.co/nvidia/parakeet-tdt_ctc-110m>`__
     - Hybrid TDT+CTC
     - ASR, timestamps
     - English
     - 110M
   * - `parakeet-rnnt-1.1b <https://huggingface.co/nvidia/parakeet-rnnt-1.1b>`__
     - RNN-T
     - ASR, timestamps
     - English
     - 1.1B
   * - `parakeet-rnnt-0.6b <https://huggingface.co/nvidia/parakeet-rnnt-0.6b>`__
     - RNN-T
     - ASR, timestamps
     - English
     - 0.6B
   * - `parakeet-ctc-1.1b <https://huggingface.co/nvidia/parakeet-ctc-1.1b>`__
     - CTC
     - ASR
     - English
     - 1.1B
   * - `parakeet-ctc-0.6b <https://huggingface.co/nvidia/parakeet-ctc-0.6b>`__
     - CTC
     - ASR
     - English
     - 0.6B
   * - `parakeet-ctc-0.6b-Vietnamese <https://huggingface.co/nvidia/parakeet-ctc-0.6b-Vietnamese>`__
     - CTC
     - ASR
     - Vietnamese
     - 0.6B
   * - `parakeet-rnnt-110m-da-dk <https://huggingface.co/nvidia/parakeet-rnnt-110m-da-dk>`__
     - RNN-T
     - ASR
     - Danish
     - 110M


Streaming Models
-----------------

Cache-aware models for real-time / low-latency inference.

.. list-table::
   :header-rows: 1

   * - Model
     - Decoder
     - Capabilities
     - Language
     - Size
   * - `nemotron-speech-streaming-en-0.6b <https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b>`__
     - Hybrid
     - ASR, streaming
     - English
     - 0.6B
   * - `multitalker-parakeet-streaming-0.6b-v1 <https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1>`__
     - RNN-T
     - ASR, multitalker, streaming
     - English
     - 0.6B
   * - `parakeet_realtime_eou_120m-v1 <https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1>`__
     - RNN-T
     - ASR, end-of-utterance, streaming
     - English
     - 120M
   * - `stt_en_fastconformer_hybrid_large_streaming_multi <https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi>`__
     - Hybrid
     - ASR, streaming, multiple look-aheads
     - English
     - Large
   * - `stt_en_fastconformer_hybrid_medium_streaming_80ms_pc <https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms_pc>`__
     - Hybrid
     - ASR, PnC, streaming
     - English
     - Medium
   * - `stt_en_fastconformer_hybrid_medium_streaming_80ms <https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms>`__
     - Hybrid
     - ASR, streaming
     - English
     - Medium
   * - `stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc <https://huggingface.co/nvidia/stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc>`__
     - Hybrid
     - ASR, PnC, streaming
     - Georgian
     - Large
   * - `stt_en_fastconformer_hybrid_large_streaming_1040ms <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_1040ms>`__
     - Hybrid
     - ASR, streaming
     - English
     - Large


FastConformer English Models (Non-Streaming)
----------------------------------------------

.. list-table::
   :header-rows: 1

   * - Model
     - Decoder
     - Capabilities
     - Language
     - Size
   * - `stt_en_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - English
     - Large
   * - `stt_en_fastconformer_ctc_large <https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large>`__
     - CTC
     - ASR
     - English
     - Large
   * - `stt_en_fastconformer_ctc_xlarge <https://huggingface.co/nvidia/stt_en_fastconformer_ctc_xlarge>`__
     - CTC
     - ASR
     - English
     - XLarge
   * - `stt_en_fastconformer_ctc_xxlarge <https://huggingface.co/nvidia/stt_en_fastconformer_ctc_xxlarge>`__
     - CTC
     - ASR
     - English
     - XXLarge
   * - `stt_en_fastconformer_transducer_large <https://huggingface.co/nvidia/stt_en_fastconformer_transducer_large>`__
     - RNN-T
     - ASR
     - English
     - Large
   * - `stt_en_fastconformer_transducer_xlarge <https://huggingface.co/nvidia/stt_en_fastconformer_transducer_xlarge>`__
     - RNN-T
     - ASR
     - English
     - XLarge
   * - `stt_en_fastconformer_transducer_xxlarge <https://huggingface.co/nvidia/stt_en_fastconformer_transducer_xxlarge>`__
     - RNN-T
     - ASR
     - English
     - XXLarge
   * - `stt_en_fastconformer_tdt_large <https://huggingface.co/nvidia/stt_en_fastconformer_tdt_large>`__
     - TDT
     - ASR
     - English
     - Large


FastConformer Multilingual Models
----------------------------------

Single-language FastConformer Hybrid models. Models with ``_pc`` suffix support punctuation and capitalization.

.. list-table::
   :header-rows: 1

   * - Model
     - Decoder
     - Capabilities
     - Language
     - Size
   * - `stt_multilingual_fastconformer_hybrid_large_pc_blend_eu <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_multilingual_fastconformer_hybrid_large_pc_blend_eu>`__
     - Hybrid
     - ASR, PnC
     - Multilingual EU
     - Large
   * - `stt_de_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_de_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - German
     - Large
   * - `stt_es_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_es_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Spanish
     - Large
   * - `stt_es_fastconformer_hybrid_large_pc_nc <https://huggingface.co/nvidia/stt_es_fastconformer_hybrid_large_pc_nc>`__
     - Hybrid
     - ASR, Punctuation only
     - Spanish
     - Large
   * - `stt_fr_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - French
     - Large
   * - `stt_it_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_it_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Italian
     - Large
   * - `stt_ru_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Russian
     - Large
   * - `stt_ua_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_ua_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Ukrainian
     - Large
   * - `stt_pl_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_pl_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Polish
     - Large
   * - `stt_hr_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_hr_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Croatian
     - Large
   * - `stt_be_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_be_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Belarusian
     - Large
   * - `stt_nl_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_nl_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Dutch
     - Large
   * - `stt_pt_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_pt_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Portuguese
     - Large
   * - `stt_fa_fastconformer_hybrid_large <https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large>`__
     - Hybrid
     - ASR
     - Farsi
     - Large
   * - `stt_ka_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_ka_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Georgian
     - Large
   * - `stt_hy_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Armenian
     - Large
   * - `stt_ar_fastconformer_hybrid_large_pc_v1.0 <https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0>`__
     - Hybrid
     - ASR, PnC
     - Arabic
     - Large
   * - `stt_ar_fastconformer_hybrid_large_pcd_v1.0 <https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0>`__
     - Hybrid
     - ASR, PnC (diacritized)
     - Arabic
     - Large
   * - `stt_uz_fastconformer_hybrid_large_pc <https://huggingface.co/nvidia/stt_uz_fastconformer_hybrid_large_pc>`__
     - Hybrid
     - ASR, PnC
     - Uzbek
     - Large
   * - `stt_kk_ru_fastconformer_hybrid_large <https://huggingface.co/nvidia/stt_kk_ru_fastconformer_hybrid_large>`__
     - Hybrid
     - ASR
     - Kazakh + Russian
     - Large


Loading Models
--------------

All models (except SALM — see :doc:`SpeechLM2 </speechlm2/intro>`) can be loaded via the ``from_pretrained()`` API:

.. code-block:: python

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")


