.. _choosing-a-model:

Choosing a Model
================

NeMo offers many pretrained speech models. This guide helps you pick the right one for your use case.

ASR: Which Model Should I Use?
------------------------------

.. list-table::
   :widths: 30 25 45
   :header-rows: 1

   * - I want to...
     - Recommended Model
     - Why
   * - Get the best accuracy on English
     - `Parakeet-TDT-0.6B V2 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2>`_
     - #1 on the `OpenASR Leaderboard <https://huggingface.co/spaces/hf-audio/open_asr_leaderboard>`_. TDT decoder provides accurate timestamps.
   * - Transcribe multiple languages
     - `Canary-1B V2 <https://huggingface.co/nvidia/canary-1b-v2>`_
     - Supports 25 EU languages + translation between them. AED decoder.
   * - Fast multilingual inference
     - `Canary-1B Flash <https://huggingface.co/nvidia/canary-1b-flash>`_
     - Optimized for speed while maintaining multilingual quality.
   * - Stream audio in real-time
     - Cache-aware Streaming FastConformer
     - Designed for low-latency streaming with caching to avoid recomputation.
   * - Minimize model size
     - `Canary-180M Flash <https://huggingface.co/nvidia/canary-180m-flash>`_
     - Smallest multilingual model. Good for edge deployment.
   * - Use CTC decoding (simpler pipeline)
     - `Parakeet-CTC-1.1B <https://huggingface.co/nvidia/parakeet-ctc-1.1b>`_
     - Non-autoregressive. Fast. Good with external language models.
   * - Integrate with an external LM
     - Any Parakeet model + NGPU-LM
     - GPU-accelerated n-gram LM fusion for CTC, RNNT, and TDT models.
   * - Transcribe multi-speaker meetings
     - `Multitalker Parakeet Streaming <https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1>`_
     - Handles overlapping speech in real-time with speaker-adapted decoding.

TTS: Which Model Should I Use?
------------------------------

.. list-table::
   :widths: 30 25 45
   :header-rows: 1

   * - I want to...
     - Recommended Model
     - Why
   * - Generate high-quality multilingual speech
     - `MagpieTTS <https://huggingface.co/nvidia/magpie_tts_multilingual_357m>`_
     - End-to-end LLM-based TTS. Supports voice cloning and multiple languages.
   * - Fast, controllable English synthesis
     - `FastPitch <https://huggingface.co/nvidia/tts_en_fastpitch>`_ + `HiFi-GAN <https://huggingface.co/nvidia/tts_hifigan>`_
     - Cascaded pipeline with pitch/duration control. Well-tested.
   * - Generate discrete audio tokens
     - Audio Codec
     - Neural audio codec for tokenizing audio. Used by MagpieTTS internally.

Speaker Tasks: Which Model Should I Use?
-----------------------------------------

.. list-table::
   :widths: 30 25 45
   :header-rows: 1

   * - I want to...
     - Recommended Model
     - Why
   * - Determine who spoke when
     - `Sortformer <https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1>`_
     - End-to-end streaming diarization. Supports up to 4 speakers.
   * - Verify/identify a speaker
     - `TitaNet <https://huggingface.co/nvidia/speakerverification_en_titanet_large>`_
     - Extracts speaker embeddings for verification and identification.
   * - Detect voice activity
     - `MarbleNet <https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0>`_
     - Frame-level VAD. Multilingual. Works as a preprocessing step.

Speech Language Models: Which Model Should I Use?
-------------------------------------------------

.. list-table::
   :widths: 30 25 45
   :header-rows: 1

   * - I want to...
     - Recommended Model
     - Why
   * - Ask questions about audio content
     - `Canary-Qwen 2.5B <https://huggingface.co/nvidia/canary-qwen-2.5b>`_ (SALM)
     - LLM augmented with speech understanding. Can transcribe, translate, and answer questions about audio.
   * - Build a speech-to-speech system
     - DuplexS2SModel
     - Full-duplex model that both understands and generates speech.


Decision Flowchart
------------------

.. code-block:: text

   What do you want to do?
   │
   ├─ Transcribe speech to text (ASR)
   │  ├─ English only? → Parakeet-TDT-0.6B V2
   │  ├─ Multiple languages? → Canary-1B V2
   │  ├─ Need streaming? → Cache-aware Streaming FastConformer
   │  └─ Multi-speaker meeting? → Multitalker Parakeet Streaming
   │
   ├─ Generate speech from text (TTS)
   │  ├─ Multilingual / voice cloning? → MagpieTTS
   │  └─ English with pitch control? → FastPitch + HiFi-GAN
   │
   ├─ Identify speakers
   │  ├─ Who spoke when? → Sortformer
   │  └─ Verify identity? → TitaNet
   │
   ├─ Enhance audio quality → See Audio Processing models
   │
   └─ Speech-aware LLM → Canary-Qwen 2.5B (SALM)


Where to Find Models
--------------------

All pretrained NeMo models are available on:

- `HuggingFace Hub (nvidia) <https://huggingface.co/nvidia>`_ — search for "nemo" or specific model names
- `NGC Model Catalog <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_ — NVIDIA's model registry

See :doc:`../checkpoints/intro` for instructions on loading pretrained models.

