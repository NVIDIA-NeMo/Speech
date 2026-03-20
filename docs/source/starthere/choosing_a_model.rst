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
     - `Canary-Qwen 2.5B <https://huggingface.co/nvidia/canary-qwen-2.5b>`_
     - State-of-the-art English ASR. For very fast offline alternatives with almost SOTA accuracy, use `Parakeet-TDT V2 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2>`_ or `Parakeet-TDT V3 <https://huggingface.co/nvidia/parakeet-tdt-1.1b>`_.
   * - Transcribe multiple languages
     - `Canary-1B V2 <https://huggingface.co/nvidia/canary-1b-v2>`_
     - Supports 25 EU languages + translation between them. AED decoder.
   * - Transcribe European languages (ASR only, auto language detection)
     - `Parakeet-TDT 0.6B V3 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`_
     - 25 European languages in one model; automatic language detection; punctuation, capitalization, and word/segment timestamps; long-form and streaming options. No speech-to-text translation—use Canary-1B V2 if you need translation.
   * - Stream audio in real-time
     - `Nemotron-Speech-Streaming <https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b>`_
     - Low-latency streaming English ASR with configurable chunk sizes. Cache-aware FastConformer + RNN-T.
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
     - `Streaming Sortformer <https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1>`_, `Offline Sortformer <https://huggingface.co/nvidia/diar_sortformer_4spk-v1>`_
     - End-to-end diarization for up to 4 speakers. Use streaming for real-time; use offline for batch.
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
   │  ├─ Best accuracy on English? → Canary-Qwen 2.5B (or Parakeet-TDT V2/V3 for fast offline)
   │  ├─ Multiple languages + translation? → Canary-1B V2
   │  ├─ European multilingual ASR (auto LID)? → Parakeet-TDT 0.6B V3
   │  ├─ Stream audio in real-time? → Nemotron-Speech-Streaming
   │  └─ Multi-speaker meeting? → Multitalker Parakeet Streaming
   │
   ├─ Generate speech from text (TTS)
   │  ├─ Multilingual / voice cloning? → MagpieTTS
   │  └─ English with pitch control? → FastPitch + HiFi-GAN
   │
   ├─ Identify speakers
   │  ├─ Who spoke when? → Streaming Sortformer or Offline Sortformer
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

