NVIDIA NeMo Toolkit Developer Docs
===================================

`NVIDIA NeMo Toolkit <https://github.com/NVIDIA/NeMo>`_ is an open-source toolkit for speech, audio, and multimodal language model research, with a clear path from experimentation to production deployment.

Models
------

- **ASR:** `Parakeet <https://huggingface.co/collections/nvidia/parakeet>`_, `Canary <https://huggingface.co/collections/nvidia/canary>`_, FastConformer -- with CTC, Transducer, TDT, and hybrid decoders
- **TTS:** `MagpieTTS <https://huggingface.co/nvidia/magpie_tts_multilingual_357m>`_, `FastPitch <https://huggingface.co/nvidia/tts_en_fastpitch>`_ + `HiFi-GAN <https://huggingface.co/nvidia/tts_hifigan>`_ -- multi-language, multi-speaker
- **Speaker:** `Sortformer <https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1>`_ streaming diarization, `TitaNet <https://huggingface.co/nvidia/speakerverification_en_titanet_large>`_ speaker recognition, `MarbleNet <https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0>`_ VAD
- **Audio:** `Speech enhancement <https://huggingface.co/nvidia/sr_ssl_flowmatching_16k_430m>`_, source separation, neural audio codecs
- **SpeechLM2:** `Canary-Qwen 2.5B <https://huggingface.co/nvidia/canary-qwen-2.5b>`_ (SALM), Duplex Speech-to-Speech -- HuggingFace Transformers backbone integration

Inference & Deployment
----------------------

- Streaming and real-time ASR with cache-aware Conformer
- GPU-accelerated decoding with `NGPU-LM <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_customization/ngpulm_language_modeling_and_customization.html>`_ language model fusion
- Export to ONNX

Voice Agent
-----------

- Open-source conversational agent framework built on `Pipecat <https://github.com/pipecat-ai/pipecat>`_
- Streaming STT + LLM + TTS pipeline with natural turn-taking
- Live speaker diarization and tool calling support

----

NeMo is built for researchers and engineers. Each collection provides prebuilt, modular components
that can be customized, extended, and composed -- from rapid prototyping to multi-node training
to production inference.

`NVIDIA NeMo Toolkit <https://github.com/NVIDIA/NeMo>`_ has separate collections for:

* :doc:`Automatic Speech Recognition (ASR) <asr/intro>`

* :doc:`Text-to-Speech (TTS) <tts/intro>`

* :doc:`Audio Processing <audio/intro>`

* :doc:`SpeechLM2 <speechlm2/intro>`

For quick guides and tutorials, see the "Getting started" section below.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: starthere
   :titlesonly:

   starthere/intro
   starthere/fundamentals
   starthere/best-practices
   starthere/tutorials

For more information, browse the developer docs for your area of interest in the contents section below or on the left sidebar.


.. toctree::
   :maxdepth: 1
   :caption: Training
   :name: Training

   features/parallelisms
   features/mixed_precision

.. toctree::
   :maxdepth: 1
   :caption: Model Checkpoints
   :name: Checkpoints

   checkpoints/intro

.. toctree::
   :maxdepth: 1
   :caption: APIs
   :name: APIs
   :titlesonly:

   apis

.. toctree::
   :maxdepth: 1
   :caption: Collections
   :name: Collections
   :titlesonly:

   collections

.. toctree::
   :maxdepth: 1
   :caption: Speech AI Tools
   :name: Speech AI Tools
   :titlesonly:

   tools/intro
