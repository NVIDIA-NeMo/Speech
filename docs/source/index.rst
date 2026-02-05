NVIDIA NeMo Framework Developer Docs
====================================

NVIDIA NeMo Framework is an end-to-end, cloud-native framework designed to build, customize, and deploy generative AI models anywhere.

`NVIDIA NeMo Framework <https://github.com/NVIDIA/NeMo>`_ key capabilities include:

- Leaderboard-topping ASR models: Parakeet, Canary, FastConformer
- Production-ready TTS with MagpieTTS
- Speech Language Models (SpeechLM2): SALM, Duplex Speech-to-Speech
- Streaming speaker diarization with Sortformer
- HuggingFace Transformers integration for backbone LLMs
- GPU-accelerated ASR decoding algorithms
- Multi-GPU/multi-node training with mixed precision
- Comprehensive Speech AI tools: forced alignment, data exploration, CTC segmentation

`NVIDIA NeMo Framework <https://github.com/NVIDIA/NeMo>`_ has separate collections for:

* :doc:`Automatic Speech Recognition (ASR) <asr/intro>`

* :doc:`Text-to-Speech (TTS) <tts/intro>`

* :doc:`Audio Processing <audio/intro>`

* :doc:`SpeechLM2 <speechlm2/intro>`

Each collection consists of prebuilt modules that include everything needed to train on your data.
Every module can easily be customized, extended, and composed to create new generative AI
model architectures.

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
