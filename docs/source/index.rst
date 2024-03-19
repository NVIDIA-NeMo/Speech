NVIDIA NeMo Framework Developer Docs
====================================

NVIDIA NeMo Framework is an end-to-end, cloud-native framework to build, customize, and deploy generative AI models anywhere.

`NVIDIA NeMo Framework <https://github.com/NVIDIA/NeMo>`_ has separate collections for:

* :doc:`Large Language Models (LLMs) <nlp/nemo_megatron/intro>`

* :doc:`Automatic Speech Recognition (ASR) <asr/intro>`

* :doc:`Multimodal (MM) Models <multimodal/mllm/intro>`

* :doc:`Text-to-Speech (TTS) <tts/intro>`

* :doc:`Computer Vision (CV)  <vision/intro>`

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
   starthere/tutorials
   starthere/best-practices

For more information, browse the developer docs for your area of interest in the contents section below or on the left sidebar.

.. toctree::
   :maxdepth: 1
   :caption: NeMo Core
   :name: core
   :titlesonly:

   core/core_index

.. toctree::
   :maxdepth: 2
   :caption: Community Model Converters
   :name: CheckpointConverters

   ckpt_converters/user_guide
   ckpt_converters/dev_guide

.. toctree::
   :maxdepth: 1
   :caption: Large Language Models (LLMs)
   :name: Large Language Models
   :titlesonly:

   nlp/nemo_megatron/intro
   nlp/models
   nlp/machine_translation/machine_translation
   nlp/megatron_onnx_export
   nlp/api


.. toctree::
   :maxdepth: 1
   :caption: Speech AI
   :name: Speech AI
   :titlesonly:

   asr/intro
   asr/speech_classification/intro
   asr/speaker_recognition/intro
   asr/speaker_diarization/intro
   asr/ssl/intro
   asr/speech_intent_slot/intro


.. toctree::
   :maxdepth: 1
   :caption: Multimodal (MM)
   :name: Multimodal
   :titlesonly:

   multimodal/mllm/intro
   multimodal/vlm/intro
   multimodal/text2img/intro
   multimodal/nerf/intro
   multimodal/api


.. toctree::
   :maxdepth: 1
   :caption: Text To Speech (TTS)
   :name: Text To Speech
   :titlesonly:

   tts/intro

.. toctree::
   :maxdepth: 2
   :caption: Vision (CV)
   :name: vision
   :titlesonly:

   vision/intro

.. toctree::
   :maxdepth: 2
   :caption: Common
   :name: Common
   :titlesonly:

   common/intro


.. toctree::
   :maxdepth: 2
   :caption: Speech Tools
   :name: Speech Tools
   :titlesonly:

   tools/intro

.. toctree::
   :maxdepth: 2
   :caption: Upgrade Guide
   :name: Upgrade Guide
   :titlesonly:

   starthere/migration-guide