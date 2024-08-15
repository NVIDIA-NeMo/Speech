NVIDIA NeMo Framework Developer Docs
====================================

NVIDIA NeMo Framework is an end-to-end, cloud-native framework designed to build, customize, and deploy generative AI models anywhere.

`NVIDIA NeMo Framework <https://github.com/NVIDIA/NeMo>`_ supports large-scale training features, including:

- Mixed Precision Training
- Parallelism
- Distributed Optimizer
- Fully Sharded Data Parallel (FSDP)
- Flash Attention
- Activation Recomputation
- Positional Embeddings and Positional Interpolation
- Post-Training Quantization (PTQ) and Quantization Aware Training (QAT) with `TensorRT Model Optimizer <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_
- Knowledge Distillation-based training with `TensorRT Model Optimizer <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_
- Sequence Packing

`NVIDIA NeMo Framework <https://github.com/NVIDIA/NeMo>`_ has separate collections for:

* :doc:`Large Language Models (LLMs) <nlp/nemo_megatron/intro>`

* :doc:`Automatic Speech Recognition (ASR) <asr/intro>`

* :doc:`Multimodal Models (MMs) <multimodal/mllm/intro>`

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
   starthere/fundamentals
   starthere/tutorials

For more information, browse the developer docs for your area of interest in the contents section below or on the left sidebar.


.. toctree::
   :maxdepth: 1
   :caption: Key Optimizations
   :name: Key Optimizations

   features/mixed_precision
   features/parallelisms
   features/moe
   features/optimizations/index

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
