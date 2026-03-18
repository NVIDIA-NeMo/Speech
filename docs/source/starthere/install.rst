.. _installation:

Installation
============

This page covers how to install NVIDIA NeMo for speech AI tasks (ASR, TTS, speaker tasks, audio processing, and speech language models).

Prerequisites
-------------

Before installing NeMo, ensure you have:

#. **Python** 3.12 or above
#. **PyTorch** 2.7+
#. **NVIDIA GPU** (required for training; CPU-only inference is possible but slow)

Install from PyPI
-----------------

The quickest way to install NeMo is via pip. Install only the collections you need:

.. code-block:: bash

   # Install ASR and TTS (most common)
   pip install nemo_toolkit[asr,tts]

   # Install everything speech-related
   pip install nemo_toolkit[asr,tts,audio]

Available extras:

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Extra
     - What it includes
   * - ``asr``
     - Automatic Speech Recognition models, data loaders, and utilities
   * - ``tts``
     - Text-to-Speech models, vocoders, and audio codecs
   * - ``audio``
     - Audio processing models (enhancement, separation)

Install from Source
-------------------

For the latest development version or if you plan to contribute:

.. code-block:: bash

   git clone https://github.com/NVIDIA/NeMo.git
   cd NeMo
   pip install -e '.[test]'


Using Docker
------------

NVIDIA provides Docker containers with NeMo pre-installed. Check the `NeMo GitHub releases <https://github.com/NVIDIA/NeMo/releases>`_ for the latest container tags.

Verify Installation
-------------------

After installing, verify that NeMo is working:

.. code-block:: python

   import nemo.collections.asr as nemo_asr
   print("NeMo ASR installed successfully!")

   # Quick test: load a pretrained model
   model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
   print(f"Model loaded: {model.__class__.__name__}")

What's Next?
------------

- :doc:`ten_minutes` — A quick tour of NeMo's speech capabilities
- :doc:`key_concepts` — Understand the fundamentals of speech AI
- :doc:`choosing_a_model` — Find the right model for your use case

