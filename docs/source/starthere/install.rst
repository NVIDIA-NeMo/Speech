.. _installation:

Installation
============

This page covers how to install NVIDIA NeMo for speech AI tasks (ASR, TTS, speaker tasks, audio processing, and speech language models).

Prerequisites
-------------

Before installing NeMo, ensure you have:

#. **Python** 3.12 or above
#. **PyTorch** 2.7+ (install **before** NeMo so CUDA wheels match your GPU driver)
#. **NVIDIA GPU** (required for training; CPU-only inference is possible but slow)

Recommended installation order
------------------------------

Install dependencies in this order when setting up a **local GPU** environment:

#. Create and activate a Python environment.
#. Install a **CUDA toolkit** (or rely on a driver + PyTorch bundle that matches your CUDA major version).
#. Install **PyTorch** (and torchvision if you need it) from the index that matches your CUDA build.
#. Install **NeMo** (from PyPI or editable source) **with the extras** for the collections you need (``asr``, ``tts``, etc.).

Putting PyTorch in place first avoids mismatched CUDA runtimes and makes NeMo’s optional GPU-dependent packages resolve correctly.

**Example (conda + pip, CUDA 13.0 PyTorch wheels):**

.. code-block:: bash

   # 1) New environment (adjust Python version if your platform requires it)
   conda create -n nemo python=3.12 -y
   conda activate nemo

   # 2) CUDA toolkit from conda (optional if you already have a compatible toolkit via the driver)
   conda install nvidia::cuda-toolkit

   # 3) PyTorch built for CUDA 13.x — change cu130 / URL if you use cu124 or CPU-only
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

   # 4) NeMo: use extras for ASR/TTS/etc. For a clone of the repo, use editable install (see below)
   pip install nemo_toolkit[asr,tts]

Adjust the PyTorch ``--index-url`` (e.g. ``cu124``, ``cu121``, or CPU) to match `PyTorch’s install matrix <https://pytorch.org/get-started/locally/>`_ and your NVIDIA driver.

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

.. _install-from-source:

Install from Source
-------------------

For the latest development version or if you plan to contribute, clone the repository and install in editable mode.

The ``test`` extra pulls in **pytest and tooling for the test suite**. It does **not** install NeMo collection dependencies (ASR, TTS, audio, etc.). Add those extras explicitly or imports like ``nemo.collections.asr`` will fail.

.. code-block:: bash

   git clone https://github.com/NVIDIA/NeMo.git
   cd NeMo

   # After PyTorch is installed (see Recommended installation order above):
   # Collections you need for development (required for nemo.collections.* imports)
   pip install -e '.[asr,tts]'

   # Optional: add test to run pytest with NeMo’s dev test dependencies
   # pip install -e '.[asr,tts,test]'

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
