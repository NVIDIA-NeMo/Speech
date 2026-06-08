.. _installation:

Installation
============

This page covers how to install NVIDIA NeMo for speech AI tasks (ASR, TTS, speaker tasks, audio processing, and speech language models).

Prerequisites
-------------

NeMo Speech works with the **Python, PyTorch, and CUDA versions of your choosing**:

#. **Python** 3.10 or above
#. **PyTorch** 2.6 or above
#. **NVIDIA GPU + CUDA** (required for training; CPU-only inference is possible but slow)

.. admonition:: Bring your own Python / PyTorch / CUDA
   :class: important

   The recommended install path is uv (below), which gives you our actively-tested stack. But NeMo Speech can also install *on top of* an existing environment: the ``nemo-toolkit`` package only requires ``torch>=2.6``, so if you already have a Python, PyTorch, and CUDA stack, your pre-installed PyTorch is **kept, not replaced** (see :ref:`the pip fallback <install-from-pypi>`).

   The versions pinned in ``uv.lock`` and shipped in the official container — **Python 3.13, PyTorch 2.12, CUDA 12.6/13.2** — are simply the combination we actively test and support. They make setup turnkey and reproducible, but they are **not** a hard requirement.

.. note::

   As of `PyTorch 2.6 <https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true>`_, ``torch.load`` defaults to ``weights_only=True``. Some checkpoints require ``weights_only=False``; in that case set ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` before loading, and only with trusted files (loading untrusted files with full pickle support risks arbitrary code execution).

.. _install-from-source:

Install from Source with uv (recommended)
------------------------------------------

The recommended way to install NeMo Speech is from source with `uv <https://docs.astral.sh/uv/>`_, which reproduces our actively-tested stack from the committed ``uv.lock``:

.. code-block:: bash

   git clone https://github.com/NVIDIA-NeMo/NeMo.git
   cd NeMo

   # CUDA 13.x (recommended). Use --extra cu12 for CUDA 12.x. uv resolves the
   # matching PyTorch CUDA wheel automatically from the pinned indexes.
   uv sync --extra all --extra cu13

   # Optional: add the test suite tooling, or the docs build dependencies
   # uv sync --extra all --extra cu13 --group test
   # uv sync --group docs

``uv sync`` creates a virtual environment in ``.venv/`` with NeMo installed in editable mode, matching our supported stack (Python 3.13, PyTorch 2.12, CUDA 13.2 by default). Run commands with ``uv run <cmd>`` or activate the environment with ``source .venv/bin/activate``.

On Linux, pass exactly one of ``--extra cu13`` (recommended) or ``--extra cu12`` — they are mutually exclusive. If you omit both, uv installs the generic PyPI PyTorch wheel instead of NVIDIA's CUDA-matched build.

Available collection extras (combine with one CUDA extra above):

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - Extra
     - What it includes
   * - ``asr``
     - Automatic Speech Recognition models, data loaders, and utilities
   * - ``tts``
     - Text-to-Speech models, vocoders, and audio codecs
   * - ``audio``
     - Audio processing models (enhancement, separation)
   * - ``speechlm2``
     - Speech language models (includes NeMo Automodel)
   * - ``all``
     - All of the collections above
   * - ``cu12`` / ``cu13``
     - Our pinned CUDA 12.x / 13.x PyTorch build (Linux; pick at most one)

.. note::

   ``test`` and ``docs`` are dependency *groups* (PEP 735), not extras. Install them with ``--group`` (e.g. ``uv sync --group test``) — the bracket form ``.[test]`` does not work.

.. _install-compiled-extras:

Optional compiled dependencies for SpeechLM2 / Automodel (``compiled`` / ``compiled-a100``)
-------------------------------------------------------------------------------------------

The Automodel backend used for SpeechLM2 **does not require any compiled dependencies — it runs without them.** The ``compiled`` and ``compiled-a100`` extras are an *optional* performance add-on: when their source-built GPU kernels are installed, Automodel can route to dedicated accelerated backends (FP8 Transformer kernels via Transformer Engine, FlashAttention, Mamba/state-space layers, and Mixture-of-Experts ops). They contain:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Package
     - Purpose
   * - ``transformer-engine``
     - NVIDIA Transformer Engine — FP8 and accelerated Transformer kernels
   * - ``flash-attn``
     - FlashAttention attention kernels
   * - ``mamba-ssm`` + ``causal-conv1d``
     - Mamba / state-space-model kernels (hybrid Mamba architectures)
   * - ``nv-grouped-gemm``
     - Grouped GEMM kernels for Mixture-of-Experts (MoE) layers
   * - ``deep_ep`` (DeepEP)
     - Expert-parallel communication kernels for MoE (``compiled`` only — see below)
   * - ``onnx-ir`` + ``onnxscript``
     - Pinned ONNX export tooling

Choose the variant that matches your GPU (the two are mutually exclusive):

* ``compiled`` — Hopper/Blackwell and newer (SM90/SM100/SM120, e.g. H100/H200/B200). Includes DeepEP.
* ``compiled-a100`` — Ampere A100 (SM80). Omits DeepEP, which requires a separately-built, patched version on A100.

.. warning::

   These packages **build from source** and need a full CUDA build environment — build tools, matching ``TORCH_CUDA_ARCH_LIST`` / ``NVTE_CUDA_ARCHS`` flags, ``--no-build-isolation``, and (for ``compiled``) extra manual build steps that the Dockerfile performs (e.g. flash-attn-4 and DeepEP patches). The supported, reproducible way to get them is the container build, which sets all of this up for you:

   .. code-block:: bash

      # Hopper/Blackwell (default GPU_TARGET=h100plus → compiled)
      docker buildx build -f docker/Dockerfile -t nemo-speech .

      # Ampere A100 (GPU_TARGET=a100 → compiled-a100)
      docker buildx build -f docker/Dockerfile \
        --build-arg BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04 \
        --build-arg GPU_TARGET=a100 -t nemo-speech .

   A bare ``uv sync --extra all --extra cu13 --extra compiled`` outside this environment will likely fail to compile.

Using Docker (turnkey, our supported stack)
--------------------------------------------

.. note::

   **NGC container:** *Coming soon — the pull command for the prebuilt NeMo Speech container image will be published here.*

To build the container from source, use the provided ``docker/Dockerfile`` (CUDA 13 / H100+ by default):

.. code-block:: bash

   git clone https://github.com/NVIDIA-NeMo/NeMo.git
   cd NeMo
   docker buildx build -f docker/Dockerfile -t nemo-speech .

See the header of ``docker/Dockerfile`` for CUDA 12 / A100 build arguments (``BASE_IMAGE``, ``GPU_TARGET``).

.. _install-from-pypi:

Install from PyPI with pip (fallback — bring your own versions)
---------------------------------------------------------------

Prefer your own Python/PyTorch/CUDA? Install your preferred PyTorch first (any version ≥ 2.6, built for your CUDA — see `PyTorch's install matrix <https://pytorch.org/get-started/locally/>`_), then add NeMo with the collections you need. Because ``nemo-toolkit`` only requires ``torch>=2.6``, your pre-installed PyTorch is kept, not replaced:

.. code-block:: bash

   # 1) Your choice of PyTorch (example: CUDA 12.6 build). Skip if you already have one.
   pip install torch --index-url https://download.pytorch.org/whl/cu126

   # 2) NeMo — your PyTorch above is kept
   pip install nemo_toolkit[asr,tts]        # also: [asr,tts,audio], [speechlm2], etc.

To have pip install our pinned PyTorch build instead, add the matching CUDA extra **and** the PyTorch wheel index. pip does not read uv's index configuration, so the ``--extra-index-url`` is required:

.. code-block:: bash

   pip install 'nemo-toolkit[asr,tts,cu13]' --extra-index-url https://download.pytorch.org/whl/cu132   # CUDA 13.x
   pip install 'nemo-toolkit[asr,tts,cu12]' --extra-index-url https://download.pytorch.org/whl/cu126   # CUDA 12.x

.. tip::

   Prefer a conda environment? Create and activate one (``conda create -n nemo python=3.10 -y && conda activate nemo``), then run the same ``uv`` or ``pip`` commands above inside it. NeMo Speech does not require a separate conda CUDA toolkit or a manual ``torchvision`` install.

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
