.. _featured-community-checkpoints:

Featured Community Checkpoints
==============================

Community fine-tunes built on NVIDIA NeMo ASR checkpoints and published on Hugging Face.
Depending on the repo, a checkpoint loads with **NeMo** (``.nemo``), **MLX** (Apple Silicon), or **GGUF** (C++ via `CrispASR <https://github.com/CrispStrobe/CrispASR>`__).

For NVIDIA-published checkpoints, see :doc:`./asr_checkpoints` and the `NVIDIA Hugging Face organization <https://huggingface.co/nvidia>`__.

.. note::

   Community checkpoints are maintained by their authors, not by the NeMo team.


NeMo
----

Load checkpoints that ship a ``.nemo`` file on Hugging Face with ``ASRModel.from_pretrained()``:

.. code-block:: python

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained("johannhartmann/parakeet_de_med")
    print(model.transcribe(["audio.wav"])[0].text)

.. list-table::
   :header-rows: 1
   :widths: 26 20 14 40

   * - Model
     - Base checkpoint
     - License
     - Highlights
   * - `akera/parakeet-tdt-salt <https://huggingface.co/akera/parakeet-tdt-salt>`__
     - `parakeet-tdt-0.6b-v3 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`__
     - See model card
     - SALT multilingual ASR for 10 East African languages. Hybrid TDT+CTC FastConformer, 600M params.
   * - `johannhartmann/parakeet_de_med <https://huggingface.co/johannhartmann/parakeet_de_med>`__
     - `parakeet-tdt-0.6b-v3 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`__
     - CC-BY-4.0
     - German medical documentation ASR. PEFT fine-tune; WER 11.73% → 3.28% on a 122-sample medical eval set.
   * - `qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC <https://huggingface.co/qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC>`__
     - `parakeet-tdt-0.6b-v3 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`__
     - See model card
     - ATC English ASR on `jacktol/ATC-ASR-Dataset <https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset>`__. Test WER 5.99%.
   * - `KasuleTrevor/parakeet-0.6b-cv-sw-5hr_v9 <https://huggingface.co/KasuleTrevor/parakeet-0.6b-cv-sw-5hr_v9>`__
     - Parakeet 0.6B (see model card)
     - CC-BY-4.0
     - Swahili ASR fine-tune on ~5 hours of Common Voice data.


.. _mlx-inference:

MLX Inference
-------------

For Apple Silicon checkpoints, use ``parakeet-mlx`` or ``mlx-audio``:

.. code-block:: bash

    pip install parakeet-mlx
    parakeet-mlx audio.wav --model NeurologyAI/neuro-parakeet-mlx

.. list-table::
   :header-rows: 1
   :widths: 26 20 14 40

   * - Model
     - Base checkpoint
     - License
     - Highlights
   * - `NeurologyAI/neuro-parakeet-mlx <https://huggingface.co/NeurologyAI/neuro-parakeet-mlx>`__
     - `parakeet-tdt-0.6b-v3 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`__
     - CC-BY-4.0
     - German medical/neurology ASR for Apple Silicon. WER 1.04% on the author's medical validation set.


.. _gguf-inference:

GGUF Inference
--------------

GGUF exports run with the `CrispASR <https://github.com/CrispStrobe/CrispASR>`_ C++ CLIs — no NeMo install required:

.. code-block:: bash

    git clone -b parakeet https://github.com/CrispStrobe/CrispASR
    cd CrispASR && cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc) --target parakeet-main canary-main

    huggingface-cli download cstr/parakeet-tdt-0.6b-v3-GGUF parakeet-tdt-0.6b-v3-q4_k.gguf --local-dir .
    ./build/bin/parakeet-main -m parakeet-tdt-0.6b-v3-q4_k.gguf -f audio.wav -t 8

.. list-table::
   :header-rows: 1
   :widths: 26 20 14 40

   * - Model
     - Base checkpoint
     - License
     - Highlights
   * - `cstr/parakeet-tdt-0.6b-v3-GGUF <https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF>`__
     - `parakeet-tdt-0.6b-v3 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`__
     - CC-BY-4.0
     - Quantised Parakeet TDT (Q4_K ~467 MB). 25 EU languages, word-level timestamps. Run with ``parakeet-main``.
   * - `cstr/canary-1b-v2-GGUF <https://huggingface.co/cstr/canary-1b-v2-GGUF>`__
     - `canary-1b-v2 <https://huggingface.co/nvidia/canary-1b-v2>`__
     - CC-BY-4.0
     - Quantised Canary 1B (Q4_K ~673 MB). Multilingual ASR and speech translation. Run with ``canary-main``.


.. _submit-a-community-checkpoint:

Submit a Community Checkpoint
-----------------------------

To suggest a checkpoint for this page, open a `GitHub issue <https://github.com/NVIDIA-NeMo/NeMo/issues/new>`__ with the Hugging Face model link, NeMo base checkpoint, task, languages, and evaluation results.
