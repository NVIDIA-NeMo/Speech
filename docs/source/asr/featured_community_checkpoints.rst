.. _featured-community-checkpoints:

Featured Community Checkpoints
==============================

Community fine-tunes built on NVIDIA NeMo ASR checkpoints and published on Hugging Face.
For NVIDIA-published checkpoints, see :doc:`./asr_checkpoints` and the `NVIDIA Hugging Face organization <https://huggingface.co/nvidia>`__.

.. note::

   Community checkpoints are maintained by their authors, not by the NeMo team.
   Use each model's Hugging Face model card and the framework project linked below for up-to-date setup and inference instructions.

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Checkpoint
     - What's special
     - Framework
   * - `akera/parakeet-tdt-salt <https://huggingface.co/akera/parakeet-tdt-salt>`__
     - SALT multilingual ASR for 10 East African languages. Hybrid TDT+CTC FastConformer (600M), fine-tuned from `parakeet-tdt-0.6b-v3 <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`__.
     - NeMo
   * - `johannhartmann/parakeet_de_med <https://huggingface.co/johannhartmann/parakeet_de_med>`__
     - German medical documentation ASR (PEFT). WER 11.73% → 3.28% on a 122-sample medical eval set.
     - NeMo
   * - `qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC <https://huggingface.co/qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC>`__
     - ATC English ASR on `jacktol/ATC-ASR-Dataset <https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset>`__. Test WER 5.99%.
     - NeMo
   * - `KasuleTrevor/parakeet-0.6b-cv-sw-5hr_v9 <https://huggingface.co/KasuleTrevor/parakeet-0.6b-cv-sw-5hr_v9>`__
     - Swahili ASR fine-tune on ~5 hours of Common Voice data.
     - NeMo
   * - `NeurologyAI/neuro-parakeet-mlx <https://huggingface.co/NeurologyAI/neuro-parakeet-mlx>`__
     - German medical/neurology ASR for Apple Silicon. WER 1.04% on the author's medical validation set.
     - MLX
   * - `cstr/parakeet-tdt-0.6b-v3-GGUF <https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF>`__
     - Quantised Parakeet TDT (Q4_K ~467 MB). 25 EU languages, word-level timestamps.
     - GGUF (`CrispASR <https://github.com/CrispStrobe/CrispASR>`__)
   * - `cstr/canary-1b-v2-GGUF <https://huggingface.co/cstr/canary-1b-v2-GGUF>`__
     - Quantised Canary 1B (Q4_K ~673 MB). Multilingual ASR and speech translation.
     - GGUF (`CrispASR <https://github.com/CrispStrobe/CrispASR>`__)


.. _submit-a-community-checkpoint:

Submit a Community Checkpoint
-----------------------------

To suggest a checkpoint for this page, open a `GitHub issue <https://github.com/NVIDIA-NeMo/NeMo/issues/new>`__ with the Hugging Face model link, NeMo base checkpoint, task, languages, evaluation results, and inference framework.
