Serving SpeechLM with vLLM
==========================

The NeMo SpeechLM vLLM plugin serves exported SpeechLM checkpoints with
their speech encoder and language backbone. It supports ordinary generation
and checkpoint-backed speculative decoding.

Installation
------------

Install the ASR runtime and vLLM 0.28.0, pinned by NeMo's ``vllm`` extra:

.. code-block:: bash

   pip install -e ".[asr,vllm]"

The equivalent uv command is:

.. code-block:: bash

   uv sync --extra asr --extra vllm

Do not combine ``vllm`` with the ``speechlm2``, ``speechlm2-only``,
``all``, ``cu12``, ``cu13``, ``compiled``, or ``compiled-a100`` extras.
These include Automodel training dependencies; vLLM owns the exact Torch
and CUDA-kernel stack for this serving environment. Automodel is not
required to serve an exported checkpoint.

Serving an exported checkpoint
------------------------------

Start ordinary BF16 serving with a vLLM-ready NeMo SpeechLM checkpoint:

.. code-block:: bash

   vllm serve /path/to/vllm-ready-speechlm-checkpoint \
     --trust-remote-code \
     --dtype bfloat16

Hybrid checkpoints with exported MTP weights can also use vLLM's MTP draft.
Set the number of speculative tokens according to the checkpoint's trained
prediction depths, and use ``--mamba-cache-mode all`` with MTP. For a separate
checkpoint-backed draft, see :doc:`vllm_dflash`.

.. _speechlm2-vllm-hybrid-precision:

Hybrid checkpoint precision
---------------------------

The SpeechLM wrapper preserves FP32 Mamba ``A``, ``D``, and ``dt_bias``
parameters before checkpoint loading, for both the target and MTP draft.
MoE routers use FP32 weights, operands, and output logits, matching the
Automodel training precision contract. Other language-model and perception
weights retain their configured serving precision. FP32 SSM cache defaults
are delegated to vLLM's NemotronH configuration hook.
