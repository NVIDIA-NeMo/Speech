Serving SpeechLM with vLLM
==========================

The NeMo SpeechLM vLLM plugin serves exported SpeechLM checkpoints with
their speech encoder and language backbone. It supports ordinary generation
and checkpoint-backed speculative decoding.

CTC word timestamp alignment
----------------------------

When CTC timestamps are requested, native SpeechLM and the vLLM timestamp
worker use the same single-pass banded CTC word aligner. Speaker mapping
uses greedy token distributions; the search retains speaker-active frame
support and scores an anchor-guided band. There is no preliminary DP pass
or selectable legacy alignment backend.

The Sortformer log-probability prior defaults to weight ``1.0``. The vLLM
checkpoint setting ``ctc_timestamps.speaker_logprob_weight`` may override
this weight without changing the alignment algorithm. Diarization-assisted
word-boundary refinement always runs, with a default cap of 12 frames
(at most 0.96 seconds). Native inference automatically chooses transcript
units from the writing script unless ``timestamp_language`` is explicit.

Remove the former ``alignment_backend`` option from native aligner
configuration; it is no longer accepted. Timestamp generation remains
opt-in and requires a CTC adapter.

Installation
------------

Install the ASR runtime and vLLM 0.32.0, pinned by NeMo's ``vllm`` extra.
vLLM 0.32 requires Python 3.11 or newer:

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
prediction depths, and verify the drafts with vLLM's ReplaySSM:

.. code-block:: bash

   vllm serve /path/to/vllm-ready-speechlm-checkpoint \
     --trust-remote-code \
     --dtype bfloat16 \
     --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none \
     --no-enable-prefix-caching \
     --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'

For a separate checkpoint-backed draft, see :doc:`vllm_dflash`.

vLLM rejects audio inputs larger than ``VLLM_MAX_AUDIO_CLIP_FILESIZE_MB``
(default 25 MB) or longer than ``VLLM_MAX_AUDIO_DECODE_DURATION_S`` (default
600 seconds). Raise both, as environment variables of the server, to transcribe
longer recordings.

.. _speechlm2-vllm-hybrid-precision:

Hybrid checkpoint precision
---------------------------

The SpeechLM wrapper preserves FP32 Mamba ``A``, ``D``, and ``dt_bias``
parameters before checkpoint loading, for both the target and MTP draft.
MoE routers use FP32 weights, operands, and output logits, matching the
Automodel training precision contract. Other language-model and perception
weights retain their configured serving precision. FP32 SSM cache defaults
are delegated to vLLM's NemotronH configuration hook.
