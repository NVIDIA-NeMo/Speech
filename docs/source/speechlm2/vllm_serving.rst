Serving SpeechLM with vLLM
==========================

The NeMo SpeechLM vLLM plugin serves exported SpeechLM checkpoints with
their speech encoder and language backbone. It supports ordinary generation
and checkpoint-backed speculative decoding.

CTC word timestamp alignment
----------------------------------------

When CTC timestamps are requested, native SpeechLM and the vLLM timestamp
worker use the same single-pass banded CTC word aligner. Speaker mapping
uses greedy token distributions; the search retains speaker-active frame
support and scores an anchor-guided band. There is no preliminary DP pass
or selectable legacy alignment backend.

The Sortformer prior, duration limits, and diarization-assisted boundary
refinement settings are checkpoint-configurable. The vLLM checkpoint setting
``ctc_timestamps.speaker_logprob_weight`` explicitly overrides the calibrated
speaker weight. Native inference automatically chooses transcript units from
the writing script unless ``timestamp_language`` is explicit.

Remove the former ``alignment_backend`` option from native aligner
configuration; it is no longer accepted. Timestamp generation remains
opt-in and requires a CTC adapter.

Checkpoint-specific CTC configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export timestamp settings in the model directory's existing ``config.json``
under ``ctc_timestamps``. No safetensors rewrite or separate YAML file is needed
when the model already contains the compatible CTC head and tokenizer tensors.
The block has three configuration mappings:

* ``decoder_config``: CTC decoder construction options, including vocabulary,
  Transformer settings, ``upsampling_factor``, and ``upsampling_mode``.
* ``tokenizer_config``: SentencePiece runtime options (the tokenizer model bytes
  remain in the checkpoint).
* ``alignment_config``: a versioned envelope containing ``schema_version: 1``,
  ``basis_version: 1``, the matching ``upsampling_factor``, and ``parameters``.
  The parameters include scalar alignment controls, interpolation profile/grid
  settings, and ``boundary_interpolation`` with start/end ``coefficients`` and
  ``offset`` arrays. Export the complete validated configuration for the head;
  these values are checkpoint-specific, not interchangeable across adapters.

Both native and vLLM loading prefer these mappings over legacy safetensors
metadata. Missing mappings fall back to metadata for older artifacts. Invalid
JSON or incompatible calibration raises an error instead of silently selecting
legacy values. Explicit caller alignment options override checkpoint settings;
vLLM also forwards its effective HF configuration, including ``hf_overrides``.
Without any supplied calibration, constructor defaults apply and boundary
interpolation is skipped. Configuration is loaded at model startup; restart the
model after editing the file.

Factor-2 upsampling of 80 ms encoder frames gives 40 ms CTC frames. Calibrated
boundary interpolation can refine the output onto a 10 ms grid before
diarization refinement; that grid is not native 10 ms acoustic resolution.
Native CTC transcription converts character, word, and segment offsets using
the decoder's output-frame rate, and intermediate CTC losses use the scaled
valid output lengths. Diarization onset snapping preserves calibrated
same-speaker order; near recording ends, onset spacing reserves the configured
minimum word duration first, subject to correction and recording bounds.
The existing ``ctc_timestamps.enabled`` and ``adapter_path`` controls retain
their behavior. Leaving ``enabled`` unspecified keeps automatic bundled-head
discovery optional; ``enabled: false`` disables it, and ``enabled: true`` makes
unavailable timestamp support an initialization error.

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
