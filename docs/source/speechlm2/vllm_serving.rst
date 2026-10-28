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

The MTP draft compiles the Triton variants of vLLM's
``eagle_prepare_inputs_padded_kernel`` while the weights load, so the first
request with a new batch size does not stall on a JIT compile.

If the checkpoint's chat template enables thinking by default, add
``--default-chat-template-kwargs '{"enable_thinking": false}'`` so OpenAI-style
clients receive plain transcripts without passing the option on every request.
Async scheduling (``--async-scheduling``) can lower the latency of a single short
clip; compare it against synchronous scheduling with your own load and audio
lengths before enabling it, because the best setting depends on both.

.. _speechlm2-vllm-hybrid-precision:

Hybrid checkpoint precision
---------------------------

The SpeechLM wrapper preserves FP32 Mamba ``A``, ``D``, and ``dt_bias``
parameters before checkpoint loading, for both the target and MTP draft.
MoE routers use FP32 weights, operands, and output logits, matching the
Automodel training precision contract. Other language-model and perception
weights retain their configured serving precision. FP32 SSM cache defaults
are delegated to vLLM's NemotronH configuration hook; with ReplaySSM, a
bfloat16 cache is faster at the same accuracy (see
:ref:`speechlm2-vllm-throughput-settings`).

.. _speechlm2-vllm-encoder-cuda-graphs:

Audio encoder CUDA graphs
-------------------------

vLLM encodes the audio of the requests it admits in each engine step, often a few dozen short clips. For a
``ParallelExpertEncoder`` at that batch size, the encoder's cost is mostly kernel launches. Setting
``encoder_cuda_graphs`` replays the encoder (everything after the mel front end) from CUDA graphs captured per
batch size and audio length:

.. code-block:: bash

   vllm serve /path/to/vllm-ready-speechlm-checkpoint \
     --trust-remote-code \
     --dtype bfloat16 \
     --hf-overrides '{"encoder_cuda_graphs": true, "encoder_compile": true}'

The keys can also be set in the checkpoint's ``config.json``.

- ``encoder_cuda_graphs``: capture graphs at the first encoder call, so audio of any length replays them. Audio up
  to two 40-second encoder windows replays one graph per batch of 1 to 128 clips. Longer audio runs the windowed
  encoder with each piece replayed from graphs: the ASR encoder over batches of windows, and every streaming
  diarization step. Each clip of a step takes the path its own length needs, so a long clip does not slow the short
  ones; long clips of similar length share a windowed pass. Capture adds start-up time and GPU memory; the graphs
  reproduce the encoder's outputs.
- ``encoder_compile``: also compile each encoder transformer block with ``torch.compile``, fusing its
  memory-bound operations. This lowers encoder GPU time further and changes the outputs at bf16 rounding level.

The graph buckets can be changed with these keys:

- ``encoder_cuda_graph_batch_sizes``: graph batch sizes, default ``[1, 2, 4, 8, 16, 32, 64, 128]``. A step's clips
  are split over them, largest first; fewer than 8 remaining clips go into one padded graph. The small sizes cut
  the encoder latency of a single request; dropping them roughly halves the capture time at start-up.
- ``encoder_cuda_graph_max_windows``: longest audio replayed as one whole-encoder graph, in encoder windows,
  default ``2``. Longer audio is graphed window by window.
- ``encoder_cuda_graph_frame_bucket``: length granularity of the whole-encoder graphs in feature frames (10 ms
  each), default ``256``.
- ``encoder_cuda_graph_long_audio``: graph audio past ``encoder_cuda_graph_max_windows`` window by window, default
  ``true``. Set it to ``false`` to save the start-up time and memory of those graphs when no request is longer; such
  audio then runs eagerly.
- ``encoder_cuda_graph_last_window_bucket``: for audio graphed window by window, the last window is padded to a
  multiple of this many feature frames, default ``1000``. Smaller values pad less and capture more graphs.

For example, a low-concurrency deployment of short clips:

.. code-block:: bash

   vllm serve /path/to/vllm-ready-speechlm-checkpoint \
     --trust-remote-code \
     --dtype bfloat16 \
     --max-num-seqs 8 \
     --hf-overrides '{"encoder_cuda_graphs": true, "encoder_compile": true,
                      "encoder_cuda_graph_batch_sizes": [1, 2, 4, 8], "encoder_cuda_graph_max_windows": 1}'

Both ``encoder_cuda_graphs`` and ``encoder_compile`` are off by default. If a capture fails, the encoder runs eagerly
instead. With CTC timestamps enabled, the graphs also produce the alignment inputs, so requests that ask for
timestamps are served from the same graphs.

.. _speechlm2-vllm-throughput-settings:

Throughput settings
-------------------

For high-throughput serving, especially of long recordings, serve the MTP checkpoint with:

.. code-block:: bash

   vllm serve /path/to/vllm-ready-speechlm-checkpoint \
     --trust-remote-code \
     --dtype bfloat16 \
     --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none \
     --no-enable-prefix-caching \
     --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
     --hf-overrides '{"encoder_cuda_graphs": true, "encoder_compile": true}' \
     --mamba-ssm-cache-dtype bfloat16 \
     --api-server-count 4

- ``encoder_cuda_graphs`` and ``encoder_compile`` (see :ref:`speechlm2-vllm-encoder-cuda-graphs`): a long recording
  is encoded window by window in the engine step that admits it, so the requests being decoded wait for it; the
  graphs replay those windows instead of launching their many small kernels one by one, and ``encoder_compile``
  fuses the encoder's memory-bound operations into fewer kernels, changing its outputs at bf16 rounding level.
- ``--mamba-ssm-cache-dtype bfloat16``: overrides the float32 Mamba state cache the checkpoint config asks for. With
  ReplaySSM, the state is written to the cache only every ``--replayssm-buffer-len`` steps, so it is rounded to
  bfloat16 once per write rather than once per step, and accuracy is unchanged. The smaller state makes each
  decoding step cheaper.
- ``--api-server-count 4``: four API-server processes prepare incoming audio in parallel, so a burst of long
  recordings reaches the engine sooner. Each keeps its own multimodal processor cache (``--mm-processor-cache-gb``).
