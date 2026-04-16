Streaming Inference
===================

The speechlm2 collection provides a streaming inference pipeline for
NemotronVoiceChat that processes audio in real time, chunk by chunk, and
produces both text and speech output incrementally.  The pipeline follows the
same methodology as the NeMo ASR Inference Pipelines (see
``nemo.collections.asr.inference``).

Overview
--------

The streaming inference stack has four layers:

.. code-block:: text

    Entry Script          s2s_streaming_infer.py (Hydra)
         │
         ▼
    Pipeline              StreamingS2SPipeline
         │                  - audio buffering
         │                  - state management
         │                  - file I/O
         ▼
    Model Wrapper         NemotronVoicechatInferenceWrapper
         │                  - infer_one_step()
         │                  - perception
         │                  - model_llm_interface    (PyTorchLLM  or VLLMLLM)
         │                  - model_eartts_interface (PyTorchEarTTS or VLLMEarTTS)
         │                  - codec decode
         ▼
    Model                 NemotronVoiceChat
                            - DuplexSTTModel + DuplexEARTTS

Quick Start
-----------

Batch Inference from a Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to run streaming inference is with the provided Hydra script:

.. code-block:: bash

    python examples/speechlm2/nemo_inference_pipelines/s2s_streaming_infer.py \
        --config-path=examples/speechlm2/nemo_inference_pipelines/conf \
        --config-name=s2s_streaming \
        audio_file=/path/to/audio_or_directory_or_manifest.json \
        output_dir=./generated \
        s2s.model_path=/path/to/checkpoint \
        s2s.speaker_name="<speaker>" \
        s2s.engine_type=native \
        s2s.system_prompt="You are a helpful assistant." \
        streaming.chunk_size_in_secs=0.24 \
        streaming.buffer_size_in_secs=1.68

This will:

1. Load the NemotronVoiceChat checkpoint.
2. Stream each audio file through the pipeline in chunks.
3. Save generated ``.wav``, stereo (input+output), and ``.txt`` files under
   ``output_dir``.

Programmatic Usage
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from nemo.collections.speechlm2.inference import S2SPipelineBuilder

    pipeline = S2SPipelineBuilder.build_pipeline(cfg)
    output = pipeline.run(audio_filepaths, options=options)

    # output.texts            -- generated agent text per file
    # output.asr_texts        -- recognized user text per file
    # output.audio_filepaths  -- paths to generated .wav files


Architecture
------------

The Core Loop
^^^^^^^^^^^^^

Like the ASR pipeline's ``BasePipeline.run()``, the S2S pipeline iterates
over chunks and calls a single step method:

.. code-block:: python

    pipeline.open_session()
    for frames in streamer:
        # Each call returns partial results for this chunk only
        step_outputs = pipeline.generate_step(frames)
        for out in step_outputs:
            # out.text / out.asr_text: new tokens from this step
            # out.audio: newly decoded audio for this step
            print(f"[stream {out.stream_id}] agent: {out.text}  user: {out.asr_text}")
    pipeline.close_session()
    return PipelineOutput(...)  # texts, token tensors, audio filepaths

Each ``generate_step()`` call returns a list of ``GenerateStepOutput`` carrying
the partial text, ASR text, and audio produced by that single chunk.  The
``run()`` method uses these to build a ``PipelineOutput`` with the aggregated
results for all streams.

Output files (wav, txt, CTM) are written as each stream finishes, so
results are available on disk before the full run completes.
Two token-level CTM files are produced per stream:
``<stem>.ctm`` for agent output tokens and ``<stem>_asr.ctm`` for user
(ASR) tokens.  Timestamps in both files reflect when the text token was
*generated* by the model, which is not necessarily when the corresponding
speech was spoken.

``generate_step()`` is the unified entry point used by **both** the batch
``run()`` method and server deployments.


What Happens Inside One Step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    generate_step(frames)
        │
        ├─ for each frame where is_first=True:
        │      │
        │      └─ _init_state(stream_id, options)
        │             1. augment_with_defaults()   ← fill None fields from YAML
        │             2. create_state(options)      ← pipeline-level state
        │             3. create context_manager     ← KV caches, decode slots
        │             4. prefill system prompt      ← populate LLM KV cache
        │
        ├─ any frames with audio?
        │      │
        │     NO → return empty outputs  (server prefill-only request)
        │      │
        │    YES ↓
        │
        └─ generate_step_for_frames()
               1. audio buffering
               2. perception encoder
               3. per-frame LLM loop
               4. per-frame TTS
               5. codec decode
               6. state updates + output accumulation
               7. return list[GenerateStepOutput]

Each call to ``generate_step(frames)`` performs:

1. **Stream init on** ``is_first`` -- If a frame has ``is_first=True``, the
   private ``_init_state()`` method runs: per-stream options are merged with
   pipeline defaults (via ``S2SRequestOptions.augment_with_defaults()``),
   a fresh ``S2SStreamingState`` is created, the context manager is
   allocated, and the LLM KV cache is prefilled with the system prompt and
   TTS speaker embedding.  This mirrors ASR's ``init_state()`` called inside
   ``transcribe_step()``.  If the frame carries no audio (zero-length
   samples), the method returns after init — this is the recommended
   pattern for latency-sensitive server deployments (see
   :ref:`init-and-latency` below).

2. **Audio buffering** -- ``BatchedAudioBufferer`` (reused from ASR
   infrastructure) maintains a sliding window of ``buffer_size_in_secs``.

3. **Model inference** via ``infer_one_step(audio_buffer, state)``:

   a. **Perception** -- The audio buffer is encoded by the streaming
      FastConformer encoder into frame embeddings.
   b. **Per-frame LLM loop** -- For each of the ``num_frames_per_chunk``
      frames, the pipeline builds an input embedding (user audio +
      previous-step text/ASR tokens), runs it through the LLM, and obtains
      predicted text and ASR tokens.
   c. **Per-frame TTS** -- Each predicted text token is fed into the EarTTS
      model to produce audio codec codes.
   d. **Codec decode** -- The accumulated codes are decoded into a waveform.

4. **State updates** -- The context manager advances ``frame_idx`` and
   updates the subword mask.

5. **Output accumulation** -- Decoded audio and text are appended to the
   per-stream ``S2SStreamingState``.


Two Kinds of State
^^^^^^^^^^^^^^^^^^

The pipeline maintains two separate state objects per stream:

**StreamingDecodeState** (model level)
    Lives in ``S2SContextManager`` slots.  Contains LLM KV cache, TTS KV
    cache, perception cache, codec cache, token workspaces (``gen_text``,
    ``gen_asr_text``), and ``frame_idx``.  Created by the wrapper, mutated
    in-place by ``infer_one_step()``, destroyed at end-of-stream.

**S2SStreamingState** (pipeline level)
    Lives in the pipeline's ``_state_pool``.  Accumulates generated audio
    chunks and text strings across steps.  In the ``run()``
    batch path, the state is freed as soon as the stream's output files are
    saved.  In the server path, the caller controls lifetime via
    ``close_session()`` or ``delete_state()``.


Inference Backends
^^^^^^^^^^^^^^^^^^

NemotronVoiceChat has two inference components that each need a backend:

- **LLM** (DuplexSTT backbone) -- takes audio embeddings from the perception
  encoder and predicts text tokens, ASR tokens, and optional function-call
  tokens at each frame.
- **TTS** (EarTTS) -- takes the predicted text token and produces audio codec
  codes (RVQ acoustic tokens).

Each component can run on **native PyTorch** or **vLLM**, selected by the
``engine_type`` config value:

.. list-table::
   :header-rows: 1
   :widths: 35 30 30

   * - ``engine_type``
     - LLM backend
     - TTS backend
   * - ``native``
     - ``PyTorchLLM``
     - ``PyTorchEarTTS``
   * - ``vllm_llm``
     - ``VLLMLLM``
     - ``PyTorchEarTTS``
   * - ``vllm_eartts``
     - ``PyTorchLLM``
     - ``VLLMEarTTS``
   * - ``vllm_llm_vllm_eartts``
     - ``VLLMLLM``
     - ``VLLMEarTTS``

All four backend classes implement the same ``ModelInterface`` ABC (defined in
``inference.model_wrappers.backend.interface``), so the inference wrapper
(``NemotronVoicechatInferenceWrapper``) can treat them uniformly via two
attributes:

- ``model_llm_interface`` -- the LLM backend
- ``model_eartts_interface`` -- the TTS backend

The backend classes live under ``inference.model_wrappers.backend/``:

.. code-block:: text

    backend/
        interface.py            # ModelInterface ABC + shared sampling
        pytorch/
            model.py            # PyTorchLLM  (wraps DuplexSTT forward pass)
            eartts.py           # PyTorchEarTTS  (wraps DuplexEARTTS.infer_codes_one_step)
        vllm/
            base.py             # VLLMModelBase  (engine lifecycle, async loop)
            llm.py              # VLLMLLM  (DuplexSTT via vLLM)
            eartts.py           # VLLMEarTTS  (EarTTS via vLLM)
    factory.py                  # create_model()  dispatches to the right class

``ModelInterface`` provides shared utilities used by the LLM backends:
top-p (nucleus) sampling, repetition penalty, and temperature scaling.
These are applied **post-hoc** on the returned logits -- vLLM internally
runs with ``skip_sampling=True`` and greedy decoding.

Each backend also exposes lifecycle methods that the wrapper calls uniformly:

- ``prefill_prompt(embeddings, ...)`` -- Warm up KV cache (native) or
  prefill the vLLM engine with system-prompt embeddings before streaming.
- ``compile()`` -- Apply ``torch.compile`` to the TTS backbone (native
  only; no-op for vLLM).
- ``setup_subword_cache(cfg)`` -- Enable the TTS subword embedding cache
  (native only; no-op for vLLM).

The ``factory.create_model()`` function is the single entry point that
dispatches to the correct class based on a per-component ``engine_type``
string (``native_llm``, ``native_eartts``, ``vllm_llm``, ``vllm_eartts``).

vLLM Integration Details
""""""""""""""""""""""""

When ``engine_type`` includes ``vllm``, the pipeline loads vLLM engines
**in-process** alongside the native PyTorch components -- there is no
disaggregated multi-server setup.  Each vLLM component runs as an
``AsyncLLM`` engine in the same Python process, sharing GPU memory with the
native perception encoder and codec decoder.

The vLLM engines manage their own KV caches via PagedAttention.  Both
``VLLMLLM`` and ``VLLMEarTTS`` inherit from ``VLLMModelBase``, which wraps a
``CustomInputAsyncVLLMEngine`` (defined in ``inference.vllm.streaming_llm_engine``)
and provides:

- An internal ``asyncio`` event loop for blocking synchronous calls
- Request lifecycle management (start, abort, restart)
- Automatic checkpoint conversion to vLLM format on first use

``CustomInputAsyncVLLMEngine`` is a thin wrapper around vLLM's ``AsyncLLM``
that adds support for custom input tensor specifications (multi-tensor
inputs like audio embeddings, subword IDs, speaker latents).  The
``engine_kind`` parameter (``"llm"`` or ``"eartts"``) selects EarTTS-specific
runtime settings (TRITON attention backend, TF32 precision, guidance scale)
without introducing inheritance between TTS and LLM engine classes.

This requires a custom vLLM fork with NemotronVoiceChat model support:

.. code-block:: bash

    pip install git+https://github.com/vklimkov-nvidia/vllm@vklimkov/voicechat


Configuration
-------------

The streaming inference configuration is defined in
``examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml``.

Key configuration groups:

S2S Model Settings (``s2s``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``model_path``
     - (required)
     - Path to the NemotronVoiceChat HuggingFace checkpoint.
   * - ``engine_type``
     - (required)
     - ``native``, ``vllm_llm``, ``vllm_eartts``, or
       ``vllm_llm_vllm_eartts``.
   * - ``speaker_name``
     - ``null``
     - Registered speaker name (must match a speaker in the checkpoint).
   * - ``system_prompt``
     - (required)
     - Text injected into the LLM KV cache before audio streaming begins.
   * - ``compute_dtype``
     - ``bfloat16``
     - Precision for LLM/embedding layers.
   * - ``use_perception_cache``
     - ``true``
     - Cache-aware streaming for the perception encoder.
   * - ``use_llm_cache``
     - ``true``
     - Use KV cache for incremental LLM decoding.
   * - ``top_p``
     - ``0.5``
     - Top-p sampling threshold.
   * - ``temperature``
     - ``0.3``
     - Sampling temperature.
   * - ``repetition_penalty``
     - ``1.1``
     - Repetition penalty applied to previously generated tokens.
   * - ``deterministic``
     - ``false``
     - Force deterministic mode (native engine only).
   * - ``profile_timing``
     - ``false``
     - Insert ``torch.cuda.synchronize()`` around each stage for accurate
       per-stage timing.  Disabled by default to avoid GPU stalls.

Streaming Settings (``streaming``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``chunk_size_in_secs``
     - (required)
     - Audio processed per inference step.  Must be a multiple of 0.08 s.
   * - ``buffer_size_in_secs``
     - (required)
     - Sliding-window size passed to the perception encoder.
   * - ``batch_size``
     - ``1``
     - Number of concurrent streams (currently only 1 is supported).
   * - ``max_len``
     - ``8192``
     - Maximum number of frames per stream.

Padding Settings (top-level)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At most one of these may be set:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``pad_audio_to_sec``
     - ``null``
     - Pad each input to a fixed duration.
   * - ``pad_silence_ratio``
     - ``null``
     - Append silence equal to this fraction of the original duration.
   * - ``pad_audio_by_sec``
     - ``null``
     - Append a fixed number of extra seconds of silence.


Server Integration
------------------

The same ``generate_step()`` method used by ``run()`` can be called directly
from a custom server.  It returns a list of ``GenerateStepOutput`` objects
(one per input frame) carrying the **incremental** audio and text produced
by this step — no need to diff against accumulated state:

.. code-block:: python

    from nemo.collections.speechlm2.inference import GenerateStepOutput

    # 1. Init stream (empty audio so prefill completes before recording)
    init_frame = Frame(
        samples=torch.empty(0),
        stream_id=stream_id,
        is_first=True, is_last=False,
        options=S2SRequestOptions(system_prompt=prompt, top_p=0.9),
    )
    pipeline.generate_step([init_frame])
    # -> client can now start recording

    # 2. Stream audio chunks and consume incremental outputs
    for i, chunk in enumerate(audio_source):
        frame = Frame(
            samples=chunk,
            stream_id=stream_id,
            is_first=False, is_last=(i == last),
        )
        outputs = pipeline.generate_step([frame])
        for out in outputs:
            send_to_client(out.audio, out.text, out.asr_text)

Per-stream options (``system_prompt``, ``top_p``, ``temperature``,
``repetition_penalty``) are attached to the ``is_first`` frame via
``S2SRequestOptions``.  Any field left as ``None`` falls back to the
pipeline-level YAML default through ``augment_with_defaults()``.

.. _init-and-latency:

Init and Latency
^^^^^^^^^^^^^^^^

When ``generate_step`` sees ``is_first``, it always runs stream
initialization (context creation, KV-cache prefill).  If the frame also
carries audio, inference runs immediately after init in the same call.

For **latency-sensitive** server deployments (real-time voice chat),
prefill can take hundreds of milliseconds or even multiple seconds.  
Clients should send ``is_first`` with **empty audio**, wait for the 
response confirming init is done, and only then start recording the 
user's microphone.  This prevents audio from queuing up during the 
expensive prefill phase.

For **batch/offline** usage (CLI ``run()``), there is no real-time
constraint.  The first frame carries both ``is_first`` and real audio,
so init and first-chunk processing happen in one call with no extra
round-trip.

The pipeline makes no distinction between these cases — it initializes
on ``is_first`` and processes whatever audio is present.  The latency
trade-off is entirely the caller's choice.


Batch Size
----------

The pipeline currently supports ``batch_size=1`` (one stream at a time).


File Layout
-----------

.. code-block:: text

    nemo/collections/speechlm2/inference/
    ├── __init__.py                          # Public exports
    ├── factory/
    │   └── s2s_pipeline_builder.py          # S2SPipelineBuilder
    ├── pipelines/
    │   ├── s2s_pipeline_interface.py        # Base: _state_pool, sessions
    │   └── streaming_s2s_pipeline.py        # StreamingS2SPipeline
    ├── model_wrappers/
    │   ├── decode_state.py                  # StreamingDecodeState, InferenceStepResult
    │   ├── nemotron_voicechat_inference_wrapper.py
    │   ├── model_factory.py                 # Native / vLLM model interfaces
    │   └── perception_cache.py              # Perception cache + CUDA graphs
    ├── streaming/
    │   ├── framing/
    │   │   └── s2s_request_options.py       # S2SRequestOptions
    │   └── state/
    │       ├── s2s_state.py                 # S2SStreamingState
    │       └── s2s_context_manager.py       # Slot-based decode-state lifecycle
    ├── utils/
    │   ├── pipeline_utils.py                # PipelineOutput, text helpers
    │   └── audio_data.py                    # Manifest / folder loading
    └── vllm/                                # Optional vLLM engine backend
