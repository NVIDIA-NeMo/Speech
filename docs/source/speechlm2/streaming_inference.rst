Streaming Inference
===================

The speechlm2 collection provides a streaming inference pipeline for
NemotronVoiceChat that processes audio chunk by chunk, producing text and speech
output incrementally.  The pipeline follows a similar API to the NeMo ASR Inference Pipelines
(see ``nemo.collections.asr.inference``).

There are two ways to use the pipeline:

* ``StreamingS2SPipeline.run()`` processes complete audio files.  It is used by
  ``s2s_streaming_infer.py`` for a single ``.wav`` file, a directory of ``.wav``
  files, or a manifest.
* ``StreamingS2SPipeline.generate_step()`` processes one batch of ``Frame``
  objects and returns incremental outputs for that step.  Use it for servers,
  microphone connectors, or other live audio sources.

.. code-block:: text

    File inputs: one or more .wav files
    (single path, directory, or manifest)
                    │
                    ▼
              run(audio_filepaths)
                    │  creates Frame chunks
                    ▼
              generate_step(frames)
                    │
                    ├─ incremental agent audio + text
                    └─ incremental user ASR text

Each audio file passed to ``run()`` is treated as one continuous audio stream. ``run()``
accumulates the per-step outputs for each stream and writes final audio/text
artifacts when the stream ends.

The script can append trailing silence so the agent is more likely to finish
speaking before the stream ends.  When a manifest contains reference ``text``
fields, it also reports WER for the recognized user speech.

.. note::

   The current implementation supports one stream at a time (``batch_size=1``).

Script Call Path
----------------

The ``s2s_streaming_infer.py`` script follows this call path:

.. code-block:: text

    Entry Script          s2s_streaming_infer.py
         │
         ▼
    Pipeline              StreamingS2SPipeline.run()
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

(With ``s2s.decode_audio=false``, the model still predicts text/ASR tokens but
skips EarTTS generation and codec decoding.)

Quick Start
-----------

File-Based Inference from a Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Call the Python script and pass configuration values as command-line overrides:

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
3. Save per-stream output files under ``output_dir``: generated ``.wav``,
   stereo input+output ``.wav``, ``.txt``, and per-token ``.ctm``.
4. Write ``output_processed.json`` and ``output_raw.json`` summarising the run.

Programmatic Usage
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from nemo.collections.speechlm2.inference import S2SPipelineBuilder

    pipeline = S2SPipelineBuilder.build_pipeline(cfg)
    outputs = pipeline.run(audio_filepaths, options=options)

    # returns list[S2SStreamingOutput], one per input file
    # each element has: .output_text_str, .output_asr_text_str, .audio_filepath, ...

File Inputs and Manifests
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``audio_file`` argument accepted by
``examples/speechlm2/nemo_inference_pipelines/s2s_streaming_infer.py`` may be:

* A single ``.wav`` file.
* A directory, in which case all ``.wav`` files in that directory are streamed.
* A line-delimited ``.json`` or ``.jsonl`` manifest listing audio files.

Manifest entries must provide ``audio_filepath`` and may also provide
``system_prompt`` and ``text``:

.. code-block:: json

    {"audio_filepath": "audio/example.wav", "system_prompt": "You are helpful.", "text": "reference user transcript"}

The JSON/JSONL manifest accepted by ``s2s_streaming_infer.py`` has this schema.
Its ``text`` field is read only as an optional reference transcript for WER on
the ASR/user side.  Generated agent text is produced by the model and written to
``pred_text`` in the output JSON.

This lightweight streaming inference manifest is distinct from the dataset
manifests used for SpeechLM2 training and offline evaluation.  For those dataset
formats, see :doc:`SpeechLM2 datasets <datasets>`.

File paths in streaming inference manifests are resolved relative to the
manifest file.  Audio from file inputs is converted to mono and
resampled to ``streaming.input_sample_rate`` before it is chunked.


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

Use ``generate_step()`` directly when audio does not come from complete files:
for example, from a microphone, socket, Triton server, or browser UI.  The
caller owns the input connector: capture audio, convert it to mono
``streaming.input_sample_rate`` samples, split it into chunks, and pass those
chunks as ``Frame`` objects.

``generate_step()`` returns one ``GenerateStepOutput`` for each input frame,
containing the audio, agent text, and user ASR text produced by that step.

.. code-block:: python

    from nemo.collections.asr.inference.streaming.framing.request import Frame
    from nemo.collections.speechlm2.inference import S2SRequestOptions

    # 1. Initialize the stream before recording starts.
    #    Send empty audio because prefill will likely take longer than chunk_size_in_secs.
    init_frame = Frame(
        samples=torch.empty(0),
        stream_id=stream_id,
        is_first=True, is_last=False,
        options=S2SRequestOptions(system_prompt=prompt, top_p=0.9),
    )
    pipeline.generate_step([init_frame])
    # -> the input connector can now start sending audio

    # 2. For each input audio chunk, run one streaming step.
    for chunk, is_last in audio_source:
        frame = Frame(
            samples=chunk,
            stream_id=stream_id,
            is_first=False, is_last=is_last,
        )
        outputs = pipeline.generate_step([frame])
        for out in outputs:
            send_to_client(out.audio, out.text, out.asr_text)

Per-stream options (``system_prompt``, ``top_p``, ``temperature``,
``repetition_penalty``) are attached to the ``is_first`` frame via
``S2SRequestOptions``.  Any field left as ``None`` falls back to the
pipeline-level YAML default through ``fill_defaults()``.

.. _init-and-latency:

Init and Latency
^^^^^^^^^^^^^^^^

When ``generate_step`` sees ``is_first``, it always runs stream
initialization (context creation, KV-cache prefill).  If the frame also
carries audio, inference runs immediately after init in the same call.

For **latency-sensitive** integrations, prefill can take hundreds of
milliseconds or even multiple seconds.  Send ``is_first`` with **empty audio**,
wait for the response confirming init is done, and only then start sending real
audio.  This prevents input audio from queuing up during the expensive prefill
phase.

For **batch/offline** usage (``run()``), there is no real-time
constraint.  The first frame carries both ``is_first`` and real audio,
so init and first-chunk processing happen in one call with no extra
round-trip.

The pipeline makes no distinction between these cases — it initializes
on ``is_first`` and processes whatever audio is present.  The latency
trade-off is entirely the caller's choice.


Architecture
------------

File Chunking in ``run()``
^^^^^^^^^^^^^^^^^^^^^^^^^^

For file inputs, ``StreamingS2SPipeline.run()`` uses
``SilencePaddedContinuousBatchedFrameStreamer`` to load the audio paths,
convert them to mono ``streaming.input_sample_rate`` audio, and emit ``Frame``
chunks.  The streamer uses the configured chunk size, batch size, and optional
silence-padding settings.  ``run()`` then passes each emitted frame batch to
``generate_step``.  Live integrations do not need this helper; they can
construct ``Frame`` objects directly and call ``generate_step``.

The Core Streaming Loop
^^^^^^^^^^^^^^^^^^^^^^^

``StreamingS2SPipeline.run()`` orchestrates the streaming loop, delegating
per-chunk inference to ``generate_step()`` and saving outputs as streams
finish.  In simplified pseudocode:

.. code-block:: python

    # Inside StreamingS2SPipeline.run() (simplified):
    self.open_session()
    for frames in streamer:
        # step_outputs[i] carries GenerateStepOutput.audio / .text / .asr_text
        # — the new agent audio and text produced by this chunk.
        step_outputs = self.generate_step(frames)
        self._finalize_and_save_finished_streams(frames, ...)
    self.close_session()
    # run() then returns list[S2SStreamingOutput], one per input file

``run()`` returns a list of finalized ``S2SStreamingOutput`` objects (one per
input audio file) with the accumulated texts, token tensors, and audio
filepaths.

``run()`` writes outputs as each stream finishes, so results appear on disk
before the full run completes.  For each stream:

* ``<stem>.txt`` - agent transcript.
* ``<stem>.ctm`` & ``<stem>_asr.ctm`` - per-token timing for agent text and ASR text.
  Timestamps reflect when the text token was generated by the model.
* ``<stem>.wav`` & ``<stem>_input_output.wav`` - generated agent audio, plus a
  stereo file with input on one channel and output on the other.

  * In the stereo file, the generated-output channel is offset by one chunk so
    playback reflects the minimum delay from waiting for a full input chunk
    before generating output (Note: actual inference time would add to this in
    a real deployment).
  * Both audio files are skipped when ``s2s.decode_audio=false``.

After all streams finish, ``s2s_streaming_infer.py`` also writes two JSON
summaries of the run: ``output_raw.json`` (full token stream including padding
tokens) and ``output_processed.json`` (padding tokens removed for legibility).

``run()`` loops over chunks of existing audio files, calling ``generate_step()``
on each; ``generate_step()`` can also be called directly when audio comes from
a non-file source.


What Happens Inside One Step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    generate_step(frames)
        │
        ├─ for each frame where is_first=True:
        │      │
        │      └─ _init_state(stream_id, options)
        │             1. fill_defaults()           ← fill None fields from YAML
        │             2. create_state(options)      ← pipeline-level state
        │             3. reset context_manager       ← fresh decode-state storage
        │             4. prefill system prompt      ← populate LLM KV cache
        │
        └─ any frames with audio?
               │
              NO → return empty outputs  (server prefill-only request)
               │
             YES → update per-stream sliding audio buffer
                      │
                      ▼
                   generate_step_for_frames()
                      1. perception encoder
                      2. per-frame LLM loop
                      3. per-frame TTS (when decode_audio=true)
                      4. codec decode  (when decode_audio=true)
                      5. state updates + output accumulation
                      6. return list[GenerateStepOutput]

Each call to ``generate_step(frames)`` performs:

1. **Stream init on** ``is_first`` -- If a frame has ``is_first=True``, the
   private ``_init_state()`` method runs: per-stream options are merged with
   pipeline defaults (via ``S2SRequestOptions.fill_defaults()``),
   a fresh ``S2SStreamingOutput`` is created, the context manager is
   allocated, and the LLM KV cache is prefilled with the system prompt and
   TTS speaker embedding.  This mirrors ASR's ``init_state()`` called inside
   ``transcribe_step()``.  If the frame carries no audio (zero-length
   samples), the method returns after init — this is the recommended
   pattern for latency-sensitive deployments (see
   :ref:`init-and-latency` above).

2. **Audio buffer update** -- ``generate_step`` updates each stream's rolling
   audio buffer so the model receives the current ``buffer_size_in_secs``-size
   window of audio.

3. **Model inference** via ``infer_one_step(audio_buffer, state)``:

   a. **Perception** -- The audio buffer is encoded by the streaming
      FastConformer encoder into frame embeddings.
   b. **Per-frame LLM loop** -- For each of the ``num_frames_per_chunk``
      frames, the pipeline builds an input embedding (user audio +
      previous-step text/ASR tokens), runs it through the LLM, and obtains
      predicted text and ASR tokens.
   c. **TTS code generation** -- When ``s2s.decode_audio=true``, predicted text
      tokens are fed into the EarTTS model to produce audio codec codes.
   d. **Codec decode** -- When ``s2s.decode_audio=true``, the accumulated codes
      are decoded into a waveform.

4. **State updates** -- The per-stream ``StreamingDecodeState`` is updated
   with model-side decode state such as generated-token history and caches.

5. **Output accumulation** -- Decoded audio and text are appended to the
   per-stream ``S2SStreamingOutput``.


Data Objects
^^^^^^^^^^^^

The streaming pipeline uses four data objects.  Two are **model-level**
(owned by the model wrapper) and two are **pipeline-level** (owned by
``StreamingS2SPipeline``):

.. code-block:: text

    Model level (decode_state.py)
    ─────────────────────────────────────────────────────────────
    StreamingDecodeState          created per stream
      GPU KV caches, token          mutated in-place by infer_one_step()
      workspaces, perception        destroyed at end-of-stream
      cache, codec cache
              │
              │ infer_one_step()
              ▼
    InferenceStepResult           created each step
      predicted tokens, text        returned to the pipeline
      strings, decoded audio        consumed immediately

    Pipeline level (streaming_s2s_pipeline.py, s2s_streaming_output.py)
    ─────────────────────────────────────────────────────────────
    S2SStreamingOutput            created per stream
      accumulates audio chunks      finalized fields (text_with_timestamps,
      and text across steps         audio_filepath, etc.) filled at end-of-stream
                                    returned by run()
              ▲
              │ each step appends
              │
    GenerateStepOutput            created each step
      incremental per-stream        returned by generate_step()
      audio + text                  used by server integrations

**StreamingDecodeState** lives in ``S2SContextManager`` and holds the heavy
GPU tensors (KV caches, perception cache, token workspaces).  It is created
by the model wrapper, mutated in-place by ``infer_one_step()``, and
destroyed at end-of-stream.

**S2SStreamingOutput** lives in the pipeline's ``_state_pool``.  During
streaming it accumulates audio chunks and text parts.  At end-of-stream the
pipeline populates its finalized fields (``text_with_timestamps``,
``raw_text``, ``audio_filepath``, token tensors) and returns the same
object from ``run()``.


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
