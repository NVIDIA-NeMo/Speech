.. _onelogger-integration:

OneLogger Integration
=====================

NeMo Speech can emit training lifecycle and throughput telemetry through
OneLogger. The integration is opt-in and is inactive unless it is explicitly
enabled.

Enabling OneLogger
------------------

The OneLogger Python packages are optional. NeMo Speech does not import them
unless telemetry is explicitly enabled, and runs normally when they are not
installed. Install the runtime packages if they are not already available:

.. code-block:: bash

    pip install "nv_one_logger_core>=2.3.1" "nv_one_logger_training_telemetry>=2.3.1"

Set ``NEMO_ONE_LOGGER_ENABLED`` to a true value before starting a training job:

.. code-block:: bash

    export NEMO_ONE_LOGGER_ENABLED=true
    python <training-script> <training-options>

Training entry points that call :func:`nemo.utils.exp_manager.exp_manager`
attach the callback automatically. In distributed jobs, only global rank zero
exports telemetry. In a single-process job, the same environment variable
enables the local process.

Unset ``NEMO_ONE_LOGGER_ENABLED``, or set it to ``false``, to leave the
integration disabled. OneLogger exporter destinations and credentials use the
standard configuration supported by the installed OneLogger packages. If
telemetry is enabled but its packages cannot be imported, NeMo Speech logs a
warning, disables the integration, and continues normally.

Reporting cadence
-----------------

Throughput is reported every 100 training batches by default. If
``trainer.log_every_n_steps`` is larger, that value is used instead. Override
the cadence with a positive batch count:

.. code-block:: bash

    export NEMO_ONE_LOGGER_THROUGHPUT_INTERVAL=250

The batch count is a minimum reporting interval. With gradient accumulation, a
window waits for the next optimizer-step boundary so per-step metrics never mix
a partial accumulated batch. A reporting window is also closed before validation
and checkpointing so those operations are not included in training throughput.
When such a boundary interrupts gradient accumulation, totals and per-second
rates are still emitted but per-step means are omitted. GPU timing and
aggregation are asynchronous during training.

Reported telemetry
------------------

Lifecycle spans use the ``nemo_speech`` namespace and cover model, data loader,
and optimizer initialization; checkpoint load and save; training; and
validation. Checkpoint save success and failure are emitted as explicit events.

Throughput is reported as the ``nemo_speech.throughput`` event. Every event
contains:

* ``policy``: the model-specific measurement policy;
* ``rank`` and ``scope``: the rank-local origin of the measurement;
* ``global_step``;
* ``window_batches``, ``window_optimizer_steps``, and ``window_seconds``;
* ``examples``, ``examples_per_second``, and ``mean_batch_size`` when the policy
  can determine the example count for every batch;
* each available work-unit total;
* a corresponding ``<unit>_per_second`` rate; and
* a corresponding ``<unit>_per_step`` mean when the window contains a
  completed optimizer step and ends at an optimizer-step boundary.

The available work units depend on the model and batch schema:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Model family
     - Input work
     - Output or target work
   * - ASR
     - ``input_audio_seconds``
     - ``target_text_tokens``
   * - TTS
     - ``input_text_tokens``
     - ``output_audio_seconds``
   * - Audio codec
     - ``input_audio_seconds``
     -
   * - Diarization
     - ``input_audio_seconds``
     -
   * - Audio-to-audio
     - ``input_audio_seconds``
     -
   * - SALM family
     - ``input_audio_seconds``
     - ``multimodal_tokens`` (text and audio tokens in the post-insertion model sequence)
   * - DuplexSTT
     - ``input_audio_seconds``
     - ``text_tokens``
   * - Speech-to-speech
     - ``input_audio_seconds``
     - ``output_audio_seconds``

Measurements use the length tensors from the actual dynamic batch. Every SALM
variant reports the exact number of non-padding tokens in the mixed-modality
model sequence after audio tokens have been inserted; packed batches use their
pre-parallelism sequence metadata. NeMo Speech does not report an individual
microbatch size, a global batch size, or a static sequence length.
``mean_batch_size`` is the total rank-local examples divided by completed
optimizer steps, so it includes gradient accumulation. Work-unit means use the
same optimizer-step denominator; no ``_per_example`` metrics are emitted. Audio
durations are omitted when a trustworthy sample rate or waveform length is
unavailable, rather than estimated from unrelated configuration.

Throughput events are rank-local measurements, not distributed global
estimates. If a model has no registered policy or its batch does not expose a
safe measurement, no throughput event is emitted for that batch. Telemetry
setup and reporting failures disable telemetry without stopping training.
