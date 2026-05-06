.. _asr-inference:

=========
Inference
=========

This page covers how to load ASR models and run inference in NeMo.


Loading Checkpoints
-------------------

**From a local file:**

.. code-block:: python

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.restore_from("path/to/checkpoint.nemo")

**From HuggingFace:**

.. code-block:: python

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")


Basic Transcription
-------------------

**Python API:**

.. code-block:: python

    outputs = model.transcribe(audio=["file1.wav", "file2.wav"], batch_size=2)
    print(outputs[0].text)

The ``audio`` argument accepts file paths (strings), lists of paths, numpy arrays, or PyTorch tensors.
Audio must be 16 kHz mono-channel.

**Numpy/Tensor inputs:**

.. code-block:: python

    import soundfile as sf
    audio, sr = sf.read("audio.wav", dtype='float32')
    outputs = model.transcribe([audio], batch_size=1)

**Command line:**

.. code-block:: bash

    python examples/asr/transcribe_speech.py \
        pretrained_name="nvidia/parakeet-tdt-0.6b-v2" \
        audio_dir=<path_to_audio_dir>

**Batch generator (for incremental processing):**

``model.transcribe()`` already handles large file lists internally via batching. Use ``transcribe_generator`` only when you need to process results incrementally (e.g., writing to disk batch-by-batch to avoid holding all outputs in memory):

.. code-block:: python

    config = model.get_transcribe_config()
    config.batch_size = 32
    for batch_outputs in model.transcribe_generator(audio_files, override_config=config):
        # write batch results to disk immediately
        ...

``TranscribeConfig`` fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``batch_size``
     - 4
     - Batch size for inference. Larger = better throughput but more memory.
   * - ``return_hypotheses``
     - False
     - If True, return ``Hypothesis`` objects (with timestamps, scores) instead of plain strings.
   * - ``use_lhotse``
     - True
     - Use Lhotse dataloading for inference.
   * - ``num_workers``
     - None
     - Number of DataLoader workers.
   * - ``channel_selector``
     - None
     - Select channel(s) from multi-channel audio. ``'average'`` to average across channels.
   * - ``timestamps``
     - None
     - If True, return word/segment timestamps (requires ``return_hypotheses=True``).
   * - ``augmentor``
     - None
     - Optional augmentation config to apply during transcription.
   * - ``verbose``
     - True
     - Show progress bar.

For multi-task models (Canary), ``MultiTaskTranscriptionConfig`` extends this with ``prompt``, ``text_field``, ``lang_field``, and ``enable_chunking``.

**Alignments:**

.. code-block:: python

    hyps = model.transcribe(audio=["file.wav"], return_hypotheses=True)
    alignments = hyps[0].alignments


Timestamps
----------

Obtain word, segment, or character timestamps with Parakeet models (CTC/RNNT/TDT):

**Simple usage:**

.. code-block:: python

    hypotheses = model.transcribe(["audio.wav"], timestamps=True)

    for stamp in hypotheses[0].timestamp['word']:
        print(f"{stamp['start']}s - {stamp['end']}s : {stamp['word']}")

    for stamp in hypotheses[0].timestamp['segment']:
        print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")

**Advanced configuration:**

For Transducer decoding strategies (greedy, beam, TSD, ALSD, mAES) and their parameters, see the Transducer Decoding section in :doc:`Configs <./configs>`.
For CTC and AED decoding classes, see the :doc:`API reference <./api>`.
For decoding customization (confidence, CUDA graphs, language models, word boosting), see :doc:`ASR Language Modeling and Customization <./asr_language_modeling_and_customization>`.

.. code-block:: python

    from omegaconf import open_dict

    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
        decoding_cfg.segment_seperators = [".", "?", "!"]
        decoding_cfg.word_seperator = " "
        model.change_decoding_strategy(decoding_cfg)

    hypotheses = model.transcribe(["audio.wav"], return_hypotheses=True)
    timestamp_dict = hypotheses[0].timestamp

    time_stride = 8 * model.cfg.preprocessor.window_stride
    for stamp in timestamp_dict['word']:
        start = stamp['start_offset'] * time_stride
        end = stamp['end_offset'] * time_stride
        word = stamp['char'] if 'char' in stamp else stamp['word']
        print(f"{start:0.2f} - {end:0.2f} : {word}")


Long Audio Inference
--------------------

For audio longer than what fits in memory (especially with Conformer's quadratic attention):

**Buffered / chunked inference:**

Divide audio into overlapping chunks and merge outputs. Scripts are in
`examples/asr/asr_chunked_inference <https://github.com/NVIDIA/NeMo/tree/main/examples/asr/asr_chunked_inference>`_.

**Local attention (recommended for Fast Conformer):**

Switch to Longformer-style local+global attention for linear-cost inference on audio >1 hour:

.. code-block:: python

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-ctc-1.1b")
    model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[128, 128]
    )

Or via CLI:

.. code-block:: bash

    python examples/asr/speech_to_text_eval.py \
        (...other parameters...) \
        ++model_change.conformer.self_attention_model="rel_pos_local_attn" \
        ++model_change.conformer.att_context_size=[128, 128]

**Subsampling memory optimization:**

For very long files where even the subsampling module runs out of memory:

.. code-block:: python

    model.change_subsampling_conv_chunking_factor(1)  # auto-chunk subsampling


Multi-task Inference (Canary)
-----------------------------

Canary models use prompt slots to control transcription behavior.

**Via manifest:**

.. code-block:: python

    from nemo.collections.asr.models import EncDecMultiTaskModel

    canary = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")
    decode_cfg = canary.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary.change_decoding_strategy(decode_cfg)

    results = canary.transcribe("manifest.json", batch_size=16)

For the manifest format required by Canary models, see :ref:`Canary Manifest Format <canary-manifest-format>`.

**Via direct parameters:**

.. code-block:: python

    results = canary.transcribe(
        audio=["audio.wav"],
        batch_size=4,
        source_lang="en",
        target_lang="en",
        pnc=True,
    )


.. _asr-enforcing-single-language:

Enforcing a Single Language
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multilingual Canary models, you can enforce a specific output language by explicitly setting ``source_lang`` and
``target_lang``. When both are set to the same language, the model will transcribe in that language only:

.. code-block:: python

    results = canary.transcribe(
        audio=["audio.wav"],
        source_lang="de",
        target_lang="de",
    )

This prevents phonetic drift where the model may switch languages mid-utterance.


Streaming Inference
-------------------

NeMo provides a unified streaming-first Pipeline API for real-time ASR under ``nemo.collections.asr.inference``.
It supports buffered CTC/RNNT/TDT pipelines (overlapping chunks with any offline model) and cache-aware CTC/RNNT pipelines (processes each frame once using cached activations).

See the `Streaming ASR Pipelines tutorial <https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/Streaming_ASR_Pipelines.ipynb>`_ for a comprehensive walkthrough covering buffered and cache-aware pipelines, per-stream options, EoU detection, word timestamps, per-stream biasing, ITN, and speech translation.

See :ref:`cache-aware streaming conformer` for model architecture details.


Apple MPS Support
-----------------

Inference on Apple M-Series GPUs is supported with PyTorch 2.0+:

.. code-block:: bash

    PYTORCH_ENABLE_MPS_FALLBACK=1 python examples/asr/speech_to_text_eval.py \
        (...other parameters...) \
        allow_mps=true


Execution Flow
--------------

When writing custom inference scripts, follow the execution flow diagram at the
`ASR examples README <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/README.md>`_.
