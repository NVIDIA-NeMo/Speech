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

**From HuggingFace or NGC:**

.. code-block:: python

    # HuggingFace (prefix with nvidia/)
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

    # NGC (no prefix)
    model = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_transducer_large")

.. note::

    For resuming an unfinished training experiment, use the Experiment Manager with ``resume_if_exists=True`` instead.


Basic Transcription
-------------------

**Python API:**

.. code-block:: python

    outputs = model.transcribe(audio=["file1.wav", "file2.wav"], batch_size=4)
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

**Batch generator (for large datasets):**

.. code-block:: python

    config = model.get_transcribe_config()
    config.batch_size = 32
    for batch_outputs in model.transcribe_generator(audio_files, config):
        # process each batch of results
        ...

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

Canary models require task tokens. Use a manifest or specify task parameters directly:

**Via manifest:**

.. code-block:: python

    from nemo.collections.asr.models import EncDecMultiTaskModel

    canary = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")
    decode_cfg = canary.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary.change_decoding_strategy(decode_cfg)

    results = canary.transcribe("manifest.json", batch_size=16)

Manifest format:

.. code-block:: json

    {"audio_filepath": "/path/to/audio.wav", "duration": null, "taskname": "asr", "source_lang": "en", "target_lang": "en", "pnc": "yes", "answer": "na"}

**Via direct parameters:**

.. code-block:: python

    results = canary.transcribe(
        audio=["audio.wav"],
        batch_size=4,
        task="asr",
        source_lang="en",
        target_lang="en",
        pnc=True,
    )


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
