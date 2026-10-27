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
`examples/asr/asr_chunked_inference <https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/asr/asr_chunked_inference>`_.

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

See the `Streaming ASR Pipelines tutorial <https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/asr/Streaming_ASR_Pipelines.ipynb>`_ for a comprehensive walkthrough covering buffered and cache-aware pipelines, per-stream options, EoU detection, word timestamps, per-stream biasing, ITN, and speech translation.

See :ref:`cache-aware streaming conformer` for model architecture details.


Prompt-conditioned Streaming (Multilingual)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prompt-conditioned cache-aware streaming models
(:class:`~nemo.collections.asr.models.EncDecRNNTBPEModelWithPrompt` and
:class:`~nemo.collections.asr.models.EncDecHybridRNNTCTCBPEModelWithPrompt`) select a language
prompt at inference time via ``target_lang``. The cache-aware simulation script accepts the
flag directly:

.. code-block:: bash

    python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
        model_path=<path_to_nemo_checkpoint> \
        dataset_manifest=<path_to_manifest> \
        target_lang=en-US \
        decoder_type=rnnt \
        strip_lang_tags=true

``target_lang`` must be a key of the model's ``prompt_dictionary``. A bare language code such as
``en`` is also accepted when it matches exactly one locale key (for example ``en-US``). An
unsupported language raises an error. ``target_lang=auto`` (the default when ``target_lang`` is
unset) selects the language-agnostic prompt; quality on languages the model was not trained on is
not guaranteed. This script applies one prompt to every sample. For per-utterance languages, use
``model.transcribe()``, which reads each manifest entry's ``lang`` field when ``target_lang`` is
unset and uses ``auto`` for entries without one.
Setting ``strip_lang_tags=true`` removes ``<xx-XX>`` language tags from the decoded text
(pattern is customizable via ``lang_tag_pattern``).

In Python, call ``set_inference_prompt`` once before decoding:

.. code-block:: python

    model = nemo_asr.models.EncDecRNNTBPEModelWithPrompt.restore_from("path/to/model.nemo")
    model.set_inference_prompt("en-US")
    # ... subsequent conformer_stream_step / streaming pipeline calls use this prompt ...

See :ref:`RNN-T with Prompt Conditioning Configuration <RNNT-Prompt_model__Config>` for the
full model config reference.


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
`ASR examples README <https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/README.md>`_.


Offline TensorRT encoder
------------------------

Run an offline ASR encoder in TensorRT while NeMo handles preprocessing and RNNT/TDT decoding in PyTorch.
Install the optional build/runtime dependencies with pip or uv:

.. code-block:: bash

    pip install 'nemo-toolkit[asr,trt]'
    # Or:
    uv pip install 'nemo-toolkit[asr,trt]'

The ``trt`` extra targets Linux with TensorRT 10.13.3 or later in the 10.x series, using its CUDA 13 package, plus Polygraphy.
ONNX is already a base NeMo dependency.
Use a CUDA 13-compatible NVIDIA driver and PyTorch installation; see the `TensorRT installation guide <https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html>`_.
Torch-TensorRT and Triton Inference Server are separate optional tools.

Export the encoder to ONNX, then build an engine:

.. code-block:: bash

    python examples/asr/export/build_trt_encoder.py \
        --onnx encoder.onnx --engine encoder.plan --features 128

Set ``--features`` to the model's feature count.
Defaults cover batches 1–16 and 8–4000 padded feature frames; the ONNX export must support those shapes and positional-cache limits.
Use ``--help`` for build options.
FP32 with TF32 disabled is the default; check recognition accuracy when enabling ``--fp16``.
Build for the GPU and TensorRT/CUDA versions you will deploy.

Transcribe mono audio at the checkpoint's sample rate, using the same checkpoint as the export:

.. code-block:: bash

    python examples/asr/export/transducer/infer_transducer_trt.py \
        --model model.nemo --engine encoder.plan \
        --audio a.wav b.wav --output predictions.jsonl \
        --batch-size 2 --compare-native

``--compare-native`` adds the PyTorch encoder's transcript to each output record.
This is an accuracy example: it retains the native encoder, uses FP32 preprocessing/decoding and disables decoder CUDA graphs.
Prompt-conditioned models are rejected, because the example passes the encoder output to the decoder without the prompt projection.
A cache-aware streaming encoder runs offline, over whole utterances without caches, and the example warns about it.

For integration into your own loop, see `TensorRTEncoder <https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/parts/submodules/tensorrt_encoder.py>`_ for tensor and stream requirements.
Streaming encoder caches, training and CUDA graph capture are not supported.


Waveform preprocessing engine
-----------------------------

Build a standalone log-mel preprocessor from the same checkpoint as the encoder:

.. code-block:: bash

    python examples/asr/export/build_trt_preprocessor.py \
        --model model.nemo --onnx preprocessor.onnx --engine preprocessor.plan

Building requires NeMo, PyTorch, TensorRT and Polygraphy.
The engine uses the checkpoint's evaluation settings, FP32 with TF32 disabled, and no dithering or training augmentation.
Dynamic export supports per-feature or no normalization with ``frame_splicing=1``.
Defaults cover batches 1–16 and 800–640000 padded waveform samples; use ``--help`` to adjust them and ensure the resulting features fit the encoder.

Prepare mono audio at the checkpoint's sample rate.
Store zero-padded float32 ``input_signal`` arrays of shape ``[batch, samples]`` and int64 ``length`` arrays of shape ``[batch]`` in an NPZ file.

Run preprocessing with NumPy, TensorRT and cuda-python, without PyTorch:

.. code-block:: bash

    python examples/asr/export/run_trt_preprocessor.py \
        --engine preprocessor.plan --input waveforms.npz --output features.npz

The outputs, ``processed_signal`` and ``processed_length``, contain features and lengths in feature frames for the encoder.
The example reloads the engine and copies CPU/GPU buffers on each call; serving should retain them and pass GPU outputs directly to the encoder.

Near-silent audio can amplify FP32 rounding during normalization, including between native CPU and CUDA execution.
Validate recognition accuracy; identical features or transcripts are not guaranteed.

Prediction network and joint engine
-----------------------------------

Build a TensorRT engine for one RNNT/TDT decoding step from NeMo's combined prediction/joint ONNX export:

.. code-block:: bash

    python examples/asr/export/build_trt_decoder_joint.py \
        --onnx decoder_joint-model.onnx --engine decoder_joint.plan

The engine takes an encoder frame, the previous label and recurrent states, then returns joint scores, prediction lengths and updated states.
Defaults are batches 1–16 and FP32 with TF32 disabled; see ``--help`` for options.
Check successive decoding steps before enabling FP16.

The caller must handle blanks, state updates, TDT durations and termination while preserving cached predictions and projections.
The encoder transcription example still uses NeMo's PyTorch decoder.


Complete TensorRT TDT serving
-----------------------------

The `waveform serving example <https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/export/transducer/README_trt_asr.md>`__ packages preprocessing, the encoder and TDT networks for one Triton request.
It retains NeMo's cached decoding loop and CUDA graphs; PyTorch remains a runtime dependency.
