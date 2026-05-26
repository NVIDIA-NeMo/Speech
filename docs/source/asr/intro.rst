Automatic Speech Recognition (ASR)
==================================

Automatic Speech Recognition (ASR), also known as Speech To Text (STT), refers to the problem of automatically transcribing spoken language.
NeMo provides open-sourced pretrained models in 25+ languages. Browse the full list in :doc:`ASR Model Checkpoints <./asr_checkpoints>`.


Quick Start
-----------

After :ref:`installing NeMo<installation>`, transcribe an audio file in 3 lines:

.. code-block:: python

    import nemo.collections.asr as nemo_asr
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    transcript = asr_model.transcribe(["path/to/audio_file.wav"])[0].text

Timestamps
^^^^^^^^^^

Obtain word, segment, or character timestamps with any Parakeet model (CTC/RNNT/TDT):

.. code-block:: python

    hypotheses = asr_model.transcribe(["path/to/audio_file.wav"], timestamps=True)
    for stamp in hypotheses[0].timestamp['word']:
        print(f"{stamp['start']}s - {stamp['end']}s : {stamp['word']}")

See :doc:`Inference <./inference>` for full details on timestamps, long audio, streaming, and multi-task inference.


NeMo supports GPU-accelerated language model fusion for all major ASR model types, including CTC, RNN-T, TDT, and AED. 
Customization is available during both greedy and beam decoding. After :ref:`training <train-ngram-lm>` an n-gram LM, you can apply it using the
`speech_to_text_eval.py <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/speech_to_text_eval.py>`_ script.

**To configure the evaluation:**

1. Select the pretrained model:
   Use the `pretrained_name` option or provide a local path using `model_path`.

2. Set up the N-gram language model:
   Provide the path to the NGPU-LM model with `ngram_lm_model`, and set LM weight with `ngram_lm_alpha`.

3. Choose the decoding strategy:

   - CTC models: `greedy_batch` or `beam_batch`
   - RNN-T models: `greedy_batch`, `malsd_batch`, or `maes_batch`
   - TDT models: `greedy_batch` or `malsd_batch`
   - AED models: `beam` (set `beam_size=1` for greedy decoding)

4. Run the evaluation script.

**Example: CTC Greedy Decoding with NGPU-LM**

.. code-block:: bash

    python examples/asr/speech_to_text_eval.py \
        pretrained_name=nvidia/parakeet-ctc-1.1b \
        amp=false \
        amp_dtype=bfloat16 \
        matmul_precision=high \
        compute_dtype=bfloat16 \
        presort_manifest=true \
        cuda=0 \
        batch_size=32 \
        dataset_manifest=<path to the evaluation JSON manifest file> \
        ctc_decoding.greedy.ngram_lm_model=<path to the .nemo/.ARPA file of the NGPU-LM model> \
        ctc_decoding.greedy.ngram_lm_alpha=0.2 \
        ctc_decoding.greedy.allow_cuda_graphs=True \
        ctc_decoding.strategy="greedy_batch"

**Example: RNN-T Beam Search with NGPU-LM**

.. code-block:: bash

    python examples/asr/speech_to_text_eval.py \
        pretrained_name=nvidia/parakeet-rnnt-1.1b \
        amp=false \
        amp_dtype=bfloat16 \
        matmul_precision=high \
        compute_dtype=bfloat16 \
        presort_manifest=true \
        cuda=0 \
        batch_size=16 \
        dataset_manifest=<path to the evaluation JSON manifest file> \
        rnnt_decoding.beam.ngram_lm_model=<path to the .nemo/.ARPA file of the NGPU-LM model> \
        rnnt_decoding.beam.ngram_lm_alpha=0.3 \
        rnnt_decoding.beam.beam_size=10 \
        rnnt_decoding.strategy="malsd_batch"

See detailed documentation here: :ref:`asr_language_modeling_and_customization`.

Transcribe long audio files (chunking mode)
-------------------------------------------
You can transcribe long audio files by using the **chunking mode**: set `enable_chunking=True` in the `transcribe` method.
Chunking is available only when you pass a single audio file or set `batch_size=1`; it is not used when the input is a pre-built DataLoader.

.. code-block:: python

    import nemo.collections.asr as nemo_asr
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    transcript = asr_model.transcribe(["path/to/audio_file.wav"], enable_chunking=True)[0].text

**Workflow**

Long audio is split into overlapping segments (chunks) of configurable duration. For each chunk the model runs normally; the per-chunk hypotheses are then merged into a single transcript. This keeps memory and compute manageable while still producing one continuous result. Consecutive chunks overlap by about 1 second so that words spanning chunk boundaries can be merged correctly in the final text.

**chunk_range**

`chunk_range` is a pair ``[min_seconds, max_seconds]`` that defines the allowed duration of each chunk.

* **Defaults:** Parakeet-style models use ``[240, 300]`` seconds; Canary-style (e.g. multi-task AED) models use ``[30, 40]`` seconds.
* **Lhotse** Chunk duration is fixed in the Lhotse-based dataloaders used for file input: `LhotseSpeechToTextBpeDataset` and `PromptedAudioToTextLhotseDataset`.
* **Tensors or numpy arrays:** When audio is passed as a tensor or numpy array, `chunk_range` is taken from :class:`TranscribeConfig` for `TranscriptionTensorDataset` (e.g. via `override_config` or the default ``[240, 300]``).

Use real-time transcription
---------------------------

It is possible to use NeMo to transcribe speech in real-time. We provide tutorial notebooks for `Cache Aware Streaming <https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb>`_ and `Buffered Streaming <https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Buffered_Streaming.ipynb>`_.

Try different ASR models
------------------------

NeMo offers a variety of open-sourced pretrained ASR models that vary by model architecture:

* **encoder architecture** (FastConformer, Conformer, Citrinet, etc.),
* **decoder architecture** (Transducer, CTC & hybrid of the two),
* **size** of the model (small, medium, large, etc.).

The pretrained models also vary by:

* **language** (English, Spanish, etc., including some **multilingual** and **code-switching** models),
* whether the output text contains **punctuation & capitalization** or not.

The NeMo ASR checkpoints can be found on `HuggingFace <https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia>`_, or on `NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_. All models released by the NeMo team can be found on NGC, and some of those are also available on HuggingFace.

All NeMo ASR checkpoints open-sourced by the NeMo team follow the following naming convention:
``stt_{language}_{encoder name}_{decoder name}_{model size}{_optional descriptor}``.

You can load the checkpoints automatically using the ``ASRModel.from_pretrained()`` class method, for example:

.. code-block:: python

    import nemo.collections.asr as nemo_asr
    # model will be fetched from NGC
    asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_transducer_large")
    # if model name is prepended with "nvidia/", the model will be fetched from huggingface
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_transducer_large")
    # you can also load open-sourced NeMo models released by other HF users using:
    # asr_model = nemo_asr.models.ASRModel.from_pretrained("<HF username>/<model name>")

**50+ Pretrained Models** — NeMo offers open-source checkpoints across 14+ languages, available on `HuggingFace <https://huggingface.co/nvidia>`__ and `NGC <https://catalog.ngc.nvidia.com/models?query=nemo>`__. Browse the full list in :doc:`All Checkpoints <./asr_checkpoints>`.

**Timestamps** — Character, word, and segment-level timestamps are supported for all Parakeet models with CTC, RNNT, and TDT decoders.

**Streaming** — Real-time transcription with cache-aware streaming Conformer models, supporting configurable latency-accuracy tradeoffs. See :ref:`cache-aware streaming conformer`.

**Multi-task (Canary)** — The Canary model family supports ASR and speech translation (AST) across 25 European languages, with built-in punctuation and capitalization. See :doc:`Featured Models <./featured_models>`.

**Language Modeling** — GPU-accelerated n-gram LM fusion (NGPU-LM) for CTC, RNN-T, TDT, and AED models improves transcription accuracy without retraining. See :ref:`asr_language_modeling_and_customization`.

**Word Boosting** — Bias decoding toward specific words or phrases without retraining. Supports global and per-stream (per-utterance) boosting. See :ref:`word_boosting`.

**Multitalker** — Streaming multi-speaker ASR with speaker kernel injection handles overlapping speech in real time. See `Multitalker Parakeet <https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1>`__.

**Long Audio** — Inference on audio over 1 hour via local attention or buffered chunked processing.

**Decoder Types** — NeMo supports CTC, RNN-T, TDT, AED, and Hybrid decoders. For a comparison of decoder types, see :ref:`asr_language_modeling_and_customization`.


ASR Customization
-----------------

NeMo supports decoding-time customization techniques to improve accuracy without retraining, including GPU-accelerated language model fusion (NGPU-LM), neural rescoring, and word boosting (GPU-PB, per-stream, Flashlight, CTC-WS). See :ref:`asr_language_modeling_and_customization` for full documentation.


Further Reading
---------------

.. toctree::
   :maxdepth: 1

   featured_models
   asr_checkpoints
   inference
   fine_tuning
   datasets
   asr_language_modeling_and_customization
   configs
   api
