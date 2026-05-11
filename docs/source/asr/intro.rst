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


Key Features
------------

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
