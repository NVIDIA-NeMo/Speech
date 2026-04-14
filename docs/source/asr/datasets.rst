Datasets
========

NeMo ASR models expect data as a set of audio files plus a manifest file describing each utterance.

.. _section-with-manifest-format-explanation:

Manifest Format
---------------

Each line of the manifest is a JSON object:

.. code-block:: json

  {"audio_filepath": "/path/to/audio.wav", "text": "the transcription of the utterance", "duration": 23.147}

* ``audio_filepath`` — absolute or relative path to the audio file (WAV recommended)
* ``text`` — the transcript
* ``duration`` — duration in seconds

There should be one manifest per dataset split (train, validation, test). Pass it via ``training_ds.manifest_filepath=<path>``.


Canary Manifest Format
~~~~~~~~~~~~~~~~~~~~~~

Canary multi-task models require additional manifest keys to control transcription, translation, punctuation, and other behaviors.
The required and optional keys differ between Canary v1 and Canary Flash / v2.

**Canary v1** (e.g., ``canary-1b``):

.. code-block:: json

  {"audio_filepath": "audio.wav", "text": "hello world", "duration": 3.5, "source_lang": "en", "task": "asr", "target_lang": "en", "pnc": "yes"}

.. list-table::
  :header-rows: 1

  * - Key
    - Required
    - Description
  * - ``source_lang``
    - Yes
    - Input audio language (ISO code, e.g. ``en``, ``de``, ``es``)
  * - ``target_lang``
    - Yes
    - Output transcription language
  * - ``task``
    - Yes
    - ``"asr"`` (transcribe) or ``"ast"`` (translate)
  * - ``pnc``
    - Yes
    - ``"yes"`` or ``"no"`` — enable punctuation and capitalization

**Canary Flash / v2** (e.g., ``canary-1b-flash``, ``canary-1b-v2``):

The ``task`` field has been removed; the model infers ASR vs translation from the language pair.
Additional optional keys control features like timestamps, ITN, and diarization.

.. code-block:: json

  {"audio_filepath": "audio.wav", "text": "hello world", "duration": 3.5, "source_lang": "en", "target_lang": "en", "pnc": "yes"}

.. list-table::
  :header-rows: 1

  * - Key
    - Required
    - Description
  * - ``source_lang``
    - Yes
    - Input audio language (ISO code)
  * - ``target_lang``
    - Yes
    - Output transcription language. Same as ``source_lang`` for ASR; different for translation.
  * - ``pnc``
    - No (default: ``"yes"``)
    - ``"yes"`` or ``"no"`` — punctuation and capitalization
  * - ``itn``
    - No (default: ``"no"``)
    - ``"yes"`` or ``"no"`` — inverse text normalization
  * - ``timestamp``
    - No (default: ``"no"``)
    - ``"yes"`` or ``"no"`` — predict word-level timestamps
  * - ``diarize``
    - No (default: ``"no"``)
    - ``"yes"`` or ``"no"`` — diarize speech
  * - ``decodercontext``
    - No (default: ``""``)
    - Previous transcript or other context to bias predictions
  * - ``emotion``
    - No (default: ``"undefined"``)
    - Speaker emotion hint (``"neutral"``, ``"angry"``, ``"happy"``, ``"sad"``, ``"undefined"``)

During fine-tuning, these keys are read from the manifest and encoded as prompt tokens.
During inference, they can be provided either in the manifest or as arguments to ``model.transcribe()``.

.. _Tarred_Datasets:

Tarred Datasets
---------------

For cluster training with distributed file systems, tar your audio files to avoid reading many small files.
Use ``is_tarred: true`` in the config and provide tarball paths via ``tarred_audio_filepaths``.

NeMo uses `WebDataset <https://github.com/tmbdev/webdataset>`_ for tarred data.

**Convert to tarred format:**

.. code-block:: bash

  python scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
    --manifest_path=<manifest> \
    --target_dir=<output_dir> \
    --num_shards=64 \
    --shuffle --shuffle_seed=0

.. _Bucketing_Datasets:

Bucketing
---------

The script ``scripts/speech_recognition/convert_to_tarred_audio_dataset.py`` offers a ``--buckets_num`` option that enables
static bucketing by sorting data into separate duration-based buckets at pre-processing time.
This approach is deprecated in favor of :ref:`dynamic bucketing <lhotse-dataloading>` enabled with Lhotse, which doesn't require special pre-processing.

If you do wish to proceed with static bucketing, pass the tarred datasets as a list of lists in your training config:

.. code-block:: yaml

  train_ds:
    manifest_filepath: [[bucket1/manifest.json], [bucket2/manifest.json], ...]
    tarred_audio_filepaths: [[bucket1/audio__OP_0..63_CL_.tar], [bucket2/audio__OP_0..63_CL_.tar], ...]
    bucketing_batch_size: null  # set to a list of ints for adaptive batch sizes per bucket


Lhotse Dataloading
------------------

NeMo supports `Lhotse <https://github.com/lhotse-speech/lhotse>`_ for advanced dataloading with dynamic batch sizes, dynamic bucketing, OOMptimizer, and multi-dataset configuration.

See :doc:`Lhotse Dataloading </dataloaders>` for full documentation.
