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

Split training samples into duration-based buckets for up to 2x speedup by reducing padding.
Pass tarred datasets as a list of lists to enable bucketing. Use ``bucket_batch_size`` for adaptive batch sizes per bucket.

For advanced dynamic bucketing with Lhotse, see :doc:`Lhotse Dataloading </dataloaders>`.


Lhotse Dataloading
------------------

NeMo supports `Lhotse <https://github.com/lhotse-speech/lhotse>`_ for advanced dataloading with dynamic batch sizes, dynamic bucketing, OOMptimizer, and multi-dataset configuration.

See :doc:`Lhotse Dataloading </dataloaders>` for full documentation.
