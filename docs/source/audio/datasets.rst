Datasets
========

The `audio` collection expect the training, validation and tests datasets in either NeMo format or Lhotse format.

NeMo Format
-----------

Each dataset consists of a set of utterances in individual audio files plus a manifest that describes the dataset, with information about one utterance per line (``.json``).
There should be one manifest file per dataset that will be passed in, therefore, if the user wants separate training and validation datasets, they should also have separate manifests. Otherwise, they will be loading validation data with their training data and vice versa.


In most applications, such as speech restoration or enhancement, the model aims to transform the input audio signal into the target audio signal. In this case, each line of the manifest should have the following structure:

.. code-block:: json

  {"input_filepath": "/path/to/input_audio.wav", "target_filepath": "/path/to/target_audio.wav", "duration": 3.147}

The :code:`input_filepath` field should provide either an absolute path to the audio file corresponding to the utterance, or a relative path with respect to the directory containing the manifest.
The :code:`target_filepath` field should provide either an absolute path to the audio file corresponding to the utterance, or a relative path with respect to the directory containing the manifest.
Note that keys for input and target audio can be custom, and can be configured in the model configuration file, as described in :ref:`Configs <audio-configs-nemo-dataset-configuration>`.

Each entry in the manifest (describing one audio file) should be bordered by ``"{"`` and ``"}"`` and must be placed on one line. The ``"key": value`` pairs should be separated by a commas as shown above. NeMo enforces no blank lines in the manifest so that the total number of lines indicates the total number of audio files in the dataset.

Once there is a manifest that describes each audio file in the dataset, assign the ``JSON`` manifest file path in the experiment config file, for example, ``training_ds.manifest_filepath=<path/to/manifest.json>``.

For more information about the individual tarred datasets and the parameters available, including shuffling options, see the corresponding class APIs in the `Datasets <./api.html#Datasets>`_ section.


Lhotse Format
-------------

NeMo supports using `Lhotse`_, a speech data handling library, as a dataloading option.
Lhotse can be easily enabled using `use_lhotse=True` in the dataset configuration file, as described in :ref:`Lhotse dataset configuration <audio-configs-lhotse-dataset-configuration>`.

Lhotse dataloading supports the following types of inputs:

* Lhotse CutSet manifests
    Regular Lhotse CutSet manifests (typically gzipped JSONL).
    See `Lhotse Cuts documentation`_ to learn more about Lhotse data formats.
* Lhotse Shar data
    Lhotse Shar is a data format that also uses tar files for sequential data loading,
    but is designed to be modular (i.e., easily extensible with new data sources and with new feature fields).
    More details can be found in the tutorial notebook: |tutorial_shar|.


.. _Lhotse: https://github.com/lhotse-speech/lhotse
.. _Lhotse Cuts documentation: https://lhotse.readthedocs.io/en/latest/cuts.html
.. |tutorial_shar| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/04-lhotse-shar.ipynb



Converting NeMo manifest to Lhotse
----------------------------------

A dataset with a manifest in NeMo format can be converted to Lhotse format using the provided `conversion script <https://github.com/NVIDIA/NeMo/blob/main/scripts/audio_to_audio/convert_nemo_to_lhotse.py>`_.

.. code:: shell

    CONVERT_SCRIPT=scripts/audio_to_audio/convert_nemo_to_lhotse.py
    INPUT_MANIFEST=/path/to/data/nemo_manifest.json
    OUTPUT_MANIFEST=/path/to/data/lhotse_manifest.jsonl

    python ${CONVERT_SCRIPT} ${INPUT_MANIFEST} ${OUTPUT_MANIFEST} -i input_filepath -t target_filepath


Creating Lhotse shar dataset
----------------------------

First, convert the NeMo manifest to Lhotse format with absolute paths.

.. code:: shell

    CONVERT_SCRIPT=scripts/audio_to_audio/convert_nemo_to_lhotse.py
    INPUT_MANIFEST=/path/to/data/nemo_manifest.json
    OUTPUT_MANIFEST=/path/to/data/lhotse_manifest_absolute_paths.jsonl

    # convert
    python ${CONVERT_SCRIPT} ${INPUT_MANIFEST} ${OUTPUT_MANIFEST} -i input_filepath -t target_filepath --force_absolute_paths


Then, create the Lhotse shar dataset.

.. code:: shell

    LHOTSE_MANIFEST=/path/to/data/lhotse_manifest_absolute_paths.jsonl
    OUTPUT_DIR=/path/to/data/shar

    # create shars, each with 2084 examples, flac audio format
    lhotse shar export --num-jobs 16 --verbose --shard-size 2084 --audio flac ${LHOTSE_MANIFEST} ${OUTPUT_DIR}