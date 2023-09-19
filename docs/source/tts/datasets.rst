Data Preprocessing
==================

NeMo TTS recipes support most of public TTS datasets that consist of multiple languages, multiple emotions, and multiple speakers. Current recipes covered English (en-US), German (de-DE), Spanish (es-ES), and Mandarin Chinese (zh-CN), while the support for many other languages is under planning. NeMo provides corpus-specific data preprocessing scripts, as shown in the directory of `scripts/data_processing/tts/ <https://github.com/NVIDIA/NeMo/tree/stable/scripts/dataset_processing/tts/>`_, to convert common public TTS datasets into the format expected by the dataloaders as defined in `nemo/collections/tts/data/dataset.py <https://github.com/NVIDIA/NeMo/tree/stable/nemo/collections/tts/data/dataset.py>`_. The ``nemo_tts`` collection expects each dataset to consist of a set of utterances in individual audio files plus a ``JSON`` manifest that describes the dataset, with information about one utterance per line. The audio files can be of any format supported by `Pydub <https://github.com/jiaaro/pydub>`_, though we recommend ``WAV`` files as they are the default and have been most thoroughly tested. NeMo supports any original sampling rates of audios, although our scripts of extracting supplementary data and model training all specify the common target sampling rates as either 44100 Hz or 22050 Hz. If the original sampling rate mismatches the target sampling rate, the `feature preprocess <https://github.com/NVIDIA/NeMo/blob/stable/nemo/collections/asr/parts/preprocessing/features.py#L124>`_ can automatically resample the original sampling rate into the target one.

There should be one ``JSON`` manifest file per dataset that will be passed in, therefore, if the user wants separate training and validation datasets, they should also have separate manifests. Otherwise, they will be loading validation data with their training data and vice versa. Each line of the manifest should be in the following format:

.. code-block:: json

    {
      "audio_filepath": "/path/to/audio.wav",
      "text": "the transcription of the utterance",
      "normalized_text": "the normalized transcription of the utterance",
      "speaker": 5,
      "duration": 15.4
    }

where ``"audio_filepath"`` provides an absolute path to the ``.wav`` file corresponding to the utterance so that audio files can be located anywhere without the constraint of being organized in the same directory as the manifest itself; ``"text"`` contains the full transcript (either graphemes or phonemes or their mixer) for the utterance; ``"normalized_text"`` contains normalized ``"text"`` that helps to bypass the normalization steps but it is fully optional; ``"speaker"`` refers to the integer speaker ID; ``"duration"`` describes the duration of the utterance in seconds.

Each entry in the manifest (describing one audio file) should be bordered by ``"{"`` and ``"}"`` and must be placed on one line. The ``"key": value`` pairs should be separated by a commas as shown above. NeMo enforces no blank lines in the manifest so that the total number of lines indicates the total number of audio files in the dataset.

Once there is a manifest that describes each audio file in the dataset, assign the ``JSON`` manifest file path in the experiment config file, for example, ``training_ds.manifest_filepath=<path/to/manifest.json>``.

Following the instructions below on how to run the corpus-specific scripts, you can get started with either directly processing those public datasets, or creating your custom scripts to preprocess your custom datasets scripts.

Public TTS Datasets
------------------------------
This table below summarizes the statistics for a collection of high-quality public datasets used by NeMo TTS. We recommend to start customizing the scripts for your custom datasets that have close sampling rate and number of speakers.

.. csv-table::
   :file: data/datasets.csv
   :align: right
   :header-rows: 1

Corpus-Specific Data Preprocessing
----------------------------------
NeMo implements model-agnostic data preprocessing scripts that wrap up steps of **downloading raw datasets, extracting files, and/or normalizing raw texts, and generating data manifest files**. Most scripts are able to be reused for any datasets with only minor adaptations. Most TTS models work out-of-the-box with the LJSpeech dataset, so it would be straightforward to start adapting your custom script from `LJSpeech script <https://github.com/NVIDIA/NeMo/tree/stable/scripts/dataset_processing/tts/ljspeech/get_data.py>`_. For some models that may require supplementary data for training and validating, such as speech/text alignment prior, pitch, speaker ID, emotion ID, energy, etc, you may need an extra step of **supplementary data extraction** by calling `script/dataset_processing/tts/extract_sup_data.py <https://github.com/NVIDIA/NeMo/tree/stable/scripts/dataset_processing/tts/extract_sup_data.py>`_ . The following sub-sections demonstrate detailed instructions for running data preprocessing scripts.

LJSpeech
~~~~~~~~
* Dataset URL: https://keithito.com/LJ-Speech-Dataset/
* Dataset Processing Script: https://github.com/NVIDIA/NeMo/tree/stable/scripts/dataset_processing/tts/ljspeech/get_data.py
* Command Line Instruction:

.. code-block:: shell-session

    $ python scripts/dataset_processing/tts/ljspeech/get_data.py \
        --data-root <your_local_dataset_root>

    $ python scripts/dataset_processing/tts/extract_sup_data.py \
        --config-path ljspeech/ds_conf \
        --config-name ds_for_fastpitch_align.yaml \
        manifest_filepath=<your_path_to_train_manifest> \
        sup_data_path=<your_path_to_where_to_save_supplementary_data>


LibriTTS
~~~~~~~~
* Dataset URL: https://www.openslr.org/60/
* Dataset Processing Script: https://github.com/NVIDIA/NeMo/tree/stable/scripts/dataset_processing/tts/libritts/get_data.py
* Command Line Instruction:

.. code-block:: console

    $ python scripts/dataset_processing/tts/libritts/get_data.py \
        --data-root <your_local_dataset_root> \
        --data-sets "ALL"
        --num-workers 4

    $ python scripts/dataset_processing/tts/extract_sup_data.py \
        --config-path ljspeech/ds_conf \
        --config-name ds_for_fastpitch_align.yaml \
        manifest_filepath=<your_path_to_train_manifest> \
        sup_data_path=<your_path_to_where_to_save_supplementary_data>

.. note::
    LibriTTS original sampling rate is **24000 Hz**, we re-use LJSpeech's config to down-sample it to **22050 Hz**.


HiFiTTS
~~~~~~~
The texts of this dataset has been normalized already. So there is no extra need to preprocess the data again. But we still need a download script and split it into manifests.

* Dataset URL: http://www.openslr.org/109/
* Dataset Processing Script: TBD
* Command Line Instruction: TBD


Thorsten Müller's German Neutral-TTS Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are two German neutral datasets released by Thorsten Müller for now, 21.02 and 22.10, respectively. Version 22.10 has been recorded with a better recording setup, such as recording chamber and better microphone. So it is advised to train models on the 22.10 version because its audio quality is better and it has a way more natural speech flow and higher character rate per second speech. The two datasets are described below and defined in `scripts/dataset_processing/tts/thorsten_neutral/get_data.py:THORSTEN_NEUTRAL <https://github.com/NVIDIA/NeMo/tree/stable/scripts/dataset_processing/tts/thorsten_neutral/get_data.py#L41-L51>`_.

.. code-block:: python

    # Thorsten Müller published two neural voice datasets, 21.02 and 22.10.
    THORSTEN_NEUTRAL = {
        "21_02": {
            "url": "https://zenodo.org/record/5525342/files/thorsten-neutral_v03.tgz?download=1",
            "dir_name": "thorsten-de_v03",
            "metadata": ["metadata.csv"],
        },
        "22_10": {
            "url": "https://zenodo.org/record/7265581/files/ThorstenVoice-Dataset_2022.10.zip?download=1",
            "dir_name": "ThorstenVoice-Dataset_2022.10",
            "metadata": ["metadata_train.csv", "metadata_dev.csv", "metadata_test.csv"],
        },
    }

* Thorsten Müller's German Datasets repo: https://github.com/thorstenMueller/Thorsten-Voice
* Dataset Processing Script: https://github.com/NVIDIA/NeMo/tree/stable/scripts/dataset_processing/tts/thorsten_neutral/get_data.py
* Command Line Instruction:

.. code-block:: bash

    # Version 22.10
    $ python scripts/dataset_processing/tts/thorsten_neutral/get_data.py \
        --data-root <your_local_dataset_root> \
        --manifests-root <your_local_manifest_root> \
        --data-version "22_10" \
        --val-size 100 \
        --test-size 100 \
        --seed-for-ds-split 100 \
        --normalize-text

    # Version 21.02
    $ python scripts/dataset_processing/tts/thorsten_neutral/get_data.py \
        --data-root <your_local_dataset_root> \
        --manifests-root <your_local_manifest_root> \
        --data-version "21_02" \
        --val-size 100 \
        --test-size 100 \
        --seed-for-ds-split 100 \
        --normalize-text

    # extract pitch and compute pitch normalization params for each version.
    $ python scripts/dataset_processing/tts/extract_sup_data.py \
        --config-path thorsten_neutral/ds_conf \
        --config-name ds_for_fastpitch_align.yaml \
        manifest_filepath=<your_path_to_train_manifest> \
        sup_data_path=<your_path_to_where_to_save_supplementary_data>

HUI Audio Corpus German
~~~~~~~~~~~~~~~~~~~~~~~
* Dataset URL: https://opendata.iisys.de/datasets.html
* Dataset Processing Script: https://github.com/NVIDIA/NeMo/tree/stable/scripts/dataset_processing/tts/hui_acg/get_data.py
* Command Line Instruction:

.. code-block:: bash

    $ python scripts/dataset_processing/tts/hui_acg/get_data.py \
        --data-root <your_local_dataset_root> \
        --manifests-root <your_local_manifest_root> \
        --set-type clean \
        --min-duration 0.1 \
        --max-duration 15 \
        --val-num-utts-per-speaker 1 \
        --test-num-utts-per-speaker 1 \
        --seed-for-ds-split 100

    $ python scripts/dataset_processing/tts/hui_acg/phonemizer.py \
        --json-manifests <your_path_to_train_manifest> <your_path_to_val_manifest> <your_path_to_test_manifest> \
        --preserve-punctuation

    $ python scripts/dataset_processing/tts/extract_sup_data.py \
        --config-path hui_acg/ds_conf \
        --config-name ds_for_fastpitch_align.yaml \
        manifest_filepath=<your_path_to_train_manifest> \
        sup_data_path=<your_path_to_where_to_save_supplementary_data>


SFSpeech Chinese/English Bilingual Speech
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Dataset URL: https://catalog.ngc.nvidia.com/orgs/nvidia/resources/sf_bilingual_speech_zh_en
* Dataset Processing Script: https://github.com/NVIDIA/NeMo/tree/stable/scripts/dataset_processing/tts/sfbilingual/get_data.py
* Command Line Instruction: please refer details in Section 1 (NGC Registry CLI installation), Section 2 (Downloading SFSpeech Dataset), and Section 3 (Creatiung Data Manifests) from https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/FastPitch_ChineseTTS_Training.ipynb. Below code block briefly describes the steps.

.. code-block:: bash

    # [prerequisite] Install and setup 'ngc' cli tool by following document https://docs.ngc.nvidia.com/cli/cmd.html

    $ ngc registry resource download-version "nvidia/sf_bilingual_speech_zh_en:v1"

    $ unzip sf_bilingual_speech_zh_en_vv1/SF_bilingual.zip -d <your_local_dataset_root>

    $ python scripts/dataset_processing/tts/sfbilingual/get_data.py \
        --data-root <your_local_dataset_root>/SF_bilingual \
        --val-size 0.005 \
        --test-size 0.01 \
        --seed-for-ds-split 100

    $ python scripts/dataset_processing/tts/extract_sup_data.py \
        --config-path sfbilingual/ds_conf \
        --config-name ds_for_fastpitch_align.yaml \
        manifest_filepath=<your_path_to_train_manifest> \
        sup_data_path=<your_path_to_where_to_save_supplementary_data>
