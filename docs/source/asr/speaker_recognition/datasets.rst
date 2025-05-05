Datasets
========

.. _HI-MIA:

HI-MIA
--------

Run the script to download and process ``hi-mia`` dataset in order to generate files in the supported format of  ``nemo_asr``. You should set the data folder of 
hi-mia using ``--data_root``. These scripts are present in ``<nemo_root>/scripts``

.. code-block:: bash

    python get_hi-mia_data.py --data_root=<data directory> 

After download and conversion, your `data` folder should contain directories with following set of files as:

* `data/<set>/train.json`
* `data/<set>/dev.json` 
* `data/<set>/{set}_all.json` 
* `data/<set>/utt2spk`


All-other Datasets
------------------

These methods can be applied to any dataset to get similar training or inference manifest files.

`filelist_to_manifest.py` script in `$<NeMo_root>/scripts/speaker_tasks/` folder generates manifest file from a text file containing paths to audio files. 

sample `filelist.txt` file contents:

.. code-block:: bash 

    /data/datasets/voxceleb/data/dev/aac_wav/id00179/Q3G6nMr1ji0/00086.wav
    /data/datasets/voxceleb/data/dev/aac_wav/id00806/VjpQLxHQQe4/00302.wav
    /data/datasets/voxceleb/data/dev/aac_wav/id01510/k2tzXQXvNPU/00132.wav

This list file is used to generate manifest file. This script has optional arguments to split the whole manifest file in to train and dev and also segment audio files to smaller segments for robust training (for testing, we don't need to create segments for each utterance).

sample usage:

.. code-block:: bash 
    
    python filelist_to_manifest.py --filelist=filelist.txt --id=-3 --out=speaker_manifest.json 

This would create a manifest containing file contents as shown below:
.. code-block:: json
    
    {"audio_filepath": "/data/datasets/voxceleb/data/dev/aac_wav/id00179/Q3G6nMr1ji0/00086.wav", "offset": 0, "duration": 4.16, "label": "id00179"}
    {"audio_filepath": "/data/datasets/voxceleb/data/dev/aac_wav/id00806/VjpQLxHQQe4/00302.wav", "offset": 0, "duration": 12.288, "label": "id00806"}
    {"audio_filepath": "/data/datasets/voxceleb/data/dev/aac_wav/id01510/k2tzXQXvNPU/00132.wav", "offset": 0, "duration": 4.608, "label": "id01510"}

For other optional arguments like splitting manifest file to train and dev and for creating segements from each utterance refer to the arguments 
described in the script.

Tarred Datasets
---------------

Similarly to ASR, you can tar your audio files and use ASR Dataset class ``TarredAudioToSpeechLabelDataset`` (corresponding to the ``AudioToSpeechLabelDataset``) for this case.

If you want to use tarred dataset, have a look at the ASR :ref:`Tarred Datasets <asr-datasets-tarred>` section.
