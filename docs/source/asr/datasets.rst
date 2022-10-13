Datasets
========

NeMo has scripts to convert several common ASR datasets into the format expected by the ``nemo_asr`` collection. You can get started
with those datasets by following the instructions to run those scripts in the section appropriate to each dataset below.

If the user has their own data and want to preprocess it to use with NeMo ASR models, refer to the `Preparing Custom ASR Data`_ section.

If the user already has a dataset that you want to convert to a tarred format, refer to the `Tarred Datasets`_ section.

.. _LibriSpeech_dataset:

LibriSpeech
-----------

Run the following scripts to download the LibriSpeech data and convert it into the format expected by `nemo_asr`. At least 250GB free
space is required.

.. code-block:: bash

    # install sox
    sudo apt-get install sox
    mkdir data
    python get_librispeech_data.py --data_root=data --data_set=ALL

After this, the ``data`` folder should contain wav files and ``.json`` manifests for NeMo ASR datalayer.

Each line is a training example. ``audio_filepath`` contains the path to the wav file, ``duration`` is the duration in seconds, and ``text`` is the transcript:

.. code-block:: json

    {"audio_filepath": "<absolute_path_to>/1355-39947-0000.wav", "duration": 11.3, "text": "psychotherapy and the community both the physician and the patient find their place in the community the life interests of which are superior to the interests of the individual"}
    {"audio_filepath": "<absolute_path_to>/1355-39947-0001.wav", "duration": 15.905, "text": "it is an unavoidable question how far from the higher point of view of the social mind the psychotherapeutic efforts should be encouraged or suppressed are there any conditions which suggest suspicion of or direct opposition to such curative work"}

Fisher English Training Speech
------------------------------

Run these scripts to convert the Fisher English Training Speech data into a format expected by the ``nemo_asr`` collection.

In brief, the following scripts convert the ``.sph`` files to ``.wav``, slices those files into smaller audio samples, matches the
smaller slices with their corresponding transcripts, and splits the resulting audio segments into train, validation, and test sets
(with one manifest each).

.. note::
  - 106 GB of space is required to run the ``.wav`` conversion
  - additional 105 GB is required for the slicing and matching
  - ``sph2pipe`` is required in order to run the ``.wav`` conversion

**Instructions**

The following scripts assume that you already have the Fisher dataset from the Linguistic Data Consortium, with a directory structure
that looks similar to the following:

.. code-block:: bash

  FisherEnglishTrainingSpeech/
  ├── LDC2004S13-Part1
  │   ├── fe_03_p1_transcripts
  │   ├── fisher_eng_tr_sp_d1
  │   ├── fisher_eng_tr_sp_d2
  │   ├── fisher_eng_tr_sp_d3
  │   └── ...
  └── LDC2005S13-Part2
      ├── fe_03_p2_transcripts
      ├── fe_03_p2_sph1
      ├── fe_03_p2_sph2
      ├── fe_03_p2_sph3
      └── ...

The transcripts that will be used are located in the ``fe_03_p<1,2>_transcripts/data/trans`` directory. The audio files (``.sph``)
are located in the remaining directories in an ``audio`` subdirectory.

#. Convert the audio files from ``.sph`` to ``.wav`` by running:

   .. code-block:: bash

     cd <nemo_root>/scripts/dataset_processing
     python fisher_audio_to_wav.py \
       --data_root=<fisher_root> --dest_root=<conversion_target_dir>

   This will place the unsliced ``.wav`` files in ``<conversion_target_dir>/LDC200[4,5]S13-Part[1,2]/audio-wav/``. It will take several
   minutes to run.

#. Process the transcripts and slice the audio data.

   .. code-block:: bash

     python process_fisher_data.py \
       --audio_root=<conversion_target_dir> --transcript_root=<fisher_root> \
       --dest_root=<processing_target_dir> \
       --remove_noises

   This script splits the full dataset into train, validation, test sets, and places the audio slices in the corresponding folders
   in the destination directory. One manifest is written out per set, which includes each slice's transcript, duration, and path.

   This will likely take around 20 minutes to run. Once finished, delete the 10 minute long ``.wav`` files.

2000 HUB5 English Evaluation Speech
-----------------------------------

Run the following script to convert the HUB5 data into a format expected by the ``nemo_asr`` collection.

Similarly, to the Fisher dataset processing scripts, this script converts the ``.sph`` files to ``.wav``, slices the audio files and
transcripts into utterances, and combines them into segments of some minimum length (default is 10 seconds). The resulting segments
are all written out to an audio directory and the corresponding transcripts are written to a manifest JSON file.

.. note::
  - 5 GB of free space is required to run this script
  - ``sph2pipe`` is also required to be installed

This script assumes you already have the 2000 HUB5 dataset from the Linguistic Data Consortium.

Run the following command to process the 2000 HUB5 English Evaluation Speech samples:

.. code-block:: bash

  python process_hub5_data.py \
    --data_root=<path_to_HUB5_data> \
    --dest_root=<target_dir>

You can optionally include ``--min_slice_duration=<num_seconds>`` if you would like to change the minimum audio segment duration.

AN4 Dataset
-----------

This is a small dataset recorded and distributed by Carnegie Mellon University. It consists of recordings of people spelling out
addresses, names, etc. Information about this dataset can be found on the `official CMU site <http://www.speech.cs.cmu.edu/databases/an4/>`_.

#. `Download and extract the dataset <http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz>`_ (which is labeled "NIST's Sphere audio (.sph) format (64M)".

#. Convert the ``.sph`` files to ``.wav`` using sox, and build one training and one test manifest.

   .. code-block:: bash

     python process_an4_data.py --data_root=<path_to_extracted_data>

After the script finishes, the ``train_manifest.json`` and ``test_manifest.json`` can be found in the ``<data_root>/an4/`` directory.

Aishell-1
---------

To download the Aishell-1 data and convert it into a format expected by ``nemo_asr``, run:

.. code-block:: bash

    # install sox
    sudo apt-get install sox
    mkdir data
    python get_aishell_data.py --data_root=data

After the script finishes, the ``data`` folder should contain a ``data_aishell`` folder which contains a wav file, a transcript folder,  and related ``.json`` and ``vocab.txt`` files.

Aishell-2
---------

To process the AIShell-2 dataset, in the command below, set the data folder of AIShell-2 using ``--audio_folder`` and where to push
these files using ``--dest_folder``. In order to generate files in the supported format of ``nemo_asr``, run:

.. code-block:: bash

    python process_aishell2_data.py --audio_folder=<data directory> --dest_folder=<destination directory>

After the script finishes, the ``train.json``, ``dev.json``, ``test.json``, and ``vocab.txt`` files can be found in the ``dest_folder`` directory.

Preparing Custom ASR Data
-------------------------

The ``nemo_asr`` collection expects each dataset to consist of a set of utterances in individual audio files plus
a manifest that describes the dataset, with information about one utterance per line (``.json``).
The audio files can be of any format supported by `Pydub <https://github.com/jiaaro/pydub>`_, though we recommend
WAV files as they are the default and have been most thoroughly tested.

There should be one manifest file per dataset that will be passed in, therefore, if the user wants separate training and validation
datasets, they should also have separate manifests. Otherwise, they will be loading validation data with their training data and vice
versa.

Each line of the manifest should be in the following format:

.. code-block:: json

  {"audio_filepath": "/path/to/audio.wav", "text": "the transcription of the utterance", "duration": 23.147}

The :code:`audio_filepath` field should provide an absolute path to the ``.wav`` file corresponding to the utterance.
The :code:`text` field should contain the full transcript for the utterance, and the :code:`duration` field should
reflect the duration of the utterance in seconds.

Each entry in the manifest (describing one audio file) should be bordered by '{' and '}' and must
be contained on one line. The fields that describe the file should be separated by commas, and have the form :code:`"field_name": value`,
as shown above. There should be no extra lines in the manifest, i.e. there should be exactly as many lines in the manifest as
there are audio files in the dataset.

Since the manifest specifies the path for each utterance, the audio files do not have to be located
in the same directory as the manifest, or even in any specific directory structure.

Once there is a manifest that describes each audio file in the dataset, use the dataset by passing
in the manifest file path in the experiment config file, e.g. as ``training_ds.manifest_filepath=<path/to/manifest.json>``.

Tarred Datasets
---------------

If experiments are run on a cluster with datasets stored on a distributed file system, the user will likely
want to avoid constantly reading multiple small files and would prefer tarring their audio files.
There are tarred versions of some NeMo ASR dataset classes for this case, such as the ``TarredAudioToCharDataset``
(corresponding to the ``AudioToCharDataset``) and the ``TarredAudioToBPEDataset`` (corresponding to the
``AudioToBPEDataset``). The tarred audio dataset classes in NeMo use `WebDataset <https://github.com/tmbdev/webdataset>`_.

To use an existing tarred dataset instead of a non-tarred dataset, set ``is_tarred: true`` in
the experiment config file. Then, pass in the paths to all of the audio tarballs in ``tarred_audio_filepaths``, either as a list
of filepaths, e.g. ``['/data/shard1.tar', '/data/shard2.tar']``, or in a single brace-expandable string, e.g.
``'/data/shard_{1..64}.tar'`` or ``'/data/shard__OP_1..64_CL_'`` (recommended, see note below).

.. note::
  For brace expansion, there may be cases where ``{x..y}`` syntax cannot be used due to shell interference. This occurs most commonly
  inside SLURM scripts. Therefore, we provide a few equivalent replacements. Supported opening braces (equivalent to ``{``) are ``(``,
  ``[``, ``<`` and the special tag ``_OP_``. Supported closing braces (equivalent to ``}``) are ``)``, ``]``, ``>`` and the special
  tag ``_CL_``. For SLURM based tasks, we suggest the use of the special tags for ease of use.

As with non-tarred datasets, the manifest file should be passed in ``manifest_filepath``. The dataloader assumes that the length
of the manifest after filtering is the correct size of the dataset for reporting training progress.

The ``tarred_shard_strategy`` field of the config file can be set if you have multiple shards and are running an experiment with
multiple workers. It defaults to ``scatter``, which preallocates a set of shards per worker which do not change during runtime.
Note that this strategy, on specific occasions (when the number of shards is not divisible with ``world_size``), will not sample
the entire dataset. As an alternative the ``replicate`` strategy, will preallocate the entire set of shards to every worker and not
change it during runtime. The benefit of this strategy is that it allows each worker to sample data points from the entire dataset
independently of others. Note, though, that more than one worker may sample the same shard, and even sample the same data points!
As such, there is no assured guarantee that all samples in the dataset will be sampled at least once during 1 epoch. Note that
for these reasons it is not advisable to use tarred datasets as validation and test datasets.

For more information about the individual tarred datasets and the parameters available, including shuffling options,
see the corresponding class APIs in the `Datasets <./api.html#Datasets>`__ section.

.. warning::
  If using multiple workers, the number of shards should be divisible by the world size to ensure an even
  split among workers. If it is not divisible, logging will give a warning but training will proceed, but likely hang at the last epoch.
  In addition, if using distributed processing, each shard must have the same number of entries after filtering is
  applied such that each worker ends up with the same number of files. We currently do not check for this in any dataloader, but the user's
  program may hang if the shards are uneven.

Conversion to Tarred Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can easily convert your existing NeMo-compatible ASR datasets using the
`conversion script here <https://github.com/NVIDIA/NeMo/tree/stable/scripts/speech_recognition/convert_to_tarred_audio_dataset.py>`_.

.. code:: bash

  python convert_to_tarred_audio_dataset.py \
    --manifest_path=<path to the manifest file> \
    --target_dir=<path to output directory> \
    --num_shards=<number of tarfiles that will contain the audio>
    --max_duration=<float representing maximum duration of audio samples> \
    --min_duration=<float representing minimum duration of audio samples> \
    --shuffle --shuffle_seed=0

This script shuffles the entries in the given manifest (if ``--shuffle`` is set, which we recommend), filter
audio files according to ``min_duration`` and ``max_duration``, and tar the remaining audio files to the directory
``--target_dir`` in ``n`` shards, along with separate manifest and metadata files.

The files in the target directory should look similar to the following:

.. code::

  target_dir/
  ├── audio_1.tar
  ├── audio_2.tar
  ├── ...
  ├── metadata.yaml
  └── tarred_audio_manifest.json

Note that file structures are flattened such that all audio files are at the top level in each tarball. This ensures that
filenames are unique in the tarred dataset and the filepaths do not contain "-sub" and forward slashes in each ``audio_filepath`` are
simply converted to underscores. For example, a manifest entry for ``/data/directory1/file.wav`` would be ``_data_directory1_file.wav``
in the tarred dataset manifest, and ``/data/directory2/file.wav`` would be converted to ``_data_directory2_file.wav``.

Bucketing Datasets
------------------

For training ASR models, audios with different lengths may be grouped into a batch. It would make it necessary to use paddings to make all the same length.
These extra paddings is a significant source of computation waste. Splitting the training samples into buckets with different lengths and sampling from the same bucket for each batch would increase the computation efficicncy.
It may result into training speeedup of more than 2X. To enable and use the bucketing feature, you need to create the bucketing version of the dataset by using `conversion script here <https://github.com/NVIDIA/NeMo/tree/stable/scripts/speech_recognition/convert_to_tarred_audio_dataset.py>`_.
You may use --buckets_num to specify the number of buckets (Recommened to use 4 to 8 buckets). It creates multiple tarred datasets, one per bucket, based on the audio durations. The range of [min_duration, max_duration) is split into equal sized buckets.


To enable the bucketing feature in the dataset section of the config files, you need to pass the multiple tarred datasets as a list of lists.
If user passes just a list of strings, then the datasets would simply get concatenated which would be different from bucketing.
Here is an example for 4 buckets and 512 shards:

.. code::

    python speech_to_text_bpe.py
    ...
    model.train_ds.manifest_filepath=[[PATH_TO_TARS/bucket1/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket2/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket3/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket4/tarred_audio_manifest.json]]
    model.train_ds.tarred_audio_filepaths=[[PATH_TO_TARS/bucket1/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket2/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket3/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket4/audio__OP_0..511_CL_.tar]]

When bucketing is enabled, in each epoch, first all GPUs would use the first bucket, then go to the second bucket, and so on. It guarantees that all GPUs are using the same bucket at the same time. It reduces the number of paddings in each batch and speedup the training significantly without hurting the accuracy significantly.

There are two types of batching:

*  Fixed-size bucketing: all batches would have the same number of samples specified by train_ds.batch_size
*  Adaptive-size bucketing: uses different batch sizes for each bucket.

Adaptive-size bucketing helps to increase the GPU utilization and speedup the training.
Batches sampled from buckets with smaller audio lengths can be larger which would increase the GPU utilization and speedup the training.
You may use train_ds.bucketing_batch_size to enable the adaptive batching and specify the batch sizes for the buckets.
When bucketing_batch_size is not set, train_ds.batch_size is going to be used for all buckets (fixed-size bucketing).

bucketing_batch_size can be set as an integer or a list of integers to explicitly specify the batch size for each bucket.
if bucketing_batch_size is set to be an integer, then linear scaling is being used to scale-up the batch sizes for batches with shorted audio size. For example, setting train_ds.bucketing_batch_size=8 for 4 buckets would use these sizes [32,24,16,8] for different buckets.
When bucketing_batch_size is set, traind_ds.batch_size need to be set to 1.

Training an ASR model on audios sorted based on length may affect the accuracy of the model. We introduced some strategies to mitigate it.
We support three types of bucketing strategies:

*   fixed_order: the same order of buckets are used for all epochs
*   synced_randomized (default): each epoch would have a different order of buckets. Order of the buckets is shuffled every epoch.
*   fully_randomized: similar to synced_randomized but each GPU has its own random order. So GPUs would not be synced.

Tha parameter train_ds.bucketing_strategy can be set to specify one of these strategies. The recommended strategy is synced_randomized which gives the highest training speedup.
The fully_randomized strategy would have lower speedup than synced_randomized but may give better accuracy.

Bucketing may improve the training speed more than 2x but may affect the final accuracy of the model slightly. Training for more epochs and using 'synced_randomized' strategy help to fill this gap.
Currently bucketing feature is just supported for tarred datasets.

Upsampling Datasets
-------------------

Buckets may also be 'weighted' to allow multiple runs through a target dataset during each training epoch. This can be beneficial in cases when a dataset is composed of several component sets of unequal sizes and one desires to mitigate bias towards the larger sets through oversampling.

Weighting is managed with the `bucketing_weights` parameter. After passing your composite tarred datasets in the format described above for bucketing, pass a list of integers (one per bucket) to indicate how many times a manifest should be read during training.

For example, by passing `[2,1,1,3]` to the code below:

.. code::

    python speech_to_text_bpe.py
    ...
    model.train_ds.manifest_filepath=[[PATH_TO_TARS/bucket1/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket2/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket3/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket4/tarred_audio_manifest.json]]
    model.train_ds.tarred_audio_filepaths=[[PATH_TO_TARS/bucket1/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket2/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket3/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket4/audio__OP_0..511_CL_.tar]]
	...
	model.train_ds.bucketing_weights=[2,1,1,3]

NeMo will configure training so that all data in `bucket1` will be present twice in a training epoch, `bucket4` will be present three times, and that of `bucket2` and `bucket3` will occur only once each. Note that this will increase the effective amount of data present during training and thus affect training time per epoch.

If using adaptive bucketing, note that the same batch size will be assigned to each instance of the upsampled data. That is, given the following:

.. code::

    python speech_to_text_bpe.py
    ...
    model.train_ds.manifest_filepath=[[PATH_TO_TARS/bucket1/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket2/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket3/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket4/tarred_audio_manifest.json]]
	...
	...
	model.train_ds.bucketing_weights=[2,1,1,3]
	model.train_ds.bucketing_batch_size=[4,4,4,2]

All instances of data from `bucket4` will still be trained with a batch size of 2 while all others would have a batch size of 4. As with standard bucketing, this requires `batch_size`` to be set to 1.
If `bucketing_batch_size` is not specified, all datasets will be passed with the same fixed batch size as specified by the `batch_size` parameter.

It is recommended to set bucketing strategies to `fully_randomized` during multi-GPU training to prevent possible dataset bias during training.