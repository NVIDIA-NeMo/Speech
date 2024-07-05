Checkpoints
===========

There are two main ways to load pretrained checkpoints in NeMo:

* Using the :code:`restore_from()` method to load a local checkpoint file (`.nemo`), or
* Using the :code:`from_pretrained()` method to download and set up a checkpoint from NGC.

See the following sections for instructions and examples for each.

Note that these instructions are for loading fully trained checkpoints for evaluation or fine-tuning.
For resuming an unfinished training experiment, please use the experiment manager to do so by setting the
``resume_if_exists`` flag to True.

Loading Local Checkpoints
-------------------------

NeMo will automatically save checkpoints of a model you are training in a `.nemo` format.
You can also manually save your models at any point using :code:`model.save_to(<checkpoint_path>.nemo)`.

If you have a local ``.nemo`` checkpoint that you'd like to load, simply use the :code:`restore_from()` method:

.. code-block:: python

  import nemo.collections.asr as nemo_asr
  model = nemo_asr.models.<MODEL_BASE_CLASS>.restore_from(restore_path="<path/to/checkpoint/file.nemo>")

Where the model base class is the ASR model class of the original checkpoint, or the general `ASRModel` class.

Speaker Label Inference
------------------------

The goal of speaker label inference is to infer speaker labels using a speaker model with known speaker labels from enrollment set. We provide `speaker_identification_infer.py` script for this purpose under `<NeMo_root>/examples/speaker_tasks/recognition` folder.
Currently supported backends are cosine_similarity and neural classifier.

The audio files should be 16KHz mono channel wav files.

The script takes two manifest files: 

* enrollment_manifest : This manifest contains enrollment data with known speaker labels.
* test_manifest: This manifest contains test data for which we map speaker labels captured from enrollment manifest using one of provided backend

sample format for each of these manifests is provided in `<NeMo_root>/examples/speaker_tasks/recognition/conf/speaker_identification_infer.yaml` config file.

To infer speaker labels using cosine_similarity backend

.. code-block:: bash
  
    python speaker_identification_infer.py data.enrollment_manifest=<path/to/enrollment_manifest> data.test_manifest=<path/to/test_manifest> backend.backend_model=cosine_similarity

    
Speaker Embedding Extraction
-----------------------------
Speaker Embedding Extraction, is to extract speaker embeddings for any wav file (from known or unknown speakers). We provide two ways to do this:

* single Python liner for extracting embeddings from a single file 
* Python script for extracting embeddings from a bunch of files provided through manifest file

For extracting embeddings from a single file:

.. code-block:: python

  speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="<pretrained_model_name or path/to/nemo/file>")
  embs = speaker_model.get_embedding('<audio_path>')

For extracting embeddings from a bunch of files:

The audio files should be 16KHz mono channel wav files.

Write audio files to a ``manifest.json`` file with lines as in format:

.. code-block:: json
    
    {"audio_filepath": "<absolute path to dataset>/audio_file.wav", "duration": "duration of file in sec", "label": "speaker_id"}
      
This python call will download best pretrained model from NGC and writes embeddings pickle file to current working directory

.. code-block:: bash
  
    python examples/speaker_tasks/recognition/extract_speaker_embeddings.py --manifest=manifest.json
   
or you can run `batch_inference()` to perform inference on the manifest with seleted batch_size to get embeddings

.. code-block:: python

  speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="<pretrained_model_name or path/to/nemo/file>")
  embs, logits, gt_labels, trained_labels = speaker_model.batch_inference(manifest, batch_size=32)

Speaker Verification Inference
------------------------------

Speaker Verification is a task of verifying if two utterances are from the same speaker or not.

We provide a helper function to verify the audio files (also in a batch) and return True if provided pair of audio files is from the same speaker, False otherwise.

The audio files should be 16KHz mono channel wav files.

.. code-block:: python

  speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
  decision = speaker_model.verify_speakers('path/to/one/audio_file','path/to/other/audio_file')
  decisions = speaker_model.verify_speakers_batch([
                                                  ('/path/to/audio_0_0', '/path/to/audio_0_1'),
                                                  ('/path/to/audio_1_0', '/path/to/audio_1_1'),
                                                  ('/path/to/audio_2_0', '/path/to/audio_2_1'),
                                                  ('/path/to/audio_3_0', '/path/to/audio_3_1')
                                                  ],  batch_size=4, device='cuda')


NGC Pretrained Checkpoints
--------------------------

The SpeakerNet-ASR collection has checkpoints of several models trained on various datasets for a variety of tasks.
`TitaNet <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large>`_ , `ECAPA_TDNN <https://ngc.nvidia.com/catalog/models/nvidia:nemo:ecapa_tdnn>`_ and `Speaker_Verification <https://ngc.nvidia.com/catalog/models/nvidia:nemo:speakerverification_speakernet>`_ model cards on NGC contain more information about each of the checkpoints available.

The tables below list the speaker embedding extractor models available from NGC, and the models can be accessed via the
:code:`from_pretrained()` method inside the EncDecSpeakerLabelModel Model class.

In general, you can load any of these models with code in the following format:

.. code-block:: python

  import nemo.collections.asr as nemo_asr
  model = nemo_asr.models.<MODEL_CLASS_NAME>.from_pretrained(model_name="<MODEL_NAME>")

where the model name is the value under "Model Name" entry in the tables below.

If you would like to programatically list the models available for a particular base class, you can use the
:code:`list_available_models()` method.

.. code-block:: python

  nemo_asr.models.<MODEL_BASE_CLASS>.list_available_models()


Speaker Recognition Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: data/speaker_results.csv
   :align: left
   :widths: 30, 30, 40
   :header-rows: 1

