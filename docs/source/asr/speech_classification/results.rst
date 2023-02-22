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


Transcribing/Inference
-----------------------
   
The audio files should be 16KHz monochannel wav files.

`Transcribe speech command segment:`
  
You may perform inference and transcribe a sample of speech after loading the model by using its 'transcribe()' method:

.. code-block:: python 

  mbn_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="<MODEL_NAME>")
  mbn_model.transcribe([list of audio files],  batch_size=BATCH_SIZE, logprobs=False) 

Setting argument ``logprobs`` to True would return the log probabilities instead of transcriptions. You may find more details in `Modules <../api.html#modules>`__.

Learn how to fine tune on your own data or on subset classes in ``<NeMo_git_root>/tutorials/asr/Speech_Commands.ipynb``


`Run VAD inference:`

.. code-block:: bash 

  python <NeMo-git-root>/examples/asr/speech_classification/vad_infer.py --config-path="../conf/vad" --config-name="vad_inference_postprocessing.yaml" dataset=<Path of json file of evaluation data. Audio files should have unique names>


This script will perform vad frame-level prediction and will help you perform postprocessing and generate speech segments as well if needed.

Have a look at configuration file ``<NeMo-git-root>/examples/asr/conf/vad/vad_inference_postprocessing.yaml`` and scripts under ``<NeMo-git-root>/scripts/voice_activity_detection`` for details regarding posterior processing, postprocessing and threshold tuning.

Posterior processing includes generating predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.

For VAD postprocessing we introduce 

Binarization: 
  - ``onset`` and ``offset`` threshold for detecting the beginning and end of a speech. 
  - padding durations ``pad_onset`` before and padding duarations ``pad_offset`` after each speech segment;

Filtering:
  - ``min_duration_on`` threshold for short speech segment deletion,
  - ``min_duration_on`` threshold for small silence deletion,
  - ``filter_speech_first`` to control whether to perform short speech segment deletion first.


`Identify language of utterance`

You may load the model and identify the language of an audio file by using `get_label()` method:

.. code-block:: python

  langid_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="<MODEL_NAME>")
  lang = langid_model.get_label('<audio_path>')
  
or you can run `batch_inference()` to perform inference on a manifest with seleted batch_size to get trained model labels and gt_labels with logits

.. code-block:: python

  langid_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="<MODEL_NAME>")
  lang_embs, logits, gt_labels, trained_labels = langid_model.batch_inference(manifest_filepath, batch_size=32)

  
NGC Pretrained Checkpoints
--------------------------

The Speech Classification collection has checkpoints of several models trained on various datasets for a variety of tasks.
These checkpoints are obtainable via NGC `NeMo Automatic Speech Recognition collection <https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels>`_.
The model cards on NGC contain more information about each of the checkpoints available.

The tables below list the Speech Classification models available from NGC, and the models can be accessed via the
:code:`from_pretrained()` method inside the ASR Model class.

In general, you can load any of these models with code in the following format.

.. code-block:: python

  import nemo.collections.asr as nemo_asr
  model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="<MODEL_NAME>")

Where the model name is the value under "Model Name" entry in the tables below.

For example, to load the MatchboxNet3x2x64_v1 model for speech command detection, run:

.. code-block:: python

  model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="commandrecognition_en_matchboxnet3x2x64_v1")

You can also call :code:`from_pretrained()` from the specific model class (such as :code:`EncDecClassificationModel`
for MatchboxNet and MarbleNet) if you will need to access specific model functionality.

If you would like to programatically list the models available for a particular base class, you can use the
:code:`list_available_models()` method.

.. code-block:: python

  nemo_asr.models.<MODEL_BASE_CLASS>.list_available_models()


Speech Classification Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabularcolumns:: 30 30 40

.. csv-table::
   :file: data/classification_results.csv
   :header-rows: 1
   :class: longtable
   :widths: 1 1 1

