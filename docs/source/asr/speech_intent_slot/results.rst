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


Inference
-----------------------
   
The audio files should be 16KHz monochannel wav files.

**Transcribe Audios to Semantics:**
  
You may perform inference on a sample of speech after loading the model by using its 'transcribe()' method:

.. code-block:: python 

  slu_model = nemo_asr.models.SLUIntentSlotBPEModel.from_pretrained(model_name="<MODEL_NAME>")
  predictions = slu_model.transcribe([list of audio files],  batch_size="<BATCH_SIZE>") 


SLU Models
-----------------------------------

Below is a list of all the Speech Intent Classification and Slot Filling models that are available in NeMo.


.. csv-table::
   :file: data/benchmark_sis.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1
