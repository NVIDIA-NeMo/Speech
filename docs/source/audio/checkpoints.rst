Checkpoints
===========

There are two main ways to load pretrained checkpoints in NeMo:

* Using the :code:`restore_from()` method to load a local checkpoint file (``.nemo``), or
* Using the :code:`from_pretrained()` method to download and set up a checkpoint from the cloud.

Note that these instructions are for loading fully trained checkpoints for evaluation or fine-tuning. For resuming an unfinished
training experiment, use the Experiment Manager to do so by setting the ``resume_if_exists`` flag to ``True``.

Local Checkpoints
-------------------------

* **Save Model Checkpoints**: NeMo automatically saves final model checkpoints with ``.nemo`` suffix. You could also manually save any model checkpoint using :code:`model.save_to(<checkpoint_path>.nemo)`.
* **Load Model Checkpoints**: if you'd like to load a checkpoint saved at ``<path/to/checkpoint/file.nemo>``, use the :code:`restore_from()` method below, where ``<MODEL_BASE_CLASS>`` is the model class of the original checkpoint.

.. code-block:: python

    import nemo.collections.audio as nemo_audio
    model = nemo_audio.models.<MODEL_BASE_CLASS>.restore_from(restore_path="<path/to/checkpoint/file.nemo>")

Pretrained Checkpoints
----------------------

The table below in :ref:`checkpoints_audio_models` list part of available pre-trained audio processing models including speech processing, restoration and extraction.

Load Model Checkpoints
^^^^^^^^^^^^^^^^^^^^^^
The models can be accessed via the :code:`from_pretrained()` method inside the audio model class. In general, you can load any of these models with code in the following format,

.. code-block:: python

    import nemo.collections.audio as nemo_audio
    model = nemo_audio.models.<MODEL_BASE_CLASS>.from_pretrained(model_name="<MODEL_NAME>")

where ``<MODEL_NAME>`` is the value in ``Model Name`` column in the tables in :ref:`checkpoints_audio_models`. These names are predefined in the each model's member function ``self.list_available_models()``. 


.. _checkpoints_audio_models:

Audio Models
------------

Speech Enhancement Models
^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: data/checkpoints_se.csv
   :align: left
   :header-rows: 1

SSL Models
^^^^^^^^^^

.. csv-table::
   :file: data/checkpoints_ssl.csv
   :align: left
   :header-rows: 1
