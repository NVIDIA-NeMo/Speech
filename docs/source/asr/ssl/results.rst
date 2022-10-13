Checkpoints
===========

Pre-trained SSL checkpoints available in NeMo need to be further fine-tuned on down-stream task. 
There are two main ways to load pretrained checkpoints in NeMo:

* Using the :code:`restore_from()` method to load a local checkpoint file (``.nemo``), or
* Using the :code:`from_pretrained()` method to download and set up a checkpoint from NGC.

Refer to the following sections for instructions and examples for each.

Note that these instructions are for fine-tuning. To resume an unfinished training experiment, 
use the Experiment Manager to do so by setting the ``resume_if_exists`` flag to ``True``.

Loading Local Checkpoints
-------------------------

NeMo automatically saves checkpoints of a model that is trained in a ``.nemo`` format. Alternatively, to manually save the model at any 
point, issue :code:`model.save_to(<checkpoint_path>.nemo)`.

If there is a local ``.nemo`` checkpoint that you'd like to load, use the :code:`restore_from()` method:

.. code-block:: python

  import nemo.collections.asr as nemo_asr
  ssl_model = nemo_asr.models.<MODEL_BASE_CLASS>.restore_from(restore_path="<path/to/checkpoint/file.nemo>")

Where the model base class is the ASR model class of the original checkpoint, or the general ``ASRModel`` class.

Loading NGC Pretrained Checkpoints
----------------------------------

The SSL collection has checkpoints of several models trained on various datasets. These checkpoints are 
obtainable via NGC `NeMo Automatic Speech Recognition collection <https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_asr>`_.
The model cards on NGC contain more information about each of the checkpoints available.

The table at the end of this page lists the SSL models available from NGC. The models can be accessed via the :code:`from_pretrained()` method inside
the ASR Model class. In general, you can load any of these models with code in the following format:

.. code-block:: python

  import nemo.collections.asr as nemo_asr
  ssl_model = nemo_asr.models.ASRModel.from_pretrained(model_name="<MODEL_NAME>")

Where the ``model_name`` is the value under "Model Name" entry in the tables below.

For example, to load the conformer Large SSL checkpoint, run:

.. code-block:: python

  ssl_model = nemo_asr.models.ASRModel.from_pretrained(model_name="ssl_en_conformer_large")

You can also call :code:`from_pretrained()` from the specific model class (such as :code:`SpeechEncDecSelfSupervisedModel`
for Conformer) if you need to access a specific model functionality.

If you would like to programatically list the models available for a particular base class, you can use the
:code:`list_available_models()` method.

.. code-block:: python

  nemo_asr.models.<MODEL_BASE_CLASS>.list_available_models()


Loading SSL checkpoint into Down-stream Model
---------------------------------------------
After loading an SSL checkpoint as shown above, it's ``state_dict`` needs to be copied to a 
down-stream model for fine-tuning. 

For example, to load a SSL checkpoint for ASR down-stream task using ``EncDecRNNTBPEModel``, run:

.. code-block:: python

  # define down-stream model
  asr_model = nemo_asr.models.EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)

  # load ssl checkpoint
  asr_model.load_state_dict(ssl_model.state_dict(), strict=False)

  # discard ssl model
  del ssl model

Refer to `SSL configs <./configs.html>`__ to do this automatically via config files. 


Fine-tuning on Downstream Datasets
-----------------------------------

After loading SSL checkpoint into down-stream model, refer to multiple ASR tutorials provided in the :ref:`Tutorials <tutorials>` section. 
Most of these tutorials explain how to fine-tune on some dataset as a demonstration.

Inference Execution Flow Diagram
--------------------------------

When preparing your own inference scripts after downstream fine-tuning, please follow the execution flow diagram order for correct inference, found at the `examples directory for ASR collection <https://github.com/NVIDIA/NeMo/blob/stable/examples/asr/README.md>`_.

SSL Models
-----------------------------------

Below is a list of all the SSL models that are available in NeMo.


.. csv-table::
   :file: data/benchmark_ssl.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1
