Checkpoints
===========

There are two main ways to load pretrained checkpoints in NeMo as described in :doc:`../asr/results`.

* Using the :code:`restore_from()` method to load a local checkpoint file (``.nemo``), or
* Using the :code:`from_pretrained()` method to download and set up a checkpoint from NGC.

Note that these instructions are for loading fully trained checkpoints for evaluation or fine-tuning. For resuming an unfinished
training experiment, use the Experiment Manager to do so by setting the ``resume_if_exists`` flag to ``True``.

Local Checkpoints
-------------------------

* **Save Model Checkpoints**: NeMo automatically saves final model checkpoints with ``.nemo`` suffix. You could also manually save any model checkpoint using :code:`model.save_to(<checkpoint_path>.nemo)`.
* **Load Model Checkpoints**: if you'd like to load a checkpoint saved at ``<path/to/checkpoint/file.nemo>``, use the :code:`restore_from()` method below, where ``<MODEL_BASE_CLASS>`` is the TTS model class of the original checkpoint.

.. code-block:: python

    import nemo.collections.tts as nemo_tts
    model = nemo_tts.models.<MODEL_BASE_CLASS>.restore_from(restore_path="<path/to/checkpoint/file.nemo>")

NGC Pretrained Checkpoints
--------------------------

The NGC `NeMo Text to Speech collection <https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_tts>`_ aggregates model cards that contain detailed information about  checkpoints of various models trained on various datasets. The tables below in :ref:`Checkpoints<NGC TTS Models>` list part of available TTS models from NGC including speech/text aligners, acoustic models, and vocoders.

Load Model Checkpoints
^^^^^^^^^^^^^^^^^^^^^^
The models can be accessed via the :code:`from_pretrained()` method inside the TTS Model class. In general, you can load any of these models with code in the following format,

.. code-block:: python

    import nemo.collections.tts as nemo_tts
    model = nemo_tts.models.<MODEL_BASE_CLASS>.from_pretrained(model_name="<MODEL_NAME>")

where ``<MODEL_NAME>`` is generally the basename of the "Model Card" entry in the tables in :ref:`Checkpoints<NGC TTS Models>`. For example, the basename of the English FastPitch mel-generator model from https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_fastpitch is :code:`"tts_en_fastpitch"`. You could load this model by running,

.. code-block:: python

    model = nemo_tts.models.FastPitchModel.from_pretrained(model_name="tts_en_fastpitch")

If you would like to programmatically list the models available for a particular base class, you can use the
:code:`list_available_models()` method,

.. code-block:: python

    nemo_tts.models.<MODEL_BASE_CLASS>.list_available_models()

Inference and Audio Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NeMo TTS supports both cascaded and end-to-end models to synthesize audios. Most of steps in between are the same except that cascaded models need to load an extra vocoder model before generating audios. Below code snippet demonstrates steps of generating a audio sample from a text input using a cascaded FastPitch and HiFiGAN models. Please refer to :ref:`NeMo TTS Collection API` for detailed implementation of model classes.

.. code-block:: python

    import nemo.collections.tts as nemo_tts
    # Load mel spectrogram generator
    spec_generator = nemo_tts.models.FastPitchModel.from_pretrained("tts_en_fastpitch")
    # Load vocoder
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained(model_name="tts_hifigan")
    # Generate audio
    import soundfile as sf
    parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    # Save the audio to disk in a file called speech.wav
    sf.write("speech.wav", audio.to('cpu').numpy(), 22050)


Fine-Tuning on Different Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are multiple TTS tutorials provided in the directory of `tutorials/tts/ <https://github.com/NVIDIA/NeMo/tree/stable/tutorials/tts>`_. Most of these tutorials demonstrate how to instantiate a pre-trained model, and prepare the model for fine-tuning on datasets with the same language or different languages, the same speaker or different speakers.

* **cross-lingual fine-tuning**: https://github.com/NVIDIA/NeMo/tree/stable/tutorials/tts/FastPitch_GermanTTS_Training.ipynb
* **cross-speaker fine-tuning**: https://github.com/NVIDIA/NeMo/tree/stable/tutorials/tts/FastPitch_Finetuning.ipynb

NGC TTS Models
-----------------------------------

Below summarizes a full list of available NeMo TTS models that have been released in `NGC NeMo Text to Speech Collection <https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_tts/entities>`_.

Speech/Text Aligners
^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: data/ngc_models_aligner.csv
   :align: left
   :header-rows: 1

Mel-Spectrogram Generators
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. csv-table::
   :file: data/ngc_models_am.csv
   :align: left
   :header-rows: 1

Vocoders
^^^^^^^^
.. csv-table::
   :file: data/ngc_models_vocoder.csv
   :align: left
   :header-rows: 1