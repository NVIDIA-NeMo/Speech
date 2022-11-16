Models
======

This page gives a brief overview of the models that NeMo's Speech Classification collection currently supports.
For Speech Classification, we support Speech Command (Keyword) Detection and Voice Activity Detection (VAD).

Each of these models can be used with the example ASR scripts (in the ``<NeMo_git_root>/examples/asr`` directory) by
specifying the model architecture in the config file used.
Examples of config files for each model can be found in the ``<NeMo_git_root>/examples/asr/conf`` directory.

For more information about the config files and how they should be structured, see the :doc:`./configs` page.

Pretrained checkpoints for all of these models, as well as instructions on how to load them, can be found on the :doc:`./results` page.
You can use the available checkpoints for immediate inference, or fine-tune them on your own datasets.
The Checkpoints page also contains benchmark results for the available ASR models.

.. _MatchboxNet_model:

MatchboxNet (Speech Commands) 
------------------------------

MatchboxNet :cite:`sc-models-matchboxnet` is an end-to-end neural network for speech command recognition based on `QuartzNet <../models.html#QuartzNet>`__.

Similarly to QuartzNet, the MatchboxNet family of models are denoted as MatchBoxNet_[BxRxC] where B is the number of blocks, and R is the number of convolutional sub-blocks within a block, and C is the number of channels. Each sub-block contains a 1-D *separable* convolution, batch normalization, ReLU, and dropout:

    .. image:: images/matchboxnet_vertical.png
        :align: center
        :alt: MatchboxNet model
        :scale: 50%

It can reach state-of-the art accuracy on the Google Speech Commands dataset while having significantly fewer parameters than similar models. 
The `_v1` and `_v2` are denoted for models trained on `v1` (30-way classification) and `v2` (35-way classification) datasets; 
And we use _subset_task to represent (10+2)-way subset (10 specific classes + other remaining classes + silence) classification task.

MatchboxNet models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecClassificationModel` class.

.. note::
  For model details and deep understanding about Speech Command Detedction training, inference, finetuning and etc., 
  please refer to  ``<NeMo_git_root>/tutorials/asr/Speech_Commands.ipynb`` and ``<NeMo_git_root>/tutorials/asr/Online_Offline_Speech_Commands_Demo.ipynb``.



.. _MarbleNet_model:

MarbleNet (VAD) 
------------------

MarbleNet :cite:`sc-models-marblenet` an end-to-end neural network for speech command recognition based on :ref:`MatchboxNet_model`, 

Similarly to MatchboxNet, the MarbleNet family of models are denoted as MarbleNet_[BxRxC] where B is the number of blocks, and R is the number of convolutional sub-blocks within a block, and C is the number of channels. Each sub-block contains a 1-D *separable* convolution, batch normalization, ReLU, and dropout:

    .. image:: images/marblenet_vertical.png
        :align: center
        :alt: MarbleNet model
        :scale: 30%

It can reach state-of-the art performance on the difficult `AVA speech dataset <https://research.google.com/ava/download.html#ava_speech_download>`_ while having significantly fewer parameters than similar models even training on simple data.
MarbleNet models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecClassificationModel` class.

.. note::
  For model details and deep understanding about VAD training, inference, postprocessing, threshold tuning and etc., 
  please refer to  ``<NeMo_git_root>/tutorials/asr/06_Voice_Activiy_Detection.ipynb`` and ``<NeMo_git_root>/tutorials/asr/Online_Offline_Microphone_VAD_Demo.ipynb``.



.. _AmberNet_model:

AmberNet (Lang ID) 
------------------

AmberNet is an end-to-end neural network for language identification moden based on `TitaNet <../speaker_recognition/models.html#titanet>`__.

It can reach state-of-the art performance on the `Voxlingua107 dataset <http://bark.phon.ioc.ee/voxlingua107/>`_ while having significantly fewer parameters than similar models.
AmberNet models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecSpeakerLabelModel` class.



References
----------------

.. bibliography:: ../asr_all.bib
    :style: plain
    :labelprefix: SC-MODELS
    :keyprefix: sc-models-
