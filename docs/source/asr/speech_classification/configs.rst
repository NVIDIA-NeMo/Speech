NeMo Speech Classification Configuration Files
================================================

This page covers NeMo configuration file setup that is specific to models in the Speech Classification collection.
For general information about how to set up and run experiments that is common to all NeMo models (e.g.
experiment manager and PyTorch Lightning trainer parameters), see the :doc:`../../core/core`  page.

The model section of NeMo Speech Classification configuration files will generally require information about the dataset(s) being
used, the preprocessor for audio files, parameters for any augmentation being performed, as well as the
model architecture specification.
The sections on this page cover each of these in more detail.

Example configuration files for all of the NeMo ASR scripts can be found in the
``<NeMo_git_root>/examples/asr/conf>``.


Dataset Configuration
---------------------

Training, validation, and test parameters are specified using the ``train_ds``, ``validation_ds``, and
``test_ds`` sections of your configuration file, respectively.
Depending on the task, you may have arguments specifying the sample rate of your audio files, labels, whether or not to shuffle the dataset, and so on.
You may also decide to leave fields such as the ``manifest_filepath`` blank, to be specified via the command line
at runtime.

Any initialization parameters that are accepted for the Dataset class used in your experiment
can be set in the config file.
See the :ref:`Datasets <asr-api-datasets>` section of the API for a list of Datasets and their respective parameters.

An example Speech Classification train and validation configuration could look like: 

.. code-block:: yaml

  model:
    sample_rate: 16000
    repeat: 2 # number of convolutional sub-blocks within a block, R in <MODEL>_[BxRxC]
    dropout: 0.0
    kernel_size_factor: 1.0
    labels: ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin',
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
    'wow', 'yes', 'zero']

    train_ds:
      manifest_filepath: ???
      sample_rate: ${model.sample_rate}
      labels: ${model.labels} # Uses the labels above
      batch_size: 128
      shuffle: True

    validation_ds:
      manifest_filepath: ???
      sample_rate: ${model.sample_rate}
      labels: ${model.labels} # Uses the labels above
      batch_size: 128
      shuffle: False # No need to shuffle the validation data

If you would like to use tarred dataset, have a look at :ref:`Datasets Configuration <asr-configs-dataset-configuration>`.


Preprocessor Configuration
--------------------------
Preprocessor helps to compute MFCC or mel spectrogram features that are given as inputs to model. 
For details on how to write this section, refer to :ref:`Preprocessor Configuration <asr-configs-preprocessor-configuration>`

Check config yaml files in ``<NeMo_git_root>/examples/asr/conf`` to find the processors been used by speech classification models.


Augmentation Configurations
---------------------------

There are a few on-the-fly spectrogram augmentation options for NeMo ASR, which can be specified by the
configuration file using the ``augmentor`` and ``spec_augment`` section.
For details on how to write this section, refer to the ASR :ref:`Augmentation Configuration <asr-configs-augmentation-configurations>` section.

Check config yaml files in ``<NeMo_git_root>/tutorials/asr/conf`` to find the processors been used by speech classification models. 


Model Architecture Configurations
---------------------------------

Each configuration file should describe the model architecture being used for the experiment.
Models in the NeMo ASR collection need a ``encoder`` section and a ``decoder`` section, with the ``_target_`` field
specifying the module to use for each.

The following sections go into more detail about the specific configurations of each model architecture.

The :ref:`MatchboxNet <MarbleNet_model>` and :ref:`MarbleNet <MarbleNet_model>` models are very similar, and they are based on :ref:`QuartzNet <QuartzNet_model>`  and as 
such the components in their configs are very similar as well.

Decoder Configurations
------------------------

After features have been computed from ConvASREncoder, we pass the features to decoder to compute embeddings and then to compute log_probs 
for training models.

.. code-block:: yaml

  model:
    ...
    decoder:
      _target_: nemo.collections.asr.modules.ConvASRDecoderClassification
      feat_in: *enc_final_filters
      return_logits: true # return logits if true, else return softmax output
      pooling_type: 'avg' # AdaptiveAvgPool1d 'avg' or AdaptiveMaxPool1d 'max'


Fine-tuning Execution Flow Diagram
----------------------------------

When preparing your own training or fine-tuning scripts, please follow the execution flow diagram order for correct inference.

Depending on the type of model, there may be extra steps that must be performed -

* Speech Classification models - `Examples directory for Classification Models <https://github.com/NVIDIA/NeMo/blob/stable/examples/asr/speech_classification/README.md>`_

