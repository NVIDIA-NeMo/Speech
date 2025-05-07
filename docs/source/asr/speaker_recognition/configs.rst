NeMo Speaker Recognition Configuration Files
============================================

This page covers NeMo configuration file setup that is specific to speaker recognition models.
For general information about how to set up and run experiments that is common to all NeMo models (e.g.
experiment manager and PyTorch Lightning trainer parameters), see the :doc:`../../core/core` page.

The model section of NeMo speaker recognition configuration files will generally require information about the dataset(s) being
used, the preprocessor for audio files, parameters for any augmentation being performed, as well as the
model architecture specification.
The sections on this page cover each of these in more detail.

Example configuration files for all of the Speaker related scripts can be found in the
config directory of the examples ``{NEMO_ROOT/examples/speaker_tasks/recognition/conf}``.


Dataset Configuration
---------------------

Training, validation, and test parameters are specified using the ``train_ds``, ``validation_ds``, and
``test_ds`` sections of your configuration file, respectively.
Depending on the task, you may have arguments specifying the sample rate of your audio files, max time length to consider for each audio file , whether or not to shuffle the dataset, and so on.
You may also decide to leave fields such as the ``manifest_filepath`` blank, to be specified via the command line
at run time.

Any initialization parameters that are accepted for the Dataset class used in your experiment
can be set in the config file.

An example TitaNet train and validation configuration could look like (``{NEMO_ROOT}examples/speaker_tasks/recognition/conf/titanet-large.yaml``):

.. code-block:: yaml

  model:
    train_ds:
      manifest_filepath: ???
      sample_rate: 16000
      labels: None   # finds labels based on manifest file
      batch_size: 32
      trim_silence: False
      shuffle: True

    validation_ds:
      manifest_filepath: ???
      sample_rate: 16000
      labels: None   # Keep None, to match with labels extracted during training
      batch_size: 32
      shuffle: False    # No need to shuffle the validation data

      
If you would like to use tarred dataset, have a look at the ASR :ref:`Tarred Datasets <Tarred_Datasets>` section.


Preprocessor Configuration
--------------------------
Preprocessor helps to compute MFCC or mel spectrogram features that are given as inputs to model. 
For details on how to write this section, refer to :ref:`Preprocessor Configuration <asr-configs-preprocessor-configuration>`.


Augmentation Configurations
---------------------------

For TitaNet training we use on-the-fly augmentations with MUSAN and RIR impulses using ``noise`` augmentor section

The following example sets up musan augmentation with audio files taken from manifest path and 
minimum and maximum SNR specified with min_snr and max_snr respectively. This section can be added to 
``train_ds`` part in model

.. code-block:: yaml

  model:
    ...
    train_ds:
      ...
      augmentor:
        noise:
          manifest_path: /path/to/musan/manifest_file
          prob: 0.2  # probability to augment the incoming batch audio with augmentor data
          min_snr_db: 5 
          max_snr_db: 15        


See the :class:`nemo.collections.asr.parts.preprocessing.perturb.AudioAugmentor`  API section for more details.


Model Architecture Configurations
---------------------------------

Each configuration file should describe the model architecture being used for the experiment.
Models in the NeMo ASR collection need a ``encoder`` section and a ``decoder`` section, with the ``_target_`` field
specifying the module to use for each.

The following sections go into more detail about the specific configurations of each model architecture.

For more information about the TitaNet Encoder models, see the :doc:`Models <./models>` page.

Decoder Configurations
------------------------

After features have been computed from TitaNet encoder, we pass these features to the decoder to compute embeddings and then to compute log probabilities
for training models.

.. code-block:: yaml

  model:
    ...
    decoder:
      _target_: nemo.collections.asr.modules.SpeakerDecoder
      feat_in: *enc_feat_out
      num_classes: 7205  # Total number of classes in voxceleb1,2 training manifest file 
      pool_mode: attention # xvector, attention
      emb_sizes: 192 # number of intermediate emb layers. can be comma separated for additional layers like 512,512
      angular: true # if true then loss will be changed to angular softmax loss and consider scale and margin from loss section else train with cross-entropy loss
    
    loss:
      scale: 30
      margin 0.2
