NeMo Audio Configuration Files
==============================
This section describes the NeMo configuration file setup that is specific to models in the audio collection.
For general information about how to set up and run experiments that is common to all NeMo models (e.g. Experiment Manager and PyTorch Lightning trainer parameters), see the :doc:`../core/core` section.

The model section of the NeMo audio configuration files generally requires information about the dataset(s) being used, parameters for any augmentation being performed, as well as the model architecture specification.

Example configuration files for all of the NeMo audio models can be found in the
`config directory of the examples <https://github.com/NVIDIA/NeMo/tree/main/examples/audio/conf>`_.


.. _audio-configs-nemo-dataset-configuration:

NeMo Dataset Configuration
--------------------------

Training, validation, and test parameters are specified using the ``model.train_ds``, ``model.validation_ds``, and ``model.test_ds`` sections in the configuration file, respectively.
Depending on the task, there may be arguments specifying the sample rate or duration of the loaded audio examples. Some fields can be left out and specified via the command-line at runtime.
Refer to the `Dataset Processing Classes <./api.html#datasets>`__ section of the API for a list of datasets classes and their respective parameters.
An example train, validation and test datasets can be configured as follows:

.. code-block:: yaml

  model:
    sample_rate: 16000
    skip_nan_grad: false

    train_ds:
      manifest_filepath: ???
      input_key: audio_filepath # key of the input signal path in the manifest
      target_key: target_filepath # key of the target signal path in the manifest
      target_channel_selector: 0 # target signal is the first channel from files in target_key
      audio_duration: 4.0 # in seconds, audio segment duration for training
      random_offset: true # if the file is longer than audio_duration, use random offset to select a subsegment
      min_duration: ${model.train_ds.audio_duration}
      batch_size: 64 # batch size may be increased based on the available memory
      shuffle: true
      num_workers: 8
      pin_memory: true

    validation_ds:
      manifest_filepath: ???
      input_key: audio_filepath # key of the input signal path in the manifest
      target_key: target_filepath # key of the target signal path in the manifest
      target_channel_selector: 0 # target signal is the first channel from files in target_key
      batch_size: 64 # batch size may be increased based on the available memory
      shuffle: false
      num_workers: 4
      pin_memory: true

    test_ds:
      manifest_filepath: ???
      input_key: audio_filepath # key of the input signal path in the manifest
      target_key: target_filepath # key of the target signal path in the manifest
      target_channel_selector: 0 # target signal is the first channel from files in target_key
      batch_size: 1 # batch size may be increased based on the available memory
      shuffle: false
      num_workers: 4
      pin_memory: true

More information about online augmentation can found in the `masking example configuration <https://github.com/NVIDIA/NeMo/blob/main/examples/audio/conf/masking.yaml>`_


.. _audio-configs-lhotse-dataset-configuration:

Lhotse Dataset Configuration
----------------------------

Lhotse CutSet
~~~~~~~~~~~~~

An example train dataset in Lhotse CutSet format can be configured as follows:

.. code-block:: yaml

  train_ds:
    use_lhotse: true # enable Lhotse data loader
    cuts_path: ??? # path to Lhotse cuts manifest with input signals and the corresponding target signals (target signals should be in the custom "target_recording" field)
    truncate_duration: 4.00 # truncate audio to 4 seconds
    truncate_offset_type: random # if the file is longer than truncate_duration, use random offset to select a subsegment
    batch_size: 64 # batch size may be increased based on the available memory
    shuffle: true
    num_workers: 8
    pin_memory: true

Lhotse CutSet with Online Augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An example train dataset in Lhotse CutSet format using online augmentation with room impulse response (RIR) convolution and additive noise can be configured as follows:

.. code-block:: yaml

  train_ds:
    use_lhotse: true # enable Lhotse data loader
    cuts_path: ??? # path to Lhotse cuts manifest with speech signals for augmentation (including custom "target_recording" field with the same signals)
    truncate_duration: 4.00 # truncate audio to 4 seconds
    truncate_offset_type: random # if the file is longer than truncate_duration, use random offset to select a subsegment
    batch_size: 64 # batch size may be increased based on the available memory
    shuffle: true
    num_workers: 8
    pin_memory: true
    rir_enabled: true # enable room impulse response augmentation
    rir_path: ??? # path to Lhotse recordings manifest with room impulse response signals
    noise_path: ??? # path to Lhotse cuts manifest with noise signals

A configuration file with Lhotse online augmentation can found in the `online augmentation example configuration <https://github.com/NVIDIA/NeMo/blob/main/examples/audio/conf/masking_with_online_augmentation.yaml>`_.
More information about the online augmentation can be found in the `tutorial notebook <https://github.com/NVIDIA/NeMo/blob/main/tutorials/audio/speech_enhancement/Speech_Enhancement_with_Online_Augmentation.ipynb>`_.


Lhotse Shar
~~~~~~~~~~~

An example train dataset in Lhotse shar format can be configured as follows:

.. code-block:: yaml

  train_ds:
    shar_path: ???
    use_lhotse: true
    truncate_duration: 4.00 # truncate audio to 4 seconds
    truncate_offset_type: random
    batch_size: 8 # batch size may be increased based on the available memory
    shuffle: true
    num_workers: 8
    pin_memory: true


A configuration file with Lhotse shar format can found in the `SSL pretraining example configuration <https://github.com/NVIDIA/NeMo/blob/main/examples/audio/conf/flow_matching_generative_ssl_pretraining.yaml>`_.


Dataset Reweighting with Temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When combining multiple datasets using nested ``input_cfg`` groups, you can control the sampling distribution using the ``reweight_temperature`` parameter. This feature allows you to balance dataset sampling without manually recalculating weights **when adding or removing datasets**.

The temperature scaling formula is:

.. math::

    \hat{w}_i = \frac{w_i^{\tau}}{\sum_{j} w_j^{\tau}}

where :math:`w_i` is the original weight of dataset :math:`i`, :math:`\tau` is the temperature, and :math:`\hat{w}_i` is the normalized sampling probability.

**How Temperature Works:**

- ``temperature = 1.0``: Preserves original weight ratios (neutral, no reweighting)
- ``temperature = 0.0``: Equalizes all datasets (each gets equal probability regardless of original weights)
- ``0 < temperature < 1.0``: Over-samples smaller datasets relative to larger ones
- ``temperature > 1.0``: Amplifies differences between dataset weights

**Configuration Options:**

The ``reweight_temperature`` parameter accepts two formats:

1. **Scalar value** (applied to all nesting levels, warning logged):

.. code-block:: yaml

    train_ds:
      use_lhotse: true
      reweight_temperature: 0.5  # Applied to all levels, warning logged
      input_cfg:
        - type: group
          input_cfg:
            - type: lhotse_shar
              shar_path: /path/to/dataset1
              weight: 900
            - type: lhotse_shar
              shar_path: /path/to/dataset2
              weight: 100
        - type: lhotse_shar
          shar_path: /path/to/dataset3
          weight: 200
        - type: nemo_tarred
          manifest_filepath: /path/to/dataset4/manifest.json
          tarred_audio_filepath: /path/to/dataset4/audio.tar
          weight: 300

2. **List matching maximum nesting depth** (one temperature per level):

.. code-block:: yaml

    train_ds:
      use_lhotse: true
      reweight_temperature: [1.0, 0.0]  # Level 1: preserve ratios, Level 2: equalize
      input_cfg:
        - type: group
          weight: 0.7
          input_cfg:
            - type: lhotse_shar
              shar_path: /path/to/dataset1
              weight: 600
            - type: lhotse_shar
              shar_path: /path/to/dataset2
              weight: 400
        - type: group
          weight: 0.3
          input_cfg:
            - type: lhotse_shar
              shar_path: /path/to/dataset3
              weight: 100

.. note::

    If ``reweight_temperature`` is provided as a list, its length **must** exactly match the maximum nesting depth of ``input_cfg``.
    A mismatch (too few or too many values) raises a ``ValueError``.
    Use a scalar value instead if you want the same temperature applied uniformly to all levels.

**Maximum Nesting Depth Calculation:**

The maximum nesting depth is calculated as the maximum depth of ``input_cfg`` keys in the configuration. Sibling groups at the same level share the same temperature value.

.. code-block:: yaml

    # This has maximum nesting depth = 2
    input_cfg:                          # Level 1
      - type: group
        input_cfg:                      # Level 2
          - type: lhotse_shar
      - type: group                     # Same level as above (sibling)
        input_cfg:                      # Level 2 (same as above)
          - type: lhotse_shar

**Example: Balancing Multiple Task Groups**

.. code-block:: yaml

    train_ds:
      use_lhotse: true
      reweight_temperature: [1.0, 0.0]  # Level 1: Preserve task ratios, Level 2: Equalize within tasks
      input_cfg:
        - type: group
          weight: 0.7
          tags:
            task: asr
          input_cfg:
            - type: nemo_tarred
              manifest_filepath: /path/to/asr1/manifest.json
              tarred_audio_filepath: /path/to/asr1/audio.tar
              weight: 600  # Large dataset
            - type: nemo_tarred
              manifest_filepath: /path/to/asr2/manifest.json
              tarred_audio_filepath: /path/to/asr2/audio.tar
              weight: 100  # Small dataset (will be upsampled with temp=0.0)
        - type: group
          weight: 0.3
          tags:
            task: ast
          input_cfg:
            - type: nemo_tarred
              manifest_filepath: /path/to/ast1/manifest.json
              tarred_audio_filepath: /path/to/ast1/audio.tar
              weight: 50
            - type: nemo_tarred
              manifest_filepath: /path/to/ast2/manifest.json
              tarred_audio_filepath: /path/to/ast2/audio.tar
              weight: 200

In this example:

- Level 1 temperature is ``1.0``: The 70/30 split between ASR and AST groups is preserved
- Level 2 temperature is ``0.0``: Within each group, all datasets are sampled equally regardless of their original weights


Model Architecture Configuration
--------------------------------
Each configuration file should describe the model architecture being used for the experiment.
An example of a simple predictive model configuration is shown below:

.. code-block:: yaml

  model:
    type: predictive
    sample_rate: 16000
    skip_nan_grad: false
    num_outputs: 1
    normalize_input: true # normalize the input signal to 0dBFS

    train_ds:
      manifest_filepath: ???
      input_key: noisy_filepath
      target_key: clean_filepath
      audio_duration: 2.00 # trim audio to 2 seconds
      random_offset: true
      normalization_signal: input_signal
      batch_size: 8 # batch size may be increased based on the available memory
      shuffle: true
      num_workers: 8
      pin_memory: true

    validation_ds:
      manifest_filepath: ???
      input_key: noisy_filepath
      target_key: clean_filepath
      batch_size: 8
      shuffle: false
      num_workers: 4
      pin_memory: true

    encoder:
      _target_: nemo.collections.audio.modules.transforms.AudioToSpectrogram
      fft_length: 510 # Number of subbands in the STFT = fft_length // 2 + 1 = 256
      hop_length: 128
      magnitude_power: 0.5
      scale: 0.33

    decoder:
      _target_: nemo.collections.audio.modules.transforms.SpectrogramToAudio
      fft_length: ${model.encoder.fft_length}
      hop_length: ${model.encoder.hop_length}
      magnitude_power: ${model.encoder.magnitude_power}
      scale: ${model.encoder.scale}

    estimator:
      _target_: nemo.collections.audio.parts.submodules.ncsnpp.SpectrogramNoiseConditionalScoreNetworkPlusPlus
      in_channels: 1 # single-channel noisy input
      out_channels: 1 # single-channel estimate
      num_res_blocks: 3 # increased number of res blocks
      pad_time_to: 64 # pad to 64 frames for the time dimension
      pad_dimension_to: 0 # no padding in the frequency dimension

    loss:
      _target_: nemo.collections.audio.losses.MSELoss # computed in the time domain

    metrics:
      val:
        sisdr: # output SI-SDR
          _target_: torchmetrics.audio.ScaleInvariantSignalDistortionRatio

    optim:
      name: adam
      lr: 1e-4
      # optimizer arguments
      betas: [0.9, 0.999]
      weight_decay: 0.0


Complete configuration file can found in the `example configuration <https://github.com/NVIDIA/NeMo/blob/main/examples/audio/conf/predictive.yaml>`_.


Finetuning Configuration
--------------------------

All scripts support easy finetuning by partially/fully loading the pretrained weights from a checkpoint into the currently instantiated model.
Note that the currently instantiated model should have parameters that match the pre-trained checkpoint so the weights may load properly.

Pre-trained weights can be provided by:

* Providing a path to a NeMo model (via ``init_from_nemo_model``)
* Providing a name of a pretrained NeMo model (which will be downloaded via the cloud) (via ``init_from_pretrained_model``)


Training from scratch
~~~~~~~~~~~~~~~~~~~~~

A model can be trained from scratch using the following command:

.. code-block:: shell

    python examples/audio/audio_to_audio_train.py \
        --config-path=<path to dir of configs>
        --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.devices=1 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50

Fine-tuning via a NeMo model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A model can be finetuned from an existing NeMo model using the following command:

.. code-block:: shell
    :emphasize-lines: 9

    python examples/audio/audio_to_audio_train.py \
        --config-path=<path to dir of configs>
        --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.devices=1 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        +init_from_nemo_model="<path to .nemo model file>"


Fine-tuning via a NeMo pretrained model name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A model can be finetuned from an pre-trained NeMo model using the following command:

.. code-block:: shell
    :emphasize-lines: 9

    python examples/audio/audio_to_audio_train.py \
        --config-path=<path to dir of configs>
        --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.devices=1 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        +init_from_pretrained_model="<name of pretrained checkpoint>"

