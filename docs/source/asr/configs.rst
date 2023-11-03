NeMo ASR Configuration Files
============================

This section describes the NeMo configuration file setup that is specific to models in the ASR collection. For general information
about how to set up and run experiments that is common to all NeMo models (e.g. Experiment Manager and PyTorch Lightning trainer
parameters), see the :doc:`../core/core` section.

The model section of the NeMo ASR configuration files generally requires information about the dataset(s) being used, the preprocessor
for audio files, parameters for any augmentation being performed, as well as the model architecture specification. The sections on
this page cover each of these in more detail.

Example configuration files for all of the NeMo ASR scripts can be found in the
`config directory of the examples <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/conf>`_.


Dataset Configuration
---------------------

Training, validation, and test parameters are specified using the ``train_ds``, ``validation_ds``, and
``test_ds`` sections in the configuration file, respectively. Depending on the task, there may be arguments specifying the sample rate
of the audio files, the vocabulary of the dataset (for character prediction), whether or not to shuffle the dataset, and so on. You may
also decide to leave fields such as the ``manifest_filepath`` blank, to be specified via the command-line at runtime.

Any initialization parameter that is accepted for the Dataset class used in the experiment can be set in the config file.
Refer to the `Datasets <./api.html#Datasets>`__ section of the API for a list of Datasets and their respective parameters.

An example ASR train and validation configuration should look similar to the following:

.. code-block:: yaml

  # Specified at the beginning of the config file
  labels: &labels [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

  model:
    train_ds:
      manifest_filepath: ???
      sample_rate: 16000
      labels: *labels   # Uses the labels above
      batch_size: 32
      trim_silence: True
      max_duration: 16.7
      shuffle: True
      num_workers: 8
      pin_memory: true
      # tarred datasets
      is_tarred: false # If set to true, uses the tarred version of the Dataset
      tarred_audio_filepaths: null     # Not used if is_tarred is false
      shuffle_n: 2048                  # Not used if is_tarred is false
      # bucketing params
      bucketing_strategy: "synced_randomized"
      bucketing_batch_size: null
      bucketing_weights: null

    validation_ds:
      manifest_filepath: ???
      sample_rate: 16000
      labels: *labels   # Uses the labels above
      batch_size: 32
      shuffle: False    # No need to shuffle the validation data
      num_workers: 8
      pin_memory: true

There are two ways to test/validate on more than one manifest:

- Specify a list in the `manifest_filepath` field. Results will be reported for each, the first one being used for overall loss / WER (specify `val_dl_idx` if you wish to change that). In this case, all manifests will share configuration parameters.
- Use the ds_item key and pass a list of config objects to it. This allows you to use differently configured datasets for validation, e.g.

.. code-block:: yaml

  model:
    validation_ds:
      ds_item:
      - name: dataset1
        manifest_filepath: ???
        # Config parameters for dataset1
        ...
      - name: dataset2
        manifest_filepath: ???
        # Config parameters for dataset2
        ...

By default, dataloaders are set up when the model is instantiated. However, dataloader setup can be deferred to
model's `setup()` method by setting ``defer_setup`` in the configuration.

For example, training data setup can be deferred as follows:

.. code-block:: yaml

  model:
    train_ds:
      # Configure training data as usual
      ...
      # Defer train dataloader setup from `__init__` to `setup`
      defer_setup: true


Preprocessor Configuration
--------------------------

If you are loading audio files for your experiment, you will likely want to use a preprocessor to convert from the
raw audio signal to features (e.g. mel-spectrogram or MFCC). The ``preprocessor`` section of the config specifies the audio
preprocessor to be used via the ``_target_`` field, as well as any initialization parameters for that preprocessor.

An example of specifying a preprocessor is as follows:

.. code-block:: yaml

  model:
    ...
    preprocessor:
      # _target_ is the audio preprocessor module you want to use
      _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
      normalize: "per_feature"
      window_size: 0.02
      ...
      # Other parameters for the preprocessor

Refer to the `Audio Preprocessors <./api.html#Audio Preprocessors>`__ API section for the preprocessor options, expected arguments,
and defaults.

Augmentation Configurations
---------------------------

There are a few on-the-fly spectrogram augmentation options for NeMo ASR, which can be specified by the
configuration file using a ``spec_augment`` section.

For example, there are options for `Cutout <https://arxiv.org/abs/1708.04552>`_ and
`SpecAugment <https://arxiv.org/abs/1904.08779>`_ available via the ``SpectrogramAugmentation`` module.

The following example sets up both ``Cutout`` (via the ``rect_*`` parameters) and ``SpecAugment`` (via the ``freq_*``
and ``time_*`` parameters).

.. code-block:: yaml

  model:
    ...
    spec_augment:
      _target_: nemo.collections.asr.modules.SpectrogramAugmentation
      # Cutout parameters
      rect_masks: 5   # Number of rectangles to cut from any given spectrogram
      rect_freq: 50   # Max cut of size 50 along the frequency dimension
      rect_time: 120  # Max cut of size 120 along the time dimension
      # SpecAugment parameters
      freq_masks: 2   # Cut two frequency bands
      freq_width: 15  # ... of width 15 at maximum
      time_masks: 5    # Cut out 10 time bands
      time_width: 25  # ... of width 25 at maximum

You can use any combination of ``Cutout``, frequency/time ``SpecAugment``, or neither of them.

With NeMo ASR, you can also add augmentation pipelines that can be used to simulate various kinds of noise
added to audio in the channel. Augmentors in a pipeline are applied on the audio data read in the data layer. Online
augmentors can be specified in the config file using an ``augmentor`` section in ``train_ds``. The following example
adds an augmentation pipeline that first adds white noise to an audio sample with a probability of 0.5 and at a level
randomly picked between -50 dB and -10 dB and then passes the resultant samples through a room impulse response randomly
picked from the manifest file provided for ``impulse`` augmentation in the config file.

.. code-block:: yaml

  model:
    ...
    train_ds:
    ...
        augmentor:
            white_noise:
                prob: 0.5
                min_level: -50
                max_level: -10
            impulse:
                prob: 0.3
                manifest_path: /path/to/impulse_manifest.json

Refer to the `Audio Augmentors <./api.html#Audio Augmentors>`__ API section for more details.

Tokenizer Configurations
------------------------

Some models utilize sub-word encoding via an external tokenizer instead of explicitly defining their vocabulary.

For such models, a ``tokenizer`` section is added  to the model config. ASR models currently support two types of
custom tokenizers:

- Google Sentencepiece tokenizers (tokenizer type of ``bpe`` in the config)
- HuggingFace WordPiece tokenizers (tokenizer type of ``wpe`` in the config)
- Aggregate tokenizers ((tokenizer type of ``agg`` in the config), see below)

In order to build custom tokenizers, refer to the ``ASR_with_Subword_Tokenization`` notebook available in the
ASR tutorials directory.

The following example sets up a ``SentencePiece Tokenizer`` at a path specified by the user:

.. code-block:: yaml

  model:
    ...
    tokenizer:
      dir: "<path to the directory that contains the custom tokenizer files>"
      type: "bpe"  # can be "bpe" or "wpe"

The Aggregate (``agg``) tokenizer feature makes it possible to combine tokenizers in order to train multilingual
models. The config file would look like this:

.. code-block:: yaml

  model:
    ...
    tokenizer:
      type: "agg"  # aggregate tokenizer
      langs:
        en:
          dir: "<path to the directory that contains the tokenizer files>"
          type: "bpe"  # can be "bpe" or "wpe"
        es:
          dir: "<path to the directory that contains the tokenizer files>"
          type: "bpe"  # can be "bpe" or "wpe"

In the above config file, each language is associated with its own pre-trained tokenizer, which gets assigned
a token id range in the order the tokenizers are listed. To train a multilingual model, one needs to populate the
``lang`` field in the manifest file, allowing the routing of each sample to the correct tokenizer. At inference time,
the routing is done based on the inferred token id range.

For models which utilize sub-word tokenization, we share the decoder module (``ConvASRDecoder``) with character tokenization models.
All parameters are shared, but for models which utilize sub-word encoding, there are minor differences when setting up the config. For
such models, the tokenizer is utilized to fill in the missing information when the model is constructed automatically.

For example, a decoder config corresponding to a sub-word tokenization model should look similar to the following:

.. code-block:: yaml

  model:
    ...
    decoder:
      _target_: nemo.collections.asr.modules.ConvASRDecoder
      feat_in: *enc_final
      num_classes: -1  # filled with vocabulary size from tokenizer at runtime
      vocabulary: []  # filled with vocabulary from tokenizer at runtime


On-the-fly Code Switching
-------------------------

Nemo supports creating code-switched synthetic utterances on-the-fly during training/validation/testing. This allows you to create ASR models which
support intra-utterance code switching. If you have Nemo formatted audio data on disk (either JSON manifests or tarred audio data), you
can easily mix as many of these audio sources together as desired by adding some extra parameters to your `train_ds`, `validation_ds`, and `test_ds`.

Please note that this allows you to mix any kind of audio sources together to create synthetic utterances which sample from all sources. The most
common use case for this is blending different languages together to create a multilingual code-switched model, but you can also blend
together different audio sources from the same languages (or language families), to create noise robust data, or mix fast and slow speech from the
same language.

For multilingual code-switched models, we recommend using AggTokenizer for your Tokenizer if mixing different languages.

The following example shows how to mix 3 different languages: English (en), German (de), and Japanese (ja) added to the `train_ds` model block, however
you can add similar logic to your `validation_ds` and `test_ds` blocks for on-the-fly code-switched validation and test data too. This example mixes
together 3 languages, but you can use as many as you want. However, be advised that the more languages you add, the higher your `min_duration` and `max_duration`
need to be set to ensure all languages are sampled into each synthetic utterance, and setting these hyperparameters higher will use more VRAM per mini-batch during
training and evaluation.

.. code-block:: yaml

  model:
    train_ds:
      manifest_filepath: [/path/to/EN/tarred_manifest.json, /path/to/DE/tarred_manifest.json, /path/to/JA/tarred_manifest.json]
      tarred_audio_filepaths: ['/path/to/EN/tars/audio__OP_0..511_CL_.tar', '/path/to/DE/tars/audio__OP_0..1023_CL_.tar', '/path/to/JA/tars/audio__OP_0..2047_CL_.tar']
      is_code_switched: true
      is_tarred: true
      shuffle: true
        code_switched:              # add this block for code-switching
          min_duration: 12          # the minimum number of seconds for each synthetic code-switched utterance
          max_duration: 20          # the maximum number of seconds for each synthetic code-switched utterance
          min_monolingual: 0.3      # the minimum percentage of utterances which will be pure monolingual (0.3 = 30%)
          probs: [0.25, 0.5, 0.25]  # the probability to sample each language (matches order of `language` above) if not provided, assumes uniform distribution
          force_monochannel: true   # if your source data is multi-channel, then setting this to True will force the synthetic utterances to be mono-channel
          sampling_scales: 0.75     # allows you to down/up sample individual languages. Can set this as an array for individual languages, or a scalar for all languages
          seed: 123                 # add a seed for replicability in future runs (highly useful for `validation_ds` and `test_ds`)


Model Architecture Configurations
---------------------------------

Each configuration file should describe the model architecture being used for the experiment. Models in the NeMo ASR collection need
an ``encoder`` section and a ``decoder`` section, with the ``_target_`` field specifying the module to use for each.

Here is the list of the parameters in the model section which are shared among most of the ASR models:

+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| **Parameter**           | **Datatype**     | **Description**                                                                                               | **Supported Values**            |
+=========================+==================+===============================================================================================================+=================================+
| :code:`log_prediction`  | bool             | Whether a random sample should be printed in the output at each step, along with its predicted transcript.    |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`ctc_reduction`   | string           | Specifies the reduction type of CTC loss. Defaults to ``mean_batch`` which would take the average over the    | :code:`none`,                   |
|                         |                  | batch after taking the average over the length of each sample.                                                | :code:`mean_batch`              |
|                         |                  |                                                                                                               | :code:`mean`, :code:`sum`       |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+

The following sections go into more detail about the specific configurations of each model architecture.

For more information about the ASR models, refer to the :doc:`Models <./models>` section.

Jasper and QuartzNet
~~~~~~~~~~~~~~~~~~~~

The `Jasper <./models.html#Jasper>`__ and `QuartzNet <./models.html#QuartzNet>`__ models are very similar, and as such the components in their
configs are very similar as well.

Both architectures use the ``ConvASREncoder`` for the ``encoder``, with parameters detailed in the table below. The encoder parameters
include details about the Jasper/QuartzNet ``[BxR]`` encoder architecture, including how many blocks to use (``B``), how many times
to repeat each sub-block (``R``), and the convolution parameters for each block.

The number of blocks ``B`` is determined by the number of list elements under ``jasper`` minus the one prologue and two epilogue blocks.
The number of sub-blocks ``R`` is determined by setting the ``repeat`` parameter.

To use QuartzNet (which uses more compact time-channel separable convolutions) instead of Jasper, add :code:`separable: true` to all
but the last block in the architecture.

Change the parameter name ``jasper``.

+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+-------------------------------------+
| **Parameter**           | **Datatype**     | **Description**                                                                                               | **Supported Values**                |
+=========================+==================+===============================================================================================================+=====================================+
| :code:`feat_in`         | int              | The number of input features. Should be equal to :code:`features` in the preprocessor parameters.             |                                     |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+-------------------------------------+
| :code:`activation`      | string           | Which activation function to use in the encoder.                                                              | :code:`hardtanh`, :code:`relu`,     |
|                         |                  |                                                                                                               | :code:`selu`, :code:`swish`         |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+-------------------------------------+
| :code:`conv_mask`       | bool             | Whether to use masked convolutions in the encoder. Defaults to ``true``.                                      |                                     |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+-------------------------------------+
| :code:`jasper`          |                  | A list of blocks that specifies your encoder architecture. Each entry in this list represents one block in    |                                     |
|                         |                  | the architecture and contains the parameters for that block, including convolution parameters, dropout, and   |                                     |
|                         |                  | the number of times the block is repeated. Refer to the `Jasper <https://arxiv.org/pdf/1904.03288.pdf>`_ and  |                                     |
|                         |                  | `QuartzNet <https://arxiv.org/pdf/1910.10261.pdf>`_ papers for details about specific model configurations.   |                                     |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+-------------------------------------+

A QuartzNet 15x5 (fifteen blocks, each sub-block repeated five times) encoder configuration should look similar to the following example:

.. code-block:: yaml

  # Specified at the beginning of the file for convenience
  n_mels: &n_mels 64    # Used for both the preprocessor and encoder as number of input features
  repeat: &repeat 5     # R=5
  dropout: &dropout 0.0
  separable: &separable true  # Set to true for QN. Set to false for Jasper.

  model:
    ...
    encoder:
      _target_: nemo.collections.asr.modules.ConvASREncoder
      feat_in: *n_mels  # Should match "features" in the preprocessor.
      activation: relu
      conv_mask: true

      jasper:   # This field name should be "jasper" for both types of models.

      # Prologue block
      - dilation: [1]
        dropout: *dropout
        filters: 256
        kernel: [33]
        repeat: 1   # Prologue block is not repeated.
        residual: false
        separable: *separable
        stride: [2]

      # Block 1
      - dilation: [1]
        dropout: *dropout
        filters: 256
        kernel: [33]
        repeat: *repeat
        residual: true
        separable: *separable
        stride: [1]

      ... # Entries for blocks 2~14

      # Block 15
      - dilation: [1]
        dropout: *dropout
        filters: 512
        kernel: [75]
        repeat: *repeat
        residual: true
        separable: *separable
        stride: [1]

      # Two epilogue blocks
      - dilation: [2]
        dropout: *dropout
        filters: 512
        kernel: [87]
        repeat: 1   # Epilogue blocks are not repeated
        residual: false
        separable: *separable
        stride: [1]

      - dilation: [1]
        dropout: *dropout
        filters: &enc_filters 1024
        kernel: [1]
        repeat: 1   # Epilogue blocks are not repeated
        residual: false
        stride: [1]

Both Jasper and QuartzNet use the ``ConvASRDecoder`` as the decoder. The decoder parameters are detailed in the following table.

+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| **Parameter**           | **Datatype**     | **Description**                                                                                               | **Supported Values**            |
+=========================+==================+===============================================================================================================+=================================+
| :code:`feat_in`         | int              | The number of input features to the decoder. Should be equal to the number of filters in the last block of    |                                 |
|                         |                  | the encoder.                                                                                                  |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`vocabulary`      | list             | A list of the valid output characters for your model. For example, for an English dataset, this could be a    |                                 |
|                         |                  | list of all lowercase letters, space, and apostrophe.                                                         |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`num_classes`     | int              | Number of output classes, i.e. the length of :code:`vocabulary`.                                              |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+

For example, a decoder config corresponding to the encoder above should look similar to the following:

.. code-block:: yaml

  model:
    ...
    decoder:
      _target_: nemo.collections.asr.modules.ConvASRDecoder
      feat_in: *enc_filters
      vocabulary: *labels
      num_classes: 28   # Length of the vocabulary list

Citrinet
~~~~~~~~

The `Citrinet <./models.html#Citrinet>`__ and `QuartzNet <./models.html#QuartzNet>`__ models are very similar, and as such the
components in their configs are very similar as well. Citrinet utilizes Squeeze and Excitation, as well as sub-word tokenization, in
contrast to QuartzNet. Depending on the dataset, we utilize different tokenizers. For Librispeech, we utilize the HuggingFace WordPiece
tokenizer, and for all other datasets we utilize the Google Sentencepiece tokenizer - usually the ``unigram`` tokenizer type.

Both architectures use the ``ConvASREncoder`` for the ``encoder``, with parameters detailed above. The encoder parameters include
details about the Citrinet-C encoder architecture, including how many filters are used per channel (``C``). The Citrinet-C
configuration is a shortform notation for Citrinet-21x5xC, such that ``B = 21`` and ``R = 5`` are the default and should generally
not be changed.

To use Citrinet instead of QuartzNet, refer to the ``citrinet_512.yaml`` configuration found inside the ``examples/asr/conf/citrinet``
directory. Citrinet is primarily comprised of the same :class:`~nemo.collections.asr.parts.submodules.jasper.JasperBlock` as ``Jasper`` or
``QuartzNet``.

While the configs for Citrinet and QuartzNet are similar, we note the additional flags used for Citrinet below. Refer to the
``JasperBlock`` documentation for the meaning of these arguments.

+---------------------------+------------------+-----------------------------------------------------------------------------------------------------------+-----------------------------------+
| **Parameter**             | **Datatype**     | **Description**                                                                                           | **Supported Values**              |
+===========================+==================+===========================================================================================================+===================================+
| :code:`se`                | bool             | Whether to apply squeeze-and-excitation mechanism or not.                                                 | :code:`true` or :code:`false`     |
+---------------------------+------------------+-----------------------------------------------------------------------------------------------------------+-----------------------------------+
| :code:`se_context_size`   | int              | SE context size. -1 means global context.                                                                 | :code:`-1` or :code:`+ve int`     |
+---------------------------+------------------+-----------------------------------------------------------------------------------------------------------+-----------------------------------+
| :code:`stride_last`       | bool             | Stride on the final repeated block or all repeated blocks.                                                | :code:`true` or :code:`false`     |
+---------------------------+------------------+-----------------------------------------------------------------------------------------------------------+-----------------------------------+
| :code:`residual_mode`     | str              | Type of residual branch to construct.                                                                     | :code:`"add"` or                  |
|                           |                  | Can be pointwise residual addition or pointwise strided residual attention                                | :code:`"stride_add"`              |
+---------------------------+------------------+-----------------------------------------------------------------------------------------------------------+-----------------------------------+

A Citrinet-512 config should look similar to the following:

.. code-block:: yaml

  model:
    ...
    # Specify some defaults across the entire model
    model_defaults:
      repeat: 5
      dropout: 0.1
      separable: true
      se: true
      se_context_size: -1
    ...
    encoder:
      _target_: nemo.collections.asr.modules.ConvASREncoder
      feat_in: *n_mels  # Should match "features" in the preprocessor.
      activation: relu
      conv_mask: true

      jasper:   # This field name should be "jasper" for the JasperBlock (which constructs Citrinet).

      # Prologue block
      - filters: 512
        repeat: 1
        kernel: [5]
        stride: [1]
        dilation: [1]
        dropout: 0.0
        residual: false
        separable: ${model.model_defaults.separable}
        se: ${model.model_defaults.se}
        se_context_size: ${model.model_defaults.se_context_size}

      # Block 1
      - filters: 512
        repeat: ${model.model_defaults.repeat}
        kernel: [11]
        stride: [2]
        dilation: [1]
        dropout: ${model.model_defaults.dropout}
        residual: true
        separable: ${model.model_defaults.separable}
        se: ${model.model_defaults.se}
        se_context_size: ${model.model_defaults.se_context_size}
        stride_last: true
        residual_mode: "stride_add"

      ... # Entries for blocks 2~21

      # Block 22
      - filters: 512
        repeat: ${model.model_defaults.repeat}
        kernel: [39]
        stride: [1]
        dilation: [1]
        dropout: ${model.model_defaults.dropout}
        residual: true
        separable: ${model.model_defaults.separable}
        se: ${model.model_defaults.se}
        se_context_size: ${model.model_defaults.se_context_size}

      # Epilogue block

      - filters: &enc_final 640
        repeat: 1
        kernel: [41]
        stride: [1]
        dilation: [1]
        dropout: 0.0
        residual: false
        separable: ${model.model_defaults.separable}
        se: ${model.model_defaults.se}
        se_context_size: ${model.model_defaults.se_context_size}

As mentioned above, Citrinet uses the ``ConvASRDecoder`` as the decoder layer similar to QuartzNet. Only the configuration must be
changed slightly as Citrinet utilizes sub-word tokenization.

.. note::
    The following information is relevant to any of the above models that implements its encoder as an :class:`~nemo.collections.asr.modules.conv_asr.ConvASREncoder`, and utilizes the ``SqueezeExcite`` mechanism.

The ``SqueezeExcite`` block within a :class:`~nemo.collections.asr.modules.conv_asr.ConvASREncoder` network can be modified to utilize a different context window after the model has been instantiated (even after the model has been trained) so as to evaluate the model with limited context. This can be achieved using the :meth:`~nemo.collections.asr.parts.mixins.mixins.ASRModuleMixin.change_conv_asr_se_context_window`

.. code-block:: python

    # Here, model can be any model that has a `ConvASREncoder` as its encoder, and utilized `SqueezeExcite` blocks
    # `context_window` : It is an integer representing the number of timeframes (each corresponding to some window stride).
    # `update_config` : Bool flag which determines whether the config of the model should be updated to reflect the new context window.

    # Here, we specify that 128 timeframes of 0.01s stride should be the context window
    # This is equivalent to 128 * 0.01s context window for `SqueezeExcite`
    model.change_conv_asr_se_context_window(context_window=128, update_config=True)

Conformer-CTC
~~~~~~~~~~~~~

The config files for Conformer-CTC model contain character-based encoding and sub-word encoding at
``<NeMo_git_root>/examples/asr/conf/conformer/conformer_ctc_char.yaml`` and ``<NeMo_git_root>/examples/asr/conf/conformer/conformer_ctc_bpe.yaml``
respectively. Some components of the configs of `Conformer-CTC <./models.html#Conformer-CTC>`__ include the following datasets:

* ``train_ds``, ``validation_ds``, and ``test_ds``
* opimizer (``optim``)
* augmentation (``spec_augment``)
* ``decoder``
* ``trainer``
* ``exp_manager``

These datasets are similar to other ASR models like `QuartzNet <./models.html#QuartzNet>`__. There should be a tokenizer section where you can
specify the tokenizer if you want to use sub-word encoding instead of character-based encoding.


The encoder section includes the details about the Conformer-CTC encoder architecture. You may find more information in the
config files and also :ref:`nemo.collections.asr.modules.ConformerEncoder <conformer-encoder-api>`.

Squeezeformer-CTC
~~~~~~~~~~~~~~~~~

The config files for Squeezeformer-CTC model contain character-based encoding and sub-word encoding at
``<NeMo_git_root>/examples/asr/conf/squeezeformer/squeezeformer_ctc_char.yaml`` and ``<NeMo_git_root>/examples/asr/conf/squeezeformer/squeezeformer_ctc_bpe.yaml``
respectively. Components of the configs of `Squeezeformer-CTC <./models.html#Squeezeformer-CTC>`__ are similar to Conformer config - `QuartzNet <./configs.html#Conformer-CTC>`__.

The encoder section includes the details about the Squeezeformer-CTC encoder architecture. You may find more information in the
config files and also :ref:`nemo.collections.asr.modules.SqueezeformerEncoder <squeezeformer-encoder-api>`.


ContextNet
~~~~~~~~~~

Please refer to the model page of `ContextNet <./models.html#ContextNet>`__ for more information on this model.

Conformer-Transducer
~~~~~~~~~~~~~~~~~~~~

Please refer to the model page of `Conformer-Transducer <./models.html#Conformer-Transducer>`__ for more information on this model.

LSTM-Transducer and LSTM-CTC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The config files for LSTM-Transducer and LSTM-CTC models can be found at ``<NeMo_git_root>/examples/asr/conf/lstm/lstm_transducer_bpe.yaml`` and ``<NeMo_git_root>/examples/asr/conf/lstm/lstm_ctc_bpe.yaml`` respectively.
Most of the of the configs of are similar to other ctc or transducer models. The main difference is the encoder part.
The encoder section includes the details about the RNN-based encoder architecture. You may find more information in the
config files and also :ref:`nemo.collections.asr.modules.RNNEncoder <rnn-encoder-api>`.


InterCTC Config
---------------

All CTC-based models also support `InterCTC loss <https://arxiv.org/abs/2102.03216>`_. To use it, you need to specify
2 parameters as in example below

.. code-block:: yaml

   model:
      # ...
      interctc:
        loss_weights: [0.3]
        apply_at_layers: [8]

which can be used to reproduce the default setup from the paper (assuming the total number of layers is 18).
You can also specify multiple CTC losses from different layers, e.g., to get 2 losses from layers 3 and 8 with
weights 0.1 and 0.3, specify:

.. code-block:: yaml

   model:
      # ...
      interctc:
        loss_weights: [0.1, 0.3]
        apply_at_layers: [3, 8]

Note that the final-layer CTC loss weight is automatically computed to normalize
all weight to 1 (0.6 in the example above).


Stochastic Depth Config
-----------------------

`Stochastic Depth <https://arxiv.org/abs/2102.03216>`_ is a useful technique for regularizing ASR model training.
Currently it's only supported for :ref:`nemo.collections.asr.modules.ConformerEncoder <conformer-encoder-api>`. To
use it, specify the following parameters in the encoder config file to reproduce the default setup from the paper:

.. code-block:: yaml

   model:
      # ...
      encoder:
        # ...
        stochastic_depth_drop_prob: 0.3
        stochastic_depth_mode: linear  # linear or uniform
        stochastic_depth_start_layer: 1

See :ref:`documentation of ConformerEncoder <conformer-encoder-api>` for more details. Note that stochastic depth
is supported for both CTC and Transducer model variations (or any other kind of model/loss that's using
conformer as encoder).


Transducer Configurations
-------------------------

All CTC-based ASR model configs can be modified to support Transducer loss training. Below, we discuss the modifications required in the config to enable Transducer training. All modifications are made to the ``model`` config.

Model Defaults
~~~~~~~~~~~~~~

It is a subsection to the model config representing the default values shared across the entire model represented as ``model.model_defaults``.

There are three values that are primary components of a transducer model. They are :

* ``enc_hidden``: The hidden dimension of the final layer of the Encoder network.
* ``pred_hidden``: The hidden dimension of the final layer of the Prediction network.
* ``joint_hidden``: The hidden dimension of the intermediate layer of the Joint network.

One can access these values inside the config by using OmegaConf interpolation as follows :

.. code-block:: yaml

    model:
      ...
      model_defaults:
        enc_hidden: 256
        pred_hidden: 256
        joint_hidden: 256
      ...
      decoder:
        ...
        prednet:
          pred_hidden: ${model.model_defaults.pred_hidden}

Acoustic Encoder Model
~~~~~~~~~~~~~~~~~~~~~~

The transducer model is comprised of three models combined. One of these models is the Acoustic (encoder) model. We should be able to drop in any CTC Acoustic model config into this section of the transducer config.

The only condition that needs to be met is that **the final layer of the acoustic model must have the hidden dimension defined in ``model_defaults.enc_hidden``**.

Decoder / Prediction Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Prediction model is generally an autoregressive, causal model that consumes text tokens and returns embeddings that will be used by the Joint model. The base config for an LSTM based Prediction network can be found in the the ``decoder`` section of `ContextNet <./models.html#ContextNet>`__ or other Transducer architectures. For further information refer to the ``Intro to Transducers`` tutorial in the ASR tutorial section.

**This config can be copy-pasted into any custom transducer model with no modification.**

Let us discuss some of the important arguments:

* ``blank_as_pad``: In ordinary transducer models, the embedding matrix does not acknowledge the ``Transducer Blank`` token (similar to CTC Blank). However, this causes the autoregressive loop to be more complicated and less efficient. Instead, this flag which is set by default, will add the ``Transducer Blank`` token to the embedding matrix - and use it as a pad value (zeros tensor). This enables more efficient inference without harming training. For further information refer to the ``Intro to Transducers`` tutorial in the ASR tutorial section.

* ``prednet.pred_hidden``: The hidden dimension of the LSTM and the output dimension of the Prediction network.

.. code-block:: yaml

  decoder:
    _target_: nemo.collections.asr.modules.RNNTDecoder
    normalization_mode: null
    random_state_sampling: false
    blank_as_pad: true

    prednet:
      pred_hidden: ${model.model_defaults.pred_hidden}
      pred_rnn_layers: 1
      t_max: null
      dropout: 0.0

Joint Model
~~~~~~~~~~~

The Joint model is a simple feed-forward Multi-Layer Perceptron network. This MLP accepts the output of the Acoustic and Prediction models and computes a joint probability distribution over the entire vocabulary space. The base config for the Joint network can be found in the the ``joint`` section of `ContextNet <./models.html#ContextNet>`__ or other Transducer architectures. For further information refer to the ``Intro to Transducers`` tutorial in the ASR tutorial section.

**This config can be copy-pasted into any custom transducer model with no modification.**

The Joint model config has several essential components which we discuss below :

* ``log_softmax``: Due to the cost of computing softmax on such large tensors, the Numba CUDA implementation of RNNT loss will implicitly compute the log softmax when called (so its inputs should be logits). The CPU version of the loss doesn't face such memory issues so it requires log-probabilities instead. Since the behaviour is different for CPU-GPU, the ``None`` value will automatically switch behaviour dependent on whether the input tensor is on a CPU or GPU device.

* ``preserve_memory``: This flag will call ``torch.cuda.empty_cache()`` at certain critical sections when computing the Joint tensor. While this operation might allow us to preserve some memory, the empty_cache() operation is tremendously slow and will slow down training by an order of magnitude or more. It is available to use but not recommended.

* ``fuse_loss_wer``: This flag performs "batch splitting" and then "fused loss + metric" calculation. It will be discussed in detail in the next tutorial that will train a Transducer model.

* ``fused_batch_size``: When the above flag is set to True, the model will have two distinct "batch sizes". The batch size provided in the three data loader configs (``model.*_ds.batch_size``) will now be the ``Acoustic model`` batch size, whereas the ``fused_batch_size`` will be the batch size of the ``Prediction model``, the ``Joint model``, the ``transducer loss`` module and the ``decoding`` module.

* ``jointnet.joint_hidden``: The hidden intermediate dimension of the joint network.

.. code-block:: yaml

  joint:
    _target_: nemo.collections.asr.modules.RNNTJoint
    log_softmax: null  # sets it according to cpu/gpu device

    # fused mode
    fuse_loss_wer: false
    fused_batch_size: 16

    jointnet:
      joint_hidden: ${model.model_defaults.joint_hidden}
      activation: "relu"
      dropout: 0.0

Sampled Softmax Joint Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are some situations where a large vocabulary with a Transducer model - such as for multilingual models with a large
number of languages. In this setting, we need to consider the cost of memory of training Transducer networks which does
not allow large vocabulary.

For such cases, one can instead utilize the ``SampledRNNTJoint`` module instead of the usual ``RNNTJoint`` module, in order
to compute the loss using a sampled subset of the vocabulary rather than the full vocabulary file.

It adds only one additional parameter :

* ``n_samples``: Specifies the minimum number of tokens to sample from the vocabulary space,
  excluding the RNNT blank token. If a given value is larger than the entire vocabulary size,
  then the full vocabulary will be used.

The only difference in config required is to replace ``nemo.collections.asr.modules.RNNTJoint`` with ``nemo.collections.asr.modules.SampledRNNTJoint``

.. code-block:: yaml

  joint:
    _target_: nemo.collections.asr.modules.SampledRNNTJoint
    n_samples: 500
    ...  # All other arguments from RNNTJoint can be used after this.


Effect of Batch Splitting / Fused Batch step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following information below explain why memory is an issue when training Transducer models and how NeMo tackles the issue with its Fused Batch step. The material can be read for a thorough understanding, otherwise, it can be skipped. You can also follow these steps in the "ASR_with_Transducers" tutorial.

**Diving deeper into the memory costs of Transducer Joint**

One of the significant limitations of Transducers is the exorbitant memory cost of computing the Joint module. The Joint module is comprised of two steps.

1) Projecting the Acoustic and Transcription feature dimensions to some standard hidden dimension (specified by model.model_defaults.joint_hidden)

2) Projecting this intermediate hidden dimension to the final vocabulary space to obtain the transcription.

Take the following example.

BS=32 ; T (after 2x stride) = 800, U (with character encoding) = 400-450 tokens, Vocabulary size V = 28 (26 alphabet chars, space and apostrophe). Let the hidden dimension of the Joint model be 640 (Most Google Transducer papers use hidden dimension of 640).

* :math:`Memory \, (Hidden, \, gb) = 32 \times 800 \times 450 \times 640 \times 4 = 29.49` gigabytes (4 bytes per float).

* :math:`Memory \, (Joint, \, gb) = 32 \times 800 \times 450 \times 28 \times 4 = 1.290` gigabytes (4 bytes per float)

**NOTE**: This is just for the forward pass! We need to double this memory to store gradients! This much memory is also just for the Joint model **alone**. Far more memory is required for the Prediction model as well as the large Acoustic model itself and its gradients!

Even with mixed precision, that's $\sim 30$ GB of GPU RAM for just 1 part of the network + its gradients.

Effect of Fused Batch Step
^^^^^^^^^^^^^^^^^^^^^^^^^^

The fundamental problem is that the joint tensor grows in size when ``[T x U]`` grows in size. This growth in memory cost is due to many reasons - either by model construction (downsampling) or the choice of dataset preprocessing (character tokenization vs. sub-word tokenization).

Another dimension that NeMo can control is **batch**. Due to how we batch our samples, small and large samples all get clumped together into a single batch. So even though the individual samples are not all as long as the maximum length of T and U in that batch, when a batch of such samples is constructed, it will consume a significant amount of memory for the sake of compute efficiency.

So as is always the case - **trade-off compute speed for memory savings**.

The fused operation goes as follows :

1) Forward the entire acoustic model in a single pass. (Use global batch size here for acoustic model - found in ``model.*_ds.batch_size``)

2) Split the Acoustic Model's logits by ``fused_batch_size`` and loop over these sub-batches.

3) Construct a sub-batch of same ``fused_batch_size`` for the Prediction model. Now the target sequence length is :math:`U_{sub-batch} < U`.

4) Feed this :math:`U_{sub-batch}` into the Joint model, along with a sub-batch from the Acoustic model (with :math:`T_{sub-batch} < T)`. Remember, we only have to slice off a part of the acoustic model here since we have the full batch of samples :math:`(B, T, D)` from the acoustic model.

5) Performing steps (3) and (4) yields :math:`T_{sub-batch}` and :math:`U_{sub-batch}`. Perform sub-batch joint step - costing an intermediate :math:`(B, T_{sub-batch}, U_{sub-batch}, V)` in memory.

6) Compute loss on sub-batch and preserve in a list to be later concatenated.

7) Compute sub-batch metrics (such as Character / Word Error Rate) using the above Joint tensor and sub-batch of ground truth labels. Preserve the scores to be averaged across the entire batch later.

8) Delete the sub-batch joint matrix  :math:`(B, T_{sub-batch}, U_{sub-batch}, V)`. Only gradients from .backward() are preserved now in the computation graph.

9) Repeat steps (3) - (8) until all sub-batches are consumed.

10) Cleanup step. Compute full batch WER and log. Concatenate loss list and pass to PTL to compute the equivalent of the original (full batch) Joint step. Delete ancillary objects necessary for sub-batching.

Transducer Decoding
~~~~~~~~~~~~~~~~~~~

Models which have been trained with CTC can transcribe text simply by performing a regular argmax over the output of their decoder. For transducer-based models, the three networks must operate in a synchronized manner in order to transcribe the acoustic features. The base config for the Transducer decoding step can be found in the the ``decoding`` section of `ContextNet <./models.html#ContextNet>`__ or other Transducer architectures. For further information refer to the ``Intro to Transducers`` tutorial in the ASR tutorial section.

**This config can be copy-pasted into any custom transducer model with no modification.**

The most important component at the top level is the ``strategy``. It can take one of many values:

* ``greedy``: This is sample-level greedy decoding. It is generally exceptionally slow as each sample in the batch will be decoded independently. For publications, this should be used alongside batch size of 1 for exact results.

* ``greedy_batch``: This is the general default and should nearly match the ``greedy`` decoding scores (if the acoustic features are not affected by feature mixing in batch mode). Even for small batch sizes, this strategy is significantly faster than ``greedy``.

* ``beam``: Runs beam search with the implicit language model of the Prediction model. It will generally be quite slow, and might need some tuning of the beam size to get better transcriptions.

* ``tsd``: Time synchronous decoding. Please refer to the paper: `Alignment-Length Synchronous Decoding for RNN Transducer <https://ieeexplore.ieee.org/document/9053040>`_ for details on the algorithm implemented. Time synchronous decoding (TSD) execution time grows by the factor T * max_symmetric_expansions. For longer sequences, T is greater and can therefore take a long time for beams to obtain good results. TSD also requires more memory to execute.

* ``alsd``: Alignment-length synchronous decoding. Please refer to the paper: `Alignment-Length Synchronous Decoding for RNN Transducer <https://ieeexplore.ieee.org/document/9053040>`_ for details on the algorithm implemented. Alignment-length synchronous decoding (ALSD) execution time is faster than TSD, with a growth factor of T + U_max, where U_max is the maximum target length expected during execution. Generally, T + U_max < T * max_symmetric_expansions. However, ALSD beams are non-unique. Therefore it is required to use larger beam sizes to achieve the same (or close to the same) decoding accuracy as TSD. For a given decoding accuracy, it is possible to attain faster decoding via ALSD than TSD.

* ``maes``: Modified Adaptive Expansion Search Decoding. Please refer to the paper `Accelerating RNN Transducer Inference via Adaptive Expansion Search <https://ieeexplore.ieee.org/document/9250505>`_. Modified Adaptive Synchronous Decoding (mAES) execution time is adaptive w.r.t the number of expansions (for tokens) required per timestep. The number of expansions can usually be constrained to 1 or 2, and in most cases 2 is sufficient. This beam search technique can possibly obtain superior WER while sacrificing some evaluation time.

.. code-block:: yaml

  decoding:
    strategy: "greedy_batch"

    # preserve decoding alignments
    preserve_alignments: false

    # Overrides the fused batch size after training.
    # Setting it to -1 will process whole batch at once when combined with `greedy_batch` decoding strategy
    fused_batch_size: Optional[int] = -1

    # greedy strategy config
    greedy:
      max_symbols: 10

    # beam strategy config
    beam:
      beam_size: 2
      score_norm: true
      softmax_temperature: 1.0  # scale the logits by some temperature prior to softmax
      tsd_max_sym_exp: 10  # for Time Synchronous Decoding, int > 0
      alsd_max_target_len: 5.0  # for Alignment-Length Synchronous Decoding, float > 1.0
      maes_num_steps: 2  # for modified Adaptive Expansion Search, int > 0
      maes_prefix_alpha: 1  # for modified Adaptive Expansion Search, int > 0
      maes_expansion_beta: 2  # for modified Adaptive Expansion Search, int >= 0
      maes_expansion_gamma: 2.3  # for modified Adaptive Expansion Search, float >= 0

Transducer Loss
~~~~~~~~~~~~~~~

This section configures the type of Transducer loss itself, along with possible sub-sections. By default, an optimized implementation of Transducer loss will be used which depends on Numba for CUDA acceleration. The base config for the Transducer loss section can be found in the the ``loss`` section of `ContextNet <./models.html#ContextNet>`__ or other Transducer architectures. For further information refer to the ``Intro to Transducers`` tutorial in the ASR tutorial section.

**This config can be copy-pasted into any custom transducer model with no modification.**

The loss config is based on a resolver pattern and can be used as follows:

1) ``loss_name``: ``default`` is generally a good option. Will select one of the available resolved losses and match the kwargs from a sub-configs passed via explicit ``{loss_name}_kwargs`` sub-config.

2) ``{loss_name}_kwargs``: This sub-config is passed to the resolved loss above and can be used to configure the resolved loss.


.. code-block:: yaml

  loss:
    loss_name: "default"
    warprnnt_numba_kwargs:
      fastemit_lambda: 0.0

FastEmit Regularization
^^^^^^^^^^^^^^^^^^^^^^^

FastEmit Regularization is supported for the default Numba based WarpRNNT loss. Recently proposed regularization approach - `FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization <https://arxiv.org/abs/2010.11148>`_ allows us near-direct control over the latency of transducer models.

Refer to the above paper for results and recommendations of ``fastemit_lambda``.


.. _Hybrid-ASR-TTS_model__Config:

Hybrid ASR-TTS Model Configuration
----------------------------------

:ref:`Hybrid ASR-TTS model <Hybrid-ASR-TTS_model>` consists of three parts:

* ASR model (``EncDecCTCModelBPE``, ``EncDecRNNTBPEModel`` or ``EncDecHybridRNNTCTCBPEModel``)
* TTS Mel Spectrogram Generator (currently, only :ref:`FastPitch <FastPitch_model>` model is supported)
* :ref:`Enhancer model <SpectrogramEnhancer_model>` (optional)

Also, the config allows to specify :ref:`text-only dataset <Hybrid-ASR-TTS_model__Text-Only-Data>`.

Main parts of the config:

* ASR model
    * ``asr_model_path``: path to the ASR model checkpoint (`.nemo`) file, loaded only once, then the config of the ASR model is stored in the ``asr_model`` field
    * ``asr_model_type``: needed only when training from scratch. ``rnnt_bpe`` corresponds to ``EncDecRNNTBPEModel``, ``ctc_bpe`` to ``EncDecCTCModelBPE``, ``hybrid_rnnt_ctc_bpe`` to ``EncDecHybridRNNTCTCBPEModel``
    * ``asr_model_fuse_bn``: fusing BatchNorm in the pretrained ASR model, can improve quality in finetuning scenario
* TTS model
    * ``tts_model_path``: path to the pretrained TTS model checkpoint (`.nemo`) file, loaded only once, then the config of the model is stored in the ``tts_model`` field
* Enhancer model
    * ``enhancer_model_path``: optional path to the enhancer model. Loaded only once, the config is stored in the ``enhancer_model`` field
* ``train_ds``
    * ``text_data``: properties related to text-only data
        * ``manifest_filepath``: path (or paths) to :ref:`text-only dataset <Hybrid-ASR-TTS_model__Text-Only-Data>` manifests
        * ``speakers_filepath``: path (or paths) to the text file containing speaker ids for the multi-speaker TTS model (speakers are sampled randomly during training)
        * ``min_words`` and ``max_words``: parameters to filter text-only manifests by the number of words
        * ``tokenizer_workers``: number of workers for initial tokenization (when loading the data). ``num_CPUs / num_GPUs`` is a recommended value.
    * ``asr_tts_sampling_technique``, ``asr_tts_sampling_temperature``, ``asr_tts_sampling_probabilities``: sampling parameters for text-only and audio-text data (if both specified). Correspond to ``sampling_technique``, ``sampling_temperature``, and ``sampling_probabilities`` parameters of the :mod:`ConcatDataset <nemo.collections.common.data.dataset.ConcatDataset>`.
    * all other components are similar to conventional ASR models
* ``validation_ds`` and ``test_ds`` correspond to the underlying ASR model


.. code-block:: yaml

  model:
    sample_rate: 16000

    # asr model
    asr_model_path: ???
    asr_model: null
    asr_model_type: null  # rnnt_bpe, ctc_bpe or hybrid_rnnt_ctc_bpe; needed only if instantiating from config, otherwise type is auto inferred
    asr_model_fuse_bn: false  # only ConformerEncoder supported now, use false for other models

    # tts model
    tts_model_path: ???
    tts_model: null

    # enhancer model
    enhancer_model_path: null
    enhancer_model: null

    train_ds:
      text_data:
        manifest_filepath: ???
        speakers_filepath: ???
        min_words: 1
        max_words: 45  # 45 - recommended value, ~16.7 sec for LibriSpeech
        tokenizer_workers: 1
      asr_tts_sampling_technique: round-robin  # random, round-robin, temperature
      asr_tts_sampling_temperature: null
      asr_tts_sampling_probabilities: null  # [0.5,0.5] – ASR,TTS
      manifest_filepath: ???
      batch_size: 16 # you may increase batch_size if your memory allows
      # other params

Finetuning with Text-Only Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To finetune existing ASR model using text-only data use ``<NeMo_git_root>/examples/asr/asr_with_tts/speech_to_text_bpe_with_text_finetune.py`` script with the corresponding config ``<NeMo_git_root>/examples/asr/conf/asr_tts/hybrid_asr_tts.yaml``.

Please specify paths to all the required models (ASR, TTS, and Enhancer checkpoints), along with ``train_ds.text_data.manifest_filepath`` and ``train_ds.text_data.speakers_filepath``.

.. code-block:: shell

    python speech_to_text_bpe_with_text_finetune.py \
        model.asr_model_path=<path to ASR model> \
        model.tts_model_path=<path to compatible TTS model> \
        model.enhancer_model_path=<optional path to enhancer model> \
        model.asr_model_fuse_bn=<true recommended if ConformerEncoder with BatchNorm, false otherwise> \
        model.train_ds.manifest_filepath=<path to manifest with audio-text pairs or null> \
        model.train_ds.text_data.manifest_filepath=<path(s) to manifest with train text> \
        model.train_ds.text_data.speakers_filepath=<path(s) to speakers list> \
        model.train_ds.text_data.tokenizer_workers=4 \
        model.validation_ds.manifest_filepath=<path to validation manifest> \
        model.train_ds.batch_size=<batch_size>

Training from Scratch
~~~~~~~~~~~~~~~~~~~~~

To train ASR model from scratch using text-only data use ``<NeMo_git_root>/examples/asr/asr_with_tts/speech_to_text_bpe_with_text.py`` script with conventional ASR model config, e.g. ``<NeMo_git_root>/examples/asr/conf/conformer/conformer_ctc_bpe.yaml`` or  ``<NeMo_git_root>/examples/asr/conf/conformer/conformer_transducer_bpe.yaml``

Please specify the ASR model type, paths to the TTS model, and (optional) enhancer, along with text-only data-related fields.
Use ``++`` or ``+`` markers for these options, since the options are not present in the original ASR model config.

.. code-block:: shell

    python speech_to_text_bpe_with_text.py \
        ++asr_model_type=<rnnt_bpe or ctc_bpe> \
        ++tts_model_path=<path to compatible tts model> \
        ++enhancer_model_path=<optional path to enhancer model> \
        ++model.train_ds.text_data.manifest_filepath=<path(s) to manifests with train text> \
        ++model.train_ds.text_data.speakers_filepath=<path(s) to speakers list> \
        ++model.train_ds.text_data.min_words=1 \
        ++model.train_ds.text_data.max_words=45 \
        ++model.train_ds.text_data.tokenizer_workers=4

Fine-tuning Configurations
--------------------------

All ASR scripts support easy fine-tuning by partially/fully loading the pretrained weights from a checkpoint into the **currently instantiated model**. Note that the currently instantiated model should have parameters that match the pre-trained checkpoint (such that weights may load properly). In order to directly fine-tune a pre-existing checkpoint, please follow the tutorial  `ASR Language Fine-tuning. <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb>`_

Models can be fine-tuned in two ways: 
* By updating or retaining current tokenizer alone
* By updating model architecture and tokenizer

Fine-tuning by updating or retaining current tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, the model architecture is not updated. The model is initialized with the pre-trained weights by 
two ways:

1) Providing a path to a NeMo model (via ``init_from_nemo_model``)
2) Providing a name of a pretrained NeMo model (which will be downloaded via the cloud) (via ``init_from_pretrained_model``)

Then users can use existing tokenizer or update the tokenizer with new vocabulary. This is useful when users don't want to update the model architecture 
but want to update the tokenizer with new vocabulary.

The same script can be used to finetune CTC, RNNT or Hybrid models as well.

<NeMo_repo>/examples/asr/speech_to_text_finetune.py script supports this type of fine-tuning with the following arguments:

.. code-block:: sh

    python examples/asr/speech_to_text_finetune.py \
        --config-path=<path to dir of configs> \
        --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        model.tokenizer.update_tokenizer=<True/False> \ # True to update tokenizer, False to retain existing tokenizer
        model.tokenizer.dir=<path to tokenizer dir> \ # Path to tokenizer dir when update_tokenizer=True
        model.tokenizer.type=<tokenizer type> \ # tokenizer type when update_tokenizer=True
        trainer.devices=-1 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        +init_from_nemo_model="<path to .nemo model file>" (or +init_from_pretrained_model="<name of pretrained checkpoint>")


Refer to <NeMo_repo>/examples/asr/conf/asr_finetune/speech_to_text_finetune.yaml for more details.

Finetune ASR Models using HuggingFace Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Users can utilize HuggingFace Datasets for finetuning NeMo ASR models. The following config file can be used for this purpose:
`<NeMo_repo>/examples/asr/conf/asr_finetune/speech_to_text_hf_finetune.yaml`

As mentioned earlier, users can update the tokenizer or use an existing one based on their requirements. If users want to create a new tokenizer 
from HuggingFace Datasets, they can use the following script:
`<NeMo_repo>/scripts/tokenizers/get_hf_text_data.py`

Fine-tuning by changing model architecture and tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If users want to update the model architecture as well they can use the following script:

For providing pretrained model, users can provide Pre-trained weights in multiple ways -

1) Providing a path to a NeMo model (via ``init_from_nemo_model``)
2) Providing a name of a pretrained NeMo model (which will be downloaded via the cloud) (via ``init_from_pretrained_model``)
3) Providing a path to a Pytorch Lightning checkpoint file (via ``init_from_ptl_ckpt``)

There are multiple ASR subtasks inside the ``examples/asr/`` directory, you can substitute the ``<subtask>`` tag below.

.. code-block:: sh

    python examples/asr/<subtask>/script_to_<script_name>.py \
        --config-path=<path to dir of configs> \
        --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.devices=-1 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        +init_from_nemo_model="<path to .nemo model file>" # (or +init_from_pretrained_model, +init_from_ptl_ckpt )

To reinitialize part of the model, to make it different from the pretrained model, users can mention them through config:

.. code-block:: yaml
    
    init_from_nemo_model: "<path to .nemo model file>"
        asr_model:
            include: ["preprocessor","encoder"]
            exclude: ["decoder"]

Fine-tuning Execution Flow Diagram
----------------------------------

When preparing your own training or fine-tuning scripts, please follow the execution flow diagram order for correct inference.

Depending on the type of model, there may be extra steps that must be performed -

* CTC Models - `Examples directory for CTC Models <https://github.com/NVIDIA/NeMo/blob/stable/examples/asr/asr_ctc/README.md>`_
* RNN Transducer Models - `Examples directory for Transducer Models <https://github.com/NVIDIA/NeMo/blob/stable/examples/asr/asr_transducer/README.md>`_
