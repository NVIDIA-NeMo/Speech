NeMo TTS Configuration Files
============================
This section describes the NeMo configuration file setup that is specific to models in the TTS collection. For general information
about how to set up and run experiments that is common to all NeMo models (e.g. Experiment Manager and PyTorch Lightning trainer
parameters), see the :doc:`../core/core` section.

The model section of the NeMo TTS configuration files generally requires information about the dataset(s) being used, the preprocessor
for audio files, parameters for any augmentation being performed, as well as the model architecture specification. The sections on
this page cover each of these in more detail.

Example configuration files for all of the NeMo TTS scripts can be found in the
`config directory of the examples <https://github.com/NVIDIA/NeMo/tree/stable/examples/tts/conf>`_.

Dataset Configuration
---------------------

Training, validation, and test parameters are specified using the ``model.train_ds``, ``model.validation_ds``, and ``model.test_ds`` sections in the configuration file, respectively. Depending on the task, there may be arguments specifying the sample rate of the audio files, supplementary data such as speech/text alignment priors and speaker IDs, etc., the threshold to trim leading and trailing silence from an audio signal, pitch normalization parameters, and so on. You may also decide to leave fields such as the ``manifest_filepath`` blank, to be specified via the command-line at runtime.

Any initialization parameter that is accepted for the class `nemo.collections.tts.data.dataset.TTSDataset
<https://github.com/NVIDIA/NeMo/tree/stable/nemo/collections/tts/data/dataset.py#L80>`_  can be set in the config
file. Refer to the `Dataset Processing Classes <./api.html#Datasets>`__ section of the API for a list of datasets classes and their respective parameters. An example TTS train and validation configuration should look similar to the following:

.. code-block:: yaml

  model:
    train_ds:
      dataset:
        _target_: nemo.collections.tts.data.dataset.TTSDataset
        manifest_filepath: ???
        sample_rate: 44100
        sup_data_path: ???
        sup_data_types: ["align_prior_matrix", "pitch"]
        n_fft: 2048
        win_length: 2048
        hop_length: 512
        window: hann
        n_mels: 80
        lowfreq: 0
        highfreq: null
        max_duration: null
        min_duration: 0.1
        ignore_file: null
        trim: false
        pitch_fmin: 65.40639132514966
        pitch_fmax: 2093.004522404789
        pitch_norm: true
        pitch_mean: 212.35873413085938
        pitch_std: 68.52806091308594
        use_beta_binomial_interpolator: true

      dataloader_params:
        drop_last: false
        shuffle: true
        batch_size: 32
        num_workers: 12
        pin_memory: true


Audio Preprocessor Configuration
--------------------------------

If you are loading audio files for your experiment, you will likely want to use a preprocessor to convert from the raw audio signal to features (e.g. mel-spectrogram or MFCC). The ``preprocessor`` section of the config specifies the audio preprocessor to be used via the ``_target_`` field, as well as any initialization parameters for that preprocessor. An example of specifying a preprocessor is as follows. Refer to the `Audio Preprocessors <../asr/api.html#Audio Preprocessors>`__ API section for the preprocessor options, expected arguments, and defaults.

.. code-block:: yaml

  model:
    preprocessor:
      _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
      features: 80
      lowfreq: 0
      highfreq: null
      n_fft: 2048
      n_window_size: 2048
      window_size: false
      n_window_stride: 512
      window_stride: false
      pad_to: 1
      pad_value: 0
      sample_rate: 44100
      window: hann
      normalize: null
      preemph: null
      dither: 0.0
      frame_splicing: 1
      log: true
      log_zero_guard_type: add
      log_zero_guard_value: 1e-05
      mag_power: 1.0

Text Normalizer Configuration
------------------------------
Text normalization (TN) converts text from written form into its verbalized form, and it is an essential preprocessing step before text-to-speech Synthesis. TN ensures that TTS can handle all input texts without skipping unknown symbols. For example, "$123" is converted to "one hundred and twenty three dollars". Currently, NeMo supports text normalizers for English, German, Spanish, and Chinese. Refer to the previous Section :doc:`../nlp/text_normalization/intro` for more details. Below shows an example of specifying text normalizer for English.

.. code-block:: yaml

  model:
    text_normalizer:
      _target_: nemo_text_processing.text_normalization.normalize.Normalizer
      lang: en
      input_case: cased

    text_normalizer_call_kwargs:
      verbose: false
      punct_pre_process: true
      punct_post_process: true

Tokenizer Configuration
------------------------
Tokenization converts input text string to a list of integer tokens. It may pad leading and/or trailing whitespaces to a string. NeMo tokenizer supports grapheme-only inputs, phoneme-only inputs, or a mixer of grapheme and phoneme inputs to disambiguate pronunciations of heteronyms for English, German, and Spanish. It also utilizes a grapheme-to-phoneme (G2P) tool to transliterate out-of-vocabulary (OOV) words. Please refer to the Section :doc:`../text_processing/g2p/g2p` and `TTS tokenizer collection <https://github.com/NVIDIA/NeMo/tree/stable/nemo/collections/common/tokenizers/text_to_speech/tts_tokenizers.py>`_ for more details. Note that G2P integration to NeMo TTS tokenizers pipeline is upcoming soon. The following example sets up a ``EnglishPhonemesTokenizer`` with a mixer of grapheme and phoneme inputs where each word shown in the heteronym list is transliterated into graphemes or phonemes by a 50% chance.

.. code-block:: yaml

  model:
    text_tokenizer:
      _target_: nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.EnglishPhonemesTokenizer
      punct: true
      stresses: true
      chars: true
      apostrophe: true
      pad_with_space: true
      g2p:
        _target_: nemo.collections.tts.g2p.models.en_us_arpabet.EnglishG2p
        phoneme_dict: ${phoneme_dict_path}
        heteronyms: ${heteronyms_path}
      phoneme_probability: 0.5


Model Architecture Configuration
--------------------------------
Each configuration file should describe the model architecture being used for the experiment. Models in the NeMo TTS collection need several module sections with the ``_target_`` field specifying which model architecture or component is used. Please refer to `TTS module collection <https://github.com/NVIDIA/NeMo/tree/stable/nemo/collections/tts/modules>`_ for details. Below shows an example of FastPitch model architecture,

.. code-block:: yaml

  model:
    input_fft: #n_embed and padding_idx are added by the model
      _target_: nemo.collections.tts.modules.transformer.FFTransformerEncoder
      n_layer: 6
      n_head: 1
      d_model: 384
      d_head: 64
      d_inner: 1536
      kernel_size: 3
      dropout: 0.1
      dropatt: 0.1
      dropemb: 0.0
      d_embed: 384

    output_fft:
      _target_: nemo.collections.tts.modules.transformer.FFTransformerDecoder
      n_layer: 6
      n_head: 1
      d_model: 384
      d_head: 64
      d_inner: 1536
      kernel_size: 3
      dropout: 0.1
      dropatt: 0.1
      dropemb: 0.0

    alignment_module:
      _target_: nemo.collections.tts.modules.aligner.AlignmentEncoder
      n_text_channels: 384

    duration_predictor:
      _target_: nemo.collections.tts.modules.fastpitch.TemporalPredictor
      input_size: 384
      kernel_size: 3
      filter_size: 256
      dropout: 0.1
      n_layers: 2

    pitch_predictor:
      _target_: nemo.collections.tts.modules.fastpitch.TemporalPredictor
      input_size: 384
      kernel_size: 3
      filter_size: 256
      dropout: 0.1
      n_layers: 2

    optim:
      name: adamw
      lr: 1e-3
      betas: [0.9, 0.999]
      weight_decay: 1e-6

      sched:
        name: NoamAnnealing
        warmup_steps: 1000
        last_epoch: -1
        d_model: 1  # Disable scaling based on model dim

Finetuning Configuration
--------------------------

All TTS scripts support easy finetuning by partially/fully loading the pretrained weights from a checkpoint into the **currently instantiated model**. Note that the currently instantiated model should have parameters that match the pre-trained checkpoint (such that weights may load properly). In order to directly finetune a pre-existing checkpoint, please follow the tutorial of `Finetuning FastPitch for a new speaker. <https://github.com/NVIDIA/NeMo/tree/stable/tutorials/tts/FastPitch_Finetuning.ipynb>`_

Pre-trained weights can be provided in multiple ways:

1) Providing a path to a NeMo model (via ``init_from_nemo_model``)
2) Providing a name of a pretrained NeMo model (which will be downloaded via the cloud) (via ``init_from_pretrained_model``)
3) Providing a path to a Pytorch Lightning checkpoint file (via ``init_from_ptl_ckpt``)

There are multiple TTS model finetuning scripts in `examples/tts/<model>_finetune.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/tts/>`_. You can finetune any model by substituting the ``<model>`` tag. An example of finetuning a HiFiGAN model is shown below.

Fine-tuning via a NeMo model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh
    :emphasize-lines: 13

    python examples/tts/hifigan_finetune.py \
        --config-path=<path to dir of configs> \
        --config-name=<name of config without .yaml>) \
        model/train_ds=train_ds_finetune \
        model/validation_ds=val_ds_finetune \
        train_dataset="<path to manifest file>" \
        validation_dataset="<path to manifest file>" \
        model.optim.lr=0.00001 \
        ~model.optim.sched \
        trainer.devices=-1 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        +init_from_nemo_model="<path to .nemo model file>"


Fine-tuning via a NeMo pretrained model name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh
    :emphasize-lines: 13

    python examples/tts/hifigan_finetune.py \
        --config-path=<path to dir of configs> \
        --config-name=<name of config without .yaml>) \
        model/train_ds=train_ds_finetune \
        model/validation_ds=val_ds_finetune \
        train_dataset="<path to manifest file>" \
        validation_dataset="<path to manifest file>" \
        model.optim.lr=0.00001 \
        ~model.optim.sched \
        trainer.devices=-1 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        +init_from_pretrained_model="<name of pretrained checkpoint>"

Fine-tuning via a Pytorch Lightning checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh
    :emphasize-lines: 13

    python examples/tts/hifigan_finetune.py \
        --config-path=<path to dir of configs> \
        --config-name=<name of config without .yaml>) \
        model/train_ds=train_ds_finetune \
        model/validation_ds=val_ds_finetune \
        train_dataset="<path to manifest file>" \
        validation_dataset="<path to manifest file>" \
        model.optim.lr=0.00001 \
        ~model.optim.sched \
        trainer.devices=-1 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        +init_from_ptl_ckpt="<name of pytorch lightning checkpoint>"
