NeMo Speech Intent Classification and Slot Filling Configuration Files
=======================================================================

This page covers NeMo configuration file setup that is specific to models in the Speech Intent Classification and Slot Filling collection.
For general information about how to set up and run experiments that is common to all NeMo models (e.g.
experiment manager and PyTorch Lightning trainer parameters), see the :doc:`../../core/core`  page.

Dataset Configuration
---------------------

Dataset configuration for Speech Intent Classification and Slot Filling model is mostly the same as for standard ASR training,
covered `here <../configs.html#Dataset Configuration>`__. One exception is that ``use_start_end_token`` must be set to ``True``.

An example of train and validation configuration should look similar to the following:

.. code-block:: yaml

  model:
    train_ds:
      manifest_filepath: ???
      sample_rate: ${model.sample_rate}
      batch_size: 16 # you may increase batch_size if your memory allows
      shuffle: true
      num_workers: 8
      pin_memory: false
      use_start_end_token: true
      trim_silence: false
      max_duration: 11.0
      min_duration: 0.0
      # tarred datasets
      is_tarred: false
      tarred_audio_filepaths: null
      shuffle_n: 2048
      # bucketing params
      bucketing_strategy: "synced_randomized"
      bucketing_batch_size: null

    validation_ds:
      manifest_filepath: ???
      sample_rate: ${model.sample_rate}
      batch_size: 16 # you may increase batch_size if your memory allows
      shuffle: false
      num_workers: 8
      pin_memory: true
      use_start_end_token: true
      min_duration: 8.0


Preprocessor Configuration
--------------------------

Preprocessor helps to compute MFCC or mel spectrogram features that are given as inputs to model.
For details on how to write this section, refer to `Preprocessor Configuration <../configs.html#preprocessor-configuration>`__

Augmentation Configurations
---------------------------


There are a few on-the-fly spectrogram augmentation options for NeMo ASR, which can be specified by the
configuration file using the ``augmentor`` and ``spec_augment`` section.
For details on how to write this section, refer to `Augmentation Configuration <../configs.html#augmentation-configurations>`__


Model Architecture Configurations
---------------------------------

The ``encoder`` of the model is a `Conformer-large <./models.html#Conformer-CTC>`__ model without the text decoder, and can be initialized with pretrained checkpoints. The ``decoder`` is a Transforemr model, with additional ``embedding`` and ``classifier`` modules.

An example config for the model can be:

.. code-block:: yaml

  pretrained_encoder:
    name: stt_en_conformer_ctc_large  # which model use to initialize the encoder, set to null if not using any. Only used to initialize training, not used in resuming from checkpoint.
    freeze: false  # whether to freeze the encoder during training.

  model:
    sample_rate: 16000
    encoder:
      _target_: nemo.collections.asr.modules.ConformerEncoder
      feat_in: ${model.preprocessor.features}
      feat_out: -1 # you may set it if you need different output size other than the default d_model
      n_layers: 17  # SSL conformer-large have only 17 layers
      d_model: 512

      # Sub-sampling params
      subsampling: striding # vggnet or striding, vggnet may give better results but needs more memory
      subsampling_factor: 4 # must be power of 2
      subsampling_conv_channels: -1 # -1 sets it to d_model

      # Reduction parameters: Can be used to add another subsampling layer at a given position.
      # Having a 2x reduction will speedup the training and inference speech while keeping similar WER.
      # Adding it at the end will give the best WER while adding it at the beginning will give the best speedup.
      reduction: null # pooling, striding, or null
      reduction_position: null # Encoder block index or -1 for subsampling at the end of encoder
      reduction_factor: 1

      # Feed forward module's params
      ff_expansion_factor: 4

      # Multi-headed Attention Module's params
      self_attention_model: rel_pos # rel_pos or abs_pos
      n_heads: 8 # may need to be lower for smaller d_models
      # [left, right] specifies the number of steps to be seen from left and right of each step in self-attention
      att_context_size: [-1, -1] # -1 means unlimited context
      xscaling: true # scales up the input embeddings by sqrt(d_model)
      untie_biases: true # unties the biases of the TransformerXL layers
      pos_emb_max_len: 5000

      # Convolution module's params
      conv_kernel_size: 31
      conv_norm_type: 'batch_norm' # batch_norm or layer_norm

      ### regularization
      dropout: 0.1 # The dropout used in most of the Conformer Modules
      dropout_pre_encoder: 0.1 # The dropout used before the encoder
      dropout_emb: 0.0 # The dropout used for embeddings
      dropout_att: 0.1 # The dropout for multi-headed attention modules

    embedding:
      _target_: nemo.collections.asr.modules.transformer.TransformerEmbedding
      vocab_size: -1
      hidden_size: ${model.encoder.d_model}
      max_sequence_length: 512
      num_token_types: 1
      embedding_dropout: 0.0
      learn_positional_encodings: false

    decoder:
      _target_: nemo.collections.asr.modules.transformer.TransformerDecoder
      num_layers: 3
      hidden_size: ${model.encoder.d_model}
      inner_size: 2048
      num_attention_heads: 8
      attn_score_dropout: 0.0
      attn_layer_dropout: 0.0
      ffn_dropout: 0.0

    classifier:
      _target_: nemo.collections.common.parts.MultiLayerPerceptron
      hidden_size: ${model.encoder.d_model}
      num_classes: -1
      num_layers: 1
      activation: 'relu'
      log_softmax: true


Loss Configurations
---------------------------------

The loss function by default is the negative log-likelihood loss, where optional label-smoothing can be applied by using the following config (default is 0.0):

.. code-block:: yaml

  loss:
    label_smoothing: 0.0


Inference Configurations
---------------------------------
During inference, three types of sequence generation strategies can be applied: ``greedy search``, ``beam search`` and ``top-k search``.

.. code-block:: yaml

  sequence_generator:
    type: greedy  # choices=[greedy, topk, beam]
    max_sequence_length: ${model.embedding.max_sequence_length}
    temperature: 1.0  # for top-k sampling
    beam_size: 1  # K for top-k sampling, N for beam search
    len_pen: 0  # for beam-search
