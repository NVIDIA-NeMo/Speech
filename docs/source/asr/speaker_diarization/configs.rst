End-to-End Speaker Diarization Configuration Files
==================================================

Hydra Configurations for Sortformer Diarizer Training 
-----------------------------------------------------

Sortformer Diarizer is an end-to-end speaker diarization model that is solely based on Transformer-encoder type of architecture.
Model name convention for Sortformer Diarizer: sortformer_diarizer_<loss_type>_<speaker count limit>-<version>.yaml


* Example `<NeMo_root>/examples/speaker_tasks/diarization/neural_diarizer/conf/sortformer_diarizer_hybrid_loss_4spk-v1.yaml`.

.. code-block:: yaml

  name: "SortformerDiarizer"
  num_workers: 18
  batch_size: 8

  model: 
    sample_rate: 16000
    pil_weight: 0.5 # Weight for Permutation Invariant Loss (PIL) used in training the Sortformer diarizer model
    ats_weight: 0.5 # Weight for Arrival Time Sort (ATS) loss in training the Sortformer diarizer model
    max_num_of_spks: 4 # Maximum number of speakers per model; currently set to 4

    model_defaults:
      fc_d_model: 512 # Hidden dimension size of the Fast-conformer Encoder
      tf_d_model: 192 # Hidden dimension size of the Transformer Encoder

    train_ds:
      manifest_filepath: ???
      sample_rate: ${model.sample_rate}
      num_spks: ${model.max_num_of_spks}
      session_len_sec: 90 # Maximum session length in seconds
      soft_label_thres: 0.5 # Threshold for binarizing target values; higher values make the model more conservative in predicting speaker activity.
      soft_targets: False # If True, use continuous values as target values when calculating cross-entropy loss
      labels: null
      batch_size: ${batch_size}
      shuffle: True
      num_workers: ${num_workers}
      validation_mode: False
      # lhotse config
      use_lhotse: False
      use_bucketing: True
      num_buckets: 10
      bucket_duration_bins: [10, 20, 30, 40, 50, 60, 70, 80, 90]
      pin_memory: True
      min_duration: 10
      max_duration: 90
      batch_duration: 400
      quadratic_duration: 1200
      bucket_buffer_size: 20000
      shuffle_buffer_size: 10000
      window_stride: ${model.preprocessor.window_stride}
      subsampling_factor: ${model.encoder.subsampling_factor}

    validation_ds:
      manifest_filepath: ???
      is_tarred: False
      tarred_audio_filepaths: null
      sample_rate: ${model.sample_rate}
      num_spks: ${model.max_num_of_spks}
      session_len_sec: 90 # Maximum session length in seconds
      soft_label_thres: 0.5 # A threshold value for setting up the binarized labels. The higher the more conservative the model becomes. 
      soft_targets: False
      labels: null
      batch_size: ${batch_size}
      shuffle: False
      num_workers: ${num_workers}
      validation_mode: True
      # lhotse config
      use_lhotse: False
      use_bucketing: False
      drop_last: False
      pin_memory: True
      window_stride: ${model.preprocessor.window_stride}
      subsampling_factor: ${model.encoder.subsampling_factor}
    
    test_ds:
      manifest_filepath: null
      is_tarred: False
      tarred_audio_filepaths: null
      sample_rate: 16000
      num_spks: ${model.max_num_of_spks}
      session_len_sec: 90 # Maximum session length in seconds
      soft_label_thres: 0.5
      soft_targets: False
      labels: null
      batch_size: ${batch_size}
      shuffle: False
      seq_eval_mode: True
      num_workers: ${num_workers}
      validation_mode: True
      # lhotse config
      use_lhotse: False
      use_bucketing: False
      drop_last: False
      pin_memory: True
      window_stride: ${model.preprocessor.window_stride}
      subsampling_factor: ${model.encoder.subsampling_factor}

    preprocessor:
      _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
      normalize: "per_feature"
      window_size: 0.025
      sample_rate: ${model.sample_rate}
      window_stride: 0.01
      window: "hann"
      features: 80
      n_fft: 512
      frame_splicing: 1
      dither: 0.00001

    sortformer_modules:
      _target_: nemo.collections.asr.modules.sortformer_modules.SortformerModules
      num_spks: ${model.max_num_of_spks} # Number of speakers per model. This is currently fixed at 4.
      dropout_rate: 0.5 # Dropout rate
      fc_d_model: ${model.model_defaults.fc_d_model}
      tf_d_model: ${model.model_defaults.tf_d_model} # Hidden layer size for linear layers in Sortformer Diarizer module

    encoder:
      _target_: nemo.collections.asr.modules.ConformerEncoder
      feat_in: ${model.preprocessor.features}
      feat_out: -1
      n_layers: 18
      d_model: ${model.model_defaults.fc_d_model}

      # Sub-sampling parameters
      subsampling: dw_striding # vggnet, striding, stacking or stacking_norm, dw_striding
      subsampling_factor: 8 # must be power of 2 for striding and vggnet
      subsampling_conv_channels: 256 # set to -1 to make it equal to the d_model
      causal_downsampling: false
      # Feed forward module's params
      ff_expansion_factor: 4
      # Multi-headed Attention Module's params
      self_attention_model: rel_pos # rel_pos or abs_pos
      n_heads: 8 # may need to be lower for smaller d_models
      # [left, right] specifies the number of steps to be seen from left and right of each step in self-attention
      att_context_size: [-1, -1] # -1 means unlimited context
      att_context_style: regular # regular or chunked_limited
      xscaling: true # scales up the input embeddings by sqrt(d_model)
      untie_biases: true # unties the biases of the TransformerXL layers
      pos_emb_max_len: 5000
      # Convolution module's params
      conv_kernel_size: 9
      conv_norm_type: 'batch_norm' # batch_norm or layer_norm or groupnormN (N specifies the number of groups)
      conv_context_size: null
      # Regularization
      dropout: 0.1 # The dropout used in most of the Conformer Modules
      dropout_pre_encoder: 0.1 # The dropout used before the encoder
      dropout_emb: 0.0 # The dropout used for embeddings
      dropout_att: 0.1 # The dropout for multi-headed attention modules
      # Set to non-zero to enable stochastic depth
      stochastic_depth_drop_prob: 0.0
      stochastic_depth_mode: linear  # linear or uniform
      stochastic_depth_start_layer: 1
    
    transformer_encoder:
      _target_: nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder
      num_layers: 18
      hidden_size: ${model.model_defaults.tf_d_model} # Needs to be multiple of num_attention_heads
      inner_size: 768
      num_attention_heads: 8
      attn_score_dropout: 0.5
      attn_layer_dropout: 0.5
      ffn_dropout: 0.5
      hidden_act: relu
      pre_ln: False
      pre_ln_final_layer_norm: True
    
    loss: 
      _target_: nemo.collections.asr.losses.bce_loss.BCELoss
      weight: null # Weight for binary cross-entropy loss. Either `null` or list type input. (e.g. [0.5,0.5])
      reduction: mean

    lr: 0.0001
    optim:
      name: adamw
      lr: ${model.lr}
      # optimizer arguments
      betas: [0.9, 0.98]
      weight_decay: 1e-3

      sched:
        name: InverseSquareRootAnnealing
        warmup_steps: 2500
        warmup_ratio: null
        min_lr: 1e-06

  trainer:
    devices: 1 # number of gpus (devices)
    accelerator: gpu 
    max_epochs: 800
    max_steps: -1 # computed at runtime if not set
    num_nodes: 1
    strategy: ddp_find_unused_parameters_true # Could be "ddp"
    accumulate_grad_batches: 1
    deterministic: True
    enable_checkpointing: False
    logger: False
    log_every_n_steps: 1  # Interval of logging.
    val_check_interval: 1.0  # Set to 0.25 to check 4 times per epoch, or an int for number of iterations

  exp_manager:
    use_datetime_version: False
    exp_dir: null
    name: ${name}
    resume_if_exists: True
    resume_from_checkpoint: null # The path to a checkpoint file to continue the training, restores the whole state including the epoch, step, LR schedulers, apex, etc.
    resume_ignore_no_checkpoint: True
    create_tensorboard_logger: True
    create_checkpoint_callback: True
    create_wandb_logger: False
    checkpoint_callback_params:
      monitor: "val_f1_acc"
      mode: "max"
      save_top_k: 9
      every_n_epochs: 1
    wandb_logger_kwargs:
      resume: True
      name: null
      project: null

Hydra Configurations for Streaming Sortformer Diarizer Training
----------------------------------------------------------------

Model name convention for Streaming Sortformer Diarizer: streaming_sortformer_diarizer_<speaker count limit>-<version>.yaml

* Example `<NeMo_root>/examples/speaker_tasks/diarization/neural_diarizer/conf/streaming_sortformer_diarizer_4spk-v2.yaml`.

.. code-block:: yaml

  name: "StreamingSortformerDiarizer"
  num_workers: 18
  batch_size: 4

  model: 
    sample_rate: 16000
    pil_weight: 0.5 # Weight for Permutation Invariant Loss (PIL) used in training the Sortformer diarizer model
    ats_weight: 0.5 # Weight for Arrival Time Sort (ATS) loss in training the Sortformer diarizer model
    max_num_of_spks: 4 # Maximum number of speakers per model; currently set to 4
    streaming_mode: True

    model_defaults:
      fc_d_model: 512 # Hidden dimension size of the Fast-conformer Encoder
      tf_d_model: 192 # Hidden dimension size of the Transformer Encoder

    train_ds:
      manifest_filepath: ???
      sample_rate: ${model.sample_rate}
      num_spks: ${model.max_num_of_spks}
      session_len_sec: 90 # Maximum session length in seconds
      soft_label_thres: 0.5 # Threshold for binarizing target values; higher values make the model more conservative in predicting speaker activity.
      soft_targets: False # If True, use continuous values as target values when calculating cross-entropy loss
      labels: null
      batch_size: ${batch_size}
      shuffle: True
      num_workers: ${num_workers}
      validation_mode: False
      # lhotse config
      use_lhotse: False
      use_bucketing: True
      num_buckets: 10
      bucket_duration_bins: [10, 20, 30, 40, 50, 60, 70, 80, 90]
      pin_memory: True
      min_duration: 10
      max_duration: 90
      batch_duration: 400
      quadratic_duration: 1200
      bucket_buffer_size: 20000
      shuffle_buffer_size: 10000
      window_stride: ${model.preprocessor.window_stride}
      subsampling_factor: ${model.encoder.subsampling_factor}

    validation_ds:
      manifest_filepath: ???
      is_tarred: False
      tarred_audio_filepaths: null
      sample_rate: ${model.sample_rate}
      num_spks: ${model.max_num_of_spks}
      session_len_sec: 90 # Maximum session length in seconds
      soft_label_thres: 0.5 # A threshold value for setting up the binarized labels. The higher the more conservative the model becomes. 
      soft_targets: False
      labels: null
      batch_size: ${batch_size}
      shuffle: False
      num_workers: ${num_workers}
      validation_mode: True
      # lhotse config
      use_lhotse: False
      use_bucketing: False
      drop_last: False
      pin_memory: True
      window_stride: ${model.preprocessor.window_stride}
      subsampling_factor: ${model.encoder.subsampling_factor}
    
    test_ds:
      manifest_filepath: null
      is_tarred: False
      tarred_audio_filepaths: null
      sample_rate: 16000
      num_spks: ${model.max_num_of_spks}
      session_len_sec: 90 # Maximum session length in seconds
      soft_label_thres: 0.5
      soft_targets: False
      labels: null
      batch_size: ${batch_size}
      shuffle: False
      seq_eval_mode: True
      num_workers: ${num_workers}
      validation_mode: True
      # lhotse config
      use_lhotse: False
      use_bucketing: False
      drop_last: False
      pin_memory: True
      window_stride: ${model.preprocessor.window_stride}
      subsampling_factor: ${model.encoder.subsampling_factor}

    preprocessor:
      _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
      normalize: "NA"
      window_size: 0.025
      sample_rate: ${model.sample_rate}
      window_stride: 0.01
      window: "hann"
      features: 128
      n_fft: 512
      frame_splicing: 1
      dither: 0.00001

    sortformer_modules:
      _target_: nemo.collections.asr.modules.sortformer_modules.SortformerModules
      num_spks: ${model.max_num_of_spks} # Maximum number of speakers the model can handle
      dropout_rate: 0.5 # Dropout rate
      fc_d_model: ${model.model_defaults.fc_d_model} # Hidden dimension size for Fast Conformer encoder
      tf_d_model: ${model.model_defaults.tf_d_model} # Hidden dimension size for Transformer encoder
      # Streaming mode parameters
      spkcache_len: 188 # Length of speaker cache buffer (total number of frames for all speakers)
      fifo_len: 0 # Length of FIFO buffer for streaming processing (0 = disabled)
      chunk_len: 188 # Number of frames processed in each streaming chunk
      spkcache_update_period: 188 # Speaker cache update period in frames
      chunk_left_context: 1 # Number of previous frames for each streaming chunk
      chunk_right_context: 1 # Number of future frames for each streaming chunk
      # Speaker cache update parameters
      spkcache_sil_frames_per_spk: 3 # Number of silence frames allocated per speaker in the speaker cache
      scores_add_rnd: 0 # Standard deviation of random noise added to scores in speaker cache update (training only)
      pred_score_threshold: 0.25 # Probability threshold for internal scores processing in speaker cache update
      max_index: 99999 # Maximum allowed index value for internal processing in speaker cache update
      scores_boost_latest: 0.05 # Gain for scores for recently added frames in speaker cache update
      sil_threshold: 0.2 # Threshold for determining silence frames to calculate average silence embedding
      strong_boost_rate: 0.75 # Rate determining number of frames per speaker that receive strong score boosting
      weak_boost_rate: 1.5 # Rate determining number of frames per speaker that receive weak score boosting
      min_pos_scores_rate: 0.5 # Rate threshold for dropping overlapping frames when enough non-overlapping exist
      # Self-attention parameters (training only)
      causal_attn_rate: 0.5 # Proportion of batches that use self-attention with limited right context
      causal_attn_rc: 7 # Right context size for self-attention with limited right context

    encoder:
      _target_: nemo.collections.asr.modules.ConformerEncoder
      feat_in: ${model.preprocessor.features}
      feat_out: -1
      n_layers: 17
      d_model: ${model.model_defaults.fc_d_model}

      # Sub-sampling parameters
      subsampling: dw_striding # vggnet, striding, stacking or stacking_norm, dw_striding
      subsampling_factor: 8 # must be power of 2 for striding and vggnet
      subsampling_conv_channels: 256 # set to -1 to make it equal to the d_model
      causal_downsampling: false
      # Feed forward module's params
      ff_expansion_factor: 4
      # Multi-headed Attention Module's params
      self_attention_model: rel_pos # rel_pos or abs_pos
      n_heads: 8 # may need to be lower for smaller d_models
      # [left, right] specifies the number of steps to be seen from left and right of each step in self-attention
      att_context_size: [-1, -1] # -1 means unlimited context
      att_context_style: regular # regular or chunked_limited
      xscaling: true # scales up the input embeddings by sqrt(d_model)
      untie_biases: true # unties the biases of the TransformerXL layers
      pos_emb_max_len: 5000
      # Convolution module's params
      conv_kernel_size: 9
      conv_norm_type: 'batch_norm' # batch_norm or layer_norm or groupnormN (N specifies the number of groups)
      conv_context_size: null
      # Regularization
      dropout: 0.1 # The dropout used in most of the Conformer Modules
      dropout_pre_encoder: 0.1 # The dropout used before the encoder
      dropout_emb: 0.0 # The dropout used for embeddings
      dropout_att: 0.1 # The dropout for multi-headed attention modules
      # Set to non-zero to enable stochastic depth
      stochastic_depth_drop_prob: 0.0
      stochastic_depth_mode: linear  # linear or uniform
      stochastic_depth_start_layer: 1
    
    transformer_encoder:
      _target_: nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder
      num_layers: 18
      hidden_size: ${model.model_defaults.tf_d_model} # Needs to be multiple of num_attention_heads
      inner_size: 768
      num_attention_heads: 8
      attn_score_dropout: 0.5
      attn_layer_dropout: 0.5
      ffn_dropout: 0.5
      hidden_act: relu
      pre_ln: False
      pre_ln_final_layer_norm: True
    
    loss: 
      _target_: nemo.collections.asr.losses.bce_loss.BCELoss
      weight: null # Weight for binary cross-entropy loss. Either `null` or list type input. (e.g. [0.5,0.5])
      reduction: mean

    lr: 0.0001
    optim:
      name: adamw
      lr: ${model.lr}
      # optimizer arguments
      betas: [0.9, 0.98]
      weight_decay: 1e-3

      sched:
        name: InverseSquareRootAnnealing
        warmup_steps: 500
        warmup_ratio: null
        min_lr: 1e-06

  trainer:
    devices: 1 # number of gpus (devices)
    accelerator: gpu 
    max_epochs: 800
    max_steps: -1 # computed at runtime if not set
    num_nodes: 1
    strategy: ddp_find_unused_parameters_true # Could be "ddp"
    accumulate_grad_batches: 1
    deterministic: True
    enable_checkpointing: False
    logger: False
    log_every_n_steps: 1  # Interval of logging.
    val_check_interval: 1.0  # Set to 0.25 to check 4 times per epoch, or an int for number of iterations

  exp_manager:
    use_datetime_version: False
    exp_dir: null
    name: ${name}
    resume_if_exists: True
    resume_from_checkpoint: null # The path to a checkpoint file to continue the training, restores the whole state including the epoch, step, LR schedulers, apex, etc.
    resume_ignore_no_checkpoint: True
    create_tensorboard_logger: True
    create_checkpoint_callback: True
    create_wandb_logger: False
    checkpoint_callback_params:
      monitor: "val_f1_acc"
      mode: "max"
      save_top_k: 9
      every_n_epochs: 1
    wandb_logger_kwargs:
      resume: True
      name: null
      project: null


Hydra Configurations for (Streaming) Sortformer Diarization Post-processing
-----------------------------------------------------------------------------

Post-processing converts the floating point number based Tensor output to time stamp output. While generating the speaker-homogeneous segments, onset and offset threshold, 
paddings can be considered to render the time stamps that can lead to the lowest diarization error rate (DER). This post-processing can be applied to both offline and streaming Sortformer diarizer.


By default, post-processing is bypassed, and only binarization is performed. If you want to reproduce DER scores reported on NeMo model cards, you need to apply post-processing steps. Use batch_size = 1 to have the longest inference window and the highest possible accuracy.

.. code-block:: yaml

  parameters: 
    onset: 0.64  # Onset threshold for detecting the beginning of a speech segment
    offset: 0.74  # Offset threshold for detecting the end of a speech segment
    pad_onset: 0.06  # Adds the specified duration at the beginning of each speech segment
    pad_offset: 0.0  # Adds the specified duration at the end of each speech segment
    min_duration_on: 0.1  # Removes short speech segments if the duration is less than the specified minimum duration
    min_duration_off: 0.15  # Removes short silences if the duration is less than the specified minimum duration


Cascaded Speaker Diarization Configuration Files
================================================

Both training and inference of cascaded speaker diarization is configured by ``.yaml`` files. The diarizer section will generally require information about the dataset(s) being used, models used in this pipeline, as well as inference related parameters such as post processing of each models. The sections on this page cover each of these in more detail.

.. note::
  For model details and deep understanding about configs, training, fine-tuning and evaluations,
  please refer to ``<NeMo_root>/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb`` and ``<NeMo_root>/tutorials/speaker_tasks/Speaker_Diarization_Training.ipynb``;
  for other applications such as possible integration with ASR, have a look at ``<NeMo_root>/tutorials/speaker_tasks/ASR_with_SpeakerDiarization.ipynb``.


Hydra Configurations for Diarization Training 
---------------------------------------------

Currently, NeMo supports Multi-scale diarization decoder (MSDD) as a neural diarizer model. MSDD is a speaker diarization model based on initializing clustering and multi-scale segmentation input. Example configuration files for MSDD model training can be found in ``<NeMo_root>/examples/speaker_tasks/diarization/conf/neural_diarizer/``.

* Model name convention for MSDD: msdd_<number of scales>scl_<longest scale in decimal second (ds)>_<shortest scale in decimal second (ds)>_<overlap percentage of window shifting>Povl_<hidden layer size>x<number of LSTM layers>x<number of CNN output channels>x<repetition count of conv layer>
* Example: ``msdd_5scl_15_05_50Povl_256x3x32x2.yaml`` has 5 scales, the longest scale is 1.5 sec, the shortest scale is 0.5 sec, with 50 percent overlap, hidden layer size is 256, 3 LSTM layers, 32 CNN channels, 2 repeated Conv layers

MSDD model checkpoint (.ckpt) and NeMo file (.nemo) contain speaker embedding model (TitaNet) and the speaker model is loaded along with standalone MSDD module. Note that MSDD models require more than one scale. Thus, the parameters in ``diarizer.speaker_embeddings.parameters`` should have more than one scale to function as a MSDD model.


General Diarizer Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The items (OmegaConfig keys) directly under ``model`` determines segmentation and clustering related parameters. Multi-scale parameters (``window_length_in_sec``, ``shift_length_in_sec`` and ``multiscale_weights``) are specified. ``max_num_of_spks``, ``scale_n``, ``soft_label_thres`` and ``emb_batch_size`` are set here and then assigned to dataset configurations.

.. code-block:: yaml

  diarizer:
    out_dir: null
    oracle_vad: True # If True, uses RTTM files provided in manifest file to get speech activity (VAD) timestamps
    speaker_embeddings:
      model_path: ??? # .nemo local model path or pretrained model name (titanet_large is recommended)
      parameters:
        window_length_in_sec: [1.5,1.25,1.0,0.75,0.5] # Window length(s) in sec (floating-point number). either a number or a list. ex) 1.5 or [1.5,1.0,0.5]
        shift_length_in_sec: [0.75,0.625,0.5,0.375,0.25] # Shift length(s) in sec (floating-point number). either a number or a list. ex) 0.75 or [0.75,0.5,0.25]
        multiscale_weights: [1,1,1,1,1] # Weight for each scale. should be null (for single scale) or a list matched with window/shift scale count. ex) [0.33,0.33,0.33]
        save_embeddings: True # Save embeddings as pickle file for each audio input.


  num_workers: ${num_workers} # Number of workers used for data-loading.
  max_num_of_spks: 2 # Number of speakers per model. This is currently fixed at 2.
  scale_n: 5 # Number of scales for MSDD model and initializing clustering.
  soft_label_thres: 0.5 # Threshold for creating discretized speaker label from continuous speaker label in RTTM files.
  emb_batch_size: 0 # If this value is bigger than 0, corresponding number of embedding vectors are attached to torch graph and trained.

Dataset Configuration
^^^^^^^^^^^^^^^^^^^^^

Training, validation, and test parameters are specified using the ``train_ds``, ``validation_ds``, and
``test_ds`` sections in the configuration YAML file, respectively. The items such as ``num_spks``, ``soft_label_thres`` and ``emb_batch_size`` follow the settings in ``model`` key. You may also leave fields such as the ``manifest_filepath`` or ``emb_dir`` blank, and then specify it via command-line interface. Note that ``test_ds`` is not used during training and only used for speaker diarization inference.

.. code-block:: yaml

  train_ds:
    manifest_filepath: ???
    emb_dir: ???
    sample_rate: ${sample_rate}
    num_spks: ${model.max_num_of_spks}
    soft_label_thres: ${model.soft_label_thres}
    labels: null
    batch_size: ${batch_size}
    emb_batch_size: ${model.emb_batch_size}
    shuffle: True

  validation_ds:
    manifest_filepath: ???
    emb_dir: ???
    sample_rate: ${sample_rate}
    num_spks: ${model.max_num_of_spks}
    soft_label_thres: ${model.soft_label_thres}
    labels: null
    batch_size: 2
    emb_batch_size: ${model.emb_batch_size}
    shuffle: False

  test_ds:
    manifest_filepath: null
    emb_dir: null
    sample_rate: 16000
    num_spks: ${model.max_num_of_spks}
    soft_label_thres: ${model.soft_label_thres}
    labels: null
    batch_size: 2
    shuffle: False
    seq_eval_mode: False


Pre-processor Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the MSDD configuration, pre-processor configuration follows the pre-processor of the embedding extractor model.

.. code-block:: yaml

  preprocessor:
    _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    normalize: "per_feature"
    window_size: 0.025
    sample_rate: ${sample_rate}
    window_stride: 0.01
    window: "hann"
    features: 80
    n_fft: 512
    frame_splicing: 1
    dither: 0.00001


Model Architecture Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hyper-parameters for MSDD models are under the ``msdd_module`` key. The model architecture can be changed by setting up the ``weighting_scheme`` and ``context_vector_type``. The detailed explanation for architecture can be found in the :doc:`Models <./models>` page.

.. code-block:: yaml

  msdd_module:
    _target_: nemo.collections.asr.modules.msdd_diarizer.MSDD_module
    num_spks: ${model.max_num_of_spks} # Number of speakers per model. This is currently fixed at 2.
    hidden_size: 256 # Hidden layer size for linear layers in MSDD module
    num_lstm_layers: 3 # Number of stacked LSTM layers
    dropout_rate: 0.5 # Dropout rate
    cnn_output_ch: 32 # Number of filters in a conv-net layer.
    conv_repeat: 2 # Determines the number of conv-net layers. Should be greater or equal to 1.
    emb_dim: 192 # Dimension of the speaker embedding vectors
    scale_n: ${model.scale_n} # Number of scales for multiscale segmentation input
    weighting_scheme: 'conv_scale_weight' # Type of weighting algorithm. Options: ('conv_scale_weight', 'attn_scale_weight')
    context_vector_type: 'cos_sim' # Type of context vector: options. Options: ('cos_sim', 'elem_prod')

Loss Configurations
^^^^^^^^^^^^^^^^^^^

Neural diarizer uses a binary cross entropy (BCE) loss. A set of weights for negative (absence of the speaker's speech) and positive (presence of the speaker's speech) can be provided to the loss function.

.. code-block:: yaml

  loss: 
    _target_: nemo.collections.asr.losses.bce_loss.BCELoss
    weight: null # Weight for binary cross-entropy loss. Either `null` or list type input. (e.g. [0.5,0.5])


Hydra Configurations for Diarization Inference
==============================================

Example configuration files for speaker diarization inference can be found in ``<NeMo_root>/examples/speaker_tasks/diarization/conf/inference/``. Choose a yaml file that fits your targeted domain. For example, if you want to diarize audio recordings of telephonic speech, choose ``diar_infer_telephonic.yaml``.

The configurations for all the components of diarization inference are included in a single file named ``diar_infer_<domain>.yaml``. Each ``.yaml`` file has a few different sections for the following modules: VAD, Speaker Embedding, Clustering and ASR.

In speaker diarization inference, the datasets provided in manifest format denote the data that you would like to perform speaker diarization on. 

Diarizer Configurations
-----------------------

An example ``diarizer``  Hydra configuration could look like:

.. code-block:: yaml

  diarizer:
    manifest_filepath: ???
    out_dir: ???
    oracle_vad: False # If True, uses RTTM files provided in manifest file to get speech activity (VAD) timestamps
    collar: 0.25 # Collar value for scoring
    ignore_overlap: True # Consider or ignore overlap segments while scoring

Under ``diarizer`` key, there are ``vad``, ``speaker_embeddings``, ``clustering`` and ``asr`` keys containing configurations for the inference of the corresponding modules.

Configurations for Voice Activity Detector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters for VAD model are provided as in the following Hydra config example.

.. code-block:: yaml

  vad:
    model_path: null # .nemo local model path or pretrained model name or none
    external_vad_manifest: null # This option is provided to use external vad and provide its speech activity labels for speaker embeddings extraction. Only one of model_path or external_vad_manifest should be set

    parameters: # Tuned parameters for CH109 (using the 11 multi-speaker sessions as dev set) 
      window_length_in_sec: 0.15  # Window length in sec for VAD context input 
      shift_length_in_sec: 0.01 # Shift length in sec for generate frame level VAD prediction
      smoothing: "median" # False or type of smoothing method (eg: median)
      overlap: 0.875 # Overlap ratio for overlapped mean/median smoothing filter
      onset: 0.4 # Onset threshold for detecting the beginning and end of a speech 
      offset: 0.7 # Offset threshold for detecting the end of a speech
      pad_onset: 0.05 # Adding durations before each speech segment 
      pad_offset: -0.1 # Adding durations after each speech segment 
      min_duration_on: 0.2 # Threshold for short speech segment deletion
      min_duration_off: 0.2 # Threshold for small non_speech deletion
      filter_speech_first: True 

Configurations for Speaker Embedding in Diarization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters for speaker embedding model are provided in the following Hydra config example. Note that multiscale parameters either accept list or single floating point number.

.. code-block:: yaml

  speaker_embeddings:
    model_path: ??? # .nemo local model path or pretrained model name (titanet_large, ecapa_tdnn or speakerverification_speakernet)
    parameters:
      window_length_in_sec: 1.5 # Window length(s) in sec (floating-point number). Either a number or a list. Ex) 1.5 or [1.5,1.25,1.0,0.75,0.5]
      shift_length_in_sec: 0.75 # Shift length(s) in sec (floating-point number). Either a number or a list. Ex) 0.75 or [0.75,0.625,0.5,0.375,0.25]
      multiscale_weights: null # Weight for each scale. should be null (for single scale) or a list matched with window/shift scale count. Ex) [1,1,1,1,1]
      save_embeddings: False # Save embeddings as pickle file for each audio input.

Configurations for Clustering in Diarization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters for clustering algorithm are provided in the following Hydra config example.

.. code-block:: yaml
  
  clustering:
    parameters:
      oracle_num_speakers: False # If True, use num of speakers value provided in the manifest file.
      max_num_speakers: 20 # Max number of speakers for each recording. If oracle_num_speakers is passed, this value is ignored.
      enhanced_count_thres: 80 # If the number of segments is lower than this number, enhanced speaker counting is activated.
      max_rp_threshold: 0.25 # Determines the range of p-value search: 0 < p <= max_rp_threshold. 
      sparse_search_volume: 30 # The higher the number, the more values will be examined with more time. 

Configurations for Diarization with ASR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following configuration needs to be appended under ``diarizer`` to run ASR with diarization to get a transcription with speaker labels. 

.. code-block:: yaml

  asr:
    model_path: ??? # Provide NGC cloud ASR model name. stt_en_conformer_ctc_* models are recommended for diarization purposes.
    parameters:
      asr_based_vad: False # if True, speech segmentation for diarization is based on word-timestamps from ASR inference.
      asr_based_vad_threshold: 50 # threshold (multiple of 10ms) for ignoring the gap between two words when generating VAD timestamps using ASR based VAD.
      asr_batch_size: null # Batch size can be dependent on each ASR model. Default batch sizes are applied if set to null.
      lenient_overlap_WDER: True # If true, when a word falls into speaker-overlapped regions, consider the word as a correctly diarized word.
      decoder_delay_in_sec: null # Native decoder delay. null is recommended to use the default values for each ASR model.
      word_ts_anchor_offset: null # Offset to set a reference point from the start of the word. Recommended range of values is [-0.05  0.2]. 
      word_ts_anchor_pos: "start" # Select which part of the word timestamp we want to use. The options are: 'start', 'end', 'mid'.
      fix_word_ts_with_VAD: False # Fix the word timestamp using VAD output. You must provide a VAD model to use this feature.
      colored_text: False # If True, use colored text to distinguish speakers in the output transcript.
      print_time: True # If True, the start of the end time of each speaker turn is printed in the output transcript.
      break_lines: False # If True, the output transcript breaks the line to fix the line width (default is 90 chars)
    
    ctc_decoder_parameters: # Optional beam search decoder (pyctcdecode)
      pretrained_language_model: null # KenLM model file: .arpa model file or .bin binary file.
      beam_width: 32
      alpha: 0.5
      beta: 2.5

    realigning_lm_parameters: # Experimental feature
      arpa_language_model: null # Provide a KenLM language model in .arpa format.
      min_number_of_words: 3 # Min number of words for the left context.
      max_number_of_words: 10 # Max number of words for the right context.
      logprob_diff_threshold: 1.2  # The threshold for the difference between two log probability values from two hypotheses.
