Configuration Files
===================

The SpeechLM2 models use YAML configuration files to define model architecture, training parameters, and data settings.
This page describes the configuration structure and important parameters for each model type in the collection.

Configuration Structure
-----------------------

SpeechLM2 configuration files typically have the following high-level structure:

.. code-block:: yaml

    model:
      # Model architecture settings
      ...
    
    trainer:
      # PyTorch Lightning trainer settings
      ...
    
    exp_manager:
      # Experiment logging settings
      ...
    
    data:
      # Dataset settings
      ...

SALM Configuration
------------------

The SALM (Speech-Augmented Language Model) configuration includes settings for the pretrained LLM, audio perception module, and training parameters.
See the `SALM paper <https://arxiv.org/abs/2310.09424>`_ for more details.

.. code-block:: yaml

    model:
      # Pretrained model paths
      pretrained_llm: "TinyLlama/TinyLlama_v1.1"  # HF model path
      pretrained_asr: "stt_en_fastconformer_hybrid_large_streaming_80ms"  # NeMo checkpoint name
      pretrained_weights: True  # Whether to load weights or just architecture
      
      # Special token settings
      audio_locator_tag: "<audio>"  # Tag to replace with audio embeddings
      
      # Freezing parameters
      freeze_params:
        - "^llm\\.model\\.layers\\.[0-4]\\..+$"  # Regex patterns for parameters to freeze
      prevent_freeze_params: []  # Override freeze_params for specific submodules
      
      # Optional LoRA settings for efficient fine-tuning
      lora:
        task_type: CAUSAL_LM
        r: 8
        lora_alpha: 32
        lora_dropout: 0.1
      
      # Audio perception module configuration
      perception:
        target: nemo.collections.speechlm2.modules.perception.AudioPerceptionModule
        
        preprocessor:
          normalize: 'NA'
        
        encoder:
          self_attention_model: rel_pos
          att_context_size: [-1, -1]
          conv_context_size: regular
          conv_norm_type: batch_norm
        
        modality_adapter:
          _target_: nemo.collections.asr.modules.ConformerEncoder
          feat_in: 1024
          feat_out: -1
          n_layers: 2
          d_model: 1024
          subsampling: dw_striding
          subsampling_factor: 1
          subsampling_conv_channels: 256
          causal_downsampling: false
          ff_expansion_factor: 4
          self_attention_model: rel_pos
          n_heads: 8
          att_context_size: [-1, -1]
          att_context_style: regular
          xscaling: true
          untie_biases: true
          pos_emb_max_len: 5000
          conv_kernel_size: 9
          conv_norm_type: batch_norm
          conv_context_size: null
          dropout: 0
          dropout_pre_encoder: 0
          dropout_emb: 0.0

DuplexS2SModel Configuration
-----------------------------

The DuplexS2SModel adds speech generation capabilities to the configuration:

.. code-block:: yaml

    model:
      # Pretrained model paths
      pretrained_llm: "TinyLlama/TinyLlama_v1.1"
      pretrained_audio_codec: "path/to/audio_codec.nemo"
      pretrained_asr: "stt_en_fastconformer_hybrid_large_streaming_80ms"
      scoring_asr: "stt_en_fastconformer_transducer_large"  # used only in validation
      
      # Loss weights
      audio_loss_weight: 4
      text_loss_weight: 3
      
      # Perception module config (similar to SALM)
      perception:
        # ... (similar to SALM perception module)

DuplexS2SSpeechDecoderModel Configuration
-----------------------------------------

The DuplexS2SSpeechDecoderModel is similar to DuplexS2SModel, but focuses on an additional speech generation transformer decoder:

.. code-block:: yaml

    model:
      # Pretrained model paths
      pretrained_llm: "TinyLlama/TinyLlama_v1.1"
      pretrained_audio_codec: "path/to/audio_codec.nemo"
      pretrained_asr: "stt_en_fastconformer_hybrid_large_streaming_80ms"
      
      # Speech decoder settings
      speech_decoder:
        target: nemo.collections.speechlm2.modules.speech_generation.TransformerARSpeechDecoder
        d_model: 1024
        n_layers: 12
        n_heads: 16
        d_kv: 64
        d_ff: 4096
        max_seq_len: 2048
        dropout: 0.1
        layernorm_epsilon: 1e-5
        activation_function: "gelu_new"
        init_method_std: 0.02
        use_cache: True
        
      # ... other settings

Trainer Configuration
---------------------

The trainer section contains PyTorch Lightning Trainer settings:

.. code-block:: yaml

    trainer:
      devices: 1
      num_nodes: 1
      accelerator: gpu
      precision: bf16-true
      logger: false
      enable_checkpointing: false  # handled by exp_manager
      replace_sampler_ddp: false   # handled by lhotse
      max_epochs: null
      max_steps: 100000
      log_every_n_steps: 10
      val_check_interval: 2000
      accumulate_grad_batches: 1
      gradient_clip_val: 1.0

Experiment Manager Configuration
--------------------------------

The exp_manager section contains settings for experiment logging and model checkpointing:

.. code-block:: yaml

    exp_manager:
      explicit_log_dir: path/to/output_dir
      exp_dir: null
      name: ${name}
      create_wandb_logger: false  # set to true if you want to use wandb
      wandb_logger_kwargs:
        project: null
        name: null
      resume_if_exists: true
      resume_ignore_no_checkpoint: true
      create_checkpoint_callback: true
      checkpoint_callback_params:
        monitor: val_loss
        filename: "{step}"  # checkpoint name will be step=<step>.ckpt
        save_top_k: 1
        mode: min
      create_tensorboard_logger: false  # set to true if you want to use tensorboard
      version: null

Data Configuration
------------------

The data section defines dataset paths, preprocessing, and data loading parameters:

.. code-block:: yaml

    data:
      train_ds:
        sample_rate: ${data.target_sample_rate}
        input_cfg:
          - type: lhotse_shar
            shar_path: /path/to/train_data
        seed: 42
        shard_seed: "randomized"
        num_workers: 4
        batch_size: 16
        # Optional bucketing settings
        # batch_duration: 100
        # bucket_duration_bins: [8.94766,10.1551,11.64118,19.30376,42.85]
        # use_bucketing: true
        # num_buckets: 5
        # bucket_buffer_size: 5000
    
      validation_ds:
        datasets:
          val_set_name:
            shar_path: /path/to/validation_data
        sample_rate: ${data.target_sample_rate}
        batch_size: 1
        seed: 42
        shard_seed: "randomized"

Depending on the model, there may be additional options available under ``data`` namespace that are passed to the corresponding Dataset class.
For example, S2S models have:

.. code-block:: yaml

    data:
      frame_length: 0.08
      source_sample_rate: 16000
      target_sample_rate: 22050
      input_roles: ["user", "User"]
      output_roles: ["agent", "Assistant"]

      train_ds: ...

Important Configuration Parameters
-----------------------------------

Model Parameters
^^^^^^^^^^^^^^^^

- **pretrained_llm**: Path to the pretrained HuggingFace LLM
- **pretrained_asr**: Name of the pretrained NeMo ASR model used for perception
- **pretrained_audio_codec**: Path to the pretrained audio codec model (for speech generation)
- **freeze_params**: Regex patterns of parameters to freeze during training
- **audio_loss_weight/text_loss_weight**: Weighting of different loss components

Perception Module
^^^^^^^^^^^^^^^^^

- **self_attention_model**: Type of attention mechanism ("rel_pos" or "abs_pos")
- **att_context_size**: Context window size for attention ([left, right])
- **conv_context_size**: Context type for convolutions ("causal" or "regular")
- **n_layers**: Number of encoder layers
- **d_model**: Model dimension size

Data Parameters
^^^^^^^^^^^^^^^

- **frame_length**: Frame duration in seconds
- **source_sample_rate/target_sample_rate**: Sample rates for input/output audio
- **input_roles/output_roles**: Speaker roles for input and output
- **batch_size**: Number of samples per batch
- **use_bucketing**: Whether to use length-based bucketing for efficient batching

Example Configuration Files
---------------------------

Example configurations for all model types can be found in the example directory:

- SALM: `examples/speechlm2/conf/salm.yaml`
- DuplexS2SModel: `examples/speechlm2/conf/s2s_duplex.yaml`
- DuplexS2SSpeechDecoderModel: `examples/speechlm2/conf/s2s_duplex_speech_decoder.yaml`

Using Configuration Files
-------------------------

You can use these configurations with the training scripts by specifying the config path:

.. code-block:: bash

    # Train SALM model
    python examples/speechlm2/salm_train.py \
      --config-path=conf \
      --config-name=salm

You can also override configuration values from the command line:

.. code-block:: bash

    python examples/speechlm2/salm_train.py \
      --config-path=conf \
      --config-name=salm \
      model.pretrained_llm="different/llm/path" \
      trainer.max_steps=1000 \
      data.train_ds.batch_size=8 