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

      # Fine-tune from a previous training checkpoint (weights only, fresh optimizer)
      init_from_checkpoint: null  # path to .ckpt, DCP dir, or HF dir

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

SALMAutomodel Configuration
----------------------------

The SALMAutomodel configuration extends the SALM configuration with NeMo Automodel
support. The key difference is ``use_nemo_automodel: true`` and the use of
``AutomodelParallelStrategy`` instead of ``DDPStrategy``.

The example below shows a configuration for training with NVIDIA Nemotron Nano V3
MoE as the LLM backbone, with Expert Parallelism across 8 GPUs:

.. code-block:: yaml

    model:
      use_nemo_automodel: true  # Selects SALMAutomodel in salm_train.py
      pretrained_llm: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
      pretrained_asr: "nvidia/canary-1b-flash"
      pretrained_weights: True
      encoder_chunk_size_seconds: 30.0

      freeze_params:
        - "^llm\\..+$"
        - "^perception\\.preprocessor\\..+$"
        - "^perception\\.encoder\\..+$"
      prevent_freeze_params: []

      # LoRA uses Automodel-native format (not HF PEFT):
      # lora:
      #   dim: 128
      #   alpha: 256
      #   dropout: 0.01
      #   target_modules: ["q_proj", "v_proj"]

      perception:
        target: nemo.collections.speechlm2.modules.perception.AudioPerceptionModule
        output_dim: 2048
        modality_adapter:
          _target_: nemo.collections.speechlm2.modules.perception.IdentityConnector
          d_model: 1024

    trainer:
      strategy:
        _target_: nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy
        ep_size: 8  # Expert Parallelism across 8 GPUs for MoE
        # tp_size: 1
        # dp_size: null  # inferred

NeMo Automodel applies MoE-specific optimizations automatically when an MoE model
is detected:

* **Grouped GEMM** — fuses expert computations into a single batched matrix multiply
  for higher GPU throughput.
* **DeepEP** (Deep Expert Parallelism) — efficient all-to-all expert routing across
  GPUs, minimizing communication overhead for MoE layers.

Note the differences from the SALM configuration:

* ``model.use_nemo_automodel: true`` — selects ``SALMAutomodel`` in the training script.
* ``model.pretrained_llm`` can point to MoE models like ``nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16``.
* ``trainer.strategy._target_`` uses ``AutomodelParallelStrategy`` instead of ``ModelParallelStrategy``.
* ``ep_size`` controls Expert Parallelism on the FSDP data-parallel axis — dense layers are sharded via FSDP2, while MoE layers use EP for expert routing on the same GPUs.
* LoRA config uses ``dim``/``alpha`` keys (Automodel native) instead of ``r``/``lora_alpha`` (HF PEFT).
* No ``embed_tokens`` freeze pattern — embeddings stay inside the LLM.
* ``encoder_chunk_size_seconds`` controls long-audio chunking for the speech encoder.
  Audio rows longer than this value are split on the time axis, encoded as a chunk
  batch, and concatenated back into one embedding sequence before the LLM forward.
  Set it to ``null`` to disable chunking. With a ``ParallelExpertEncoder`` and
  ``packed_encoder_sequences: true``, this same value instead chunks both the ASR
  and diarization branches after feature stacking; set the data audio-token
  estimator's ``chunk_size_seconds`` to ``null`` because this does not change the
  total encoder-token count.

SALMAutomodel-Specific Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SALMAutomodel config exposes a few extra knobs that pass through to NeMo
Automodel. All are optional — defaults preserve standard behavior.

**Garbage collection:**

.. code-block:: yaml

    model:
      # Optional positive optimizer-step interval; null keeps automatic GC.
      gc_every_steps: null

Setting ``gc_every_steps`` to a positive integer disables Python's automatic
garbage collector at fit start and uses NeMo Automodel's generation-1 collector
at that optimizer-step cadence. This avoids an occasional generation-2 scan on
one distributed rank delaying all peers at the next collective. The cadence is
counted in optimizer steps, so gradient accumulation does not increase the
collection frequency. Leave it ``null`` unless profiling shows GC-related rank
stragglers.

**MoE training:**

.. code-block:: yaml

    model:
      # MoE auxiliary load-balancing loss coefficient. > 0 to enable.
      # Gradients are injected during backward; reported CE loss is unchanged.
      aux_loss_coeff: 0.0

      # When true, unfreezes Gate.weight so the router can adapt to new data.
      # Default false keeps pretrained routing frozen.
      train_gate: false

      # Per-step expert balance / utilization metrics.
      moe_metrics:
        enabled: true
        mode: brief                  # "brief" or "detailed"
        detailed_every_steps: null   # null = every step when mode=detailed
        top_k_experts: 5             # top/bottom utilization experts to report

When ``aux_loss_coeff > 0``, SALMAutomodel sets ``MoEAuxLossAutoScaler.main_loss_backward_scale``
to the DP group size at ``on_fit_start`` so FSDP's gradient averaging cancels out and the
net aux-loss gradient scale stays at 1.

**torch.compile:**

.. code-block:: yaml

    model:
      compile:
        enabled: false
        mode: default          # "default" | "reduce-overhead" | "max-autotune"
        fullgraph: false
        dynamic: true          # Recommended for variable-length audio
        backend: null          # null = inductor
        dynamo_cache_size_limit: 256

**Backend dispatch (attention / linear / norm / MoE kernels):**

.. code-block:: yaml

    model:
      automodel_backend:
        attn: te                  # "te" | "sdpa" | "flex"
        linear: te                # "torch" | "te"
        rms_norm: torch_fp32      # "torch" | "torch_fp32" | "te"
        rope_fusion: false          # Current Automodel main force-disables fused RoPE
        experts: torch_mm         # "torch" | "te" | "gmm" | "torch_mm"
        dispatcher: deepep        # "torch" | "deepep" | "hybridep" | "uccl_ep"
        dispatcher_num_sms: 20

      # Pin SDPA kernel when automodel_backend.attn=sdpa.
      # E.g. ["flash_attention"] forces FA2 and errors if unavailable.
      sdpa_method: null

Defaults come from Automodel's ``BackendConfig`` and select available kernels and
dispatchers; override here to pin a specific backend (for example, ``attn: sdpa``
to bypass TE). ``enable_deepep`` is no longer supported; set ``dispatcher`` and
``experts`` explicitly.

**Packed sequences (THD):**

.. code-block:: yaml

    model:
      packed_sequences: true   # default false (right-padded BSHD path)
      automodel_backend:
        attn: te               # THD path dispatches TE varlen FlashAttention

When ``packed_sequences`` is true, ``SALMAutomodel.prepare_inputs`` packs
each minibatch into a single flat ``[T_total, H]`` sequence with a
``cu_seqlens`` index instead of right-padding to ``[B, T_max, H]``.
``SALMAutomodel`` then forwards the THD metadata (``qkv_format``,
``cu_seqlens``, ``position_ids``, ``max_seqlen``) through ``forward()`` to
the LLM. The TE attention preprocessor splits the singular ``max_seqlen``
into the ``max_seqlen_q`` / ``max_seqlen_kv`` pair that
``DotProductAttention`` requires for ``qkv_format="thd"``. The packing also
rounds each utterance's flat length up to a multiple of ``2 * cp_size`` so
the same THD batch satisfies TE's CP DualChunkSwap contract — see the
"Context Parallelism (CP)" subsection in
:doc:`training_and_scaling` for the recommended pairing with ``cp_size > 1``.

Padding overhead drops from ``O(B * (T_max - T_avg))`` to
``O(per-utt rounding to 2*cp_size)``. Throughput improvement scales with
the variance of utterance lengths in your bucketing.

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

DuplexSTTModel Configuration
--------------------------------------

The DuplexSTTModel is a speech-to-text model that processes duplex audio conversations and generates agent text responses:

.. code-block:: yaml

    model:
      # Pretrained model paths
      pretrained_llm: "TinyLlama/TinyLlama_v1.1"
      pretrained_asr: "stt_en_fastconformer_hybrid_large_streaming_80ms"
        
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
- **encoder_chunk_size_seconds**: Speech-encoder chunk size in seconds for long audio inputs
  (supported by both ``SALM`` and ``SALMAutomodel``). Leave as ``null`` to encode each audio
  row directly
- **pretrained_audio_codec**: Path to the pretrained audio codec model (for speech generation)
- **init_from_checkpoint**: Path to a training checkpoint to initialize model weights from (see :ref:`fine-tuning-from-checkpoint` below)
- **freeze_params**: Regex patterns of parameters to freeze during training
- **audio_loss_weight/text_loss_weight**: Weighting of different loss components

Perception Module
^^^^^^^^^^^^^^^^^

- **self_attention_model**: Type of attention mechanism ("rel_pos" or "abs_pos")
- **att_context_size**: Context window size for attention ([left, right])
- **conv_context_size**: Context type for convolutions ("causal" or "regular")
- **n_layers**: Number of encoder layers
- **d_model**: Model dimension size
- **sync_max_audio_length**: Whether the encoder all-reduces the longest input length across ranks
  on every forward. Set it to ``false`` when ranks run a data-dependent number of encoder forwards
  (for example decode-only validation), or DDP can deadlock. When a ``ParallelExpertEncoder`` is
  mounted, this value is applied to both of its branches, unless the encoder's own settings below
  set it.

Parallel Expert Encoder
^^^^^^^^^^^^^^^^^^^^^^^

A ``ParallelExpertEncoder`` replaces the perception encoder, either from a self-contained bundle
(``model.pe_encoder_path``, optionally with ``model.pe_encoder_overrides``) or assembled from an ASR
and a Sortformer checkpoint (``model.parallel_expert_encoder`` with ``asr_model``, ``diar_model``
and any encoder constructor argument). Every applied override is logged when the encoder is
mounted. An HF export replaces ``model.pe_encoder_path`` with ``model.pe_encoder_config``, the
bundle's config with the overrides applied, which rebuilds the same encoder without the bundle
file (see :doc:`/speechlm2/evaluation`).

StreamingSTT models decode chunk by chunk, so they mount a ``StreamingParallelExpertEncoder`` on
both routes: a bundle always mounts as the streaming class, whatever ``target`` its
``model_config.yaml`` names, and the two-checkpoint route does so unless
``model.parallel_expert_encoder.streaming`` is ``false``. A bundle's weights and offline outputs are
the same for both classes; the streaming class adds the cache-aware streaming interface, whose step
applies the diarizer input's normalization to each chunk on its own (see ``diar_normalize_type``
below). SALM's bundle mount is unchanged (a local ``.nemo`` mounts as the plain
``ParallelExpertEncoder``).

The ASR branch receives un-normalized mels and replays ``asr_normalize_type`` itself. On the
two-checkpoint route, ``model.parallel_expert_encoder.asr_normalize_type`` takes these values:

- absent: ``per_feature``, as before, with a warning. This ignores the ASR checkpoint's own
  ``preprocessor.normalize``, so an ASR model trained with ``normalize: NA`` is fed re-normalized
  mels;
- ``null`` or ``NA``: no normalization;
- ``per_feature`` or ``all_features``: that normalization;
- ``auto``: the ASR checkpoint's ``preprocessor.normalize`` (``NA`` or ``null`` gives no
  normalization; ``per_feature`` and ``all_features`` are replayed as given). Any other checkpoint
  value, or none, is an error. The resolved value (``NA`` for no normalization) replaces ``auto``
  in the model config and in the saved hyperparameters, so a checkpoint or an export reloads with
  the value it was trained with.

``auto`` needs the ASR checkpoint's preprocessor config, so a bundle (``model.pe_encoder_overrides``)
rejects it.

The diarizer branch also receives un-normalized mels, which bypass the diarizer's own preprocessor.
``diar_normalize_type`` sets the normalization applied to them, the same way on both encoder classes
and both routes (the bundle config, ``model.pe_encoder_overrides``, or
``model.parallel_expert_encoder``):

- absent: the diarizer's own ``preprocessor.normalize``, the normalization it was trained with (its
  preprocessor's default if its config states none). The encoder logs what it resolved. A value
  other than ``NA``, ``null``, ``per_feature`` or ``all_features`` is an error: set
  ``diar_normalize_type`` for such a diarizer;
- ``null`` or ``NA``: no normalization;
- ``per_feature`` or ``all_features``: that normalization, whatever the diarizer's own is.

Any other value is an error. The offline (training) forward and online inference normalize each
whole utterance. The streaming step of a ``StreamingParallelExpertEncoder`` normalizes each chunk on
its own, as it does the ASR input with ``asr_normalize_type``, so a stream feeds the diarizer
differently normalized mels than training did. The first time the streaming class normalizes the
diarizer input, it logs a warning saying so.

The effective value is written to the bundle config that an HF export embeds, as ``null`` for none.
A bundle saved with ``ParallelExpertEncoderPT.save_to_nemo`` and, on the two-checkpoint route,
``model.parallel_expert_encoder`` in the model config and in the saved hyperparameters record it as
``NA`` for none. ``null`` and ``NA`` both reload as no normalization, so a reload normalizes the
diarizer input the same way, also into the other encoder class or after the diarizer checkpoint
changed.

The keys accepted in ``model.pe_encoder_overrides``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Effect
   * - ``speaker_feature_config_version``, ``speaker_feature_mode``, ``speaker_activity_threshold``
     - The speaker-feature fusion contract (thresholded or continuous speaker activity).
   * - ``asr_normalize_type``, ``spk_kernel_scale``
     - Normalization replayed on the ASR branch (``per_feature``, ``all_features``, or ``null``/``NA``
       for none; ``auto`` is rejected); scale of the speaker kernel.
   * - ``missing_rttm_target``
     - Rows whose speaker targets are all at or below this value have no RTTM and are filled from
       the diarizer. Default ``-1.0``.
   * - ``sync_max_audio_length``
     - Applied to both branches (the ASR encoder and the diarizer's encoder). Takes precedence over
       ``model.perception.encoder.sync_max_audio_length``. Unset keeps each branch's own value.
       ``model.parallel_expert_encoder.sync_max_audio_length`` does the same on the two-checkpoint
       route.
   * - ``diar_normalize_type``
     - Normalization applied to the diarizer input (``per_feature``, ``all_features``, or
       ``null``/``NA`` for none), on both encoder classes. Unset keeps the diarizer's own
       ``preprocessor.normalize`` (see above).
   * - ``chunk_size_seconds``, ``frame_shift_seconds``
     - Rejected with an error naming the key, because nothing in this encoder reads them. Use
       ``model.encoder_chunk_size_seconds`` to chunk long audio.

Speaker Counts (StreamingSTT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three settings describe speakers in a multi-speaker StreamingSTT recipe:

- ``model.speaker_tokens.max_speakers``: the number of ``<spk:N>`` tags registered as single
  tokens;
- ``data.dataset.multispeaker_cfg.num_speakers``: the number of speaker-target columns the dataset
  builds from the RTTMs;
- the speaker count of a mounted ``ParallelExpertEncoder`` (its diarizer's ``num_spks``), which is
  the number of columns it fuses.

When the model is built with a dataset config whose ``multispeaker_cfg`` is enabled (as
``streaming_stt_train.py`` does), construction checks that:

- ``max_speakers`` is at most ``num_speakers``. Fewer tags than columns is valid: for example an
  8-speaker Sortformer with ``num_speakers: 8`` and ``max_speakers: 4``;
- ``num_speakers`` equals the encoder's speaker count. Otherwise the first forward with speaker
  targets fails with a shape error;
- ``multispeaker_cfg.missing_rttm_target`` equals the encoder's ``missing_rttm_target``. Otherwise
  rows without an RTTM are fused as real speaker activity instead of being filled from the
  diarizer;
- ``model.speaker_tokens.turn_start_token`` (none while the speaker tokens are disabled) equals the
  dataset's ``turn_start_token``, both unset included, each also read under its deprecated name
  (``switch_token``, ``speaker_switch_token``). The model registers the token and the dataset writes
  it into the targets, so a token set on one side only is an error.

The validation dataset config is checked the same way. ``StreamingSTTModelAutomodel`` runs the
encoder checks in ``configure_model()``, after it mounts the encoder. The dataset requires
``<spk:0>`` and a turn-start token to be single tokens and logs how many consecutive tags are
registered.

``model.speaker_tokens`` accepts the keys ``enable``, ``template``, ``max_speakers``,
``base_token_id`` and ``turn_start_token``, plus the deprecated ``switch_token``; any other key is
an error. The turn-start token is described under "StreamingSTTDataset speaker tags" in
:doc:`datasets`.

Without a dataset config nothing is compared: ``from_pretrained`` (an HF export, as
``streaming_stt_generate.py`` loads one) and ``streaming_stt_to_hf.py`` build the model from its
model config alone. Lightning's ``load_from_checkpoint`` restores the ``data_cfg`` and
``val_data_cfg`` saved in the checkpoint, so it runs the checks on them and refuses, for example,
an older checkpoint whose ``multispeaker_cfg`` still has the retired ``no_rttm_to_ones`` key. To
skip the checks, pass both ``data_cfg=None`` and ``val_data_cfg=None``; with only one, the other
config is still checked. A checkpoint trained with online forced alignment also needs
``forced_aligner=None``, because the saved aligner is restored too and online forced alignment
requires a dataset config.

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
- SALMAutomodel: `examples/speechlm2/conf/salm_automodel.yaml`
- DuplexS2SModel: `examples/speechlm2/conf/s2s_duplex.yaml`
- DuplexS2SSpeechDecoderModel: `examples/speechlm2/conf/s2s_duplex_speech_decoder.yaml`
- DuplexSTTModel: `examples/speechlm2/conf/duplex_stt.yaml`

Using Configuration Files
-------------------------

You can use these configurations with the training scripts by specifying the config path:

.. code-block:: bash

    # Train SALM model
    python examples/speechlm2/salm_train.py \
      --config-path=conf \
      --config-name=salm

    # Train SALMAutomodel
    python examples/speechlm2/salm_train.py \
      --config-name=salm_automodel

You can also override configuration values from the command line:

.. code-block:: bash

    python examples/speechlm2/salm_train.py \
      --config-path=conf \
      --config-name=salm \
      model.pretrained_llm="different/llm/path" \
      trainer.max_steps=1000 \
      data.train_ds.batch_size=8

.. _fine-tuning-from-checkpoint:

Fine-Tuning from a Previous Checkpoint
---------------------------------------

To start a new training run initialized from a previous checkpoint — with a fresh
optimizer, LR scheduler, and step counter — set ``model.init_from_checkpoint``:

.. code-block:: yaml

    model:
      init_from_checkpoint: /path/to/checkpoints/step=6375.ckpt

Or pass it as a Hydra override:

.. code-block:: bash

    python examples/speechlm2/salm_train.py \
      --config-name=salm_automodel \
      ++model.init_from_checkpoint=/path/to/checkpoints/step=6375.ckpt

This differs from ``exp_manager.resume_from_checkpoint`` which restores the
**full** training state (optimizer, scheduler, step counter) to continue an
interrupted run. ``init_from_checkpoint`` only loads model weights, giving you a
clean starting point for fine-tuning on different data or with different
hyperparameters.

Supported Checkpoint Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three checkpoint formats are supported:

* **Distributed checkpoints (DCP)**: Directories with a ``.metadata`` file, produced
  by ``ModelParallelStrategy`` / ``AutomodelParallelStrategy``. This is the default
  format when training with FSDP2 or TP. DCP loading handles automatic resharding
  when the parallelism configuration differs between the source and target runs.

* **HuggingFace model directories**: Directories containing ``model.safetensors``,
  such as the output of ``to_hf.py``.

* **Single-file checkpoints**: Standard ``.ckpt`` or ``.pt`` files with a
  ``state_dict`` key.

The model architecture is still defined by ``pretrained_llm`` and ``pretrained_asr``
(needed for config and tokenizer initialization), but all weights are overridden by
the checkpoint.

This feature works with both ``SALM`` and ``SALMAutomodel``.

.. note::
   ``init_from_checkpoint`` requires the source and target models to use the
   same model class (e.g., both ``SALMAutomodel``). Cross-model loading
   (e.g., ``SALM`` checkpoint into ``SALMAutomodel``) will encounter state dict
   key mismatches because the two classes structure the embedding layer differently.
