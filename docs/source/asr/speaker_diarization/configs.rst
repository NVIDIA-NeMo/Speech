Speaker Diarization Configuration Files
========================================

.. note::

   For the full configuration files, see the YAML configs on GitHub:

   - `sortformer_diarizer_hybrid_loss_4spk-v1.yaml <https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speaker_tasks/diarization/conf/neural_diarizer/sortformer_diarizer_hybrid_loss_4spk-v1.yaml>`__
   - `streaming_sortformer_diarizer_4spk-v2.yaml <https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speaker_tasks/diarization/conf/neural_diarizer/streaming_sortformer_diarizer_4spk-v2.yaml>`__

Hydra Configurations for Sortformer Diarizer Training
-----------------------------------------------------

Sortformer Diarizer is an end-to-end speaker diarization model that is solely based on Transformer-encoder type of architecture.
Model name convention for Sortformer Diarizer: ``sortformer_diarizer_<loss_type>_<speaker count limit>-<version>.yaml``

* Example: ``<NeMo_root>/examples/speaker_tasks/diarization/neural_diarizer/conf/sortformer_diarizer_hybrid_loss_4spk-v1.yaml``

Key parameters:

.. code-block:: yaml

  name: "SortformerDiarizer"
  batch_size: 8

  model:
    sample_rate: 16000
    pil_weight: 0.5       # Weight for Permutation Invariant Loss (PIL)
    ats_weight: 0.5       # Weight for Arrival Time Sort (ATS) loss
    max_num_of_spks: 4    # Maximum number of speakers per model

    model_defaults:
      fc_d_model: 512     # Hidden dimension size of the Fast-Conformer Encoder
      tf_d_model: 192     # Hidden dimension size of the Transformer Encoder

    train_ds:
      manifest_filepath: ???
      session_len_sec: 90
      # ...

    preprocessor:
      _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
      normalize: "per_feature"
      window_stride: 0.01
      features: 80

    encoder:
      n_layers: 18
      d_model: ${model.model_defaults.fc_d_model}
      subsampling: dw_striding
      subsampling_factor: 8

    transformer_encoder:
      num_layers: 18
      hidden_size: ${model.model_defaults.tf_d_model}
      num_attention_heads: 8

Hydra Configurations for Streaming Sortformer Diarizer Training
----------------------------------------------------------------

Model name convention for Streaming Sortformer Diarizer: ``streaming_sortformer_diarizer_<speaker count limit>-<version>.yaml``

* Example: ``<NeMo_root>/examples/speaker_tasks/diarization/neural_diarizer/conf/streaming_sortformer_diarizer_4spk-v2.yaml``

The Streaming Sortformer config extends the offline config with ``streaming_mode: True`` and additional speaker cache parameters:

.. code-block:: yaml

  name: "StreamingSortformerDiarizer"
  batch_size: 4

  model:
    sample_rate: 16000
    pil_weight: 0.5
    ats_weight: 0.5
    max_num_of_spks: 4
    streaming_mode: True

    model_defaults:
      fc_d_model: 512
      tf_d_model: 192

    preprocessor:
      _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
      normalize: "NA"  # Required for streaming (no per-feature normalization)
      window_stride: 0.01
      features: 128

    sortformer_modules:
      num_spks: ${model.max_num_of_spks}
      # Streaming-specific parameters
      spkcache_len: 188          # Length of speaker cache buffer (frames for all speakers)
      chunk_len: 188             # Number of frames processed per streaming chunk
       chunk_left_context: 1
      chunk_right_context: 1
      # ...

    encoder:
      n_layers: 17
      d_model: ${model.model_defaults.fc_d_model}
      subsampling: dw_striding
      subsampling_factor: 8

    transformer_encoder:
      num_layers: 18
      hidden_size: ${model.model_defaults.tf_d_model}
      num_attention_heads: 8

See the full YAML configs on GitHub: `Sortformer <https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speaker_tasks/diarization/conf/neural_diarizer/sortformer_diarizer_hybrid_loss_4spk-v1.yaml>`__ · `Streaming Sortformer <https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speaker_tasks/diarization/conf/neural_diarizer/streaming_sortformer_diarizer_4spk-v2.yaml>`__


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
