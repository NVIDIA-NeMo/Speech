NeMo Speaker Diarization Configuration Files
============================================

Both training and inference of speaker diarization is configured by ``.yaml`` files. The diarizer section will generally require information about the dataset(s) being used, models used in this pipeline, as well as inference related parameters such as post processing of each models. The sections on this page cover each of these in more detail.

.. note::
  For model details and deep understanding about configs, training, fine-tuning and evaluations,
  please refer to ``<NeMo_git_root>/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb`` and ``<NeMo_git_root>/tutorials/speaker_tasks/Speaker_Diarization_Training.ipynb``;
  for other applications such as possible integration with ASR, have a look at ``<NeMo_git_root>/tutorials/speaker_tasks/ASR_with_SpeakerDiarization.ipynb``.


Hydra Configurations for Diarization Training 
=============================================

Currently, NeMo supports Multi-scale diarization decoder (MSDD) as a neural diarizer model. MSDD is a speaker diarization model based on initializing clustering and multi-scale segmentation input. Example configuration files for MSDD model training can be found in ``<NeMo_git_root>/examples/speaker_tasks/diarization/conf/neural_diarizer/``.

* Model name convention for MSDD: msdd_<number of scales>scl_<longest scale in decimal second (ds)>_<shortest scale in decimal second (ds)>_<overlap percentage of window shifting>Povl_<hidden layer size>x<number of LSTM layers>x<number of CNN output channels>x<repetition count of conv layer>
* Example: ``msdd_5scl_15_05_50Povl_256x3x32x2.yaml`` has 5 scales, the longest scale is 1.5 sec, the shortest scale is 0.5 sec, with 50 percent overlap, hidden layer size is 256, 3 LSTM layers, 32 CNN channels, 2 repeated Conv layers

MSDD model checkpoint (.ckpt) and NeMo file (.nemo) contain speaker embedding model (TitaNet) and the speaker model is loaded along with standalone MSDD module. Note that MSDD models require more than one scale. Thus, the parameters in ``diarizer.speaker_embeddings.parameters`` should have more than one scale to function as a MSDD model.


General Diarizer Configuration
------------------------------

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
---------------------

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
---------------------------

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
---------------------------------

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
-------------------

Neural diarizer uses a binary cross entropy (BCE) loss. A set of weights for negative (absence of the speaker's speech) and positive (presence of the speaker's speech) can be provided to the loss function.

.. code-block:: yaml

  loss: 
    _target_: nemo.collections.asr.losses.bce_loss.BCELoss
    weight: null # Weight for binary cross-entropy loss. Either `null` or list type input. (e.g. [0.5,0.5])


Hydra Configurations for Diarization Inference
==============================================

Example configuration files for speaker diarization inference can be found in ``<NeMo_git_root>/examples/speaker_tasks/diarization/conf/inference/``. Choose a yaml file that fits your targeted domain. For example, if you want to diarize audio recordings of telephonic speech, choose ``diar_infer_telephonic.yaml``.

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
------------------------------------------

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
      min_duration_on: 0.2 # Threshold for small non_speech deletion
      min_duration_off: 0.2 # Threshold for short speech segment deletion
      filter_speech_first: True 

Configurations for Speaker Embedding in Diarization
---------------------------------------------------

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
--------------------------------------------

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
---------------------------------------

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
