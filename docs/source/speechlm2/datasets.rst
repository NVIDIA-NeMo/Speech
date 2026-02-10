Datasets
========

The speechlm2 collection supports datasets that contain both audio and text data for training models that can understand speech and generate appropriate responses.
This section describes the dataset format, preparation, and usage with the speechlm2 models.

Dataset Format
--------------

Duplex S2S models use the Lhotse framework for audio data management. The primary datasets used are:

1. **DuplexS2SDataset**: For general duplex speech-to-speech models
2. **SALMDataset**: Specifically for the Speech-Augmented Language Model (SALM), which processes speech+text and outputs text.
3. **DuplexEARTTSDataset**: Dataset for Duplex EARTTS model, extending DuplexS2SDataset with additional output fields for TTS, including audio prompting. It optionally prepends an audio prompt (speaker reference) to target_audio, which is used to initialize speaker conditioning in the EARTTS model. The dataset provides audio_prompt, audio_prompt_lens, non_prompt_mask, aligned_attention_mask, and aligned_position_ids, and supports custom speaker reference audio through the context_audio field, while preserving full compatibility with the original data format.

DuplexS2S Dataset Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A typical dataset for speechlm2 models consists of:

1. **Audio files**: Contains source audio (input speech) and possibly target audio (output speech)
2. **Text transcriptions**: Associated text for both input and output speech
3. **Role identifiers**: To distinguish between speakers (e.g., "user" vs "agent")

The dataset organization is built around the concept of conversation turns, with each turn containing audio and text from either a user or an agent/assistant.

The datasets are primarily managed using Lhotse's CutSet format, which provides efficient handling of audio data and annotations. A typical Lhotse manifest includes:

- Audio recording information (path, duration, sample rate)
- Supervision information (transcripts, speaker roles, timing)
- Optional additional annotations

Example of a Lhotse cut:

.. code-block:: python

    {
        "id": "conversation_1",
        "start": 0,
        "duration": 10.7,
        "channel": 0,
        "supervisions": [
            {
                "id": "conversation_1_turn_0",
                "text": "Can you help me with this problem?",
                "start": 0,
                "duration": 5.2,
                "speaker": "user"
            },
            {
                "id": "conversation_1_turn_1",
                "text": "I can help you with that.",
                "start": 5.2,
                "duration": 3.1,
                "speaker": "assistant"
            }
        ],
        "recording": {
            "id": "conversation_1_user",
            "path": "/path/to/audio/conversation_1_user.wav",
            "sampling_rate": 16000,
            "num_samples": 171200,
            "duration": 10.7
        },
        "custom": {
            "target_audio": {
                "id": "conversation_1_assistant",
                "path": "/path/to/audio/conversation_1_assistant.wav",
                "sampling_rate": 22050,
                "num_samples": 235935,
                "duration": 10.7
            }
        }
    }

The DuplexS2SDataset performs several key operations when processing data:

1. **Turn Identification**: Each cut contains a list of `supervisions` with objects of type `lhotse.SupervisionSegment` that represent conversation turns with corresponding text and speaker information.

2. **Speaker Role Separation**: The text of each supervision is tokenized and identified as the model's output (when `supervision.speaker` is in `output_roles`, e.g., "agent" or "Assistant") or the model's input (when in `input_roles`, e.g., "user" or "User").

3. **Token Sequence Generation**:
   - `target_tokens` and `source_tokens` arrays are created with a length equal to `lhotse.utils.compute_num_frames(cut.duration, frame_length, cut.sampling_rate)`
   - The `frame_length` parameter (typically 80ms) determines the temporal resolution of token assignments
   - Each token is assigned to a position based on its corresponding audio segment's timing

4. **Token Offset Calculation**:
   - The starting position for each turn's tokens is determined using `lhotse.utils.compute_num_frames(supervision.start, frame_length)`
   - This ensures tokens are aligned with their corresponding audio segments

5. **Length Validation**:
   - If token sequences are too long compared to the audio duration, warnings are emitted
   - Tokens that extend beyond the audio length are truncated

This process ensures that the model can correctly align audio input with corresponding text, and learn to generate appropriate responses based on the conversation context.

DuplexS2SDataset
****************

This dataset class is designed for models that handle both speech understanding and speech generation. It processes audio inputs and prepares them for the model along with corresponding text.

.. code-block:: python

    from nemo.collections.speechlm2.data import DuplexS2SDataset
    
    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,                   # Text tokenizer
        frame_length=0.08,                          # Frame length in seconds
        source_sample_rate=16000,                   # Input audio sample rate
        target_sample_rate=22050,                   # Output audio sample rate
        input_roles=["user", "User"],               # Roles considered as input
        output_roles=["agent", "Assistant"]         # Roles considered as output
    )

SALMDataset Structure
^^^^^^^^^^^^^^^^^^^^^

Data used for SALM can be either regular speech-to-text data (in any NeMo or Lhotse format), or a dataset of multi-turn conversions.
For the most part, please refer to :doc:`the ASR datasets documentation <../asr/datasets>` for details on data formats and multimodal dataloading.

When using speech-to-text data, you'll need read it with a special ``lhotse_as_conversation`` data reader
that creates a two-turn, query+response, multi-modal conversation data types out of regular Lhotse cuts.
This approach makes SALM training more flexible, allowing straightforward combination of single-turn and multi-turn data.

Each audio turn is represented as a single token, defined in ``audio_locator_tag`` property, and automatically added to the model's tokenizer inside model code.
This token is replaced during the training/generation pass with its corresponding audio segment representation.

Example YAML configuration using existing ASR datasets with ``lhotse_as_conversation``:

.. code-block:: yaml

    data:
      train_ds:
        prompt_format: "llama3"  # Choose based on your model
        token_equivalent_duration: 0.08
        input_cfg:
          # Example 1: Using standard ASR Lhotse manifests (JSONL)
          - type: lhotse_as_conversation
            cuts_path: /path/to/librispeech_train_clean_100.jsonl.gz
            audio_locator_tag: "<|audioplaceholder|>"
            tags:
              context: "Transcribe the following audio:"
              # Optional system prompt can be uncommented
              # system_prompt: "You are a helpful assistant that transcribes audio accurately."
          
          # Example 2: Using tarred NeMo manifests
          - type: lhotse_as_conversation
            manifest_filepath: /path/to/tedlium_train_manifest.jsonl.gz
            tarred_audio_filepaths: /path/to/tedlium_shards/shard-{000000..000009}.tar
            audio_locator_tag: "<|audioplaceholder|>"
            tags:
              context: "Write down what is said in this recording:"
              
          # Example 3: Using Lhotse SHAR format
          - type: lhotse_as_conversation
            shar_path: /path/to/fisher_shar/
            audio_locator_tag: "<|audioplaceholder|>"
            tags:
              context: "Listen to this clip and write a transcript:"
    
      # ... other settings

Alternatively, one can provide an existing YAML file with their dataset composition and wrap 
it in a ``lhotse_as_conversation`` reader as follows:

.. code-block:: yaml

    data:
      train_ds:
        input_cfg:
          - type: lhotse_as_conversation
            input_cfg: /path/to/dataset_config.yaml
            audio_locator_tag: "<|audioplaceholder|>"
            tags:
              context: "Transcribe the following audio:"
              # Optional system prompt can be uncommented
              # system_prompt: "You are a helpful assistant that transcribes audio accurately."


The ``lhotse_as_conversation`` reader automatically creates a two-turn conversation from each ASR example:
1. Optionally, if ``system_prompt`` tag is provided, it's added as a special system turn for LLM models that support system prompts.
2. A user turn containing the audio and a text context (from the ``context`` tag)
3. An assistant turn containing the transcription (from the cut's supervision text)

If a ``context`` tag is provided in the configuration, it's added as a text turn before the audio.

SALMDataset
***********

This dataset class is specialized for the SALM model, which focuses on understanding speech input and generating text output.

.. code-block:: python

    from nemo.collections.speechlm2.data import SALMDataset
    
    dataset = SALMDataset(
        tokenizer=model.tokenizer,                   # Text tokenizer
    )

DataModule
----------

The DataModule class in the speechlm2 collection manages dataset loading, preparation, and batching for PyTorch Lightning training:

.. code-block:: python

    from nemo.collections.speechlm2.data import DataModule
    
    datamodule = DataModule(
        cfg_data,                  # Configuration dictionary for data
        tokenizer=model.tokenizer, # Text tokenizer
        dataset=dataset            # Instance of DuplexS2SDataset or SALMDataset
    )

The DataModule takes care of:
1. Setting up proper data parallel ranks for dataloaders
2. Instantiating the dataloaders with configuration from YAML
3. Managing multiple datasets for validation/testing

Bucketing for Efficient Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DataModule supports bucketing for more efficient training. Bucketing groups samples of similar lengths together, which reduces padding and improves training efficiency. The key bucketing parameters are:

1. **batch_duration**: Target cumulative duration (in seconds) of samples in a batch
2. **bucket_duration_bins**: List of duration thresholds for bucketing
3. **use_bucketing**: Flag to enable/disable bucketing
4. **num_buckets**: Number of buckets to create
5. **bucket_buffer_size**: Number of samples to load into memory for bucket assignment

Example bucketing configuration:

.. code-block:: yaml

    train_ds:
      # ... other settings
      batch_duration: 100  # Target 100 seconds per batch
      bucket_duration_bins: [8.94766, 10.1551, 11.64118, 19.30376, 42.85]  # Duration thresholds
      use_bucketing: true  # Enable bucketing
      num_buckets: 5  # Create 5 buckets
      bucket_buffer_size: 5000  # Buffer size for bucket assignment

When bucketing is enabled:

1. Samples are grouped into buckets based on their duration
2. Each batch contains samples from the same bucket
3. The actual batch size can vary to maintain a consistent total duration
4. The target batch_duration ensures efficient GPU memory usage

Bucketing helps to:
- Reduce padding and increase effective batch size
- Improve training efficiency and convergence
- Manage memory usage with variable-length inputs

Data Configuration
------------------

A typical data configuration in YAML includes:

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
        # Optional bucketing settings
        batch_duration: 100
        bucket_duration_bins: [8.94766, 10.1551, 11.64118, 19.30376, 42.85]
        use_bucketing: true
        num_buckets: 5
        bucket_buffer_size: 5000
        # batch_size: 4  # alternative to bucketing
    
      validation_ds:
        datasets:
          val_set_name_0:
            shar_path: /path/to/validation_data_0
          val_set_name_1:
            shar_path: /path/to/validation_data_1
        sample_rate: ${data.target_sample_rate}
        batch_size: 4
        seed: 42
        shard_seed: "randomized"

Note that the actual dataset paths and blend are defined by the YAML config, not Python code. This makes it easy to change the dataset composition without modifying the code.
To learn more about the YAML data config, see :ref:`the Extended multi-dataset configuration format <asr-dataset-config-format>` section in the ASR documentation.

Preparing S2S Datasets
----------------------

Creating Lhotse Manifests
^^^^^^^^^^^^^^^^^^^^^^^^^

To prepare your own dataset, you'll need to create Lhotse manifests from your audio files and transcripts:

.. code-block:: python

    from lhotse import CutSet, Recording, SupervisionSegment
    
    # Create a recording for user and assistant
    recording_user = Recording(
        id="conversation_1_user",
        path="/path/to/audio/conversation_1_user.wav",
        sampling_rate=16000,
        num_samples=171200,
        duration=10.7
    )
    recording_assistant = Recording(
        id="conversation_1_assistant",
        path="/path/to/audio/conversation_1_assistant.wav",
        sampling_rate=22050,
        num_samples=235935,
        duration=10.7
    )
    
    # Create supervision for this recording
    supervisions = [
        SupervisionSegment(
            id="conversation_1_turn_0",
            recording_id="conversation_1",
            start=0,
            duration=5.2,
            text="Can you help me with this problem?",
            speaker="user"
        ),
        SupervisionSegment(
            id="conversation_1_turn_1",
            recording_id="conversation_1",
            start=5.5,
            duration=3.1,
            text="I can help you with that.",
            speaker="assistant"
        ),
    ]
    
    # Create a CutSet
    # The assistant's response is located in target_audio field which makes it easy to replace
    # when using multiple models or speakers for synthetic data generation.
    cut = recording.to_cut()
    cut.supervisions = supervisions
    cut.target_audio = recording_assistant
    cutset = CutSet([cut])
    
    # Save to disk
    cutset.to_file("path/to/manifest.jsonl.gz")

Converting to SHAR Format
^^^^^^^^^^^^^^^^^^^^^^^^^

For efficient training, it's recommended to convert your Lhotse manifests to SHAR (SHarded ARchive) format:

.. code-block:: python

    from lhotse import CutSet
    from lhotse.shar import SharWriter
    
    cutset = CutSet.from_file("path/to/manifest.jsonl.gz")
    cutset.to_shar("path/to/train_shar", fields={"recording": "flac", "target_audio": "flac"}, shard_size=100)
    

Training with Prepared Datasets
-------------------------------

Once your datasets are prepared, you can use them to train a model:

.. code-block:: python

    # Load configuration
    config_path = "path/to/config.yaml"
    cfg = OmegaConf.load(config_path)
    
    # The training data paths are available in the config file:
    # cfg.data.train_ds.input_cfg[0].shar_path = "path/to/train_shar"
    
    # Create dataset and datamodule
    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    
    # Train the model
    trainer.fit(model, datamodule)

Example S2S Datasets
--------------------

While there are no publicly available datasets specifically formatted for Duplex S2S models yet, you can adapt conversation datasets with audio recordings such as:

1. Fisher Corpus
2. Switchboard Corpus
3. CallHome
4. Synthetic conversation datasets generated using TTS

You would need to format these datasets as Lhotse manifests with appropriate speaker role annotations to use them with the speechlm2 S2S models. 