Models
======

The Duplex Speech-to-Speech (S2S) collection consists of several model architectures designed to enable conversational AI systems with speech understanding and generation capabilities. These models combine audio perception components with language models and speech synthesis to create end-to-end speech-to-speech systems.

Core Model Architectures
------------------------

The collection includes the following core model architectures:


DuplexEARTTS
^^^^^^^^^^^^

DuplexEARTTS is a streaming text-to-speech model designed for duplex speech-to-speech systems. It focuses on low-latency, fully streamable speech generation by converting text tokens into audio representations in real time.

The architecture is based on the Streaming TTS model proposed in `Audio Flamingo 3 <https://arxiv.org/abs/2507.08128>`_, with several extensions for duplex interaction:

* **Gated fusion of text and audio representations**: (`GatedProjectedSumRMSNorm`), enabling better multimodal integration.
* **Subword-aware embeddings**: (`SubwordFlagEmbedding`) to improve pronunciation for words composed of multiple text tokens.
* **Custom BOS/EOS embeddings**: (`BOSEOSEmbedding`) for interruption-aware, multi-turn duplex generation.


Key components:

* **RVQVAEModel**: An RVQ-based neural audio codec that compresses speech into discrete acoustic tokens using a convolutional encoder and reconstructs high-quality audio via a convolutional decoder.
* **RVQEARTTSModel**: A streaming speech generation model that predicts multiple RVQ codebooks in parallel using a Mixture-of-Gaussians (MoG) prediction head. It produces audio tokens autoregressively from text representations with minimal latency.

DuplexEARTTS is particularly useful for:
* Duplex speech-to-speech systems requiring interruption-aware synthesis.
* Low-latency text-to-speech generation.
* Real-time conversational agents with streamed audio output.


SALM (Speech-Augmented Language Model)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SALM is a speech-augmented language model that integrates an audio perception module with a pre-trained LLM. The model is designed to understand speech input and generate text responses.
This is an implementation of the `SALM paper <https://arxiv.org/abs/2310.09424>`_.

Key components:

* **Audio Perception Module**: Processes audio inputs and converts them into embeddings that can be understood by the LLM. This module leverages a pre-trained ASR model's encoder.
* **LLM**: A pre-trained large language model that receives the audio embeddings and generates appropriate text responses.
* **Audio Locator Tag**: A special token that serves as a placeholder in the input text, which gets replaced with the audio embeddings during processing.

SALM is particularly useful for:
* Speech-to-text applications where high-quality text generation is required
* Speech understanding tasks that benefit from the contextual understanding of an LLM
* Applications that need to handle mixed text and speech inputs

DuplexS2SModel
^^^^^^^^^^^^^^

The DuplexS2SModel extends the SALM architecture to enable full speech-to-speech capabilities, adding the ability to generate speech output.

Key components:

* **Audio Perception Module**: Processes audio inputs into embeddings for the LLM.
* **LLM**: Processes audio embeddings and generates text or audio token responses.
* **Audio Codec**: Converts discrete audio tokens into speech waveforms.
* **Audio Head**: Predicts audio tokens for speech generation.
* **Embed Audio Tokens**: Embedding layers for audio token representation.

This model is particularly useful for:
* Complete conversational agents that can listen and respond with speech
* Voice assistants and interactive dialogue systems
* Applications requiring natural-sounding spoken responses

DuplexS2SSpeechDecoderModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This model focuses on the speech generation aspect of the duplex system, optimizing the decoder for high-quality speech output.

Key components:

* **Audio Perception Module**: Similar to other models, processes audio into embeddings.
* **Speech Decoder**: A specialized component for generating high-quality speech from text or audio representations.
* **TransformerARSpeechDecoder**: An autoregressive transformer-based decoder for speech generation.

This model is particularly useful for:
* Applications focusing on high-quality speech synthesis
* Systems where speech generation quality is the primary concern
* Specialized voice output applications

DuplexSTTModel
^^^^^^^^^^^^^^

This model focuses on speech-to-text conversion in duplex conversations, processing both user speech and generating agent text responses.

Key components:

* **Audio Perception Module**: Processes audio inputs into embeddings for the LLM.
* **LLM**: Processes audio embeddings and generates text responses.
* **Tokenizer**: Converts LLM outputs into readable text.

This model is particularly useful for:
* Speech-to-text applications in conversational settings
* Duplex systems where text responses are needed instead of speech
* Applications requiring transcript generation from spoken dialogue

Model Components
----------------

Audio Perception Module
^^^^^^^^^^^^^^^^^^^^^^^

The audio perception module is responsible for converting speech signals into embeddings that can be processed by language models. It typically consists of:

1. **Preprocessor**: Converts raw audio waveforms into spectral features
2. **Encoder**: Processes these features to create meaningful representations
3. **Modality Adapter**: Adapts the encoder outputs to be compatible with the LLM's input space

Speech Generation
^^^^^^^^^^^^^^^^^

Speech generation components convert text or token representations back into speech. The collection offers:

1. **TransformerARSpeechDecoder**: An autoregressive transformer-based speech decoder
2. **Audio Codec Integration**: Works with audio codecs to generate natural speech from discrete tokens
3. **DuplexEARTTS**: A ready-to-use duplex text-to-speech model that supports user interruption via a special text interruption token. The model integrates an RVQ-based audio codec with a streaming speech generation module to enable low-latency, real-time synthesis.

Implementation Details
----------------------

The DuplexS2SModel implementation contains several key methods that handle different aspects of the model's functionality:

Model Initialization
^^^^^^^^^^^^^^^^^^^^

The constructor (`__init__`) initializes the following components:

1. **Pretrained ASR encoder/perception**: Loads a pretrained NeMo ASR model and adapts it for audio perception
2. **LLM**: Loads a pretrained language model using HuggingFace's AutoModel
3. **Audio codec**: Loads a pretrained NeMo audio codec for speech generation
4. **Token prediction heads**: Adds separate heads for text token and audio token prediction

Forward Method
^^^^^^^^^^^^^^

The `forward` method:

1. Accepts input representations (sum of audio perception and text embedding outputs)
2. Runs an offline forward pass through the language model
3. Generates logits from both text and audio token prediction heads
4. Returns these logits for loss computation

Training Step
^^^^^^^^^^^^^

The `training_step` method:

1. Builds input representations using the `prepare_inputs` method
2. Runs the `forward` method to get text and audio logits
3. Computes losses for both text and audio tokens
4. Logs training metrics (loss, learning rate, etc.)
5. Returns the loss for backpropagation

Prepare Inputs
^^^^^^^^^^^^^^

The `prepare_inputs` method:

1. Processes source audio through the perception module (with gradients enabled)
2. Processes target audio through the audio codec (with gradients disabled)
3. Truncates source/target audio and target text sequences to have the same sequence length, if needed
4. Performs additional truncation for sequence lengths to be divisible by tensor parallelism world size (if enabled)
5. Returns a dictionary with input and label tensors for training

.. code-block:: python

    def prepare_inputs(self, batch):
        # Process source audio through perception
        audio_embs, audio_emb_lens = self.perception(
            input_signal=batch["source_audio"], 
            input_signal_length=batch["source_audio_lens"]
        )
        
        # Process target audio through codec (no gradients)
        with torch.no_grad():
            target_audio_tokens, target_audio_token_lens = self.audio_codec.encode(batch["target_audio"], batch["target_audio_lens"])
        
        # Truncate sequences if needed
        # ... (truncation logic)

        # Embed text tokens and combine them with audio representations
        # ... (text embedding logic)

        # Return processed inputs and labels
        return {
            "audio_embeds": audio_embs,
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "target_text_ids": target_text_ids,
            "target_audio_ids": target_audio_ids,
        }

Validation
^^^^^^^^^^

The validation process:

1. Clears GPU memory to avoid OOM issues
2. Loads a scoring ASR model into GPU for evaluation
3. Initializes metric aggregation for each dataset
4. Processes each validation dataset separately
5. Computes BLEU scores for text and ASR-decoded audio
6. Logs and clears metrics after validation is complete

Scaling Support
---------------

The DuplexS2SModel includes a `configure_model` method that sets up model parallelism for large-scale training. This method:

1. Detects the parallelism strategy from the trainer's device mesh
2. Applies Fully Sharded Data Parallel (FSDP) sharding to appropriate modules
3. Applies Tensor Parallelism (TP) and Sequence Parallelism (SP) when configured
4. Handles model-specific adaptations for different LLM architectures

The scaling approach supports:

* Pure FSDP2 for distributing parameters across GPUs
* Pure TP/SP for splitting computation across GPUs
* 2D parallelism combining both approaches

Pretrained Model Usage
----------------------

All models in the speechlm2 collection can be instantiated from pretrained checkpoints:

.. code-block:: python

    import nemo.collections.speechlm2 as slm
    
    # Load SALM model
    salm_model = slm.models.SALM.from_pretrained("path/to/checkpoint")
    
    # Load DuplexS2SModel
    duplex_model = slm.models.DuplexS2SModel.from_pretrained("path/to/checkpoint")

    # Load DuplexS2SSpeechDecoderModel
    speech_decoder_model = slm.models.DuplexS2SSpeechDecoderModel.from_pretrained("path/to/checkpoint")

    # Load DuplexSTTModel
    stt_model = slm.models.DuplexSTTModel.from_pretrained("path/to/checkpoint")

    # Load DuplexEARTTS
    ear_tts_model = slm.models.DuplexEARTTS.from_pretrained("path/to/checkpoint")

Model Configuration
-------------------

All models in this collection use a configuration-based approach, where a YAML configuration file specifies the model architecture, components, and training parameters. See the :doc:`configurations documentation <configs>` for details on these configuration files.

For information about scaling and training these models at scale, see the :doc:`training and scaling documentation <training_and_scaling>`. 