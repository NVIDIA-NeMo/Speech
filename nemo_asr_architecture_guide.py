"""
================================================================================
NEMO ASR (AUTOMATIC SPEECH RECOGNITION) ARCHITECTURE GUIDE
================================================================================

This comprehensive guide documents the complete ASR architecture in NVIDIA NeMo,
including all mechanisms, functions, classes, and their relationships.

Author: Generated Architecture Documentation
Date: 2025-11-22
Repository: NVIDIA NeMo
Focus: nemo/collections/asr/

================================================================================
TABLE OF CONTENTS
================================================================================

1. DIRECTORY STRUCTURE
2. MODEL ARCHITECTURES
   2.1 Base Classes
   2.2 CTC Models
   2.3 RNN-T (Transducer) Models
   2.4 Hybrid Models
   2.5 Other Architectures
3. CORE COMPONENTS
   3.1 Encoders (Conformer, Squeezeformer, Convolutional, RNN)
   3.2 Decoders (RNN-T Prediction Networks, CTC, Joint Networks)
   3.3 Preprocessors (Mel Spectrogram, MFCC)
   3.4 Tokenizers (BPE, Character-level)
4. INFERENCE PIPELINE
   4.1 CTC Decoding (Greedy, Beam Search)
   4.2 RNN-T Decoding (Greedy, TSD, ALSD, MAES)
   4.3 Context Biasing
   4.4 Confidence Estimation
5. TRAINING PIPELINE
   5.1 Loss Functions (CTC, RNN-T, Hybrid)
   5.2 Forward Pass
   5.3 Optimization
6. DATA PROCESSING
   6.1 Dataset Classes
   6.2 Audio Augmentation
   6.3 Spectrogram Augmentation
7. ADVANCED FEATURES
   7.1 Streaming & Cache-Aware Models
   7.2 Multi-blank & TDT
   7.3 Adapter Modules
   7.4 N-gram Language Models
8. CONFIGURATION MANAGEMENT
9. DETAILED CLASS & FUNCTION REFERENCE

================================================================================
"""

# ============================================================================
# 1. DIRECTORY STRUCTURE
# ============================================================================

DIRECTORY_STRUCTURE = """
/home/user/NeMo/nemo/collections/asr/
├── data/                      # Data loading and processing
│   ├── audio_to_text.py           - AudioToCharDataset, AudioToBPEDataset
│   ├── audio_to_text_lhotse.py    - Lhotse integration for modern data loading
│   ├── audio_to_text_dali.py      - NVIDIA DALI acceleration
│   └── ...
│
├── inference/                 # Inference pipelines and utilities
│   ├── pipelines/
│   │   ├── buffered_ctc_pipeline.py
│   │   ├── buffered_rnnt_pipeline.py
│   │   ├── cache_aware_ctc_pipeline.py
│   │   └── cache_aware_rnnt_pipeline.py
│   ├── factory/
│   │   └── pipeline_builder.py
│   └── utils/
│
├── losses/                    # Loss functions
│   ├── ctc.py                     - CTCLoss wrapper
│   └── rnnt.py                    - Multi-backend RNN-T loss
│
├── metrics/                   # Evaluation metrics
│   └── wer.py                     - WER, CER computation
│
├── models/                    # Main model implementations
│   ├── asr_model.py               - ASRModel base class
│   ├── ctc_models.py              - EncDecCTCModel
│   ├── ctc_bpe_models.py          - EncDecCTCModelBPE
│   ├── rnnt_models.py             - EncDecRNNTModel
│   ├── rnnt_bpe_models.py         - EncDecRNNTBPEModel
│   ├── hybrid_rnnt_ctc_models.py  - Multi-task learning
│   └── ...
│
├── modules/                   # Core neural network modules
│   ├── conformer_encoder.py       - ConformerEncoder (SOTA)
│   ├── squeezeformer_encoder.py   - Optimized Conformer variant
│   ├── conv_asr.py                - ConvASREncoder (Jasper/QuartzNet)
│   ├── rnn_encoder.py             - LSTM/GRU encoder
│   ├── rnnt.py                    - RNNTDecoder, RNNTJoint
│   ├── rnnt_abstract.py           - Abstract interfaces
│   ├── audio_preprocessing.py     - Mel spectrogram extraction
│   └── ...
│
└── parts/                     # Supporting components
    ├── context_biasing/           - Context-aware decoding
    │   ├── context_graph_ctc.py
    │   └── context_graph_universal.py
    │
    ├── mixins/                    - Reusable model mixins
    │   ├── mixins.py              - ASRModuleMixin, ASRBPEMixin
    │   ├── transcription.py       - ASRTranscriptionMixin
    │   ├── interctc_mixin.py      - InterCTCMixin
    │   └── streaming.py           - StreamingEncoder
    │
    ├── numba/                     - Optimized CUDA/Numba implementations
    │   ├── rnnt_loss/             - Fast RNN-T loss
    │   └── spec_augment/          - Fast SpecAugment
    │
    ├── preprocessing/             - Audio preprocessing
    │   ├── features.py            - Feature extraction
    │   ├── perturb.py             - Audio augmentation
    │   └── segment.py             - AudioSegment
    │
    ├── submodules/                - Decoding and other submodules
    │   ├── ctc_decoding.py        - CTCDecoding orchestrator
    │   ├── ctc_greedy_decoding.py - CTCGreedyDecoder
    │   ├── ctc_beam_decoding.py   - CTCBeamDecoder (KenLM)
    │   ├── rnnt_decoding.py       - RNNTDecoding orchestrator
    │   ├── rnnt_greedy_decoding.py - Greedy RNN-T inference
    │   ├── rnnt_beam_decoding.py  - Beam search variants
    │   ├── ngram_lm.py            - N-gram LM support
    │   ├── spectr_augment.py      - SpecAugment, SpecCutout
    │   └── adapters/              - Parameter-efficient fine-tuning
    │
    └── utils/                     - Utility functions
        ├── manifest_utils.py
        └── parsers.py
"""

# ============================================================================
# 2. MODEL ARCHITECTURES
# ============================================================================

class ASR_ARCHITECTURE_OVERVIEW:
    """
    ========================================================================
    2.1 BASE CLASSES
    ========================================================================

    ASRModel (nemo/collections/asr/models/asr_model.py)
    ---------------------------------------------------
    Base class for all ASR models.

    Inheritance Chain:
        ModelPT <- ASRModel <- Specific ASR Models

    Key Responsibilities:
        - Validation/test epoch management
        - WER/BLEU metric computation
        - Auxiliary loss management (adapter losses)
        - CUDA graphs support for inference speedup
        - Optimization flags (skip_nan_grad)

    Key Methods:
        - validation_step(batch, batch_idx, dataloader_idx=0)
        - test_step(batch, batch_idx, dataloader_idx=0)
        - multi_validation_epoch_end(outputs, dataloader_idx=0)
        - multi_test_epoch_end(outputs, dataloader_idx=0)

    Mixins:
        - WithOptionalCudaGraphs: CUDA graph optimization
        - ABC: Abstract base class

    ========================================================================

    ExportableEncDecModel (nemo/collections/asr/models/asr_model.py)
    ---------------------------------------------------------------
    Mixin for encoder-decoder export capabilities.

    Features:
        - ONNX export support
        - Streaming cache management for export
        - Input/output names for deployment

    Key Methods:
        - list_export_subnets() -> List[str]
        - get_export_subnet_name() -> str

    ========================================================================
    2.2 CTC MODELS
    ========================================================================

    EncDecCTCModel (nemo/collections/asr/models/ctc_models.py)
    ----------------------------------------------------------
    Primary CTC (Connectionist Temporal Classification) model.

    Inheritance:
        ASRModel, ExportableEncDecModel, ASRModuleMixin,
        InterCTCMixin, ASRTranscriptionMixin

    Architecture Components:
        1. preprocessor: Audio -> Mel Spectrogram
           - AudioToMelSpectrogramPreprocessor
           - Converts raw audio to mel features

        2. encoder: Mel Features -> Acoustic Embeddings
           - ConformerEncoder (most common)
           - Jasper, QuartzNet, ContextNet, etc.
           - Outputs: (batch, time, features)

        3. decoder: Acoustic Embeddings -> Logits
           - Simple linear projection
           - Projects encoder output to vocabulary size
           - Outputs: (batch, time, vocab_size)

        4. loss: CTCLoss
           - Computes CTC loss for training
           - Handles alignment internally

        5. spec_augmentation: SpecAugment (optional)
           - Time and frequency masking
           - Applied during training only

        6. decoding: CTCDecoding
           - Greedy or beam search decoding
           - For inference/evaluation

    Forward Pass (Training):
        audio_signal (B, T)
        -> preprocessor -> (B, T', F)
        -> spec_augment -> (B, T', F)
        -> encoder -> (B, T'', H)
        -> decoder -> (B, T'', V)
        -> log_softmax -> (B, T'', V)
        -> CTCLoss

    Forward Pass (Inference):
        audio_signal -> preprocessor -> encoder -> decoder
        -> log_softmax -> decoding.decode() -> text

    Key Methods:
        - forward(input_signal, input_signal_length, processed_signal,
                  processed_signal_length)
        - training_step(batch, batch_idx)
        - validation_step(batch, batch_idx)
        - transcribe(audio, batch_size=4) -> List[str]

    Variants:
        - EncDecCTCModel: Character-level
        - EncDecCTCModelBPE: Subword (BPE) tokenization
          (nemo/collections/asr/models/ctc_bpe_models.py)

    ========================================================================
    2.3 RNN-T (TRANSDUCER) MODELS
    ========================================================================

    EncDecRNNTModel (nemo/collections/asr/models/rnnt_models.py)
    ------------------------------------------------------------
    Primary RNN-T (Recurrent Neural Network Transducer) model.

    Inheritance:
        ASRModel, ASRModuleMixin, ExportableEncDecModel,
        ASRTranscriptionMixin

    Architecture Components:
        1. preprocessor: Audio -> Features
           - AudioToMelSpectrogramPreprocessor

        2. encoder: Features -> Acoustic Embeddings
           - ConformerEncoder, Squeezeformer, etc.
           - Outputs: f(x) with shape (B, T, H_enc)

        3. decoder: Previous Tokens -> Prediction Embeddings
           - RNNTDecoder (LSTM-based, stateful)
           - StatelessTransducerDecoder (context window)
           - Outputs: g(y) with shape (B, U, H_dec)

        4. joint: Combines encoder and decoder
           - RNNTJoint network
           - Computes: h(f(x), g(y)) = joint(encoder, decoder)
           - Outputs: (B, T, U, V) where V = vocab_size + 1 (blank)

        5. loss: RNNTLoss
           - Multiple backends: warprnnt_numba, warprnnt, pytorch
           - Computes forward-backward algorithm

        6. decoding: RNNTDecoding
           - Greedy, beam search (TSD, ALSD, MAES)
           - For inference/evaluation

    Forward Pass (Training):
        audio_signal (B, T)
        -> preprocessor -> (B, T', F)
        -> encoder -> f(x) (B, T', H_enc)

        targets (B, U)
        -> decoder -> g(y) (B, U, H_dec)

        f(x), g(y) -> joint -> (B, T', U, V)
        -> RNNTLoss

    Forward Pass (Inference):
        For each timestep t:
            1. Get encoder output f(x_t)
            2. Initialize decoder state
            3. Loop until blank:
                a. Get decoder output g(y_i)
                b. Compute joint(f(x_t), g(y_i))
                c. Get next token
                d. If blank: move to next timestep
                e. Else: emit token, update decoder state

    Key Methods:
        - forward(input_signal, input_signal_length, transcript,
                  transcript_length)
        - training_step(batch, batch_idx)
        - validation_step(batch, batch_idx)
        - transcribe(audio, batch_size=4) -> List[str]

    Variants:
        - EncDecRNNTModel: Character-level
        - EncDecRNNTBPEModel: Subword tokenization
          (nemo/collections/asr/models/rnnt_bpe_models.py)

    ========================================================================
    2.4 HYBRID MODELS
    ========================================================================

    EncDecHybridRNNTCTCModel
    (nemo/collections/asr/models/hybrid_rnnt_ctc_models.py)
    -------------------------------------------------------
    Multi-task learning combining CTC and RNN-T objectives.

    Architecture:
        - Shared encoder
        - CTC decoder head
        - RNN-T decoder + joint
        - Combined loss: α*CTC_loss + (1-α)*RNNT_loss

    Benefits:
        - Better convergence (CTC provides auxiliary supervision)
        - Can use either decoder at inference
        - Often better WER than single-task models

    Variants:
        - EncDecHybridRNNTCTCModel: Character-level
        - EncDecHybridRNNTCTCBPEModel: BPE tokenization
          (nemo/collections/asr/models/hybrid_rnnt_ctc_bpe_models.py)

    ========================================================================
    2.5 OTHER ARCHITECTURES
    ========================================================================

    - Transformer models (nemo/collections/asr/models/transformer_bpe_models.py)
    - AED (Attention Encoder-Decoder) (aed_multitask_models.py)
    - Multi-talker ASR (multitalker_asr_models.py)
    - Classification models (classification_models.py)
    """
    pass


# ============================================================================
# 3. CORE COMPONENTS
# ============================================================================

class ENCODER_ARCHITECTURES:
    """
    ========================================================================
    3.1 ENCODERS
    ========================================================================

    ConformerEncoder (nemo/collections/asr/modules/conformer_encoder.py)
    -------------------------------------------------------------------
    State-of-the-art encoder architecture (2020).

    Paper: "Conformer: Convolution-augmented Transformer for Speech Recognition"
    https://arxiv.org/abs/2005.08100

    Architecture:
        Each Conformer block contains:
        1. Feed-forward module (1st half)
        2. Multi-head self-attention module
        3. Convolution module
        4. Feed-forward module (2nd half)
        5. Layer normalization

        Block formula:
        x = x + 0.5 * FFN(x)
        x = x + MHSA(x)
        x = x + Conv(x)
        x = x + 0.5 * FFN(x)
        x = LayerNorm(x)

    Key Parameters:
        - feat_in: Input feature dimension (e.g., 80 for mel features)
        - n_layers: Number of Conformer blocks (12-24 typical)
        - d_model: Model dimension (256, 512, 1024)
        - n_heads: Number of attention heads (4-8)
        - conv_kernel_size: Convolution kernel size (31 typical)
        - ff_expansion_factor: FFN expansion (4 typical)
        - subsampling: Input subsampling strategy
          - 'vggnet': VGG-style strided convolutions
          - 'striding': Simple striding
          - 'dw_striding': Depthwise separable striding
          - 'stacking': Frame stacking
        - subsampling_factor: Temporal reduction factor (4 typical)
        - self_attention_model: Attention type
          - 'rel_pos': Relative positional encoding (default)
          - 'abs_pos': Absolute positional encoding
          - 'rel_pos_local_attn': Local attention variant

    Advanced Features:
        - Stochastic depth: Random layer dropping (regularization)
        - Local attention: For long-form audio (>30s)
        - Streaming support: Cache-aware attention
        - Causal convolution: For streaming models

    Input/Output:
        Input: (batch, time, feat_in)
        Output: (batch, time // subsampling_factor, d_model)

    Example Configuration:
        encoder:
          _target_: nemo.collections.asr.modules.ConformerEncoder
          feat_in: 80
          n_layers: 18
          d_model: 256
          n_heads: 4
          conv_kernel_size: 31
          subsampling: dw_striding
          subsampling_factor: 4

    Key Methods:
        - forward(audio_signal, length)
        - set_max_audio_length(max_audio_length)  # For CUDA graphs

    -----------------------------------------------------------------------

    SqueezeformerEncoder (nemo/collections/asr/modules/squeezeformer_encoder.py)
    ---------------------------------------------------------------------------
    Optimized Conformer variant with temporal U-Net structure.

    Paper: "Squeezeformer: An Efficient Transformer for ASR"
    https://arxiv.org/abs/2206.00888

    Key Differences from Conformer:
        - Temporal U-Net: Downsampling then upsampling
        - Unified attention context
        - Reduced parameters (~30% fewer)
        - Similar or better accuracy

    Use Case:
        - When you need efficient inference
        - Resource-constrained environments
        - Mobile/edge deployment

    -----------------------------------------------------------------------

    ConvASREncoder (nemo/collections/asr/modules/conv_asr.py)
    ---------------------------------------------------------
    Convolutional encoder for Jasper, QuartzNet models.

    Architecture:
        - Stack of JasperBlocks
        - Each block: Conv1D + BN + Activation + Dropout
        - Residual connections (skip connections)
        - Optional Squeeze-and-Excitation

    Models:
        - Jasper: Dense residual connections
        - QuartzNet: Time-channel separable convolutions (efficient)

    Key Parameters:
        - jasper: List of block configurations
        - activation: relu, silu, hardtanh, etc.
        - feat_in: Input features
        - residual_mode: 'add', 'max', 'stride_add'

    Use Case:
        - Older but proven architecture
        - Good for smaller datasets
        - Fast inference (pure convolution)

    -----------------------------------------------------------------------

    RNNEncoder (nemo/collections/asr/modules/rnn_encoder.py)
    --------------------------------------------------------
    LSTM/GRU-based encoder.

    Architecture:
        - Bidirectional LSTM/GRU
        - Multiple layers
        - Dropout between layers

    Use Case:
        - Legacy models
        - Simple baselines
        - Generally outperformed by Conformer

    ========================================================================
    """
    pass


class DECODER_ARCHITECTURES:
    """
    ========================================================================
    3.2 DECODERS
    ========================================================================

    RNN-T Decoder / Prediction Network
    -----------------------------------
    Location: nemo/collections/asr/modules/rnnt.py
    Abstract Interface: nemo/collections/asr/modules/rnnt_abstract.py

    Abstract Base: AbstractRNNTDecoder
    ----------------------------------
    Interface for all RNN-T prediction networks.

    Required Methods:
        - forward(targets, target_length, states=None)
        - predict(y, state, add_sos, batch_size)
        - initialize_state(batch_size)
        - batch_initialize_states(batch_size, device)
        - batch_select_state(states, idx)
        - batch_concat_states(states_list)

    -----------------------------------------------------------------------

    RNNTDecoder (Stateful LSTM-based)
    ----------------------------------
    The standard RNN-T prediction network.

    Architecture:
        1. Embedding layer: token_id -> embedding
        2. Multiple LSTM layers (typically 2)
        3. Linear projection to hidden dimension

    Components:
        - embedding: nn.Embedding(vocab_size, pred_hidden)
        - lstm: nn.LSTM(pred_hidden, pred_hidden, num_layers)
        - projection: nn.Linear(pred_hidden, output_dim)

    State Management:
        - Maintains hidden and cell states for LSTM
        - States shape: (num_layers, batch, pred_hidden)
        - Crucial for autoregressive decoding

    Forward Pass (Training):
        Input: targets (B, U), target_length (B,)
        1. Embed tokens -> (B, U, pred_hidden)
        2. Pass through LSTM -> (B, U, pred_hidden)
        3. Project -> (B, U, output_dim)
        Output: (B, U, output_dim), final_states

    Forward Pass (Inference):
        Input: previous_token (1,), state
        1. Embed token -> (1, 1, pred_hidden)
        2. LSTM with state -> (1, 1, pred_hidden), new_state
        3. Project -> (1, 1, output_dim)
        Output: (1, 1, output_dim), new_state

    Key Parameters:
        - pred_hidden: Hidden dimension (640 typical)
        - pred_rnn_layers: Number of LSTM layers (2 typical)
        - forget_gate_bias: LSTM forget gate bias (1.0 default)

    -----------------------------------------------------------------------

    StatelessTransducerDecoder
    ---------------------------
    Stateless variant of the prediction network.

    Key Difference:
        - No recurrent state
        - Uses context window of previous tokens
        - Faster inference (parallel processing)
        - Slightly lower accuracy than stateful

    Architecture:
        1. Embedding layer
        2. Context window concatenation
        3. Feed-forward network or Transformer

    Use Case:
        - When inference speed is critical
        - Deployment scenarios
        - Trade accuracy for latency

    -----------------------------------------------------------------------

    Joint Network (AbstractRNNTJoint)
    ---------------------------------
    Combines encoder and decoder outputs.

    Interface Methods:
        - forward(encoder_outputs, decoder_outputs)
        - project_encoder(encoder_output)
        - project_decoder(decoder_output)
        - joint(f, g)

    RNNTJoint Implementation:
    -------------------------
    Standard joint network.

    Architecture:
        1. Linear projection of encoder output
           f' = Linear(H_enc -> joint_hidden)
        2. Linear projection of decoder output
           g' = Linear(H_dec -> joint_hidden)
        3. Element-wise addition: h = f' + g'
        4. Activation: h = activation(h)
        5. Final projection: logits = Linear(joint_hidden -> vocab_size+1)

    Forward Pass:
        Input:
            encoder_output (B, T, H_enc)
            decoder_output (B, U, H_dec)

        Process:
            1. Expand dimensions for broadcasting
               encoder: (B, T, 1, H_enc)
               decoder: (B, 1, U, H_dec)
            2. Project and add
               joint: (B, T, U, joint_hidden)
            3. Activation + final projection
               logits: (B, T, U, V)

        Output: (B, T, U, V) where V = vocab_size + 1

    Key Parameters:
        - joint_hidden: Hidden dimension (640 typical)
        - activation: tanh, relu, silu
        - dropout: Regularization (0.1-0.2)

    -----------------------------------------------------------------------

    CTC Decoder
    -----------
    Simple linear projection layer.

    Architecture:
        decoder = nn.Linear(encoder_dim, vocab_size + 1)

    Output:
        logits (B, T, V) where V = vocab_size + 1 (includes blank token)

    Note:
        - No learnable alignment (handled by CTC loss)
        - Blank token is typically index 0 or vocab_size

    ========================================================================
    """
    pass


class PREPROCESSORS_AND_TOKENIZERS:
    """
    ========================================================================
    3.3 PREPROCESSORS
    ========================================================================

    AudioToMelSpectrogramPreprocessor
    ----------------------------------
    Location: nemo/collections/asr/modules/audio_preprocessing.py

    Primary feature extractor for ASR models.

    Processing Pipeline:
        Raw Audio (waveform)
        -> Windowing (Hann window)
        -> STFT (Short-Time Fourier Transform)
        -> Magnitude spectrum
        -> Mel filterbank
        -> Log compression
        -> Normalization
        -> Mel spectrogram features

    Key Parameters:
        - sample_rate: Audio sample rate (16000 Hz typical)
        - window_size: Window length in seconds (0.025 = 25ms)
        - window_stride: Hop length in seconds (0.01 = 10ms)
        - n_fft: FFT size (512 typical)
        - features: Number of mel bins (80 or 64 typical)
        - normalize: Normalization strategy
          - 'per_feature': Per mel bin (z-score per frequency)
          - 'all_features': Global normalization
          - None: No normalization
        - dither: Gaussian noise for numerical stability (1e-5)
        - preemph: Pre-emphasis coefficient (0.97 typical)
        - mag_power: Power spectrum (2 for energy, 1 for magnitude)
        - log: Apply log to features (True typical)
        - log_zero_guard_type: Handling log(0) ('add' or 'clamp')
        - log_zero_guard_value: Small constant (1e-05)

    Forward Pass:
        Input:
            input_signal (B, T_audio)
            length (B,)
        Output:
            features (B, n_features, T_frames)
            processed_length (B,)

        Where T_frames ≈ T_audio / (sample_rate * window_stride)

    Example Configuration:
        preprocessor:
          _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
          normalize: per_feature
          window_size: 0.025
          window_stride: 0.01
          window: hann
          features: 80
          n_fft: 512
          frame_splicing: 1
          dither: 0.00001
          stft_conv: false

    -----------------------------------------------------------------------

    AudioToMFCCPreprocessor
    -----------------------
    MFCC (Mel-Frequency Cepstral Coefficients) feature extractor.

    Additional Processing:
        Mel Spectrogram -> DCT -> MFCC features

    Key Parameters:
        - n_mfcc: Number of MFCC coefficients (20-40 typical)
        - All mel spectrogram parameters

    Use Case:
        - Traditional ASR systems
        - Less common in deep learning (mel spectrograms preferred)

    -----------------------------------------------------------------------

    Feature Implementation Details
    ------------------------------
    Location: nemo/collections/asr/parts/preprocessing/features.py

    FilterbankFeatures:
        Core class for mel spectrogram computation.

        Methods:
            - get_seq_len(seq_len): Compute output sequence length
            - forward(x, seq_len): Extract features

        Backends:
            - torchaudio (fast, recommended)
            - custom (fallback)

    ========================================================================
    3.4 TOKENIZERS
    ========================================================================

    ASRBPEMixin
    -----------
    Location: nemo/collections/asr/parts/mixins/mixins.py

    Manages tokenizer setup and vocabulary for BPE models.

    Tokenizer Types:
        1. BPE (Byte Pair Encoding) - SentencePiece
           - Subword tokenization
           - Vocabulary size: 128-8192 typical
           - Handles OOV (out-of-vocabulary) gracefully
           - Artifact: tokenizer.model

        2. WPE (WordPiece) - BERT-style
           - Similar to BPE
           - Different merging strategy

        3. Aggregate Tokenizers
           - Multilingual models
           - Language-specific vocabularies

    Key Methods:
        - _setup_tokenizer(tokenizer_cfg):
            Initialize tokenizer from config.

        - _setup_monolingual_tokenizer(cfg):
            Setup single-language tokenizer.

            Parameters:
                - tokenizer.dir: Directory with tokenizer files
                - tokenizer.type: 'bpe', 'wpe'
                - tokenizer.model_path: Path to tokenizer model

        - register_artifact('tokenizer', tokenizer_path):
            Register tokenizer for model export.

    Tokenizer Artifacts:
        - tokenizer.model: SentencePiece model file
        - vocab.txt: Vocabulary text file
        - tokenizer.vocab: SentencePiece vocabulary

    Usage in Model:
        # During initialization
        self._setup_tokenizer(cfg.tokenizer)

        # Encoding
        tokens = self.tokenizer.text_to_ids(text)

        # Decoding
        text = self.tokenizer.ids_to_text(tokens)

    -----------------------------------------------------------------------

    Character-level Tokenization
    ----------------------------
    No explicit tokenizer needed.

    Vocabulary:
        - List of characters: ['a', 'b', ..., 'z', ' ', "'", etc.]
        - Vocabulary size: ~30-50 for English
        - Defined in model config

    Usage:
        # Character to ID
        char_to_id = {c: i for i, c in enumerate(vocabulary)}

        # ID to character
        id_to_char = {i: c for i, c in enumerate(vocabulary)}

    ========================================================================
    """
    pass


# ============================================================================
# 4. INFERENCE PIPELINE
# ============================================================================

class DECODING_MECHANISMS:
    """
    ========================================================================
    4.1 CTC DECODING
    ========================================================================

    CTCDecoding
    -----------
    Location: nemo/collections/asr/parts/submodules/ctc_decoding.py

    Main orchestrator for CTC decoding strategies.

    Supported Strategies:
        1. greedy
        2. greedy_batch (faster batched version)
        3. beam (KenLM-based beam search)

    Initialization:
        decoding = CTCDecoding(
            decoding_cfg=cfg.decoding,
            vocabulary=model.vocabulary
        )

    Decoding Call:
        predictions = decoding.decode(
            log_probs,      # (B, T, V)
            log_probs_length  # (B,)
        )

    -----------------------------------------------------------------------

    CTCGreedyDecoder
    ----------------
    Location: nemo/collections/asr/parts/submodules/ctc_greedy_decoding.py

    Greedy CTC decoding algorithm.

    Algorithm:
        For each timestep:
            1. Select token with highest probability: argmax(log_probs[t])
            2. Remove consecutive duplicates
            3. Remove blank tokens
            4. Convert token IDs to text

    Example:
        Input logits: [b, b, c, a, a, t, b, b]  (b = blank)
        After dedup: [b, c, a, t, b]
        Remove blank: [c, a, t]
        Output: "cat"

    Features:
        - CUDA graphs support (10-100x speedup)
        - Batched processing
        - Confidence scoring
        - Timestamp computation

    Key Methods:
        - forward(log_probs, log_probs_length)
        - _greedy_decode(log_probs, log_probs_length)

    Confidence Computation:
        - Frame-level: Max probability at each timestep
        - Token-level: Aggregated frame confidence
        - Word-level: Aggregated token confidence

    -----------------------------------------------------------------------

    CTCBeamDecoder
    --------------
    Location: nemo/collections/asr/parts/submodules/ctc_beam_decoding.py

    Beam search with external language model (KenLM).

    Algorithm:
        1. Maintain top-k hypotheses (beam)
        2. For each hypothesis and timestep:
            - Extend with all possible tokens
            - Score: acoustic_score + α*LM_score + β*word_count
        3. Prune to beam_size
        4. Return best hypothesis

    Key Parameters:
        - beam_size: Number of hypotheses (128 typical)
        - beam_alpha: LM weight (1.0-2.0 typical)
        - beam_beta: Word insertion bonus (0.0-2.0)
        - kenlm_path: Path to KenLM .arpa or .bin file

    Example Configuration:
        decoding:
          strategy: beam
          beam:
            beam_size: 128
            beam_alpha: 1.5
            beam_beta: 1.0
            kenlm_path: /path/to/lm.arpa

    Use Case:
        - When you have domain-specific text corpus
        - Improving accuracy (5-20% WER reduction typical)
        - Trade-off: Slower inference (~10x)

    ========================================================================
    4.2 RNN-T DECODING
    ========================================================================

    RNNTDecoding
    ------------
    Location: nemo/collections/asr/parts/submodules/rnnt_decoding.py

    Main orchestrator for RNN-T decoding strategies.

    Supported Strategies:
        1. greedy - Single sample greedy
        2. greedy_batch - Batched greedy (recommended)
        3. beam - Beam search variants
           - tsd: Time Synchronous Decoding
           - alsd: Alignment-Length Synchronous Decoding
           - maes: Modified Adaptive Expansion Search

    -----------------------------------------------------------------------

    Greedy RNN-T Decoding
    ---------------------
    Location: nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py

    Classes:
        - _GreedyRNNTInfer: Base class
        - GreedyRNNTInfer: Single-sample inference
        - GreedyBatchedRNNTInfer: Batched inference (faster)

    Algorithm (for each acoustic timestep t):
        1. Get encoder output: f(x_t)
        2. Initialize: hypothesis = [], state = initial_state
        3. Loop until blank or max_symbols:
            a. Get decoder output: g(y) using current hypothesis
            b. Compute joint: h = joint(f(x_t), g(y))
            c. Get next token: k = argmax(h)
            d. If k == blank:
                - Break (move to next timestep)
            e. Else:
                - Append k to hypothesis
                - Update decoder state
                - Continue loop
        4. Return final hypothesis

    Key Parameters:
        - max_symbols_per_step: Maximum tokens per timestep (10 typical)
          Prevents infinite loops for pathological cases.

    Pseudocode:
        ```
        def greedy_decode(encoder_output):
            hypothesis = []
            state = decoder.initialize_state()

            for t in range(T):  # For each acoustic frame
                f = encoder_output[t]
                not_blank = True
                symbols_added = 0

                while not_blank and symbols_added < max_symbols:
                    g, state = decoder.predict(hypothesis[-1] if hypothesis else SOS, state)
                    logits = joint(f, g)
                    k = argmax(logits)

                    if k == blank:
                        not_blank = False
                    else:
                        hypothesis.append(k)
                        symbols_added += 1

            return hypothesis
        ```

    -----------------------------------------------------------------------

    Beam Search RNN-T
    -----------------
    Location: nemo/collections/asr/parts/submodules/rnnt_beam_decoding.py

    BeamRNNTInfer: Multiple beam search algorithms.

    1. TSD (Time Synchronous Decoding)
    -----------------------------------
    Standard beam search for RNN-T.

    Algorithm:
        - Maintain beam of top-k hypotheses
        - At each timestep, expand all hypotheses
        - Prune to beam_size based on score

    Complexity: O(T * U * beam_size * vocab_size)

    2. ALSD (Alignment-Length Synchronous Decoding)
    ------------------------------------------------
    Faster variant of beam search.

    Key Idea:
        - Synchronize by alignment length (U) instead of time (T)
        - Reduces redundant computation

    Speedup: 2-3x faster than TSD

    3. MAES (Modified Adaptive Expansion Search)
    ---------------------------------------------
    Adaptive beam expansion.

    Key Idea:
        - Dynamically adjust beam size
        - Expand only promising hypotheses
        - Prune aggressively

    Speedup: 3-5x faster than TSD
    Accuracy: Close to TSD

    Key Parameters:
        - beam_size: 4-8 typical for RNN-T (smaller than CTC)
        - search_type: 'tsd', 'alsd', 'maes'
        - score_norm: Normalize by length (True recommended)
        - return_best_hypothesis: Return top-1 (True) or all beams (False)

    Example Configuration:
        decoding:
          strategy: beam
          beam:
            beam_size: 4
            search_type: alsd
            score_norm: true
            return_best_hypothesis: true

    -----------------------------------------------------------------------

    CUDA Graphs Greedy Decoding
    ---------------------------
    Location: nemo/collections/asr/parts/submodules/cuda_graph_rnnt_greedy_decoding.py

    Optimized greedy decoding using CUDA graphs.

    Benefits:
        - 10-100x speedup for greedy decoding
        - Pre-recorded computation graphs
        - Reduced kernel launch overhead

    Limitations:
        - Fixed maximum length
        - Fixed batch size
        - Not compatible with dynamic shapes

    Use Case:
        - Production inference with consistent input sizes
        - Latency-critical applications

    ========================================================================
    4.3 CONTEXT BIASING
    ========================================================================

    Context Biasing
    ---------------
    Location: nemo/collections/asr/parts/context_biasing/

    Boost probabilities of specific words/phrases during decoding.

    Use Cases:
        - Domain-specific vocabulary (medical, legal terms)
        - Named entities (person names, locations)
        - Custom commands
        - Rare words that appear frequently in your domain

    Implementation:
        - Trie-based data structure for fast lookup
        - Score boosting during beam search
        - GPU-accelerated

    Key Classes:
        - GPUBoostingTreeModel: Efficient trie structure
        - ContextGraphCTC: CTC-specific biasing
        - ContextGraphUniversal: General-purpose biasing

    Usage:
        # Define context words
        context_words = ["COVID-19", "pneumonia", "hypertension"]

        # Create context graph
        context_graph = GPUBoostingTreeModel(
            context_words,
            boost_score=10.0  # Boost amount
        )

        # During decoding
        decoding = RNNTDecoding(
            decoding_cfg=cfg.decoding,
            context_graph=context_graph
        )

    Parameters:
        - boost_score: How much to boost (1.0-20.0)
        - context_words: List of words/phrases to boost

    Impact:
        - Can significantly improve accuracy for specified terms
        - 20-50% error reduction for biased words
        - Minimal impact on other words

    ========================================================================
    4.4 CONFIDENCE ESTIMATION
    ========================================================================

    Confidence Scoring
    ------------------
    Estimate reliability of predictions.

    Levels:
        1. Frame-level: Confidence at each time step
        2. Token-level: Confidence for each predicted token
        3. Word-level: Confidence for each word

    Methods:
        1. max_prob: Maximum probability
           conf = max(prob)

        2. entropy: Normalized entropy
           H = -Σ p*log(p)
           conf = 1 - H/log(vocab_size)

           Variants:
           - entropy (Shannon)
           - gibbs (Gibbs entropy, temperature-based)
           - tsallis (Tsallis entropy, with parameter q)
           - renyi (Renyi entropy, with parameter alpha)

    Aggregation (for word-level from token-level):
        - mean: Average confidence
        - min: Minimum confidence (conservative)
        - max: Maximum confidence (optimistic)
        - prod: Product of confidences

    Configuration:
        decoding:
          preserve_frame_confidence: true
          confidence_cfg:
            preserve_token_confidence: true
            preserve_word_confidence: true
            aggregation: min
            method_cfg:
              name: entropy
              entropy_type: tsallis
              alpha: 0.33

    Output Format:
        {
            'text': 'hello world',
            'confidence': 0.95,
            'word_confidence': [0.97, 0.93],
            'token_confidence': [0.98, 0.96, 0.95, 0.94, 0.92],
            'frame_confidence': [...]
        }

    Use Cases:
        - Filter low-confidence predictions
        - Identify segments needing review
        - Active learning (select uncertain samples)
        - Confidence thresholding for production

    ========================================================================
    """
    pass


# ============================================================================
# 5. TRAINING PIPELINE
# ============================================================================

class TRAINING_COMPONENTS:
    """
    ========================================================================
    5.1 LOSS FUNCTIONS
    ========================================================================

    CTCLoss
    -------
    Location: nemo/collections/asr/losses/ctc.py

    Wrapper around PyTorch's CTCLoss.

    CTC (Connectionist Temporal Classification):
        - Handles alignment between input and output sequences
        - Allows multiple frames to map to same token (or blank)
        - Marginalizes over all possible alignments

    Key Concepts:
        - Blank token: Represents no output
        - Alignment: Mapping from frames to tokens
        - Forward-backward algorithm: Efficiently computes loss

    Loss Computation:
        1. For each alignment path:
           - Compute probability using forward algorithm
        2. Sum probabilities of all valid paths
        3. Take negative log likelihood

    Key Parameters:
        - blank_id: Index of blank token (typically 0 or len(vocab))
        - zero_infinity: Replace inf values with 0 (stability)
        - reduction: How to aggregate loss
          - 'mean_batch': Average over batch
          - 'mean_volume': Weighted by sequence length
          - 'sum': Sum all losses
          - 'mean': Simple mean
          - 'none': No reduction

    Input Format:
        - log_probs: (B, T, V) - Log probabilities
        - targets: (B, U) - Target token IDs
        - input_lengths: (B,) - Acoustic sequence lengths
        - target_lengths: (B,) - Target sequence lengths

    Output:
        - loss: Scalar (if reduction != 'none')
        - loss: (B,) if reduction == 'none'

    Example Usage:
        loss_fn = CTCLoss(
            num_classes=vocab_size + 1,
            zero_infinity=True,
            reduction='mean_batch'
        )

        loss = loss_fn(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths
        )

    -----------------------------------------------------------------------

    RNNTLoss
    --------
    Location: nemo/collections/asr/losses/rnnt.py

    Multi-backend RNN-T loss implementation.

    RNN-T Loss:
        - Transducer loss for sequence-to-sequence learning
        - Joint probability over acoustic and label sequences
        - More flexible than CTC (allows label repetition)

    Available Backends (in preference order):
        1. warprnnt_numba (RECOMMENDED)
           - Numba-optimized CUDA kernels
           - Fastest implementation
           - Location: nemo/collections/asr/parts/numba/rnnt_loss/

        2. warprnnt
           - Original C++/CUDA implementation
           - Requires compilation

        3. pytorch
           - Pure PyTorch (slow)
           - For debugging only

        4. multiblank_rnnt
           - Multi-blank variant
           - Better for long-form audio

        5. tdt (Token-Duration Transducer)
           - Predicts duration explicitly
           - Better timestamps

        6. graph_rnnt, graph_w_transducer
           - K2-based graph losses
           - Experimental

    Algorithm (Forward-Backward):
        Forward pass:
            1. Compute alpha (forward probabilities)
               alpha[t, u] = P(output u in time 0..t)
            2. Use recursion:
               alpha[t, u] = alpha[t-1, u]*blank + alpha[t, u-1]*label

        Backward pass:
            1. Compute beta (backward probabilities)
               beta[t, u] = P(output u..U in time t..T)
            2. Similar recursion backward

        Loss:
            -log(alpha[T, U])

        Gradient:
            Computed using alpha and beta

    Key Parameters:
        - loss_name: Backend name ('default' = warprnnt_numba)
        - reduction: 'mean_batch', 'mean_volume', etc.
        - fastemit_lambda: FastEmit regularization (0.0-0.01)
          - Encourages earlier token emission
          - Reduces latency for streaming

    Input Format:
        - joint: (B, T, U, V) - Joint network output
        - targets: (B, U) - Target sequences
        - input_lengths: (B,) - Acoustic lengths
        - target_lengths: (B,) - Target lengths

    Output:
        - loss: Scalar or (B,) depending on reduction

    Example Usage:
        loss_fn = RNNTLoss(
            num_classes=vocab_size + 1,
            reduction='mean_batch',
            loss_name='default',  # warprnnt_numba
            fastemit_lambda=0.001
        )

        loss = loss_fn(
            acts=joint_output,
            labels=targets,
            act_lens=input_lengths,
            label_lens=target_lengths
        )

    -----------------------------------------------------------------------

    Numba RNN-T Implementation Details
    -----------------------------------
    Location: nemo/collections/asr/parts/numba/rnnt_loss/

    Files:
        - rnnt.py: Main loss function
        - rnnt_pytorch.py: PyTorch integration
        - utils/cuda_utils/: CUDA kernels

    Key Components:
        1. compute_alphas_kernel: Forward pass CUDA kernel
        2. compute_betas_kernel: Backward pass CUDA kernel
        3. compute_grad_kernel: Gradient computation

    Optimizations:
        - Efficient memory usage
        - Parallelized across batch and time
        - Numerical stability (log-space computation)

    ========================================================================
    5.2 FORWARD PASS
    ========================================================================

    CTC Model Training Step
    -----------------------

    def training_step(self, batch, batch_idx):
        # Unpack batch
        audio_signal, audio_lengths, targets, target_lengths = batch

        # Preprocessing: Audio -> Features
        processed_signal, processed_lengths = self.preprocessor(
            input_signal=audio_signal,
            length=audio_lengths
        )

        # Spec augmentation (training only)
        if self.spec_augmentation is not None:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal,
                length=processed_lengths
            )

        # Encoder: Features -> Acoustic embeddings
        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal,
            length=processed_lengths
        )

        # Decoder: Embeddings -> Logits
        log_probs = self.decoder(encoder_output=encoded)

        # CTC Loss
        loss = self.loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=encoded_len,
            target_lengths=target_lengths
        )

        # Auxiliary losses (e.g., InterCTC, adapters)
        if hasattr(self, 'compute_auxiliary_losses'):
            aux_loss = self.compute_auxiliary_losses()
            loss = loss + aux_loss

        return loss

    -----------------------------------------------------------------------

    RNN-T Model Training Step
    -------------------------

    def training_step(self, batch, batch_idx):
        # Unpack batch
        audio_signal, audio_lengths, targets, target_lengths = batch

        # Preprocessing
        processed_signal, processed_lengths = self.preprocessor(
            input_signal=audio_signal,
            length=audio_lengths
        )

        # Spec augmentation
        if self.spec_augmentation is not None:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal,
                length=processed_lengths
            )

        # Encoder: Audio -> Acoustic embeddings
        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal,
            length=processed_lengths
        )

        # Decoder: Targets -> Prediction embeddings
        # Note: Targets are shifted (teacher forcing)
        # Input to decoder: [SOS, y_1, y_2, ..., y_{n-1}]
        decoder_output, _ = self.decoder(
            targets=targets,
            target_length=target_lengths
        )

        # Joint network: Combine encoder and decoder
        joint_output = self.joint(
            encoder_outputs=encoded,
            decoder_outputs=decoder_output
        )

        # RNN-T Loss
        loss = self.loss(
            acts=joint_output,
            labels=targets,
            act_lens=encoded_len,
            label_lens=target_lengths
        )

        return loss

    -----------------------------------------------------------------------

    Hybrid Model Training Step
    ---------------------------

    def training_step(self, batch, batch_idx):
        # Shared preprocessing and encoding
        audio_signal, audio_lengths, targets, target_lengths = batch
        processed_signal, processed_lengths = self.preprocessor(...)
        encoded, encoded_len = self.encoder(...)

        # CTC branch
        ctc_log_probs = self.ctc_decoder(encoded)
        ctc_loss = self.ctc_loss(
            log_probs=ctc_log_probs,
            targets=targets,
            input_lengths=encoded_len,
            target_lengths=target_lengths
        )

        # RNN-T branch
        decoder_output, _ = self.decoder(targets, target_lengths)
        joint_output = self.joint(encoded, decoder_output)
        rnnt_loss = self.loss(
            acts=joint_output,
            labels=targets,
            act_lens=encoded_len,
            label_lens=target_lengths
        )

        # Combined loss
        # alpha = 0.3 typical (30% CTC, 70% RNN-T)
        alpha = self.cfg.model.ctc_alpha
        loss = alpha * ctc_loss + (1 - alpha) * rnnt_loss

        return loss

    ========================================================================
    5.3 OPTIMIZATION
    ========================================================================

    Optimizer Configuration
    -----------------------

    Typical Configuration:
        optim:
          name: adamw
          lr: 0.001
          betas: [0.9, 0.999]
          weight_decay: 0.0001

          sched:
            name: CosineAnnealing
            warmup_steps: 10000
            warmup_ratio: null
            min_lr: 1e-6

    Common Optimizers:
        1. AdamW: Adam with decoupled weight decay (recommended)
        2. Adam: Standard Adam
        3. Novograd: Novel optimizer for large batches
        4. SGD: Stochastic Gradient Descent (rare for ASR)

    Learning Rate Schedules:
        1. CosineAnnealing: Cosine decay with warmup
        2. WarmupAnnealing: Linear warmup + decay
        3. SquareRootAnnealing: sqrt(t) decay
        4. PolynomialDecayAnnealing: Polynomial decay

    -----------------------------------------------------------------------

    Special Training Features
    -------------------------

    1. skip_nan_grad (ASRModel)
       - Skip optimizer step if NaN gradients detected
       - Prevents model divergence

    2. CUDA Graphs (WithOptionalCudaGraphs)
       - Cache computation graphs for inference
       - Massive speedup (10-100x)

    3. Gradient Clipping
       - Prevent exploding gradients
       - Typical value: 1.0

    4. Mixed Precision (AMP)
       - Use float16 for faster training
       - Automatic loss scaling

    Example Training Configuration:
        trainer:
          devices: 8
          num_nodes: 1
          accelerator: gpu
          strategy: ddp
          precision: 16
          max_epochs: 100
          val_check_interval: 1.0
          gradient_clip_val: 1.0

    ========================================================================
    """
    pass


# ============================================================================
# 6. DATA PROCESSING
# ============================================================================

class DATA_PIPELINE:
    """
    ========================================================================
    6.1 DATASET CLASSES
    ========================================================================

    Manifest Format
    ---------------
    NeMo uses JSON manifest files for dataset specification.

    Format (one JSON object per line):
        {
            "audio_filepath": "/path/to/audio.wav",
            "text": "the transcription",
            "duration": 3.5,
            "offset": 0.0,
            "lang": "en"
        }

    Optional fields:
        - offset: Start time in audio file
        - duration: Length of segment
        - speaker: Speaker ID
        - lang: Language code

    -----------------------------------------------------------------------

    AudioToCharDataset
    ------------------
    Location: nemo/collections/asr/data/audio_to_text.py

    Character-level dataset for CTC/RNN-T models.

    Initialization:
        dataset = AudioToCharDataset(
            manifest_filepath="train_manifest.json",
            labels=vocabulary,  # List of characters
            sample_rate=16000,
            int_values=False,
            augmentor=augmentor,  # Optional audio augmentation
            max_duration=20.0,
            min_duration=0.1,
            trim=False,
            parser='en'  # Text parser
        )

    Processing:
        1. Load audio from manifest
        2. Apply audio augmentation (if specified)
        3. Normalize/resample audio
        4. Parse and encode text to character IDs
        5. Return: (audio, audio_len, tokens, token_len)

    -----------------------------------------------------------------------

    AudioToBPEDataset
    -----------------
    Subword tokenization dataset.

    Key Difference:
        - Uses tokenizer instead of character labels
        - tokenizer.text_to_ids(text) for encoding
        - More flexible vocabulary

    Initialization:
        dataset = AudioToBPEDataset(
            manifest_filepath="train_manifest.json",
            tokenizer=tokenizer,  # SentencePiece tokenizer
            sample_rate=16000,
            ...
        )

    -----------------------------------------------------------------------

    Tarred Datasets
    ---------------
    For large-scale training (>100k hours).

    Classes:
        - TarredAudioToCharDataset
        - TarredAudioToBPEDataset

    Benefits:
        - Efficient storage (webdataset format)
        - Fast I/O (streaming from tar files)
        - Good for distributed training

    Format:
        data_00000.tar:
            - 00000.wav
            - 00000.txt
            - 00001.wav
            - 00001.txt
            ...

    -----------------------------------------------------------------------

    Lhotse Integration
    ------------------
    Location: nemo/collections/asr/data/audio_to_text_lhotse.py

    Modern data loading with Lhotse library.

    Benefits:
        - Better performance
        - Dynamic batching by duration
        - On-the-fly augmentation
        - Cut management

    Usage:
        config.model.train_ds = {
            'use_lhotse': True,
            'cuts_path': 'cuts.jsonl.gz',
            'batch_duration': 600.0,  # seconds
            'num_workers': 8,
            ...
        }

    -----------------------------------------------------------------------

    Collation Function
    ------------------

    _speech_collate_fn:
        Combines samples into batches.

        Input: List of (audio, audio_len, tokens, token_len)

        Processing:
            1. Find max audio length in batch
            2. Pad all audio to max length
            3. Find max token length in batch
            4. Pad all token sequences
            5. Create length tensors

        Output:
            - audio_signal: (B, max_audio_len)
            - audio_lengths: (B,)
            - tokens: (B, max_token_len)
            - token_lengths: (B,)

    ========================================================================
    6.2 AUDIO AUGMENTATION
    ========================================================================

    Perturbation Classes
    --------------------
    Location: nemo/collections/asr/parts/preprocessing/perturb.py

    1. SpeedPerturbation
    --------------------
    Resample audio to different speed.

    Parameters:
        - min_speed_rate: 0.9 (90% speed)
        - max_speed_rate: 1.1 (110% speed)
        - p: Probability of applying (0.0-1.0)

    Effect:
        - Changes duration
        - Does NOT preserve pitch
        - Increases data diversity

    Example:
        perturb = SpeedPerturbation(
            min_speed_rate=0.95,
            max_speed_rate=1.05,
            p=0.5
        )

    2. TimeStretchPerturbation
    --------------------------
    Change speed while preserving pitch.

    Parameters:
        - min_speed_rate: 0.9
        - max_speed_rate: 1.1

    Uses: Phase vocoder algorithm

    3. GainPerturbation
    -------------------
    Random volume adjustment.

    Parameters:
        - min_gain_dbfs: -10 dB
        - max_gain_dbfs: 10 dB
        - p: 0.5

    Effect: Simulates different microphone gains

    4. ShiftPerturbation
    --------------------
    Time-shift the audio.

    Parameters:
        - min_shift_ms: -5.0 ms
        - max_shift_ms: 5.0 ms

    Effect: Temporal jittering

    5. NoisePerturbation
    --------------------
    Add background noise from corpus.

    Parameters:
        - manifest_path: Path to noise manifest
        - min_snr_db: 10 dB (minimum signal-to-noise ratio)
        - max_snr_db: 50 dB

    Noise Manifest:
        {
            "audio_filepath": "/path/to/noise.wav",
            "duration": 10.0
        }

    Effect: Simulates real-world noisy conditions

    6. WhiteNoisePerturbation
    -------------------------
    Add Gaussian white noise.

    Parameters:
        - min_level: -90 dB
        - max_level: -46 dB

    Effect: Simulates electronic noise

    7. ImpulsePerturbation
    ----------------------
    Apply room impulse response (RIR).

    Parameters:
        - manifest_path: Path to RIR manifest

    Effect: Simulates room acoustics, reverberation

    -----------------------------------------------------------------------

    AudioAugmentor
    --------------
    Combines multiple perturbations.

    Configuration:
        augmentor:
          speed:
            prob: 0.5
            min_speed_rate: 0.95
            max_speed_rate: 1.05

          noise:
            prob: 0.5
            manifest_path: noise_manifest.json
            min_snr_db: 10
            max_snr_db: 50

          gain:
            prob: 0.5
            min_gain_dbfs: -10
            max_gain_dbfs: 10

    Usage:
        augmentor = AudioAugmentor.from_config(config.augmentor)
        augmented_audio = augmentor.perturb(audio)

    ========================================================================
    6.3 SPECTROGRAM AUGMENTATION
    ========================================================================

    SpecAugment
    -----------
    Location: nemo/collections/asr/parts/submodules/spectr_augment.py

    Paper: "SpecAugment: A Simple Data Augmentation Method for ASR"
    https://arxiv.org/abs/1904.08779

    Augmentation Types:
        1. Frequency Masking: Zero out frequency bins
        2. Time Masking: Zero out time steps

    Parameters:
        - freq_masks: Number of frequency masks (2 typical)
        - freq_width: Max width of each mask (27 typical)
        - time_masks: Number of time masks (10 typical)
        - time_width: Max width of each mask (0.05 = 5% of length)
        - rect_masks: Number of rectangular masks (5 typical)
        - rect_time: Max time dimension (5)
        - rect_freq: Max frequency dimension (20)

    Algorithm:
        For each mask:
            1. Sample random width w ~ Uniform(0, max_width)
            2. Sample random position p ~ Uniform(0, T - w)
            3. Set features[p:p+w] = 0

    Example Configuration:
        spec_augment:
          _target_: nemo.collections.asr.modules.SpecAugment
          freq_masks: 2
          freq_width: 27
          time_masks: 10
          time_width: 0.05

    Implementation:
        - GPU-accelerated (vectorized operations)
        - Applied after preprocessing, before encoder
        - Training only (disabled during validation/test)

    Effectiveness:
        - 5-20% WER reduction typical
        - Acts as strong regularizer
        - Prevents overfitting to specific frequencies/times

    -----------------------------------------------------------------------

    SpecCutout
    ----------
    Cutout augmentation for spectrograms.

    Similar to SpecAugment but:
        - Rectangular regions (both time and frequency)
        - Different masking strategy

    Parameters:
        - rect_masks: Number of rectangles
        - rect_time: Max time dimension
        - rect_freq: Max frequency dimension

    -----------------------------------------------------------------------

    Numba-Accelerated SpecAugment
    -----------------------------
    Location: nemo/collections/asr/parts/numba/spec_augment/

    Even faster implementation using Numba CUDA kernels.

    Benefits:
        - 2-3x faster than standard implementation
        - Lower memory usage
        - Seamless replacement

    ========================================================================
    """
    pass


# ============================================================================
# 7. ADVANCED FEATURES
# ============================================================================

class ADVANCED_CAPABILITIES:
    """
    ========================================================================
    7.1 STREAMING & CACHE-AWARE MODELS
    ========================================================================

    Streaming ASR
    -------------
    Process audio incrementally as it arrives.

    Requirements:
        1. Causal encoder (no future context)
        2. Efficient state management
        3. Chunk-based processing

    -----------------------------------------------------------------------

    Cache-Aware Streaming
    ---------------------
    Location: nemo/collections/asr/models/configs/asr_models_config.py

    Configuration:
        CacheAwareStreamingConfig:
            - chunk_size: [1.6, 1.6] # [left, right] in seconds
            - shift_size: [0.4, 0.4]
            - cache_size: [3.2, 3.2]
            - valid_out_len: 1.6

    Mechanism:
        1. Encoder maintains cache of previous frames
        2. Process audio in chunks
        3. Use left context from cache
        4. Limited right context (lookahead)
        5. Update cache for next chunk

    Example Encoder Configuration:
        encoder:
          att_context_size: [64, 64]  # Left, right attention context
          conv_context_size: [left, right]
          use_cache_aware_streaming: true

    -----------------------------------------------------------------------

    Streaming Inference Pipeline
    ----------------------------
    Location: nemo/collections/asr/inference/pipelines/

    Pipelines:
        - CacheAwareCTCPipeline
        - CacheAwareRNNTPipeline
        - BufferedCTCPipeline
        - BufferedRNNTPipeline

    Usage:
        pipeline = CacheAwareRNNTPipeline(
            model=model,
            chunk_size=1.6,  # seconds
            shift_size=0.4
        )

        # Process audio chunks
        for chunk in audio_chunks:
            partial_result = pipeline.process_chunk(chunk)
            print(partial_result)

        # Finalize
        final_result = pipeline.finalize()

    Features:
        - Real-time processing
        - Low latency (<500ms typical)
        - Buffering and smoothing
        - Endpointing detection

    ========================================================================
    7.2 MULTI-BLANK & TDT
    ========================================================================

    Multi-blank RNN-T
    -----------------
    Paper: "Multi-blank Transducers for Speech Recognition"
    https://arxiv.org/abs/2211.03541

    Concept:
        - Instead of 1 blank token, use K blank tokens
        - Each blank represents different "wait duration"
        - Better alignment for long-form audio
        - Improved WER (5-10% reduction)

    Architecture Change:
        - Joint network outputs: [vocab, blank_0, blank_1, ..., blank_K]
        - K = 2-8 typical

    Loss:
        - Modified RNN-T loss
        - Backend: 'multiblank_rnnt'

    Configuration:
        joint:
          num_extra_outputs: 2  # K-1 (total K+1 with original blank)

        loss:
          loss_name: multiblank_rnnt
          num_blanks: 3  # K

    -----------------------------------------------------------------------

    TDT (Token-and-Duration Transducer)
    -----------------------------------
    Paper: "Token-and-Duration Transducer for ASR"
    https://arxiv.org/abs/2304.06795

    Concept:
        - Predict token AND its duration explicitly
        - Joint network has two heads:
          1. Token prediction
          2. Duration prediction
        - Better timestamp accuracy
        - Improved alignment

    Architecture:
        joint:
          token_head: Linear(joint_hidden -> vocab_size + 1)
          duration_head: Linear(joint_hidden -> max_duration)

    Loss:
        - Combined token and duration loss
        - Backend: 'tdt'

    Benefits:
        - Accurate word timestamps (crucial for subtitles, alignment)
        - Better handling of long tokens
        - 5-15% WER improvement in some cases

    Configuration:
        loss:
          loss_name: tdt
          tdt_cfg:
            max_duration: 10
            duration_loss_weight: 0.5

    ========================================================================
    7.3 ADAPTER MODULES
    ========================================================================

    Adapter Modules
    ---------------
    Location: nemo/collections/asr/parts/submodules/adapters/

    Concept: Parameter-Efficient Fine-Tuning (PEFT)

    Idea:
        1. Freeze pre-trained model
        2. Add small adapter layers (1-5% parameters)
        3. Fine-tune only adapters

    Benefits:
        - Fast adaptation to new domain
        - Low memory (only adapter gradients)
        - Preserve base model knowledge
        - Multiple adapters for different domains

    Adapter Types:
        1. Bottleneck adapters (after transformer layers)
        2. LoRA (Low-Rank Adaptation)
        3. Prefix tuning adapters

    Configuration:
        model:
          encoder:
            adapter_modules:
              _target_: nemo.collections.asr.modules.AdapterModule
              hidden_size: 512
              adapter_dim: 64
              dropout: 0.1

    Usage:
        # Add adapter
        model.add_adapter("medical", adapter_config)

        # Fine-tune on medical data
        trainer.fit(model, medical_dataloader)

        # Switch adapters
        model.set_active_adapter("medical")

        # Inference
        transcription = model.transcribe(audio)

    -----------------------------------------------------------------------

    ASRAdapterModelMixin
    --------------------
    Location: nemo/collections/asr/parts/mixins/asr_adapter_mixins.py

    Methods:
        - add_adapter(name, cfg)
        - remove_adapter(name)
        - set_active_adapter(name)
        - freeze_adapter(name)
        - unfreeze_adapter(name)
        - list_adapters() -> List[str]

    ========================================================================
    7.4 N-GRAM LANGUAGE MODELS
    ========================================================================

    N-gram LM Integration
    ---------------------
    Location: nemo/collections/asr/parts/submodules/ngram_lm.py

    Purpose: Improve accuracy with external language model.

    Supported Formats:
        1. KenLM (.arpa or .bin files)
        2. Custom n-gram models

    Integration Methods:
        1. Shallow Fusion (during beam search)
           score = acoustic_score + α * lm_score

        2. Deep Fusion (integrate into model)
           Less common for ASR

    -----------------------------------------------------------------------

    NGramGPULanguageModel
    ---------------------
    GPU-accelerated n-gram LM.

    Features:
        - Fast lookup on GPU
        - Batch processing
        - Configurable LM weight (alpha)

    Usage with Beam Search:
        decoding:
          strategy: beam
          beam:
            beam_size: 128
            ngram_lm_path: /path/to/lm.arpa
            ngram_lm_alpha: 1.5  # LM weight
            ngram_lm_beta: 1.0   # Word insertion bonus

    -----------------------------------------------------------------------

    Building Custom LM
    ------------------

    Using KenLM:
        # Install KenLM
        # Prepare text corpus: corpus.txt

        # Build LM
        lmplz -o 4 < corpus.txt > lm.arpa

        # Binary format (faster)
        build_binary lm.arpa lm.bin

    Effect:
        - 5-20% WER reduction typical
        - Larger improvement with domain-specific corpus
        - Trade-off: Slower inference (10-100x)

    ========================================================================
    """
    pass


# ============================================================================
# 8. CONFIGURATION MANAGEMENT
# ============================================================================

class CONFIGURATION_SYSTEM:
    """
    ========================================================================
    HYDRA CONFIGURATION SYSTEM
    ========================================================================

    NeMo uses Hydra and OmegaConf for configuration management.

    Benefits:
        - Type-safe configurations
        - Hierarchical composition
        - Command-line overrides
        - Config inheritance

    -----------------------------------------------------------------------

    Configuration Files
    -------------------
    Location: examples/asr/conf/

    Structure:
        conf/
        ├── config.yaml              # Main config
        ├── model/
        │   ├── conformer_ctc.yaml   # CTC model
        │   ├── conformer_rnnt.yaml  # RNN-T model
        │   └── ...
        ├── encoder/
        │   ├── conformer.yaml
        │   └── squeezeformer.yaml
        └── ...

    -----------------------------------------------------------------------

    Example Full Configuration
    --------------------------

    model:
      # Model type
      _target_: nemo.collections.asr.models.EncDecRNNTBPEModel

      # Sample rate
      sample_rate: 16000

      # Preprocessor
      preprocessor:
        _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
        normalize: per_feature
        window_size: 0.025
        window_stride: 0.01
        window: hann
        features: 80
        n_fft: 512
        dither: 0.00001

      # SpecAugment
      spec_augment:
        _target_: nemo.collections.asr.modules.SpecAugment
        freq_masks: 2
        freq_width: 27
        time_masks: 10
        time_width: 0.05

      # Encoder
      encoder:
        _target_: nemo.collections.asr.modules.ConformerEncoder
        feat_in: 80
        n_layers: 18
        d_model: 256
        n_heads: 4
        ff_expansion_factor: 4
        conv_kernel_size: 31
        subsampling: dw_striding
        subsampling_factor: 4
        dropout: 0.1
        dropout_att: 0.1
        self_attention_model: rel_pos

      # Decoder (Prediction Network)
      decoder:
        _target_: nemo.collections.asr.modules.RNNTDecoder
        prednet:
          pred_hidden: 640
          pred_rnn_layers: 2
          dropout: 0.2

      # Joint Network
      joint:
        _target_: nemo.collections.asr.modules.RNNTJoint
        jointnet:
          joint_hidden: 640
          activation: relu
          dropout: 0.2

      # Loss
      loss:
        loss_name: default  # warprnnt_numba
        reduction: mean_batch

      # Decoding
      decoding:
        strategy: greedy_batch
        greedy:
          max_symbols_per_step: 10

      # Tokenizer
      tokenizer:
        dir: tokenizer/
        type: bpe

      # Training data
      train_ds:
        manifest_filepath: train_manifest.json
        sample_rate: 16000
        batch_size: 16
        shuffle: true
        num_workers: 8

        augmentor:
          speed:
            prob: 0.5
            min_speed_rate: 0.95
            max_speed_rate: 1.05
          noise:
            prob: 0.5
            manifest_path: noise.json
            min_snr_db: 10
            max_snr_db: 50

      # Validation data
      validation_ds:
        manifest_filepath: dev_manifest.json
        sample_rate: 16000
        batch_size: 32
        shuffle: false
        num_workers: 4

      # Optimizer
      optim:
        name: adamw
        lr: 0.001
        betas: [0.9, 0.999]
        weight_decay: 0.0001

        sched:
          name: CosineAnnealing
          warmup_steps: 10000
          min_lr: 1e-6

    # Trainer configuration
    trainer:
      devices: 8
      num_nodes: 1
      accelerator: gpu
      strategy: ddp
      precision: 16
      max_epochs: 100
      val_check_interval: 1.0
      gradient_clip_val: 1.0
      accumulate_grad_batches: 1

    -----------------------------------------------------------------------

    Command-Line Overrides
    ----------------------

    python train_asr.py \
      model.train_ds.batch_size=32 \
      model.optim.lr=0.0005 \
      trainer.max_epochs=200 \
      trainer.devices=4

    -----------------------------------------------------------------------

    Config Dataclasses
    ------------------
    Location: nemo/collections/asr/models/configs/

    Type-safe configuration using dataclasses:
        - asr_models_config.py
        - classification_models_config.py
        - quartznet_config.py

    Example:
        @dataclass
        class ConformerEncoderConfig:
            feat_in: int
            n_layers: int
            d_model: int
            n_heads: int = 4
            conv_kernel_size: int = 31
            ...

    ========================================================================
    """
    pass


# ============================================================================
# 9. DETAILED CLASS & FUNCTION REFERENCE
# ============================================================================

class DETAILED_REFERENCE:
    """
    ========================================================================
    KEY FILE LOCATIONS & PRIMARY CLASSES
    ========================================================================

    MODELS
    ------
    nemo/collections/asr/models/asr_model.py
        - ASRModel: Base class
        - ExportableEncDecModel: Export mixin

    nemo/collections/asr/models/ctc_models.py
        - EncDecCTCModel: CTC character model

    nemo/collections/asr/models/ctc_bpe_models.py
        - EncDecCTCModelBPE: CTC BPE model

    nemo/collections/asr/models/rnnt_models.py
        - EncDecRNNTModel: RNN-T character model

    nemo/collections/asr/models/rnnt_bpe_models.py
        - EncDecRNNTBPEModel: RNN-T BPE model

    nemo/collections/asr/models/hybrid_rnnt_ctc_models.py
        - EncDecHybridRNNTCTCModel: Hybrid model

    -----------------------------------------------------------------------

    ENCODERS
    --------
    nemo/collections/asr/modules/conformer_encoder.py
        - ConformerEncoder: SOTA encoder
        - ConformerBlock: Single Conformer block
        - ConformerConvolution: Convolution module
        - MultiHeadAttention: Self-attention

    nemo/collections/asr/modules/squeezeformer_encoder.py
        - SqueezeformerEncoder: Efficient variant

    nemo/collections/asr/modules/conv_asr.py
        - ConvASREncoder: Jasper/QuartzNet encoder
        - JasperBlock: Convolutional block

    -----------------------------------------------------------------------

    DECODERS & JOINT
    ----------------
    nemo/collections/asr/modules/rnnt.py
        - RNNTDecoder: LSTM prediction network
        - RNNTJoint: Joint network
        - StatelessTransducerDecoder: Stateless variant

    nemo/collections/asr/modules/rnnt_abstract.py
        - AbstractRNNTDecoder: Interface
        - AbstractRNNTJoint: Interface

    nemo/collections/asr/modules/hybrid_autoregressive_transducer.py
        - HATJoint: Hybrid autoregressive joint

    -----------------------------------------------------------------------

    PREPROCESSING
    -------------
    nemo/collections/asr/modules/audio_preprocessing.py
        - AudioToMelSpectrogramPreprocessor: Mel features
        - AudioToMFCCPreprocessor: MFCC features

    nemo/collections/asr/parts/preprocessing/features.py
        - FilterbankFeatures: Core feature extraction
        - WaveformFeaturizer: Audio loading

    nemo/collections/asr/parts/preprocessing/perturb.py
        - SpeedPerturbation
        - NoisePerturbation
        - GainPerturbation
        - ImpulsePerturbation
        - ShiftPerturbation
        - WhiteNoisePerturbation
        - TimeStretchPerturbation
        - AudioAugmentor: Orchestrator

    nemo/collections/asr/parts/preprocessing/segment.py
        - AudioSegment: Audio file handling

    -----------------------------------------------------------------------

    LOSSES
    ------
    nemo/collections/asr/losses/ctc.py
        - CTCLoss: CTC loss wrapper

    nemo/collections/asr/losses/rnnt.py
        - RNNTLoss: Multi-backend RNN-T loss

    nemo/collections/asr/parts/numba/rnnt_loss/rnnt.py
        - Numba-optimized RNN-T loss implementation

    -----------------------------------------------------------------------

    DECODING
    --------
    CTC:
        nemo/collections/asr/parts/submodules/ctc_decoding.py
            - CTCDecoding: Orchestrator

        nemo/collections/asr/parts/submodules/ctc_greedy_decoding.py
            - CTCGreedyDecoder: Greedy decoding

        nemo/collections/asr/parts/submodules/ctc_beam_decoding.py
            - CTCBeamDecoder: Beam search with LM

        nemo/collections/asr/parts/submodules/ctc_batched_beam_decoding.py
            - Batched beam search

    RNN-T:
        nemo/collections/asr/parts/submodules/rnnt_decoding.py
            - RNNTDecoding: Orchestrator

        nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py
            - GreedyRNNTInfer: Single-sample greedy
            - GreedyBatchedRNNTInfer: Batched greedy

        nemo/collections/asr/parts/submodules/rnnt_beam_decoding.py
            - BeamRNNTInfer: Multiple beam algorithms

        nemo/collections/asr/parts/submodules/cuda_graph_rnnt_greedy_decoding.py
            - CUDA graphs optimization

    -----------------------------------------------------------------------

    DATA
    ----
    nemo/collections/asr/data/audio_to_text.py
        - AudioToCharDataset: Character-level
        - AudioToBPEDataset: BPE tokenization
        - TarredAudioToCharDataset: Tarred format
        - TarredAudioToBPEDataset: Tarred BPE
        - ASRManifestProcessor: Manifest handling
        - _speech_collate_fn: Batch collation

    nemo/collections/asr/data/audio_to_text_lhotse.py
        - LhotseDataLoadingConfig: Lhotse integration

    -----------------------------------------------------------------------

    MIXINS
    ------
    nemo/collections/asr/parts/mixins/mixins.py
        - ASRModuleMixin: Dataset management
        - ASRBPEMixin: Tokenizer management

    nemo/collections/asr/parts/mixins/transcription.py
        - ASRTranscriptionMixin: Transcription methods

    nemo/collections/asr/parts/mixins/interctc_mixin.py
        - InterCTCMixin: Intermediate CTC

    nemo/collections/asr/parts/mixins/streaming.py
        - StreamingEncoder: Streaming support

    -----------------------------------------------------------------------

    AUGMENTATION
    ------------
    nemo/collections/asr/parts/submodules/spectr_augment.py
        - SpecAugment: SpecAugment implementation
        - SpecCutout: Cutout augmentation

    nemo/collections/asr/parts/numba/spec_augment/
        - Numba-optimized SpecAugment

    -----------------------------------------------------------------------

    CONTEXT BIASING
    ---------------
    nemo/collections/asr/parts/context_biasing/context_graph_ctc.py
        - CTC context biasing

    nemo/collections/asr/parts/context_biasing/context_graph_universal.py
        - Universal context graph

    -----------------------------------------------------------------------

    INFERENCE PIPELINES
    -------------------
    nemo/collections/asr/inference/pipelines/buffered_ctc_pipeline.py
        - BufferedCTCPipeline

    nemo/collections/asr/inference/pipelines/buffered_rnnt_pipeline.py
        - BufferedRNNTPipeline

    nemo/collections/asr/inference/pipelines/cache_aware_ctc_pipeline.py
        - CacheAwareCTCPipeline

    nemo/collections/asr/inference/pipelines/cache_aware_rnnt_pipeline.py
        - CacheAwareRNNTPipeline

    -----------------------------------------------------------------------

    METRICS
    -------
    nemo/collections/asr/metrics/wer.py
        - WER: Word Error Rate computation
        - CER: Character Error Rate

    ========================================================================
    KEY METHODS REFERENCE
    ========================================================================

    ASRModel (Base)
    ---------------
    - __init__(cfg, trainer)
    - training_step(batch, batch_idx)
    - validation_step(batch, batch_idx, dataloader_idx=0)
    - test_step(batch, batch_idx, dataloader_idx=0)
    - multi_validation_epoch_end(outputs, dataloader_idx=0)
    - multi_test_epoch_end(outputs, dataloader_idx=0)

    EncDecCTCModel
    --------------
    - forward(input_signal, input_signal_length, ...)
    - training_step(batch, batch_idx)
    - validation_step(batch, batch_idx, dataloader_idx=0)
    - test_step(batch, batch_idx, dataloader_idx=0)
    - change_vocabulary(new_vocabulary)
    - transcribe(audio, batch_size=4, logprobs=False)
    - _setup_dataloader_from_config(config, shuffle)

    EncDecRNNTModel
    ---------------
    - forward(input_signal, input_signal_length, transcript, transcript_length)
    - training_step(batch, batch_idx)
    - validation_step(batch, batch_idx, dataloader_idx=0)
    - transcribe(audio, batch_size=4)
    - change_vocabulary(new_vocabulary, decoding_cfg=None)

    ConformerEncoder
    ----------------
    - __init__(feat_in, n_layers, d_model, ...)
    - forward(audio_signal, length, cache_last_channel=None, ...)
    - set_max_audio_length(max_audio_length)
    - update_max_seq_length(seq_length, device)

    RNNTDecoder
    -----------
    - forward(targets, target_length, states=None)
    - predict(y, state, add_sos=False, batch_size=None)
    - initialize_state(batch_size)
    - batch_initialize_states(batch_size, device)

    RNNTJoint
    ---------
    - forward(encoder_outputs, decoder_outputs)
    - joint(f, g)
    - project_encoder(encoder_output)
    - project_decoder(decoder_output)

    CTCDecoding
    -----------
    - __init__(decoding_cfg, vocabulary)
    - decode(log_probs, log_probs_length)

    RNNTDecoding
    ------------
    - __init__(decoding_cfg, decoder, joint, blank_id)
    - decode(encoder_output, encoded_lengths)

    ASRTranscriptionMixin
    ---------------------
    - transcribe(audio, batch_size=4, logprobs=False, ...)
    - transcribe_file(audio_file, output_file=None)

    ========================================================================
    """
    pass


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

class USAGE_EXAMPLES:
    """
    ========================================================================
    EXAMPLE 1: LOADING PRE-TRAINED MODEL
    ========================================================================

    import nemo.collections.asr as nemo_asr

    # Load pre-trained Conformer-CTC model
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name='stt_en_conformer_ctc_large'
    )

    # Transcribe audio
    transcriptions = model.transcribe(
        audio=['audio1.wav', 'audio2.wav'],
        batch_size=4
    )

    for text in transcriptions:
        print(text)

    ========================================================================
    EXAMPLE 2: TRAINING FROM SCRATCH
    ========================================================================

    import pytorch_lightning as pl
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf

    # Load configuration
    config = OmegaConf.load('config.yaml')

    # Create model
    model = EncDecRNNTBPEModel(cfg=config.model, trainer=None)

    # Setup trainer
    trainer = pl.Trainer(
        devices=8,
        accelerator='gpu',
        strategy='ddp',
        max_epochs=100,
        precision=16
    )

    # Train
    trainer.fit(model)

    ========================================================================
    EXAMPLE 3: FINE-TUNING
    ========================================================================

    # Load pre-trained model
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        model_name='stt_en_conformer_transducer_large'
    )

    # Update data configuration
    model.cfg.train_ds.manifest_filepath = 'custom_train.json'
    model.cfg.validation_ds.manifest_filepath = 'custom_dev.json'

    # Setup data
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    # Optionally freeze encoder
    model.encoder.freeze()

    # Fine-tune
    trainer = pl.Trainer(devices=4, max_epochs=10)
    trainer.fit(model)

    ========================================================================
    EXAMPLE 4: INFERENCE WITH BEAM SEARCH
    ========================================================================

    from omegaconf import OmegaConf

    # Load model
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        model_name='stt_en_conformer_transducer_large'
    )

    # Configure beam search
    decoding_cfg = OmegaConf.create({
        'strategy': 'beam',
        'beam': {
            'beam_size': 4,
            'search_type': 'alsd',
            'score_norm': True,
            'return_best_hypothesis': True
        }
    })

    # Update decoding
    model.change_decoding_strategy(decoding_cfg)

    # Transcribe
    transcriptions = model.transcribe(audio=['test.wav'])
    print(transcriptions[0])

    ========================================================================
    EXAMPLE 5: STREAMING INFERENCE
    ========================================================================

    from nemo.collections.asr.inference.pipelines import CacheAwareRNNTPipeline

    # Load streaming model
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        model_name='stt_en_conformer_transducer_large_streaming'
    )

    # Create pipeline
    pipeline = CacheAwareRNNTPipeline(
        model=model,
        chunk_size=1.6,
        shift_size=0.4
    )

    # Process audio chunks
    import soundfile as sf
    audio, sr = sf.read('long_audio.wav')

    chunk_samples = int(1.6 * sr)
    shift_samples = int(0.4 * sr)

    for start in range(0, len(audio), shift_samples):
        chunk = audio[start:start + chunk_samples]
        partial_result = pipeline.process_chunk(chunk)
        if partial_result:
            print(f"Partial: {partial_result}")

    final_result = pipeline.finalize()
    print(f"Final: {final_result}")

    ========================================================================
    EXAMPLE 6: CONTEXT BIASING
    ========================================================================

    from nemo.collections.asr.parts.context_biasing import GPUBoostingTreeModel

    # Define context words
    context_words = ["COVID-19", "pneumonia", "hypertension", "diabetes"]

    # Create context graph
    context_graph = GPUBoostingTreeModel(
        context_words=context_words,
        boost_score=10.0
    )

    # Update model decoding with context biasing
    model.decoding.context_graph = context_graph

    # Transcribe (context words will be boosted)
    transcriptions = model.transcribe(audio=['medical_audio.wav'])

    ========================================================================
    EXAMPLE 7: CUSTOM AUDIO AUGMENTATION
    ========================================================================

    from nemo.collections.asr.parts.preprocessing.perturb import (
        AudioAugmentor, SpeedPerturbation, NoisePerturbation, GainPerturbation
    )

    # Create augmentations
    augmentations = [
        SpeedPerturbation(
            min_speed_rate=0.95,
            max_speed_rate=1.05,
            p=0.5
        ),
        NoisePerturbation(
            manifest_path='noise_manifest.json',
            min_snr_db=10,
            max_snr_db=50,
            p=0.5
        ),
        GainPerturbation(
            min_gain_dbfs=-10,
            max_gain_dbfs=10,
            p=0.5
        )
    ]

    # Create augmentor
    augmentor = AudioAugmentor(perturbations=augmentations)

    # Use in dataset configuration
    model.cfg.train_ds.augmentor = augmentor

    ========================================================================
    """
    pass


# ============================================================================
# COMMON WORKFLOWS
# ============================================================================

class COMMON_WORKFLOWS:
    """
    ========================================================================
    WORKFLOW 1: PREPARING DATA
    ========================================================================

    Step 1: Organize Audio Files
    ----------------------------
    data/
    ├── train/
    │   ├── audio1.wav
    │   ├── audio2.wav
    │   └── ...
    ├── dev/
    │   └── ...
    └── test/
        └── ...

    Step 2: Create Manifest Files
    -----------------------------
    # train_manifest.json (one JSON object per line)
    {"audio_filepath": "data/train/audio1.wav", "text": "hello world", "duration": 2.5}
    {"audio_filepath": "data/train/audio2.wav", "text": "good morning", "duration": 3.1}
    ...

    # Script to create manifest
    import json
    import soundfile as sf

    manifest = []
    for audio_path, text in zip(audio_files, transcriptions):
        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr
        manifest.append({
            'audio_filepath': audio_path,
            'text': text,
            'duration': duration
        })

    with open('train_manifest.json', 'w') as f:
        for item in manifest:
            f.write(json.dumps(item) + '\n')

    Step 3: (Optional) Train Tokenizer
    ----------------------------------
    # For BPE models, train SentencePiece tokenizer
    python <NeMo>/scripts/tokenizers/process_asr_text_tokenizer.py \
      --manifest=train_manifest.json \
      --vocab_size=1024 \
      --tokenizer=spe \
      --spe_type=bpe \
      --output_dir=tokenizer/

    ========================================================================
    WORKFLOW 2: TRAINING A MODEL
    ========================================================================

    Step 1: Prepare Configuration
    -----------------------------
    # config.yaml (see CONFIGURATION_SYSTEM above)

    Step 2: Train
    -------------
    python train_asr.py \
      --config-path=conf \
      --config-name=config \
      model.train_ds.manifest_filepath=train_manifest.json \
      model.validation_ds.manifest_filepath=dev_manifest.json \
      trainer.devices=8 \
      trainer.max_epochs=100

    Step 3: Monitor Training
    -----------------------
    # TensorBoard logs
    tensorboard --logdir=nemo_experiments/

    # View metrics: WER, loss, learning rate

    Step 4: Evaluate
    ---------------
    python evaluate_asr.py \
      --model_path=nemo_experiments/checkpoints/best.nemo \
      --test_manifest=test_manifest.json

    ========================================================================
    WORKFLOW 3: INFERENCE AT SCALE
    ========================================================================

    Step 1: Load Model
    -----------------
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
        'model.nemo'
    )
    model.freeze()
    model.eval()

    Step 2: Prepare Audio List
    --------------------------
    audio_files = [
        'audio1.wav',
        'audio2.wav',
        ...
    ]

    Step 3: Batch Transcribe
    -----------------------
    transcriptions = model.transcribe(
        audio=audio_files,
        batch_size=32,  # Adjust based on GPU memory
        num_workers=4
    )

    Step 4: Save Results
    -------------------
    with open('results.json', 'w') as f:
        for audio, text in zip(audio_files, transcriptions):
            f.write(json.dumps({
                'audio': audio,
                'text': text
            }) + '\n')

    ========================================================================
    WORKFLOW 4: DOMAIN ADAPTATION WITH ADAPTERS
    ========================================================================

    Step 1: Load Base Model
    ----------------------
    base_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        'stt_en_conformer_transducer_large'
    )

    Step 2: Add Adapter
    ------------------
    from omegaconf import OmegaConf

    adapter_cfg = OmegaConf.create({
        '_target_': 'nemo.collections.asr.modules.AdapterModule',
        'hidden_size': 1024,
        'adapter_dim': 128,
        'dropout': 0.1
    })

    base_model.add_adapter('medical', adapter_cfg)

    Step 3: Freeze Base Model
    ------------------------
    base_model.encoder.freeze()
    base_model.decoder.freeze()
    base_model.joint.freeze()

    Step 4: Fine-tune Adapter
    -------------------------
    base_model.set_active_adapter('medical')

    # Setup domain-specific data
    base_model.cfg.train_ds.manifest_filepath = 'medical_train.json'
    base_model.setup_training_data(base_model.cfg.train_ds)

    # Train (only adapter parameters updated)
    trainer = pl.Trainer(devices=4, max_epochs=10)
    trainer.fit(base_model)

    Step 5: Save Adapter
    -------------------
    base_model.save_adapters('medical_adapter.nemo', 'medical')

    ========================================================================
    """
    pass


# ============================================================================
# PERFORMANCE OPTIMIZATION TIPS
# ============================================================================

class OPTIMIZATION_TIPS:
    """
    ========================================================================
    INFERENCE OPTIMIZATION
    ========================================================================

    1. CUDA Graphs
    --------------
    - Enable for greedy decoding
    - 10-100x speedup
    - Fixed input size required

    Usage:
        model.encoder.set_max_audio_length(max_len)
        model.decoder.cuda_graphs_enable()

    2. Batch Size
    -------------
    - Larger batch = better GPU utilization
    - Limited by GPU memory
    - Typical: 8-64 depending on model size

    3. Mixed Precision
    ------------------
    - Use float16 instead of float32
    - 2x speedup, 2x memory reduction
    - Minimal accuracy loss

    Usage:
        model.half()  # Convert to FP16

    4. TorchScript
    --------------
    - JIT compile for faster inference
    - Remove Python overhead

    Usage:
        scripted_model = torch.jit.script(model)

    5. ONNX Export
    --------------
    - Deploy to non-PyTorch environments
    - Use TensorRT for optimization

    Usage:
        model.export('model.onnx')

    ========================================================================
    TRAINING OPTIMIZATION
    ========================================================================

    1. Distributed Training
    ----------------------
    - Multi-GPU: Use DDP (DistributedDataParallel)
    - Multi-node: Scale to many GPUs

    Config:
        trainer:
          devices: 8
          num_nodes: 4
          strategy: ddp

    2. Gradient Accumulation
    ------------------------
    - Simulate larger batch size
    - Useful when GPU memory limited

    Config:
        trainer:
          accumulate_grad_batches: 4

    3. Mixed Precision Training
    ---------------------------
    - Use AMP (Automatic Mixed Precision)
    - Faster training, less memory

    Config:
        trainer:
          precision: 16

    4. Data Loading
    ---------------
    - Use multiple workers
    - Prefetch data
    - Use tarred datasets for very large scale

    Config:
        train_ds:
          num_workers: 8
          pin_memory: true

    5. Spec Augmentation Optimization
    ---------------------------------
    - Use Numba-accelerated version
    - Apply on GPU, not CPU

    ========================================================================
    MEMORY OPTIMIZATION
    ========================================================================

    1. Gradient Checkpointing
    ------------------------
    - Trade compute for memory
    - Recompute activations during backward

    Usage:
        model.encoder.use_checkpoint = True

    2. Smaller Batch Size
    ---------------------
    - Reduce batch size
    - Use gradient accumulation to maintain effective batch size

    3. Model Parallelism
    -------------------
    - Split model across GPUs
    - For very large models

    4. Activation Checkpointing
    --------------------------
    - Don't store all intermediate activations
    - Recompute when needed

    ========================================================================
    """
    pass


# ============================================================================
# SUMMARY
# ============================================================================

SUMMARY = """
================================================================================
NEMO ASR ARCHITECTURE - QUICK REFERENCE
================================================================================

MAIN MODEL TYPES:
    1. CTC Models: EncDecCTCModel, EncDecCTCModelBPE
    2. RNN-T Models: EncDecRNNTModel, EncDecRNNTBPEModel
    3. Hybrid Models: EncDecHybridRNNTCTCModel

CORE COMPONENTS:
    - Encoders: ConformerEncoder (SOTA), SqueezeformerEncoder, ConvASREncoder
    - Decoders: RNNTDecoder (LSTM), StatelessTransducerDecoder
    - Preprocessors: AudioToMelSpectrogramPreprocessor
    - Tokenizers: BPE (SentencePiece), Character-level

DECODING STRATEGIES:
    CTC:
        - Greedy: CTCGreedyDecoder
        - Beam: CTCBeamDecoder (with KenLM)

    RNN-T:
        - Greedy: GreedyBatchedRNNTInfer
        - Beam: BeamRNNTInfer (TSD, ALSD, MAES)

TRAINING:
    - Losses: CTCLoss, RNNTLoss (warprnnt_numba backend)
    - Optimizers: AdamW, Novograd
    - Augmentation: SpecAugment, Audio perturbations

DATA:
    - Datasets: AudioToCharDataset, AudioToBPEDataset
    - Formats: JSON manifest, Tarred datasets
    - Augmentation: Speed, Noise, Gain, Impulse, SpecAugment

ADVANCED FEATURES:
    - Streaming: Cache-aware models
    - Context Biasing: GPUBoostingTreeModel
    - Adapters: Parameter-efficient fine-tuning
    - Multi-blank: Improved alignment
    - TDT: Token-Duration Transducer

KEY FILES:
    Models: nemo/collections/asr/models/
    Modules: nemo/collections/asr/modules/
    Data: nemo/collections/asr/data/
    Losses: nemo/collections/asr/losses/
    Decoding: nemo/collections/asr/parts/submodules/
    Preprocessing: nemo/collections/asr/parts/preprocessing/

================================================================================
END OF NEMO ASR ARCHITECTURE GUIDE
================================================================================
"""

if __name__ == "__main__":
    print(__doc__)
    print(DIRECTORY_STRUCTURE)
    print(SUMMARY)
