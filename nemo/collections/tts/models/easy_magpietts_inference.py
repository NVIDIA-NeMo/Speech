# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.modules.magpietts_modules import (
    CharAwareSubwordEncoder,
    CodecHelper,
    LocalTransformerHelper,
    LocalTransformerType,
    SpecialAudioToken,
    add_special_tokens,
)
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging


@dataclass
class TrainingMode:
    """
    Configuration for a training mode in multi-mode training.
    We can configure our model to have different delays for phoneme and speech streams in each mode.
    During training, we choose one of the modes randomly for each batch.
    For inference, we can specify the inference mode to use.

    Attributes:
        text_input_mode: Either "full" or "streaming"
        streaming_phonemes_delay: Delay for phoneme stream (only used in streaming mode)
        streaming_speech_delay: Delay for speech stream (only used in streaming mode)
        mode_idx: Index of this mode in the list of modes (used for task embedding lookup)
    """

    text_input_mode: str
    streaming_phonemes_delay: int
    streaming_speech_delay: int
    mode_idx: int

    @property
    def name(self) -> str:
        """Derived identifier used for inference selection and logging."""
        return f"{self.text_input_mode}_{self.streaming_phonemes_delay}_{self.streaming_speech_delay}"


@dataclass
class StreamingConfig:
    """
    Static configuration for streaming TTS inference, set once during streaming_init.

    Attributes:
        batch_size: Number of items in the batch.
        device: Device tensors are on.
        training_mode: The training mode being used for inference.
        use_cfg: Whether classifier-free guidance is enabled.
        cfg_scale: CFG scale factor.
        use_local_transformer: Whether to use local transformer for inference.
        temperature: Sampling temperature.
        topk: Top-k sampling parameter.
        phoneme_input_type: 'gt' or 'pred' for phoneme tokens.
        phoneme_sampling_method: 'argmax' or 'sample' for phoneme token selection.
        dummy_context_embedding_unconditional: Unconditional embedding for CFG (if enabled).
    """

    batch_size: int
    device: torch.device
    training_mode: TrainingMode
    use_cfg: bool
    cfg_scale: float
    use_local_transformer: bool
    temperature: float
    topk: int
    phoneme_input_type: str
    phoneme_sampling_method: str
    dummy_context_embedding_unconditional: Optional[torch.Tensor]


@dataclass
class StreamingState:
    """
    Mutable state for streaming TTS inference with batch support.

    This dataclass maintains the dynamically changing state for autoregressive
    streaming generation. Static configuration lives in StreamingConfig (accessed
    via the `config` field).

    The streaming operates in four phases (per batch item):
    1. Context phase (context_position < full_context_lens): Processing remaining context
    2. Prompt phase (text_tokens_seen < phoneme_delay): Only text, no predictions
    3. Phoneme-only phase (phoneme_delay <= text_tokens_seen < speech_delay): Phoneme predictions only
    4. Audio phase (text_tokens_seen >= speech_delay): Both phoneme and audio predictions
    """

    config: StreamingConfig
    past_key_values: Optional[Tuple]
    cache_seq_len: int
    all_predictions: List[torch.Tensor]
    all_phoneme_predictions: List[torch.Tensor]
    context_audio_codes: torch.Tensor
    context_audio_codes_lens: torch.Tensor
    context_lens: torch.Tensor
    full_context_embedding: torch.Tensor
    full_context_lens: torch.Tensor
    context_position: torch.Tensor
    text_tokens_seen: torch.Tensor
    phoneme_steps: torch.Tensor
    audio_steps: torch.Tensor
    phoneme_stream_ended: torch.Tensor
    phoneme_eos_detected: torch.Tensor
    finished: torch.Tensor
    last_hidden: torch.Tensor
    text_finished: torch.Tensor
    last_phoneme_tokens: Optional[torch.Tensor]
    last_audio_codes: Optional[torch.Tensor]
    audio_prediction_start_idx: torch.Tensor
    audio_prediction_end_idx: torch.Tensor
    phoneme_prediction_start_idx: torch.Tensor
    phoneme_prediction_end_idx: torch.Tensor
    gt_phoneme_embeddings: Optional[torch.Tensor] = None  # (B, T', E) pre-computed GT embeddings
    gt_phoneme_lens: Optional[torch.Tensor] = None  # (B,) lengths after stacking
    gt_audio_embeddings: Optional[torch.Tensor] = None  # (B, T', E) pre-computed GT audio embeddings
    gt_audio_lens: Optional[torch.Tensor] = None  # (B,) lengths after stacking


@dataclass
class StreamingFinalizeOutput:
    """Output from streaming_finalize containing audio and phoneme predictions."""

    audio: torch.Tensor  # (B, max_audio_len) generated audio waveform
    audio_len: torch.Tensor  # (B,) length of audio per batch item
    audio_codes: torch.Tensor  # (B, num_codebooks, T) generated audio codes
    audio_codes_len: torch.Tensor  # (B,) length of codes per batch item
    phoneme_tokens: List[List[int]]  # List of phoneme token sequences per batch item
    phoneme_text: List[str]  # Decoded phoneme strings per batch item


@dataclass
class InferBatchOutput:
    """Output dataclass for EasyMagpieTTS infer_batch method."""

    predicted_audio: torch.Tensor  # (B, T_audio)
    predicted_audio_lens: torch.Tensor  # (B,)
    predicted_codes: torch.Tensor  # (B, num_codebooks, T_frames)
    predicted_codes_lens: torch.Tensor  # (B,)
    rtf_metrics: Dict[str, Any]
    predicted_phoneme_tokens: Optional[torch.Tensor] = None  # (B, phoneme_stacking_factor, T_phoneme_steps)
    predicted_phoneme_tokens_lens: Optional[torch.Tensor] = None  # (B,) number of valid phoneme steps per item
    phoneme_prediction_start_idx: Optional[torch.Tensor] = None  # (B,) start index into predicted_phoneme_tokens


@dataclass
class EasyModelInferenceParameters:
    """Inference parameters for the decoder-only EasyMagpieTTS model.

    Attributes:
        max_decoder_steps: Maximum number of decoder steps.
        temperature: Sampling temperature.
        topk: Number of top-probability tokens to consider in sampling.
        cfg_scale: Scale factor for classifier-free guidance.
    """

    max_decoder_steps: int = 300
    temperature: float = 0.7
    topk: int = 80
    cfg_scale: float = 2.5

    @classmethod
    def from_dict(cls, data: dict) -> 'EasyModelInferenceParameters':
        field_names = {field.name for field in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


class EasyMagpieTTSInferenceModel(ModelPT):
    """
    Inference-only base class for EasyMagpieTTS decoder-only model.

    Contains the model architecture (codec, embeddings, decoder, local transformer),
    shared building-block methods, and all inference methods (streaming_init,
    streaming_step, streaming_finalize, infer_batch, do_tts).

    EasyMagpieTTSModel subclasses this to add training, validation, and data loading.
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # load codec
        codec_model = AudioCodecModel.restore_from(cfg.get('codecmodel_path'), strict=False)
        self.sample_rate = codec_model.sample_rate
        self.output_sample_rate = codec_model.output_sample_rate

        if hasattr(codec_model, "discriminator"):
            # del codec discriminator to free memory
            del codec_model.discriminator

        # Set up codebook configuration
        vector_quantizer = cfg.get('vector_quantizer')
        if vector_quantizer is not None:
            vector_quantizer = instantiate(vector_quantizer)
            num_audio_codebooks = vector_quantizer.num_codebooks
            codebook_size = vector_quantizer.codebook_size
            codec_converter = VectorQuantizerIndexConverter(
                vector_quantizer_original=codec_model.vector_quantizer,
                vector_quantizer_new=vector_quantizer,
            )
            data_num_audio_codebooks = codec_model.vector_quantizer.num_codebooks
        else:
            num_audio_codebooks = codec_model.num_codebooks
            data_num_audio_codebooks = num_audio_codebooks
            codebook_size = codec_model.codebook_size
            codec_converter = None

        # The dataloader needs to know the number of codebooks that the context codes were stored in
        # In the case where there are no context codes saved, and there is no context audio (in the text context path),
        # We create a dummy context code tensor that is only [context_BOS, context_EOS] that is repeated for
        # data_num_audio_codebooks
        self.data_num_audio_codebooks = data_num_audio_codebooks
        self.num_audio_codebooks = num_audio_codebooks
        self.codebook_size = codebook_size

        self.codec_model_samples_per_frame = codec_model.samples_per_frame
        # Our codebooks start with actual audio codec tokens, followed by special tokens.
        # The `forced_*` options are for backward compatibility for models trained with older code.
        get_token_index = partial(SpecialAudioToken.get_index, base_codebook_size=self.codebook_size)
        self.audio_bos_id = get_token_index(SpecialAudioToken.AUDIO_BOS)
        self.audio_eos_id = get_token_index(SpecialAudioToken.AUDIO_EOS)
        self.context_audio_bos_id = get_token_index(SpecialAudioToken.AUDIO_CONTEXT_BOS)
        self.context_audio_eos_id = get_token_index(SpecialAudioToken.AUDIO_CONTEXT_EOS)
        self.mask_token_id = get_token_index(SpecialAudioToken.MASK_TOKEN)
        self.num_all_tokens_per_codebook = self.codebook_size + len(SpecialAudioToken)
        self.use_bpe_char_tokenizer = cfg.get('use_bpe_char_tokenizer', False)

        # If specified, use this as the text conditioning tokenizer. Otherwise, use the first tokenizer.
        self.text_conditioning_tokenizer_name = cfg.get('text_conditioning_tokenizer_name', None)
        if self.text_conditioning_tokenizer_name is None:
            self.text_conditioning_tokenizer_name = list(cfg.text_tokenizers.keys())[0]

        self.cfg_unconditional_prob = cfg.get('cfg_unconditional_prob', 0.0)

        # Multi-mode training configuration
        # The model trains with multiple text input modes (full, streaming with various delays)
        # Each mode has its own task embedding that is prepended to the context
        training_modes_cfg = cfg.get('training_modes', None)
        if training_modes_cfg is None:
            # Create a default training mode for backward compatibility
            self.training_modes = [
                TrainingMode(
                    text_input_mode="streaming",
                    streaming_phonemes_delay=4,
                    streaming_speech_delay=8,
                    mode_idx=0,
                )
            ]

        else:
            self.training_modes = []
            for mode_idx, mode_cfg in enumerate(training_modes_cfg):
                mode = TrainingMode(
                    text_input_mode=mode_cfg.text_input_mode,
                    streaming_phonemes_delay=mode_cfg.get('streaming_phonemes_delay', 0),
                    streaming_speech_delay=mode_cfg.get('streaming_speech_delay', 0),
                    mode_idx=mode_idx,
                )
                self.training_modes.append(mode)

        logging.info(f"Multi-mode training with {len(self.training_modes)} modes:")
        for mode in self.training_modes:
            logging.info(
                f"  - {mode.name}: text_input_mode={mode.text_input_mode}, "
                f"streaming_phonemes_delay={mode.streaming_phonemes_delay}, "
                f"streaming_speech_delay={mode.streaming_speech_delay}"
            )

        # Create a mapping from mode name to mode object for easy lookup during inference
        self.mode_name_to_mode = {mode.name: mode for mode in self.training_modes}
        # Default mode for inference if not specified (first mode in the list)
        self.default_inference_mode = self.training_modes[0].name

        self.frame_stacking_factor = cfg.get('frame_stacking_factor', 1)

        self.tokenizer = setup_tokenizers(
            all_tokenizers_config=cfg.text_tokenizers,
            mode='train',
        )

        num_tokens_tokenizer = len(self.tokenizer.tokens)
        num_tokens = num_tokens_tokenizer + 3  # +3 for BOS, EOS, CFG_UNK
        self.bos_id = num_tokens - 3
        self.eos_id = num_tokens - 2
        self.cfg_unk_token_id = num_tokens - 1
        self.phoneme_tokenizer = None
        if cfg.get('phoneme_tokenizer', None) is not None:
            self.phoneme_tokenizer = instantiate(cfg.phoneme_tokenizer)
            self.phoneme_stacking_factor = cfg.get('phoneme_stacking_factor', 1)
            self.phoneme_vocab_size = self.phoneme_tokenizer.vocab_size
            if cfg.get('phoneme_corruption_batch_prob', None) is None:
                # Legacy mode: remove the UNK token from the phoneme vocabulary
                # TODO: Remove this.
                self.phoneme_vocab_size -= 1
            # If max phoneme probability is below this threshold at inference-time,
            # replace the predicted timestep with UNK to reduce error propagation.
            self.phoneme_confidence_unk_threshold = cfg.get('phoneme_confidence_unk_threshold', 0.0)

        self.pad_context_text_to_max_duration = False
        self.add_language_to_context_text = cfg.get('add_language_to_context_text', False)
        self.ignore_phoneme_languages = cfg.get('ignore_phoneme_languages', [])

        super().__init__(cfg=cfg, trainer=trainer)

        # This needs to happen after super().__init__()
        self._codec_model = codec_model
        self._codec_model.freeze()  # Lightning does requires_grad = False and self.eval()
        self._codec_converter = codec_converter
        self._codec_helper = CodecHelper(self._codec_model, self._codec_converter)

        # Audio embedding dimension - can be smaller than hidden_dim to reduce parameters
        self.audio_embedding_dim = cfg.get('audio_embedding_dim', cfg.hidden_dim)

        audio_embeddings = []
        for _ in range(self.num_audio_codebooks * self.frame_stacking_factor):
            audio_embeddings.append(nn.Embedding(self.num_all_tokens_per_codebook, self.audio_embedding_dim))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        # Projection from audio_embedding_dim to embedding_dim (Identity if same)
        if self.audio_embedding_dim != cfg.embedding_dim:
            self.audio_in_projection = nn.Linear(self.audio_embedding_dim, cfg.embedding_dim)
        else:
            self.audio_in_projection = nn.Identity()

        if self.phoneme_tokenizer is not None:
            phoneme_embeddings = []
            for _ in range(self.phoneme_stacking_factor):
                phoneme_embeddings.append(nn.Embedding(self.phoneme_vocab_size, cfg.embedding_dim))
            self.phoneme_embeddings = nn.ModuleList(phoneme_embeddings)
            self.phoneme_final_proj = nn.Linear(cfg.hidden_dim, self.phoneme_vocab_size * self.phoneme_stacking_factor)

        # Decoder backend selection - supports HuggingFace models or NemotronH
        self.decoder_type = cfg.get('decoder_type', 'huggingface')  # backward compatible default
        logging.info(f"Using decoder type: {self.decoder_type}")

        if self.decoder_type == 'huggingface':
            # Existing HuggingFace path
            self.transformer_backend_config = AutoConfig.from_pretrained(
                cfg.transformer_hf_backend,
                trust_remote_code=True,
            )

            # This is set True during inference to save time on loading the model.
            use_meta = cfg.get('use_meta_init_for_decoder', False)
            if use_meta:
                logging.info("Using meta device for decoder init (weights will be loaded from checkpoint)")
                with torch.device('meta'):
                    hf_transformer = AutoModelForCausalLM.from_config(self.transformer_backend_config)
                hf_transformer = hf_transformer.to_empty(device='cpu')
                for module in hf_transformer.modules():
                    if 'RotaryEmbedding' in type(module).__name__ and hasattr(module, 'config'):
                        type(module).__init__(module, module.config, device='cpu')
            else:
                hf_transformer = AutoModelForCausalLM.from_config(self.transformer_backend_config)
            self.decoder = hf_transformer.model
            self.lm_text_head = hf_transformer.lm_head

        elif self.decoder_type == 'nemotron_h':
            # NemotronH hybrid Mamba2/Attention backend
            from nemo.collections.tts.modules.nemotron_h_decoder import NemotronHConfig, NemotronHForCausalLM

            # Build config from YAML parameters
            nemotron_h_config_dict = dict(cfg.get('nemotron_h_config', {}))
            # Ensure hidden_size matches embedding_dim for compatibility
            if 'hidden_size' not in nemotron_h_config_dict:
                nemotron_h_config_dict['hidden_size'] = cfg.embedding_dim
            nemotron_config = NemotronHConfig(**nemotron_h_config_dict)
            nemotron_model = NemotronHForCausalLM(nemotron_config)
            self.decoder = nemotron_model.backbone
            self.lm_text_head = nemotron_model.lm_head
            logging.info(
                f"NemotronH config: {nemotron_config.num_hidden_layers} layers, pattern={nemotron_config.hybrid_override_pattern[:20]}..."
            )

        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}. Supported: 'huggingface', 'nemotron_h'")

        self.text_embedding = nn.Embedding(num_tokens, cfg.embedding_dim)
        self.decoder.set_input_embeddings(self.text_embedding)

        # Task embedding for multi-mode training
        # Each mode has a unique task embedding that is prepended to the context
        # Only create task embedding if there are multiple modes
        num_modes = len(self.training_modes)
        if num_modes > 1:
            self.task_embedding = nn.Embedding(num_modes, cfg.embedding_dim)
            logging.info(f"Created task embedding with {num_modes} modes, embedding_dim={cfg.embedding_dim}")
        else:
            self.task_embedding = None
            logging.info(f"Single training mode '{self.training_modes[0].name}', skipping task embedding")

        if self.use_bpe_char_tokenizer:
            # BPE char tokenizer
            assert len(self.tokenizer.tokenizers) == 1, "BPE char tokenizer should only be used with one tokenizer"
            tokenizer_name = self.tokenizer.tokenizer_names[0]
            tokenizer = self.tokenizer.tokenizers[tokenizer_name]
            subword_vocab = tokenizer.get_vocab()
            # special tokens will be stored as it is in the char_vocab
            # Each special token will only be mapped to one char id
            special_vocab = {
                '<BOS>': self.bos_id,
                '<EOS>': self.eos_id,
                '<CFG_UNK>': self.cfg_unk_token_id,
            }
            self.cas_encoder = CharAwareSubwordEncoder(
                d_embed=cfg.embedding_dim,
                llm_tokenizer_vocab=subword_vocab,
                subword_padding_idx=self.tokenizer.pad,
                special_vocab=special_vocab,
            )

        # Projection from hidden_dim to audio_embedding_dim before final_proj (Identity if same)
        if self.audio_embedding_dim != cfg.hidden_dim:
            self.audio_out_projection = nn.Linear(cfg.hidden_dim, self.audio_embedding_dim)
        else:
            self.audio_out_projection = nn.Identity()

        self.final_proj = nn.Linear(
            self.audio_embedding_dim,
            self.num_audio_codebooks * self.num_all_tokens_per_codebook * self.frame_stacking_factor,
        )

        self.local_transformer_type = LocalTransformerType(cfg.get('local_transformer_type', 'none').lower())
        logging.info(f"Local transformer type: {self.local_transformer_type}")
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            local_transformer_hidden_dim = cfg.get('local_transformer_hidden_dim', 256)
            if local_transformer_hidden_dim != cfg.hidden_dim:
                self.local_transformer_in_projection = nn.Linear(cfg.hidden_dim, local_transformer_hidden_dim)
            else:
                self.local_transformer_in_projection = nn.Identity()
            self.local_transformer = transformer_2501.Transformer(
                n_layers=self.cfg.get('local_transformer_n_layers', 2),
                d_model=local_transformer_hidden_dim,
                d_ffn=local_transformer_hidden_dim * 4,
                sa_n_heads=self.cfg.get('local_transformer_n_heads', 1),
                kernel_size=1,
                is_causal=self.local_transformer_type == LocalTransformerType.AR,
                max_length_causal_mask=self.num_audio_codebooks * self.frame_stacking_factor + 2,
                use_learnable_pos_emb=True,
            )
            # Projection from local_transformer_hidden_dim to audio_embedding_dim (Identity if same)
            if self.audio_embedding_dim != local_transformer_hidden_dim:
                self.local_transformer_audio_out_projection = nn.Linear(
                    local_transformer_hidden_dim, self.audio_embedding_dim
                )
            else:
                self.local_transformer_audio_out_projection = nn.Identity()
            local_transformer_out_projections = []
            for _ in range(self.num_audio_codebooks * self.frame_stacking_factor):
                # Have a separate projection layer for each codebook, to distinguish between them
                local_transformer_out_projections.append(
                    nn.Linear(self.audio_embedding_dim, self.num_all_tokens_per_codebook)
                )
            self.local_transformer_out_projections = nn.ModuleList(local_transformer_out_projections)

            # EasyMagpie stacks frames into the channel dimension (B, C*S, T_stacked)
            # via stack_codes, unlike Magpie which keeps them interleaved in time (B, C, T_full).
            # We pass num_audio_codebooks=C*S and frame_stacking_factor=1 so the helper
            # treats each stacked channel as an independent codebook without time-domain striding.
            self._lt_helper = LocalTransformerHelper(
                local_transformer=self.local_transformer,
                audio_embeddings=self.audio_embeddings,
                audio_in_projection=self.audio_in_projection,
                local_transformer_in_projection=self.local_transformer_in_projection,
                local_transformer_audio_out_projection=self.local_transformer_audio_out_projection,
                local_transformer_out_projections=self.local_transformer_out_projections,
                num_audio_codebooks=self.num_audio_codebooks * self.frame_stacking_factor,
                frame_stacking_factor=1,
                audio_eos_id=self.audio_eos_id,
                mask_token_id=self.mask_token_id,
                codebook_size=self.codebook_size,
            )

    def _get_state_dict_keys_to_exclude(self) -> List[str]:
        return [
            '_codec_model',
        ]

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if hasattr(self, '_no_state_dict') and self._no_state_dict:
            return {}
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_substrings_to_exclude = self._get_state_dict_keys_to_exclude()
        for key in list(state_dict.keys()):
            if any(substring in key for substring in keys_substrings_to_exclude):
                del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        if not strict:
            super().load_state_dict(state_dict, strict=False)
        modules_to_skip = self._get_state_dict_keys_to_exclude()
        for name, child in self.named_children():
            if name in modules_to_skip:
                continue
            if any(param.numel() > 0 for param in child.parameters()):
                new_state_dict = {}
                for key in state_dict.keys():
                    name_with_dot = f"{name}."
                    if key.startswith(name_with_dot):
                        new_state_dict[key[len(name_with_dot) :]] = state_dict[key]
                child.load_state_dict(new_state_dict)

    def setup_optimizer_param_groups(self):
        """Exclude frozen eval/inference-only models from the optimizer."""
        modules_to_exclude = set(self._get_state_dict_keys_to_exclude())

        excluded_param_ids = set()
        for name, module in self.named_children():
            if name in modules_to_exclude:
                for param in module.parameters():
                    excluded_param_ids.add(id(param))

        trainable_params = [p for p in self.parameters() if id(p) not in excluded_param_ids]

        logging.info(
            f"setup_optimizer_param_groups: {len(trainable_params)} params in optimizer, "
            f"{len(excluded_param_ids)} params excluded (eval models)"
        )

        self._optimizer_param_groups = [{"params": trainable_params}]

    def setup_training_data(self, train_data_config=None):
        pass

    def setup_validation_data(self, val_data_config=None):
        pass

    def _prepare_codes_for_decode(self, codes, codes_len, min_len=4):
        """Unstack frame-stacked codes and pad short sequences before decoding."""
        if self.frame_stacking_factor > 1 and codes.size(1) == self.num_audio_codebooks * self.frame_stacking_factor:
            codes, codes_len = self.unstack_codes(codes, codes_len, self.frame_stacking_factor)
        if min_len > 0 and codes_len.min() < min_len:
            codes = torch.nn.functional.pad(input=codes, pad=(0, min_len - codes_len.min()), value=0)
            codes_len = torch.where(codes_len < min_len, torch.ones_like(codes_len) * min_len, codes_len)
            codes = codes[:, :, : codes_len.max()]
        return codes, codes_len

    def embed_audio_tokens(self, audio_tokens):
        # audio_tokens: (B, C, T')
        # Add and average the embeddings of the audio tokens across the codebooks
        audio_embedding = None
        for c in range(audio_tokens.size(1)):
            embedding = self.audio_embeddings[c](audio_tokens[:, c, :])
            if audio_embedding is None:
                audio_embedding = embedding
            else:
                audio_embedding = audio_embedding + embedding
        audio_embedding = audio_embedding / audio_tokens.size(1)
        # Project from audio_embedding_dim to embedding_dim
        audio_embedding = self.audio_in_projection(audio_embedding)
        return audio_embedding

    def embed_phoneme_tokens(self, phoneme_tokens):
        # phoneme_tokens: (B, S, T')
        phoneme_embedding = None
        for c in range(phoneme_tokens.size(1)):
            embedding = self.phoneme_embeddings[c](phoneme_tokens[:, c, :])
            if phoneme_embedding is None:
                phoneme_embedding = embedding
            else:
                phoneme_embedding = phoneme_embedding + embedding
        phoneme_embedding = phoneme_embedding / phoneme_tokens.size(1)
        return phoneme_embedding

    def forward(self, inputs_embeds, attention_mask, use_cache=False, past_key_values=None, cache_position=None):
        # Only pass cache_position for NemotronH (HF transformers may not accept it)
        if self.decoder_type == 'nemotron_h':
            backend_out = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
        else:
            backend_out = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
        return backend_out

    def logits_to_audio_codes(self, all_code_logits, audio_codes_lens):
        # all_code_logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # audio_codes_lens: (B,)
        all_preds = []
        for idx in range(self.num_audio_codebooks * self.frame_stacking_factor):
            si = idx * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = all_code_logits[:, :, si:ei]
            codebook_probs = torch.softmax(codebook_logits, dim=-1)  # (B, T', num_tokens_per_codebook)
            # argmax to get the tokens
            codebook_preds = torch.argmax(codebook_probs, dim=-1)  # (B, T')
            all_preds.append(codebook_preds)

        all_preds = torch.stack(all_preds, dim=1)  # (B, C, T')
        audio_mask = get_mask_from_lengths(audio_codes_lens)
        all_preds = all_preds * audio_mask.unsqueeze(1)

        return all_preds

    def sample_codes_from_logits(
        self, all_code_logits_t, temperature=0.7, topk=80, unfinished_items={}, finished_items={}
    ):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = []
        for idx in range(self.num_audio_codebooks * self.frame_stacking_factor):
            si = idx * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)
            # Replace NaN/inf then clamp to prevent extreme values causing NaN in softmax
            codebook_logits = torch.nan_to_num(codebook_logits, nan=0.0, posinf=100.0, neginf=-100.0)
            codebook_logits = codebook_logits.clamp(min=-100.0, max=100.0)
            for item_idx in unfinished_items:
                codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                codebook_logits[item_idx, :] = float('-inf')
                codebook_logits[item_idx, self.audio_eos_id] = 0.0
            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                -1
            )  # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            if temperature <= 0.0:
                # Argmax sampling for deterministic output
                codebook_preds = codebook_logits_rescored.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                codebook_probs = torch.softmax(
                    codebook_logits_rescored / temperature, dim=-1
                )  # (B, num_tokens_per_codebook)
                codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
            all_preds.append(codebook_preds)
        all_preds = torch.cat(all_preds, dim=1).long()  # (B, num_codebooks)
        return all_preds

    def sample_codes_from_logits_phoneme(self, all_code_logits_t, temperature=0.7, topk=80):
        # all_code_logits_t: (B, phoneme_stacking_factor * phoneme_vocab_size), logits at a given timestep
        all_preds = []
        for idx in range(self.phoneme_stacking_factor):
            si = idx * self.phoneme_vocab_size
            ei = si + self.phoneme_vocab_size
            codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)
            # Replace NaN/inf then clamp to prevent extreme values causing NaN in softmax
            codebook_logits = torch.nan_to_num(codebook_logits, nan=0.0, posinf=100.0, neginf=-100.0)
            codebook_logits = codebook_logits.clamp(min=-100.0, max=100.0)
            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                -1
            )  # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            if temperature <= 0.0:
                # Argmax sampling for deterministic output
                codebook_preds = codebook_logits_rescored.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                codebook_probs = torch.softmax(
                    codebook_logits_rescored / temperature, dim=-1
                )  # (B, num_tokens_per_codebook)
                codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
            all_preds.append(codebook_preds)
        all_preds = torch.cat(all_preds, dim=1).long()  # (B, num_codebooks)
        return all_preds

    def join_embeddings_temporally(
        self,
        embeddings: Sequence[torch.Tensor],  # [ (B, Ti, E), … ]
        lengths: Sequence[torch.Tensor],  # [ (B,), … ]  same order/size as `embeddings`
        pad_embed: torch.Tensor | None = None,  # (E,)  defaults to zeros
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merges Multiple Embedding sequences into a single Embedding Sequence.

        Args:
            embeddings  : Sequence of tensors, each of shape (B, Ti, E) — batch, time, embedding
            lengths     : Sequence of tensors, each of shape (B,)
            pad_embed   : (E,)  — embedding to use for padding, defaults to zeros

        Returns:
            joined      : (B, max_sum_len, E)  — merged & padded
            out_lengths : (B,)  — total lengths of each batch element after merging
        """
        if len(embeddings) == 0:
            raise ValueError("contexts must be non-empty")

        B, _, E = embeddings[0].shape
        device = embeddings[0].device
        dtype = embeddings[0].dtype

        # 1. compute output sizes
        len_stack = torch.stack(tuple(lengths), dim=0)  # (N, B)
        out_lengths = len_stack.sum(0)
        max_len = int(out_lengths.max())

        if pad_embed is None:
            pad_embed = torch.zeros(E, dtype=dtype, device=device)

        joined = pad_embed.expand(B, max_len, E).clone()  # (B,max_len,E)

        # batch row indices
        batch_rows = torch.arange(B, device=device).unsqueeze(1)  # (B,1)

        # running offset keeps "write cursor" for each row
        offset = torch.zeros(B, dtype=torch.long, device=device)  # (B,)

        for i, (embedding_i, len_i) in enumerate(zip(embeddings, lengths)):
            Ti = embedding_i.shape[1]
            t_idx = torch.arange(Ti, device=device)  # (Ti,)
            mask = t_idx.unsqueeze(0) < len_i.unsqueeze(1)  # (B,Ti)

            # destination columns: offset + t
            dest_cols = offset.unsqueeze(1) + t_idx  # (B,Ti)

            # Assign embedding_i to the correct positions in joined
            # Ensure dtype matches to avoid errors during mixed-precision training
            joined[batch_rows.expand_as(mask)[mask], dest_cols[mask]] = embedding_i[mask].to(joined.dtype)

            # move cursor past this segment
            offset += len_i

        return joined, out_lengths

    def prepare_context_tensors(
        self,
        context_text_tokens: torch.Tensor,
        context_text_tokens_lens: torch.Tensor,
        context_audio_codes: Optional[torch.Tensor] = None,
        context_audio_codes_lens: Optional[torch.Tensor] = None,
        context_audio: Optional[torch.Tensor] = None,
        context_audio_lens: Optional[torch.Tensor] = None,
        training_mode: Optional[TrainingMode] = None,
        dropout_conditional_input: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare context tensors (without text) for the simplified process_batch.

        This function processes context audio and context text to create the combined
        context embedding.
        Args:
            context_text_tokens: Context text token IDs for speaker/style conditioning (B, L)
            context_text_tokens_lens: Length of context text for each batch item (B,)
            context_audio_codes: Pre-computed audio codes for context audio (B, C, T').
                If None, will be computed from context_audio.
            context_audio_codes_lens: Length of context audio codes (B,).
                Required if context_audio_codes is provided.
            context_audio: Raw context audio waveform (B, T).
                Used to compute context_audio_codes if not provided.
            context_audio_lens: Length of context audio (B,).
                Required if context_audio is provided.
            training_mode: Optional TrainingMode object specifying the mode to use.
                If None, uses the first mode from training_modes as default.
            dropout_conditional_input: If True, replace context with CFG unconditional token.

        Returns:
            Tuple of:
                - context_embedding: Combined context embedding (B, T_context, E)
                - context_lens: Total context length per batch item (B,)
                - context_audio_codes: Processed audio codes with special tokens (B, C, T')
                - context_audio_codes_lens: Length of processed context audio codes (B,)
        """
        # Determine the mode parameters to use
        if training_mode is None:
            training_mode = self.training_modes[0]

        current_mode_idx = training_mode.mode_idx
        batch_size = context_text_tokens.size(0)
        device = context_text_tokens.device

        # Context Audio
        if context_audio_codes is None:
            if context_audio is None:
                raise ValueError("Either context_audio_codes or context_audio must be provided")
            context_audio_codes, context_audio_codes_lens = self._codec_helper.audio_to_codes(
                context_audio, context_audio_lens
            )

        if self._codec_converter is not None:
            context_audio_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=context_audio_codes, audio_lens=context_audio_codes_lens
            ).long()

        context_audio_codes, context_audio_codes_lens = add_special_tokens(
            codes=context_audio_codes,
            codes_len=context_audio_codes_lens,
            bos_id=self.context_audio_bos_id,
            eos_id=self.context_audio_eos_id,
        )

        context_audio_codes, context_audio_codes_lens = self.stack_codes(
            context_audio_codes,
            context_audio_codes_lens,
            self.context_audio_bos_id,
            self.context_audio_eos_id,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )
        context_audio_embedded = self.embed_audio_tokens(context_audio_codes)  # (B, T', E)

        # Context Text
        context_text_lens = context_text_tokens_lens
        context_text_embedded = self.decoder.get_input_embeddings()(context_text_tokens)  # (B, L, E)

        # Prepare task embedding for multi-mode training
        task_embedding = None
        task_embedding_lens = None
        if self.task_embedding is not None and current_mode_idx is not None:
            mode_idx_tensor = torch.full((batch_size,), current_mode_idx, dtype=torch.long, device=device)
            task_embedding = self.task_embedding(mode_idx_tensor).unsqueeze(1)  # (B, 1, E)
            task_embedding_lens = torch.ones(batch_size, dtype=torch.long, device=device)  # (B,)

        # Combine context embeddings: [task_embedding | context_audio | context_text]
        if task_embedding is not None:
            context_embedding, context_lens = self.join_embeddings_temporally(
                embeddings=[task_embedding, context_audio_embedded, context_text_embedded],
                lengths=[task_embedding_lens, context_audio_codes_lens, context_text_lens],
            )
        else:
            context_embedding, context_lens = self.join_embeddings_temporally(
                embeddings=[context_audio_embedded, context_text_embedded],
                lengths=[context_audio_codes_lens, context_text_lens],
            )

        # Handle CFG unconditional dropout
        if dropout_conditional_input:
            cfg_token_id = self.cfg_unk_token_id
            cfg_token_embedding = self.decoder.get_input_embeddings()(
                torch.full((batch_size, 1), cfg_token_id, device=device)
            )  # (B, 1, E)
            # Expand CFG token to match context embedding size
            context_embedding = cfg_token_embedding.expand(-1, context_embedding.size(1), -1)  # (B, T_context, E)

        return context_embedding, context_lens, context_audio_codes, context_audio_codes_lens

    def stack_codes(self, codes, codes_lens, bos_id, eos_id, stacking_factor, num_codebooks):
        """
        Stack multiple time steps into the channel dimension to reduce sequence length.

        This function reshapes audio/phoneme codes by grouping consecutive time steps together
        and placing them in the channel dimension. This allows the model to process multiple
        frames in parallel while reducing the sequence length.

        Args:
            codes: Input codes tensor of shape (B, C, T) where B is batch size,
                   C is number of codebooks, and T is sequence length.
            codes_lens: Length of valid codes for each batch item, shape (B,).
            bos_id: Beginning-of-sequence token ID used to detect and handle BOS tokens.
            eos_id: End-of-sequence token ID used for padding.
            stacking_factor: Number of time steps to stack together. If 1, no stacking is performed.
            num_codebooks: Number of codebooks in the input.

        Returns:
            Tuple of:
                - stacked_codes: Reshaped codes of shape (B, C * stacking_factor, T // stacking_factor).
                  If input contains BOS tokens, they are preserved at the beginning.
                - new_lens: Updated sequence lengths after stacking, shape (B,).
        """
        if stacking_factor == 1:
            return codes, codes_lens

        contains_bos = codes[0, 0, 0].item() == bos_id
        if contains_bos:
            bos_tensor_repeated = torch.full(
                (codes.size(0), (stacking_factor) * num_codebooks, 1), bos_id, device=codes.device
            )  # (B,stacking_factor*C, 1)
            codes = codes[:, :, 1:]  # Remove the bos token
            codes_lens = codes_lens - 1  # Remove the bos token
        B, C, T = codes.shape
        s = int(stacking_factor)

        # --- Compute max padding needed ---
        pad_t = (-T) % s  # pad so that T' is divisible by s
        pad_tail = torch.full((B, C, pad_t), eos_id, dtype=codes.dtype, device=codes.device)
        codes = torch.cat([codes, pad_tail], dim=-1)

        # --- Stack time into channel dimension ---
        Tp = codes.shape[-1]
        T_out = Tp // s
        codes = codes.view(B, C, T_out, s)
        codes = codes.permute(0, 1, 3, 2).reshape(B, C * s, T_out)

        new_lens = torch.div(codes_lens + s - 1, s, rounding_mode='floor')
        if contains_bos:
            codes = torch.cat([bos_tensor_repeated, codes], dim=2)
            new_lens = new_lens + 1

        return codes, new_lens

    def unstack_codes(self, stacked_codes, stacked_lens, stacking_factor):
        """
        Reverse the stacking operation to recover the original time dimension.

        This is the inverse of `stack_codes`. It takes codes that have been stacked
        in the channel dimension and expands them back into the time dimension.

        Args:
            stacked_codes: Stacked codes tensor of shape (B, C * stacking_factor, T_stacked)
                          where T_stacked = T_original // stacking_factor.
            stacked_lens: Length of valid stacked sequences for each batch item, shape (B,).
            stacking_factor: The stacking factor used in the original `stack_codes` call.
                            If 1, no unstacking is performed.

        Returns:
            Tuple of:
                - unstacked_codes: Codes with restored time dimension, shape (B, C, T_stacked * stacking_factor).
                - orig_lens: Recovered sequence lengths, shape (B,). Note that these are the
                  maximum possible lengths; actual valid lengths may be shorter due to
                  padding applied during stacking.
        """
        if stacking_factor == 1:
            return stacked_codes, stacked_lens

        B, CxS, T_out = stacked_codes.shape
        s = int(stacking_factor)
        assert CxS % s == 0, f"Channel dim ({CxS}) must be divisible by stacking_factor ({s})"

        C = CxS // s
        # Reshape: split channels back into (C, s)
        x = stacked_codes.view(B, C, s, T_out)
        # Bring s back into time dimension
        x = x.permute(0, 1, 3, 2).reshape(B, C, T_out * s)

        # Recover original lengths (before padding)
        orig_lens = stacked_lens * s

        return x, orig_lens

    def _sample_audio_codes(
        self,
        last_hidden: torch.Tensor,
        all_code_logits_t: torch.Tensor,
        temperature: float,
        topk: int,
        use_local_transformer_for_inference: bool,
        use_cfg: bool,
        cfg_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample audio codes from logits using either local transformer or parallel sampling.

        Returns:
            audio_codes_next: Sampled codes with temperature/topk (B, num_codebooks)
            all_codes_next_argmax: Argmax sampled codes for EOS detection (B, num_codebooks)
        """
        if use_local_transformer_for_inference:
            if self.local_transformer_type == LocalTransformerType.AR:
                audio_codes_next = self._lt_helper.sample_autoregressive(
                    dec_output=last_hidden[:, -1, :],
                    temperature=temperature,
                    topk=topk,
                    use_cfg=use_cfg,
                    cfg_scale=cfg_scale,
                    use_kv_cache=False,
                    sanitize_logits=True,
                )
                # Base class returns (B, C, S); flatten to (B, C*S) for downstream code
                audio_codes_next = audio_codes_next.permute(0, 2, 1)
                audio_codes_next = audio_codes_next.reshape(audio_codes_next.size(0), -1)
            else:
                raise ValueError(
                    f"Local transformer inference requested but local transformer type is {self.local_transformer_type}"
                )
            # TODO @rfejgin: should we add argmax sampling for EOS here too?
            all_codes_next_argmax = audio_codes_next
        else:
            # Parallel sampling from all codebook logits
            audio_codes_next = self.sample_codes_from_logits(all_code_logits_t, temperature=temperature, topk=topk)
            # Argmax sampling for reliable EOS detection
            if temperature <= 0.0:
                all_codes_next_argmax = audio_codes_next  # already argmax
            else:
                all_codes_next_argmax = self.sample_codes_from_logits(all_code_logits_t, temperature=0.01)

        return audio_codes_next, all_codes_next_argmax

    def streaming_init(
        self,
        context_audio_codes: torch.Tensor,
        context_audio_codes_lens: torch.Tensor,
        context_text_tokens: torch.Tensor,
        context_text_tokens_lens: torch.Tensor,
        inference_mode: Optional[str] = None,
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        use_local_transformer: bool = False,
        temperature: float = 0.7,
        topk: int = 80,
        phoneme_input_type: str = 'predicted',
        phoneme_sampling_method: str = 'argmax',
        gt_phoneme_tokens: Optional[torch.Tensor] = None,
        gt_phoneme_tokens_lens: Optional[torch.Tensor] = None,
        gt_audio_codes: Optional[torch.Tensor] = None,
        gt_audio_codes_lens: Optional[torch.Tensor] = None,
        use_inference_mode: bool = True,
    ) -> StreamingState:
        """
        Initialize streaming TTS inference state.

        This prepares the model for streaming inference by processing the context
        (audio + context text) and returning a StreamingState that can be used
        with streaming_step() to incrementally generate audio.

        Note: This function does NOT take the main text input. Text tokens are
        provided incrementally via streaming_step().

        For batched inference, each batch item can have a different context length.
        This function processes only up to the minimum context length across the batch,
        storing the remaining context to be processed in streaming_step's context phase.

        The streaming inference follows phases (per batch item):
        1. Context phase: Processing remaining context (if any) for items with longer context.
        2. Prompt phase: First `streaming_speech_delay` text tokens are processed
           without generating audio (building up context).
        3. Generation phase: Audio BOS is added and audio codes are generated
           autoregressively, with remaining text tokens added to audio embeddings.

        Args:
            context_audio_codes: Pre-computed audio codes for context audio (B, C, T').
            context_audio_codes_lens: Length of context audio codes (B,).
            context_text_tokens: Context text token IDs for speaker/style conditioning (B, L).
            context_text_tokens_lens: Length of context text (B,).
            inference_mode: Name of the inference mode to use (e.g., "streaming_4_8").
                If None, uses the default inference mode.
            use_cfg: Whether to use classifier-free guidance.
            cfg_scale: CFG scale factor (higher = stronger conditioning).
            use_local_transformer: Whether to use local transformer for AR sampling.
            temperature: Sampling temperature for audio codes.
            topk: Top-k sampling parameter.
            phoneme_input_type: 'gt' or 'predicted' for phoneme tokens (use 'predicted' for streaming).
            phoneme_sampling_method: 'argmax' or 'sample' for phoneme token selection.
            gt_phoneme_tokens: Optional GT phoneme tokens (B, L) with BOS/EOS for teacher forcing.
            gt_phoneme_tokens_lens: Lengths of GT phoneme tokens (B,).
            gt_audio_codes: Optional GT audio codes (B, C*S, T) already stacked with BOS/EOS,
                input portion ([:, :, :-1]) for teacher forcing. Pre-processed by caller.
            gt_audio_codes_lens: Lengths of GT audio codes (B,) after stacking.

        Returns:
            StreamingState: Initial state for streaming inference.
        """
        grad_ctx = torch.inference_mode if use_inference_mode else torch.no_grad
        with grad_ctx():
            batch_size = context_audio_codes.size(0)
            device = context_audio_codes.device

            # Resolve inference mode
            mode_name = inference_mode if inference_mode is not None else self.default_inference_mode
            if mode_name not in self.mode_name_to_mode:
                available_modes = list(self.mode_name_to_mode.keys())
                raise ValueError(f"Unknown inference mode '{mode_name}'. Available modes: {available_modes}")

            selected_training_mode = self.mode_name_to_mode[mode_name]

            # Prepare context embedding using shared helper
            context_embedding, context_lens, context_audio_codes, context_audio_codes_lens = (
                self.prepare_context_tensors(
                    context_text_tokens=context_text_tokens,
                    context_text_tokens_lens=context_text_tokens_lens,
                    context_audio_codes=context_audio_codes,
                    context_audio_codes_lens=context_audio_codes_lens,
                    training_mode=selected_training_mode,
                    dropout_conditional_input=False,
                )
            )

            # Store full context embedding and lens before any CFG manipulation
            full_context_embedding = context_embedding.clone()  # (B, T_max, E)
            full_context_lens = context_lens.clone()  # (B,)

            # Compute min context length - we only process up to this in init
            min_context_len = context_lens.min().item()

            # Setup classifier-free guidance if enabled
            dummy_context_embedding_unconditional = None
            if use_cfg:
                dummy_context_embedding_unconditional = self.decoder.get_input_embeddings()(
                    torch.full((1, 1), self.cfg_unk_token_id, device=device)
                )
                # Create unconditional context (same length as conditional)
                dummy_context_expanded = dummy_context_embedding_unconditional.expand(
                    batch_size, context_embedding.size(1), -1
                )
                # Concatenate conditional and unconditional: (2*B, T, E)
                context_embedding = torch.cat([context_embedding, dummy_context_expanded], dim=0)

            # First forward pass to process context - only up to min_context_len
            cache_position = torch.arange(min_context_len, device=device)
            transformer_out = self.forward(
                inputs_embeds=context_embedding[:, :min_context_len, :],
                attention_mask=None,
                use_cache=True,
                past_key_values=None,
                cache_position=cache_position,
            )

            last_hidden = transformer_out.last_hidden_state
            past_kv = transformer_out.past_key_values
            current_cache_seq_len = min_context_len

            # Process GT phoneme tokens if provided (for teacher forcing)
            gt_phoneme_embeddings = None
            gt_phoneme_lens = None
            if gt_phoneme_tokens is not None and gt_phoneme_tokens_lens is not None:
                gt_phoneme_expanded = gt_phoneme_tokens.unsqueeze(1)  # (B, 1, L)
                gt_phoneme_stacked, gt_phoneme_lens = self.stack_codes(
                    gt_phoneme_expanded,
                    gt_phoneme_tokens_lens,
                    self.phoneme_tokenizer.bos_token_id,
                    self.phoneme_tokenizer.eos_token_id,
                    self.phoneme_stacking_factor,
                    1,
                )
                gt_phoneme_embeddings = self.embed_phoneme_tokens(gt_phoneme_stacked)  # (B, T', E)

            # Process GT audio codes if provided (for teacher forcing)
            gt_audio_embeddings = None
            gt_audio_lens_state = None
            if gt_audio_codes is not None and gt_audio_codes_lens is not None:
                gt_audio_embeddings = self.embed_audio_tokens(gt_audio_codes)  # (B, T', E)
                gt_audio_lens_state = gt_audio_codes_lens

            # Initialize static config and mutable streaming state
            config = StreamingConfig(
                batch_size=batch_size,
                device=device,
                training_mode=selected_training_mode,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
                use_local_transformer=use_local_transformer,
                temperature=temperature,
                topk=topk,
                phoneme_input_type=phoneme_input_type,
                phoneme_sampling_method=phoneme_sampling_method,
                dummy_context_embedding_unconditional=dummy_context_embedding_unconditional,
            )

            state = StreamingState(
                config=config,
                past_key_values=past_kv,
                cache_seq_len=current_cache_seq_len,
                all_predictions=[],
                all_phoneme_predictions=[],
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                context_lens=context_lens,
                full_context_embedding=full_context_embedding,
                full_context_lens=full_context_lens,
                context_position=torch.full((batch_size,), min_context_len, dtype=torch.long, device=device),
                text_tokens_seen=torch.zeros(batch_size, dtype=torch.long, device=device),
                phoneme_steps=torch.zeros(batch_size, dtype=torch.long, device=device),
                audio_steps=torch.zeros(batch_size, dtype=torch.long, device=device),
                phoneme_stream_ended=torch.zeros(batch_size, dtype=torch.bool, device=device),
                phoneme_eos_detected=torch.zeros(batch_size, dtype=torch.bool, device=device),
                finished=torch.zeros(batch_size, dtype=torch.bool, device=device),
                last_hidden=last_hidden,
                text_finished=torch.zeros(batch_size, dtype=torch.bool, device=device),
                last_phoneme_tokens=None,
                last_audio_codes=None,
                audio_prediction_start_idx=torch.full((batch_size,), -1, dtype=torch.long, device=device),
                audio_prediction_end_idx=torch.full((batch_size,), -1, dtype=torch.long, device=device),
                phoneme_prediction_start_idx=torch.full((batch_size,), -1, dtype=torch.long, device=device),
                phoneme_prediction_end_idx=torch.full((batch_size,), -1, dtype=torch.long, device=device),
                gt_phoneme_embeddings=gt_phoneme_embeddings,
                gt_phoneme_lens=gt_phoneme_lens,
                gt_audio_embeddings=gt_audio_embeddings,
                gt_audio_lens=gt_audio_lens_state,
            )

            return state

    def streaming_step(
        self,
        state: StreamingState,
        text_tokens: Optional[torch.Tensor] = None,
        force_dropout_text: bool = False,
        use_inference_mode: bool = True,
    ) -> Tuple[StreamingState, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform one streaming inference step with batch support.

        Orchestrates three phases: (1) prepare the input embedding for this step,
        (2) run the transformer forward pass, (3) extract predictions and update state.

        Args:
            state: Current StreamingState from streaming_init or previous streaming_step.
            text_tokens: Next text token for each batch item, shape (B,), or None if text has finished.
                For items still in context phase, the text_token value is ignored (can be 0).
                When None is passed, the model continues generating until EOS.
            force_dropout_text: Whether to zero out text embeddings.
            use_inference_mode: Whether to use torch.inference_mode (vs torch.no_grad).

        Returns:
            Tuple of:
                - Updated StreamingState
                - Predicted audio codes for this step (B, C, S) unstacked, or None if no items in audio phase
                - Predicted phoneme tokens for this step (B, phoneme_stacking_factor) or None
        """
        if state.finished.all():
            return state, None, None

        grad_ctx = torch.inference_mode if use_inference_mode else torch.no_grad
        with grad_ctx():
            device = state.config.device

            # Phase 1: Prepare input embedding and determine per-item phase masks
            next_input, needs_context, needs_phoneme, needs_audio = self._prepare_streaming_input(
                state, text_tokens, force_dropout_text
            )

            # Phase 2: Transformer forward pass
            cache_position = torch.tensor([state.cache_seq_len], device=device)
            transformer_out = self.forward(
                inputs_embeds=next_input,
                attention_mask=None,
                use_cache=True,
                past_key_values=state.past_key_values,
                cache_position=cache_position,
            )

            state.last_hidden = transformer_out.last_hidden_state
            state.past_key_values = transformer_out.past_key_values
            state.cache_seq_len += 1

            # Phase 3: Update counters and extract predictions
            audio_codes_next, pred_phoneme_tokens = self._process_predictions(
                state, needs_context, needs_phoneme, needs_audio
            )

            return state, audio_codes_next, pred_phoneme_tokens

    def _prepare_streaming_input(
        self,
        state: StreamingState,
        text_tokens: Optional[torch.Tensor],
        force_dropout_text: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the input embedding for one streaming step.

        Determines per-batch-item phase (context / text / phoneme / audio), composes
        the corresponding embeddings, handles CFG doubling, and updates text_finished
        and phoneme_stream_ended flags on state.

        Returns:
            Tuple of (next_input, needs_context, needs_phoneme, needs_audio) where
            next_input is (B, 1, E) or (2*B, 1, E) with CFG, and the masks are (B,) bool tensors.
        """
        device = state.config.device
        batch_size = state.config.batch_size
        streaming_speech_delay = state.config.training_mode.streaming_speech_delay
        streaming_phonemes_delay = state.config.training_mode.streaming_phonemes_delay

        # Determine phases per batch item
        needs_context = state.context_position < state.full_context_lens  # (B,) bool
        needs_text = (~needs_context) & (~state.text_finished)
        needs_phoneme = (
            (~needs_context) & (state.text_tokens_seen >= streaming_phonemes_delay) & (~state.phoneme_stream_ended)
        )
        needs_audio = (~needs_context) & (state.text_tokens_seen >= streaming_speech_delay) & (~state.finished)

        next_input = torch.zeros(batch_size, 1, self.cfg.embedding_dim, device=device)

        # --- Context phase items: use next context embedding ---
        if needs_context.any():
            ctx_positions = state.context_position.clone()  # (B,)
            ctx_positions = ctx_positions.clamp(max=state.full_context_embedding.size(1) - 1)
            ctx_emb = state.full_context_embedding[
                torch.arange(batch_size, device=device), ctx_positions, :
            ].unsqueeze(
                1
            )  # (B, 1, E)
            context_mask = needs_context.view(batch_size, 1, 1).float()
            next_input = next_input + ctx_emb * context_mask

        # --- Non-context phase items: handle text embedding ---
        if text_tokens is not None and needs_text.any():
            text_tokens_2d = text_tokens.unsqueeze(1)  # (B, 1)
            text_embedded = self.decoder.get_input_embeddings()(text_tokens_2d)  # (B, 1, E)

            if self.use_bpe_char_tokenizer:
                text_mask = torch.ones_like(text_tokens_2d, dtype=torch.bool)
                cas_embedding = self.cas_encoder(text_tokens_2d, subword_mask=text_mask)  # (B, 1, E)
                text_embedded = text_embedded + cas_embedding

            if force_dropout_text:
                text_embedded = text_embedded * 0

            is_eos_token = (text_tokens == self.eos_id) & needs_text  # (B,) bool
            text_add_mask = needs_text.view(batch_size, 1, 1).float()
            next_input = next_input + text_embedded * text_add_mask
            state.text_finished = state.text_finished | is_eos_token

        elif text_tokens is None:
            state.text_finished = state.text_finished | ~needs_context

        # --- Phoneme embedding for phoneme and audio phase items ---
        if self.phoneme_tokenizer is not None:
            if needs_phoneme.any():
                phoneme_emb = torch.zeros(batch_size, 1, self.cfg.embedding_dim, device=device)

                if state.config.phoneme_input_type == 'gt' and state.gt_phoneme_embeddings is not None:
                    within_gt_len = state.phoneme_steps < state.gt_phoneme_lens  # (B,)
                    positions = state.phoneme_steps.clamp(max=state.gt_phoneme_embeddings.size(1) - 1)
                    gt_emb = state.gt_phoneme_embeddings[
                        torch.arange(batch_size, device=device), positions, :
                    ].unsqueeze(
                        1
                    )  # (B, 1, E)
                    phoneme_mask = (needs_phoneme & within_gt_len).view(batch_size, 1, 1).float()
                    phoneme_emb = phoneme_emb + gt_emb * phoneme_mask
                else:
                    first_phoneme_step = needs_phoneme & (state.phoneme_steps == 0)
                    has_last_phoneme = needs_phoneme & (~first_phoneme_step) & (state.last_phoneme_tokens is not None)

                    if first_phoneme_step.any():
                        phoneme_bos = torch.full(
                            (batch_size, self.phoneme_stacking_factor, 1),
                            self.phoneme_tokenizer.bos_token_id,
                            device=device,
                        ).long()
                        phoneme_bos_emb = self.embed_phoneme_tokens(phoneme_bos)  # (B, 1, E)
                        first_mask = first_phoneme_step.view(batch_size, 1, 1).float()
                        phoneme_emb = phoneme_emb + phoneme_bos_emb * first_mask

                    if has_last_phoneme.any() and state.last_phoneme_tokens is not None:
                        last_phoneme_emb = self.embed_phoneme_tokens(
                            state.last_phoneme_tokens.unsqueeze(2)
                        )  # (B, 1, E)
                        last_mask = has_last_phoneme.view(batch_size, 1, 1).float()
                        phoneme_emb = phoneme_emb + last_phoneme_emb * last_mask

                    state.phoneme_stream_ended = state.phoneme_stream_ended | state.phoneme_eos_detected

                next_input = next_input + phoneme_emb

        # --- Audio embedding for audio phase items ---
        audio_emb = None
        if needs_audio.any():
            audio_emb = torch.zeros(batch_size, 1, self.cfg.embedding_dim, device=device)

            if state.gt_audio_embeddings is not None:
                within_gt_len = state.audio_steps < state.gt_audio_lens  # (B,)
                positions = state.audio_steps.clamp(max=state.gt_audio_embeddings.size(1) - 1)
                gt_emb = state.gt_audio_embeddings[torch.arange(batch_size, device=device), positions, :].unsqueeze(
                    1
                )  # (B, 1, E)
                audio_mask = (needs_audio & within_gt_len).view(batch_size, 1, 1).float()
                audio_emb = audio_emb + gt_emb * audio_mask
            else:
                first_audio_step = needs_audio & (state.audio_steps == 0)
                has_last_audio = needs_audio & ~first_audio_step & (state.last_audio_codes is not None)

                if first_audio_step.any():
                    audio_bos = torch.full(
                        (batch_size, self.num_audio_codebooks * self.frame_stacking_factor, 1),
                        self.audio_bos_id,
                        device=device,
                    ).long()
                    audio_bos_emb = self.embed_audio_tokens(audio_bos)  # (B, 1, E)
                    first_mask = first_audio_step.view(batch_size, 1, 1).float()
                    audio_emb = audio_emb + audio_bos_emb * first_mask

                if has_last_audio.any() and state.last_audio_codes is not None:
                    last_audio_emb = self.embed_audio_tokens(state.last_audio_codes.unsqueeze(2))  # (B, 1, E)
                    last_mask = has_last_audio.view(batch_size, 1, 1).float()
                    audio_emb = audio_emb + last_audio_emb * last_mask

            next_input = next_input + audio_emb

        # --- Handle CFG ---
        if state.config.use_cfg:
            next_input_unconditional_context = state.config.dummy_context_embedding_unconditional.expand(
                batch_size, 1, -1
            )
            next_input_unconditional_zeros = torch.zeros_like(next_input_unconditional_context)
            context_mask = needs_context.view(batch_size, 1, 1).float()
            next_input_unconditional = (
                context_mask * next_input_unconditional_context + (1 - context_mask) * next_input_unconditional_zeros
            )

            if needs_audio.any():
                audio_mask = needs_audio.view(batch_size, 1, 1).float()
                next_input_unconditional = next_input_unconditional * (1 - audio_mask) + audio_emb * audio_mask

            next_input = torch.cat([next_input, next_input_unconditional], dim=0)

        return next_input, needs_context, needs_phoneme, needs_audio

    def _process_predictions(
        self,
        state: StreamingState,
        needs_context: torch.Tensor,
        needs_phoneme: torch.Tensor,
        needs_audio: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Update state counters and extract phoneme/audio predictions from the transformer output.

        Args:
            state: Current StreamingState (last_hidden must already be set from the forward pass).
            needs_context: (B,) bool mask of items still in context phase.
            needs_phoneme: (B,) bool mask of items in phoneme prediction phase.
            needs_audio: (B,) bool mask of items in audio prediction phase.

        Returns:
            Tuple of (audio_codes_next, pred_phoneme_tokens), either may be None.
        """
        batch_size = state.config.batch_size
        device = state.config.device

        # Update counters
        state.context_position = state.context_position + needs_context.long()
        state.text_tokens_seen = state.text_tokens_seen + (~needs_context).long()
        state.phoneme_steps = state.phoneme_steps + needs_phoneme.long()
        state.audio_steps = state.audio_steps + needs_audio.long()

        pred_phoneme_tokens = None
        audio_codes_next = None

        # --- Phoneme predictions ---
        if needs_phoneme.any() and self.phoneme_tokenizer is not None:
            first_phoneme_step = needs_phoneme & (state.phoneme_prediction_start_idx == -1)
            if first_phoneme_step.any():
                current_phoneme_step_idx = len(state.all_phoneme_predictions)
                state.phoneme_prediction_start_idx = torch.where(
                    first_phoneme_step,
                    torch.full_like(state.phoneme_prediction_start_idx, current_phoneme_step_idx),
                    state.phoneme_prediction_start_idx,
                )

            pred_phoneme_tokens = self._predict_phoneme_tokens(state)  # (B, phoneme_stacking_factor)
            state.last_phoneme_tokens = pred_phoneme_tokens
            state.all_phoneme_predictions.append(pred_phoneme_tokens)

            phoneme_eos_detected = needs_phoneme & (pred_phoneme_tokens == self.phoneme_tokenizer.eos_token_id).any(
                dim=1
            )  # (B,)

            state.phoneme_eos_detected = state.phoneme_eos_detected | phoneme_eos_detected

            newly_ended_phoneme = phoneme_eos_detected & (state.phoneme_prediction_end_idx == -1)
            if newly_ended_phoneme.any():
                current_phoneme_step_idx = len(state.all_phoneme_predictions)
                state.phoneme_prediction_end_idx = torch.where(
                    newly_ended_phoneme,
                    torch.full_like(state.phoneme_prediction_end_idx, current_phoneme_step_idx),
                    state.phoneme_prediction_end_idx,
                )

        # --- Audio predictions ---
        if needs_audio.any():
            first_audio_step = needs_audio & (state.audio_prediction_start_idx == -1)
            if first_audio_step.any():
                current_frame_idx = sum(p.size(-1) for p in state.all_predictions)
                state.audio_prediction_start_idx = torch.where(
                    first_audio_step,
                    torch.full_like(state.audio_prediction_start_idx, current_frame_idx),
                    state.audio_prediction_start_idx,
                )

            audio_codes_next_stacked, all_codes_next_argmax = self._predict_audio_codes(state)  # (B, C*S)

            S = self.frame_stacking_factor
            C = self.num_audio_codebooks
            audio_codes_unstacked = audio_codes_next_stacked.view(batch_size, C, S)  # (B, C, S)

            if state.last_audio_codes is None:
                state.last_audio_codes = audio_codes_next_stacked
            else:
                update_mask = needs_audio.view(batch_size, 1).expand_as(audio_codes_next_stacked)
                state.last_audio_codes = torch.where(update_mask, audio_codes_next_stacked, state.last_audio_codes)

            # EOS detection (skip in teacher-forced mode)
            if state.gt_audio_embeddings is None:
                all_codes_argmax_unstacked = all_codes_next_argmax.view(batch_size, C, S)

                eos_in_sampled = audio_codes_unstacked == self.audio_eos_id  # (B, C, S)
                eos_in_argmax = all_codes_argmax_unstacked == self.audio_eos_id  # (B, C, S)
                eos_any_codebook = eos_in_sampled.any(dim=1) | eos_in_argmax.any(dim=1)  # (B, S)

                eos_frame_idx = torch.where(
                    eos_any_codebook.any(dim=1),
                    eos_any_codebook.int().argmax(dim=1),
                    torch.full((batch_size,), S, device=device),
                )  # (B,)

                audio_eos_detected = eos_any_codebook.any(dim=1) & needs_audio
                state.finished = state.finished | audio_eos_detected

                newly_ended_audio = audio_eos_detected & (state.audio_prediction_end_idx == -1)
                if newly_ended_audio.any():
                    current_frame_count = len(state.all_predictions) * self.frame_stacking_factor
                    end_frame_idx = current_frame_count + eos_frame_idx
                    state.audio_prediction_end_idx = torch.where(
                        newly_ended_audio, end_frame_idx, state.audio_prediction_end_idx
                    )

            state.all_predictions.append(audio_codes_unstacked)
            audio_codes_next = audio_codes_unstacked

        # Force-finish items when GT audio is exhausted (teacher forcing)
        if state.gt_audio_embeddings is not None and state.gt_audio_lens is not None:
            gt_exhausted = needs_audio & (state.audio_steps >= state.gt_audio_lens)
            state.finished = state.finished | gt_exhausted

        return audio_codes_next, pred_phoneme_tokens

    def _predict_phoneme_tokens(self, state: StreamingState) -> torch.Tensor:
        """Predict phoneme tokens from the last hidden state."""
        actual_batch_size = state.config.batch_size
        last_hidden = state.last_hidden

        # Get phoneme logits
        all_code_logits_t_phoneme = self.phoneme_final_proj(last_hidden[:, -1, :])
        all_code_logits_t_phoneme = all_code_logits_t_phoneme[:actual_batch_size]
        phoneme_logits = all_code_logits_t_phoneme.view(
            actual_batch_size, self.phoneme_stacking_factor, self.phoneme_vocab_size
        )
        max_probs = torch.softmax(phoneme_logits, dim=-1).max(dim=-1).values  # (B, phoneme_stacking_factor)

        # Sample phonemes
        if state.config.phoneme_sampling_method == 'argmax':
            pred_phoneme_tokens = self.sample_codes_from_logits_phoneme(all_code_logits_t_phoneme, temperature=0.0)
        else:
            pred_phoneme_tokens = self.sample_codes_from_logits_phoneme(
                all_code_logits_t_phoneme, temperature=state.config.temperature, topk=state.config.topk
            )

        # In prediction mode, low-confidence phoneme steps are replaced with UNK across
        # all stacked channels (except steps where EOS is predicted).
        if (
            state.config.phoneme_input_type != 'gt'
            and hasattr(self.phoneme_tokenizer, 'unk_token_id')
            and self.phoneme_confidence_unk_threshold > 0.0
        ):
            underconfident_step = (max_probs < self.phoneme_confidence_unk_threshold).any(
                dim=1, keepdim=True
            )  # (B, 1)
            eos_predicted_step = (pred_phoneme_tokens == self.phoneme_tokenizer.eos_token_id).any(dim=1, keepdim=True)
            replace_with_unk = underconfident_step & (~eos_predicted_step)
            if replace_with_unk.any():
                unk_tokens = torch.full_like(pred_phoneme_tokens, self.phoneme_tokenizer.unk_token_id)
                pred_phoneme_tokens = torch.where(replace_with_unk, unk_tokens, pred_phoneme_tokens)
        # (B, phoneme_stacking_factor)
        return pred_phoneme_tokens

    def _predict_audio_codes(self, state: StreamingState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict audio codes from the last hidden state."""
        actual_batch_size = state.config.batch_size
        last_hidden = state.last_hidden

        # Compute audio logits
        last_hidden_audio = self.audio_out_projection(last_hidden[:, -1, :])
        all_code_logits_t = self.final_proj(last_hidden_audio)

        # Apply CFG if enabled
        if state.config.use_cfg:
            conditional_logits = all_code_logits_t[:actual_batch_size]
            unconditional_logits = all_code_logits_t[actual_batch_size:]
            all_code_logits_t = (
                state.config.cfg_scale * conditional_logits + (1.0 - state.config.cfg_scale) * unconditional_logits
            )

        # Sample audio codes
        audio_codes_next, all_codes_next_argmax = self._sample_audio_codes(
            last_hidden=last_hidden,
            all_code_logits_t=all_code_logits_t,
            temperature=state.config.temperature,
            topk=state.config.topk,
            use_local_transformer_for_inference=state.config.use_local_transformer,
            use_cfg=state.config.use_cfg,
            cfg_scale=state.config.cfg_scale,
        )

        return audio_codes_next, all_codes_next_argmax

    def streaming_finalize(
        self,
        state: StreamingState,
        use_inference_mode: bool = True,
    ) -> StreamingFinalizeOutput:
        """
        Finalize streaming and return the complete generated audio and phoneme predictions.

        This function should be called after all streaming_step() calls are complete
        (i.e., when state.finished.all() is True or max steps reached).

        Args:
            state: Final StreamingState after streaming is complete.

        Returns:
            StreamingFinalizeOutput containing audio, codes, and phoneme predictions.
        """
        batch_size = state.config.batch_size
        device = state.config.device

        # Extract and decode phoneme predictions
        phoneme_tokens_list: List[List[int]] = []
        phoneme_text_list: List[str] = []
        if self.phoneme_tokenizer is not None and len(state.all_phoneme_predictions) > 0:
            # Stack phoneme predictions: each is (B, phoneme_stacking_factor)
            all_phonemes = torch.stack(state.all_phoneme_predictions, dim=-1)  # (B, S, T)
            for i in range(batch_size):
                start = max(0, state.phoneme_prediction_start_idx[i].item())
                end = state.phoneme_prediction_end_idx[i].item()
                if end < 0:
                    end = all_phonemes.size(-1)
                # Flatten stacked phonemes back to sequence
                tokens = all_phonemes[i, :, start:end].T.reshape(-1).tolist()
                # Remove special tokens (BOS, EOS, PAD)
                special = {self.phoneme_tokenizer.bos_token_id, self.phoneme_tokenizer.eos_token_id}
                if hasattr(self.phoneme_tokenizer, 'pad_token_id'):
                    special.add(self.phoneme_tokenizer.pad_token_id)
                tokens = [t for t in tokens if t not in special]
                phoneme_tokens_list.append(tokens)
                phoneme_text_list.append(self.phoneme_tokenizer.decode(tokens))
        else:
            phoneme_tokens_list = [[] for _ in range(batch_size)]
            phoneme_text_list = ["" for _ in range(batch_size)]

        if len(state.all_predictions) == 0:
            return StreamingFinalizeOutput(
                audio=torch.zeros(batch_size, 0, device=device),
                audio_len=torch.zeros(batch_size, dtype=torch.long, device=device),
                audio_codes=torch.zeros(batch_size, self.num_audio_codebooks, 0, device=device),
                audio_codes_len=torch.zeros(batch_size, dtype=torch.long, device=device),
                phoneme_tokens=phoneme_tokens_list,
                phoneme_text=phoneme_text_list,
            )

        grad_ctx = torch.inference_mode if use_inference_mode else torch.no_grad
        with grad_ctx():
            # Concatenate all predictions - each is (B, C, S), concat gives (B, C, T_total_frames)
            all_codes = torch.cat(state.all_predictions, dim=-1)  # (B, C, T_total_frames)
            total_frames = all_codes.size(-1)
            num_codebooks = all_codes.size(1)

            # Start and end indices are in frames (not steps)
            # If start_idx is -1, item never started audio predictions - use 0
            # If end_idx is -1, item never ended - use total_frames
            start_indices = torch.clamp(state.audio_prediction_start_idx, min=0)
            end_indices = torch.where(
                state.audio_prediction_end_idx >= 0,
                state.audio_prediction_end_idx,
                torch.full_like(state.audio_prediction_end_idx, total_frames),
            )

            # Calculate per-item lengths (in frames)
            predicted_codes_lens = end_indices - start_indices
            max_len = predicted_codes_lens.max().item()

            # Handle case where all items have zero-length predictions
            if max_len == 0:
                return StreamingFinalizeOutput(
                    audio=torch.zeros(batch_size, 0, device=device),
                    audio_len=torch.zeros(batch_size, dtype=torch.long, device=device),
                    audio_codes=torch.zeros(batch_size, num_codebooks, 0, device=device, dtype=all_codes.dtype),
                    audio_codes_len=torch.zeros(batch_size, dtype=torch.long, device=device),
                    phoneme_tokens=phoneme_tokens_list,
                    phoneme_text=phoneme_text_list,
                )

            # Create padded output tensor and slice each item's valid predictions
            predicted_codes = torch.zeros(batch_size, num_codebooks, max_len, dtype=all_codes.dtype, device=device)
            for i in range(batch_size):
                start = start_indices[i].item()
                end = end_indices[i].item()
                length = end - start
                if length > 0:
                    predicted_codes[i, :, :length] = all_codes[i, :, start:end]

            # No need to remove EOS - end_indices already point to the frame before EOS
            # Decode to audio (codes are already unstacked: B, C, T)
            predicted_codes, predicted_codes_lens = self._prepare_codes_for_decode(
                predicted_codes, predicted_codes_lens
            )
            audio, audio_len, decoded_codes = self._codec_helper.codes_to_audio(
                predicted_codes,
                predicted_codes_lens,
            )

            return StreamingFinalizeOutput(
                audio=audio,
                audio_len=audio_len,
                audio_codes=predicted_codes,
                audio_codes_len=predicted_codes_lens,
                phoneme_tokens=phoneme_tokens_list,
                phoneme_text=phoneme_text_list,
            )

    def infer_batch(
        self,
        batch: Dict[str, torch.Tensor],
        max_decoder_steps: int = 500,
        temperature: float = 0.7,
        topk: int = 80,
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        use_local_transformer_for_inference: bool = False,
        phoneme_input_type: str = 'pred',
        phoneme_sampling_method: str = 'argmax',
        force_dropout_text: bool = False,
        use_teacher_forced: bool = False,
        use_inference_mode: bool = True,
    ) -> InferBatchOutput:
        """
        Batch inference using streaming infrastructure.

        This is a simple wrapper around streaming_init, streaming_step, and streaming_finalize
        that processes a batch dictionary similar to training_step/validation_step.

        Args:
            batch: Dictionary containing:
                - text: Text token IDs (B, L)
                - text_lens: Lengths (B,)
                - context_text_tokens: Context text tokens (B, L')
                - context_text_tokens_lens: Lengths (B,)
                - context_audio_codes: Context audio codes (B, C, T) OR
                - context_audio / context_audio_lens: Raw context audio to encode
                - phoneme_tokens (optional): GT phoneme tokens (B, L'')
                - phoneme_tokens_lens (optional): Lengths (B,)
                For teacher forcing (use_teacher_forced=True), also requires:
                - audio_codes / audio_codes_lens: GT audio codes (B, C, T) OR
                - audio / audio_lens: Raw audio waveforms to encode
            max_decoder_steps: Maximum number of decoder steps.
            temperature: Sampling temperature for audio codes. Use 0.0 for argmax.
            topk: Top-k sampling parameter.
            use_cfg: Whether to use classifier-free guidance.
            cfg_scale: CFG scale factor.
            use_local_transformer_for_inference: Whether to use local transformer.
            phoneme_input_type: 'gt' or 'pred' for phoneme tokens.
            phoneme_sampling_method: 'argmax' or 'sample' for phoneme token selection.
            force_dropout_text: Whether to dropout text embeddings.
            use_teacher_forced: If True, feed GT audio codes (and force GT phonemes, argmax sampling)
                instead of predicted codes at each streaming step.

        Returns:
            InferBatchOutput containing predicted audio, codes, and RTF metrics.
        """
        grad_ctx = torch.inference_mode if use_inference_mode else torch.no_grad
        with grad_ctx():
            start_time = time.time()

            # Extract tensors from batch
            text = batch['text']
            text_lens = batch['text_lens']
            context_text_tokens = batch['context_text_tokens']
            context_text_tokens_lens = batch['context_text_tokens_lens']

            # Handle context audio - either use codes directly or encode from audio
            if 'context_audio_codes' in batch:
                context_audio_codes = batch['context_audio_codes']
                context_audio_codes_lens = batch['context_audio_codes_lens']
            else:
                context_audio = batch['context_audio']
                context_audio_lens = batch['context_audio_lens']
                context_audio_codes, context_audio_codes_lens = self._codec_helper.audio_to_codes(
                    context_audio, context_audio_lens
                )

            # Optional GT phoneme tokens for teacher forcing
            gt_phoneme_tokens = batch.get('phoneme_tokens')
            gt_phoneme_tokens_lens = batch.get('phoneme_tokens_lens')

            # Prepare GT audio codes for teacher forcing if requested
            gt_audio_codes_for_init = None
            gt_audio_codes_lens_for_init = None
            if use_teacher_forced:
                # Force GT phoneme input and argmax sampling
                phoneme_input_type = 'gt'
                temperature = 0.0

                # Get GT audio codes
                if 'audio_codes' in batch:
                    gt_audio_codes = batch['audio_codes']
                    gt_audio_codes_lens = batch['audio_codes_lens']
                elif 'audio' in batch:
                    gt_audio = batch['audio']
                    gt_audio_lens = batch['audio_lens']
                    gt_audio_codes, gt_audio_codes_lens = self._codec_helper.audio_to_codes(gt_audio, gt_audio_lens)
                else:
                    raise ValueError("Teacher forcing requires 'audio_codes' or 'audio' in batch")

                # Convert and add special tokens, then stack
                if self._codec_converter is not None:
                    gt_audio_codes = self._codec_converter.convert_original_to_new(
                        audio_tokens=gt_audio_codes, audio_lens=gt_audio_codes_lens
                    ).long()

                gt_audio_codes_processed, gt_audio_codes_lens_processed = add_special_tokens(
                    codes=gt_audio_codes,
                    codes_len=gt_audio_codes_lens,
                    bos_id=self.audio_bos_id,
                    eos_id=self.audio_eos_id,
                )
                gt_audio_codes_processed, gt_audio_codes_lens_processed = self.stack_codes(
                    gt_audio_codes_processed,
                    gt_audio_codes_lens_processed,
                    self.audio_bos_id,
                    self.audio_eos_id,
                    self.frame_stacking_factor,
                    self.num_audio_codebooks,
                )

                # Input portion: all tokens except the last (teacher forcing shift)
                gt_audio_codes_for_init = gt_audio_codes_processed[:, :, :-1]
                gt_audio_codes_lens_for_init = gt_audio_codes_lens_processed - 1

            batch_size = text.size(0)

            # Initialize streaming state
            state = self.streaming_init(
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                context_text_tokens=context_text_tokens,
                context_text_tokens_lens=context_text_tokens_lens,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
                use_local_transformer=use_local_transformer_for_inference,
                temperature=temperature,
                topk=topk,
                phoneme_input_type=phoneme_input_type,
                phoneme_sampling_method=phoneme_sampling_method,
                gt_phoneme_tokens=gt_phoneme_tokens,
                gt_phoneme_tokens_lens=gt_phoneme_tokens_lens,
                gt_audio_codes=gt_audio_codes_for_init,
                gt_audio_codes_lens=gt_audio_codes_lens_for_init,
                use_inference_mode=use_inference_mode,
            )

            time_to_first_prediction = None
            generation_start_time = time.time()
            device = text.device

            # Generate until all items are finished or max steps reached
            logging.info("Generation started")
            gen_step = 0
            while not state.finished.all() and len(state.all_predictions) < max_decoder_steps:
                gen_step += 1
                if gen_step % 10 == 0:
                    logging.info(f"Generation step {gen_step}")
                # Gather the correct text token for each batch item based on text_tokens_seen
                # Items in context phase will have their token ignored by streaming_step
                positions = state.text_tokens_seen.clamp(max=text.size(1) - 1)
                current_tokens = text[torch.arange(batch_size, device=device), positions]

                # For items that have exhausted their text, provide EOS token
                text_exhausted = state.text_tokens_seen >= text_lens
                current_tokens = torch.where(
                    text_exhausted, torch.full_like(current_tokens, self.eos_id), current_tokens
                )

                state, audio_codes, phoneme_tokens = self.streaming_step(
                    state=state,
                    text_tokens=current_tokens,
                    force_dropout_text=force_dropout_text,
                    use_inference_mode=use_inference_mode,
                )

                # Record time to first audio prediction
                if time_to_first_prediction is None and audio_codes is not None:
                    time_to_first_prediction = time.time() - start_time

            tts_generation_time = time.time() - generation_start_time

            # Finalize and decode audio
            finalize_output = self.streaming_finalize(state, use_inference_mode=use_inference_mode)

            end_time = time.time()
            total_time = end_time - start_time

            # Compute RTF metrics
            total_audio_samples = finalize_output.audio_len.sum().item()
            total_audio_duration = total_audio_samples / self.output_sample_rate
            num_frames = len(state.all_predictions)
            tts_generation_time_per_frame = tts_generation_time / num_frames if num_frames > 0 else 0.0

            rtf_metrics = {
                'rtf': total_audio_duration / total_time if total_time > 0 else 0.0,
                'time_to_first_prediction': time_to_first_prediction,
                'tts_generation_time': tts_generation_time,
                'total_time': total_time,
                'total_audio_duration': total_audio_duration,
                'total_audio_samples': total_audio_samples,
                'num_decoder_steps': num_frames,
                'tts_generation_time_per_frame': tts_generation_time_per_frame,
            }

            # Prepare phoneme token output if available
            predicted_phoneme_tokens = None
            predicted_phoneme_tokens_lens = None
            phoneme_prediction_start_idx_out = None
            if self.phoneme_tokenizer is not None and len(state.all_phoneme_predictions) > 0:
                predicted_phoneme_tokens = torch.stack(state.all_phoneme_predictions, dim=-1)  # (B, S, T)
                # Per-item valid phoneme prediction lengths
                phoneme_start = torch.clamp(state.phoneme_prediction_start_idx, min=0)
                phoneme_end = torch.where(
                    state.phoneme_prediction_end_idx >= 0,
                    state.phoneme_prediction_end_idx,
                    torch.full_like(state.phoneme_prediction_end_idx, predicted_phoneme_tokens.size(-1)),
                )
                predicted_phoneme_tokens_lens = phoneme_end - phoneme_start
                phoneme_prediction_start_idx_out = phoneme_start

            return InferBatchOutput(
                predicted_audio=finalize_output.audio,
                predicted_audio_lens=finalize_output.audio_len,
                predicted_codes=finalize_output.audio_codes,
                predicted_codes_lens=finalize_output.audio_codes_len,
                rtf_metrics=rtf_metrics,
                predicted_phoneme_tokens=predicted_phoneme_tokens,
                predicted_phoneme_tokens_lens=predicted_phoneme_tokens_lens,
                phoneme_prediction_start_idx=phoneme_prediction_start_idx_out,
            )

    @staticmethod
    def _load_audio_for_inference(audio_path: str, target_sample_rate: int) -> torch.Tensor:
        """
        Load context audio and resample if needed.
        Returns tensor of shape (1, num_samples).
        """
        audio, sr = sf.read(audio_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != target_sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
        return torch.from_numpy(audio).unsqueeze(0)

    @staticmethod
    def _adjust_audio_to_duration_for_inference(
        audio: torch.Tensor,
        sample_rate: int,
        target_duration: float,
        codec_model_samples_per_frame: int,
    ) -> torch.Tensor:
        """
        Match the same duration-alignment logic used in magpietts_streaming_inference.py.
        """
        num_codec_frames = int(target_duration * sample_rate / codec_model_samples_per_frame)
        target_num_samples = num_codec_frames * codec_model_samples_per_frame
        current_num_samples = audio.size(1)

        if current_num_samples >= target_num_samples:
            audio = audio[:, :target_num_samples]
        else:
            num_repeats = int(np.ceil(target_num_samples / current_num_samples))
            audio_repeated = audio.repeat(1, num_repeats)
            audio = audio_repeated[:, :target_num_samples]
        return audio

    def do_tts(
        self,
        transcript: str,
        context_audio_file_path: Optional[str] = None,
        context_text: str = "[NO TEXT CONTEXT]",
        main_tokenizer_name: Optional[str] = None,
        context_audio_duration: float = 5.0,
        use_cfg: bool = True,
        cfg_scale: float = 2.5,
        use_local_transformer: bool = True,
        temperature: float = 0.7,
        topk: int = 80,
        max_steps: int = 330,
        gt_phoneme_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate speech from transcript using EasyMagpie inference with optional context text/audio.
        Optionally accepts ground-truth phoneme text (IPA string) for decoder-only inference.
        """
        if transcript is None or transcript.strip() == "":
            raise ValueError("`transcript` must be a non-empty string.")

        device = next(self.parameters()).device
        transcript = transcript.strip()
        context_text = (context_text or "[NO TEXT CONTEXT]").strip()

        if main_tokenizer_name is None:
            # Match model init behavior: default to first configured tokenizer.
            main_tokenizer_name = list(self.cfg.text_tokenizers.keys())[0]
        if main_tokenizer_name not in self.tokenizer.tokenizers:
            raise ValueError(
                f"Unknown main_tokenizer_name='{main_tokenizer_name}'. "
                f"Available tokenizers: {list(self.tokenizer.tokenizers.keys())}"
            )

        text_tokens = self.tokenizer.encode(transcript, tokenizer_name=main_tokenizer_name) + [self.eos_id]
        text = torch.tensor([text_tokens], dtype=torch.long, device=device)
        text_lens = torch.tensor([len(text_tokens)], dtype=torch.long, device=device)

        context_text_tokens = self.tokenizer.encode(context_text, tokenizer_name=self.text_conditioning_tokenizer_name)
        context_text_tensor = torch.tensor([context_text_tokens], dtype=torch.long, device=device)
        context_text_lens = torch.tensor([len(context_text_tokens)], dtype=torch.long, device=device)

        if context_audio_file_path is not None and context_audio_file_path.strip() != "":
            context_audio = self._load_audio_for_inference(context_audio_file_path, self.sample_rate)
            context_audio = self._adjust_audio_to_duration_for_inference(
                context_audio,
                self.sample_rate,
                context_audio_duration,
                self.codec_model_samples_per_frame,
            )
            context_audio = context_audio.to(device)
            context_audio_lens = torch.tensor([context_audio.size(1)], dtype=torch.long, device=device)
            with torch.inference_mode():
                context_audio_codes, context_audio_codes_lens = self._codec_helper.audio_to_codes(
                    context_audio, context_audio_lens
                )
        else:
            context_audio_codes = torch.zeros(
                1,
                self.data_num_audio_codebooks,
                0,
                dtype=torch.long,
                device=device,
            )
            context_audio_codes_lens = torch.zeros(1, dtype=torch.long, device=device)

        batch = {
            'text': text,
            'text_lens': text_lens,
            'context_text_tokens': context_text_tensor,
            'context_text_tokens_lens': context_text_lens,
            'context_audio_codes': context_audio_codes,
            'context_audio_codes_lens': context_audio_codes_lens,
        }
        phoneme_input_type = 'pred'
        if gt_phoneme_text is not None:
            if self.phoneme_tokenizer is None:
                raise ValueError(
                    "Model does not have a phoneme tokenizer configured, but gt_phoneme_text was provided."
                )
            gt_phoneme_text = gt_phoneme_text.strip()
            if gt_phoneme_text == "":
                raise ValueError("`gt_phoneme_text` must be a non-empty string when provided.")
            gt_phoneme_tokens = self.phoneme_tokenizer.encode(gt_phoneme_text)
            gt_phoneme_tokens = (
                [self.phoneme_tokenizer.bos_token_id] + gt_phoneme_tokens + [self.phoneme_tokenizer.eos_token_id]
            )
            if len(gt_phoneme_tokens) == 0:
                raise ValueError("Failed to encode `gt_phoneme_text` into phoneme tokens.")
            batch['phoneme_tokens'] = torch.tensor([gt_phoneme_tokens], dtype=torch.long, device=device)
            batch['phoneme_tokens_lens'] = torch.tensor([len(gt_phoneme_tokens)], dtype=torch.long, device=device)
            phoneme_input_type = 'gt'

        with torch.inference_mode():
            output = self.infer_batch(
                batch=batch,
                max_decoder_steps=max_steps,
                temperature=temperature,
                topk=topk,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
                use_local_transformer_for_inference=use_local_transformer,
                phoneme_input_type=phoneme_input_type,
                phoneme_sampling_method='argmax',
                use_teacher_forced=False,
                use_inference_mode=True,
            )
        return output.predicted_audio, output.predicted_audio_lens

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []
