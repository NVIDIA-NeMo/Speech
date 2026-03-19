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
import copy
import json
import os
import random
import re
import time

from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import wandb
from hydra.utils import instantiate
from lhotse.serialization import load_yaml
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch import nn
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import MagpieTTSLhotseDataset, setup_tokenizers
from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss
from nemo.collections.tts.losses.moe_loss import MoEAuxiliaryLoss, compute_expert_usage
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.modules.aligner import AlignmentEncoder
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.modules.magpietts_modules import (
    CharAwareSubwordEncoder,
    CodecHelper,
    EOSDetectionMethod,
    LocalTransformerHelper,
    LocalTransformerType,
    SpecialAudioToken,
    add_special_tokens,
    clear_forbidden_logits,
    pad_audio_codes,
    remove_bos_token,
    remove_embedded_bos_token,
    remove_embedded_eos_token,
    remove_eos_token,
    remove_special_tokens,
    worker_init_fn,
)
from nemo.collections.tts.parts.utils.helpers import (
    binarize_attention_parallel,
    get_mask_from_lengths,
    plot_alignment_to_numpy,
    plot_expert_usage_heatmap_to_numpy,
)
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    chunk_text_for_inference,
    get_tokenizer_for_language,
    stack_tensors,
)
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging


@dataclass
class InferBatchOutput:
    """Output dataclass for MagpieTTS infer_batch method.

    This provides a consistent return type regardless of which optional outputs
    are requested.

    Attributes:
        predicted_audio: Generated audio waveforms. Shape: (B, T_audio).
        predicted_audio_lens: Length of each audio in samples. Shape: (B,).
        predicted_codes: Generated audio codec tokens. Shape: (B, num_codebooks, T_frames).
        predicted_codes_lens: Length of each code sequence in frames. Shape: (B,).
        rtf_metrics: Dictionary containing real-time factor and timing metrics.
        cross_attention_maps: Optional cross-attention visualization maps.
            List of numpy arrays, one per batch item. Only populated if
            return_cross_attn_probs=True.
        headwise_cross_attention_maps: Optional per-head cross-attention maps.
            Only populated if return_cross_attn_probs=True and
            compute_all_heads_attn_maps=True.
    """

    predicted_audio: torch.Tensor
    predicted_audio_lens: torch.Tensor
    predicted_codes: torch.Tensor
    predicted_codes_lens: torch.Tensor
    rtf_metrics: Dict[str, Any]
    cross_attention_maps: Optional[List[Any]] = None
    headwise_cross_attention_maps: Optional[List[Any]] = None


@dataclass
class ContextTensorsOutput:
    """Output container for prepare_context_tensors method.

    This dataclass provides typed access to all tensors prepared for the decoder,
    replacing the previous untyped dictionary return.

    Attributes:
        text_encoder_out: Encoded text from the encoder. Shape: (B, T_text, E).
        text_embedded: Embedded text before encoding. Shape: (B, T_text, E).
        text_mask: Boolean mask for text. Shape: (B, T_text).
        text_lens: Length of each text sequence. Shape: (B,).
        text: Original text token IDs. Shape: (B, T_text).
        cond: Conditioning tensor(s) for decoder cross-attention.
            Either a single tensor or list of tensors for multi-encoder models.
        cond_mask: Mask(s) for conditioning tensors.
        attn_prior: Attention prior matrix for guided attention.
            Can be None, a tensor, or a list of tensors per layer.
        prior_used: Whether attention prior is being used.
        multi_encoder_mapping: Mapping for multi-encoder models (or None).
        additional_decoder_input: Context embeddings prepended to decoder input.
        additional_decoder_mask: Mask for additional decoder input.
        dec_context_size: Number of context frames prepended to decoder.
        context_audio_codes: Extracted context audio codes. Shape: (B, C, T_ctx).
        context_audio_codes_lens: Length of context audio codes. Shape: (B,).
        beta_binomial_attn_prior: Original beta-binomial prior from batch.
    """

    text_encoder_out: torch.Tensor
    text_embedded: torch.Tensor
    text_mask: torch.Tensor
    text_lens: torch.Tensor
    text: torch.Tensor
    cond: Union[torch.Tensor, List[torch.Tensor]]
    cond_mask: Union[torch.Tensor, List[torch.Tensor]]
    attn_prior: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None
    prior_used: bool = False
    multi_encoder_mapping: Optional[Dict[str, Any]] = None
    additional_decoder_input: Optional[torch.Tensor] = None
    additional_decoder_mask: Optional[torch.Tensor] = None
    dec_context_size: int = 0
    context_audio_codes: Optional[torch.Tensor] = None
    context_audio_codes_lens: Optional[torch.Tensor] = None
    beta_binomial_attn_prior: Optional[torch.Tensor] = None


@dataclass
class ChunkedDecoderState:
    """Tracks state during chunked speech generation (single- or multi-chunk).

    This dataclass encapsulates all the mutable state variables used in the
    autoregressive decoding loop of generate_speech, reducing parameter
    passing and improving code organization.

    Attributes:
        audio_codes_input: Current audio codes buffer. Shape: (B, num_codebooks, T).
        audio_codes_lens: Length of each audio sequence. Shape: (B,).
        audio_codes_mask: Mask for audio codes. Shape: (B, T).
        attended_timestep_counter: List of dicts tracking attention counts per timestep.
        all_predictions: List of predicted audio code tensors.
        chunk_end_dict: Maps batch indices to their chunk end timesteps.
        unfinished_texts: Maps batch indices to whether text is still being processed.
        finished_texts_counter: Maps batch indices to counts of timesteps near text end.
        attn_prior: Current attention prior tensor. Shape: (B, 1, T_text).
    """

    audio_codes_input: torch.Tensor
    audio_codes_lens: torch.Tensor
    audio_codes_mask: torch.Tensor
    attended_timestep_counter: List[Dict[int, int]]
    all_predictions: List[torch.Tensor]
    chunk_end_dict: Dict[int, int]
    unfinished_texts: Dict[int, bool]
    finished_texts_counter: Dict[int, int]
    attn_prior: Optional[torch.Tensor] = None


@dataclass
class ChunkedInferenceConfig:
    """Immutable configuration for chunked inference tuning parameters.

    These parameters control the behavior of chunked (single- or multi-chunk) speech generation.
    Initialized once in MagpieTTSModel.__init__ and accessed via self.chunked_inference_config.

    Attributes:
        history_len_heuristic: Maximum history tokens to retain across chunks.
        prior_weights_init: Attention prior weights for chunk initialization.
        prior_weights: Attention prior weights during generation (history, current, +1, +2, +3, +4).
        finished_limit_with_eot: Steps after text end before allowing EOS (multi-chunk).
        finished_limit_without_eot: Steps after chunk end before allowing EOS (multi-chunk).
        finished_limit_first_chunk: Steps near text end before forcing EOS for first/single chunk.
            Matches the threshold used in infer_batch() for consistent single-chunk behavior.
        forceful_chunk_end_threshold: Threshold for forceful chunk termination.
        argmax_temperature: Temperature for argmax sampling in EOS detection.
        short_sentence_threshold: Sentences shorter than this skip attention prior.
        attention_sink_threshold: Times attended before position is considered a sink.
        near_end_threshold: Positions from text end to consider "near end".
    """

    history_len_heuristic: int = 20
    prior_weights_init: Tuple[float, ...] = (0.5, 1.0, 0.8, 0.2, 0.2)
    prior_weights: Tuple[float, ...] = (0.2, 1.0, 0.6, 0.4, 0.2, 0.2)
    finished_limit_with_eot: int = 5
    finished_limit_without_eot: int = 1
    finished_limit_first_chunk: int = 20
    forceful_chunk_end_threshold: int = 3
    argmax_temperature: float = 0.01
    short_sentence_threshold: int = 35
    attention_sink_threshold: int = 10
    near_end_threshold: int = 3


@dataclass
class ChunkState:
    """Mutable state persisting across chunks during chunked generation.

    Created by the inference runner via model.create_chunk_state(),
    passed to generate_speech(), and updated in-place across chunk iterations.

    Attributes:
        batch_size: Number of items in the batch.
        history_text: Text tokens from previous chunks. Shape: (B, T).
        history_text_lens: Lengths of history text per batch item. Shape: (B,).
        history_context_tensor: Encoder output from previous chunks. Shape: (B, T, E).
        end_indices: Maps batch indices to overall timestep where they ended.
        overall_idx: Global timestep counter across all chunks.
        left_offset: Sliding window offset per batch item for attention tracking.
        previous_attn_len: Attention lengths from previous chunk per batch item.
        last_attended_timesteps: Tracking of attended positions across decoding.
    """

    batch_size: int
    history_text: Optional[torch.Tensor] = None
    history_text_lens: Optional[torch.Tensor] = None
    history_context_tensor: Optional[torch.Tensor] = None
    end_indices: Dict[int, int] = field(default_factory=dict)
    overall_idx: int = 0
    left_offset: List[int] = field(default_factory=list)
    previous_attn_len: List[int] = field(default_factory=list)
    last_attended_timesteps: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize batch-sized lists if not provided."""
        if not self.left_offset:
            self.left_offset = [0] * self.batch_size
        if not self.last_attended_timesteps:
            self.last_attended_timesteps = [[1] * self.batch_size]


@dataclass
class ModelInferenceParameters:
    """Model specific parameters that are sent to inference functions.

    This dataclass should contain all parameters that are model specific and should not change on a per run basis.

    Attributes:
        max_decoder_steps (int): Maximum number of decoder steps. Autoregressive for loop will terminate here.
        temperature (float): Sampling temperature.
        topk (int): Number of top-probability tokens to consider in sampling.
        cfg_scale (float): Scale factor for classifier-free guidance. Only used if use_cfg=True.
        apply_attention_prior (bool): Whether to apply attention prior.
        attention_prior_epsilon (float): Base probability for non-targeted positions.
        attention_prior_lookahead_window (int): Size of the forward-looking window to search for the next attended
            timestep. Determines how far ahead from the last attended timestep to look.
        estimate_alignment_from_layers (Optional[List[int]]): Layers to use for alignment estimation.
        apply_prior_to_layers (Optional[List[int]]): Layers to apply prior to.
        start_prior_after_n_audio_steps (int): Which step to start enabling the attention prior.
        use_LT_kv_cache (bool): Whether to use KV cache for the autoregressive local transformer.
        ignore_finished_sentence_tracking (bool): Whether to ignore finished sentence tracking.
        eos_detection_method (str): EOS detection method. See the EOSDetectionMethod class.
        min_generated_frames (int): Setting this greater than 0 prevents rare cases of first-frame termination. Any
            number greater between 1 and 4 should work, but 4 lines up with the codec's minimum frame requirement.
    """

    max_decoder_steps: int = 500
    temperature: float = 0.7
    topk: int = 80
    cfg_scale: float = 2.5
    apply_attention_prior: bool = True
    attention_prior_epsilon: float = 0.1
    attention_prior_lookahead_window: int = 5
    estimate_alignment_from_layers: Optional[List[int]] = None
    apply_prior_to_layers: Optional[List[int]] = None
    start_prior_after_n_audio_steps: int = 0
    use_LT_kv_cache: bool = True
    ignore_finished_sentence_tracking: bool = True
    eos_detection_method: str = "argmax_or_multinomial_any"
    min_generated_frames: int = 4

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelInferenceParameters':
        # Get the names of fields defined in the dataclass
        field_names = {field.name for field in fields(cls)}
        # Filter the input dictionary to include only valid fields
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        # Instantiate the dataclass with the filtered data

        # Double check for renamed fields: prior_epsilon and lookahead_window_size
        # These fields are currently used in nvidia/magpie_tts_multilingual_357m with commit hash: 291da79
        if 'prior_epsilon' in data:
            filtered_data['attention_prior_epsilon'] = data['prior_epsilon']
        if 'lookahead_window_size' in data:
            filtered_data['attention_prior_lookahead_window'] = data['lookahead_window_size']
        return cls(**filtered_data)


class MagpieTTSModel(ModelPT):
    """
    Magpie-TTS Model Base Class used for training a TTS model that can generate audio codes from transcript and a context
    audio/text

    Supports multiple model types:

    - multi_encoder_context_tts: Transcript and context audio go to different encoders. Transcript encoding feeds to
      layers given by cfg.model.transcript_decoder_layers and the context encoding feeds into the layers given by
      context_decoder_layers .Also supports text context which gets encoded by the same encoder as context audio.
      Only one of context audio or contex text is supported.

    - decoder_context_tts: Text goes into the encoder; context & target audio go to the decoder. Also supports text
      context. Supports fixed sized context so we set context_duration_min and context_duration_max to the same
      value (5 seconds). Text context, which is usually shorter than number of codec frames of 5 second of audio, is
      padded to the max context duration in this model.

    - decoder_ce: Same as decoder_context_tts except there is a small neural network between the context tensors and
      the decoder input.
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # Register tokenizer artifacts (phoneme_dict, heteronyms, etc.) for .nemo packaging
        self._register_tokenizer_artifacts(cfg)
        self._setup_inference_parameters(cfg)

        # load codec, disable loading of loss modules not needed during inference
        codec_model_path = cfg.get('codecmodel_path')
        if codec_model_path.startswith('nvidia/'):
            codec_model = AudioCodecModel.from_pretrained(codec_model_path)
        else:
            codec_model_cfg = AudioCodecModel.restore_from(codec_model_path, return_config=True)
            if "use_scl_loss" in codec_model_cfg:
                codec_model_cfg.use_scl_loss = False
            codec_model = AudioCodecModel.restore_from(
                codec_model_path, strict=False, override_config_path=codec_model_cfg
            )
        self.sample_rate = codec_model.sample_rate
        self.output_sample_rate = codec_model.output_sample_rate
        self.codec_model_samples_per_frame = codec_model.samples_per_frame
        # del codec discriminator to free memory
        del codec_model.discriminator

        # When using FSQ tokens, the codebook structure can be changed at any time.
        # An FSQ definition can be provided in `vector_quantizer` config to train with a codebook structure
        # that is different than in the audio codec checkpoint.
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

        # Our codebooks start with actual audio codec tokens, followed by special tokens.
        # The `forced_*` options are for backward compatibility for models trained with older code.
        get_token_index = partial(SpecialAudioToken.get_index, base_codebook_size=self.codebook_size)
        self.audio_bos_id = cfg.get('forced_audio_bos_id', get_token_index(SpecialAudioToken.AUDIO_BOS))
        self.audio_eos_id = cfg.get('forced_audio_eos_id', get_token_index(SpecialAudioToken.AUDIO_EOS))
        self.context_audio_bos_id = cfg.get(
            'forced_context_audio_bos_id', get_token_index(SpecialAudioToken.AUDIO_CONTEXT_BOS)
        )
        self.context_audio_eos_id = cfg.get(
            'forced_context_audio_eos_id', get_token_index(SpecialAudioToken.AUDIO_CONTEXT_EOS)
        )
        self.mask_token_id = cfg.get('forced_mask_token_id', get_token_index(SpecialAudioToken.MASK_TOKEN))
        self.num_all_tokens_per_codebook = cfg.get(
            'forced_num_all_tokens_per_codebook', self.codebook_size + len(SpecialAudioToken)
        )
        self.use_bpe_char_tokenizer = cfg.get('use_bpe_char_tokenizer', False)

        # The frame stacking factor controls how many consecutive frames are processed together by the base decoder
        # (and then refined into individual frames by the local transformer). A frame stacking factor of 1 means no
        # frame stacking. We have a separate embedding table for each of the stacked frames, e.g. for frame stacking
        # factor of 3, the entries of codebook 0 appear 3 times in the embedding table.
        self.frame_stacking_factor = cfg.get('frame_stacking_factor', 1)
        assert 'downsample_factor' not in cfg, '`downsample_factor` is deprecated, use `frame_stacking_factor` instead'
        # Setup tokenizer
        if hasattr(cfg, 'text_tokenizer'):
            # For backward compatibility for English-only models
            with open_dict(cfg):
                cfg.text_tokenizers = {"english_phoneme": cfg.text_tokenizer}
                del cfg['text_tokenizer']

        self.use_text_conditioning_encoder = cfg.get('use_text_conditioning_encoder', False)
        # Using google-t5/t5-small as default text conditioning tokenizer for backward compatibility.
        self.text_conditioning_tokenizer_name = cfg.get('text_conditioning_tokenizer_name', None)
        self.legacy_text_conditioning = cfg.get('legacy_text_conditioning', False)

        if self.legacy_text_conditioning:
            if self.text_conditioning_tokenizer_name is None:
                self.text_conditioning_tokenizer_name = "google-t5/t5-small"

            tokenizer_target = "AutoTokenizer"
            if self.text_conditioning_tokenizer_name == "google-t5/t5-small":
                tokenizer_target = "T5Tokenizer"

            with open_dict(cfg):
                cfg.text_tokenizers[self.text_conditioning_tokenizer_name] = {
                    '_target_': tokenizer_target,
                    'pretrained_model': self.text_conditioning_tokenizer_name,
                }
        elif self.text_conditioning_tokenizer_name is None:
            # If no text_conditioning_tokenizer_name is specified, use the first one as default
            # For text context tokenization
            self.text_conditioning_tokenizer_name = list(cfg.text_tokenizers.keys())[0]

        # TODO @xueyang: both tokenizers are only used to get some token ids. We
        # should kill them to save a small amount of mem resources since dataloader will initialize them
        # again after the worker processes are spawned.
        self.tokenizer = setup_tokenizers(
            all_tokenizers_config=cfg.text_tokenizers,
            mode='train',
        )

        num_tokens_tokenizer = len(self.tokenizer.tokens)
        if self.legacy_text_conditioning:
            # Text context tokens are not a part of the the regular transcript embedding table in legacy models
            num_tokens_tokenizer -= self.tokenizer.num_tokens_per_tokenizer[self.text_conditioning_tokenizer_name]

        num_tokens = num_tokens_tokenizer + 2  # +2 for BOS and EOS
        self.bos_id = num_tokens - 2
        self.eos_id = num_tokens - 1

        self.model_type = cfg.get('model_type', None)
        self.pad_context_text_to_max_duration = self.model_type in ['decoder_context_tts', 'decoder_ce']
        self.use_kv_cache_for_inference = cfg.get('use_kv_cache_for_inference', False)

        # Below args (text_context_remapping_json, text_context_remapping_prob) are
        # for combining multiple context_texts into a single one during training.
        # Eg. if we want to treat Emma_neutral and Emma_conversational as one speaker,
        # we can create an override dict {'Emma_neutral' : 'Emma', 'Emma_conversational' : 'Emma'}
        # This dict is saved in a json file given by cfg.model.text_context_remapping_json
        # If we want to preserve both behaviours i.e (Emma_neutral, Emma_conversational) and just (Emma)
        # we can do this mapping with a probability during training, as specified by text_context_remapping_prob
        self.text_context_remapping = None
        text_context_remapping_json = cfg.get('text_context_remapping_json', None)
        self.text_context_remapping_prob = cfg.get('text_context_remapping_prob', 0.0)
        if text_context_remapping_json is not None:
            with open(text_context_remapping_json, 'r') as f:
                self.text_context_remapping = json.load(f)

        super().__init__(cfg=cfg, trainer=trainer)

        if self.legacy_text_conditioning:
            tc_tokenizer = self.tokenizer.tokenizers[self.text_conditioning_tokenizer_name]
            tc_vocab_size = tc_tokenizer.vocab_size
            # In transformers v5+, T5Tokenizer is a fast tokenizer whose vocab_size includes
            # extra_id sentinel tokens (e.g. 32100 = 32000 + 100). Subtract them to match
            # the vocab size used when training legacy checkpoints.
            if hasattr(tc_tokenizer, '_extra_ids'):
                tc_vocab_size -= tc_tokenizer._extra_ids
            self.context_text_embedding = nn.Embedding(tc_vocab_size, cfg.embedding_dim)

        # This needs to happen after super().__init__()
        self._codec_model = codec_model
        self._codec_model.freeze()  # Lightning does requires_grad = False and self.eval()
        self._codec_converter = codec_converter
        self._codec_helper = CodecHelper(self._codec_model, self._codec_converter)

        audio_embeddings = []
        for _ in range(self.num_audio_codebooks * self.frame_stacking_factor):
            audio_embeddings.append(nn.Embedding(self.num_all_tokens_per_codebook, cfg.embedding_dim))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        # Identity projections required by LocalTransformerHelper methods.
        # MagpieTTSModel embeds directly in embedding_dim, so no projection is needed.
        self.audio_in_projection = nn.Identity()
        self.local_transformer_audio_out_projection = nn.Identity()

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
            }
            self.cas_encoder = CharAwareSubwordEncoder(
                d_embed=cfg.embedding_dim,
                llm_tokenizer_vocab=subword_vocab,
                subword_padding_idx=self.tokenizer.pad,
                special_vocab=special_vocab,
            )
        else:
            # Regular text embedding
            self.text_embedding = nn.Embedding(num_tokens, cfg.embedding_dim)

        self.encoder = transformer_2501.Transformer(**dict(cfg.encoder))
        self.decoder = transformer_2501.Transformer(**dict(cfg.decoder))

        self.final_proj = nn.Linear(
            cfg.decoder.d_model,
            self.num_audio_codebooks * self.num_all_tokens_per_codebook * self.frame_stacking_factor,
        )

        self.local_transformer_type = LocalTransformerType(cfg.get('local_transformer_type', 'none').lower())
        logging.info(f"Local transformer type: {self.local_transformer_type}")
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            local_transformer_hidden_dim = cfg.get('local_transformer_hidden_dim', 256)
            if local_transformer_hidden_dim != cfg.decoder.d_model:
                self.local_transformer_in_projection = nn.Linear(cfg.decoder.d_model, local_transformer_hidden_dim)
            else:
                self.local_transformer_in_projection = nn.Identity()
            self.local_transformer = transformer_2501.Transformer(
                n_layers=self.cfg.get('local_transformer_n_layers', 2),
                d_model=local_transformer_hidden_dim,
                d_ffn=local_transformer_hidden_dim * 4,
                sa_n_heads=self.cfg.get('local_transformer_n_heads', 1),
                kernel_size=1,
                is_causal=self.local_transformer_type == LocalTransformerType.AR,
                max_length_causal_mask=self.frame_stacking_factor * self.num_audio_codebooks + 2,
                use_learnable_pos_emb=True,
            )
            local_transformer_out_projections = []
            for _ in range(self.num_audio_codebooks * self.frame_stacking_factor):
                # Have a separate projection layer for each codebook, to distinguish between them
                local_transformer_out_projections.append(
                    nn.Linear(local_transformer_hidden_dim, self.num_all_tokens_per_codebook)
                )
            self.local_transformer_out_projections = nn.ModuleList(local_transformer_out_projections)

            self._lt_helper = LocalTransformerHelper(
                local_transformer=self.local_transformer,
                audio_embeddings=self.audio_embeddings,
                audio_in_projection=self.audio_in_projection,
                local_transformer_in_projection=self.local_transformer_in_projection,
                local_transformer_audio_out_projection=self.local_transformer_audio_out_projection,
                local_transformer_out_projections=self.local_transformer_out_projections,
                num_audio_codebooks=self.num_audio_codebooks,
                frame_stacking_factor=self.frame_stacking_factor,
                audio_eos_id=self.audio_eos_id,
                mask_token_id=self.mask_token_id,
                codebook_size=self.codebook_size,
            )

        if cfg.get('use_alignment_encoder', False):
            self.alignment_encoder = AlignmentEncoder(
                n_mel_channels=cfg.embedding_dim,
                n_text_channels=cfg.embedding_dim,
                dist_type="cosine",
                temperature=15.0,
            )

        if self.model_type == 'multi_encoder_context_tts':
            logging.warning(f"The multi_encoder_context_tts model type for {self} is deprecated.")

            # Transcript and context audio/text go to different encoders.
            # Output of the encoders goes to the decoder through the cross-attention layers
            self.transcript_decoder_layers = cfg.get('transcript_decoder_layers', [3, 4, 5, 6, 7, 8])
            self.context_decoder_layers = cfg.get(
                'context_decoder_layers', [0, 1, 2, 9, 10, 11]
            )  # For backward compatibility
            multi_encoder_mapping = [None for _ in range(self.decoder.n_layers)]
            for layer in self.transcript_decoder_layers:
                multi_encoder_mapping[layer] = 0  # 0 means text goes to this layer, 1 means context goes to this layer
            for layer in self.context_decoder_layers:
                multi_encoder_mapping[layer] = 1
            self.multi_encoder_mapping = multi_encoder_mapping
            # Create context encoder (filter out MoE loss coefficients if present)
            # Note: Loss coefficients are model-level config, not passed to Transformer module
            context_encoder_cfg = dict(cfg.context_encoder)
            context_encoder_cfg.pop('router_load_balancing_loss_coeff', None)
            context_encoder_cfg.pop('router_z_loss_coeff', None)
            self.context_encoder = transformer_2501.Transformer(**context_encoder_cfg)
        elif self.model_type == 'decoder_context_tts':
            # Context audio/text goes directly to the decoder (before the target audio codes)
            self.transcript_decoder_layers = [
                idx for idx in range(self.decoder.n_layers)
            ]  # All layers are used for text
        elif self.model_type == 'decoder_ce':
            # Similar to decoder_context_tts, but we use context encoder
            # Decoder gets output from context encoder instead of raw context tokens embeddings
            context_encoder_cfg = dict(cfg.context_encoder)
            context_encoder_cfg.pop('router_load_balancing_loss_coeff', None)
            context_encoder_cfg.pop('router_z_loss_coeff', None)
            self.context_encoder = transformer_2501.Transformer(**context_encoder_cfg)
            self.transcript_decoder_layers = [
                idx for idx in range(cfg.decoder.n_layers)
            ]  # All layers are used for text
            # Baked context embedding: nn.Embedding with flattened (N, T*D), reshaped to (N, T, D) at retrieval
            # register_buffer does not work with nn.Embedding, so we use a regular variable.
            self.baked_context_embedding: Optional[nn.Embedding] = None
            self.register_buffer('_baked_embedding_T', None)  # Time dimension
            self.register_buffer('_baked_embedding_D', None)  # Embedding dimension
            self.register_buffer('baked_context_embedding_len', None)  # Per-speaker lengths (N,)
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.alignment_loss_scale = cfg.get('alignment_loss_scale', 0.0)
        self.alignment_encoder_loss_scale = cfg.get('alignment_encoder_loss_scale', 0.0)
        if self.alignment_loss_scale > 0.0:
            self.alignment_loss = ForwardSumLoss(loss_scale=self.alignment_loss_scale)
        if self.alignment_encoder_loss_scale > 0.0:
            self.alignment_encoder_loss = ForwardSumLoss(loss_scale=self.alignment_encoder_loss_scale)

        # Initialize MoE losses if MoE is enabled in decoder
        self.use_moe = cfg.get('use_moe', False)
        if self.use_moe:
            num_experts = cfg.decoder.get('num_experts', 8)
            routing_strategy = cfg.decoder.get('routing_strategy', 'top_k')

            router_load_balancing_loss_coeff = cfg.get('router_load_balancing_loss_coeff', 0.01)
            router_z_loss_coeff = cfg.get('router_z_loss_coeff', 0.001)

            # Sinkhorn routing already ensures balanced expert assignment through its doubly stochastic property
            # Load balancing loss is redundant and incompatible with Sinkhorn
            if routing_strategy == 'sinkhorn' and router_load_balancing_loss_coeff > 0:
                raise ValueError(
                    f"Invalid configuration: routing_strategy='sinkhorn' with router_load_balancing_loss_coeff={router_load_balancing_loss_coeff} > 0. "
                    f"Sinkhorn routing already ensures balanced expert load through doubly stochastic constraints. "
                    f"Set router_load_balancing_loss_coeff=0.0 when using Sinkhorn routing to avoid redundant penalization."
                )

            self.moe_auxiliary_loss = MoEAuxiliaryLoss(
                num_experts=num_experts,
                load_balancing_loss_scale=router_load_balancing_loss_coeff,
                router_z_loss_scale=router_z_loss_coeff,
            )
            logging.info(
                f"MoE enabled in decoder with {num_experts} experts, routing_strategy={routing_strategy}. "
                f"Each expert has d_ffn={cfg.decoder.d_ffn}. "
                f"Loss scales: router_load_balancing={router_load_balancing_loss_coeff}, router_z={router_z_loss_coeff}"
            )
            # Training-side accumulator for layer-wise expert usage heatmap.
            # Accumulated every training_step, rendered + reset at each validation interval.
            self._moe_num_experts = num_experts
            self._moe_train_layer_usage_accum: Optional[torch.Tensor] = None  # (n_layers, num_experts)
            self._moe_train_accum_steps: int = 0

        # Define cfg parameters into self parameters
        self.prior_end_step = self.cfg.prior_end_step
        self.prior_scaledown_start_step = self.cfg.prior_scaledown_start_step
        self.indefinite_prior_prob = self.cfg.get('indefinite_prior_prob', 0.0)
        self.ctc_prior_layer_ids = self.cfg.get('ctc_prior_layer_ids', self.transcript_decoder_layers)
        self.cfg_unconditional_prob = self.cfg.get('cfg_unconditional_prob', 0.0)
        self.decoder_input_dropout_prob = self.cfg.get('decoder_input_dropout_prob', 0.0)
        self.binarize_attn_method = self.cfg.get('binarize_attn_method', 'argmax')
        self.binarize_repeat_audio_factor = self.cfg.get('binarize_repeat_audio_factor', 2)
        self.prior_future_decay = self.cfg.get('prior_future_decay', 1.0)
        self.prior_past_decay = self.cfg.get('prior_past_decay', 1.0)
        self.binarized_prior_epsilon = self.cfg.get('binarized_prior_epsilon', 0.0)
        self.prior_future_context = self.cfg.get('prior_future_context', 1)
        self.prior_past_context = self.cfg.get('prior_past_context', 1)
        self.binarize_prior_after_step = self.cfg.get('binarize_prior_after_step', 0)
        self.codebook_loss_scale = self.cfg.get('codebook_loss_scale', 1.0)
        self.local_transformer_loss_scale = self.cfg.get('local_transformer_loss_scale', 1.0)
        self.use_alignment_encoder = self.cfg.get('use_alignment_encoder', False)
        self.use_prior_for_aligner = self.cfg.get('use_prior_for_aligner', False)
        self.aligner_encoder_train_steps = self.cfg.get('aligner_encoder_train_steps', float('inf'))
        self.dec_random_input_max = self.cfg.get('dec_random_input_max', self.num_all_tokens_per_codebook)

        # Configuration validity checks
        self.check_frame_stacking_config_validity()

        # Class-level cache for text normalizers. Used during inference.
        self._text_normalizers: Dict[str, Any] = {}

        # Chunked inference configuration (immutable tuning parameters)
        self.chunked_inference_config = ChunkedInferenceConfig()

    def _register_tokenizer_artifacts(self, cfg: DictConfig) -> None:
        """
        Register tokenizer file artifacts (phoneme_dict, heteronyms, etc.) for .nemo packaging.

        This method iterates through all tokenizer configs and registers any local file paths
        as artifacts. When the model is saved to a .nemo file, these files will be packaged
        inside the archive and automatically restored when loading from .nemo.

        Supported artifact types:
        - g2p.phoneme_dict: Phoneme dictionary file for G2P conversion
        - g2p.heteronyms: Heteronyms file for G2P conversion

        Args:
            cfg: Model configuration containing text_tokenizers config
        """
        if 'text_tokenizers' not in cfg:
            return

        for tokenizer_name in cfg.text_tokenizers:
            tokenizer_cfg = cfg.text_tokenizers[tokenizer_name]

            # Skip HuggingFace tokenizers (AutoTokenizer, T5Tokenizer) - they don't need local files
            if hasattr(tokenizer_cfg, '_target_') and tokenizer_cfg._target_ in ['AutoTokenizer', 'T5Tokenizer']:
                continue

            # Register G2P artifacts if present
            if hasattr(tokenizer_cfg, 'g2p') and tokenizer_cfg.g2p is not None:
                g2p_cfg = tokenizer_cfg.g2p

                # Register phoneme_dict (or resolve nemo: path if restoring from .nemo)
                phoneme_dict_path = (
                    g2p_cfg.get('phoneme_dict', None)
                    if hasattr(g2p_cfg, 'get')
                    else getattr(g2p_cfg, 'phoneme_dict', None)
                )
                if phoneme_dict_path and isinstance(phoneme_dict_path, str) and phoneme_dict_path.strip():
                    try:
                        # register_artifact handles both:
                        # - Local paths: registers for .nemo packaging, returns absolute path
                        # - nemo: paths: resolves to extracted file location
                        artifact_path = self.register_artifact(
                            f'text_tokenizers.{tokenizer_name}.g2p.phoneme_dict',
                            phoneme_dict_path,
                            verify_src_exists=True,
                        )
                        if artifact_path:
                            with open_dict(cfg):
                                cfg.text_tokenizers[tokenizer_name].g2p.phoneme_dict = artifact_path
                    except FileNotFoundError:
                        logging.warning(
                            f"phoneme_dict file not found for tokenizer '{tokenizer_name}': "
                            f"{phoneme_dict_path}. Artifact will not be packaged in .nemo file."
                        )

                # Register heteronyms (or resolve nemo: path if restoring from .nemo)
                heteronyms_path = (
                    g2p_cfg.get('heteronyms', None)
                    if hasattr(g2p_cfg, 'get')
                    else getattr(g2p_cfg, 'heteronyms', None)
                )
                if heteronyms_path and isinstance(heteronyms_path, str) and heteronyms_path.strip():
                    try:
                        artifact_path = self.register_artifact(
                            f'text_tokenizers.{tokenizer_name}.g2p.heteronyms',
                            heteronyms_path,
                            verify_src_exists=True,
                        )
                        if artifact_path:
                            with open_dict(cfg):
                                cfg.text_tokenizers[tokenizer_name].g2p.heteronyms = artifact_path
                    except FileNotFoundError:
                        logging.warning(
                            f"heteronyms file not found for tokenizer '{tokenizer_name}': "
                            f"{heteronyms_path}. Artifact will not be packaged in .nemo file."
                        )

    def _setup_inference_parameters(self, cfg: DictConfig) -> None:
        """
        Create the self.inference_parameters which instantiates the InferenceParameters dataclass
        """
        self.inference_parameters = ModelInferenceParameters.from_dict(cfg.get("inference_parameters", {}))

    def _get_state_dict_keys_to_exclude(self):
        """
        We remove _speaker_verification_model and _codec_model
        from the checkpoint and optimizer param groups. The codec model is saved in a separate checkpoint.
        _speaker_verification_model is only included in older checkpoints with the older single_encoder_sv_tts
        model_type that is no longer supported and can likely be removed in a future version.
        If the model has a baked context embedding, the context_encoder weights are also excluded
        since they are no longer needed for inference.
        """
        keys = ['_speaker_verification_model', '_codec_model']
        if self.has_baked_context_embedding:
            keys.append('context_encoder')
        return keys

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Only used for saving checkpoints.
        We exclude the keys in the state_dict that are in the list returned by _get_state_dict_keys_to_exclude.
        """
        if hasattr(self, '_no_state_dict') and self._no_state_dict:
            return {}
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_substrings_to_exclude = self._get_state_dict_keys_to_exclude()
        for key in list(state_dict.keys()):
            if any(substring in key for substring in keys_substrings_to_exclude):
                del state_dict[key]
        return state_dict

    def setup_optimizer_param_groups(self):
        """Exclude frozen eval/inference-only models from the optimizer.
        Saves memory by excluding the keys in the state_dict that are in the list returned by _get_state_dict_keys_to_exclude.
        """
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

    def check_frame_stacking_config_validity(self):
        """
        Check if the configuration is compatible with frame stacking.
        """
        if self.frame_stacking_factor > 1:
            # The settings below are not supported with frame stacking.
            # Some of them may work - but they have not been tested.

            # disallow alignment encoder
            if self.use_alignment_encoder:
                raise ValueError("Alignment encoder is not supported for frame stacking")
            # disallow alignment loss
            if self.alignment_loss_scale > 0.0:
                raise ValueError("Alignment loss is not supported for frame stacking")
            # disallow training prior
            if self.cfg.prior_scaling_factor is not None and self.cfg.prior_scaling_factor > 0:
                raise ValueError("Training-time attention prior is not supported for frame stacking")
            # disallow text conditioning
            if self.use_text_conditioning_encoder:
                raise ValueError("Text conditioning is not supported for frame stacking")

    @property
    def has_baked_context_embedding(self) -> bool:
        """Check if the model has a baked context embedding.

        Returns:
            True if baked_context_embedding is set with valid dimensions.
        """
        return (
            self.model_type == 'decoder_ce'
            and self.baked_context_embedding is not None
            and self._baked_embedding_T is not None
            and self._baked_embedding_D is not None
        )

    @property
    def num_baked_speakers(self) -> int:
        """Return number of baked speakers.

        Returns:
            0 if no baked embedding, N for embedding with N speakers.
        """
        if not self.has_baked_context_embedding:
            return 0
        return self.baked_context_embedding.num_embeddings

    @property
    def validation_step_outputs(self):
        """Always use list-of-lists structure for uniform single/multi-dataloader handling.

        Overrides ModelPT which uses a flat list for single dataloader and list-of-lists
        for multiple dataloaders. This override always returns list-of-lists so that
        validation_step, on_validation_epoch_end, etc. don't need conditional branching.
        """
        if self._validation_step_outputs is not None:
            return self._validation_step_outputs
        num_dl = len(self._validation_dl) if self._validation_dl is not None else 1
        self._validation_step_outputs = [[] for _ in range(num_dl)]
        return self._validation_step_outputs

    @validation_step_outputs.setter
    def validation_step_outputs(self, value):
        self._validation_step_outputs = value

    def _normalize_speaker_indices(
        self,
        speaker_indices: Optional[Union[int, List[int], torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize speaker_indices to a tensor of shape (batch_size,).

        Args:
            speaker_indices: Speaker selection. Can be:
                - None: Use first speaker (index 0) for all batch elements
                - int: Same speaker for all batch elements
                - List[int] or Tensor: One speaker index per batch element
            batch_size: Number of elements in the batch.
            device: Device to create tensor on.

        Returns:
            Tensor of shape (batch_size,) with speaker indices.

        Raises:
            ValueError: If speaker_indices length doesn't match batch_size or indices are out of range.
        """
        # Default to first speaker (index 0) if none specified
        if speaker_indices is None:
            speaker_indices = 0

        # Normalize to tensor
        if isinstance(speaker_indices, int):
            indices = torch.full((batch_size,), speaker_indices, dtype=torch.long, device=device)
        elif isinstance(speaker_indices, list):
            if len(speaker_indices) != batch_size:
                raise ValueError(
                    f"speaker_indices length ({len(speaker_indices)}) must match batch_size ({batch_size})"
                )
            indices = torch.tensor(speaker_indices, dtype=torch.long, device=device)
        elif isinstance(speaker_indices, torch.Tensor):
            if speaker_indices.numel() != batch_size:
                raise ValueError(
                    f"speaker_indices length ({speaker_indices.numel()}) must match batch_size ({batch_size})"
                )
            indices = speaker_indices.to(device=device, dtype=torch.long)
        else:
            raise ValueError(f"speaker_indices must be int, list, or tensor, got {type(speaker_indices)}")

        # Validate indices
        if (indices < 0).any() or (indices >= self.num_baked_speakers).any():
            raise ValueError(
                f"speaker_indices values must be in range [0, {self.num_baked_speakers - 1}], "
                f"got min={indices.min().item()}, max={indices.max().item()}"
            )

        return indices

    def get_baked_context_embeddings_batch(
        self,
        batch_size: int,
        speaker_indices: Optional[Union[int, List[int], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get baked context embeddings for a batch, with per-element speaker selection.

        Args:
            batch_size: Number of elements in the batch.
            speaker_indices: Speaker selection. Can be:
                - None: Use first speaker (index 0) for all batch elements
                - int: Same speaker for all batch elements
                - List[int] or Tensor: One speaker index per batch element (length must match batch_size)

        Returns:
            Tuple of (embeddings, lengths) where:
                - embeddings: (B, T, D) tensor
                - lengths: (B,) tensor with embedding lengths per batch element

        Raises:
            ValueError: If speaker_indices length doesn't match batch_size or indices are out of range.
        """
        if not self.has_baked_context_embedding:
            raise ValueError("No baked context embedding available")

        device = self.baked_context_embedding.weight.device
        indices = self._normalize_speaker_indices(speaker_indices, batch_size, device)

        # Lookup flattened embeddings via nn.Embedding: (B,) -> (B, T*D)
        flat_embeddings = self.baked_context_embedding(indices)

        # Reshape to 3D: (B, T*D) -> (B, T, D)
        T = self._baked_embedding_T.item()
        D = self._baked_embedding_D.item()
        embeddings = flat_embeddings.view(batch_size, T, D)

        lengths = self.baked_context_embedding_len[indices]  # (B,)
        return embeddings, lengths

    def update_ckpt(self, state_dict):
        """
        Backward compatibility for checkpoints saved with old model names.
        """
        new_state_dict = {}
        for key in state_dict.keys():
            if 't5_encoder' in key:
                new_key = key.replace('t5_encoder', 'encoder')
                new_state_dict[new_key] = state_dict[key]
            elif 't5_decoder' in key:
                new_key = key.replace('t5_decoder', 'decoder')
                new_state_dict[new_key] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        return new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Modify load_state_dict so that we don't restore weights to _speaker_verification_model and _codec_model when
        strict is True.
        When strict is False, we can call pytorch's load_state_dict.
        When strict is True, we loop through all parameters and rename them to enable loading.

        _speaker_verification_model is only included in older checkpoints with the older single_encoder_sv_tts
        model_type that is no longer supported and can likely be removed in a future version.

        Also handles loading baked context embeddings. If the checkpoint contains baked_speaker_embedding.weight,
        context_encoder weights are not expected to be present. The embedding is stored in flattened format
        (N, T*D) and reconstructed to (N, T, D) at inference time using stored T and D dimensions.
        """
        state_dict = self.update_ckpt(state_dict)

        # Check if checkpoint has baked context embedding (nn.Embedding format)
        has_baked_embedding_in_ckpt = 'baked_context_embedding.weight' in state_dict

        # Load baked embedding if present
        if has_baked_embedding_in_ckpt:
            weight = state_dict['baked_context_embedding.weight']  # (N, T*D)
            self._baked_embedding_T = state_dict['_baked_embedding_T']
            self._baked_embedding_D = state_dict['_baked_embedding_D']
            self.baked_context_embedding_len = state_dict['baked_context_embedding_len']

            num_speakers = weight.size(0)
            embedding_dim = weight.size(1)
            T = self._baked_embedding_T.item()
            D = self._baked_embedding_D.item()

            # Create nn.Embedding and load weights (no gradients for inference)
            self.baked_context_embedding = nn.Embedding(num_speakers, embedding_dim)
            self.baked_context_embedding.weight.data = weight
            self.baked_context_embedding.weight.requires_grad_(False)

            logging.info(
                f"Loaded baked context embedding: num_speakers={num_speakers}, T={T}, D={D}, "
                f"shape=({num_speakers}, {embedding_dim}), lengths={self.baked_context_embedding_len.tolist()}"
            )

        if not strict:
            super().load_state_dict(state_dict, strict=False)

        # Build list of modules to skip
        modules_to_skip = [
            '_speaker_verification_model',
            '_codec_model',
            '_reference_model',
            'eval_asr_model',
            'eval_speaker_verification_model',
            'whisper_model',
            'squim_objective_model',
        ]
        # Skip context_encoder if checkpoint has baked embedding (weights won't be in checkpoint)
        if has_baked_embedding_in_ckpt:
            modules_to_skip.append('context_encoder')

        for name, child in self.named_children():
            if name in modules_to_skip:
                continue
            if any(param.numel() > 0 for param in child.parameters()):
                # If the module has parameters, we want to change the default mapping so that the state_dict gets
                # loaded.
                # Ex: state_dict[encoder.position_embeddings.weight] -> new_state_dict[position_embeddings.weight]
                new_state_dict = {}
                for key in state_dict.keys():
                    name_with_dot = f"{name}."
                    if key.startswith(name_with_dot):
                        new_state_dict[key[len(name_with_dot) :]] = state_dict[key]
                child.load_state_dict(new_state_dict)

    def embed_audio_tokens(self, audio_tokens, audio_tokens_lens):
        B, C, T = audio_tokens.shape
        audio_tokens = pad_audio_codes(audio_tokens, self.frame_stacking_factor).long()
        audio_embedding = None
        for i in range(self.frame_stacking_factor):
            for c in range(C):
                tokens = audio_tokens[:, c, i :: self.frame_stacking_factor]
                embedding = self.audio_embeddings[c + i * C](tokens)
                if audio_embedding is None:
                    audio_embedding = embedding
                else:
                    audio_embedding += embedding
        audio_embedding = audio_embedding / (C * self.frame_stacking_factor)  # [B, T, E]

        audio_embedding_lens = torch.ceil(audio_tokens_lens / self.frame_stacking_factor).long()
        mask = get_mask_from_lengths(audio_embedding_lens)
        audio_embedding = audio_embedding * mask.unsqueeze(2)

        return audio_embedding, audio_embedding_lens

    def compute_loss(self, logits, audio_codes, audio_codes_lens, mask_tokens_mask=None, frame_stacking_factor=1):
        """
        Computes the audio codebook loss. Used by:

        (1) The main Magpie-TTS transformer
        (2) The local transformer, for both autoregressive and MaskGit methods

        Args:
            logits: (B, T', num_codebooks * num_tokens_per_codebook)
            audio_codes: (B, C, T')
            audio_codes_lens: (B,)
            mask_tokens_mask: (B, C, T') True for tokens that were replaced with the MASK_TOKEN and should
                therefore be the only ones included in the loss computation (for MaskGit).
            frame_stacking_factor: int, the stacking factor used in the model
        """
        loss_mask = get_mask_from_lengths(audio_codes_lens, pad_to_factor=frame_stacking_factor)
        if mask_tokens_mask is not None:
            # For MaskGit we only compute loss for the masked tokens.
            # *Both* conditions must be true:
            # 1. the token is masked
            # 2. the token is not padding
            loss_mask = loss_mask.unsqueeze(1) * mask_tokens_mask
            if not loss_mask.any():
                # Without this we were very rarely getting NaNs in the loss
                logging.warning("No tokens valid were found in compute_loss()!")
                return torch.tensor(0.0, device=loss_mask.device), loss_mask
        else:
            # repeat loss mask for each codebook to simplify code below
            loss_mask = loss_mask.unsqueeze(1).repeat(1, audio_codes.size(1), 1)
        total_codebook_loss = None
        audio_codes = pad_audio_codes(audio_codes, self.frame_stacking_factor).long()
        for fs_index in range(frame_stacking_factor):
            for codebook in range(audio_codes.size(1)):
                si = (codebook + self.num_audio_codebooks * fs_index) * self.num_all_tokens_per_codebook
                ei = si + self.num_all_tokens_per_codebook
                codebook_logits = logits[:, :, si:ei]  # (B, T', num_tokens_per_codebook)
                codebook_targets = audio_codes[:, codebook, fs_index::frame_stacking_factor]  # (B, T')
                codebook_loss = self.cross_entropy_loss(
                    codebook_logits.permute(0, 2, 1), codebook_targets  # (B, num_tokens_per_codebook, T')
                )  # (B, T')
                codebook_loss_mask = loss_mask[:, codebook, fs_index::frame_stacking_factor]
                codebook_loss = codebook_loss * codebook_loss_mask
                if codebook_loss_mask.sum() == 0:
                    logging.warning(f"Loss mask for codebook {codebook} is all zeros, global_step: {self.global_step}")
                    continue
                codebook_loss = codebook_loss.sum() / codebook_loss_mask.sum()
                if total_codebook_loss is None:
                    total_codebook_loss = codebook_loss
                else:
                    total_codebook_loss = total_codebook_loss + codebook_loss

        total_codebook_loss = total_codebook_loss / (audio_codes.size(1) * frame_stacking_factor)
        return total_codebook_loss, loss_mask

    def forward(self, dec_input_embedded, dec_input_mask, cond, cond_mask, attn_prior, multi_encoder_mapping):
        """
        Forward pass through the decoder transformer, followed by a linear projection to audio codebook logits.

        Args:
            dec_input_embedded (torch.Tensor): Embedded decoder input of shape (B, T, C).
            dec_input_mask (torch.Tensor): Boolean mask for decoder input of shape (B, T).
            cond (torch.Tensor or List[torch.Tensor]): Conditioning tensor(s) for cross-attention.
            cond_mask (torch.Tensor or List[torch.Tensor]): Mask(s) for conditioning tensor(s).
            attn_prior (torch.Tensor or None): Prior attention weights for cross-attention.
            multi_encoder_mapping (List[Optional[int]] or None): Per-layer mapping to conditioning inputs.

        Returns:
            Tuple of:

            - all_code_logits (torch.Tensor): Logits of shape (B, T', num_codebooks * num_tokens_per_codebook).
            - attn_probabilities (list): Attention probabilities from each decoder layer.
            - dec_output (torch.Tensor): Raw decoder output of shape (B, T', d_model).
            - moe_routing_info (list or None): None if MoE is disabled. If MoE is enabled,
              a list of dicts (one per layer) each containing:

              - 'router_logits' (torch.Tensor): Raw router logits (B, T, num_experts).
              - 'router_probs' (torch.Tensor): Router probabilities (B, T, num_experts).
              - 'expert_indices' (torch.Tensor): Selected expert indices (B, T, top_k).
        """
        decoder_out = self.decoder(
            dec_input_embedded,
            dec_input_mask,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=attn_prior,
            multi_encoder_mapping=multi_encoder_mapping,
        )
        attn_probabilities = decoder_out['attn_probabilities']
        moe_routing_info = decoder_out.get('moe_routing_info', None)  # Extract MoE routing info for loss computation
        all_code_logits = self.final_proj(decoder_out['output'])  # (B, T', num_codebooks * num_tokens_per_codebook)
        return all_code_logits, attn_probabilities, decoder_out['output'], moe_routing_info

    def logits_to_audio_codes(self, all_code_logits, audio_codes_lens):
        # all_code_logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # audio_codes_lens: (B,)
        all_preds = [[] for _ in range(self.frame_stacking_factor)]
        for fs_index in range(self.frame_stacking_factor):
            for idx in range(self.num_audio_codebooks):
                si = (idx + self.num_audio_codebooks * fs_index) * self.num_all_tokens_per_codebook
                ei = si + self.num_all_tokens_per_codebook
                codebook_logits = all_code_logits[:, :, si:ei]
                codebook_probs = torch.softmax(codebook_logits, dim=-1)  # (B, T', num_tokens_per_codebook)
                # argmax to get the tokens
                codebook_preds = torch.argmax(codebook_probs, dim=-1)  # (B, T')
                all_preds[fs_index].append(codebook_preds)
        all_preds = [
            torch.stack(p, dim=1) for p in all_preds
        ]  # list of `frame_stacking_factor`` elements of shape (B,C,T) each
        all_preds = torch.stack(all_preds, dim=-1)  # B, C, T, frame_stacking_factor
        # undo the frame stacking
        all_preds = all_preds.reshape(all_preds.size(0), all_preds.size(1), -1)  # B, C, T*frame_stacking_factor
        pred_max_len = all_preds.size(2)
        real_max_len = audio_codes_lens.max()
        assert (pred_max_len - real_max_len) < self.frame_stacking_factor
        # trim padding introduced for frame stacking
        all_preds = all_preds[:, :, :real_max_len]
        audio_mask = get_mask_from_lengths(audio_codes_lens)
        all_preds = all_preds * audio_mask.unsqueeze(1)

        return all_preds

    def visualize_codes(self, codes, mask_id=2020, frame_stacking_rate=2):
        """
        Visualize codes for analysis purposes
        codes: (B, C)
        """

        def code_to_str(code):
            if code == mask_id:
                return "M    "
            else:
                return f"{code:04d} "

        B, C = codes.shape
        if B > 1:
            logging.debug("Warning: visualizing only first batch element")
        codes = codes.clone().detach().cpu().numpy()[0]
        codes = [code_to_str(c) for c in codes]
        output_str = ""
        for i, c in enumerate(codes):
            if (i) % (C / frame_stacking_rate) == 0:
                output_str += "|timestep| "
            output_str += c
        logging.debug(output_str)

    def sample_codes_from_logits(
        self,
        all_code_logits_t: torch.Tensor,
        temperature: float = 0.7,
        topk: int = 80,
        unfinished_items: Dict[int, bool] = {},
        finished_items: Dict[int, bool] = {},
        forbid_audio_eos: bool = False,
    ) -> torch.Tensor:
        """
        Sample codes for all codebooks at a given timestep. Uses multinomial sampling
        with temperature and top-k. If frame stacking is on (i.e. `frame_stacking_factor
        > 1`), this function will sample across the entire frame stack.

        Special handling:
        * forbids special tokens (like AUDIO_BOS, AUDIO_CONTEXT_EOS, etc.) from being sampled
        * forces / forbids EOS for finished / unfinished items respectively
        * optionally, globally forbids audio EOS (useful early in the generation process)

        Args:
            all_code_logits_t (torch.Tensor): Logits at a given timestep with shape
                (B, num_tokens_per_codebook * num_codebooks * frame_stacking_factor)
            temperature (float, optional): Sampling temperature
            topk (int, optional): Number of top-probability tokens to consider in sampling.
            unfinished_items (dict, optional): Dictionary containing indices of batch
            items that we are confident have not completed generation. For these items, audio EOS
                sampling is forbidden.
            finished_items (dict, optional): Dictionary containing indices of batch
                items that we are confident are completed. For these items, audio EOS sampling
                is forced.
            forbid_audio_eos (bool, optional): Whether to globally forbid audio EOS for the entire
                batch.

        Returns:
            torch.Tensor: Sampled audio codes with shape (B, num_codebooks, frame_stacking_factor).
        """
        all_preds = [[] for _ in range(self.frame_stacking_factor)]
        for fs_index in range(self.frame_stacking_factor):
            for idx in range(self.num_audio_codebooks):
                si = (idx + self.num_audio_codebooks * fs_index) * self.num_all_tokens_per_codebook
                ei = si + self.num_all_tokens_per_codebook
                codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)

                for item_idx in unfinished_items:
                    codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
                for item_idx in finished_items:
                    codebook_logits[item_idx, :] = float('-inf')
                    codebook_logits[item_idx, self.audio_eos_id] = 0.0

                # Disallow generation of special tokens
                codebook_logits = clear_forbidden_logits(
                    codebook_logits.unsqueeze(1), self.codebook_size, forbid_audio_eos=forbid_audio_eos
                ).squeeze(1)

                codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
                indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                    -1
                )  # (B, num_tokens_per_codebook)
                codebook_logits_rescored = codebook_logits.clone()
                codebook_logits_rescored[indices_to_remove] = float('-inf')

                codebook_probs = torch.softmax(
                    codebook_logits_rescored / temperature, dim=-1
                )  # (B, num_tokens_per_codebook)
                codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
                all_preds[fs_index].append(codebook_preds)

        all_preds = [
            torch.cat(ds_preds, dim=1) for ds_preds in all_preds
        ]  # list of `frame_stacking_factor` elements, each of shape (B, num_codebooks)
        all_preds = torch.stack(all_preds, dim=2)  # (B, num_codebooks, frame_stacking_factor)
        return all_preds

    def _prepare_attention_images(
        self,
        attention_prob_matrix: List[torch.Tensor],
        audio_codes_lens: torch.Tensor,
        text_lens: torch.Tensor,
        dec_context_size: int = 0,
        max_examples: int = 3,
    ) -> List[np.ndarray]:
        """
        Convert attention probability matrices to numpy images for logging.

        Args:
            attention_prob_matrix: List of attention tensors, each (B, H, audio_timesteps, text_timesteps).
            audio_codes_lens: Audio sequence lengths per example.
            text_lens: Text sequence lengths per example.
            dec_context_size: Number of context audio frames to skip in attention visualization.
            max_examples: Maximum number of examples to generate images for.

        Returns:
            List of numpy arrays in HWC format, one per example.
        """
        with torch.no_grad():
            # Concatenate attention heads and average
            attention_prob_matrix = torch.cat(attention_prob_matrix, dim=1)  # (B, C, audio_timesteps, text_timesteps)
            attention_prob_matrix_mean = attention_prob_matrix.mean(dim=1)  # (B, audio_timesteps, text_timesteps)

            images = []
            num_examples = min(max_examples, attention_prob_matrix_mean.size(0))
            for idx in range(num_examples):
                # Slice attention matrix to valid region (excluding context frames)
                audio_len = int(audio_codes_lens[idx])
                text_len = int(text_lens[idx])
                item_attn_matrix = attention_prob_matrix_mean[idx][
                    dec_context_size : dec_context_size + audio_len, :text_len
                ]
                item_attn_matrix = item_attn_matrix.detach().cpu().numpy()
                img_np = plot_alignment_to_numpy(item_attn_matrix.T)
                images.append(img_np)

            return images

    def _prepare_audio_examples(
        self,
        logits: torch.Tensor,
        target_audio_codes: torch.Tensor,
        audio_codes_lens: torch.Tensor,
        context_audio_codes: Optional[torch.Tensor] = None,
        context_audio_codes_lens: Optional[torch.Tensor] = None,
        max_examples: int = 3,
    ) -> Dict[str, List[Optional[np.ndarray]]]:
        """
        Decode audio codes to waveforms and convert to numpy arrays for logging.

        Args:
            logits: Model output logits to convert to predicted audio.
            target_audio_codes: Ground truth audio codes.
            audio_codes_lens: Lengths of target audio codes.
            context_audio_codes: Optional context audio codes for voice cloning.
            context_audio_codes_lens: Lengths of context audio codes.
            max_examples: Maximum number of examples to process.

        Returns:
            Dict with keys 'pred_audios', 'target_audios', 'context_audios',
            each containing a list of numpy arrays (or None for context if unavailable).
        """
        with torch.no_grad():
            # Decode predictions: convert logits to codes, remove EOS token, then decode to audio
            pred_audio_codes = self.logits_to_audio_codes(logits, audio_codes_lens)
            pred_audio_codes, pred_audio_codes_lens = remove_eos_token(
                codes=pred_audio_codes, codes_len=audio_codes_lens
            )
            pred_audio, pred_audio_lens, _ = self._codec_helper.codes_to_audio(pred_audio_codes, pred_audio_codes_lens)

            # Decode targets: remove EOS token, then decode to audio
            target_audio_codes, target_audio_codes_lens = remove_eos_token(
                codes=target_audio_codes, codes_len=audio_codes_lens
            )
            target_audio, target_audio_lens, _ = self._codec_helper.codes_to_audio(
                target_audio_codes, target_audio_codes_lens
            )

            # Decode context audio if available (shape check ensures it's not a dummy tensor used in text context)
            # This does not handle the case in which a batch has a mixture of text and audio context examples
            context_audio, context_audio_lens = None, None
            if context_audio_codes is not None and context_audio_codes.shape[2] > 3:
                context_audio_codes, context_audio_codes_lens = remove_special_tokens(
                    codes=context_audio_codes, codes_len=context_audio_codes_lens
                )
                context_audio, context_audio_lens, _ = self._codec_helper.codes_to_audio(
                    context_audio_codes, context_audio_codes_lens
                )

            pred_audios = []
            target_audios = []
            context_audios = []

            num_examples = min(max_examples, pred_audio.size(0))
            for idx in range(num_examples):
                # Convert to numpy and trim to actual length
                pred_audio_np = pred_audio[idx, : pred_audio_lens[idx]].float().cpu().numpy()
                target_audio_np = target_audio[idx, : target_audio_lens[idx]].float().cpu().numpy()

                pred_audios.append(pred_audio_np)
                target_audios.append(target_audio_np)

                if context_audio is not None:
                    context_audio_np = context_audio[idx, : context_audio_lens[idx]].float().cpu().numpy()
                    context_audios.append(context_audio_np)
                else:
                    context_audios.append(None)

            return {
                'pred_audios': pred_audios,
                'target_audios': target_audios,
                'context_audios': context_audios,
            }

    def _collect_wandb_media_and_log_tb(
        self,
        *,
        dataset_prefix: str,
        pred_audios: List[np.ndarray],
        target_audios: List[np.ndarray],
        context_audios: List[Optional[np.ndarray]],
        attention_data: Dict[str, List[np.ndarray]],
        global_step: int,
    ) -> Dict[str, Any]:
        """
        Collect WandB media entries and log audio/attention to TensorBoard.

        TensorBoard logging happens directly within this method.
        WandB media is returned as a dict to be merged with other WandB media
        (e.g., MoE heatmaps) into a single wandb.log() call by the caller,
        ensuring all media shares the same WandB step index.

        Args:
            dataset_prefix: Prefix for log keys (e.g., 'val', 'val_set_0').
            pred_audios: List of predicted audio waveforms as numpy arrays.
            target_audios: List of target audio waveforms as numpy arrays.
            context_audios: List of context audio waveforms (or None per entry if unavailable).
            attention_data: Dict mapping attention names to lists of numpy images.
            global_step: Current training step for logging.

        Returns:
            Dict of WandB-ready media entries (audio + attention images).
            Empty dict if no WandB logger is configured.
        """
        wandb_media: Dict[str, Any] = {}

        for logger in self.loggers:
            is_wandb = isinstance(logger, WandbLogger)
            is_tb = isinstance(logger, TensorBoardLogger)
            if not is_wandb and not is_tb:
                raise ValueError(
                    f"Unsupported logger type: {type(logger)}. "
                    f"Only WandbLogger and TensorBoardLogger are supported for media logging."
                )

            for idx, (pred_audio_np, target_audio_np, context_audio_np) in enumerate(
                zip(pred_audios, target_audios, context_audios)
            ):
                if is_wandb:
                    audio_list = []
                    if context_audio_np is not None and context_audio_np.shape[0] > 0:
                        audio_list.append(
                            wandb.Audio(context_audio_np, sample_rate=self.output_sample_rate, caption="context")
                        )
                    audio_list.append(
                        wandb.Audio(pred_audio_np, sample_rate=self.output_sample_rate, caption="prediction")
                    )
                    audio_list.append(
                        wandb.Audio(target_audio_np, sample_rate=self.output_sample_rate, caption="target")
                    )
                    wandb_media[f"Audio:{dataset_prefix}/Example_{idx:02d}"] = audio_list

                if is_tb:
                    if context_audio_np is not None and context_audio_np.shape[0] > 0:
                        logger.experiment.add_audio(
                            f'{dataset_prefix}/Example_{idx}/context',
                            context_audio_np,
                            global_step=global_step,
                            sample_rate=self.output_sample_rate,
                        )
                    logger.experiment.add_audio(
                        f'{dataset_prefix}/Example_{idx}/prediction',
                        pred_audio_np,
                        global_step=global_step,
                        sample_rate=self.output_sample_rate,
                    )
                    logger.experiment.add_audio(
                        f'{dataset_prefix}/Example_{idx}/target',
                        target_audio_np,
                        global_step=global_step,
                        sample_rate=self.output_sample_rate,
                    )

            # Log attention images
            for attn_key, images in attention_data.items():
                # Determine log prefix: 'overall' uses dataset_prefix directly, others are nested
                if attn_key == 'overall':
                    prefix = dataset_prefix
                else:
                    prefix = f"{dataset_prefix}/{attn_key}"

                if is_wandb:
                    wandb_media[f"Image:{prefix}/attention_matrix"] = [
                        wandb.Image(img_np, caption=f"Example_{idx:02d}") for idx, img_np in enumerate(images)
                    ]

                if is_tb:
                    for idx, img_np in enumerate(images):
                        logger.experiment.add_image(
                            f'{prefix}/attention_matrix/Example_{idx:02d}',
                            img_np,
                            global_step=global_step,
                            dataformats="HWC",
                        )

        return wandb_media

    def scale_prior(self, prior, global_step):
        if prior is None:
            return None
        if global_step < self.prior_scaledown_start_step:
            return prior
        elif global_step >= self.prior_end_step:
            if random.random() < self.indefinite_prior_prob:
                print("Using Prior")
                return prior
            else:
                print("Not using Prior")
                return None
        else:
            with torch.no_grad():
                # Interpolate between all ones and the prior
                residual = 1.0 - prior
                new_prior = prior + (
                    residual
                    * (global_step - self.prior_scaledown_start_step)
                    / (self.prior_end_step - self.prior_scaledown_start_step)
                )
                return new_prior

    def embed_text(self, text, text_mask):
        if self.use_bpe_char_tokenizer:
            text_embedded = self.cas_encoder(text, subword_mask=text_mask)
        else:
            text_embedded = self.text_embedding(text)

        return text_embedded

    def compute_alignment_loss(self, attention_scores, text_lens, audio_lens, dec_context_size=0):
        # attention scores: List of (B, C, audio_timesteps, text_timesteps)
        attention_scores_combined = torch.cat(attention_scores, dim=1)  # (B, C, audio_timesteps, text_timesteps)
        attention_scores_mean = attention_scores_combined.mean(
            dim=1, keepdim=True
        )  # (B, 1, audio_timesteps, text_timesteps)
        attention_scores_mean = attention_scores_mean[
            :, :, dec_context_size:, :
        ]  # Remove the context audio embeddings from the attention scores
        alignment_loss = self.alignment_loss(
            attn_logprob=attention_scores_mean, in_lens=text_lens, out_lens=audio_lens
        )
        return alignment_loss

    def embed_context_text(self, context_text_tokens):
        if self.legacy_text_conditioning:
            context_text_tokens = (
                context_text_tokens - self.tokenizer.tokenizer_offsets[self.text_conditioning_tokenizer_name]
            )
            context_text_embedded = self.context_text_embedding(context_text_tokens)  # (B, L, E)
        else:
            context_text_embedded = self.text_embedding(context_text_tokens)  # (B, L, E)

        return context_text_embedded

    def _encode_text_input(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text from batch.

        Args:
            batch: Dictionary containing 'text' and 'text_lens'.

        Returns:
            Tuple of (text, text_lens, text_mask, text_embedded, text_encoder_out).
        """
        text = batch['text']
        text_lens = batch['text_lens']
        text_mask = get_mask_from_lengths(text_lens)  # (B, T)
        text_embedded = self.embed_text(text, text_mask)  # (B, T, E)
        text_encoder_out = self.encoder(text_embedded, text_mask, cond=None, cond_mask=None)['output']  # (B, T, E)
        return text, text_lens, text_mask, text_embedded, text_encoder_out

    def _get_context_audio_codes(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract or compute context audio codes from batch.

        Args:
            batch: Dictionary containing either 'context_audio_codes' or 'context_audio'.

        Returns:
            Tuple of (context_audio_codes, context_audio_codes_lens) where codes are
            padded according to frame_stacking_factor.
        """
        if 'context_audio_codes' in batch:
            codes = batch['context_audio_codes']
            lens = batch['context_audio_codes_lens']
        else:
            codes, lens = self._codec_helper.audio_to_codes(
                batch['context_audio'],
                batch['context_audio_lens'],
                sample_rate=batch.get('context_sample_rate'),
            )

        if self._codec_converter is not None:
            codes = self._codec_converter.convert_original_to_new(audio_tokens=codes, audio_lens=lens)

        codes, lens = add_special_tokens(
            codes=codes,
            codes_len=lens,
            bos_id=self.context_audio_bos_id,
            eos_id=self.context_audio_eos_id,
        )

        return codes, lens

    def _pad_tensors_to_match(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad two 3D tensors along dim=1 so they have the same sequence length.

        Args:
            tensor_a: First tensor of shape (B, T_a, E).
            tensor_b: Second tensor of shape (B, T_b, E).

        Returns:
            Tuple of (tensor_a, tensor_b) both with shape (B, max(T_a, T_b), E).
        """
        len_a, len_b = tensor_a.size(1), tensor_b.size(1)
        if len_a < len_b:
            padding = torch.zeros(
                tensor_a.size(0), len_b - len_a, tensor_a.size(2), device=tensor_a.device, dtype=tensor_a.dtype
            )
            tensor_a = torch.cat([tensor_a, padding], dim=1)
        elif len_a > len_b:
            padding = torch.zeros(
                tensor_b.size(0), len_a - len_b, tensor_b.size(2), device=tensor_b.device, dtype=tensor_b.dtype
            )
            tensor_b = torch.cat([tensor_b, padding], dim=1)
        return tensor_a, tensor_b

    def _get_context_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
        context_audio_codes: torch.Tensor,
        context_audio_codes_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get context embeddings, handling text conditioning if enabled.

        Args:
            batch: Batch dictionary containing context tokens if text conditioning is used.
            context_audio_codes: Context audio codes. Shape: (B, C, T_ctx).
            context_audio_codes_lens: Length of context audio codes. Shape: (B,).

        Returns:
            Tuple of (context_embedded, context_lens) where:
                context_embedded: Combined context embedding. Shape: (B, T, E).
                context_lens: Length of context sequences. Shape: (B,).
        """
        context_audio_embedded, context_lens = self.embed_audio_tokens(
            audio_tokens=context_audio_codes, audio_tokens_lens=context_audio_codes_lens
        )  # (B, T/frame_stacking, E)

        if not self.use_text_conditioning_encoder:
            return context_audio_embedded, context_lens

        # Text conditioning path
        context_text_tokens = batch['context_text_tokens']
        context_text_lens = batch['context_text_tokens_lens']
        context_text_embedded = self.embed_context_text(context_text_tokens)  # (B, L, E)

        # Pad tensors to match sequence lengths
        context_audio_embedded, context_text_embedded = self._pad_tensors_to_match(
            context_audio_embedded, context_text_embedded
        )

        # For 3D tensor - need to broadcast the boolean mask
        has_text_context = batch['has_text_context'].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1), bool
        context_embedded = torch.where(has_text_context, context_text_embedded, context_audio_embedded)

        # For 1D tensor - direct use
        context_lens = torch.where(batch['has_text_context'], context_text_lens, context_audio_codes_lens)
        context_embedded = context_embedded[:, : context_lens.max(), :]

        return context_embedded, context_lens

    def _prepare_multi_encoder_context(
        self,
        context_input_embedded: torch.Tensor,
        context_mask: torch.Tensor,
        text_encoder_out: torch.Tensor,
        text_mask: torch.Tensor,
        attn_prior: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[Dict], Optional[List], int]:
        """Prepare context tensors for multi_encoder_context_tts model type.

        Args:
            context_input_embedded: Context embeddings. Shape: (B, T_ctx, E).
            context_mask: Mask for context. Shape: (B, T_ctx).
            text_encoder_out: Text encoder output. Shape: (B, T_text, E).
            text_mask: Mask for text. Shape: (B, T_text).
            attn_prior: Attention prior matrix.

        Returns:
            Tuple of (cond, cond_mask, multi_encoder_mapping, attn_prior_list, dec_context_size).
        """
        context_embeddings = self.context_encoder(context_input_embedded, context_mask, cond=None, cond_mask=None)[
            'output'
        ]
        cond = [text_encoder_out, context_embeddings]
        cond_mask = [text_mask, context_mask]
        multi_encoder_mapping = self.multi_encoder_mapping
        attn_prior_list = [attn_prior, None]
        return cond, cond_mask, multi_encoder_mapping, attn_prior_list, 0

    def _prepare_decoder_context(
        self,
        context_input_embedded: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        text_encoder_out: torch.Tensor,
        text_mask: torch.Tensor,
        attn_prior: Optional[torch.Tensor],
        speaker_indices: Optional[torch.Tensor],
        text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, Optional[torch.Tensor], int, torch.Tensor, torch.Tensor]:
        """Prepare context tensors for decoder_context_tts and decoder_ce model types.

        Args:
            batch: Full batch dictionary.
            context_input_embedded: Context embeddings. Shape: (B, T_ctx, E).
                Can be None if model_type is 'decoder_ce' with baked context embedding.
            context_mask: Mask for context. Shape: (B, T_ctx).
                Can be None if model_type is 'decoder_ce' with baked context embedding.
            text_encoder_out: Text encoder output. Shape: (B, T_text, E).
            text_mask: Mask for text. Shape: (B, T_text).
            attn_prior: Attention prior matrix.
            speaker_indices: Speaker indices for multi-speaker baked embeddings.
            text: Text tensor (used for batch_size).

        Returns:
            Tuple of (cond, cond_mask, multi_encoder_mapping, attn_prior,
                     dec_context_size, additional_decoder_input, additional_decoder_mask).
        """
        if self.model_type == 'decoder_context_tts':
            context_embeddings = context_input_embedded
        elif self.model_type == 'decoder_ce':
            if self.has_baked_context_embedding:
                # Baked context embedding replaces the context encoder
                batch_size = text.size(0)
                context_embeddings, context_input_lens = self.get_baked_context_embeddings_batch(
                    batch_size=batch_size, speaker_indices=speaker_indices
                )
                context_input_lens = context_input_lens.to(text.device)
                context_mask = get_mask_from_lengths(context_input_lens)
            else:
                context_embeddings = self.context_encoder(
                    context_input_embedded, context_mask, cond=None, cond_mask=None
                )['output']
        else:
            raise ValueError(f"Unsupported model type for decoder context: {self.model_type}")

        dec_context_size = context_mask.size(1)

        # Pad attention prior if present
        if attn_prior is not None:
            padding_zeros = torch.zeros(
                attn_prior.size(0), dec_context_size, attn_prior.size(2), device=attn_prior.device
            )
            attn_prior = torch.cat([padding_zeros, attn_prior], dim=1)

        return (
            text_encoder_out,  # cond
            text_mask,  # cond_mask
            None,  # multi_encoder_mapping
            attn_prior,
            dec_context_size,
            context_embeddings,  # additional_decoder_input
            context_mask,  # additional_decoder_mask
        )

    def _apply_ctc_prior_layers(
        self, attn_prior: Optional[Union[torch.Tensor, List]]
    ) -> Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]]:
        """Apply CTC prior layer filtering to attention prior.

        Args:
            attn_prior: Attention prior tensor or list of tensors.

        Returns:
            Filtered attention prior with None for layers not in ctc_prior_layer_ids.
        """
        if attn_prior is None or self.ctc_prior_layer_ids is None:
            return attn_prior

        if self.model_type == 'multi_encoder_context_tts':
            text_attn_prior = [
                attn_prior[0] if layer_idx in self.ctc_prior_layer_ids else None
                for layer_idx in range(self.decoder.n_layers)
            ]
            return [text_attn_prior, attn_prior[1]]
        else:
            return [
                attn_prior if layer_idx in self.ctc_prior_layer_ids else None
                for layer_idx in range(self.decoder.n_layers)
            ]

    def prepare_context_tensors(self, batch: Dict[str, torch.Tensor]) -> ContextTensorsOutput:
        """Prepare all context tensors for the decoder.

        This method orchestrates text encoding, context extraction, and model-type-specific
        processing to prepare tensors for decoder inference or training.

        Args:
            batch: Dictionary containing:
                - 'text': Text token IDs. Shape: (B, T_text).
                - 'text_lens': Text lengths. Shape: (B,).
                - 'context_audio_codes' or 'context_audio': Context audio.
                - 'align_prior_matrix' (optional): Beta-binomial attention prior.
                - 'speaker_indices' (optional): Speaker IDs for multi-speaker models.
                - Text conditioning fields if use_text_conditioning_encoder is True.

        Returns:
            ContextTensorsOutput dataclass containing all prepared tensors.

        Raises:
            ValueError: If model_type is not supported.
        """
        # Step 1: Encode text input (always needed)
        text, text_lens, text_mask, text_embedded, text_encoder_out = self._encode_text_input(batch)

        # Step 2: Get and scale attention prior
        _attn_prior = batch.get('align_prior_matrix', None)
        _attn_prior = self.scale_prior(_attn_prior, self.global_step)
        speaker_indices = batch.get('speaker_indices', None)

        # Step 3: Process context based on model type
        if self.model_type not in ['multi_encoder_context_tts', 'decoder_context_tts', 'decoder_ce']:
            raise ValueError(f"Unsupported model type {self.model_type}")

        # For decoder_ce with baked context embedding, skip context audio/text processing entirely
        # The baked embedding replaces the context encoder, so we don't need context inputs
        skip_context_processing = self.model_type == 'decoder_ce' and self.has_baked_context_embedding

        if skip_context_processing:
            # Use baked context embedding directly - no need for context audio/text
            context_audio_codes = None
            context_audio_codes_lens = None
            context_input_embedded = None
            context_mask = None
        else:
            # Extract context audio codes and compute embeddings
            context_audio_codes, context_audio_codes_lens = self._get_context_audio_codes(batch)
            context_input_embedded, context_input_lens = self._get_context_embeddings(
                batch, context_audio_codes, context_audio_codes_lens
            )
            context_mask = get_mask_from_lengths(context_input_lens)

        # Step 4: Dispatch to model-type-specific handler
        if self.model_type == 'multi_encoder_context_tts':
            cond, cond_mask, multi_encoder_mapping, attn_prior, dec_context_size = self._prepare_multi_encoder_context(
                context_input_embedded, context_mask, text_encoder_out, text_mask, _attn_prior
            )
            additional_decoder_input = None
            additional_decoder_mask = None
        else:  # decoder_context_tts or decoder_ce
            (
                cond,
                cond_mask,
                multi_encoder_mapping,
                attn_prior,
                dec_context_size,
                additional_decoder_input,
                additional_decoder_mask,
            ) = self._prepare_decoder_context(
                context_input_embedded,
                context_mask,
                text_encoder_out,
                text_mask,
                _attn_prior,
                speaker_indices,
                text,
            )

        # Step 5: Apply CTC prior layer filtering
        attn_prior = self._apply_ctc_prior_layers(attn_prior)

        # Step 6: Return typed output
        return ContextTensorsOutput(
            text_encoder_out=text_encoder_out,
            text_embedded=text_embedded,
            text_mask=text_mask,
            text_lens=text_lens,
            text=text,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=attn_prior,
            prior_used=_attn_prior is not None,
            multi_encoder_mapping=multi_encoder_mapping,
            additional_decoder_input=additional_decoder_input,
            additional_decoder_mask=additional_decoder_mask,
            dec_context_size=dec_context_size,
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            beta_binomial_attn_prior=batch.get('align_prior_matrix', None),
        )

    def replace_beta_binomial_prior_with_binarized(self, attn_prior, aligner_attn_hard):
        # aligner_attn_hard B, audio_timesteps, text_timesteps
        if self.model_type == 'multi_encoder_context_tts':
            text_attn_prior = attn_prior[0]
        else:
            text_attn_prior = attn_prior

        assert text_attn_prior is not None, "Prior is None"

        if isinstance(text_attn_prior, list):
            # Layer wise prior
            prior_updated = False
            for idx, prior in enumerate(text_attn_prior):
                if prior is not None:
                    text_attn_prior[idx][:, -aligner_attn_hard.size(1) :, :] = aligner_attn_hard
                    prior_updated = True
            assert prior_updated, "Did not find any prior to update"
        else:
            # Same prior for all layers
            text_attn_prior[:, -aligner_attn_hard.size(1) :, :] = aligner_attn_hard

        if self.model_type == 'multi_encoder_context_tts':
            attn_prior[0] = text_attn_prior
        else:
            attn_prior = text_attn_prior

        return attn_prior

    def get_binarized_prior_matrix(self, aligner_attn_soft, audio_lens, text_lens):
        # aligner_attn_soft B, 1, audio_timesteps, text_timesteps
        if self.binarize_attn_method == 'nemo_binarize':
            logging.debug("Binarizing attention using nemo_binarize")
            binarize_repeat_audio_factor = self.binarize_repeat_audio_factor
            aligner_attn_soft_repeated = aligner_attn_soft.repeat_interleave(
                binarize_repeat_audio_factor, dim=2
            )  # B, 1, 2*audio_timesteps, text_timesteps
            aligner_attn_hard = binarize_attention_parallel(
                aligner_attn_soft_repeated, text_lens, audio_lens * binarize_repeat_audio_factor
            ).squeeze(
                1
            )  # B, 2*audio_timesteps, text_timesteps
            aligner_attn_hard = aligner_attn_hard[:, ::2, :]  # B, audio_timesteps, text_timesteps
        elif self.binarize_attn_method == 'argmax':
            logging.debug("Binarizing attention using argmax")
            aligner_attn_hard = torch.argmax(aligner_attn_soft.squeeze(1), dim=-1)
            aligner_attn_hard = torch.nn.functional.one_hot(
                aligner_attn_hard, num_classes=aligner_attn_soft.size(-1)
            ).float()
        else:
            raise ValueError(
                f"self.binarize_attn_method '{self.binarize_attn_method}' must be one of 'nemo_binarize' or 'argmax'."
            )

        aligner_attn_hard_wider = aligner_attn_hard + self.binarized_prior_epsilon

        for future_timestep in range(self.prior_future_context):
            decay_factor = self.prior_future_decay ** (future_timestep + 1)
            aligner_attn_hard_wider[:, :, future_timestep + 1 :] += (
                decay_factor * aligner_attn_hard[:, :, : -(future_timestep + 1)]
            )

        for past_timestep in range(self.prior_past_context):
            decay_factor = self.prior_past_decay ** (past_timestep + 1)
            aligner_attn_hard_wider[:, :, : -past_timestep - 1] += (
                decay_factor * aligner_attn_hard[:, :, past_timestep + 1 :]
            )

        aligner_attn_hard_wider = torch.clamp(aligner_attn_hard_wider, 0.0, 1.0)
        return aligner_attn_hard_wider

    def prepare_dummy_cond_for_cfg(self, cond, cond_mask, additional_decoder_input, additional_dec_mask):
        dummy_additional_decoder_input = None
        dummy_additional_dec_mask = None
        if additional_decoder_input is not None:
            dummy_additional_decoder_input = torch.zeros_like(additional_decoder_input)
            # all ones mask means dont ignore any timesteps (so that it is consistent with usual decoder mask)
            dummy_additional_dec_mask = torch.ones_like(additional_dec_mask)

        if isinstance(cond, list):
            # multi encoder conditioning
            dummy_cond = [torch.zeros_like(cond_item) for cond_item in cond]
            attn_prior = [None for _ in cond]
            dummy_mask = []
            for mask_item in cond_mask:
                # ignore all timesteps except the first one
                mask = torch.zeros_like(mask_item)
                mask[:, 0] = 1  # Make first timestep all zeros
                dummy_mask.append(mask)

        elif isinstance(cond, torch.Tensor):
            # single encoder conditioning
            dummy_cond = torch.zeros_like(cond)
            dummy_mask = torch.zeros_like(cond_mask)
            dummy_mask[:, 0] = 1  # ignore all timesteps except the first one
            attn_prior = None
        else:
            raise ValueError(f"Unsupported type for cond {type(cond)}")

        return dummy_cond, dummy_mask, dummy_additional_decoder_input, dummy_additional_dec_mask, attn_prior

    def process_batch(self, batch):
        context_tensors = self.prepare_context_tensors(batch)
        disable_alignment_loss = False

        if 'audio_codes' not in batch:
            audio_codes, audio_codes_lens = self._codec_helper.audio_to_codes(
                batch['audio'],
                batch['audio_lens'],
                sample_rate=batch.get('sample_rate'),
            )
        else:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']

        if self._codec_converter:
            audio_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=audio_codes, audio_lens=audio_codes_lens
            )

        audio_codes, audio_codes_lens = add_special_tokens(
            codes=audio_codes,
            codes_len=audio_codes_lens,
            bos_id=self.audio_bos_id,
            eos_id=self.audio_eos_id,
            num_bos_tokens=self.frame_stacking_factor,
        )  # (B, C, T)

        audio_codes_embedded_all, audio_codes_lens_all = self.embed_audio_tokens(
            audio_tokens=audio_codes, audio_tokens_lens=audio_codes_lens
        )  # (B, T/frame_stacking_factor, E)
        # Note: if a tensor lacks the `_unstacked` suffix, it can be assumed to be in the frame-stacked domain

        # Remove EOS token for decoder inputs
        audio_codes_embedded_input, audio_codes_lens_input = remove_embedded_eos_token(
            embedded=audio_codes_embedded_all, embedded_len=audio_codes_lens_all
        )
        use_cfg = self.training and (self.cfg_unconditional_prob > 0.0) and (context_tensors.cond is not None)
        if use_cfg and torch.rand(1).item() < self.cfg_unconditional_prob:
            cond, cond_mask, additional_decoder_input, additional_decoder_mask, attn_prior = (
                self.prepare_dummy_cond_for_cfg(
                    context_tensors.cond,
                    context_tensors.cond_mask,
                    context_tensors.additional_decoder_input,
                    context_tensors.additional_decoder_mask,
                )
            )
            disable_alignment_loss = True
        else:
            cond = context_tensors.cond
            cond_mask = context_tensors.cond_mask
            additional_decoder_input = context_tensors.additional_decoder_input
            additional_decoder_mask = context_tensors.additional_decoder_mask
            attn_prior = context_tensors.attn_prior

            if self.training and self.decoder_input_dropout_prob > 0.0 and torch.rand(1).item() < 0.5:
                # For some batches (half of them), replace decoder_input_dropout_prob of the timesteps with random tokens
                max_codebook_val = self.dec_random_input_max
                # @pneekhara: Keeping dec_random_input_max configurable since num_all_tokens_per_codebook usually has padding tokens
                # which can cause errors when doing codes_to_audio for audio_codes_input. We are not currently calling codes_to_audio on
                # audio_codes_input so should not matter if we don't supply dec_random_input_max.
                random_audio_tokens = torch.randint(
                    low=0, high=max_codebook_val, size=audio_codes.size(), device=audio_codes_embedded_input.device
                )  # (B, C, T)
                random_embedded, random_embedded_lens = self.embed_audio_tokens(
                    audio_tokens=random_audio_tokens, audio_tokens_lens=audio_codes_lens
                )  # (B T E)
                random_embedded, random_embedded_lens = remove_embedded_eos_token(
                    embedded=random_embedded, embedded_len=random_embedded_lens
                )
                dec_dropout_mask = (
                    torch.rand((1, 1, audio_codes_embedded_input.size(2)), device=audio_codes_embedded_input.device)
                    > self.decoder_input_dropout_prob
                )  # (1, 1, T)
                audio_codes_embedded_input = torch.where(
                    dec_dropout_mask,
                    audio_codes_embedded_input,
                    random_embedded,
                )

        audio_codes_mask = get_mask_from_lengths(audio_codes_lens_input)
        if additional_decoder_input is not None:
            audio_codes_embedded_input = torch.cat([additional_decoder_input, audio_codes_embedded_input], dim=1)
            audio_codes_mask = torch.cat([additional_decoder_mask, audio_codes_mask], dim=1)

        # Remove BOS token for aligner targets
        audio_codes_embedded_target, audio_codes_lens_target = remove_embedded_bos_token(
            embedded=audio_codes_embedded_all, embedded_len=audio_codes_lens_all
        )
        aligner_encoder_loss = None
        aligner_attn_soft = None
        aligner_attn_hard = None
        if self.use_alignment_encoder and not disable_alignment_loss:
            aligner_prior = None
            if self.use_prior_for_aligner:
                aligner_prior = context_tensors.beta_binomial_attn_prior

            train_aligner = self.global_step < self.aligner_encoder_train_steps

            with torch.set_grad_enabled(train_aligner):
                # Passing target audio embeddings to the alignment encoder
                aligner_queries = audio_codes_embedded_target.permute(0, 2, 1)  # (B, E, T')
                aligner_keys = context_tensors.text_encoder_out.permute(0, 2, 1)  # (B, E, T)
                # Aligner uses inverted mask
                aligner_mask = ~context_tensors.text_mask.unsqueeze(-1)  # (B, T, 1)
                aligner_attn_soft, aligner_attn_logprobs = self.alignment_encoder(
                    queries=aligner_queries,
                    keys=aligner_keys,
                    mask=aligner_mask,
                    attn_prior=aligner_prior,
                )

            if train_aligner:
                aligner_encoder_loss = self.alignment_encoder_loss(
                    attn_logprob=aligner_attn_logprobs,
                    in_lens=context_tensors.text_lens,
                    out_lens=audio_codes_lens_target,
                )

            with torch.no_grad():
                aligner_attn_hard = self.get_binarized_prior_matrix(
                    aligner_attn_soft, audio_codes_lens_input, context_tensors.text_lens
                )
                if (self.global_step > self.binarize_prior_after_step) and context_tensors.prior_used:
                    attn_prior = self.replace_beta_binomial_prior_with_binarized(attn_prior, aligner_attn_hard)

        logits, attn_info, dec_out, moe_routing_info = self.forward(
            dec_input_embedded=audio_codes_embedded_input,
            dec_input_mask=audio_codes_mask,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=attn_prior,
            multi_encoder_mapping=context_tensors.multi_encoder_mapping,
        )
        # logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # dec_out: (B, T', E)
        # moe_routing_info: List of routing info dicts from each layer (if MoE enabled)
        dec_context_size = context_tensors.dec_context_size
        logits = logits[:, dec_context_size:, :]  # Remove the context audio embeddings from the logits

        # Remove BOS tokens from decoder targets
        audio_codes_target_unstacked, audio_codes_lens_target_unstacked = remove_bos_token(
            codes=audio_codes, codes_len=audio_codes_lens, num_tokens=self.frame_stacking_factor
        )
        # Codebook loss (parallel)
        codebook_loss, loss_mask = self.compute_loss(
            logits,
            audio_codes_target_unstacked,
            audio_codes_lens_target_unstacked,
            frame_stacking_factor=self.frame_stacking_factor,
        )
        # Alignment loss
        alignment_loss = None
        if self.alignment_loss_scale > 0.0 and not disable_alignment_loss:
            text_lens = context_tensors.text_lens
            cross_attention_scores = [
                attn['cross_attn_probabilities'][1]
                for layer_idx, attn in enumerate(attn_info)
                if layer_idx in self.ctc_prior_layer_ids
            ]
            alignment_loss = self.compute_alignment_loss(
                cross_attention_scores, text_lens, audio_codes_lens_target, dec_context_size
            )
            loss = self.codebook_loss_scale * codebook_loss + alignment_loss
        else:
            loss = self.codebook_loss_scale * codebook_loss

        # Local Transformer loss
        local_transformer_loss = None
        local_transformer_logits = None
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            if self.local_transformer_type == LocalTransformerType.MASKGIT:
                # Maskgit
                # randomly replace some positions with MASK_TOKEN
                audio_codes_masked, mask_tokens_mask = self._lt_helper.apply_random_mask(audio_codes_target_unstacked)
                # TODO @rfejgin: the very last position might be padding but the local transformer might look at it as part of
                #                of a pair where the first position is valid. Is this an issue?
                local_transformer_logits = self._lt_helper.compute_logits(
                    dec_out[:, dec_context_size:, :], audio_codes_masked, targets_offset_by_one=True
                )
                local_transformer_loss, _ = self.compute_loss(
                    local_transformer_logits,
                    audio_codes_target_unstacked,
                    audio_codes_lens_target_unstacked,
                    mask_tokens_mask,
                    frame_stacking_factor=self.frame_stacking_factor,
                )
            else:
                # Autoregressive
                assert self.local_transformer_type == LocalTransformerType.AR, "Unexpected local transformer type"
                local_transformer_logits = self._lt_helper.compute_logits(
                    dec_out[:, dec_context_size:, :], audio_codes_target_unstacked, targets_offset_by_one=False
                )
                local_transformer_loss, _ = self.compute_loss(
                    local_transformer_logits,
                    audio_codes_target_unstacked,
                    audio_codes_lens_target_unstacked,
                    None,
                    frame_stacking_factor=self.frame_stacking_factor,
                )
            loss = loss + self.local_transformer_loss_scale * local_transformer_loss

        if aligner_encoder_loss is not None:
            loss = loss + aligner_encoder_loss

        # Compute MoE auxiliary losses and expert usage statistics if MoE is enabled
        moe_load_balancing_loss = None
        moe_router_z_loss = None
        moe_expert_usage_stats = None

        if self.use_moe and moe_routing_info is not None:
            # The decoder input is: [context_audio | target_audio | padding]. MoE routing runs on this full concatenated
            # sequence, so router_logits, router_probs, and expert_indices contain context audio dimensions. We include
            # context audio in the MoE loss computation (not stripped like the main CE loss) because:
            #   1. Load balancing loss needs to see all tokens the router dispatches, including context. Excluding
            #      context would make experts that specialize in processing context audio look underused, producing
            #      misleading gradients.
            #   2. At inference, context audio is always present and routed through experts. Training the router to
            #      balance load only on target tokens would create a train/inference mismatch in routing behavior.
            # Padding is excluded via x_mask. Router already masks padded positions, router_logits/router_probs=0,
            # expert_indices=-1, and we pass x_mask to loss functions to ensure averages are computed only over valid (non-padding) tokens.
            all_router_logits = []
            all_router_probs = []
            all_expert_indices = []
            for layer_routing_info in moe_routing_info:
                all_router_logits.append(layer_routing_info['router_logits'])
                all_router_probs.append(layer_routing_info['router_probs'])
                all_expert_indices.append(layer_routing_info['expert_indices'])

            # Concatenate across layers (batch dimension)
            stacked_logits = torch.stack(all_router_logits, dim=0)  # (n_layers, B, T, num_experts)
            stacked_probs = torch.stack(all_router_probs, dim=0)  # (n_layers, B, T, num_experts)
            stacked_indices = torch.stack(all_expert_indices, dim=0)  # (n_layers, B, T, top_k)

            # Reshape for loss computation
            # merged_logits and merged_probs are (n_layers*B, T, num_experts)
            merged_logits = stacked_logits.view(-1, stacked_logits.size(2), stacked_logits.size(3))
            merged_probs = stacked_probs.view(-1, stacked_probs.size(2), stacked_probs.size(3))
            # merged_indices is (n_layers*B, T, top_k)
            merged_indices = stacked_indices.view(-1, stacked_indices.size(2), stacked_indices.size(3))

            # Repeat mask for each layer: (B, T) -> (n_layers*B, T)
            # Include ALL decoder input positions (context audio + target audio) in loss computation
            # Context audio routing is important for inference quality. We want Expert specialization where some experts
            # may specialize in processing context, or some may specialize in generating target, or both.
            merged_mask = (
                audio_codes_mask.unsqueeze(0).repeat(len(moe_routing_info), 1, 1).view(-1, audio_codes_mask.size(1))
            )

            # Compute MoE losses using the loss module (both train and val)
            # Pass mask to ensure losses are computed only over valid tokens (excluding padding)
            moe_load_balancing_loss, moe_router_z_loss, moe_total_loss = self.moe_auxiliary_loss(
                router_logits=merged_logits,
                router_probs=merged_probs,
                x_mask=merged_mask,
            )

            # Compute expert usage statistics
            with torch.no_grad():
                num_experts = stacked_probs.size(-1)
                n_moe_layers = stacked_probs.size(0)

                # Per-layer expert usage: (n_layers, num_experts)
                layer_expert_usage = torch.stack(
                    [compute_expert_usage(stacked_probs[i], audio_codes_mask) for i in range(n_moe_layers)]
                )

                # Global expert usage: mean across layers (for scalar logging)
                expert_usage = layer_expert_usage.mean(dim=0)  # (num_experts,)

                # Compute how often each expert is selected in top-k
                # For padded positions, expert_indices=-1, so they don't match any valid expert (0 to num_experts-1)
                expert_selection_counts = torch.zeros(num_experts, device=merged_probs.device)
                for expert_idx in range(num_experts):
                    expert_selection_counts[expert_idx] = (merged_indices == expert_idx).float().sum()

                # Normalize to get selection frequency over valid (non-padded) selections only
                # Padded positions have expert_indices=-1, which don't match any valid expert
                valid_selections = (merged_indices != -1).sum().float().clamp_min(1.0)
                expert_selection_freq = expert_selection_counts / valid_selections

                moe_expert_usage_stats = {
                    'expert_usage': expert_usage.detach(),  # (num_experts,)
                    'layer_expert_usage': layer_expert_usage.detach(),  # (n_layers, num_experts)
                    'expert_selection_freq': expert_selection_freq.detach(),  # (num_experts,)
                    'batch_expert_usage_variance': expert_usage.var().detach(),
                    'ideal_usage': 1.0 / num_experts,
                }

            # Add MoE loss to total loss (only in training mode)
            if self.training:
                loss = loss + moe_total_loss

        return {
            'logits': logits,
            'attn_info': attn_info,
            'loss': loss,
            'codebook_loss': codebook_loss,
            'local_transformer_loss': local_transformer_loss,
            'local_transformer_logits': local_transformer_logits,
            'loss_mask': loss_mask,
            'alignment_loss': alignment_loss,
            'aligner_encoder_loss': aligner_encoder_loss,
            'moe_load_balancing_loss': moe_load_balancing_loss,
            'moe_router_z_loss': moe_router_z_loss,
            'moe_expert_usage_stats': moe_expert_usage_stats,
            'audio_codes_target': audio_codes_target_unstacked,
            'audio_codes_lens_target': audio_codes_lens_target_unstacked,
            'text': context_tensors.text,
            'text_lens': context_tensors.text_lens,
            'context_audio_codes': context_tensors.context_audio_codes,
            'context_audio_codes_lens': context_tensors.context_audio_codes_lens,
            'dec_context_size': dec_context_size,
            'aligner_attn_soft': aligner_attn_soft,
            'aligner_attn_hard': aligner_attn_hard,
        }

    def training_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch)
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        self.log('Loss:train/codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
        if self.cfg_unconditional_prob == 0.0:
            # Only log alignment loss when not using cfg to avoid sync issues when
            # alignment loss is None on some ranks
            alignment_loss = batch_output['alignment_loss']
            if alignment_loss is not None:
                self.log('Loss:train/alignment_loss', alignment_loss, prog_bar=True, sync_dist=True)
        self.log('Loss:train/loss', loss, prog_bar=True, sync_dist=True)
        local_transformer_loss = batch_output['local_transformer_loss']
        if local_transformer_loss is not None:
            self.log('Loss:train/local_transformer_loss', local_transformer_loss, prog_bar=True, sync_dist=True)

        # Log MoE losses and expert usage if MoE is enabled
        moe_load_balancing_loss = batch_output.get('moe_load_balancing_loss', None)
        moe_router_z_loss = batch_output.get('moe_router_z_loss', None)
        moe_expert_usage_stats = batch_output.get('moe_expert_usage_stats', None)
        if moe_load_balancing_loss is not None and self.moe_auxiliary_loss.load_balancing_loss.loss_scale > 0:
            self.log('Loss:train/moe_load_balancing_loss', moe_load_balancing_loss, prog_bar=True, sync_dist=True)
        if moe_router_z_loss is not None and self.moe_auxiliary_loss.router_z_loss.loss_scale > 0:
            self.log('Loss:train/moe_router_z_loss', moe_router_z_loss, prog_bar=True, sync_dist=True)
        if moe_expert_usage_stats is not None:
            expert_usage = moe_expert_usage_stats['expert_usage']
            layer_expert_usage = moe_expert_usage_stats['layer_expert_usage']

            self.log(
                'Loss:train/moe_expert_usage_variance',
                moe_expert_usage_stats['batch_expert_usage_variance'],
                sync_dist=True,
            )

            # Per-expert usage scalars
            for eidx in range(len(expert_usage)):
                self.log(f'MoE:train/Expert_{eidx:02d}_usage', expert_usage[eidx], sync_dist=True)

            # Accumulate layer-wise usage for training heatmap
            if self._moe_train_layer_usage_accum is None:
                self._moe_train_layer_usage_accum = torch.zeros_like(layer_expert_usage)
            self._moe_train_layer_usage_accum += layer_expert_usage.detach()
            self._moe_train_accum_steps += 1

        # Log batch info
        batch_size, text_token_max_len = batch["text"].shape
        text_token_total_num = batch["text_lens"].sum()
        batch_info_dict = {
            "BatchInfo:train/batch_size": batch_size,
            "BatchInfo:train/text_token_max_len": text_token_max_len,
            "BatchInfo:train/text_token_total_num_in_batch": text_token_total_num.item(),
            "BatchInfo:train/text_token_pad_ratio_percent_in_batch": 100
            * (1 - text_token_total_num / (batch_size * text_token_max_len)),
        }

        if "audio_codes" in batch:
            audio_codes_max_len = batch["audio_codes"].shape[-1]
            audio_codes_total_num = batch["audio_codes_lens"].sum()
            batch_info_dict.update(
                {
                    "BatchInfo:train/audio_codes_max_len": audio_codes_max_len,
                    "BatchInfo:train/audio_codes_total_num_in_batch": audio_codes_total_num.item(),
                    "BatchInfo:train/audio_codes_pad_ratio_percent_in_batch": 100
                    * (1 - audio_codes_total_num / (batch_size * audio_codes_max_len)),
                }
            )
        else:
            audio_samples_max_len = batch["audio"].shape[-1]
            audio_samples_total_num = batch["audio_lens"].sum()
            batch_info_dict.update(
                {
                    "BatchInfo:train/audio_samples_max_len": audio_samples_max_len,
                    "BatchInfo:train/audio_samples_total_num_in_batch": audio_samples_total_num.item(),
                    "BatchInfo:train/audio_samples_pad_ratio_percent_in_batch": 100
                    * (1 - audio_samples_total_num / (batch_size * audio_samples_max_len)),
                }
            )

        self.log_dict(batch_info_dict, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step with support for multiple dataloaders.

        Args:
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: Index of the dataloader (0 for single dataloader)
        """
        batch_output = self.process_batch(batch)
        # self.process_batch returns a dict. We currently only log "logits" which come from the parallel prediction
        # head. If we use local_transformer, then the local_transformer returns "local_transformer_logits"

        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        alignment_loss = batch_output['alignment_loss']
        aligner_encoder_loss = batch_output['aligner_encoder_loss']
        local_transformer_loss = batch_output['local_transformer_loss']

        # Extract MoE losses and expert usage statistics if MoE is enabled
        moe_load_balancing_loss = batch_output.get('moe_load_balancing_loss', None)
        moe_router_z_loss = batch_output.get('moe_router_z_loss', None)
        moe_expert_usage_stats = batch_output.get('moe_expert_usage_stats', None)

        logits = batch_output['logits']
        audio_codes_target = batch_output['audio_codes_target']
        audio_codes_lens_target = batch_output['audio_codes_lens_target']
        context_audio_codes = batch_output['context_audio_codes']
        context_audio_codes_lens = batch_output['context_audio_codes_lens']
        attn_info = batch_output['attn_info']
        text_lens = batch_output['text_lens']
        dec_context_size = batch_output['dec_context_size']

        val_output = {
            'val_loss': loss,
            'val_codebook_loss': codebook_loss,
        }

        # Only add optional losses if they were computed (not None)
        if alignment_loss is not None:
            val_output['val_alignment_loss'] = alignment_loss
        if local_transformer_loss is not None:
            val_output['val_local_transformer_loss'] = local_transformer_loss
        if aligner_encoder_loss is not None:
            val_output['val_aligner_encoder_loss'] = aligner_encoder_loss
        if moe_load_balancing_loss is not None:
            val_output['val_moe_load_balancing_loss'] = moe_load_balancing_loss
        if moe_router_z_loss is not None:
            val_output['val_moe_router_z_loss'] = moe_router_z_loss
        if moe_expert_usage_stats is not None:
            val_output['val_moe_expert_usage_stats'] = moe_expert_usage_stats

        # Prepare media data for logging (only first batch of each dataloader, rank 0 only).
        if batch_idx == 0 and self.global_rank == 0:
            dataset_prefix = self.get_validation_dataloader_prefix(dataloader_idx)

            # Prepare audio examples (decode via vocoder, convert to numpy)
            audio_data = self._prepare_audio_examples(
                logits=logits,
                target_audio_codes=audio_codes_target,
                audio_codes_lens=audio_codes_lens_target,
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                max_examples=3,
            )

            # Prepare attention images (only when cross-attention is available)
            attention_data = {}
            has_cross_attn = (
                self.model_type != 'decoder_pretrain_synthesizer'
                and len(attn_info[self.transcript_decoder_layers[0]].get('cross_attn_probabilities', [])) > 1
            )

            if has_cross_attn:
                # Overall attention: average across CTC prior layers
                cross_attention_probs = [
                    attn['cross_attn_probabilities'][0]
                    for layer_idx, attn in enumerate(attn_info)
                    if layer_idx in self.ctc_prior_layer_ids
                ]
                attention_data['overall'] = self._prepare_attention_images(
                    cross_attention_probs,
                    audio_codes_lens_target,
                    text_lens,
                    dec_context_size=dec_context_size,
                    max_examples=3,
                )

                # Per-layer attention visualization
                for layer_idx in self.transcript_decoder_layers:
                    layer_cross_attention_probs = [attn_info[layer_idx]['cross_attn_probabilities'][0]]
                    attention_data[f'layer_{layer_idx:02d}'] = self._prepare_attention_images(
                        layer_cross_attention_probs,
                        audio_codes_lens_target,
                        text_lens,
                        dec_context_size=dec_context_size,
                        max_examples=3,
                    )

                # Aligner encoder attention (if available)
                if batch_output['aligner_attn_soft'] is not None:
                    attention_data['aligner_encoder_attn'] = self._prepare_attention_images(
                        [batch_output['aligner_attn_soft']],
                        audio_codes_lens_target,
                        text_lens,
                        dec_context_size=0,
                        max_examples=3,
                    )

                if batch_output['aligner_attn_hard'] is not None:
                    attention_data['aligner_encoder_attn_hard'] = self._prepare_attention_images(
                        [batch_output['aligner_attn_hard'].unsqueeze(1)],
                        audio_codes_lens_target,
                        text_lens,
                        dec_context_size=0,
                        max_examples=3,
                    )

            val_output['media_data'] = {
                'dataset_prefix': dataset_prefix,
                'pred_audios': audio_data['pred_audios'],
                'target_audios': audio_data['target_audios'],
                'context_audios': audio_data['context_audios'],
                'attention_data': attention_data,
            }

        self.validation_step_outputs[dataloader_idx].append(val_output)

        return val_output

    def get_cross_attention_scores(self, attn_probs, filter_layers=None):
        """
        Returns the cross attention probabilities for the last audio timestep
        """
        mean_cross_attn_scores = []
        all_heads_cross_attn_scores = []
        for lidx, layerwise_attn_prob in enumerate(attn_probs):
            if (filter_layers is not None and lidx not in filter_layers) or (
                lidx not in self.transcript_decoder_layers
            ):
                continue
            cross_attn_prob = layerwise_attn_prob['cross_attn_probabilities'][
                0
            ]  # B, H, audio_timesteps, text_timesteps
            mean_cross_attn_scores.append(cross_attn_prob.mean(dim=1))  # B, audio_timesteps, text_timesteps
            for head_idx in range(cross_attn_prob.size(1)):
                all_heads_cross_attn_scores.append(cross_attn_prob[:, head_idx, -1, :])  # B, text_timesteps

        mean_cross_attn_scores = torch.stack(mean_cross_attn_scores, dim=1)  # B, L, audio_timesteps, text_timesteps
        mean_cross_attn_scores = mean_cross_attn_scores.mean(dim=1)  # B, audio_timesteps, text_timesteps
        last_audio_timestep_scores = mean_cross_attn_scores[:, -1, :]  # B, text_timesteps
        return last_audio_timestep_scores, all_heads_cross_attn_scores

    def get_most_attended_text_timestep(
        self,
        alignment_attention_scores,
        last_attended_timesteps,
        text_lens,
        lookahead_window_size,
        attended_timestep_counter,
        batch_size,
        left_offset=[],
    ):
        """
        Returns the most attended timestep for each batch item

        This method identifies which text token is most attended to within a lookahead window, starting from
        the last attended timestep. It includes logic to detect attention sinks (tokens attended to excessively)
        and move past them. The method also tracks how many times each timestep has been attended.

        Args:
            alignment_attention_scores (torch.Tensor): Attention scores between audio and text tokens.
                Shape: (batch_size, text_length).
            last_attended_timesteps (list): List containing the last attended timestep for each batch item.
                The last element [-1] should be a list/tensor of length batch_size.
            text_lens (torch.Tensor): Length of text sequence for each batch item. Shape: (batch_size,).
            lookahead_window_size (int): Size of the forward-looking window to search for the next attended
                timestep. Determines how far ahead from the last attended timestep to look.
            attended_timestep_counter (list): List of dictionaries (one per batch item) tracking how many
                times each timestep has been attended. Used to detect attention sinks.
            batch_size (int): Number of items in the batch.
            left_offset (list, optional): List of offsets to adjust timestep indices for each batch item,
                used in chunked inference when text is provided in chunks. Relevant only in multi-chunk
                generation.

        Returns:
            tuple: A tuple containing:
                - text_time_step_attended (list): List of integers, one per batch item, indicating the most
                  attended text timestep for that item.
                - attended_timestep_counter (list): Updated counter tracking attendance frequency for each
                  timestep across all batch items.
        """
        if len(left_offset) == 0:
            left_offset = [0 for _ in range(batch_size)]
        text_time_step_attended = []
        for bidx in range(batch_size):
            last_attended_timestep = last_attended_timesteps[-1][bidx]
            if attended_timestep_counter[bidx].get(last_attended_timestep, 0) >= 8:
                # This is probably an attention sink! Move to the next timestep
                last_attended_timestep += 1
            last_attended_timestep_in_this_window = last_attended_timestep - left_offset[bidx]
            window_size = lookahead_window_size
            window_end = min(
                last_attended_timestep_in_this_window + window_size, text_lens[bidx] - 3
            )  # Ignore the last 3 timesteps
            item_attention_scores = alignment_attention_scores[bidx, last_attended_timestep_in_this_window:window_end]
            if item_attention_scores.size(0) == 0:
                # This means the sentence has ended
                attended_timestep = text_lens[bidx].item() - 1 + left_offset[bidx]
            else:
                attended_timestep = item_attention_scores.argmax().item() + last_attended_timestep
            text_time_step_attended.append(attended_timestep)
            attended_timestep_counter[bidx][attended_timestep] = (
                attended_timestep_counter[bidx].get(attended_timestep, 0) + 1
            )
        return text_time_step_attended, attended_timestep_counter

    def construct_inference_prior(
        self,
        prior_epsilon,
        cross_attention_scores,
        text_lens,
        text_time_step_attended,
        attended_timestep_counter,
        unfinished_texts,
        finished_texts_counter,
        end_indices,
        lookahead_window_size,
        batch_size,
    ):
        # Attn prior for the next timestep
        _attn_prior = torch.zeros(cross_attention_scores.shape[0], 1, cross_attention_scores.shape[1]) + prior_epsilon
        _attn_prior = _attn_prior.to(cross_attention_scores.device)
        for bidx in range(cross_attention_scores.shape[0]):
            if bidx < batch_size:
                _text_len = text_lens[bidx]
                if text_lens[bidx] <= 5:
                    # Very short sentences, No Prior
                    _attn_prior[bidx, 0, :] = 1.0
                else:
                    _attn_prior[bidx, 0, max(1, text_time_step_attended[bidx] - 1)] = (
                        1.0  # Slight exposure to history for better pronounciation. Not very important.
                    )
                    _attn_prior[bidx, 0, text_time_step_attended[bidx]] = (
                        1.0  # Slightly bias to continue moving forward. Not very important.
                    )
                    for ind in range(1, lookahead_window_size + 1):
                        _attn_prior[bidx, 0, min(text_time_step_attended[bidx] + ind, _text_len - 1)] = 1.0

                # Penalize timesteps that have been attended to more than 10 times
                for _timestep in attended_timestep_counter[bidx]:
                    if attended_timestep_counter[bidx][_timestep] >= 10:
                        # This means the timestep has been attended to more than 10 times (To avoid getting stuck)
                        _attn_prior[bidx, 0, : _timestep + 1] = prior_epsilon

                unfinished_texts[bidx] = False
                if text_time_step_attended[bidx] < text_lens[bidx] - 3:
                    # This means the sentence has not ended
                    if bidx not in end_indices:
                        unfinished_texts[bidx] = True

                if text_time_step_attended[bidx] >= text_lens[bidx] - 2 or bidx in end_indices:
                    if bidx not in finished_texts_counter:
                        finished_texts_counter[bidx] = 0

        for bidx in finished_texts_counter:
            finished_texts_counter[bidx] += 1
            if finished_texts_counter[bidx] > 5:
                # This means we have been within the text EOS window for at least 5 timesteps
                # We should allow EOS to be predicted now.
                unfinished_texts[bidx] = False

        return _attn_prior, unfinished_texts, finished_texts_counter

    def get_inference_attention_plots(
        self,
        cross_attention_scores_all_timesteps,
        all_heads_cross_attn_scores_all_timesteps,
        text_lens,
        predicted_codes_lens,
        batch_size,
        compute_all_heads_attn_maps,
        last_attended_timestep,
    ):
        last_attended_timestep = np.array(last_attended_timestep).T
        cross_attention_scores_all_timesteps = torch.stack(
            cross_attention_scores_all_timesteps, dim=2
        )  # B, text_timesteps, T'
        headwise_cross_attention_scores_all_timesteps = []
        for hidx in range(len(all_heads_cross_attn_scores_all_timesteps[0])):
            head_cross_attention_all_timesteps = torch.stack(
                [x[hidx] for x in all_heads_cross_attn_scores_all_timesteps], dim=2
            )  # B, text_timesteps, T'
            headwise_cross_attention_scores_all_timesteps.append(head_cross_attention_all_timesteps)

        cross_attention_maps = []
        headwise_cross_attention_maps = []
        for bidx in range(batch_size):
            item_cross_attention_scores = cross_attention_scores_all_timesteps[
                bidx, : text_lens[bidx], : predicted_codes_lens[bidx]
            ]
            cross_attn_np = plot_alignment_to_numpy(
                item_cross_attention_scores.cpu().numpy(),
                attended=last_attended_timestep[bidx, : predicted_codes_lens[bidx]],
            )
            cross_attention_maps.append(cross_attn_np)
            item_all_head_cross_attn_maps = []
            if compute_all_heads_attn_maps:
                for hidx in range(len(all_heads_cross_attn_scores_all_timesteps[0])):
                    item_headwise_cross_attention_scores = headwise_cross_attention_scores_all_timesteps[hidx][
                        bidx, : text_lens[bidx], : predicted_codes_lens[bidx]
                    ]
                    headwise_cross_attn_np = plot_alignment_to_numpy(
                        item_headwise_cross_attention_scores.cpu().numpy(),
                        attended=last_attended_timestep[bidx, : predicted_codes_lens[bidx]],
                    )
                    item_all_head_cross_attn_maps.append(headwise_cross_attn_np)
                headwise_cross_attention_maps.append(item_all_head_cross_attn_maps)

        return cross_attention_maps, headwise_cross_attention_maps

    def find_eos_frame_index(self, codes, eos_detection_method) -> Union[int, float]:
        """
        Checks for EOS in the predicted codes. Returns the index of the first frame within the frame stack
        that contains an EOS token across any codebook, or `None` if no EOS is found.
        Args:
            codes: (num_codebooks, frame_stacking_factor)
        Returns:
            index (within the frame stack) of the first frame with EOS, or `float('inf')` if no EOS is found
        """
        eos_mask = codes == self.audio_eos_id  # (codebooks, frame_stacking_factor)
        detection_type = EOSDetectionMethod.detection_type(eos_detection_method)
        if detection_type == "any":
            eos_per_frame = eos_mask.any(
                dim=0
            )  # (frame_stacking_factor,) - True if any codebook has EOS in this frame
        elif detection_type == "all":
            eos_per_frame = eos_mask.all(
                dim=0
            )  # (frame_stacking_factor,) - True if all codebooks have EOS in this frame
        elif detection_type == "zero_cb":
            eos_per_frame = eos_mask[:1, :].any(
                dim=0
            )  # (frame_stacking_factor,) - True if zeroth codebook has EOS in this frame
        else:
            raise ValueError(f"Invalid EOS detection method: {eos_detection_method}")
        # find first frame with EOS
        if eos_per_frame.any():
            # return index of the first frame with EOS
            return eos_per_frame.nonzero()[0].item()
        return float('inf')

    def detect_eos(self, audio_codes_multinomial, audio_codes_argmax, eos_detection_method) -> Union[int, float]:
        """
        Detects EOS in the predicted codes. Returns the index of the first frame within the frame stack
        that triggers EOS detection, or `float('inf')` if no EOS is found.
        Args:
            audio_codes_multinomial: (num_codebooks, frame_stacking_factor) - Multinomial samples
            audio_codes_argmax: (num_codebooks, frame_stacking_factor) - Argmax samples
            eos_detection_method: EOS detection method
        Returns:
            index (within the frame stack) of the first frame with EOS, or `float('inf')` if no EOS is found
        """
        sampling_type = EOSDetectionMethod.sampling_type(eos_detection_method)
        if sampling_type == "argmax":
            return self.find_eos_frame_index(audio_codes_argmax, eos_detection_method)
        elif sampling_type == "argmax_or_multinomial":
            argmax_eos_frame = self.find_eos_frame_index(audio_codes_argmax, eos_detection_method)
            multinomial_eos_frame = self.find_eos_frame_index(audio_codes_multinomial, eos_detection_method)
            return min(argmax_eos_frame, multinomial_eos_frame)
        else:
            raise ValueError(f"Invalid EOS detection method: {eos_detection_method}")

    def infer_batch(
        self,
        batch,
        use_cfg=False,
        return_cross_attn_probs=False,
        compute_all_heads_attn_maps=False,
        use_local_transformer_for_inference=False,
        maskgit_n_steps=3,
        maskgit_noise_scale=0.0,
        maskgit_fixed_schedule=None,
        maskgit_dynamic_cfg_scale=False,
        maskgit_sampling_type=None,
    ):
        """
        The behaviour of this function is strongly dependent on self.inference_parameters
        """
        cfg_scale = self.inference_parameters.cfg_scale

        eos_detection_method = EOSDetectionMethod(self.inference_parameters.eos_detection_method)
        with torch.no_grad():
            start_time = time.time()
            self.decoder.reset_cache(use_cache=self.use_kv_cache_for_inference)

            context_tensors = self.prepare_context_tensors(batch)
            text = context_tensors.text
            audio_codes_input = torch.full(
                size=(text.size(0), self.num_audio_codebooks, self.frame_stacking_factor),
                fill_value=self.audio_bos_id,
                device=text.device,
            )
            audio_codes_lens = torch.full(
                size=[text.size(0)], fill_value=self.frame_stacking_factor, device=text.device, dtype=torch.long
            )

            all_predictions = []
            end_indices = {}

            if use_cfg:
                dummy_cond, dummy_cond_mask, dummy_additional_decoder_input, dummy_addition_dec_mask, _ = (
                    self.prepare_dummy_cond_for_cfg(
                        context_tensors.cond,
                        context_tensors.cond_mask,
                        context_tensors.additional_decoder_input,
                        context_tensors.additional_decoder_mask,
                    )
                )

            cross_attention_scores_all_timesteps = []
            all_heads_cross_attn_scores_all_timesteps = []
            _attn_prior = None
            unfinished_texts = {}
            finished_texts_counter = {}
            attended_timestep_counter = [{} for _ in range(text.size(0))]
            last_attended_timesteps = [
                [1 for _ in range(text.size(0))]
            ]  # Maintain a list of attended timesteps as we predict audio for each batch item
            time_to_first_prediction = 0.0
            for idx in range(self.inference_parameters.max_decoder_steps // self.frame_stacking_factor):
                if idx == 1:
                    time_to_first_prediction = time.time() - start_time
                if idx % 20 == 0:
                    print(f"Decoding timestep {idx}")
                audio_codes_embedded, audio_codes_embedded_lens = self.embed_audio_tokens(
                    audio_tokens=audio_codes_input, audio_tokens_lens=audio_codes_lens
                )
                audio_codes_mask = get_mask_from_lengths(audio_codes_embedded_lens)

                if context_tensors.additional_decoder_input is not None:
                    _audio_codes_embedded = torch.cat(
                        [context_tensors.additional_decoder_input, audio_codes_embedded], dim=1
                    )
                    _audio_codes_mask = torch.cat([context_tensors.additional_decoder_mask, audio_codes_mask], dim=1)
                else:
                    _audio_codes_embedded = audio_codes_embedded
                    _audio_codes_mask = audio_codes_mask

                if self.inference_parameters.apply_prior_to_layers is not None:
                    attn_prior = [None for _ in range(self.decoder.n_layers)]
                    for layer_idx in self.inference_parameters.apply_prior_to_layers:
                        attn_prior[layer_idx] = _attn_prior
                else:
                    attn_prior = _attn_prior

                if self.model_type == 'multi_encoder_context_tts':
                    attn_prior = [attn_prior, None]

                if use_cfg:
                    batch_size = audio_codes_embedded.size(0)
                    if isinstance(context_tensors.cond, list):
                        cfg_cond = [
                            torch.cat([cond_item, dummy_cond_item], dim=0)
                            for cond_item, dummy_cond_item in zip(context_tensors.cond, dummy_cond)
                        ]
                        cfg_cond_mask = [
                            torch.cat([cond_mask_item, dummy_cond_mask_item], dim=0)
                            for cond_mask_item, dummy_cond_mask_item in zip(context_tensors.cond_mask, dummy_cond_mask)
                        ]
                    else:
                        cfg_cond = torch.cat([context_tensors.cond, dummy_cond], dim=0)
                        cfg_cond_mask = torch.cat([context_tensors.cond_mask, dummy_cond_mask], dim=0)
                    cfg_audio_codes_embedded = torch.cat([_audio_codes_embedded, _audio_codes_embedded], dim=0)
                    cfg_audio_codes_mask = torch.cat([_audio_codes_mask, _audio_codes_mask], dim=0)
                    if dummy_additional_decoder_input is not None:
                        cfg_audio_codes_embedded[batch_size:, : dummy_additional_decoder_input.size(1)] = (
                            dummy_additional_decoder_input
                        )
                        cfg_audio_codes_mask[batch_size:, : dummy_additional_decoder_input.size(1)] = (
                            dummy_addition_dec_mask
                        )

                    combined_logits, attn_probs, dec_out, _ = self.forward(
                        dec_input_embedded=cfg_audio_codes_embedded,
                        dec_input_mask=cfg_audio_codes_mask,
                        cond=cfg_cond,
                        cond_mask=cfg_cond_mask,
                        attn_prior=attn_prior,
                        multi_encoder_mapping=context_tensors.multi_encoder_mapping,
                    )

                    cond_logits = combined_logits[:batch_size]
                    uncond_logits = combined_logits[batch_size:]
                    all_code_logits = (1 - cfg_scale) * uncond_logits + cfg_scale * cond_logits
                else:
                    batch_size = audio_codes_embedded.size(0)
                    all_code_logits, attn_probs, dec_out, _ = self.forward(
                        dec_input_embedded=_audio_codes_embedded,
                        dec_input_mask=_audio_codes_mask,
                        cond=context_tensors.cond,
                        cond_mask=context_tensors.cond_mask,
                        attn_prior=attn_prior,
                        multi_encoder_mapping=context_tensors.multi_encoder_mapping,
                    )

                if return_cross_attn_probs or self.inference_parameters.apply_attention_prior:
                    cross_attention_scores, all_heads_cross_attn_scores = self.get_cross_attention_scores(
                        attn_probs
                    )  # B, text_timesteps
                    alignment_attention_scores = cross_attention_scores
                    if self.inference_parameters.estimate_alignment_from_layers is not None:
                        alignment_attention_scores, _ = self.get_cross_attention_scores(
                            attn_probs, filter_layers=self.inference_parameters.estimate_alignment_from_layers
                        )  # B, text_timesteps

                    cross_attention_scores_all_timesteps.append(cross_attention_scores)
                    all_heads_cross_attn_scores_all_timesteps.append(all_heads_cross_attn_scores)

                if (
                    self.inference_parameters.apply_attention_prior
                    and idx >= self.inference_parameters.start_prior_after_n_audio_steps
                ):
                    text_time_step_attended, attended_timestep_counter = self.get_most_attended_text_timestep(
                        alignment_attention_scores=alignment_attention_scores,
                        last_attended_timesteps=last_attended_timesteps,
                        text_lens=context_tensors.text_lens,
                        lookahead_window_size=self.inference_parameters.attention_prior_lookahead_window,
                        attended_timestep_counter=attended_timestep_counter,
                        batch_size=batch_size,
                    )
                    last_attended_timesteps.append(text_time_step_attended)
                    _attn_prior, unfinished_texts, finished_texts_counter = self.construct_inference_prior(
                        prior_epsilon=self.inference_parameters.attention_prior_epsilon,
                        cross_attention_scores=cross_attention_scores,
                        text_lens=context_tensors.text_lens,
                        text_time_step_attended=text_time_step_attended,
                        attended_timestep_counter=attended_timestep_counter,
                        unfinished_texts=unfinished_texts,
                        finished_texts_counter=finished_texts_counter,
                        end_indices=end_indices,
                        lookahead_window_size=self.inference_parameters.attention_prior_lookahead_window,
                        batch_size=batch_size,
                    )

                if self.inference_parameters.ignore_finished_sentence_tracking:
                    finished_items = {}
                    unfinished_items = {}
                else:
                    finished_items = {
                        k: v for k, v in finished_texts_counter.items() if v >= 20
                    }  # Items that have been close to the end for atleast 20 timesteps
                    unfinished_items = {k: v for k, v in unfinished_texts.items() if v}

                # Don't allow termination until we have generated at least `min_generated_frames` frames (rounded up to the nearest multiple of frame_stacking_factor)
                # This guards against rare cases of termination right at the start of generation.
                forbid_audio_eos = idx * self.frame_stacking_factor < self.inference_parameters.min_generated_frames

                all_code_logits_t = all_code_logits[:, -1, :]  # (B, num_codebooks * num_tokens_per_codebook)
                if use_local_transformer_for_inference:
                    if self.local_transformer_type == LocalTransformerType.AR:
                        # Autoregressive sampling with local transformer
                        audio_codes_next = self._lt_helper.sample_autoregressive(
                            dec_output=dec_out[:, -1, :],
                            temperature=self.inference_parameters.temperature,
                            topk=self.inference_parameters.topk,
                            unfinished_items=unfinished_items,
                            finished_items=finished_items,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale,
                            use_kv_cache=self.inference_parameters.use_LT_kv_cache,
                            forbid_audio_eos=forbid_audio_eos,
                        )
                    elif self.local_transformer_type == LocalTransformerType.MASKGIT:
                        audio_codes_next = self._lt_helper.sample_maskgit(
                            dec_output=dec_out[:, -1, :],
                            temperature=self.inference_parameters.temperature,
                            topk=self.inference_parameters.topk,
                            unfinished_items=unfinished_items,
                            finished_items=finished_items,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale,
                            n_steps=maskgit_n_steps,
                            noise_scale=maskgit_noise_scale,
                            fixed_schedule=maskgit_fixed_schedule,
                            dynamic_cfg_scale=maskgit_dynamic_cfg_scale,
                            sampling_type=maskgit_sampling_type,
                            forbid_audio_eos=forbid_audio_eos,
                        )
                    else:
                        raise ValueError(
                            f"Local transformer inference requested by but local transformer type is {self.local_transformer_type}"
                        )
                else:
                    # Parallel sampling from all codebooks
                    audio_codes_next = self.sample_codes_from_logits(
                        all_code_logits_t,
                        temperature=self.inference_parameters.temperature,
                        topk=self.inference_parameters.topk,
                        unfinished_items=unfinished_items,
                        finished_items=finished_items,
                        forbid_audio_eos=forbid_audio_eos,
                    )  # (B, num_codebooks, frame_stacking_factor)
                all_codes_next_argmax = self.sample_codes_from_logits(
                    all_code_logits_t,
                    temperature=0.01,
                    topk=1,
                    unfinished_items=unfinished_items,
                    finished_items=finished_items,
                    forbid_audio_eos=forbid_audio_eos,
                )  # (B, num_codebooks, frame_stacking_factor)

                for item_idx in range(all_codes_next_argmax.size(0)):
                    if item_idx not in end_indices:
                        end_frame_index = self.detect_eos(
                            audio_codes_next[item_idx], all_codes_next_argmax[item_idx], eos_detection_method
                        )
                        if end_frame_index != float('inf'):
                            global_index = idx * self.frame_stacking_factor + end_frame_index
                            end_indices[item_idx] = global_index
                            print(f"End detected for item {item_idx} at decoder timestep: {idx}")

                all_predictions.append(audio_codes_next)
                audio_codes_input = torch.cat([audio_codes_input, audio_codes_next], dim=-1)  # (B, C, T')
                audio_codes_lens = audio_codes_lens + self.frame_stacking_factor
                if len(end_indices) == text.size(0) and len(all_predictions) >= 4:
                    # Codec must be of atleast 4 timesteps to be decoded properly
                    print("All ends reached")
                    break
            tts_generation_time = time.time() - start_time
            tts_generation_time_per_frame = tts_generation_time / (len(all_predictions) * self.frame_stacking_factor)

            # Concatenate the list of predictions along the time dimension. Note that when frame stacking is on,
            # this also undoes the stacking.
            predicted_codes = torch.cat(all_predictions, dim=-1)  # (B, num_codebooks, T')
            predicted_lens = [
                end_indices.get(idx, self.inference_parameters.max_decoder_steps) for idx in range(text.size(0))
            ]  #  Ensure that the codec is atleast of length 4
            predicted_codes_lens = torch.tensor(predicted_lens, device=text.device).long()
            predicted_codes = predicted_codes[:, :, : predicted_codes_lens.max()]

            predicted_audio, predicted_audio_lens, predicted_codes = self._codec_helper.codes_to_audio(
                predicted_codes, predicted_codes_lens
            )
            end_time = time.time()
            total_audio_duration_generated = (
                predicted_audio_lens.max().item() * predicted_audio_lens.shape[0]
            ) / self.output_sample_rate
            rtf = total_audio_duration_generated / (end_time - start_time)
            rtf_metrics = {
                'rtf': rtf,
                'time_to_first_prediction': time_to_first_prediction,
                'tts_generation_time': tts_generation_time,
                'max_frames_generated': len(all_predictions),
                'tts_generation_time_per_frame': tts_generation_time_per_frame,
                'batch_size': text.size(0),
            }
            torch.cuda.empty_cache()
            cross_attention_maps = None
            headwise_cross_attention_maps = None
            if return_cross_attn_probs:
                cross_attention_maps, headwise_cross_attention_maps = self.get_inference_attention_plots(
                    cross_attention_scores_all_timesteps,
                    all_heads_cross_attn_scores_all_timesteps,
                    context_tensors.text_lens,
                    predicted_codes_lens,
                    text.size(0),
                    compute_all_heads_attn_maps,
                    last_attended_timesteps,
                )

            return InferBatchOutput(
                predicted_audio=predicted_audio,
                predicted_audio_lens=predicted_audio_lens,
                predicted_codes=predicted_codes,
                predicted_codes_lens=predicted_codes_lens,
                rtf_metrics=rtf_metrics,
                cross_attention_maps=cross_attention_maps,
                headwise_cross_attention_maps=headwise_cross_attention_maps,
            )

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            test_dl_batch_size = self._test_dl.batch_size
            use_cfg = self.cfg.get('inference_use_cfg', False)
            self.inference_parameters.max_decoder_steps = self.cfg.get('max_decoder_steps', 500)
            self.inference_parameters.temperature = self.cfg.get('inference_temperature', 0.7)
            self.inference_parameters.topk = self.cfg.get('inference_topk', 80)
            self.inference_parameters.cfg_scale = self.cfg.get('inference_cfg_scale', 1.0)

            output = self.infer_batch(
                batch,
                use_cfg=use_cfg,
            )
            predicted_audio = output.predicted_audio
            predicted_audio_lens = output.predicted_audio_lens

            for logger in self.loggers:
                is_wandb = isinstance(logger, WandbLogger)
                is_tb = isinstance(logger, TensorBoardLogger)
                if not is_wandb and not is_tb:
                    raise ValueError(
                        "Invalid logger type for audio logging: {type(logger)}. Only `WandbLogger` and `TensorBoardLogger` are supported."
                    )

                for idx in range(predicted_audio.size(0)):
                    predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                    predicted_audio_np = predicted_audio_np[: predicted_audio_lens[idx]]
                    item_idx = batch_idx * test_dl_batch_size + idx

                    if is_wandb:
                        log_dict = {
                            "test/predicted_audio": wandb.Audio(
                                predicted_audio_np, sample_rate=self.output_sample_rate, caption="Predicted Audio"
                            ),
                        }
                        logger.experiment.log(log_dict, step=item_idx)

                    if is_tb:
                        logger.experiment.add_audio(
                            'test/predicted_audio',
                            predicted_audio_np,
                            global_step=item_idx,
                            sample_rate=self.output_sample_rate,
                        )

                    # Save the predicted audio
                    log_dir = logger.log_dir
                    audio_dir = os.path.join(log_dir, 'audios')
                    if not os.path.exists(audio_dir):
                        os.makedirs(audio_dir)
                    audio_path = os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}.wav')
                    sf.write(audio_path, predicted_audio_np, self.output_sample_rate)

    def multi_validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Called for each validation dataloader at the end of validation epoch.
        Computes metrics for this specific dataloader.

        Args:
            outputs: List of outputs from validation_step for this specific dataloader
            dataloader_idx: Index of the current dataloader

        Returns:
            A tuple of (log_dict, moe_expert_data):
                - log_dict: scalar metrics suitable for self.log()
                - moe_expert_data: per-expert usage/selection_freq tensors of shape (num_experts,), or None
        """

        def collect_required_metric(outputs, key, dim=None):
            values = [x[key] for x in outputs if key in x and x[key] is not None]
            if len(values) == 0:
                raise ValueError(
                    f"No valid values found for required metric '{key}' in validation outputs "
                    f"for dataloader {dataloader_idx}. This indicates an issue with validation."
                )
            return torch.stack(values).mean(dim=dim)

        def collect_optional_metric(outputs, key, dim=None):
            """Collect optional metric - returns None if not found."""
            values = [x[key] for x in outputs if key in x and x[key] is not None]
            if len(values) == 0:
                return None
            return torch.stack(values).mean(dim=dim)

        if len(outputs) == 0:
            raise ValueError(
                f"No validation outputs for dataloader {dataloader_idx}. "
                f"This indicates an issue with the validation dataloader or validation step."
            )

        # Compute required metrics
        val_loss = collect_required_metric(outputs, 'val_loss')
        val_codebook_loss = collect_required_metric(outputs, 'val_codebook_loss')

        log_dict = {
            'loss': val_loss,
            'codebook_loss': val_codebook_loss,
        }

        # Compute optional metrics
        VAL_OPTIONAL_METRICS = [
            'val_alignment_loss',
            'val_aligner_encoder_loss',
            'val_local_transformer_loss',
            'val_moe_load_balancing_loss',
            'val_moe_router_z_loss',
        ]
        for metric_key in VAL_OPTIONAL_METRICS:
            metric_value = collect_optional_metric(outputs, metric_key)
            if metric_value is not None:
                log_dict[metric_key.removeprefix('val_')] = metric_value

        # Exclude MoE metrics whose loss scale is disabled
        if self.use_moe:
            if self.moe_auxiliary_loss.load_balancing_loss.loss_scale <= 0:
                log_dict.pop('moe_load_balancing_loss', None)
            if self.moe_auxiliary_loss.router_z_loss.loss_scale <= 0:
                log_dict.pop('moe_router_z_loss', None)

        # Collect per-expert usage vectors
        val_moe_expert_usage_stats = [
            x.get('val_moe_expert_usage_stats') for x in outputs if x.get('val_moe_expert_usage_stats') is not None
        ]
        moe_expert_data = None
        if len(val_moe_expert_usage_stats) > 0:
            val_moe_expert_usage = collect_required_metric(val_moe_expert_usage_stats, 'expert_usage', dim=0)
            val_moe_expert_selection_freq = collect_required_metric(
                val_moe_expert_usage_stats, 'expert_selection_freq', dim=0
            )
            val_layer_expert_usage = collect_required_metric(val_moe_expert_usage_stats, 'layer_expert_usage', dim=0)
            ideal_usage = val_moe_expert_usage_stats[0]['ideal_usage']
            moe_expert_data = {
                'moe_expert_usage': val_moe_expert_usage,
                'moe_expert_selection_freq': val_moe_expert_selection_freq,
                'layer_expert_usage': val_layer_expert_usage,
                'ideal_usage': ideal_usage,
            }

        return log_dict, moe_expert_data

    def on_validation_epoch_end(self):
        """
        Computes and logs metrics across all validation dataloaders.

        Three-phase structure:
        1. Compute — aggregates metrics and collect media/heatmap data from all dataloaders.
        2. WandB media — logs all non-scalar media (audio, attention images, MoE heatmaps).
        3. Scalars — logs loss metrics and per-expert usage scalars.
        """
        if len(self.validation_step_outputs) == 0:
            return {}

        num_dataloaders = len(self.validation_step_outputs)

        # --- Phase 1: Compute all metrics + collect media data ---
        all_moe_expert_data: List[Tuple[str, Dict[str, torch.Tensor]]] = []
        all_media_data: List[Dict[str, Any]] = []
        per_dl_logs: List[Tuple[str, Dict[str, torch.Tensor]]] = []
        aggregated_metrics: Dict[str, List[torch.Tensor]] = {}

        for dataloader_idx, val_outputs in enumerate(self.validation_step_outputs):
            if len(val_outputs) == 0:
                raise ValueError(
                    f"Validation dataloader {dataloader_idx} produced no outputs. "
                    f"Check that the dataset is not empty and validation_step is working correctly."
                )

            dataloader_logs, moe_expert_data = self.multi_validation_epoch_end(
                val_outputs, dataloader_idx=dataloader_idx
            )

            dataloader_prefix = self.get_validation_dataloader_prefix(dataloader_idx)
            per_dl_logs.append((dataloader_prefix, dataloader_logs))

            if moe_expert_data is not None:
                all_moe_expert_data.append((dataloader_prefix, moe_expert_data))

            if len(val_outputs) > 0 and 'media_data' in val_outputs[0]:
                all_media_data.append(val_outputs[0]['media_data'])

            for metric_name, metric_value in dataloader_logs.items():
                aggregated_metrics.setdefault(metric_name, []).append(metric_value)

        for idx in range(num_dataloaders):
            self.validation_step_outputs[idx].clear()

        # Validate required metrics were collected
        for required_metric in ['loss', 'codebook_loss']:
            if required_metric not in aggregated_metrics or len(aggregated_metrics[required_metric]) == 0:
                raise ValueError(f"No {required_metric} collected from any dataloader.")

        # --- Phase 2: Single WandB media log (rank 0 only) ---
        if self.global_rank == 0:
            global_step = int(self.global_step)
            wandb_media: Dict[str, Any] = {}

            for media_data in all_media_data:
                media_entries = self._collect_wandb_media_and_log_tb(**media_data, global_step=global_step)
                wandb_media.update(media_entries)

            # heatmaps show layer×expert routing structure
            if all_moe_expert_data:
                for dataset_name, moe_data in all_moe_expert_data:
                    heatmap_np = plot_expert_usage_heatmap_to_numpy(
                        layer_expert_usage=moe_data['layer_expert_usage'].float().cpu().numpy(),
                        ideal_usage=moe_data['ideal_usage'],
                        title=f"MoE Expert Usage — {dataset_name} (step {int(self.global_step)})",
                    )
                    wandb_media[f"MoE:{dataset_name}/expert_usage_heatmap"] = wandb.Image(heatmap_np)

                if self._moe_train_layer_usage_accum is not None and self._moe_train_accum_steps > 0:
                    avg_layer_usage = self._moe_train_layer_usage_accum / self._moe_train_accum_steps
                    heatmap_np = plot_expert_usage_heatmap_to_numpy(
                        layer_expert_usage=avg_layer_usage.float().cpu().numpy(),
                        ideal_usage=1.0 / self._moe_num_experts,
                        title=f"MoE Expert Usage — train ({self._moe_train_accum_steps} steps avg, step {int(self.global_step)})",
                    )
                    wandb_media["MoE:train/expert_usage_heatmap"] = wandb.Image(heatmap_np)

                    self._moe_train_layer_usage_accum.zero_()
                    self._moe_train_accum_steps = 0

            if wandb_media:
                for logger in self.loggers:
                    if isinstance(logger, WandbLogger):
                        logger.experiment.log(wandb_media, commit=False)

        # --- Phase 3: Scalar metrics ---
        for dataloader_prefix, dataloader_logs in per_dl_logs:
            for metric_name, metric_value in dataloader_logs.items():
                self.log(
                    f"Loss:{dataloader_prefix}/{metric_name}",
                    metric_value,
                    prog_bar=(num_dataloaders == 1),
                    sync_dist=True,
                )

        checkpoint_loss = aggregated_metrics['loss'][0]
        if num_dataloaders > 1:
            for metric_name, metric_values in aggregated_metrics.items():
                if "loss" in metric_name:
                    avg_value = torch.stack(metric_values).mean()
                    self.log(f"Loss:val_avg/{metric_name}", avg_value, prog_bar=True, sync_dist=True)
                    if metric_name == 'loss':
                        checkpoint_loss = avg_value

        self.log(
            "val_loss",
            checkpoint_loss,
            prog_bar=False,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            logger=False,
            enable_graph=False,
        )

        if all_moe_expert_data:
            for dataset_name, moe_data in all_moe_expert_data:
                expert_usage = moe_data['moe_expert_usage']
                expert_sel_freq = moe_data['moe_expert_selection_freq']

                for eidx in range(len(expert_usage)):
                    self.log(f'MoE:{dataset_name}/Expert_{eidx:02d}_usage', expert_usage[eidx], sync_dist=True)
                    self.log(
                        f'MoE:{dataset_name}/Expert_{eidx:02d}_selection_freq', expert_sel_freq[eidx], sync_dist=True
                    )

        return {}

    def get_dataset(self, dataset_cfg, dataset_type):
        if 'datasets' not in dataset_cfg or not isinstance(dataset_cfg.datasets, (dict, DictConfig)):
            raise ValueError(
                "Expected 'datasets' key (dict) in dataset config with _target_, dataset_meta, etc. "
                f"Got keys: {list(dataset_cfg.keys())}"
            )

        dataset = instantiate(
            dataset_cfg.datasets,
            sample_rate=self.sample_rate,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            num_audio_codebooks=self.data_num_audio_codebooks,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            prior_scaling_factor=self.cfg.prior_scaling_factor,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=dataset_type,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            use_text_conditioning_tokenizer=self.cfg.use_text_conditioning_encoder,
            text_conditioning_tokenizer_name=self.text_conditioning_tokenizer_name,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
            text_context_remapping=self.text_context_remapping,
            text_context_remapping_prob=self.text_context_remapping_prob,
        )
        dataset.load_16khz_audio = False
        dataset.tokenizer_config = (
            self.cfg.text_tokenizers
        )  # This will be used in worker_init_fn for instantiating tokenizer
        return dataset

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """
        Setup validation data with support for multiple datasets.
        Overrides parent class to handle both non-lhotse and lhotse dataloaders.

        Non-lhotse config (datasets is a dict -- single dataloader, multiplicity via dataset_meta)::

            validation_ds:
                datasets:
                    _target_: nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDataset
                    dataset_meta: ...
                    min_duration: 0.2
                    max_duration: 20.0
                dataloader_params: ...

        Note: Non-lhotse creates a single dataloader even when dataset_meta contains
        multiple entries (e.g., ``{en: ..., es: ...}``). All datasets are mixed
        in one dataloader, so validation metrics are logged jointly (e.g.,
        prefix ``"en+es"``) rather than per-dataset. For per-dataset validation
        metrics, use the lhotse config with separate datasets list entries.

        Lhotse config (datasets is a list -- multiple dataloaders)::

            validation_ds:
                use_lhotse: true
                # ... shared settings ...
                datasets:
                    - name: "val_set_0"
                      input_cfg: [...] or path to an external YAML file
                    - name: "val_set_1"
                      input_cfg: [...] or path to an external YAML file
        """
        # Set placeholders that may be overridden
        self._val_dl_idx: int = 0
        self._validation_names: Optional[List[str]] = None
        self._validation_dl: Optional[torch.utils.data.DataLoader] = None

        # Preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        if 'datasets' not in val_data_config:
            raise ValueError(
                "validation_ds config must contain a 'datasets' key. "
                "For non-lhotse: a dict with _target_, dataset_meta, etc. "
                "For lhotse: a list of dataset configurations. "
                "See magpietts.yaml or magpietts_lhotse.yaml for examples."
            )

        datasets_value = val_data_config.datasets

        # Non-lhotse: datasets is a dict (single dataloader, multiplicity via dataset_meta)
        if isinstance(datasets_value, (dict, DictConfig)):
            dataset_meta = datasets_value.get('dataset_meta', {})
            if dataset_meta:
                val_name = '+'.join(dataset_meta.keys())
            else:
                val_name = 'val_set_0'
            logging.info(f"Setting up single non-lhotse validation dataloader: '{val_name}'")
            self._validation_names = [val_name]
            self._validation_dl = [self._setup_test_dataloader(val_data_config)]
            return

        # Lhotse: datasets is a path to an external YAML file (supports local paths and remote URLs like s3://) or a list
        if isinstance(datasets_value, (str, Path)):
            logging.info(f"Loading validation datasets from external file: {datasets_value}")
            datasets_list = OmegaConf.create(load_yaml(datasets_value))
        elif isinstance(datasets_value, (list, ListConfig)):
            datasets_list = datasets_value
        else:
            raise ValueError(
                f"Lhotse 'datasets' in `validation_ds` must be a non-empty list of dataset configurations. "
                f"Got: {type(datasets_value).__name__}"
            )

        if len(datasets_list) == 0:
            raise ValueError("Lhotse 'datasets' in `validation_ds` must be a non-empty list.")

        logging.info(f"Setting up {len(datasets_list)} validation dataset(s)")

        dataloaders = []
        dataset_names = []

        # Extract shared config (everything except 'datasets' key)
        shared_config = OmegaConf.create(val_data_config)
        shared_config.pop('datasets', None)

        for idx, dataset_config in enumerate(datasets_list):
            merged_config = OmegaConf.merge(shared_config, dataset_config)

            if isinstance(dataset_config, (dict, DictConfig)) and 'name' in dataset_config:
                dataset_name = dataset_config['name']
            else:
                dataset_name = f"val_set_{idx}"

            dataset_names.append(dataset_name)

            # Remove 'name' field from config as it's not needed for dataloader setup
            temp_config = OmegaConf.create(merged_config)
            temp_config.pop('name', None)

            dataloader = self._setup_test_dataloader(temp_config)
            dataloaders.append(dataloader)
            logging.info(f"  - Validation dataset {idx}: '{dataset_name}'")

        self._validation_names = dataset_names
        self._validation_dl = dataloaders
        logging.info(f"Successfully setup {len(dataloaders)} validation dataloader(s)")

    def get_lhotse_dataloader(self, dataset_cfg, mode='train') -> torch.utils.data.DataLoader:
        # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
        #   cfg is a classifier-free guidance.
        dataset = MagpieTTSLhotseDataset(
            sample_rate=self.sample_rate,
            volume_norm=dataset_cfg.volume_norm,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            num_audio_codebooks=self.data_num_audio_codebooks,
            prior_scaling_factor=self.cfg.prior_scaling_factor,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=mode,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            load_16khz_audio=False,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
            use_text_conditioning_tokenizer=self.cfg.use_text_conditioning_encoder,
            text_conditioning_tokenizer_name=self.text_conditioning_tokenizer_name,
            tokenizer_config=self.cfg.text_tokenizers,
            text_context_remapping=self.text_context_remapping,
            text_context_remapping_prob=self.text_context_remapping_prob,
        )
        data_loader = get_lhotse_dataloader_from_config(
            config=dataset_cfg,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=dataset,
        )
        return data_loader

    def setup_training_data(self, dataset_cfg):
        if dataset_cfg.get("use_lhotse", False):
            # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
            #   cfg is a classifier-free guidance.

            # specify target sampling rate the same as codec model's because lhotse config defaults 16_000.
            if not isinstance(dataset_cfg, DictConfig):
                dataset_cfg = OmegaConf.create(dataset_cfg)
            OmegaConf.set_struct(dataset_cfg, False)
            dataset_cfg.update({"sample_rate": self.sample_rate})
            OmegaConf.set_struct(dataset_cfg, True)

            self._train_dl = self.get_lhotse_dataloader(dataset_cfg, mode='train')
        else:
            dataset = self.get_dataset(dataset_cfg, dataset_type='train')
            sampler = dataset.get_sampler(dataset_cfg.dataloader_params.batch_size, world_size=self.trainer.world_size)
            persistent_workers = True
            if dataset_cfg.dataloader_params.num_workers == 0:
                persistent_workers = False
                # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
                dataset.text_tokenizer = setup_tokenizers(
                    all_tokenizers_config=self.cfg.text_tokenizers,
                    mode='train',
                )
            self._train_dl = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                sampler=sampler,
                **dataset_cfg.dataloader_params,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )

    def _setup_test_dataloader(self, dataset_cfg) -> torch.utils.data.DataLoader:
        if dataset_cfg.get("use_lhotse", False):
            # specify target sampling rate the same as codec model's because lhotse config defaults 16_000.
            if not isinstance(dataset_cfg, DictConfig):
                dataset_cfg = OmegaConf.create(dataset_cfg)
            OmegaConf.set_struct(dataset_cfg, False)
            dataset_cfg.update({"sample_rate": self.sample_rate})
            OmegaConf.set_struct(dataset_cfg, True)
            data_loader = self.get_lhotse_dataloader(dataset_cfg, mode='test')
        else:
            dataset = self.get_dataset(dataset_cfg, dataset_type='test')
            persistent_workers = True
            if dataset_cfg.dataloader_params.num_workers == 0:
                persistent_workers = False
                # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
                dataset.text_tokenizer = setup_tokenizers(all_tokenizers_config=self.cfg.text_tokenizers, mode='test')

            data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                **dataset_cfg.dataloader_params,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )
        return data_loader

    def setup_validation_data(self, dataset_cfg):
        """Required by ModelPT (abstract). Use setup_multiple_validation_data instead."""
        self._validation_names = ['val_set_0']
        self._validation_dl = [self._setup_test_dataloader(dataset_cfg)]

    def setup_test_data(self, dataset_cfg):
        self._test_dl = self._setup_test_dataloader(dataset_cfg)

    def _get_normalized_text(self, transcript: str, language: str) -> str:
        """Get normalized text using cached normalizer for the specified language.

        Args:
            transcript: Raw text to normalize.
            language: Language code (e.g., 'en', 'de', 'es').

        Returns:
            Normalized text, or original text if normalization fails/unavailable.
        """
        # Check if normalizer for this language is already cached
        if language not in self._text_normalizers:
            try:
                from nemo_text_processing.text_normalization.normalize import Normalizer

                normalizer = Normalizer(input_case='cased', lang=language)
                self._text_normalizers[language] = normalizer
                logging.info(f"Initialized text normalizer for language: {language}")
            except ImportError:
                self._text_normalizers[language] = None
                logging.warning(
                    "nemo_text_processing not installed. Skipping text normalization. "
                    "Install with: pip install nemo_text_processing"
                )
            except Exception as e:
                # Handle unsupported language or other initialization errors
                self._text_normalizers[language] = None
                logging.warning(
                    f"Failed to initialize text normalizer for language '{language}': {e}. "
                    f"Skipping text normalization. Text will be used as-is."
                )

        # Use cached normalizer if available
        normalizer = self._text_normalizers[language]
        if normalizer is not None:
            normalized_text = normalizer.normalize(transcript, verbose=False)
            return normalized_text

        return transcript

    def do_tts(
        self,
        transcript: str,
        language: str = "en",
        apply_TN: bool = False,
        use_cfg: bool = True,
        speaker_index: Optional[int] = None,
    ) -> tuple:
        """
        Generate speech from raw text transcript.

        This is a convenience method for single-utterance text-to-speech synthesis.
        For batch processing, use `infer_batch` directly. Only supports baked context embedding
        context injection, NO audio conditioning and text conditioning.
        Custom voice generation is not supported by this method.

        Args:
            transcript: Raw text to synthesize.
            language: Language code for text normalization and tokenization.
                Supported values depend on model's tokenizer configuration.
                Common: "en" (English), "de" (German), "es" (Spanish), etc.
            apply_TN: Whether to apply text normalization to the transcript.
                If True, uses nemo_text_processing for normalization.
            use_cfg: Whether to use classifier-free guidance.
            speaker_index: Speaker index for multi-speaker baked embeddings.
                Valid range: [0, num_baked_speakers - 1]. If None, uses speaker 0.
                Only applicable for models with baked context embeddings.

        Returns:
            Tuple of (audio, audio_len) where:
                audio: Generated audio waveform. Shape: (1, T_audio).
                audio_len: Length of generated audio in samples. Shape: (1,).

        Raises:
            ValueError: If model does not have a baked context embedding.
            ValueError: If speaker_index is out of valid range.
            ImportError: If apply_TN=True but nemo_text_processing is not installed.

        Example:
            >>> # If text does not need to be normalized
            >>> audio, audio_len = model.do_tts("Hello, how are you today?")
            >>>
            >>> # If text needs to be normalized
            >>> audio, audio_len = model.do_tts(
            ...     "Hello, how are you today?",
            ...     apply_TN=True,
            ... )
            >>>
            >>> # Use a specific speaker (for multi-speaker models)
            >>> audio, audio_len = model.do_tts(
            ...     "Hello!", speaker_index=2
            ... )
        """
        if not self.has_baked_context_embedding:
            raise ValueError(
                "Model does not have a baked context embedding. Please use a checkpoint with a baked context embedding."
            )
        # Workaround for bug in Ja normalizer, Ja normalizer does not work well with spaces.
        if language == "ja":
            transcript = re.sub(r'\s+', '', transcript)
        # Apply text normalization if requested
        normalized_text = (
            self._get_normalized_text(transcript=transcript, language=language) if apply_TN else transcript
        )

        # Determine tokenizer name based on language using centralized mapping
        available_tokenizers = list(self.tokenizer.tokenizers.keys())
        tokenizer_name = get_tokenizer_for_language(language, available_tokenizers)
        logging.info(f"Using tokenizer '{tokenizer_name}' for language '{language}'")

        # Unified inference path: chunk_text_for_inference automatically decides
        # whether to split based on language-specific thresholds
        # - Short text (below threshold): returns single chunk
        # - Long text (above threshold): returns multiple sentence chunks
        chunked_tokens, chunked_tokens_len, _ = chunk_text_for_inference(
            text=normalized_text,
            language=language,
            tokenizer_name=tokenizer_name,
            text_tokenizer=self.tokenizer,
            eos_token_id=self.eos_id,
        )

        num_chunks = len(chunked_tokens)

        with torch.no_grad():
            chunk_state = self.create_chunk_state(batch_size=1)
            all_codes = []

            for chunk_idx, (tokens, tokens_len) in enumerate(zip(chunked_tokens, chunked_tokens_len)):
                batch = {
                    'text': tokens.unsqueeze(0).to(self.device),
                    'text_lens': torch.tensor([tokens_len], device=self.device, dtype=torch.long),
                    'speaker_indices': speaker_index,
                }
                end_of_text = [chunk_idx == num_chunks - 1]
                beginning_of_text = chunk_idx == 0

                output = self.generate_speech(
                    batch,
                    chunk_state=chunk_state,
                    end_of_text=end_of_text,
                    beginning_of_text=beginning_of_text,
                    use_cfg=use_cfg,
                    use_local_transformer_for_inference=True,
                )
                if output.predicted_codes_lens[0] > 0:
                    all_codes.append(output.predicted_codes[0, :, : output.predicted_codes_lens[0]])

            # Concatenate and convert to audio
            if len(all_codes) > 0:
                concatenated_codes = torch.cat(all_codes, dim=1).unsqueeze(0)
                codes_lens = torch.tensor([concatenated_codes.shape[2]], device=self.device, dtype=torch.long)
                predicted_audio, predicted_audio_lens, _ = self._codec_helper.codes_to_audio(
                    concatenated_codes, codes_lens
                )
                return predicted_audio, predicted_audio_lens
            else:
                return torch.zeros(1, 0, device=self.device), torch.zeros(1, device=self.device, dtype=torch.long)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def create_chunk_state(self, batch_size: int) -> ChunkState:
        """Create fresh state for chunked inference over a batch.

        This method creates a ChunkState dataclass instance that tracks
        mutable state across multiple calls to generate_speech() when
        processing text in one or more chunks.

        The returned state object should be:
        1. Created once per batch by the inference runner
        2. Passed to each call of generate_speech()
        3. Updated in-place during generation

        Args:
            batch_size: Number of items in the batch.

        Returns:
            ChunkState with initialized state for the batch.

        Example:
            >>> chunk_state = model.create_chunk_state(batch_size=4)
            >>> for chunk in text_chunks:
            ...     output = model.generate_speech(batch, chunk_state, ...)
        """
        return ChunkState(batch_size=batch_size)

    def _set_attention_prior_weights(
        self,
        attn_prior: torch.Tensor,
        batch_idx: int,
        attended_pos: int,
        text_len: int,
        eps_sq: float,
    ) -> None:
        """
        Set attention prior weights around the currently attended position.

        Creates a distribution that:
        - Strongly suppresses positions before (attended - 1)
        - Peaks at the current attended position
        - Gradually decays for lookahead positions
        - Suppresses far-future positions

        Args:
            attn_prior: Prior tensor to modify in-place. Shape: (B, 1, T_text).
            batch_idx: Index of current batch item.
            attended_pos: Currently attended text position (chunk-relative).
            text_len: Length of text for this batch item.
            eps_sq: Squared epsilon for strong suppression.
        """
        prior_weights = self.chunked_inference_config.prior_weights

        # Suppress history (before attended - 1)
        history_end = max(1, attended_pos - 1)
        attn_prior[batch_idx, 0, :history_end] = eps_sq

        # Set weights around attended position
        attn_prior[batch_idx, 0, history_end] = prior_weights[0]  # History exposure
        attn_prior[batch_idx, 0, attended_pos] = prior_weights[1]  # Current (peak)

        # Lookahead positions with bounds checking
        for offset, weight in enumerate(prior_weights[2:], start=1):
            pos = attended_pos + offset
            if pos < text_len:
                attn_prior[batch_idx, 0, pos] = weight

        # Suppress far future (position +5 onwards)
        future_start = attended_pos + len(prior_weights) - 1
        if future_start < text_len:
            attn_prior[batch_idx, 0, future_start:] = eps_sq

    def _penalize_attention_sinks(
        self,
        attn_prior: torch.Tensor,
        batch_idx: int,
        attended_timestep_counter: Dict[int, int],
        left_offset: int,
        eps_sq: float,
    ) -> None:
        """
        Penalize timesteps that have been over-attended (attention sinks).

        When a position is attended more than the threshold, suppress all
        positions up to and including it to force the model to move forward.

        Args:
            attn_prior: Prior tensor to modify in-place. Shape: (B, 1, T_text).
            batch_idx: Index of current batch item.
            attended_timestep_counter: Dict tracking attention counts per timestep.
            left_offset: Chunk offset for this batch item.
            eps_sq: Squared epsilon for strong suppression.
        """
        threshold = self.chunked_inference_config.attention_sink_threshold

        for timestep, count in attended_timestep_counter.items():
            if timestep > left_offset and count >= threshold:
                logging.debug(f"Attention sink at timestep {timestep} for batch {batch_idx}, count: {count}")
                relative_pos = timestep - left_offset
                attn_prior[batch_idx, 0, : relative_pos + 1] = eps_sq

    def _update_text_completion_state(
        self,
        batch_idx: int,
        attended_pos: int,
        text_len: int,
        is_finished: bool,
        unfinished_texts: Dict[int, bool],
        finished_texts_counter: Dict[int, int],
    ) -> None:
        """
        Update tracking state for text completion detection.

        A text is considered "near end" when the attended position is within
        ``near_end_threshold`` positions of the text end.

        Args:
            batch_idx: Index of current batch item.
            attended_pos: Currently attended text position (chunk-relative).
            text_len: Length of text for this batch item.
            is_finished: Whether this batch item has already finished.
            unfinished_texts: Dict to update in-place.
            finished_texts_counter: Dict to update in-place.
        """
        is_near_end = attended_pos >= text_len - self.chunked_inference_config.near_end_threshold

        # Text is unfinished if not near end AND not already marked finished
        unfinished_texts[batch_idx] = not is_near_end and not is_finished

        # Start counting when near end or already finished
        if is_near_end or is_finished:
            finished_texts_counter.setdefault(batch_idx, 0)

    def construct_multi_chunk_prior(
        self,
        prior_epsilon: float,
        cross_attention_scores: torch.Tensor,
        text_lens: torch.Tensor,
        text_time_step_attended: List[int],
        attended_timestep_counter: List[Dict[int, int]],
        unfinished_texts: Dict[int, bool],
        finished_texts_counter: Dict[int, int],
        end_indices: Dict[int, int],
        chunk_end_dict: Dict[int, int],
        batch_size: int,
        left_offset: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, bool], Dict[int, int]]:
        """
        Construct attention prior for multi-chunk inference with chunked text.

        Builds a soft attention prior that guides the decoder to attend to appropriate
        text positions, preventing attention drift and encouraging monotonic progression.

        Args:
            prior_epsilon: Base probability for non-targeted positions.
            cross_attention_scores: Attention scores for shape/device inference.
                Shape: (effective_batch, text_length).
            text_lens: Length of text for each batch item. Shape: (batch_size,).
            text_time_step_attended: Most attended text position (absolute) per batch item.
            attended_timestep_counter: Per-batch dicts tracking attention counts per timestep.
            unfinished_texts: Updated in-place. True if text still being processed.
            finished_texts_counter: Updated in-place. Counts consecutive near-end timesteps.
            end_indices: Batch indices that have reached end-of-sequence.
            chunk_end_dict: Batch indices that have reached chunk end.
            batch_size: Number of items in the batch.
            left_offset: Chunk offset for each batch item. Defaults to zeros.

        Returns:
            Tuple of (attention_prior, unfinished_texts, finished_texts_counter).
        """
        # Initialize with safe default (avoid mutable default argument)
        if left_offset is None:
            left_offset = [0] * batch_size

        # Extract shape info and create prior tensor
        device = cross_attention_scores.device
        effective_batch = cross_attention_scores.shape[0]  # 2 * batch_size if CFG else batch_size
        text_dim = cross_attention_scores.shape[1]
        eps_sq = prior_epsilon * prior_epsilon

        attn_prior = torch.full((effective_batch, 1, text_dim), prior_epsilon, device=device)

        # Process each batch item
        for bidx in range(min(effective_batch, batch_size)):
            text_len = int(text_lens[bidx])
            attended_pos = text_time_step_attended[bidx] - left_offset[bidx]
            is_finished = bidx in end_indices or bidx in chunk_end_dict

            # Short sentences: uniform prior (no guidance needed)
            if text_len <= self.chunked_inference_config.short_sentence_threshold:
                attn_prior[bidx, 0, :] = 1.0
            else:
                # Set attention weights around attended position
                self._set_attention_prior_weights(attn_prior, bidx, attended_pos, text_len, eps_sq)

            # Penalize attention sinks (stuck positions)
            if not is_finished:
                self._penalize_attention_sinks(
                    attn_prior, bidx, attended_timestep_counter[bidx], left_offset[bidx], eps_sq
                )

            # Update text completion tracking
            self._update_text_completion_state(
                bidx, attended_pos, text_len, is_finished, unfinished_texts, finished_texts_counter
            )

        return attn_prior, unfinished_texts, finished_texts_counter

    @staticmethod
    def _to_int(value: Union[int, torch.Tensor]) -> int:
        """Convert tensor scalar to Python int if needed."""
        return value.item() if not isinstance(value, int) else value

    def _check_eos_and_update_state(
        self,
        chunk_state: ChunkState,
        audio_codes_next: torch.Tensor,
        all_codes_next_argmax: torch.Tensor,
        chunk_end_dict: Dict[int, int],
        chunk_end_frame_lens: Dict[int, int],
        finished_texts_counter: Dict[int, int],
        end_of_text: List[bool],
        eos_detection_method: 'EOSDetectionMethod',
        current_step: int,
        batch_size: int,
    ) -> None:
        """
        Check for EOS tokens and update chunk/end tracking state.

        Args:
            chunk_state: Mutable state object tracking history across chunks.
            audio_codes_next: Sampled audio codes. Shape: (B, num_codebooks, frame_stacking_factor).
                Always 3D; when frame stacking is disabled (frame_stacking_factor=1) the last dim is 1.
            all_codes_next_argmax: Argmax sampled codes for EOS detection.
            chunk_end_dict: Maps batch indices to chunk end timesteps.
            chunk_end_frame_lens: Maps batch indices to frame-level length (for codes_to_audio); aligned with infer().
            finished_texts_counter: Counter for near-end timesteps.
            end_of_text: Whether text has ended for each batch item.
            eos_detection_method: Method for detecting end-of-sequence.
            current_step: Current decoding step index.
            batch_size: Number of items in the batch.
        """
        for item_idx in range(batch_size):
            if item_idx in chunk_state.end_indices or item_idx in chunk_end_dict:
                continue

            end_frame_index = self.detect_eos(
                audio_codes_next[item_idx], all_codes_next_argmax[item_idx], eos_detection_method
            )

            # End of speech detected. Update the state.
            if end_frame_index != float('inf'):
                frame_len = current_step * self.frame_stacking_factor + end_frame_index
                chunk_end_frame_lens[item_idx] = frame_len
                if end_of_text[item_idx]:
                    # Speech for entire multi-chunk text has ended. Update the state.
                    chunk_state.end_indices[item_idx] = chunk_state.overall_idx
                    chunk_end_dict[item_idx] = current_step
                    logging.info(
                        f"End detected for item {item_idx} at local timestep {current_step} "
                        f"and overall timestep {chunk_state.overall_idx}"
                    )
                elif item_idx not in chunk_end_dict:
                    # Chunk end detected. Update the state.
                    chunk_end_dict[item_idx] = current_step
                    logging.info(f"Chunk end detected for item {item_idx} at local timestep {current_step}")
            elif (
                not end_of_text[item_idx]
                and finished_texts_counter.get(item_idx, -1)
                >= self.chunked_inference_config.forceful_chunk_end_threshold
            ):
                chunk_end_dict[item_idx] = current_step
                chunk_end_frame_lens[item_idx] = (current_step + 1) * self.frame_stacking_factor
                logging.info(f"Forceful chunk end detected for item {item_idx} at local timestep {current_step}")

    def _should_terminate_loop(
        self,
        chunk_state: ChunkState,
        chunk_end_dict: Dict[int, int],
        end_of_text: List[bool],
        batch_size: int,
    ) -> bool:
        """
        Check if all batch items have reached their end condition.

        Args:
            chunk_state: Mutable state object tracking history across chunks.
            chunk_end_dict: Maps batch indices to chunk end timesteps.
            end_of_text: Whether text has ended for each batch item.
            batch_size: Number of items in the batch.

        Returns:
            True if all items have reached end, False otherwise.
        """
        if len(chunk_state.end_indices) == batch_size:
            logging.info("All ends reached")
            return True

        completed_count = 0
        for bidx in range(batch_size):
            if not end_of_text[bidx] and bidx in chunk_end_dict:
                completed_count += 1
            elif end_of_text[bidx] and bidx in chunk_state.end_indices:
                completed_count += 1

        if completed_count == batch_size:
            logging.info("All ends reached via chunk end")
            return True

        return False

    def _run_chunked_forward_with_cfg(
        self,
        context_tensors: Dict[str, Any],
        audio_codes_embedded: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        attn_prior: Any,
        use_cfg: bool,
        cfg_scale: float,
        dummy_cond: Optional[torch.Tensor],
        dummy_cond_mask: Optional[torch.Tensor],
        dummy_additional_decoder_input: Optional[torch.Tensor],
        dummy_addition_dec_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        """
        Run forward pass with optional classifier-free guidance.

        Args:
            context_tensors: Context tensors from prepare_context_tensors.
            audio_codes_embedded: Embedded audio codes. Shape: (B, T, E).
            audio_codes_mask: Mask for audio codes. Shape: (B, T).
            attn_prior: Attention prior tensor or list.
            use_cfg: Whether to use classifier-free guidance.
            cfg_scale: Scale factor for CFG.
            dummy_cond: Dummy conditioning for unconditional branch.
            dummy_cond_mask: Mask for dummy conditioning.
            dummy_additional_decoder_input: Dummy additional decoder input.
            dummy_addition_dec_mask: Mask for dummy additional input.
            batch_size: Number of items in the batch.

        Returns:
            Tuple of (logits, attention_probs, decoder_output).
        """
        if use_cfg:
            # Combine conditional and unconditional inputs
            if isinstance(context_tensors.cond, list):
                cfg_cond = [torch.cat([c, d], dim=0) for c, d in zip(context_tensors.cond, dummy_cond)]
                cfg_cond_mask = [torch.cat([c, d], dim=0) for c, d in zip(context_tensors.cond_mask, dummy_cond_mask)]
            else:
                cfg_cond = torch.cat([context_tensors.cond, dummy_cond], dim=0)
                cfg_cond_mask = torch.cat([context_tensors.cond_mask, dummy_cond_mask], dim=0)

            cfg_audio_embedded = torch.cat([audio_codes_embedded, audio_codes_embedded], dim=0)
            cfg_audio_mask = torch.cat([audio_codes_mask, audio_codes_mask], dim=0)

            if dummy_additional_decoder_input is not None:
                cfg_audio_embedded[batch_size:, : dummy_additional_decoder_input.size(1)] = (
                    dummy_additional_decoder_input
                )
                cfg_audio_mask[batch_size:, : dummy_additional_decoder_input.size(1)] = dummy_addition_dec_mask

            combined_logits, attn_probs, dec_out, _ = self.forward(
                dec_input_embedded=cfg_audio_embedded,
                dec_input_mask=cfg_audio_mask,
                cond=cfg_cond,
                cond_mask=cfg_cond_mask,
                attn_prior=attn_prior,
                multi_encoder_mapping=context_tensors.multi_encoder_mapping,
            )

            cond_logits = combined_logits[:batch_size]
            uncond_logits = combined_logits[batch_size:]
            all_code_logits = (1 - cfg_scale) * uncond_logits + cfg_scale * cond_logits
            # NOTE: Keep dec_out doubled for local transformer CFG handling
        else:
            all_code_logits, attn_probs, dec_out, _ = self.forward(
                dec_input_embedded=audio_codes_embedded,
                dec_input_mask=audio_codes_mask,
                cond=context_tensors.cond,
                cond_mask=context_tensors.cond_mask,
                attn_prior=attn_prior,
                multi_encoder_mapping=context_tensors.multi_encoder_mapping,
            )

        return all_code_logits, attn_probs, dec_out

    def _initialize_chunked_attn_prior(
        self,
        chunk_state: ChunkState,
        current_chunk_len: torch.Tensor,
        batch_text_lens: torch.Tensor,
        max_text_len: int,
        batch_size: int,
        use_cfg: bool,
        prior_epsilon: float,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Initialize attention prior for chunked generation with left offset tracking.

        This method constructs the initial attention prior when continuing from
        previous chunks, accounting for the sliding window over text history.

        Args:
            chunk_state: Mutable state object tracking history across chunks.
            current_chunk_len: Length of the current text chunk for each batch item.
            batch_text_lens: Text lengths for each batch item.
            max_text_len: Maximum text length in the batch.
            batch_size: Number of items in the batch.
            use_cfg: Whether classifier-free guidance is being used.
            prior_epsilon: Base epsilon value for attention prior.
            device: Target device for tensors.

        Returns:
            Attention prior tensor or None if no history exists.
        """
        if len(chunk_state.previous_attn_len) == 0:
            return None

        # Initialize prior tensor
        cfg_multiplier = 2 if use_cfg else 1
        _attn_prior = torch.zeros(batch_size * cfg_multiplier, 1, max_text_len).to(device) + prior_epsilon

        for _idx in range(batch_size):
            # Calculate left offset for sliding window
            delta_in_len = self._to_int(current_chunk_len[_idx])
            len_to_delete = self._to_int(chunk_state.previous_attn_len[_idx] + delta_in_len - batch_text_lens[_idx])
            chunk_state.left_offset[_idx] = self._to_int(chunk_state.left_offset[_idx] + len_to_delete)

            # Skip if text has ended
            if _idx in chunk_state.end_indices and chunk_state.end_indices[_idx] is not None:
                continue

            # Set prior weights for new chunk
            current_starting_point = batch_text_lens[_idx] - current_chunk_len[_idx]
            prior_weights = self.chunked_inference_config.prior_weights_init
            _attn_prior[_idx, :, :current_starting_point] = prior_epsilon * prior_epsilon
            for offset, weight in enumerate(prior_weights[:5]):
                idx = current_starting_point + offset
                if idx < max_text_len:
                    _attn_prior[_idx, :, idx] = weight

        return _attn_prior

    def _update_context_from_history(
        self,
        chunk_state: ChunkState,
        context_tensors: Dict[str, Any],
        current_chunk_len: torch.Tensor,
        max_text_len: int,
        beginning_of_text: bool,
        batch_text_lens: torch.Tensor,
        batch_size: int,
    ) -> None:
        """
        Update context tensors with cached history for chunked generation.

        This method splices historical context embeddings into the current context
        tensors to maintain continuity across text chunks.

        Args:
            chunk_state: Mutable state object tracking history across chunks.
            context_tensors: ContextTensorsOutput containing 'cond' tensor to update.
            current_chunk_len: Length of the current text chunk for each batch item.
            max_text_len: Maximum text length in the batch.
            beginning_of_text: Whether this is the first chunk.
            batch_text_lens: Text lengths for each batch item.
            batch_size: Number of items in the batch.
        """
        for _idx in range(batch_size):
            # Skip if text has ended
            if _idx in chunk_state.end_indices and chunk_state.end_indices[_idx] is not None:
                continue
            if not beginning_of_text:
                pad_len_idx = max_text_len - batch_text_lens[_idx]
                context_tensors.cond[_idx, : -current_chunk_len[_idx] - pad_len_idx] = (
                    chunk_state.history_context_tensor[
                        _idx, -(context_tensors.cond[_idx].shape[0] - current_chunk_len[_idx] - pad_len_idx) :
                    ]
                )
        chunk_state.history_context_tensor = context_tensors.cond

    def _prepare_chunked_text_tensors(
        self,
        chunk_state: ChunkState,
        batch: Dict[str, torch.Tensor],
        current_chunk_len: torch.Tensor,
        beginning_of_text: bool,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Prepare text tensors with history for chunked inference.

        This method handles the sliding window logic for text tokens, combining
        historical text with new chunks and applying window size constraints.

        Args:
            chunk_state: Mutable state object tracking history across chunks.
            batch: Input batch containing 'text' and 'text_lens'.
            current_chunk_len: Length of the current text chunk for each batch item.
            beginning_of_text: Whether this is the first chunk.
            device: Target device for tensors.

        Returns:
            Tuple of (modified batch, max_text_len).
        """
        batch_size = batch["text"].size(0)
        text_tensors = []

        for _idx in range(batch_size):
            # If text has ended, use minimal placeholder
            if _idx in chunk_state.end_indices and chunk_state.end_indices[_idx] is not None:
                batch['text_lens'][_idx] = torch.tensor(1).to(device).long()
                text_tensors.append(batch['text'][_idx])
                continue

            # Combine history with current chunk
            if chunk_state.history_text is not None:
                current_text = torch.cat(
                    [
                        chunk_state.history_text[_idx][: chunk_state.history_text_lens[_idx]],
                        batch["text"][_idx][: current_chunk_len[_idx]],
                    ]
                )
            else:
                current_text = batch["text"][_idx][: current_chunk_len[_idx]]

            # Apply sliding window
            history_len = min(current_chunk_len[_idx], self.chunked_inference_config.history_len_heuristic)
            true_window_size = current_chunk_len[_idx] + history_len
            if not beginning_of_text:
                current_text = current_text[max(0, current_text.shape[0] - true_window_size) :]

            current_text_lens = current_text.shape[0]
            text_tensors.append(current_text)
            batch['text_lens'][_idx] = torch.tensor(current_text_lens).to(device).long()

        # Pad and stack text tensors
        max_text_len = max(batch['text_lens']).item()
        batch['text'] = stack_tensors(text_tensors, max_lens=[max_text_len])

        # Update history
        chunk_state.history_text = batch['text']
        chunk_state.history_text_lens = batch['text_lens']

        return batch, max_text_len

    def generate_speech(
        self,
        batch,
        chunk_state: ChunkState,
        end_of_text,
        beginning_of_text,
        use_cfg=True,
        use_local_transformer_for_inference=False,
        maskgit_n_steps=3,
        maskgit_noise_scale=0.0,
        maskgit_fixed_schedule=None,
        maskgit_dynamic_cfg_scale=False,
        maskgit_sampling_type=None,
    ):
        """
        Unified speech generation supporting both single-chunk and multi-chunk modes.

        This method is the unified inference entry point. For short text (single chunk where
        beginning_of_text=True and end_of_text=[True]), it behaves similarly to standard inference.
        For long text (multiple chunks), it maintains a sliding window over text and audio histories,
        tracking how many audio tokens were generated for each text position.

        The behaviour is strongly dependent on self.inference_parameters.

        Args:
            batch (dict): Input batch containing 'text' and 'text_lens'.
            chunk_state (ChunkState): Mutable state object tracking history across chunks.
                Created via model.create_chunk_state() and updated in-place.
            end_of_text (List[bool]): Whether entire text has been provided for each batch item.
            beginning_of_text (bool): Whether this is the first chunk.
            use_cfg (bool): Whether to use classifier-free guidance.
            use_local_transformer_for_inference (bool): Whether to use local transformer for sampling.
            maskgit_n_steps (int): Number of MaskGit refinement steps.
            maskgit_noise_scale (float): Noise scale for MaskGit sampling.
            maskgit_fixed_schedule (Optional[List[int]]): Fixed schedule for MaskGit.
            maskgit_dynamic_cfg_scale (bool): Whether to use dynamic CFG scale in MaskGit.
            maskgit_sampling_type (Optional[str]): Type of MaskGit sampling.

        Returns:
            InferBatchOutput: Contains predicted_codes, predicted_codes_lens, and empty audio fields.
        """
        cfg_scale = self.inference_parameters.cfg_scale

        eos_detection_method = EOSDetectionMethod(self.inference_parameters.eos_detection_method)
        device = batch['text'].device
        with torch.no_grad():
            current_chunk_len = copy.deepcopy(batch['text_lens'].detach())
            batch_size = batch["text"].size(0)

            # Prepare text tensors with history
            batch, max_text_len = self._prepare_chunked_text_tensors(
                chunk_state, batch, current_chunk_len, beginning_of_text, device
            )
            context_tensors = self.prepare_context_tensors(batch)

            # Update context with historical embeddings
            self._update_context_from_history(
                chunk_state,
                context_tensors,
                current_chunk_len,
                max_text_len,
                beginning_of_text,
                batch['text_lens'],
                batch_size,
            )

            audio_codes_input = (
                torch.full(
                    (batch_size, self.num_audio_codebooks, self.frame_stacking_factor),
                    self.audio_bos_id,
                )
                .long()
                .to(device)
            )
            audio_codes_frame_lens = torch.full(
                (batch_size,), self.frame_stacking_factor, device=device, dtype=torch.long
            )
            audio_codes_lens = torch.full((batch_size,), 1, device=device, dtype=torch.long)
            audio_codes_mask = get_mask_from_lengths(audio_codes_lens)

            # Initialize dummy variables for CFG
            dummy_cond = None
            dummy_cond_mask = None
            dummy_additional_decoder_input = None
            dummy_addition_dec_mask = None
            if use_cfg:
                dummy_cond, dummy_cond_mask, dummy_additional_decoder_input, dummy_addition_dec_mask, _ = (
                    self.prepare_dummy_cond_for_cfg(
                        context_tensors.cond,
                        context_tensors.cond_mask,
                        context_tensors.additional_decoder_input,
                        context_tensors.additional_decoder_mask,
                    )
                )

            # Initialize attention prior for chunked generation
            initial_attn_prior = self._initialize_chunked_attn_prior(
                chunk_state,
                current_chunk_len,
                batch['text_lens'],
                max_text_len,
                batch_size,
                use_cfg,
                self.inference_parameters.attention_prior_epsilon,
                device,
            )
            chunk_state.previous_attn_len = copy.deepcopy(batch['text_lens'].detach().tolist())

            # Create decoder state object to track all local mutable state
            state = ChunkedDecoderState(
                audio_codes_input=audio_codes_input,
                audio_codes_lens=audio_codes_lens,
                audio_codes_mask=audio_codes_mask,
                attended_timestep_counter=[{} for _ in range(batch_size)],
                all_predictions=[],
                chunk_end_dict={},
                unfinished_texts={},
                finished_texts_counter={},
                attn_prior=initial_attn_prior,
            )
            # Frame-level lengths for this chunk only: batch_idx -> number of codec frames to keep
            # per item (used for predicted_codes_lens and trimming). Filled when EOS or chunk end
            # is detected.
            chunk_end_frame_lens: Dict[int, int] = {}

            max_steps = self.inference_parameters.max_decoder_steps // self.frame_stacking_factor
            for idx in range(max_steps):
                if idx % 30 == 0:
                    logging.info(f"Decoding timestep {idx}")

                forbid_audio_eos = idx * self.frame_stacking_factor < self.inference_parameters.min_generated_frames

                # Embed audio codes and concatenate with additional decoder input
                audio_codes_embedded, audio_codes_embedded_lens = self.embed_audio_tokens(
                    state.audio_codes_input, audio_tokens_lens=audio_codes_frame_lens
                )
                state.audio_codes_mask = get_mask_from_lengths(audio_codes_embedded_lens)
                if context_tensors.additional_decoder_input is not None:
                    _audio_codes_embedded = torch.cat(
                        [context_tensors.additional_decoder_input, audio_codes_embedded], dim=1
                    )
                    _audio_codes_mask = torch.cat(
                        [context_tensors.additional_decoder_mask, state.audio_codes_mask], dim=1
                    )
                else:
                    _audio_codes_embedded = audio_codes_embedded
                    _audio_codes_mask = state.audio_codes_mask

                # Prepare attention prior for layers
                if self.inference_parameters.apply_prior_to_layers is not None:
                    attn_prior = [None for _ in range(self.cfg.decoder.n_layers)]
                    for layer_idx in self.inference_parameters.apply_prior_to_layers:
                        attn_prior[layer_idx] = state.attn_prior
                else:
                    attn_prior = state.attn_prior

                if self.model_type == 'multi_encoder_context_tts':
                    attn_prior = [attn_prior, None]

                # Run forward pass with optional CFG
                all_code_logits, attn_probs, dec_out = self._run_chunked_forward_with_cfg(
                    context_tensors=context_tensors,
                    audio_codes_embedded=_audio_codes_embedded,
                    audio_codes_mask=_audio_codes_mask,
                    attn_prior=attn_prior,
                    use_cfg=use_cfg,
                    cfg_scale=cfg_scale,
                    dummy_cond=dummy_cond,
                    dummy_cond_mask=dummy_cond_mask,
                    dummy_additional_decoder_input=dummy_additional_decoder_input,
                    dummy_addition_dec_mask=dummy_addition_dec_mask,
                    batch_size=batch_size,
                )

                if self.inference_parameters.apply_attention_prior:
                    # Get cross-attention scores (optionally from specific layers for alignment)
                    alignment_attention_scores, _ = self.get_cross_attention_scores(
                        attn_probs, filter_layers=self.inference_parameters.estimate_alignment_from_layers
                    )  # B, text_timesteps

                    text_time_step_attended, state.attended_timestep_counter = self.get_most_attended_text_timestep(
                        alignment_attention_scores=alignment_attention_scores,
                        last_attended_timesteps=chunk_state.last_attended_timesteps,
                        text_lens=context_tensors.text_lens,
                        lookahead_window_size=self.inference_parameters.attention_prior_lookahead_window,
                        attended_timestep_counter=state.attended_timestep_counter,
                        batch_size=batch_size,
                        left_offset=chunk_state.left_offset,
                    )
                    chunk_state.last_attended_timesteps.append(
                        text_time_step_attended.detach()
                        if isinstance(text_time_step_attended, torch.Tensor)
                        else text_time_step_attended
                    )

                    # Use different attention priors for first chunk vs subsequent chunks:
                    # - First chunk: use standard inference prior (more permissive, no history suppression)
                    # - Subsequent chunks: use multi-chunk prior (more restrictive, suppresses history/future)
                    if beginning_of_text:
                        # First chunk: use standard inference prior
                        (state.attn_prior, state.unfinished_texts, state.finished_texts_counter) = (
                            self.construct_inference_prior(
                                prior_epsilon=self.inference_parameters.attention_prior_epsilon,
                                cross_attention_scores=alignment_attention_scores,
                                text_lens=context_tensors.text_lens,
                                text_time_step_attended=text_time_step_attended,
                                attended_timestep_counter=state.attended_timestep_counter,
                                unfinished_texts=state.unfinished_texts,
                                finished_texts_counter=state.finished_texts_counter,
                                end_indices=chunk_state.end_indices,
                                lookahead_window_size=self.inference_parameters.attention_prior_lookahead_window,
                                batch_size=batch_size,
                            )
                        )
                    else:
                        # Subsequent chunks: use multi-chunk inference prior
                        (state.attn_prior, state.unfinished_texts, state.finished_texts_counter) = (
                            self.construct_multi_chunk_prior(
                                prior_epsilon=self.inference_parameters.attention_prior_epsilon,
                                cross_attention_scores=alignment_attention_scores,
                                text_lens=context_tensors.text_lens,
                                text_time_step_attended=text_time_step_attended,
                                attended_timestep_counter=state.attended_timestep_counter,
                                unfinished_texts=state.unfinished_texts,
                                finished_texts_counter=state.finished_texts_counter,
                                end_indices=chunk_state.end_indices,
                                chunk_end_dict=state.chunk_end_dict,
                                batch_size=batch_size,
                                left_offset=chunk_state.left_offset,
                            )
                        )

                if not beginning_of_text:
                    # Only increment here for multi-chunk path; construct_inference_prior
                    # (used when beginning_of_text=True) already increments internally.
                    for key in state.finished_texts_counter:
                        state.finished_texts_counter[key] += 1
                        limit = (
                            self.chunked_inference_config.finished_limit_with_eot
                            if end_of_text[key]
                            else self.chunked_inference_config.finished_limit_without_eot
                        )
                        if state.finished_texts_counter[key] > limit:
                            state.unfinished_texts[key] = False

                if self.inference_parameters.ignore_finished_sentence_tracking:
                    finished_items = {}
                    unfinished_items = {}
                else:
                    finished_threshold = (
                        self.chunked_inference_config.finished_limit_first_chunk
                        if beginning_of_text
                        else self.chunked_inference_config.finished_limit_with_eot
                    )
                    finished_items = {k: v for k, v in state.finished_texts_counter.items() if v >= finished_threshold}
                    unfinished_items = {k: v for k, v in state.unfinished_texts.items() if v}

                all_code_logits_t = all_code_logits[:, -1, :]  # (B, num_codebooks * num_tokens_per_codebook)

                if use_local_transformer_for_inference:
                    if self.local_transformer_type == LocalTransformerType.AR:
                        # Autoregressive sampling with local transformer
                        audio_codes_next = self._lt_helper.sample_autoregressive(
                            dec_output=dec_out[:, -1, :],
                            temperature=self.inference_parameters.temperature,
                            topk=self.inference_parameters.topk,
                            unfinished_items=unfinished_items,
                            finished_items=finished_items,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale,
                            use_kv_cache=self.inference_parameters.use_LT_kv_cache,
                            forbid_audio_eos=forbid_audio_eos,
                        )
                    elif self.local_transformer_type == LocalTransformerType.MASKGIT:
                        audio_codes_next = self._lt_helper.sample_maskgit(
                            dec_output=dec_out[:, -1, :],
                            temperature=self.inference_parameters.temperature,
                            topk=self.inference_parameters.topk,
                            unfinished_items=unfinished_items,
                            finished_items=finished_items,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale,
                            n_steps=maskgit_n_steps,
                            noise_scale=maskgit_noise_scale,
                            fixed_schedule=maskgit_fixed_schedule,
                            dynamic_cfg_scale=maskgit_dynamic_cfg_scale,
                            sampling_type=maskgit_sampling_type,
                            forbid_audio_eos=forbid_audio_eos,
                        )
                    else:
                        raise ValueError(
                            f"Local transformer inference requested but local transformer type is {self.local_transformer_type}"
                        )
                else:
                    audio_codes_next = self.sample_codes_from_logits(
                        all_code_logits_t,
                        temperature=self.inference_parameters.temperature,
                        topk=self.inference_parameters.topk,
                        unfinished_items=unfinished_items,
                        finished_items=finished_items,
                        forbid_audio_eos=forbid_audio_eos,
                    )  # (B, num_codebooks)
                all_codes_next_argmax = self.sample_codes_from_logits(
                    all_code_logits_t,
                    temperature=self.chunked_inference_config.argmax_temperature,
                    topk=1,
                    unfinished_items=unfinished_items,
                    finished_items=finished_items,
                    forbid_audio_eos=forbid_audio_eos,
                )  # (B, num_codebooks)

                # Check for EOS and update state
                self._check_eos_and_update_state(
                    chunk_state,
                    audio_codes_next,
                    all_codes_next_argmax,
                    state.chunk_end_dict,
                    chunk_end_frame_lens,
                    state.finished_texts_counter,
                    end_of_text,
                    eos_detection_method,
                    idx,
                    batch_size,
                )

                state.all_predictions.append(audio_codes_next)

                state.audio_codes_input = torch.cat([state.audio_codes_input, audio_codes_next], dim=-1)  # (B, C, T')
                audio_codes_frame_lens = audio_codes_frame_lens + self.frame_stacking_factor

                # Check termination condition
                if self._should_terminate_loop(chunk_state, state.chunk_end_dict, end_of_text, batch_size):
                    break

                chunk_state.overall_idx += 1

            # Concatenate the list of predictions along the time dimension.
            # Note that when frame stacking is on, this also undoes the stacking.
            predicted_codes = torch.cat(state.all_predictions, dim=-1)  # (B, C, F*T_steps)
            num_steps = len(state.all_predictions)
            default_frame_len = num_steps * self.frame_stacking_factor
            predicted_codes_lens = torch.tensor(
                [chunk_end_frame_lens.get(item_idx, default_frame_len) for item_idx in range(batch_size)],
                device=device,
            )
            predicted_codes = predicted_codes[:, :, : predicted_codes_lens.max()]

            return InferBatchOutput(
                predicted_audio=torch.empty(0, device=device),
                predicted_audio_lens=torch.empty(0, device=device, dtype=torch.long),
                predicted_codes=predicted_codes,
                predicted_codes_lens=predicted_codes_lens,
                rtf_metrics={},
                cross_attention_maps=[],
                headwise_cross_attention_maps=[],
            )