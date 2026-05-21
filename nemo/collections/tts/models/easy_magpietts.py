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
import json
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data.distributed import DistributedSampler

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.fallback import FallbackDataset
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import MagpieTTSLhotseDataset, setup_tokenizers
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset

from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel, TrainingMode
from nemo.collections.tts.modules.magpietts_modules import (
    LocalTransformerType,
    add_special_tokens,
    remove_eos_token,
    remove_special_tokens,
    worker_init_fn,
)
from nemo.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
    get_speaker_embeddings_from_filepaths,
    process_text_for_cer,
    transcribe_with_whisper,
    transcribe_with_whisper_from_filepaths,
)
from nemo.core.classes.common import safe_instantiate
from nemo.utils import logging

try:
    from nemo.collections.tts.modules.utmosv2 import UTMOSv2Calculator

    HAVE_UTMOSV2 = True
except (ImportError, ModuleNotFoundError):
    HAVE_UTMOSV2 = False

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import List

@dataclass
class ProcessBatchOutput:
    """
    Output dataclass from process_batch containing loss values and model predictions.

    Attributes:
        loss: Total combined loss (codebook_loss + phoneme_loss + local_transformer_loss)
        codebook_loss: Cross-entropy loss for parallel audio codebook prediction
        phoneme_loss: Cross-entropy loss for phoneme prediction (None if no phoneme tokenizer)
        local_transformer_loss: Loss from local transformer (None if not used)
        local_transformer_logits: Logits from local transformer (None if not used)
        logits: Predicted logits for audio codes (B, T', num_codebooks * num_tokens_per_codebook)
        phoneme_logits: Predicted logits for phoneme tokens (None if no phoneme tokenizer)
        phoneme_tokens_target: Target phoneme tokens for loss computation
        phoneme_tokens_lens_target: Lengths of target phoneme tokens
        audio_codes_target: Target audio codes for loss computation (B, C, T'-1)
        audio_codes_lens_target: Lengths of target audio codes (B,)
        context_audio_codes: Processed context audio codes (B, C, T')
        context_audio_codes_lens: Length of processed context audio codes (B,)
        selected_training_mode: Name of the training mode used for this batch (e.g., "streaming_4_8")
    """

    loss: torch.Tensor
    codebook_loss: torch.Tensor
    phoneme_loss: Optional[torch.Tensor]
    local_transformer_loss: Optional[torch.Tensor]
    local_transformer_logits: Optional[torch.Tensor]
    logits: torch.Tensor
    phoneme_logits: Optional[torch.Tensor]
    phoneme_tokens_target: Optional[torch.Tensor]
    phoneme_tokens_lens_target: Optional[torch.Tensor]
    audio_codes_target: torch.Tensor
    audio_codes_lens_target: torch.Tensor
    context_audio_codes: torch.Tensor
    context_audio_codes_lens: torch.Tensor
    selected_training_mode: Optional[str]


class EasyMagpieTTSModel(EasyMagpieTTSInferenceModel):
    """
    Magpie-TTS Model Decoder Only Model with training support.

    Subclasses EasyMagpieTTSInferenceModel to add training_step, validation_step,
    process_batch, data loading, and training-specific configuration (loss weights,
    phoneme corruption, eval models for validation metrics).
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)

        # Training-specific configuration
        self.dropout_text_input_prob = cfg.get('dropout_text_input_prob', 0.0)
        self.phoneme_corruption_batch_prob = cfg.get('phoneme_corruption_batch_prob', 0.0)
        self.phoneme_corruption_timestep_ratio = cfg.get('phoneme_corruption_timestep_ratio', 0.0)
        self.phoneme_corruption_unk_mode_prob = cfg.get('phoneme_corruption_unk_mode_prob', 0.5)
        self.phoneme_corruption_type = cfg.get('phoneme_corruption_type', 'repeat_skip_unk')
        self.phoneme_loss_weight = cfg.get('phoneme_loss_weight', 1.0)
        self.parallel_codebook_loss_scale = cfg.get('parallel_codebook_loss_scale', 1.0)
        self.local_transformer_loss_scale = cfg.get('local_transformer_loss_scale', 1.0)
        self.phoneme_as_text_prob = cfg.get('phoneme_as_text_prob', 0.0)

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        # Validation inference with metrics (optional)
        self.run_val_inference = cfg.get('run_val_inference', False)
        self.use_multilingual_asr = cfg.get('use_multilingual_asr', False)
        if self.run_val_inference:
            logging.info("Loading eval models for validation inference (ASR and speaker verification)...")
            if self.use_multilingual_asr:
                self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
                self.whisper_model.eval()
                for param in self.whisper_model.parameters():
                    param.requires_grad = False
                self._eval_asr_model = None
            else:
                self._eval_asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                    model_name="nvidia/parakeet-ctc-0.6b"
                )
                self._eval_asr_model.freeze()
                self.whisper_processor = None
                self.whisper_model = None
            self._eval_speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name='titanet_large'
            )
            self._eval_speaker_verification_model.freeze()
            logging.info("Eval models loaded successfully.")

        # UTMOSv2 naturalness scoring for validation (optional)
        self.use_utmos = cfg.get('use_utmos', False)
        if self.use_utmos:
            assert HAVE_UTMOSV2, (
                "UTMOSv2 is required for UTMOS scoring but is not installed. "
                "Install it with: pip install git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1"
            )
            self._utmos_calculator = UTMOSv2Calculator(device='cpu')
            logging.info("UTMOSv2 calculator initialized for validation naturalness scoring")

    def _get_state_dict_keys_to_exclude(self):
        return super()._get_state_dict_keys_to_exclude() + [
            '_speaker_verification_model',
            '_eval_asr_model',
            '_eval_speaker_verification_model',
            'whisper_model',
            'whisper_processor',
            '_utmos_calculator',
        ]


    def compute_loss(
        self,
        logits,
        audio_codes,
        audio_codes_lens,
        agent_mask_target=None,
    ):
        """
        Computes the audio codebook loss. Used by
        (1) The main Magpie-TTS transformer
        (2) The local transformer

        logits: (B, T', num_codebooks * num_tokens_per_codebook)
        audio_codes: (B, C, T')
        audio_codes_lens: (B,)
        """
        loss_mask = get_mask_from_lengths(audio_codes_lens)
        loss_mask = loss_mask.unsqueeze(1).repeat(1, audio_codes.size(1), 1)

        if agent_mask_target is not None:
            agent_mask_target = agent_mask_target.to(device=audio_codes.device, dtype=loss_mask.dtype)

        total_codebook_loss = None
        for codebook in range(audio_codes.size(1)):
            si = codebook * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = logits[:, :, si:ei]
            codebook_targets = audio_codes[:, codebook]
            raw_loss = self.cross_entropy_loss(
                codebook_logits.permute(0, 2, 1),
                codebook_targets.long(),
            )  # (B, T')
            effective_mask = loss_mask[:, codebook, :]
            if agent_mask_target is not None:
                effective_mask = effective_mask * agent_mask_target
            codebook_loss = raw_loss * effective_mask
            codebook_loss = codebook_loss.sum() / effective_mask.sum().clamp_min(1.0)
            total_codebook_loss = codebook_loss if total_codebook_loss is None else total_codebook_loss + codebook_loss

        total_codebook_loss = total_codebook_loss / audio_codes.size(1)
        return total_codebook_loss, loss_mask

    def compute_phoneme_loss(self, logits, phoneme_tokens, phoneme_tokens_lens):
        loss_mask = get_mask_from_lengths(phoneme_tokens_lens)
        total_phoneme_loss = None
        for codebook in range(self.phoneme_stacking_factor):
            si = codebook * self.phoneme_vocab_size
            ei = si + self.phoneme_vocab_size
            phoneme_logits = logits[:, :, si:ei]
            phoneme_targets = phoneme_tokens[:, codebook]
            phoneme_loss = self.cross_entropy_loss(phoneme_logits.permute(0, 2, 1), phoneme_targets)
            phoneme_loss = phoneme_loss * loss_mask
            phoneme_loss = phoneme_loss.sum() / loss_mask.sum()
            if total_phoneme_loss is None:
                total_phoneme_loss = phoneme_loss
            else:
                total_phoneme_loss = total_phoneme_loss + phoneme_loss
        total_phoneme_loss = total_phoneme_loss / self.phoneme_stacking_factor
        return total_phoneme_loss, loss_mask

    def log_val_audio_example(
        self,
        logits,
        target_audio_codes,
        audio_codes_lens_target,
        context_audio_codes=None,
        context_audio_codes_lens=None,
    ):
        wandb_audio_log = {}

        pred_audio_codes = self.logits_to_audio_codes(logits, audio_codes_lens_target)
        pred_audio_codes, _ = remove_eos_token(
            codes=pred_audio_codes,
            codes_len=audio_codes_lens_target,
        )
        pred_audio_codes, pred_audio_codes_lens = self._prepare_codes_for_decode(
            pred_audio_codes, audio_codes_lens_target - 1
        )
        pred_audio, pred_audio_lens, _ = self._codec_helper.codes_to_audio(
            pred_audio_codes,
            pred_audio_codes_lens,
        )
        target_audio_codes, _ = remove_eos_token(
            codes=target_audio_codes,
            codes_len=audio_codes_lens_target,
        )
        target_audio_codes, target_audio_codes_lens = self._prepare_codes_for_decode(
            target_audio_codes, audio_codes_lens_target - 1
        )
        target_audio, target_audio_lens, _ = self._codec_helper.codes_to_audio(
            target_audio_codes,
            target_audio_codes_lens,
        )

        context_audio, context_audio_lens = None, None
        if context_audio_codes is not None and context_audio_codes.shape[2] > 3:
            # > 3 ensures, it is a valid context audio tensor (and not dummy tensor used in text context)
            context_audio_codes, context_audio_codes_lens = remove_special_tokens(
                codes=context_audio_codes,
                codes_len=context_audio_codes_lens,
            )
            context_audio_codes, context_audio_codes_lens = self._prepare_codes_for_decode(
                context_audio_codes, context_audio_codes_lens
            )
            context_audio, context_audio_lens, _ = self._codec_helper.codes_to_audio(
                context_audio_codes,
                context_audio_codes_lens,
            )

        for logger in self.loggers:
            is_wandb = isinstance(logger, WandbLogger)
            is_tb = isinstance(logger, TensorBoardLogger)
            if not is_wandb and not is_tb:
                raise ValueError(
                    f"Invalid logger type for audio logging: {type(logger)}. Only `WandbLogger` and `TensorBoardLogger` are supported."
                )

            for idx in range(min(3, pred_audio.size(0))):
                pred_audio_np = pred_audio[idx].float().detach().cpu().numpy()
                target_audio_np = target_audio[idx].float().detach().cpu().numpy()
                pred_audio_np = pred_audio_np[: pred_audio_lens[idx]]
                target_audio_np = target_audio_np[: target_audio_lens[idx]]
                context_audio_np = None
                if context_audio is not None:
                    context_audio_np = context_audio[idx].float().detach().cpu().numpy()
                    context_audio_np = context_audio_np[: context_audio_lens[idx]]

                if is_wandb:
                    wandb_audio_log[f"Audio/Example_{idx}"] = list()
                    if context_audio_np is not None:
                        wandb_audio_log[f"Audio/Example_{idx}"].append(
                            wandb.Audio(context_audio_np, sample_rate=self.output_sample_rate, caption="context")
                        )
                    wandb_audio_log[f"Audio/Example_{idx}"].append(
                        wandb.Audio(pred_audio_np, sample_rate=self.output_sample_rate, caption="prediction")
                    )
                    wandb_audio_log[f"Audio/Example_{idx}"].append(
                        wandb.Audio(target_audio_np, sample_rate=self.output_sample_rate, caption="target")
                    )

                if is_tb:
                    if context_audio_np is not None:
                        logger.experiment.add_audio(
                            f'Example_{idx}/context',
                            context_audio_np,
                            global_step=self.global_step,
                            sample_rate=self.output_sample_rate,
                        )
                    logger.experiment.add_audio(
                        f'Example_{idx}/prediction',
                        pred_audio_np,
                        global_step=self.global_step,
                        sample_rate=self.output_sample_rate,
                    )
                    logger.experiment.add_audio(
                        f'Example_{idx}/target',
                        target_audio_np,
                        global_step=self.global_step,
                        sample_rate=self.output_sample_rate,
                    )

        return wandb_audio_log

    def prepare_text_channel_embeddings(
        self,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        delay: torch.Tensor,
        dropout_text_input: bool = False,
        is_multiturn: bool = False,
        text_pad_id: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare text embeddings as a channel input with delay handling.

        This function embeds text tokens and prepends zero-padding based on the delay
        parameter. The delay represents the number of zero positions to prepend before
        the text embeddings, aligning the text channel with other channels.

        Args:
            text: Input text token IDs (B, L)
            text_lens: Length of text for each batch item (B,)
            delay: Number of zero positions to prepend for each batch item (B,).
                   For text channel, this is typically just context_lens.
            dropout_text_input: If True, return all zeros (for text dropout regularization).

        Returns:
            Tuple of:
                - text_channel_embedding: Text embeddings with zero-padded delay (B, T_delay + T_text, E)
                - text_channel_lens: Total length of text channel for each batch item (B,)
        """
        batch_size = text.size(0)
        device = text.device

        # Embed text tokens (CAS-only when disable_subword_embedding=True).
        text_embedded = self.embed_text_tokens(text, text_lens=text_lens, is_multiturn=is_multiturn)  # (B, L, E)

        # Handle text dropout - zero out the embeddings
        if dropout_text_input:
            text_embedded = text_embedded * 0.0

        # multiturn dataset returns a special pad text tokens until it matches the audio len, to keep compatible with regular dataset zero-out those values
        if is_multiturn:
            text_embedded[text == text_pad_id] = 0.0

        # Create zero tensor for delay padding
        max_delay = delay.max().item()
        zero_delay_tensor = torch.zeros(batch_size, max_delay, self.cfg.embedding_dim, device=device)

        # Join delay zeros with text embeddings
        text_channel_embedding, text_channel_lens = self.join_embeddings_temporally(
            embeddings=[zero_delay_tensor, text_embedded],
            lengths=[delay, text_lens],
        )

        return text_channel_embedding, text_channel_lens

    def prepare_phoneme_channel_embeddings(
        self,
        phoneme_tokens: torch.Tensor,
        phoneme_tokens_lens: torch.Tensor,
        delay: torch.Tensor,
        apply_corruption: bool = False,
        dropout_complete_phoneme_channel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[str]]:
        """
        Prepare phoneme embeddings as a channel input with delay handling.

        This function stacks phoneme tokens (if configured), embeds them, and prepends
        zero-padding based on the delay parameter. The delay represents the number of
        zero positions to prepend before the phoneme embeddings.

        Args:
            phoneme_tokens: Phoneme token IDs (B, L)
            phoneme_tokens_lens: Length of phoneme tokens for each batch item (B,)
            delay: Number of zero positions to prepend for each batch item (B,).
                   This is typically context_lens + phoneme_delay.
            apply_corruption: If True, apply phoneme-token corruption before embedding.
            dropout_complete_phoneme_channel: If True, zero-out the whole phoneme channel embedding.

        Returns:
            Tuple of:
                - phoneme_channel_embedding: Phoneme embeddings with zero-padded delay (B, T_delay + T_phoneme, E)
                - phoneme_channel_lens: Total length of phoneme channel for each batch item (B,)
                - phoneme_tokens_stacked: Stacked phoneme tokens (B, S, T')
                - phoneme_tokens_lens_stacked: Length of stacked phoneme tokens (B,)
                - phoneme_tokens_stacked_clean: Clean stacked phoneme tokens before corruption (B, S, T')
                - corruption_mode: None, "unk", or "repeat_skip"
        """
        batch_size = phoneme_tokens.size(0)
        device = phoneme_tokens.device

        # Stack phoneme tokens
        phoneme_tokens_expanded = phoneme_tokens.unsqueeze(1)  # (B, 1, L)
        phoneme_tokens_stacked, phoneme_tokens_lens_stacked = self.stack_codes(
            phoneme_tokens_expanded,
            phoneme_tokens_lens,
            self.phoneme_tokenizer.bos_token_id,
            self.phoneme_tokenizer.eos_token_id,
            self.phoneme_stacking_factor,
            1,
        )
        phoneme_tokens_stacked_clean = phoneme_tokens_stacked.clone()

        phoneme_corruption_mode = None
        if apply_corruption:
            phoneme_tokens_stacked, phoneme_corruption_mode = self.corrupt_stacked_phoneme_tokens(
                phoneme_tokens_stacked=phoneme_tokens_stacked,
                phoneme_tokens_lens_stacked=phoneme_tokens_lens_stacked,
            )

        # Embed phoneme tokens
        phoneme_embedded = self.embed_phoneme_tokens(phoneme_tokens_stacked)  # (B, T', E)

        # Apply mask to zero out padding
        if self.cfg.get("use_multiturn_dataset", False):
            phoneme_pad_id = getattr(self.phoneme_tokenizer, "pad", -1)
            phoneme_mask = (phoneme_tokens_stacked[:, 0, :] != phoneme_pad_id) # Check the first layer of the stack
            # Apply mask to zero out padding
            phoneme_embedded = phoneme_embedded * phoneme_mask.unsqueeze(2)  # (B, T', E)
        else:
            phoneme_mask = get_mask_from_lengths(phoneme_tokens_lens_stacked)
            phoneme_embedded = phoneme_embedded * phoneme_mask.unsqueeze(2)  # (B, T', E)

        # Handle phoneme dropout - zero out the embeddings
        if dropout_complete_phoneme_channel:
            phoneme_embedded = phoneme_embedded * 0.0

        # Create zero tensor for delay padding
        max_delay = delay.max().item()
        zero_delay_tensor = torch.zeros(batch_size, max_delay, self.cfg.embedding_dim, device=device)

        # Join delay zeros with phoneme embeddings
        phoneme_channel_embedding, phoneme_channel_lens = self.join_embeddings_temporally(
            embeddings=[zero_delay_tensor, phoneme_embedded],
            lengths=[delay, phoneme_tokens_lens_stacked],
        )

        return (
            phoneme_channel_embedding,
            phoneme_channel_lens,
            phoneme_tokens_stacked,
            phoneme_tokens_lens_stacked,
            phoneme_tokens_stacked_clean,
            phoneme_corruption_mode,
        )

    def corrupt_stacked_phoneme_tokens(
        self,
        phoneme_tokens_stacked: torch.Tensor,
        phoneme_tokens_lens_stacked: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[str]]:
        """
        Corrupt stacked phoneme tokens for robustness to phoneme prediction errors.

        Two corruption modes are supported:
        1. UNK replacement at selected timesteps (all stacked channels replaced).
        2. Repeat/skip corruption via a shared index remapping over the valid prefix.
        """
        if self.phoneme_tokenizer is None:
            return phoneme_tokens_stacked, None
        if self.phoneme_corruption_batch_prob <= 0.0:
            return phoneme_tokens_stacked, None
        if self.phoneme_corruption_timestep_ratio <= 0.0:
            return phoneme_tokens_stacked, None
        if torch.rand(1).item() >= self.phoneme_corruption_batch_prob:
            return phoneme_tokens_stacked, None

        min_len = int(phoneme_tokens_lens_stacked.min().item())
        # Need room for BOS and EOS plus at least one interior timestep.
        if min_len <= 2:
            return phoneme_tokens_stacked, None

        # Corrupt only interior steps, keeping BOS/EOS untouched.
        valid_start = 1
        valid_end = min_len - 1  # exclusive
        num_valid_steps = max(0, valid_end - valid_start)
        if num_valid_steps == 0:
            return phoneme_tokens_stacked, None

        num_corrupt_steps = int(round(num_valid_steps * self.phoneme_corruption_timestep_ratio))
        num_corrupt_steps = max(1, min(num_valid_steps, num_corrupt_steps))

        corrupted = phoneme_tokens_stacked.clone()
        mode = 'unk' if torch.rand(1).item() < self.phoneme_corruption_unk_mode_prob else 'repeat_skip'

        candidate_steps = torch.arange(valid_start, valid_end, device=phoneme_tokens_stacked.device)
        corrupt_steps = candidate_steps[torch.randperm(num_valid_steps, device=phoneme_tokens_stacked.device)][
            :num_corrupt_steps
        ]

        if mode == 'unk':
            if not hasattr(self.phoneme_tokenizer, 'unk_token_id'):
                raise ValueError("Phoneme tokenizer is missing `unk_token_id` required for UNK corruption.")
            corrupted[:, :, corrupt_steps] = self.phoneme_tokenizer.unk_token_id
            return corrupted, mode

        # Repeat/skip corruption with a shared remap over [0, min_len).
        # This keeps batched execution efficient and applies the same corrupted timeline across the batch.
        step_delta = torch.ones(min_len, device=phoneme_tokens_stacked.device, dtype=torch.long)
        op_is_repeat = torch.rand(corrupt_steps.numel(), device=phoneme_tokens_stacked.device) < 0.5
        step_delta[corrupt_steps] = torch.where(
            op_is_repeat, torch.zeros_like(corrupt_steps), torch.full_like(corrupt_steps, 2)
        )
        source_index = torch.cumsum(step_delta, dim=0) - step_delta[0]
        source_index = torch.clamp(source_index, min=0, max=min_len - 1)
        source_index[0] = 0
        source_index[-1] = min_len - 1

        corrupted_prefix = phoneme_tokens_stacked[:, :, :min_len].index_select(dim=2, index=source_index)
        corrupted[:, :, :min_len] = corrupted_prefix
        return corrupted, mode

    def prepare_audio_channel_embeddings(
        self,
        audio_codes: torch.Tensor,
        audio_codes_lens: torch.Tensor,
        delay: torch.Tensor,
        speech_eos_mask: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None,
        current_streaming_speech_delay: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare audio embeddings as a channel input with delay handling.

        This function processes audio codes by adding special tokens, stacking them,
        and embedding them. It prepends zero-padding based on the delay parameter.
        Also prepares input/target split for autoregressive training.

        Args:
            audio_codes: Audio codes (B, C, T) - raw codes without special tokens
            audio_codes_lens: Length of audio codes for each batch item (B,)
            delay: Number of zero positions to prepend for each batch item (B,).
                   In full mode: context_lens + text_lens + speech_delay
                   In streaming mode: context_lens + speech_delay

        Returns:
            Tuple of:
                - audio_channel_embedding: Audio embeddings with zero-padded delay (B, T_delay + T_audio, E)
                - audio_channel_lens: Total length of audio channel for each batch item (B,)
                - audio_codes_target: Target audio codes for loss computation (B, C, T'-1)
                - audio_codes_lens_target: Length of target audio codes (B,)
                - loss_agent_mask: Optional mask used for loss masking; None when no agent_mask is provided.
        """
        batch_size = audio_codes.size(0)
        device = audio_codes.device

        # Apply codec conversion if configured
        if self._codec_converter is not None:
            audio_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=audio_codes, audio_lens=audio_codes_lens
            ).long()

        # Add BOS and EOS tokens
        audio_codes, audio_codes_lens = add_special_tokens(
            codes=audio_codes,
            codes_len=audio_codes_lens,
            bos_id=self.audio_bos_id,
            eos_id=self.audio_eos_id,
            num_eos_tokens=1 if speech_eos_mask is None else 0,
        )

        # Stack audio codes across codebooks
        audio_codes, audio_codes_lens = self.stack_codes(
            audio_codes,
            audio_codes_lens,
            self.audio_bos_id,
            self.audio_eos_id,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )

        if speech_eos_mask is not None:
            audio_codes_before_speech_eos = audio_codes.clone()
            # Shift +1 for BOS alignment and +1 more so EOS is injected after the marked frame.
            B_mask, T_mask = speech_eos_mask.shape
            shifted_mask = torch.zeros((B_mask, T_mask + 2), dtype=torch.bool, device=device)
            shifted_mask[:, 2:] = speech_eos_mask

            # 2. Find the minimum overlapping time dimension
            t_mask = shifted_mask.size(1)
            t_audio = audio_codes.size(2)
            min_t = min(t_mask, t_audio)

            # 3. Slice both to the valid overlap and broadcast the C dimension
            valid_mask = shifted_mask[:, :min_t]
            expanded_mask = valid_mask.unsqueeze(1).expand(-1, audio_codes.size(1), -1)

            # Inject the EOS token only into the overlapping region
            audio_codes[:, :, :min_t][expanded_mask] = self.audio_eos_id

        # Prepare input and target for autoregressive training
        # Input: all tokens except the last (teacher forcing)
        # Target: all tokens except the first (shifted by one)
        audio_codes_lens_target = audio_codes_lens - 1
        audio_codes_target = audio_codes[:, :, 1:]  # (B, C, T'-1)
        audio_codes_input = audio_codes[:, :, :-1]  # (B, C, T'-1)

        # Drop some EOS frames from audio input so the model learns recovery when inference misses EOS.
        # sample_prob keeps some samples untouched, so the model still learns the normal EOS-input behavior.
        if speech_eos_mask is not None and self.training:
            drop_eos_sample_prob = float(self.cfg.get("drop_eos_from_audio_input_sample_prob", 0.0))
            drop_eos_frame_prob = float(self.cfg.get("drop_eos_from_audio_input_frame_prob", 0.5))

            if drop_eos_sample_prob > 0.0 and drop_eos_frame_prob > 0.0:
                eos_frame_mask = (audio_codes_input == self.audio_eos_id).any(dim=1)  # [B, T]

                sample_drop_mask = torch.rand(batch_size, device=device) < drop_eos_sample_prob  # [B]

                frame_drop_mask = (
                    eos_frame_mask
                    & sample_drop_mask.unsqueeze(1)
                    & (torch.rand_like(eos_frame_mask.float()) < drop_eos_frame_prob)
                )  # [B, T]

                audio_codes_input_backup = audio_codes_before_speech_eos[:, :, :-1]

                audio_codes_input = torch.where(
                    frame_drop_mask.unsqueeze(1),
                    audio_codes_input_backup,
                    audio_codes_input,
                )

        # deal with agent mask
        loss_agent_mask = None
        if agent_mask is not None:
            target_T = audio_codes_target.size(2)

            # Align dataloader agent_mask to audio_codes_target time.
            if agent_mask.size(1) < target_T:
                pad = torch.zeros(
                    agent_mask.size(0),
                    target_T - agent_mask.size(1),
                    device=agent_mask.device,
                    dtype=torch.bool,
                )
                agent_mask = torch.cat([agent_mask.bool(), pad], dim=1)
            else:
                agent_mask = agent_mask[:, :target_T].bool()

            agent_mask = agent_mask.to(audio_codes_target.device)

            valid = get_mask_from_lengths(audio_codes_lens_target).bool().to(audio_codes_target.device)
            agent_mask = agent_mask & valid

            # Keep EOS and the frame before EOS supervised.
            eos_any = (audio_codes_target == self.audio_eos_id).any(dim=1) & valid

            eos_prev1 = torch.zeros_like(eos_any)
            eos_prev1[:, :-1] = eos_any[:, 1:]

            agent_mask = agent_mask | eos_prev1 | eos_any
            target_agent_mask = agent_mask & valid
            loss_agent_mask = target_agent_mask

            if self.cfg.get("debug_decode_agent_mask", False) and self.training and self.global_step < 5:
                self.debug_decode_mask_regions(
                    audio_codes_target=audio_codes_target,
                    audio_codes_lens_target=audio_codes_lens_target,
                    agent_mask=agent_mask,
                    out_dir=os.path.join(self.trainer.log_dir, "mask_debug", f"step_{self.global_step}"),
                    prefix=f"batch_{self.global_rank}_{self.global_step}",
                )

            # Replace user/non-agent regions with a learned token.
            # Important: audio_codes_input predicts audio_codes_target, so input mask must be shifted.
            if self.cfg.get("use_user_speaking_token", False):
                target_non_agent = (~target_agent_mask) & valid

                # audio_codes_input[:, :, t] is the previous token used to predict target t.
                input_agent_mask = torch.zeros_like(target_agent_mask)
                input_agent_mask[:, 1:] = target_agent_mask[:, :-1]
                input_agent_mask[:, 0] = True  # Keep first/BOS input untouched.

                input_valid = torch.zeros_like(valid)
                input_valid[:, 1:] = valid[:, :-1]
                input_valid[:, 0] = valid[:, 0]

                input_non_agent = (~input_agent_mask) & input_valid

                user_tok_input = torch.full_like(audio_codes_input, self.audio_user_speaking_id)
                audio_codes_input = torch.where(
                    input_non_agent.unsqueeze(1),
                    user_tok_input,
                    audio_codes_input,
                )

                user_tok_target = torch.full_like(audio_codes_target, self.audio_user_speaking_id)
                audio_codes_target = torch.where(
                    target_non_agent.unsqueeze(1),
                    user_tok_target,
                    audio_codes_target,
                )

            # Put audio_user_speaking_end_id in the input slot that predicts the first agent frame
            # after a non-agent region.
            if self.cfg.get("use_user_speaking_end_token", False):
                user_to_agent = torch.zeros_like(target_agent_mask)

                user_to_agent[:, 1:] = (
                    target_agent_mask[:, 1:]
                    & (~target_agent_mask[:, :-1])
                    & valid[:, 1:]
                    & valid[:, :-1]
                )

                end_tok_input = torch.full_like(audio_codes_input, self.audio_user_speaking_end_id)

                audio_codes_input = torch.where(
                    user_to_agent.unsqueeze(1),
                    end_tok_input,
                    audio_codes_input,
                )

            # Note that consider the current_streaming_speech_delay tokens/user speaking tokens on the loss,
            # allowing to predict them in autoregressive way
            transition_prefix = int(current_streaming_speech_delay or 0)
            if self.cfg.get("agent_mask_include_transition_prefix", False) and transition_prefix > 0:
                agent_i = target_agent_mask.float().unsqueeze(1)

                agent_i = torch.nn.functional.pad(agent_i, (0, transition_prefix))
                loss_agent_mask = (
                    torch.nn.functional.max_pool1d(
                        agent_i,
                        kernel_size=transition_prefix + 1,
                        stride=1,
                    )
                    .squeeze(1)
                    .bool()
                    & valid
                )

        # Embed audio tokens
        audio_embedded = self.embed_audio_tokens(audio_codes_input)  # (B, T'-1, E)

        # Create zero tensor for delay padding
        max_delay = delay.max().item()
        zero_delay_tensor = torch.zeros(batch_size, max_delay, self.cfg.embedding_dim, device=device)

        # Join delay zeros with audio embeddings
        audio_channel_embedding, audio_channel_lens = self.join_embeddings_temporally(
            embeddings=[zero_delay_tensor, audio_embedded],
            lengths=[delay, audio_codes_lens_target],
        )

        return audio_channel_embedding, audio_channel_lens, audio_codes_target, audio_codes_lens_target, loss_agent_mask

    def slice_sequence_embeddings(self, sequence_embeddings, context_lens, target_lens):
        """
        Slices sequence embeddings to get the predicted embeddings for the target sequence.
        Args:
            sequence_embeddings: (B, T, E)
            context_lens: (B,) - start index of target per batch
            target_lens: (B,) - length of target per batch

        Returns: (B, T_max, E) tensor where T_max = max(target_lens)
        """
        B, T, E = sequence_embeddings.shape
        device = sequence_embeddings.device

        # Compute max target length in batch for padding
        max_len = target_lens.max().item()

        # Build index tensor for each batch element
        # Shape: (B, max_len)
        range_indices = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        gather_indices = context_lens.unsqueeze(1) + range_indices  # (B, max_len)
        gather_indices = torch.clamp(gather_indices, max=sequence_embeddings.size(1) - 1)

        # Expand to shape (B, max_len, E) for gather
        gather_indices_exp = gather_indices.unsqueeze(2).expand(-1, -1, E)
        sliced = torch.gather(sequence_embeddings, dim=1, index=gather_indices_exp)
        return sliced

    def process_batch(
        self,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        context_text_tokens: torch.Tensor,
        context_text_tokens_lens: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_lens: torch.Tensor,
        context_audio_codes: torch.Tensor,
        context_audio_codes_lens: torch.Tensor,
        phoneme_tokens: Optional[torch.Tensor] = None,
        phoneme_tokens_lens: Optional[torch.Tensor] = None,
        mode: str = "train",
        training_mode: Optional[TrainingMode] = None,
        task: Optional[List[str]] = None,
        agent_mask: Optional[torch.Tensor] = None,
        user_audio_embedded: Optional[torch.Tensor] = None
    ) -> ProcessBatchOutput:
        """
        Simplified batch processing using channel-based embedding architecture.

        This function provides a cleaner implementation of process_batch where:
        1. Context is prepared separately (without text)
        2. Text, phoneme, and audio are each treated as channels with delay-based alignment
        3. Channels are summed element-wise and joined temporally with context

        The delay handling ensures proper temporal alignment:
        - Text channel delay: context_lens (no additional delay)
        - Phoneme channel delay: context_lens + phoneme_delay
        - Audio channel delay: context_lens + text_lens + speech_delay (full mode)
                              or context_lens + speech_delay (streaming mode)

        Args:
            text: Input text token IDs (B, L)
            text_lens: Length of text for each batch item (B,)
            context_text_tokens: Context text token IDs for conditioning (B, L_ctx)
            context_text_tokens_lens: Length of context text (B,)
            audio_codes: Audio codes (B, C, T) - raw codes without special tokens
            audio_codes_lens: Length of audio codes (B,)
            context_audio_codes: Pre-computed context audio codes (B, C, T')
            context_audio_codes_lens: Length of context audio codes (B,)
            phoneme_tokens: Phoneme token IDs (optional) (B, L_phoneme)
            phoneme_tokens_lens: Length of phoneme tokens (B,)
            mode: Training mode, either "train" or "val"
            training_mode: Optional TrainingMode object

        Returns:
            ProcessBatchOutput: Contains loss values and model predictions
        """
        # Select training mode
        selected_training_mode = training_mode
        if selected_training_mode is None:
            if mode == 'train':
                selected_training_mode = random.choice(self.training_modes)
            else:
                selected_training_mode = self.training_modes[0]

        current_text_input_mode = selected_training_mode.text_input_mode
        current_streaming_speech_delay = selected_training_mode.streaming_speech_delay
        current_streaming_phonemes_delay = selected_training_mode.streaming_phonemes_delay

        # Determine dropout flags
        dropout_text_input = (random.random() < self.dropout_text_input_prob) if mode == 'train' else False

        # Determine CFG unconditional dropout
        dropout_conditional_input = False
        if mode == 'train' and self.cfg_unconditional_prob > 0.0:
            if torch.rand(1).item() < self.cfg_unconditional_prob:
                dropout_conditional_input = True

        # 1. Prepare context tensors (without text)
        context_embedding, context_lens, context_audio_codes_processed, context_audio_codes_lens_processed = (
            self.prepare_context_tensors(
                context_text_tokens=context_text_tokens,
                context_text_tokens_lens=context_text_tokens_lens,
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                training_mode=selected_training_mode,
                dropout_conditional_input=dropout_conditional_input,
            )
        )

        # 2. Compute delays for each channel based on mode
        # Text channel delay: always context_lens
        text_delay = context_lens.clone()

        # Phoneme channel delay: context_lens + phoneme_delay (both modes)
        phoneme_delay = context_lens + current_streaming_phonemes_delay

        # Audio channel delay depends on mode
        if current_text_input_mode == 'full':
            # Full mode: context_lens + text_lens + speech_delay
            audio_delay = context_lens + text_lens + current_streaming_speech_delay
        else:
            # Streaming mode: context_lens + speech_delay
            audio_delay = context_lens + current_streaming_speech_delay

        speech_eos_mask = None
        if self.cfg.get("use_multiturn_dataset", False):
            speech_eos_mask = (text == self.interruption_token_id)  # (B, T)
            # remove the interruption token for all task, expect for interruption
            if not task or "interruption" not in str(task[0]):
                text[speech_eos_mask] = self.tokenizer.pad  # Clean up the text channel
            # else: # ToDo: move self.interruption_token_id forward by  audio_delay so that soon it saw the interruption token it is forced to stop instead of await audio_delay tokens

        # 3. Prepare text channel embeddings
        text_channel_embedding, text_channel_lens = self.prepare_text_channel_embeddings(
            text=text,
            text_lens=text_lens,
            delay=text_delay,
            dropout_text_input=dropout_text_input or dropout_conditional_input,
            is_multiturn=self.cfg.get("use_multiturn_dataset", False),
            text_pad_id=self.pad_id,
        )

        # 4. Prepare phoneme channel embeddings (if phoneme tokenizer is configured)
        phoneme_channel_embedding = None
        phoneme_tokens_stacked = None
        phoneme_tokens_lens_stacked = None
        phoneme_tokens_stacked_clean = None
        phoneme_corruption_mode = None
        dropout_complete_phoneme_channel = False
        if self.phoneme_tokenizer is not None and phoneme_tokens is not None:
            # Corrupt phonemes only when text input is not dropped.
            apply_phoneme_corruption = (
                mode == 'train'
                and (not dropout_text_input)
                and (not dropout_conditional_input)
                and self.phoneme_corruption_type == 'repeat_skip_unk'
            )
            dropout_complete_phoneme_channel = mode == 'train' and (
                dropout_conditional_input
                or (
                    self.phoneme_corruption_type == 'complete_channel'
                    and torch.rand(1).item() < self.phoneme_corruption_batch_prob
                )
            )
            (
                phoneme_channel_embedding,
                phoneme_channel_lens,
                phoneme_tokens_stacked,
                phoneme_tokens_lens_stacked,
                phoneme_tokens_stacked_clean,
                phoneme_corruption_mode,
            ) = self.prepare_phoneme_channel_embeddings(
                phoneme_tokens=phoneme_tokens,
                phoneme_tokens_lens=phoneme_tokens_lens,
                delay=phoneme_delay,
                apply_corruption=apply_phoneme_corruption,
                dropout_complete_phoneme_channel=dropout_complete_phoneme_channel,
            )

        # 5. Prepare audio channel embeddings
        (
            audio_channel_embedding,
            audio_channel_lens,
            audio_codes_target,
            audio_codes_lens_target,
            agent_mask,
        ) = self.prepare_audio_channel_embeddings(
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
            delay=audio_delay,
            speech_eos_mask=speech_eos_mask,
            agent_mask=agent_mask,
            current_streaming_speech_delay=current_streaming_speech_delay,
        )

        # 6. Sum the channel embeddings element-wise
        # First, align all channels to the same length (max of all channel lengths)
        max_channel_len = max(
            text_channel_embedding.size(1),
            audio_channel_embedding.size(1),
            phoneme_channel_embedding.size(1) if phoneme_channel_embedding is not None else 0,
        )

        # Pad text channel if needed
        if text_channel_embedding.size(1) < max_channel_len:
            padding = torch.zeros(
                text_channel_embedding.size(0),
                max_channel_len - text_channel_embedding.size(1),
                text_channel_embedding.size(2),
                device=text_channel_embedding.device,
            )
            text_channel_embedding = torch.cat([text_channel_embedding, padding], dim=1)

        # Pad audio channel if needed
        if audio_channel_embedding.size(1) < max_channel_len:
            padding = torch.zeros(
                audio_channel_embedding.size(0),
                max_channel_len - audio_channel_embedding.size(1),
                audio_channel_embedding.size(2),
                device=audio_channel_embedding.device,
            )
            audio_channel_embedding = torch.cat([audio_channel_embedding, padding], dim=1)

        # Sum channels
        combined_channel_embedding = text_channel_embedding + audio_channel_embedding

        # Add phoneme channel if available
        if phoneme_channel_embedding is not None:
            if phoneme_channel_embedding.size(1) < max_channel_len:
                padding = torch.zeros(
                    phoneme_channel_embedding.size(0),
                    max_channel_len - phoneme_channel_embedding.size(1),
                    phoneme_channel_embedding.size(2),
                    device=phoneme_channel_embedding.device,
                )
                phoneme_channel_embedding = torch.cat([phoneme_channel_embedding, padding], dim=1)
            combined_channel_embedding = combined_channel_embedding + phoneme_channel_embedding

        if user_audio_embedded is not None:
            bos_user_pad = torch.zeros(
                user_audio_embedded.size(0),
                1,
                user_audio_embedded.size(2),
                device=user_audio_embedded.device,
                dtype=user_audio_embedded.dtype,
            )
            user_audio_embedded = torch.cat([bos_user_pad, user_audio_embedded], dim=1)

            # Align user conditioning to audio_codes_target timeline,
            # same as agent_mask from prepare_audio_channel_embeddings().
            target_T = audio_codes_target.size(2)

            if user_audio_embedded.size(1) < target_T:
                pad_len = target_T - user_audio_embedded.size(1)
                user_audio_embedded = torch.cat(
                    [
                        user_audio_embedded,
                        torch.zeros(
                            user_audio_embedded.size(0),
                            pad_len,
                            user_audio_embedded.size(2),
                            device=user_audio_embedded.device,
                            dtype=user_audio_embedded.dtype,
                        ),
                    ],
                    dim=1,
                )
            else:
                user_audio_embedded = user_audio_embedded[:, :target_T]

            batch_size = user_audio_embedded.size(0)
            device = user_audio_embedded.device

            max_delay = audio_delay.max().item()
            zero_delay_tensor = torch.zeros(
                batch_size,
                max_delay,
                self.cfg.embedding_dim,
                device=device,
                dtype=user_audio_embedded.dtype,
            )

            user_audio_lens = audio_codes_lens_target.to(audio_delay.device)

            user_audio_channel_embedding, _ = self.join_embeddings_temporally(
                embeddings=[zero_delay_tensor, user_audio_embedded],
                lengths=[audio_delay, user_audio_lens],
            )

            if user_audio_channel_embedding.size(1) < max_channel_len:
                pad_len = max_channel_len - user_audio_channel_embedding.size(1)
                user_audio_channel_embedding = torch.cat(
                    [
                        user_audio_channel_embedding,
                        torch.zeros(
                            batch_size,
                            pad_len,
                            user_audio_channel_embedding.size(2),
                            device=user_audio_channel_embedding.device,
                            dtype=user_audio_channel_embedding.dtype,
                        ),
                    ],
                    dim=1,
                )
            else:
                user_audio_channel_embedding = user_audio_channel_embedding[:, :max_channel_len]

            combined_channel_embedding = combined_channel_embedding + user_audio_channel_embedding

        # 7. Join context with combined channel embeddings
        # The combined_channel_lens is the max of all channel lens for each batch item
        combined_channel_lens = (
            torch.stack(
                [
                    text_channel_lens,
                    audio_channel_lens,
                    phoneme_channel_lens if phoneme_channel_embedding is not None else audio_channel_lens,
                ],
                dim=0,
            )
            .max(dim=0)
            .values
        )

        # Right pad context embedding
        context_padding = torch.zeros(
            context_embedding.size(0),
            combined_channel_embedding.size(1) - context_embedding.size(1),
            context_embedding.size(2),
            device=context_embedding.device,
        )
        context_embedding_padded = torch.cat([context_embedding, context_padding], dim=1)

        full_embedding = context_embedding_padded + combined_channel_embedding

        # 8. Forward pass through transformer
        transformer_out = self.forward(
            inputs_embeds=full_embedding,
            attention_mask=get_mask_from_lengths(combined_channel_lens),
        )
        transformer_hidden_states = transformer_out.last_hidden_state  # (B, T_total, E)

        # 9. Extract prediction embeddings and compute losses
        # Audio predictions start at audio_delay
        pred_embeddings = self.slice_sequence_embeddings(
            transformer_hidden_states,
            context_lens=audio_delay,
            target_lens=audio_codes_lens_target,
        )

        # Project to audio logits
        pred_embeddings_audio = self.audio_out_projection(pred_embeddings)
        logits = self.final_proj(pred_embeddings_audio)

        # Compute codebook loss
        codebook_loss, _ = self.compute_loss(logits, audio_codes_target, audio_codes_lens_target, agent_mask_target=agent_mask if self.cfg.get("mask_user_on_loss", False) else None)
        loss = self.parallel_codebook_loss_scale * codebook_loss

        # Compute local transformer loss if applicable
        local_transformer_loss = None
        local_transformer_logits = None
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            assert self.local_transformer_type == LocalTransformerType.AR, "Unexpected local transformer type"
            local_transformer_logits = self._lt_helper.compute_logits(
                pred_embeddings, audio_codes_target, targets_offset_by_one=False
            )
            local_transformer_loss, _ = self.compute_loss(
                local_transformer_logits, audio_codes_target, audio_codes_lens_target, agent_mask_target=agent_mask if self.cfg.get("mask_user_on_loss", False) else None
            )

            loss = loss + self.local_transformer_loss_scale * local_transformer_loss

        # Compute phoneme loss if applicable
        phoneme_loss = None
        pb_phoneme_logits = None
        pb_phoneme_tokens_target = None
        pb_phoneme_tokens_lens_target = None
        if self.phoneme_tokenizer is not None and phoneme_tokens_stacked is not None:
            # Phoneme predictions start at phoneme_delay
            pred_embeddings_phoneme = self.slice_sequence_embeddings(
                transformer_hidden_states,
                context_lens=phoneme_delay,
                target_lens=phoneme_tokens_lens_stacked - 1,
            )
            pb_phoneme_logits = self.phoneme_final_proj(pred_embeddings_phoneme)
            pb_phoneme_tokens_target = phoneme_tokens_stacked_clean[:, :, 1:].long()
            pb_phoneme_tokens_lens_target = phoneme_tokens_lens_stacked - 1

            if (phoneme_corruption_mode != 'repeat_skip') and not (
                dropout_complete_phoneme_channel or dropout_conditional_input or dropout_text_input
            ):
                phoneme_loss, _ = self.compute_phoneme_loss(
                    pb_phoneme_logits, pb_phoneme_tokens_target, pb_phoneme_tokens_lens_target
                )
            else:
                phoneme_loss = torch.tensor(0.0, device=logits.device)

            loss = loss + self.phoneme_loss_weight * phoneme_loss

        return ProcessBatchOutput(
            loss=loss,
            codebook_loss=codebook_loss,
            phoneme_loss=phoneme_loss,
            local_transformer_loss=local_transformer_loss,
            local_transformer_logits=local_transformer_logits,
            logits=logits,
            phoneme_logits=pb_phoneme_logits,
            phoneme_tokens_target=pb_phoneme_tokens_target,
            phoneme_tokens_lens_target=pb_phoneme_tokens_lens_target,
            audio_codes_target=audio_codes_target,
            audio_codes_lens_target=audio_codes_lens_target,
            context_audio_codes=context_audio_codes_processed,
            context_audio_codes_lens=context_audio_codes_lens_processed,
            selected_training_mode=selected_training_mode.name if selected_training_mode is not None else None,
        )

    def training_step(self, batch, batch_idx):
        if 'context_audio_codes' in batch:
            context_audio_codes = batch['context_audio_codes']
            context_audio_codes_lens = batch['context_audio_codes_lens']
        else:
            context_audio = batch['context_audio']
            context_audio_lens = batch['context_audio_lens']
            context_audio_codes, context_audio_codes_lens = self._codec_helper.audio_to_codes(
                context_audio, context_audio_lens
            )

        if 'audio_codes' in batch:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']
        else:
            audio = batch['audio']
            audio_lens = batch['audio_lens']
            audio_codes, audio_codes_lens = self._codec_helper.audio_to_codes(audio, audio_lens)

        # augment tts data to looks more like multiturn data by adding pad on the begining and emulating user speaking.
        if self.cfg.get("use_multiturn_dataset", False) and "tts" in batch['task']:
            prob = self.cfg.get("add_tts_sil_begining_prob", 0.0)
            if prob > 0 and torch.rand(1).item() < prob:
                audio_codes_lens_max = audio_codes_lens.max()
                
                # 1. Calculate the raw shift (with the -1 safety buffer)
                raw_pad_lens = torch.clamp(audio_codes_lens_max - audio_codes_lens - 4, min=0)

                # 2. Round DOWN to the nearest multiple of the stacking factor
                pad_lens = (raw_pad_lens // self.frame_stacking_factor) * self.frame_stacking_factor

                # 3. Calculate perfectly aligned text padding
                text_pad_lens = pad_lens // self.frame_stacking_factor

                if pad_lens.max() > 0:
                    device = audio_codes.device
                    B, C, T_audio = audio_codes.shape

                    # --- Vectorized Audio Shift ---
                    idx_a = torch.arange(T_audio, device=device).unsqueeze(0)
                    src_idx_a = idx_a - pad_lens.unsqueeze(1)

                    valid_mask_a = (src_idx_a >= 0) & (src_idx_a < audio_codes_lens.unsqueeze(1))
                    safe_src_idx_a = src_idx_a.clamp(min=0, max=T_audio - 1)

                    safe_src_idx_a_exp = safe_src_idx_a.unsqueeze(1).expand(-1, C, -1)
                    valid_mask_a_exp = valid_mask_a.unsqueeze(1).expand(-1, C, -1)

                    gathered_audio = torch.gather(audio_codes, 2, safe_src_idx_a_exp)
                    silence_pad = self.codec_sil_codes_unconverted.view(1, C, 1).expand(B, C, T_audio)

                    audio_codes = torch.where(valid_mask_a_exp, gathered_audio, silence_pad)
                    audio_codes_lens = torch.clamp(audio_codes_lens + pad_lens, max=T_audio)

                    # Vectorized Text Shift
                    old_text = batch['text']
                    text_lens = batch['text_lens']

                    new_T_text = old_text.size(1)
                    new_text_lens = torch.clamp(text_lens + text_pad_lens, max=new_T_text)
                    idx_t = torch.arange(new_T_text, device=device).unsqueeze(0)
                    src_idx_t = idx_t - text_pad_lens.unsqueeze(1)

                    valid_mask_t = (src_idx_t >= 0) & (src_idx_t < text_lens.unsqueeze(1))
                    safe_src_idx_t = src_idx_t.clamp(min=0, max=old_text.size(1) - 1)
                    gathered_text = torch.gather(old_text, 1, safe_src_idx_t)

                    batch['text'] = torch.where(valid_mask_t, gathered_text, self.pad_id)
                    batch['text_lens'] = new_text_lens

                    # Vectorized Phoneme Shift
                    if (
                        self.phoneme_tokenizer is not None
                        and batch.get("phoneme_tokens") is not None
                        and batch.get("phoneme_tokens_lens") is not None
                    ):
                        old_phonemes = batch["phoneme_tokens"]
                        phoneme_lens = batch["phoneme_tokens_lens"]

                        new_T_phoneme = old_phonemes.size(1)
                        new_phoneme_lens = torch.clamp(phoneme_lens + text_pad_lens, max=new_T_phoneme)

                        idx_p = torch.arange(new_T_phoneme, device=device).unsqueeze(0)
                        src_idx_p = idx_p - text_pad_lens.unsqueeze(1)

                        valid_mask_p = (src_idx_p >= 0) & (src_idx_p < phoneme_lens.unsqueeze(1))
                        safe_src_idx_p = src_idx_p.clamp(min=0, max=old_phonemes.size(1) - 1)

                        gathered_phonemes = torch.gather(old_phonemes, 1, safe_src_idx_p)

                        phoneme_pad_id = getattr(self.phoneme_tokenizer, "pad", -1)
                        batch["phoneme_tokens"] = torch.where(
                            valid_mask_p,
                            gathered_phonemes,
                            torch.full_like(gathered_phonemes, phoneme_pad_id),
                        )
                        batch["phoneme_tokens_lens"] = new_phoneme_lens

                    # change batch["agent_mask"]  to consider this augmentation (in practice adding zeros/False where we are adding silence )
                    if self.cfg.get("use_multiturn_dataset", False) and "agent_mask" in batch:
                        old_agent_mask = batch["agent_mask"].bool()
                        T_mask = old_agent_mask.size(1)

                        idx_m = torch.arange(T_mask, device=device).unsqueeze(0)
                        src_idx_m = idx_m - text_pad_lens.unsqueeze(1)

                        valid_mask_m = (src_idx_m >= 0) & (src_idx_m < old_agent_mask.size(1))
                        safe_src_idx_m = src_idx_m.clamp(min=0, max=old_agent_mask.size(1) - 1)

                        gathered_agent_mask = torch.gather(old_agent_mask, 1, safe_src_idx_m)

                        # New prepended silence/user region should be non-agent.
                        batch["agent_mask"] = torch.where(
                            valid_mask_m,
                            gathered_agent_mask,
                            torch.zeros_like(gathered_agent_mask),
                        )

        if (
            self.cfg.get("use_multiturn_dataset", False)
            and batch["user_audio_turn_splitted"] is not None
            and self.cfg.get("condition_on_user_speech", False)
        ):
            input_samples_per_frame = self.codec_model_samples_per_frame * self.frame_stacking_factor

            user_audio = batch["user_audio_turn_splitted"]
            user_audio_lens = batch["user_audio_turn_splitted_lens"]

            silence_prob = float(self.cfg.get("user_cond_silence_augmentation_prob", 0.0) or 0.0)
            if self.training and silence_prob > 0.0:
                silence_mask = torch.rand(
                    user_audio.size(0),
                    device=user_audio.device,
                ) < silence_prob

                if silence_mask.any():
                    user_audio = user_audio.clone()
                    user_audio[silence_mask] = 0.0

            user_audio_codes, user_audio_codes_lens = self._codec_helper.audio_to_codes(
                user_audio,
                user_audio_lens,
            )

            if self._codec_converter is not None:
                user_audio_codes = self._codec_converter.convert_original_to_new(
                    audio_tokens=user_audio_codes,
                    audio_lens=user_audio_codes_lens,
                ).long()

            user_audio_codes, user_audio_codes_lens = self.stack_codes(
                user_audio_codes,
                user_audio_codes_lens,
                self.audio_bos_id,
                self.audio_eos_id,
                self.frame_stacking_factor,
                self.num_audio_codebooks,
            )

            user_audio_embedded = self.embed_audio_tokens(user_audio_codes)

            B = batch["text"].shape[0]
            T = batch["text"].shape[1]
            D = user_audio_embedded.shape[-1]

            user_audio_embedded_restored = user_audio_embedded.new_zeros(B, T, D)
            user_audio_embedded_mask = torch.zeros(B, T, device=user_audio_embedded.device, dtype=torch.bool)

            sample_prob = float(self.cfg.get("user_cond_trim_augmentation_sample_prob", 0.0) or 0.0)
            turn_prob = float(self.cfg.get("user_cond_trim_augmentation_turn_prob", 0.0) or 0.0)
            base_trim = int(self.cfg.get("user_cond_trim_augmentation_base", 0) or 0)

            if self.training and sample_prob > 0.0 and turn_prob > 0.0 and base_trim > 0:
                sample_trim_aug = torch.rand(B, device=user_audio_embedded.device) < sample_prob
            else:
                sample_trim_aug = torch.zeros(B, device=user_audio_embedded.device, dtype=torch.bool)

            indices = batch["user_audio_turn_splitted_indices"].to(user_audio_embedded.device)
            for turn_idx, (b, start_sample, end_sample) in enumerate(indices):
                b = int(b.item())
                if b < 0:
                    continue

                start_frame = int(torch.ceil(start_sample.float() / input_samples_per_frame).item())
                end_frame = int(end_sample.item()) // input_samples_per_frame

                start_frame = max(0, min(start_frame, T))
                end_frame = max(start_frame, min(end_frame, T))

                seq_len = end_frame - start_frame
                if seq_len <= 0:
                    continue

                boundary_trim = self.cfg.get("user_audio_boundary_trim", 0)
                boundary_trim = 0 if boundary_trim is None else int(boundary_trim)

                if boundary_trim == 0:
                    real_start = 0
                    real_end = int(user_audio_codes_lens[turn_idx].item())
                else:
                    turn_len_with_special = int(user_audio_codes_lens[turn_idx].item())
                    real_start = 1
                    real_end = max(real_start, turn_len_with_special - 1)

                turn_emb = user_audio_embedded[turn_idx, real_start:real_end]

                copy_len = min(seq_len, turn_emb.size(0))
                if copy_len <= 0:
                    continue

                turn_emb = turn_emb[:copy_len].clone()
                turn_mask = torch.ones(copy_len, device=user_audio_embedded.device, dtype=torch.bool)

                if boundary_trim > 0:
                    trim = min(boundary_trim, copy_len // 2)
                    if trim > 0:
                        turn_emb[:trim] = 0.0
                        turn_emb[copy_len - trim:] = 0.0
                        turn_mask[:trim] = False
                        turn_mask[copy_len - trim:] = False

                if bool(sample_trim_aug[b].item()):
                    do_turn_aug = torch.rand((), device=user_audio_embedded.device).item() < turn_prob

                    if do_turn_aug:
                        trim_delta = int(
                            torch.randint(
                                low=-1,
                                high=2,  # {-1, 0, 1}
                                size=(),
                                device=user_audio_embedded.device,
                            ).item()
                        )

                        trim_amount = max(1, base_trim + trim_delta)
                        trim_amount = min(trim_amount, max(1, copy_len - 1))

                        aug_choice = random.choices(
                            ["left", "right", "both"],
                            weights=[0.3, 0.3, 0.4],
                            k=1,
                        )[0]

                        zero_emb_pad = turn_emb.new_zeros(trim_amount, turn_emb.size(-1))
                        zero_mask_pad = torch.zeros(trim_amount, device=turn_mask.device, dtype=turn_mask.dtype)

                        if aug_choice == "left":
                            # Remove tokens from the left, then right-pad zeros.
                            kept_emb = turn_emb[trim_amount:]
                            kept_mask = turn_mask[trim_amount:]
                            turn_emb = torch.cat([kept_emb, zero_emb_pad], dim=0)
                            turn_mask = torch.cat([kept_mask, zero_mask_pad], dim=0)

                        elif aug_choice == "right":
                            # Remove tokens from the right, then right-pad zeros.
                            # This preserves timing of the left side and removes the transition/right edge.
                            kept_emb = turn_emb[: copy_len - trim_amount]
                            kept_mask = turn_mask[: copy_len - trim_amount]

                            turn_emb = torch.cat([kept_emb, zero_emb_pad], dim=0)
                            turn_mask = torch.cat([kept_mask, zero_mask_pad], dim=0)

                        else:  # "both"
                            # Remove trim_amount total tokens split across left and right.
                            left_trim = trim_amount // 2
                            right_trim = trim_amount - left_trim

                            # If trim_amount is odd, randomly decide which side loses the extra token.
                            if trim_amount % 2 == 1 and torch.rand((), device=user_audio_embedded.device).item() < 0.5:
                                left_trim, right_trim = right_trim, left_trim

                            kept_emb = turn_emb[left_trim : copy_len - right_trim]
                            kept_mask = turn_mask[left_trim : copy_len - right_trim]

                            turn_emb = torch.cat([kept_emb, zero_emb_pad], dim=0)
                            turn_mask = torch.cat([kept_mask, zero_mask_pad], dim=0)

                        # Safety: keep exact same length for restore assignment.
                        turn_emb = turn_emb[:copy_len]
                        turn_mask = turn_mask[:copy_len]

                dst_start = start_frame
                dst_end = start_frame + copy_len

                user_audio_embedded_restored[b, dst_start:dst_end] = turn_emb
                user_audio_embedded_mask[b, dst_start:dst_end] = turn_mask

            user_audio_embedded = user_audio_embedded_restored
            user_audio_mask = user_audio_embedded_mask

            # compare these two masks showing count batch level overlaps, left and right overlap per item batch. Consider only where both are ones.
            """
            if "agent_mask" in batch and batch["agent_mask"] is not None:
                user_cmp = user_audio_mask.bool()
                agent_cmp = batch["agent_mask"].to(user_cmp.device).bool()
                T_cmp = min(user_cmp.size(1), agent_cmp.size(1))
                valid = torch.arange(T_cmp, device=user_cmp.device)[None, :] < batch["text_lens"].to(user_cmp.device)[:, None]
                valid = valid[:, :T_cmp]
                user_cmp = user_cmp[:, :T_cmp] & valid             
                agent_cmp = agent_cmp[:, :T_cmp] & valid

                overlap = user_cmp & agent_cmp

                left_overlap = torch.zeros(B, device=user_cmp.device, dtype=torch.long)
                right_overlap = torch.zeros(B, device=user_cmp.device, dtype=torch.long)
                middle_overlap = torch.zeros(B, device=user_cmp.device, dtype=torch.long)

                boundary_width = int(self.cfg.get("user_agent_overlap_boundary_width", 10))

                for bi in range(B):
                    user_idx = user_cmp[bi].nonzero(as_tuple=False).flatten()
                    if user_idx.numel() == 0:
                        continue

                    breaks = torch.where(user_idx[1:] != user_idx[:-1] + 1)[0] + 1
                    spans = torch.tensor_split(user_idx, breaks.cpu().tolist())

                    for span in spans:
                        s = int(span[0].item())
                        e = int(span[-1].item()) + 1

                        overlap_span = overlap[bi, s:e]
                        if not overlap_span.any():
                            continue

                        left_end = min(e, s + boundary_width)
                        right_start = max(s, e - boundary_width)

                        left_overlap[bi] += overlap[bi, s:left_end].sum()
                        right_overlap[bi] += overlap[bi, right_start:e].sum()

                        boundary_mask = torch.zeros_like(overlap_span)
                        boundary_mask[: left_end - s] = True
                        boundary_mask[right_start - s :] = True

                        middle_overlap[bi] += (overlap_span & ~boundary_mask).sum()

                logging.info(
                    "[user/agent-mask overlap debug] "
                    f"overlap_frames={overlap.sum(dim=1).detach().cpu().tolist()} "
                    f"left_overlap={left_overlap.detach().cpu().tolist()} "
                    f"right_overlap={right_overlap.detach().cpu().tolist()} "
                    f"middle_overlap={middle_overlap.detach().cpu().tolist()} "
                    f"user_frames={user_cmp.sum(dim=1).detach().cpu().tolist()} "
                    f"agent_frames={agent_cmp.sum(dim=1).detach().cpu().tolist()}"
                )
            """
        else:
            user_audio_embedded = None
            user_audio_mask = None

        batch_output = self.process_batch(
            text=batch['text'],
            text_lens=batch['text_lens'],
            context_text_tokens=batch['context_text_tokens'],
            context_text_tokens_lens=batch['context_text_tokens_lens'],
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            phoneme_tokens=batch.get('phoneme_tokens'),
            phoneme_tokens_lens=batch.get('phoneme_tokens_lens'),
            mode="train",
            task=batch["task"] if self.cfg.get("use_multiturn_dataset", False) else None,
            agent_mask=batch["agent_mask"] if self.cfg.get("use_multiturn_dataset", False) else None,
            user_audio_embedded=user_audio_embedded
        )
        loss = batch_output.loss
        codebook_loss = batch_output.codebook_loss
        self.log('train/codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)

        if self.phoneme_tokenizer is not None:
            phoneme_loss = batch_output.phoneme_loss
            self.log('train/phoneme_loss', phoneme_loss, prog_bar=True, sync_dist=True)

        local_transformer_loss = batch_output.local_transformer_loss
        if local_transformer_loss is not None:
            self.log('train/local_transformer_loss', local_transformer_loss, prog_bar=True, sync_dist=True)

        # Log training mode info for multi-mode training
        if batch_output.selected_training_mode is not None:
            # Log which mode was selected for this batch
            # Convert mode name to an index for logging
            mode_idx = self.mode_name_to_mode[batch_output.selected_training_mode].mode_idx
            self.log('train/training_mode_idx', float(mode_idx), on_step=True)

        # Log batch info
        batch_size, text_token_max_len = batch["text"].shape
        text_token_total_num = batch["text_lens"].sum()
        batch_info_dict = {
            "train/batch_size": batch_size,
            "train/text_token_max_len": text_token_max_len,
            "train/text_token_total_num_in_batch": text_token_total_num,
            "train/text_token_pad_ratio_percent_in_batch": 100
            * (1 - text_token_total_num / (batch_size * text_token_max_len)),
        }

        if "audio_codes" in batch:
            audio_codes_max_len = batch["audio_codes"].shape[-1]
            audio_codes_total_num = batch["audio_codes_lens"].sum()
            batch_info_dict.update(
                {
                    "train/audio_codes_max_len": audio_codes_max_len,
                    "train/audio_codes_total_num_in_batch": audio_codes_total_num,
                    "train/audio_codes_pad_ratio_percent_in_batch": 100
                    * (1 - audio_codes_total_num / (batch_size * audio_codes_max_len)),
                }
            )
        else:
            audio_samples_max_len = batch["audio"].shape[-1]
            audio_samples_total_num = batch["audio_lens"].sum()
            batch_info_dict.update(
                {
                    "train/audio_samples_max_len": audio_samples_max_len,
                    "train/audio_samples_total_num_in_batch": audio_samples_total_num,
                    "train/audio_samples_pad_ratio_percent_in_batch": 100
                    * (1 - audio_samples_total_num / (batch_size * audio_samples_max_len)),
                }
            )

        self.log_dict(batch_info_dict, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Extract inputs from batch and pass explicitly to process_batch
        print(
            f"[Validation] global_rank: {self.global_rank}, "
            f"local_rank: {self.local_rank}, "
            f"world_size: {self.trainer.world_size}, "
            f"batch_idx: {batch_idx}"
        )
        if 'context_audio_codes' in batch:
            context_audio_codes = batch['context_audio_codes']
            context_audio_codes_lens = batch['context_audio_codes_lens']
        else:
            context_audio = batch['context_audio']
            context_audio_lens = batch['context_audio_lens']
            context_audio_codes, context_audio_codes_lens = self._codec_helper.audio_to_codes(
                context_audio, context_audio_lens
            )

        if 'audio_codes' in batch:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']
        else:
            audio = batch['audio']
            audio_lens = batch['audio_lens']
            audio_codes, audio_codes_lens = self._codec_helper.audio_to_codes(audio, audio_lens)

        batch_output = self.process_batch(
            text=batch['text'],
            text_lens=batch['text_lens'],
            context_text_tokens=batch['context_text_tokens'],
            context_text_tokens_lens=batch['context_text_tokens_lens'],
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            phoneme_tokens=batch.get('phoneme_tokens'),
            phoneme_tokens_lens=batch.get('phoneme_tokens_lens'),
            mode="val",
            task=batch["task"] if "task" in batch else None,
            agent_mask=batch["agent_mask"] if "agent_mask" in batch else None,
        )
        # Access ProcessBatchOutput dataclass attributes
        # logits come from the parallel prediction head
        # If using local_transformer, local_transformer_logits are also available
        loss = batch_output.loss
        codebook_loss = batch_output.codebook_loss
        logits = batch_output.logits
        audio_codes_target = batch_output.audio_codes_target
        audio_codes_lens_target = batch_output.audio_codes_lens_target
        context_audio_codes = batch_output.context_audio_codes
        context_audio_codes_lens = batch_output.context_audio_codes_lens

        if batch_idx == 0 and self.global_rank == 0:
            # Prepare dictionary for aggregated wandb logging
            wandb_log_dict = {}

            # Get audio data for logging
            wandb_log_dict.update(
                self.log_val_audio_example(
                    logits, audio_codes_target, audio_codes_lens_target, context_audio_codes, context_audio_codes_lens
                )
            )

            # Perform single wandb log call if wandb is active and there is data
            for logger in self.loggers:
                if isinstance(logger, WandbLogger) and wandb_log_dict:
                    logger.experiment.log(wandb_log_dict)

        local_transformer_loss = batch_output.local_transformer_loss
        val_output = {
            'val_loss': loss,
            'val_codebook_loss': codebook_loss,
            'val_local_transformer_loss': local_transformer_loss,
        }

        if self.phoneme_tokenizer is not None:
            phoneme_loss = batch_output.phoneme_loss
            val_output['val_phoneme_loss'] = phoneme_loss

        # Run inference and compute metrics if enabled
        if self.run_val_inference:
            infer_output = self.infer_batch(
                batch,
                max_decoder_steps=330,
                temperature=0.7,
                topk=80,
                use_local_transformer_for_inference=self.local_transformer_type == LocalTransformerType.AR,
                use_cfg=self.cfg.get('inference_use_cfg_in_val', True),
                cfg_scale=2.5,
            )

            # Get audio output directory
            audio_dir = self.trainer.log_dir
            audio_dir = os.path.join(audio_dir, 'val_audios')
            os.makedirs(audio_dir, exist_ok=True)

            # Save predicted and context audio, collect paths for metrics
            predicted_audio_paths = []
            context_audio_paths = []

            context_audio_codes_cleaned, context_audio_codes_lens_cleaned = remove_special_tokens(
                codes=context_audio_codes,
                codes_len=context_audio_codes_lens,
            )
            context_audio_codes_cleaned, context_audio_codes_lens_cleaned = self._prepare_codes_for_decode(
                context_audio_codes_cleaned,
                context_audio_codes_lens_cleaned,
            )
            context_audio_cleaned, context_audio_lens_cleaned, _ = self._codec_helper.codes_to_audio(
                context_audio_codes_cleaned,
                context_audio_codes_lens_cleaned,
            )

            for idx in range(infer_output.predicted_audio.size(0)):
                audio_np = infer_output.predicted_audio[idx].float().detach().cpu().numpy()
                audio_np = audio_np[: infer_output.predicted_audio_lens[idx]]

                # Log first batch on first device to wandb/tensorboard (first 3 samples)
                if batch_idx == 0 and self.global_rank == 0 and idx < 3:
                    for logger in self.loggers:
                        if isinstance(logger, WandbLogger):
                            logger.experiment.log(
                                {
                                    f"Audio_Generated/Example_{idx}": wandb.Audio(
                                        audio_np, sample_rate=self.output_sample_rate, caption="generated"
                                    )
                                }
                            )
                        elif isinstance(logger, TensorBoardLogger):
                            logger.experiment.add_audio(
                                f'Example_{idx}/generated',
                                audio_np,
                                global_step=self.global_step,
                                sample_rate=self.output_sample_rate,
                            )

                # Save predicted audio to disk
                if audio_dir:
                    audio_path = os.path.join(audio_dir, f'rank{self.global_rank}_batch{batch_idx}_idx{idx}.wav')
                    sf.write(audio_path, audio_np, self.output_sample_rate)
                    predicted_audio_paths.append(audio_path)

                    # Save context audio for SSIM computation
                    ctx_audio_np = (
                        context_audio_cleaned[idx].float().detach().cpu().numpy()[: context_audio_lens_cleaned[idx]]
                    )
                    ctx_path = os.path.join(audio_dir, f'rank{self.global_rank}_batch{batch_idx}_idx{idx}_context.wav')
                    sf.write(ctx_path, ctx_audio_np, self.output_sample_rate)
                    context_audio_paths.append(ctx_path)

            # Compute metrics if we have audio paths
            if predicted_audio_paths and context_audio_paths:
                with torch.no_grad():
                    # ASR transcription for CER/WER
                    if self.use_multilingual_asr:
                        self.whisper_model.to(self.device)
                        languages = batch.get('languages', None)
                        if languages is None:
                            languages = ['en'] * len(predicted_audio_paths)
                        try:
                            transcripts = transcribe_with_whisper_from_filepaths(
                                audio_filepaths=predicted_audio_paths,
                                language=languages,
                                whisper_processor=self.whisper_processor,
                                whisper_model=self.whisper_model,
                                device=self.device,
                                normalizer=None,
                            )
                            pred_transcripts = [process_text_for_cer(transcript) for transcript in transcripts]
                        except Exception as e:
                            logging.warning(
                                f"Val batched ASR transcription failed, falling back to per-file mode: {e}"
                            )
                            pred_transcripts = []
                            for item_idx, audio_path in enumerate(predicted_audio_paths):
                                lang = languages[item_idx] if item_idx < len(languages) else 'en'
                                try:
                                    transcript = transcribe_with_whisper(
                                        audio_path,
                                        lang,
                                        self.whisper_processor,
                                        self.whisper_model,
                                        self.device,
                                        normalizer=None,
                                    )
                                    pred_transcripts.append(process_text_for_cer(transcript))
                                except Exception as inner_e:
                                    logging.warning(f"Val ASR transcription failed for {audio_path}: {inner_e}")
                                    pred_transcripts.append(None)
                    else:
                        pred_transcripts = self._eval_asr_model.transcribe(
                            predicted_audio_paths,
                            batch_size=len(predicted_audio_paths),
                            override_config=TranscribeConfig(
                                use_lhotse=False, batch_size=len(predicted_audio_paths), num_workers=0
                            ),
                        )
                        pred_transcripts = [process_text_for_cer(t.text) for t in pred_transcripts]

                    # Speaker embeddings for SSIM
                    try:
                        pred_embeddings = get_speaker_embeddings_from_filepaths(
                            predicted_audio_paths, self._eval_speaker_verification_model, self.device
                        )
                        ctx_embeddings = get_speaker_embeddings_from_filepaths(
                            context_audio_paths, self._eval_speaker_verification_model, self.device
                        )
                    except Exception as e:
                        logging.warning(f"Val speaker embeddings failed: {e}")
                        pred_embeddings = ctx_embeddings = None

                    utmos_scores = None
                    if getattr(self, 'use_utmos', False) and hasattr(self, '_utmos_calculator'):
                        utmos_batch_size = max(int(self.cfg.get('utmos_batch_size', len(predicted_audio_paths))), 1)
                        utmos_num_workers = max(int(self.cfg.get('utmos_num_workers', 0)), 0)
                        try:
                            val_list = [os.path.basename(p) for p in predicted_audio_paths]
                            batch_results = self._utmos_calculator.process_directory(
                                audio_dir,
                                batch_size=utmos_batch_size,
                                num_workers=utmos_num_workers,
                                val_list=val_list,
                            )
                            utmos_scores = [float(item['predicted_mos']) for item in batch_results]
                        except Exception as e:
                            raise RuntimeError(f"Val UTMOSv2 batched scoring failed: {e}") from e

                    # Compute per-sample metrics for successful cases only
                    batch_cer, batch_wer, batch_ssim, batch_utmos = [], [], [], []
                    for idx in range(len(predicted_audio_paths)):
                        if pred_transcripts[idx] is None:
                            continue
                        gt_transcript = process_text_for_cer(batch['raw_texts'][idx])
                        cer = min(word_error_rate([pred_transcripts[idx]], [gt_transcript], use_cer=True), 1.0)
                        wer = min(word_error_rate([pred_transcripts[idx]], [gt_transcript], use_cer=False), 1.0)
                        batch_cer.append(cer)
                        batch_wer.append(wer)
                        ssim = None
                        if pred_embeddings is not None and ctx_embeddings is not None:
                            pred_emb = pred_embeddings[idx].cpu().float().numpy()
                            ctx_emb = ctx_embeddings[idx].cpu().float().numpy()
                            ssim = float(
                                np.dot(pred_emb, ctx_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ctx_emb))
                            )
                            batch_ssim.append(ssim)

                        # UTMOSv2 naturalness score (MOS on 1-5 scale)
                        utmos_score = None if utmos_scores is None else float(utmos_scores[idx])
                        if utmos_score is not None:
                            batch_utmos.append(utmos_score)

                        utmos_str = f", UTMOS={utmos_score:.4f}" if utmos_score is not None else ""
                        logging.info(
                            f"[Val] rank{self.global_rank}_batch{batch_idx}_idx{idx}: "
                            f"CER={cer:.4f}, WER={wer:.4f}{utmos_str} | GT: '{gt_transcript[:50]}...' | Pred: '{pred_transcripts[idx][:50]}...'"
                        )

                        # Save per-audio metrics JSON file alongside the audio file
                        if audio_dir:
                            metrics_dict = {
                                'cer': float(cer),
                                'wer': float(wer),
                                'ssim': ssim,
                                'utmos': utmos_score,
                                'gt_transcript': gt_transcript,
                                'pred_transcript': pred_transcripts[idx],
                                'audio_path': predicted_audio_paths[idx],
                                'epoch': self.trainer.current_epoch,
                                'global_step': self.global_step,
                            }
                            metrics_path = os.path.join(
                                audio_dir, f'rank{self.global_rank}_batch{batch_idx}_idx{idx}_metrics.json'
                            )
                            with open(metrics_path, 'w') as f:
                                json.dump(metrics_dict, f, indent=2)

                    if batch_cer:
                        val_output['val_cer'] = torch.tensor(np.mean(batch_cer), device=self.device)
                        val_output['val_wer'] = torch.tensor(np.mean(batch_wer), device=self.device)
                        if self.use_multilingual_asr:
                            langs = batch.get('languages', ['en'] * len(predicted_audio_paths))
                            val_output['val_languages'] = [
                                langs[i] for i in range(len(pred_transcripts)) if pred_transcripts[i] is not None
                            ]
                            val_output['val_cer_list'] = batch_cer
                            val_output['val_wer_list'] = batch_wer
                    if batch_ssim:
                        val_output['val_ssim'] = torch.tensor(np.mean(batch_ssim), device=self.device)
                    if batch_utmos:
                        val_output['val_utmos'] = torch.tensor(np.mean(batch_utmos), device=self.device)

        self.validation_step_outputs.append(val_output)

        return val_output

    def on_fit_start(self):
        super().on_fit_start()
        if not hasattr(self, "_codec_sil_codes_buffer"):
            self._generate_codec_silence_buffer()

    def on_validation_epoch_start(self) -> None:
        if torch.distributed.is_initialized():
            self.trainer.strategy.model.require_backward_grad_sync = False

    def on_validation_epoch_end(self):
        collect = lambda key: torch.stack([x[key] for x in self.validation_step_outputs]).mean()
        val_loss = collect("val_loss")
        val_codebook_loss = collect("val_codebook_loss")

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/codebook_loss", val_codebook_loss, prog_bar=True, sync_dist=True)

        if self.local_transformer_type != LocalTransformerType.NO_LT:
            val_local_transformer_loss = collect("val_local_transformer_loss")
            self.log("val/local_transformer_loss", val_local_transformer_loss, prog_bar=True, sync_dist=True)

        if self.phoneme_tokenizer is not None:
            val_phoneme_loss = collect("val_phoneme_loss")
            self.log("val/phoneme_loss", val_phoneme_loss, prog_bar=True, sync_dist=True)

        if self.run_val_inference:
            # Collect metrics only from outputs that have them
            def collect_if_exists(key):
                values = [x[key] for x in self.validation_step_outputs if key in x]
                if values:
                    return torch.stack(values).mean()
                return None

            val_metrics = ["val_cer", "val_wer", "val_ssim", "val_utmos"]
            for val_metric in val_metrics:
                metric_value = collect_if_exists(val_metric)
                if metric_value is not None:
                    self.log(val_metric.replace("val_", "val/", 1), metric_value, prog_bar=True, sync_dist=True)

            if self.use_multilingual_asr:
                lang_cer = {}
                lang_wer = {}
                for x in self.validation_step_outputs:
                    if 'val_languages' not in x or 'val_cer_list' not in x or 'val_wer_list' not in x:
                        continue
                    for lang, cer, wer in zip(x['val_languages'], x['val_cer_list'], x['val_wer_list']):
                        lang_cer.setdefault(lang, []).append(cer)
                        lang_wer.setdefault(lang, []).append(wer)
                for lang in lang_cer:
                    self.log(
                        f"val/cer_lang_{lang}",
                        torch.tensor(np.mean(lang_cer[lang]), device=self.device),
                        prog_bar=True,
                        sync_dist=True,
                    )
                for lang in lang_wer:
                    self.log(
                        f"val/wer_lang_{lang}",
                        torch.tensor(np.mean(lang_wer[lang]), device=self.device),
                        prog_bar=True,
                        sync_dist=True,
                    )

        self.validation_step_outputs.clear()  # free memory

        if torch.distributed.is_initialized():
            self.trainer.strategy.model.require_backward_grad_sync = True

    def get_dataset(self, dataset_cfg, dataset_type):
        dataset = safe_instantiate(
            dataset_cfg.dataset,
            sample_rate=self.sample_rate,
            bos_id=None,
            eos_id=self.eos_id,
            num_audio_codebooks=self.data_num_audio_codebooks,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            prior_scaling_factor=0.0,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=dataset_type,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            use_text_conditioning_tokenizer=True,
            text_conditioning_tokenizer_name=self.text_conditioning_tokenizer_name,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            add_language_to_context_text=self.add_language_to_context_text,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
            ignore_phoneme_languages=self.cfg.get("ignore_phoneme_languages", []),
            phoneme_as_text_prob=self.phoneme_as_text_prob if dataset_type == 'train' else 0.0,
            pronunciation_control_g2p=self.cfg.get("pronunciation_control_g2p", None),
        )
        dataset.load_16khz_audio = False
        dataset.tokenizer_config = (
            self.cfg.text_tokenizers
        )  # This will be used in worker_init_fn for instantiating tokenizer
        if self.phoneme_tokenizer is not None:
            dataset.phoneme_tokenizer_config = self.cfg.phoneme_tokenizer

        return dataset

    def get_lhotse_dataloader(self, dataset_cfg, mode='train') -> torch.utils.data.DataLoader:
        # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
        #   cfg is a classifier-free guidance.
        if self.cfg.get("use_multiturn_dataset", False):
            dataset = MagpieTTSLhotseMultiturnDataset(
                sample_rate=self.sample_rate,
                volume_norm=dataset_cfg.volume_norm,
                codec_model_samples_per_frame=self.codec_model_samples_per_frame,
                codec_model_input_sample_rate=self.codec_model_input_sample_rate,
                frame_stacking_factor=self.frame_stacking_factor,
                num_audio_codebooks=self.data_num_audio_codebooks,
                prior_scaling_factor=0.0,
                load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
                dataset_type=mode,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
                load_16khz_audio=False,
                pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
                context_duration_min=self.cfg.context_duration_min,
                context_duration_max=self.cfg.context_duration_max,
                use_text_conditioning_tokenizer=True,
                text_conditioning_tokenizer_name=self.text_conditioning_tokenizer_name,
                tokenizer_config=self.cfg.text_tokenizers,
                phoneme_tokenizer_config=self.cfg.get("phoneme_tokenizer", None),
                ignore_phoneme_languages=self.cfg.get("ignore_phoneme_languages", []),
                add_language_to_context_text=self.add_language_to_context_text,
                source_sample_rate=self.sample_rate,
                input_roles=["user", "User"],
                output_roles=["assistant", "Assistant", "agent", "Agent"],
                add_text_bos=self.cfg.get("add_text_bos", False),
                remove_user_turns_prob=self.cfg.get("remove_user_turns_prob", None),
            )
            dataset = FallbackDataset(dataset)
        else:
            dataset = MagpieTTSLhotseDataset(
                sample_rate=self.sample_rate,
                volume_norm=dataset_cfg.volume_norm,
                codec_model_samples_per_frame=self.codec_model_samples_per_frame,
                num_audio_codebooks=self.data_num_audio_codebooks,
                prior_scaling_factor=0.0,
                load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
                dataset_type=mode,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
                load_16khz_audio=False,
                pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
                context_duration_min=self.cfg.context_duration_min,
                context_duration_max=self.cfg.context_duration_max,
                use_text_conditioning_tokenizer=True,
                text_conditioning_tokenizer_name=self.text_conditioning_tokenizer_name,
                tokenizer_config=self.cfg.text_tokenizers,
                phoneme_tokenizer_config=self.cfg.get("phoneme_tokenizer", None),
                ignore_phoneme_languages=self.cfg.get("ignore_phoneme_languages", []),
                phoneme_as_text_prob=self.phoneme_as_text_prob if mode == 'train' else 0.0,
                pronunciation_control_g2p=self.cfg.get("pronunciation_control_g2p", None),
                add_language_to_context_text=self.add_language_to_context_text,
            )

        data_loader = get_lhotse_dataloader_from_config(
            config=dataset_cfg.dataset,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=dataset,
        )

        return data_loader

    def setup_training_data(self, dataset_cfg):
        if dataset_cfg.get("use_lhotse", False):
            # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
            #   cfg is a classifier-free guidance.
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
                if self.cfg.get("phoneme_tokenizer", None) is not None:
                    dataset.phoneme_tokenizer = safe_instantiate(self.cfg.phoneme_tokenizer)

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
            data_loader = self.get_lhotse_dataloader(dataset_cfg, mode='test')
        else:
            dataset = self.get_dataset(dataset_cfg, dataset_type='test')
            persistent_workers = True
            if dataset_cfg.dataloader_params.num_workers == 0:
                persistent_workers = False
                # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
                dataset.text_tokenizer = setup_tokenizers(all_tokenizers_config=self.cfg.text_tokenizers, mode='test')
                if self.cfg.get("phoneme_tokenizer", None) is not None:
                    dataset.phoneme_tokenizer = safe_instantiate(self.cfg.phoneme_tokenizer)

            data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                **dataset_cfg.dataloader_params,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )
        return data_loader

    def setup_validation_data(self, cfg):
        self._validation_uses_lhotse = cfg.get("use_lhotse", False)
        self._validation_dl = self._setup_test_dataloader(cfg)

    def setup_test_data(self, cfg):
        self._test_dl = self._setup_test_dataloader(cfg)

    def val_dataloader(self):
        """
        Override val_dataloader to lazily wrap with DistributedSampler for non-lhotse
        validation. This is needed because use_distributed_sampler=False is set for lhotse
        training, which also prevents Lightning from auto-wrapping the non-lhotse validation
        dataloader. We do this lazily (here instead of in setup_validation_data) because
        distributed is not yet initialized when setup_validation_data is called during __init__.
        """
        if self._validation_dl is None:
            self._validation_dl = []

        if getattr(self, '_validation_uses_lhotse', False):
            print(f"[val_dataloader] rank={self.global_rank}: Using lhotse, skipping DistributedSampler wrap")
            return self._validation_dl

        if not torch.distributed.is_initialized():
            print(
                f"[val_dataloader] rank={self.global_rank}: Distributed not initialized, skipping DistributedSampler wrap"
            )
            return self._validation_dl

        if getattr(self, '_val_dl_wrapped_with_dist_sampler', False):
            return self._validation_dl

        # Wrap the validation dataloader(s) with DistributedSampler
        dataloaders = self._validation_dl if isinstance(self._validation_dl, list) else [self._validation_dl]
        wrapped = []
        for i, dl in enumerate(dataloaders):
            if dl is not None and not isinstance(dl.sampler, DistributedSampler):
                print(
                    f"[val_dataloader] rank={self.global_rank}: Wrapping val dataloader {i} with DistributedSampler "
                    f"(dataset_len={len(dl.dataset)}, world_size={torch.distributed.get_world_size()}, "
                    f"batch_size={dl.batch_size}, num_workers={dl.num_workers})"
                )
                sampler = DistributedSampler(dl.dataset, shuffle=False)
                new_dl = torch.utils.data.DataLoader(
                    dl.dataset,
                    sampler=sampler,
                    batch_size=dl.batch_size,
                    num_workers=dl.num_workers,
                    collate_fn=dl.collate_fn,
                    pin_memory=dl.pin_memory,
                    drop_last=dl.drop_last,
                    worker_init_fn=dl.worker_init_fn,
                    persistent_workers=dl.persistent_workers,
                )
                wrapped.append(new_dl)
            else:
                sampler_type = type(dl.sampler).__name__ if dl is not None else "N/A"
                print(
                    f"[val_dataloader] rank={self.global_rank}: Val dataloader {i} already has "
                    f"sampler={sampler_type}, skipping wrap"
                )
                wrapped.append(dl)

        if isinstance(self._validation_dl, list):
            self._validation_dl = wrapped
        else:
            self._validation_dl = wrapped[0]

        self._val_dl_wrapped_with_dist_sampler = True
        return self._validation_dl

    def debug_decode_mask_regions(
        self,
        audio_codes_target,
        audio_codes_lens_target,
        agent_mask,
        out_dir,
        prefix="debug_mask",
    ):
        os.makedirs(out_dir, exist_ok=True)

        device = audio_codes_target.device
        B, C, T = audio_codes_target.shape

        agent_mask = agent_mask.to(device).bool()

        if agent_mask.size(1) < T:
            pad = torch.zeros(B, T - agent_mask.size(1), device=device, dtype=torch.bool)
            agent_mask = torch.cat([agent_mask, pad], dim=1)
        else:
            agent_mask = agent_mask[:, :T]

        valid = get_mask_from_lengths(audio_codes_lens_target).bool().to(device)
        agent_mask = agent_mask & valid

        C_base = self.num_audio_codebooks
        S = self.frame_stacking_factor
        C_target = audio_codes_target.size(1)

        sil = self.codec_sil_codes.to(device=device, dtype=audio_codes_target.dtype)

        if C_target == C_base:
            sil = sil.view(1, C_base, 1).expand(B, C_base, T)

        elif C_target == C_base * S:
            sil_unstacked = sil.view(1, C_base, 1).expand(B, C_base, T * S).contiguous()
            sil_stacked, _ = self.stack_codes(
                sil_unstacked,
                torch.full((B,), T * S, dtype=torch.long, device=device),
                bos_id=self.audio_bos_id,
                eos_id=self.audio_eos_id,
                stacking_factor=S,
                num_codebooks=C_base,
            )
            sil = sil_stacked[:, :, :T]
        else:
            raise RuntimeError(
                f"Unexpected codebook dim: target C={C_target}, "
                f"base C={C_base}, stacking_factor={S}"
            )

        def decode_and_save(codes, lens, name):
            codes = codes.clone()
            codes, lens = self._prepare_codes_for_decode(codes, lens)
            audio, audio_len, _ = self._codec_helper.codes_to_audio(codes, lens)

            for b in range(B):
                wav = audio[b, : audio_len[b]].float().detach().cpu().numpy()
                sf.write(
                    os.path.join(out_dir, f"{prefix}_b{b}_{name}.wav"),
                    wav,
                    self.output_sample_rate,
                )

        # 1. full target
        decode_and_save(audio_codes_target, audio_codes_lens_target, "full_target")

        # 2. only agent region, silence elsewhere
        agent_codes = torch.where(agent_mask[:, None, :], audio_codes_target, sil)
        decode_and_save(agent_codes, audio_codes_lens_target, "agent_only_sil_elsewhere")

        # 3. only masked-out region, silence elsewhere
        non_agent_codes = torch.where((~agent_mask & valid)[:, None, :], audio_codes_target, sil)
        decode_and_save(non_agent_codes, audio_codes_lens_target, "non_agent_only_sil_elsewhere")

        # 4. each contiguous agent segment independently
        for b in range(B):
            mask_b = agent_mask[b]
            idx = mask_b.nonzero(as_tuple=False).flatten()

            if idx.numel() == 0:
                continue

            # contiguous runs
            breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
            chunks = torch.tensor_split(idx, breaks.cpu().tolist())

            for seg_i, seg_idx in enumerate(chunks):
                start = int(seg_idx[0])
                end = int(seg_idx[-1]) + 1

                seg_codes = audio_codes_target[b : b + 1, :, start:end].clone()
                seg_lens = torch.tensor([end - start], device=device, dtype=torch.long)

                seg_codes, seg_lens = self._prepare_codes_for_decode(seg_codes, seg_lens)
                audio, audio_len, _ = self._codec_helper.codes_to_audio(seg_codes, seg_lens)

                wav = audio[0, : audio_len[0]].float().detach().cpu().numpy()
                sf.write(
                    os.path.join(out_dir, f"{prefix}_b{b}_agent_segment{seg_i}_frames{start}-{end}.wav"),
                    wav,
                    self.output_sample_rate,
                )

        logging.info(
            f"[mask_debug] saved mask decode files to {out_dir}; "
            f"agent coverage frames={agent_mask.sum(dim=1).detach().cpu().tolist()} / "
            f"{audio_codes_lens_target.detach().cpu().tolist()}"
        )