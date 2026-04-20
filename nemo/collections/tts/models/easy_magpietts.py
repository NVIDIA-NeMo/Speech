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
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data.distributed import DistributedSampler

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import MagpieTTSLhotseDataset, setup_tokenizers
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
from nemo.utils import logging

try:
    from nemo.collections.tts.modules.utmosv2 import UTMOSv2Calculator

    HAVE_UTMOSV2 = True
except (ImportError, ModuleNotFoundError):
    HAVE_UTMOSV2 = False

from transformers import WhisperForConditionalGeneration, WhisperProcessor


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

    def compute_loss(self, logits, audio_codes, audio_codes_lens):
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
        total_codebook_loss = None
        for codebook in range(audio_codes.size(1)):
            si = codebook * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = logits[:, :, si:ei]  # (B, T', num_tokens_per_codebook)
            codebook_targets = audio_codes[:, codebook]  # (B, T')
            codebook_loss = self.cross_entropy_loss(
                codebook_logits.permute(0, 2, 1), codebook_targets.long()  # (B, num_tokens_per_codebook, T')
            )  # (B, T')
            codebook_loss = codebook_loss * loss_mask[:, codebook, :]
            codebook_loss = codebook_loss.sum() / loss_mask[:, codebook, :].sum()
            if total_codebook_loss is None:
                total_codebook_loss = codebook_loss
            else:
                total_codebook_loss = total_codebook_loss + codebook_loss

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

        # Embed text tokens
        text_embedded = self.decoder.get_input_embeddings()(text)  # (B, L, E)

        # Apply CAS encoding if using BPE char tokenizer
        if self.use_bpe_char_tokenizer:
            text_mask = get_mask_from_lengths(text_lens)
            cas_embedding = self.cas_encoder(text, subword_mask=text_mask)  # (B, L, E)
            text_embedded = text_embedded + cas_embedding

        # Handle text dropout - zero out the embeddings
        if dropout_text_input:
            text_embedded = text_embedded * 0.0

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Prepare input and target for autoregressive training
        # Input: all tokens except the last (teacher forcing)
        # Target: all tokens except the first (shifted by one)
        audio_codes_lens_target = audio_codes_lens - 1
        audio_codes_target = audio_codes[:, :, 1:]  # (B, C, T'-1)
        audio_codes_input = audio_codes[:, :, :-1]  # (B, C, T'-1)

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

        return audio_channel_embedding, audio_channel_lens, audio_codes_target, audio_codes_lens_target

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

        # 3. Prepare text channel embeddings
        text_channel_embedding, text_channel_lens = self.prepare_text_channel_embeddings(
            text=text,
            text_lens=text_lens,
            delay=text_delay,
            dropout_text_input=dropout_text_input or dropout_conditional_input,
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
        ) = self.prepare_audio_channel_embeddings(
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
            delay=audio_delay,
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
        codebook_loss, _ = self.compute_loss(logits, audio_codes_target, audio_codes_lens_target)
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
                local_transformer_logits, audio_codes_target, audio_codes_lens_target
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
            audio_dir = os.path.join(audio_dir, 'val_audios', f'epoch_{self.trainer.current_epoch}')
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

    def get_dataset(self, dataset_cfg, dataset_type):
        dataset = instantiate(
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
                    dataset.phoneme_tokenizer = instantiate(self.cfg.phoneme_tokenizer)

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
                    dataset.phoneme_tokenizer = instantiate(self.cfg.phoneme_tokenizer)

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
