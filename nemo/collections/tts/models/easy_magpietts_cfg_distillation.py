# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""
Adapted EasyMagpieTTSModel class for classifier-free guidance distillation.
"""
import copy
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from io import TextIOWrapper
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
from nemo.collections.tts.losses.magpietts_cfg_distillation import (
    CodesCrossEntropyLoss,
    KLDivergenceLoss,
    NRMSELogitsLoss,
)
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_inference import InferBatchOutput, StreamingConfig, TrainingMode
from nemo.collections.tts.modules.magpietts_modules import (
    LocalTransformerType,
    clear_forbidden_logits,
    remove_special_tokens,
)
from nemo.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
    get_speaker_embeddings_from_filepaths,
    process_text_for_cer,
    transcribe_with_whisper,
    transcribe_with_whisper_from_filepaths,
)
from nemo.lightning.callback_group import CallbackGroup
from nemo.utils import logging

try:
    from nemo.collections.tts.modules.utmosv2 import UTMOSv2Calculator

    _HAVE_UTMOSV2 = True
except (ImportError, ModuleNotFoundError):
    _HAVE_UTMOSV2 = False

__all__ = ["EasyMagpieCFGDistillation"]


_STATE_DICT_EXCLUDE_NAMES: list[str] = ["_teacher_model"]


@dataclass
class _DefaultParams:
    """Default hyperparameters for EasyMagpieTTS CFG distillation."""

    # Maximum number of decoding steps during audio rollout generation.
    max_decoder_steps: int = 300
    # Sampling temperature during rollout generation.
    rollout_temperature: float = 0.7
    # Top-k sampling limit for token selection.
    rollout_topk: int = 80
    # Whether to use key-value cache during rollout inference.
    use_kv_cache_during_rollout: bool = True
    # Classifier-free guidance (CFG) scale used during distillation.
    distillation_cfg_scale: float = 2.5
    # Temperature used for softening logits during distillation (1.0 means no change).
    distillation_temperature: float = 1.0
    # Weight coefficient of the KLD distillation loss.
    audio_kl_loss_weight: float = 0.1
    # Weight coefficient of the cross-entropy distillation loss.
    audio_ce_loss_weight: float = 0.3
    # Weight coefficient for the NRMSE component in the distillation loss.
    audio_nrmse_loss_weight: float = 2.0
    # Fraction of the ground-truth sequence length used for finalized rollout rejection.
    lower_rejection_threshold: Optional[float] = 0.5
    # Fraction of the ground-truth sequence length used as a cutoff for early rollout truncation.
    upper_rejection_threshold: Optional[float] = 1.5
    # Weight assigned to truncated samples when computing the loss (used to down-weight rejected rollouts).
    rejection_weight: Optional[float] = 0.01
    # Whether to enable distillation of the local transformer head in addition to the main decoder logits.
    distill_local_transformer: bool = True
    # Target mixing weight for the local-transformer distillation loss in the final total loss.
    lt_loss_weight: float = 0.2
    # Global training step at which local-transformer distillation becomes active.
    lt_distillation_start_step: int = 1000
    # Number of steps used to linearly ramp the local-transformer loss weight from 0 to `lt_loss_weight`.
    lt_distillation_ramp_len: int = 2000
    # Weight coefficient for the phoneme-channel loss contribution in the final total loss.
    phoneme_ce_loss_weight: float = 0.01
    # Whether to save rejected teacher rollouts for debugging.
    use_rejection_debug_mode: bool = False


_DEFAULT_PARAMS = _DefaultParams()


def _validate_configuration(cfg: DictConfig) -> None:
    """Validate CFG-distillation hyperparameters.

    Args:
        cfg: Distillation model configuration to validate.

    Raises:
        ValueError: If the configuration contains unsupported values or an invalid
                combination of distillation settings.
    """
    if hasattr(cfg, "distillation_temperature") and cfg.get("distillation_temperature") <= 0:
        raise ValueError(
            "`distillation_temperature` must be greater than 0. "
            "Typical values for distillation are in the range [1.0, 4.0]."
        )

    if hasattr(cfg, "audio_kl_loss_weight") and cfg.get("audio_kl_loss_weight") < 0:
        raise ValueError("`audio_kl_loss_weight` must be non-negative.")

    if hasattr(cfg, "audio_ce_loss_weight") and cfg.get("audio_ce_loss_weight") < 0:
        raise ValueError("`audio_ce_loss_weight` must be non-negative.")

    audio_kl_loss_weight = cfg.get("audio_kl_loss_weight", _DEFAULT_PARAMS.audio_kl_loss_weight)
    audio_ce_loss_weight = cfg.get("audio_ce_loss_weight", _DEFAULT_PARAMS.audio_ce_loss_weight)

    if audio_kl_loss_weight == 0.0 and audio_ce_loss_weight == 0.0:
        raise ValueError("`audio_kl_loss_weight` and `audio_ce_loss_weight` cannot both be zero.")

    if hasattr(cfg, "audio_nrmse_loss_weight") and cfg.get("audio_nrmse_loss_weight") < 0:
        raise ValueError("`audio_nrmse_loss_weight` must be non-negative.")

    lower_rejection_threshold = cfg.get("lower_rejection_threshold")

    if lower_rejection_threshold is not None and not (0.0 <= lower_rejection_threshold <= 1.0):
        raise ValueError("`lower_rejection_threshold` must be in the range [0, 1] or `None`.")

    upper_rejection_threshold = cfg.get("upper_rejection_threshold")

    if upper_rejection_threshold is not None and upper_rejection_threshold < 1.0:
        raise ValueError("`upper_rejection_threshold` must be greater than or equal to 1.0 or `None`.")

    rejection_weight = cfg.get("rejection_weight")

    if rejection_weight is not None and not (0.0 <= rejection_weight <= 1.0):
        raise ValueError("`rejection_weight` must be in the range [0, 1] or `None`.")

    if hasattr(cfg, "lt_loss_weight") and not (0.0 <= cfg.get("lt_loss_weight") <= 1.0):
        raise ValueError("`lt_loss_weight` must be in the range [0, 1].")

    if hasattr(cfg, "lt_distillation_start_step") and cfg.get("lt_distillation_start_step") < 0:
        raise ValueError("`lt_distillation_start_step` must be non-negative.")

    if hasattr(cfg, "lt_distillation_ramp_len") and cfg.get("lt_distillation_ramp_len") < 0:
        raise ValueError("`lt_distillation_ramp_len` must be non-negative.")

    if hasattr(cfg, "phoneme_ce_loss_weight") and cfg.get("phoneme_ce_loss_weight") < 0:
        raise ValueError("`phoneme_ce_loss_weight` must be non-negative.")


def _get_teacher_model(cfg: DictConfig) -> EasyMagpieTTSModel:
    """Load and prepare the teacher model for distillation.

    Args:
        cfg: Distillation configuration containing the teacher
            checkpoint path and model settings.

    Returns:
        The frozen teacher model in evaluation mode.

    Raises:
        ValueError: If the teacher checkpoint format is unsupported.
    """
    model_path = Path(cfg.teacher_model_path)
    teacher_model_cfg = copy.deepcopy(cfg)

    with open_dict(teacher_model_cfg):
        teacher_model_cfg.train_ds = None
        teacher_model_cfg.validation_ds = None

    if model_path.suffix == ".ckpt":
        teacher_model = EasyMagpieTTSModel(cfg=teacher_model_cfg)
        ckpt = torch.load(model_path.as_posix(), map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        teacher_model.load_state_dict(state_dict)

    elif model_path.suffix == ".nemo":
        teacher_model = EasyMagpieTTSModel.restore_from(
            restore_path=model_path.as_posix(),
            override_config_path=teacher_model_cfg,
            map_location="cpu",
            strict=cfg.get("init_strict", True),
        )
    else:
        raise ValueError(f"Unsupported teacher model format: {model_path.suffix}")

    teacher_model.freeze()
    teacher_model.eval()
    teacher_model._no_state_dict = True

    return teacher_model


@dataclass
class _StudentOutput:
    """Outputs produced by the student forward pass for distillation."""

    logits: Tensor
    logits_lt: Optional[Tensor]
    logits_phonemes: Optional[Tensor]
    audio_codes_gt_lt: Optional[Tensor]
    audio_codes_lens_gt_lt: Optional[Tensor]


def _lt_sample_autoregressive(
    model: EasyMagpieTTSModel,
    dec_output: Tensor,
    temperature: float = 0.7,
    topk: int = 80,
    cfg_scale: float = 1.0,
    use_kv_cache: bool = True,
    forbid_audio_eos: bool = False,
    sanitize_logits: bool = False,
) -> tuple[Tensor, Tensor]:
    """Sample stacked audio codes autoregressively with the local transformer.

    Args:
        model: EasyMagpie model providing the local transformer and audio-token
            projections.
        dec_output: Decoder output for the current audio frame, including
            conditional and unconditional CFG batches.
        temperature: Sampling temperature. Values less than or equal to zero use
            greedy decoding.
        topk: Number of highest-logit tokens retained for sampling per codebook.
        cfg_scale: Classifier-free guidance scale applied to conditional and
            unconditional logits.
        use_kv_cache: Whether to use the local transformer's key-value cache.
        forbid_audio_eos: Whether to prevent sampling audio EOS tokens.
        sanitize_logits: Whether to replace non-finite logits and clamp their
            range before sampling.

    Returns:
        A tuple containing the sampled codec codes with shape
        `(batch_size, num_audio_codebooks, frame_stacking_factor)` and the
        corresponding CFG logits for the conditional batch.
    """
    model.local_transformer.reset_cache(use_cache=use_kv_cache)
    dec_output = dec_output.unsqueeze(1)
    local_transformer_input = model.local_transformer_in_projection(dec_output)
    predicted_codes = []
    predicted_logits = []

    for codebook_num in range(model.num_audio_codebooks * model.frame_stacking_factor):
        size = (local_transformer_input.size(0), local_transformer_input.size(1))
        _mask = torch.ones(*size, device=local_transformer_input.device)
        local_transformer_output = model.local_transformer(local_transformer_input, _mask)["output"]

        lt_out_for_proj = model.local_transformer_audio_out_projection(local_transformer_output[:, -1, :])
        codebook_logits = model.local_transformer_out_projections[codebook_num](lt_out_for_proj)

        bs = codebook_logits.size(0) // 2
        conditional_logits = codebook_logits[:bs]
        unconditional_logits = codebook_logits[bs:]
        cfg_logits = cfg_scale * conditional_logits + (1.0 - cfg_scale) * unconditional_logits
        codebook_logits[:bs] = cfg_logits
        predicted_logits.append(codebook_logits.clone())

        if sanitize_logits:
            codebook_logits = torch.nan_to_num(codebook_logits, nan=0.0, posinf=100.0, neginf=-100.0)
            codebook_logits = codebook_logits.clamp(min=-100.0, max=100.0)

        codebook_logits = clear_forbidden_logits(
            logits=codebook_logits.unsqueeze(1),
            codebook_size=model.codebook_size,
            forbid_audio_eos=forbid_audio_eos,
        )
        codebook_logits = codebook_logits.squeeze(1)

        codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]
        indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(-1)
        codebook_logits_rescored = codebook_logits.clone()
        codebook_logits_rescored[indices_to_remove] = float("-inf")

        if temperature <= 0.0:
            codebook_preds = codebook_logits_rescored.argmax(dim=-1, keepdim=True)
        else:
            codebook_probs = torch.softmax(codebook_logits_rescored / temperature, dim=-1)
            codebook_preds = torch.multinomial(codebook_probs, 1)

        codebook_preds[bs:] = codebook_preds[:bs]
        predicted_codes.append(codebook_preds)

        next_local_transformer_input = model.audio_embeddings[codebook_num](codebook_preds.squeeze(-1)).unsqueeze(1)
        next_local_transformer_input = model.audio_in_projection(next_local_transformer_input)
        next_local_transformer_input = model.local_transformer_in_projection(next_local_transformer_input)
        local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1)

    predicted_codes = torch.cat(predicted_codes, dim=1)[:bs]
    predicted_codes = predicted_codes.view(bs, model.num_audio_codebooks, model.frame_stacking_factor)
    predicted_logits = torch.cat(predicted_logits, dim=1)[:bs]

    return predicted_codes, predicted_logits


@dataclass
class _AudioCodesStep:
    """Predicted audio-code data for one rollout step."""

    codes_t: Tensor
    codes_t_argmax: Tensor
    logits_t: Tensor

    codes_t_lt: Optional[Tensor]
    codes_t_argmax_lt: Optional[Tensor]
    logits_t_lt: Optional[Tensor]

    def unstack(
        self,
        batch_size: int,
        num_codebooks: int,
        fs_factor: int,
    ) -> None:
        """Reshape flattened audio-code predictions into codec layout.

        Args:
            batch_size: Number of batch items represented by the flattened tensors.
            num_codebooks: Number of codec codebooks.
            fs_factor: Number of audio frames stacked into each prediction step.
        """
        self.codes_t = self.codes_t.view(batch_size, num_codebooks, fs_factor)
        self.codes_t_argmax = self.codes_t_argmax.view(batch_size, num_codebooks, fs_factor)

        if self.codes_t_lt is not None and self.codes_t_argmax_lt is not None and self.logits_t_lt is not None:
            self.codes_t_lt = self.codes_t_lt.view(batch_size, num_codebooks, fs_factor)
            self.codes_t_argmax_lt = self.codes_t_argmax_lt.view(batch_size, num_codebooks, fs_factor)


@dataclass
class _PhonemeTokensStep:
    """Predicted phoneme data for one rollout step."""

    tokens_t: Tensor
    logits_t: Tensor


@dataclass
class _ChannelStatus:
    """Track status masks for each model input channel."""

    context: Tensor
    text: Tensor
    phoneme: Tensor
    audio: Tensor


@dataclass
class _StreamingState:
    """Maintain state across streaming teacher-rollout steps."""

    config: StreamingConfig
    past_key_values: Optional[tuple]
    cache_seq_len: int

    pred_codes: list[Tensor]
    pred_codes_logits: list[Tensor]
    pred_codes_lt: list[Tensor]
    pred_codes_logits_lt: list[Tensor]
    pred_phoneme_tokens: list[Tensor]
    pred_phoneme_logits: list[Tensor]

    context_audio_codes: Tensor
    context_audio_codes_lens: Tensor
    context_lens: Tensor
    full_context_embedding: Tensor
    full_context_lens: Tensor

    context_position: Tensor
    text_tokens_seen: Tensor
    phoneme_steps: Tensor
    audio_steps: Tensor
    phoneme_stream_ended: Tensor
    phoneme_eos_detected: Tensor
    finished: Tensor
    last_hidden: Tensor
    text_finished: Tensor

    last_audio_codes: Optional[Tensor]
    last_audio_codes_lt: Optional[Tensor]
    last_phoneme_tokens: Optional[Tensor]

    audio_prediction_start_idx: Tensor
    audio_prediction_end_idx: Tensor
    phoneme_prediction_start_idx: Tensor
    phoneme_prediction_end_idx: Tensor

    gt_phoneme_embeddings: Optional[Tensor] = None
    gt_phoneme_lens: Optional[Tensor] = None

    rejected: Optional[Tensor] = None
    sample_weights: Optional[Tensor] = None

    @classmethod
    def create(
        cls,
        config: StreamingConfig,
        last_hidden: Tensor,
        past_kv: list[Tensor],
        min_context_len: Tensor,
        context_audio_codes: Tensor,
        context_audio_codes_lens: Tensor,
        context_lens: Tensor,
        full_context_embedding: Tensor,
        full_context_lens: Tensor,
        gt_phoneme_embeddings: Tensor,
        gt_phoneme_lens: Tensor,
        use_rollout_rejection: bool,
        rejection_weight: Optional[float],
    ) -> "_StreamingState":
        """Create an initialized streaming state for a teacher rollout.

        Initializes cache metadata, per-item counters, prediction buffers, and optional
        rollout-rejection state.

        Args:
            config: Static configuration for the streaming rollout.
            last_hidden: Hidden states from the prepared context forward pass.
            past_kv: Initial transformer key-value cache.
            min_context_len: Shared cached-context-prefix length, equal to the minimum
                prepared-context length across the batch.
            context_audio_codes: Prepared context audio codec tokens.
            context_audio_codes_lens: Lengths of the context audio-code sequences.
            context_lens: Lengths of the prepared context sequences.
            full_context_embedding: Embeddings for the complete prepared context.
            full_context_lens: Lengths of the complete prepared-context embeddings.
            gt_phoneme_embeddings: Ground-truth phoneme embeddings used for guidance.
            gt_phoneme_lens: Lengths of the ground-truth phoneme sequences.
            use_rollout_rejection: Whether to initialize per-item rejection tracking.
            rejection_weight: Optional loss weight to assign to rejected rollouts.

        Returns:
            An initialized streaming state.
        """
        bs = config.batch_size
        device = config.device
        rejected = None
        sample_weights = None

        if use_rollout_rejection:
            rejected = torch.zeros(bs, dtype=torch.bool, device=device)

        if rejection_weight is not None:
            sample_weights = torch.ones(bs, dtype=torch.float, device=device)

        obj = _StreamingState(
            config=config,
            past_key_values=past_kv,
            cache_seq_len=min_context_len,
            pred_codes=[],
            pred_codes_logits=[],
            pred_codes_lt=[],
            pred_codes_logits_lt=[],
            pred_phoneme_tokens=[],
            pred_phoneme_logits=[],
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            context_lens=context_lens,
            full_context_embedding=full_context_embedding,
            full_context_lens=full_context_lens,
            context_position=torch.full((bs,), min_context_len, dtype=torch.long, device=device),
            text_tokens_seen=torch.zeros(bs, dtype=torch.long, device=device),
            phoneme_steps=torch.zeros(bs, dtype=torch.long, device=device),
            audio_steps=torch.zeros(bs, dtype=torch.long, device=device),
            phoneme_stream_ended=torch.zeros(bs, dtype=torch.bool, device=device),
            phoneme_eos_detected=torch.zeros(bs, dtype=torch.bool, device=device),
            finished=torch.zeros(bs, dtype=torch.bool, device=device),
            last_hidden=last_hidden,
            text_finished=torch.zeros(bs, dtype=torch.bool, device=device),
            last_audio_codes=None,
            last_audio_codes_lt=None,
            last_phoneme_tokens=None,
            audio_prediction_start_idx=torch.full((bs,), -1, dtype=torch.long, device=device),
            audio_prediction_end_idx=torch.full((bs,), -1, dtype=torch.long, device=device),
            phoneme_prediction_start_idx=torch.full((bs,), -1, dtype=torch.long, device=device),
            phoneme_prediction_end_idx=torch.full((bs,), -1, dtype=torch.long, device=device),
            gt_phoneme_embeddings=gt_phoneme_embeddings,
            gt_phoneme_lens=gt_phoneme_lens,
            rejected=rejected,
            sample_weights=sample_weights,
        )
        return obj

    def update_counters(self, status: _ChannelStatus) -> None:
        """Update per-item context, text, phoneme and audio rollout counters.

        Args:
            status: Status masks indicating which channels progress for each batch item.
        """
        self.context_position = self.context_position + status.context.long()
        self.text_tokens_seen = self.text_tokens_seen + (~status.context).long()
        self.phoneme_steps = self.phoneme_steps + status.phoneme.long()
        self.audio_steps = self.audio_steps + status.audio.long()

    def update_phoneme_start_idx(self, needs_phoneme: Tensor) -> None:
        """Record the first rollout step that predicts phonemes for each item.

        Args:
            needs_phoneme: Mask identifying items that require phoneme prediction.
        """
        first_phoneme_step = needs_phoneme & (self.phoneme_prediction_start_idx == -1)

        if not first_phoneme_step.any():
            return

        self.phoneme_prediction_start_idx = torch.where(
            condition=first_phoneme_step,
            input=torch.full_like(self.phoneme_prediction_start_idx, self.current_phoneme_step_idx),
            other=self.phoneme_prediction_start_idx,
        )

    def update_phoneme_end_status(
        self,
        tokens: _PhonemeTokensStep,
        needs_phoneme: Tensor,
        eos_id: int,
    ) -> None:
        """Track phoneme EOS predictions and record phoneme end steps.

        Args:
            tokens: Predicted phoneme tokens for the current rollout step.
            needs_phoneme: Mask identifying items that require phoneme prediction.
            eos_id: Phoneme end-of-sequence token ID.
        """
        eos_detected = needs_phoneme & (tokens.tokens_t == eos_id).any(dim=1)
        self.phoneme_eos_detected = self.phoneme_eos_detected | eos_detected

        newly_ended = eos_detected & (self.phoneme_prediction_end_idx == -1)

        if newly_ended.any():
            self.phoneme_prediction_end_idx = torch.where(
                condition=newly_ended,
                input=torch.full_like(self.phoneme_prediction_end_idx, self.current_phoneme_step_idx),
                other=self.phoneme_prediction_end_idx,
            )

    def update_phoneme_step(
        self,
        tokens: _PhonemeTokensStep,
        fs_factor: int,
        vocab_size: int,
    ) -> None:
        """Store phoneme predictions and logits from the current rollout step.

        Args:
            tokens: Predicted phoneme tokens and their logits.
            fs_factor: Number of phoneme frames stacked per rollout step.
            vocab_size: Size of the phoneme vocabulary.
        """
        self.last_phoneme_tokens = tokens.tokens_t
        self.pred_phoneme_tokens.append(tokens.tokens_t.unsqueeze(1))  # (B, 1, FS)
        logits = tokens.logits_t.view(self.batch_size, 1, fs_factor * vocab_size)
        self.pred_phoneme_logits.append(logits)

    def update_audio_start_idx(self, needs_audio: Tensor) -> None:
        """Record the first rollout frame that predicts audio for each item.

        Args:
            needs_audio: Mask identifying items that require audio prediction.
        """
        first_audio_step = needs_audio & (self.audio_prediction_start_idx == -1)

        if not first_audio_step.any():
            return

        self.audio_prediction_start_idx = torch.where(
            condition=first_audio_step,
            input=torch.full_like(self.audio_prediction_start_idx, self.current_frame_idx),
            other=self.audio_prediction_start_idx,
        )

    def update_last_audio_codes(
        self,
        audio_codes: _AudioCodesStep,
        needs_audio: Tensor,
    ) -> None:
        """Update the most recently predicted audio codes for active items.

        Args:
            audio_codes: Backbone and optional local-transformer audio-code predictions.
            needs_audio: Mask identifying items that require audio prediction.
        """
        codes_t = audio_codes.codes_t.reshape(self.batch_size, -1)

        if self.last_audio_codes is None:
            self.last_audio_codes = codes_t
        else:
            update_mask = needs_audio.view(self.batch_size, 1).expand_as(codes_t)
            self.last_audio_codes = torch.where(update_mask, codes_t, self.last_audio_codes)

        if self.use_lt:
            codes_t_lt = audio_codes.codes_t_lt.reshape(self.batch_size, -1)

            if self.last_audio_codes_lt is None:
                self.last_audio_codes_lt = codes_t_lt
            else:
                update_mask = needs_audio.view(self.batch_size, 1).expand_as(codes_t_lt)
                self.last_audio_codes_lt = torch.where(update_mask, codes_t_lt, self.last_audio_codes_lt)

    def update_audio_end_status(
        self,
        audio_codes: _AudioCodesStep,
        needs_audio: Tensor,
        eos_id: int,
        fs_factor: int,
    ) -> list[int]:
        """Track audio EOS predictions and record completed rollout end frames.

        Args:
            audio_codes: Sampled and greedy audio-code predictions for the current step.
            needs_audio: Mask identifying items that require audio prediction.
            eos_id: Audio end-of-sequence token ID.
            fs_factor: Number of audio frames stacked per rollout step.

        Returns:
            Batch indices of items whose audio rollout ended at the current step.
        """
        # Use backbone codes to preserve EOS prediction policy.
        codes_t = audio_codes.codes_t
        codes_t_argmax = audio_codes.codes_t_argmax

        eos_in_sampled = codes_t == eos_id
        eos_in_argmax = codes_t_argmax == eos_id
        eos_any_codebook = eos_in_sampled.any(dim=1) | eos_in_argmax.any(dim=1)
        eos_detected = eos_any_codebook.any(dim=1) & needs_audio
        self.finished = self.finished | eos_detected

        newly_ended = eos_detected & (self.audio_prediction_end_idx == -1)

        if not newly_ended.any():
            return []

        # Intentionally retain the full stacked frame containing EOS.
        current_frame_count = len(self.pred_codes) * fs_factor
        end_frame_idx = torch.full_like(self.audio_prediction_end_idx, current_frame_count + fs_factor)
        self.audio_prediction_end_idx = torch.where(newly_ended, end_frame_idx, self.audio_prediction_end_idx)

        return torch.nonzero(newly_ended, as_tuple=False).flatten().tolist()

    def add_audio_codes(
        self,
        audio_codes: _AudioCodesStep,
    ) -> None:
        """Append audio-code predictions from the current rollout step.

        Args:
            audio_codes: Backbone and optional local-transformer audio-code predictions.
        """
        self.pred_codes.append(audio_codes.codes_t)
        self.pred_codes_logits.append(audio_codes.logits_t.unsqueeze(1))

        if self.use_lt:
            self.pred_codes_lt.append(audio_codes.codes_t_lt)
            self.pred_codes_logits_lt.append(audio_codes.logits_t_lt.unsqueeze(1))

    @property
    def current_phoneme_step_idx(self) -> int:
        """Get the current phoneme rollout step index."""
        return len(self.pred_phoneme_tokens)

    @property
    def current_frame_idx(self) -> int:
        """Get the current audio frame index."""
        return sum(p.size(-1) for p in self.pred_codes)

    @property
    def batch_size(self) -> int:
        """Get the rollout batch size."""
        return self.config.batch_size

    @property
    def temperature(self) -> float:
        """Get the audio sampling temperature."""
        return self.config.temperature

    @property
    def topk(self) -> int:
        """Get the top-k audio sampling limit."""
        return self.config.topk

    @property
    def cfg_scale(self) -> float:
        """Get the classifier-free guidance scale."""
        return self.config.cfg_scale

    @property
    def device(self) -> torch.device:
        """Get the rollout device."""
        return self.config.device

    @property
    def use_lt(self) -> bool:
        """Get whether local-transformer rollout is enabled."""
        return self.config.use_local_transformer

    @property
    def phoneme_sampling_method(self) -> str:
        """Get the phoneme sampling method."""
        return self.config.phoneme_sampling_method

    @property
    def streaming_speech_delay(self) -> int:
        """Get the text-token delay before audio generation begins."""
        return self.config.training_mode.streaming_speech_delay

    @property
    def streaming_phonemes_delay(self) -> int:
        """Get the text-token delay before phoneme generation begins."""
        return self.config.training_mode.streaming_phonemes_delay


def _create_status_from_state(state: _StreamingState) -> _ChannelStatus:
    """Create per-channel status masks for the current rollout step.

    Args:
        state: Current streaming rollout state.

    Returns:
        A container with status masks indicating which context, text,
        phoneme and audio channels are active for each batch item.
    """
    context = state.context_position < state.full_context_lens
    text = (~context) & (~state.text_finished)
    phoneme = (~context) & (state.text_tokens_seen >= state.streaming_phonemes_delay) & (~state.phoneme_stream_ended)
    audio = (~context) & (state.text_tokens_seen >= state.streaming_speech_delay) & (~state.finished)

    return _ChannelStatus(
        context=context,
        text=text,
        phoneme=phoneme,
        audio=audio,
    )


@dataclass
class _InitContext:
    """Store prepared context inputs used to initialize a teacher rollout."""

    embeds: Tensor
    lens: Tensor
    min_len: Tensor
    full_embeds: Tensor
    full_lens: Tensor
    dummy_embeds_uncond: Tensor
    cache_position: Tensor
    audio_codes: Tensor
    audio_codes_lens: Tensor


@dataclass
class _TeacherPredictedCodes:
    """Store audio-code predictions generated during a teacher rollout."""

    codes: Tensor
    logits: Tensor
    codes_len: Tensor
    codes_lt: Optional[Tensor]
    logits_lt: Optional[Tensor]


@dataclass
class _TeacherOutput:
    """Outputs produced by the teacher rollout used as distillation targets."""

    codes: Tensor
    logits: Tensor
    lens: Tensor
    sample_weights: Optional[Tensor]
    rejected: Optional[Tensor]

    codes_lt: Optional[Tensor]
    logits_lt: Optional[Tensor]


def _validate_predicted_codes(
    codes: Tensor,
    pred_codes_lens: Tensor,
    start_indices: Tensor,
    end_indices: Tensor,
    fs_factor: int,
) -> None:
    """Validate teacher rollout audio-code boundaries and lengths.

    Args:
        codes: Predicted audio codes for the rollout.
        pred_codes_lens: Finalized predicted-code lengths for each batch item.
        start_indices: Audio prediction start indices for each batch item.
        end_indices: Audio prediction end indices for each batch item.
        fs_factor: Number of audio frames stacked per decoder step.

    Raises:
        RuntimeError: If rollout boundaries or lengths are invalid, unaligned with
            stacked decoder steps or contain no audio predictions.
    """
    # If an item never entered audio generation, give it zero length by
    # using its end position as its start position. Clamping -1 to zero
    # would incorrectly assign predictions generated for other batch items to this item.
    invalid_bounds = (start_indices < 0) | (end_indices < start_indices) | (end_indices > codes.size(-1))

    if invalid_bounds.any():
        invalid_items = torch.nonzero(invalid_bounds, as_tuple=False).flatten()
        raise RuntimeError(
            "Invalid teacher rollout boundaries: "
            f"items={invalid_items.tolist()}, "
            f"starts={start_indices[invalid_items].tolist()}, "
            f"ends={end_indices[invalid_items].tolist()}, "
            f"available_frames={codes.size(-1)}."
        )

    # All boundaries must align with complete stacked decoder steps because
    # each stored logit position corresponds to one block of fs_factor unstacked codec frames.
    unaligned_bounds = (
        (start_indices % fs_factor != 0) | (end_indices % fs_factor != 0) | (pred_codes_lens % fs_factor != 0)
    )
    if unaligned_bounds.any():
        invalid_items = torch.nonzero(unaligned_bounds, as_tuple=False).flatten()
        raise RuntimeError(
            "Teacher rollout boundaries are not aligned with the audio "
            f"items={invalid_items.tolist()}, "
            f"starts={start_indices[invalid_items].tolist()}, "
            f"ends={end_indices[invalid_items].tolist()}, "
            f"lengths={pred_codes_lens[invalid_items].tolist()}."
        )

    if pred_codes_lens.max().item() == 0:
        raise RuntimeError("Teacher rollout produced no audio predictions.")


class _TeacherInferenceEngine:
    """Teacher-model inference engine for generating distillation targets."""

    def __init__(
        self,
        model: EasyMagpieTTSModel,
    ) -> None:
        self.model = model

    @property
    def phoneme_stacking_factor(self) -> int:
        """Get the number of phoneme frames stacked per prediction step."""
        return self.model.phoneme_stacking_factor

    @property
    def audio_stacking_factor(self) -> int:
        """Get the number of audio frames stacked per prediction step."""
        return self.model.frame_stacking_factor

    @property
    def phoneme_vocab_size(self) -> int:
        """Get the phoneme vocabulary size."""
        return self.model.phoneme_vocab_size

    @property
    def num_audio_codebooks(self) -> int:
        """Get the number of audio codebooks."""
        return self.model.num_audio_codebooks

    @property
    def num_audio_tokens_per_codebook(self) -> int:
        """Get the number of tokens in each audio codebook."""
        return self.model.num_all_tokens_per_codebook

    @property
    def phoneme_bos_id(self) -> int:
        """Get the phoneme BOS token ID."""
        return self.model.phoneme_tokenizer.bos_token_id

    @property
    def phoneme_eos_id(self) -> int:
        """Get the phoneme EOS token ID."""
        return self.model.phoneme_tokenizer.eos_token_id

    @property
    def audio_bos_id(self) -> int:
        """Get the audio BOS token ID."""
        return self.model.audio_bos_id

    @property
    def audio_eos_id(self) -> int:
        """Get the audio EOS token ID."""
        return self.model.audio_eos_id

    @property
    def eos_id(self) -> int:
        """Get the model EOS token ID."""
        return self.model.eos_id

    @property
    def embedding_dim(self) -> int:
        """Get the model embedding dimension."""
        return self.model.cfg.embedding_dim

    @property
    def use_multiturn_dataset(self) -> bool:
        """Get whether the model uses the multiturn dataset format."""
        return self.model.cfg.get("use_multiturn_dataset", False)

    def _prepare_init_context(
        self,
        context_audio_codes: Tensor,
        context_audio_codes_lens: Tensor,
        context_text_tokens: Tensor,
        context_text_tokens_lens: Tensor,
        training_mode: TrainingMode,
    ) -> _InitContext:
        """Prepare context tensors for initializing a CFG teacher rollout.

        Args:
            context_audio_codes: Context audio codec tokens.
            context_audio_codes_lens: Lengths of the context audio-code sequences.
            context_text_tokens: Context text token IDs.
            context_text_tokens_lens: Lengths of the context text-token sequences.
            training_mode: Training-mode settings used to prepare context inputs.

        Returns:
            A container with prepared context tensors, cache positions and embeddings
            for the unconditional CFG branch.
        """
        bs = context_audio_codes.size(0)
        device = context_audio_codes.device

        embeds, lens, audio_codes, audio_codes_lens = self.model.prepare_context_tensors(
            context_text_tokens=context_text_tokens,
            context_text_tokens_lens=context_text_tokens_lens,
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            training_mode=training_mode,
            dropout_conditional_input=False,
        )
        full_embeds, full_lens = embeds.clone(), lens.clone()
        min_len = lens.min().item()
        cache_position = torch.arange(min_len, device=device)

        dummy_embeds_uncond = self.model.embed_text_tokens(
            torch.full((1, 1), self.model.cfg_unk_token_id, device=device),
            text_lens=torch.ones(1, dtype=torch.long, device=device),
            disable_cas_embedding=self.model.disable_cas_for_context_text,
        )
        dummy_embeds_expanded = dummy_embeds_uncond.expand(bs, embeds.size(1), -1)
        embeds = torch.cat([embeds, dummy_embeds_expanded], dim=0)

        return _InitContext(
            embeds=embeds,
            lens=lens,
            min_len=min_len,
            full_embeds=full_embeds,
            full_lens=full_lens,
            dummy_embeds_uncond=dummy_embeds_uncond,
            cache_position=cache_position,
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
        )

    def _prepare_phoneme_embeds(
        self,
        gt_phoneme_tokens: Optional[Tensor],
        gt_phoneme_tokens_lens: Optional[Tensor],
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Prepare stacked embeddings for optional ground-truth phoneme tokens.

        Args:
            gt_phoneme_tokens: Optional ground-truth phoneme token IDs.
            gt_phoneme_tokens_lens: Optional lengths of the phoneme-token sequences.

        Returns:
            A tuple with ground-truth phoneme embeddings and their stacked sequence lengths.
        """
        embeds, lens = None, None

        if gt_phoneme_tokens is not None and gt_phoneme_tokens_lens is not None:
            gt_phoneme_expanded = gt_phoneme_tokens.unsqueeze(1)
            gt_phoneme_stacked, lens = self.model.stack_codes(
                codes=gt_phoneme_expanded,
                codes_lens=gt_phoneme_tokens_lens,
                bos_id=self.phoneme_bos_id,
                eos_id=self.phoneme_eos_id,
                stacking_factor=self.phoneme_stacking_factor,
                num_codebooks=1,
            )
            embeds = self.model.embed_phoneme_tokens(gt_phoneme_stacked)

            if self.use_multiturn_dataset:
                phoneme_pad_id = getattr(self.model.phoneme_tokenizer, "pad", -1)
                phoneme_mask = gt_phoneme_stacked[:, 0, :] != phoneme_pad_id
                embeds = embeds * phoneme_mask.unsqueeze(2)

        return embeds, lens

    def _streaming_init(
        self,
        context_audio_codes: Tensor,
        context_audio_codes_lens: Tensor,
        context_text_tokens: Tensor,
        context_text_tokens_lens: Tensor,
        cfg_scale: float = 1.0,
        use_lt: bool = False,
        temperature: float = 0.7,
        topk: int = 80,
        phoneme_input_type: str = "predicted",
        phoneme_sampling_method: str = "argmax",
        gt_phoneme_tokens: Optional[Tensor] = None,
        gt_phoneme_tokens_lens: Optional[Tensor] = None,
        use_rollout_rejection: bool = False,
        rejection_weight: Optional[float] = None,
    ) -> _StreamingState:
        """Initialize the streaming state and transformer cache for a teacher rollout.

        Args:
            context_audio_codes: Context audio codec tokens.
            context_audio_codes_lens: Lengths of the context audio-code sequences.
            context_text_tokens: Context text token IDs.
            context_text_tokens_lens: Lengths of the context text-token sequences.
            cfg_scale: Classifier-free guidance scale for the rollout.
            use_lt: Whether to enable local-transformer predictions.
            temperature: Audio sampling temperature.
            topk: Number of highest-logit audio tokens retained for sampling.
            phoneme_input_type: Source of phoneme inputs during rollout.
            phoneme_sampling_method: Method used to sample predicted phoneme tokens.
            gt_phoneme_tokens: Optional ground-truth phoneme token IDs.
            gt_phoneme_tokens_lens: Optional lengths of the ground-truth phoneme sequences.
            use_rollout_rejection: Whether to track rejected rollouts.
            rejection_weight: Optional loss weight for rejected rollouts.

        Returns:
            An initialized streaming-state container.
        """
        with torch.no_grad():
            training_mode = self.model.mode_name_to_mode[self.model.default_inference_mode]

            init_context = self._prepare_init_context(
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                context_text_tokens=context_text_tokens,
                context_text_tokens_lens=context_text_tokens_lens,
                training_mode=training_mode,
            )
            transformer_out = self.model.forward(
                inputs_embeds=init_context.embeds[:, : init_context.min_len, :],
                attention_mask=None,
                use_cache=True,
                past_key_values=None,
                cache_position=init_context.cache_position,
            )
            gt_phoneme_embeds, gt_phoneme_lens = self._prepare_phoneme_embeds(
                gt_phoneme_tokens, gt_phoneme_tokens_lens
            )
            config = StreamingConfig(
                batch_size=context_audio_codes.size(0),
                device=context_audio_codes.device,
                training_mode=training_mode,
                use_cfg=True,
                cfg_scale=cfg_scale,
                use_local_transformer=use_lt,
                temperature=temperature,
                topk=topk,
                phoneme_input_type=phoneme_input_type,
                phoneme_sampling_method=phoneme_sampling_method,
                dummy_context_embedding_unconditional=init_context.dummy_embeds_uncond,
            )
            state = _StreamingState.create(
                config=config,
                last_hidden=transformer_out.last_hidden_state,
                past_kv=transformer_out.past_key_values,
                min_context_len=init_context.min_len,
                context_audio_codes=init_context.audio_codes,
                context_audio_codes_lens=init_context.audio_codes_lens,
                context_lens=init_context.lens,
                full_context_embedding=init_context.full_embeds,
                full_context_lens=init_context.full_lens,
                gt_phoneme_embeddings=gt_phoneme_embeds,
                gt_phoneme_lens=gt_phoneme_lens,
                use_rollout_rejection=use_rollout_rejection,
                rejection_weight=rejection_weight,
            )
            return state

    def _prepare_input_context_phase(
        self,
        next_input: Tensor,
        state: _StreamingState,
        status: _ChannelStatus,
    ) -> Tensor:
        """Add active context embeddings to the next streaming-model input.

        Args:
            next_input: Input embeddings for the next rollout step.
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.

        Returns:
            Next-step input embeddings with active context embeddings added.
        """
        if status.context.any():
            ctx_positions = state.context_position.clone()
            ctx_positions = ctx_positions.clamp(max=state.full_context_embedding.size(1) - 1)
            indexes = torch.arange(state.batch_size, device=state.device)
            ctx_emb = state.full_context_embedding[indexes, ctx_positions, :].unsqueeze(1)
            context_mask = status.context.view(state.batch_size, 1, 1).float()
            next_input = next_input + ctx_emb * context_mask
        return next_input

    def _prepare_input_text_phase(
        self,
        next_input: Tensor,
        state: _StreamingState,
        status: _ChannelStatus,
        text_tokens: Optional[Tensor],
    ) -> tuple[Tensor, _StreamingState]:
        """Add active text embeddings to the next streaming-model input.

        Args:
            next_input: Input embeddings for the next rollout step.
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.
            text_tokens: Optional text token IDs for the current step.

        Returns:
            A tuple containing the updated input embeddings and streaming state.
        """
        if text_tokens is not None and status.text.any():
            text_tokens_2d = text_tokens.unsqueeze(1)
            text_lens = torch.ones(state.batch_size, dtype=torch.long, device=state.device)

            text_embedded = self.model.embed_text_tokens(
                text_tokens=text_tokens_2d,
                text_lens=text_lens,
                is_multiturn=self.use_multiturn_dataset,
            )
            if self.use_multiturn_dataset:
                text_embedded[text_tokens_2d == self.model.pad_id] = 0.0

            is_eos_token = (text_tokens == self.eos_id) & status.text
            text_add_mask = status.text.view(state.batch_size, 1, 1).float()
            next_input = next_input + text_embedded * text_add_mask
            state.text_finished = state.text_finished | is_eos_token

        elif text_tokens is None:
            state.text_finished = state.text_finished | ~status.context

        return next_input, state

    def _prepare_input_phoneme_phase(
        self,
        next_input: Tensor,
        state: _StreamingState,
        status: _ChannelStatus,
    ) -> tuple[Tensor, _StreamingState]:
        """Add active phoneme embeddings to the next streaming-model input.

        Args:
            next_input: Input embeddings for the next rollout step.
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.

        Returns:
            A tuple containing the updated input embeddings and streaming state.
        """
        if self.model.phoneme_tokenizer is None:
            return next_input, state

        bs = state.batch_size
        device = state.device

        if status.phoneme.any():
            phoneme_emb = torch.zeros(bs, 1, self.embedding_dim, device=device)

            if state.config.phoneme_input_type == "gt" and state.gt_phoneme_embeddings is not None:
                within_gt_len = state.phoneme_steps < state.gt_phoneme_lens
                positions = state.phoneme_steps.clamp(max=state.gt_phoneme_embeddings.size(1) - 1)
                gt_emb = state.gt_phoneme_embeddings[torch.arange(bs, device=device), positions, :].unsqueeze(1)
                phoneme_mask = (status.phoneme & within_gt_len).view(bs, 1, 1).float()
                phoneme_emb = phoneme_emb + gt_emb * phoneme_mask
            else:
                first_phoneme_step = status.phoneme & (state.phoneme_steps == 0)
                has_last_phoneme = status.phoneme & (~first_phoneme_step) & (state.last_phoneme_tokens is not None)

                if first_phoneme_step.any():
                    size = (bs, self.phoneme_stacking_factor, 1)
                    phoneme_bos = torch.full(size, self.phoneme_bos_id, device=device).long()
                    phoneme_bos_emb = self.model.embed_phoneme_tokens(phoneme_bos)
                    first_mask = first_phoneme_step.view(bs, 1, 1).float()
                    phoneme_emb = phoneme_emb + phoneme_bos_emb * first_mask

                if has_last_phoneme.any() and state.last_phoneme_tokens is not None:
                    last_phoneme_emb = self.model.embed_phoneme_tokens(state.last_phoneme_tokens.unsqueeze(2))
                    last_mask = has_last_phoneme.view(bs, 1, 1).float()
                    phoneme_emb = phoneme_emb + last_phoneme_emb * last_mask

                state.phoneme_stream_ended = state.phoneme_stream_ended | state.phoneme_eos_detected

            next_input = next_input + phoneme_emb

        return next_input, state

    def _prepare_input_audio_phase(
        self,
        next_input: Tensor,
        state: _StreamingState,
        status: _ChannelStatus,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Add active audio embeddings to the next streaming-model input.

        Args:
            next_input: Input embeddings for the next rollout step.
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.

        Returns:
            A tuple containing the updated input embeddings and the optional audio
            embeddings used for the current step.
        """
        bs = state.batch_size
        device = state.device
        audio_emb = None

        if status.audio.any():
            audio_emb = torch.zeros(bs, 1, self.embedding_dim, device=device)
            # Use backbone codes to preserve rollout policy.
            last_audio_codes = state.last_audio_codes

            first_audio_step = status.audio & (state.audio_steps == 0)
            has_last_audio = status.audio & ~first_audio_step & (last_audio_codes is not None)

            if first_audio_step.any():
                size = (bs, self.num_audio_codebooks * self.audio_stacking_factor, 1)
                audio_bos = torch.full(size, self.audio_bos_id, device=device).long()
                audio_bos_emb = self.model.embed_audio_tokens(audio_bos)
                first_mask = first_audio_step.view(bs, 1, 1).float()
                audio_emb = audio_emb + audio_bos_emb * first_mask

            if has_last_audio.any() and last_audio_codes is not None:
                last_audio_emb = self.model.embed_audio_tokens(last_audio_codes.unsqueeze(2))
                last_mask = has_last_audio.view(bs, 1, 1).float()
                audio_emb = audio_emb + last_audio_emb * last_mask

            next_input = next_input + audio_emb

        return next_input, audio_emb

    def _prepare_input_cfg_phase(
        self,
        next_input: Tensor,
        state: _StreamingState,
        status: _ChannelStatus,
        audio_emb: Optional[Tensor],
    ) -> Tensor:
        """Append unconditional CFG inputs to the next streaming-model input.

        Args:
            next_input: Conditional input embeddings for the next rollout step.
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.
            audio_emb: Optional audio embeddings for the current step.

        Returns:
            Input embeddings containing conditional and unconditional CFG branches.
        """
        uncond_context = state.config.dummy_context_embedding_unconditional.expand(state.batch_size, 1, -1)
        uncond_zeros = torch.zeros_like(uncond_context)
        context_mask = status.context.view(state.batch_size, 1, 1).float()
        uncond_next_input = context_mask * uncond_context + (1 - context_mask) * uncond_zeros

        if status.audio.any():
            audio_mask = status.audio.view(state.batch_size, 1, 1).float()
            uncond_next_input = uncond_next_input * (1 - audio_mask) + audio_emb * audio_mask

        next_input = torch.cat([next_input, uncond_next_input], dim=0)

        return next_input

    def _prepare_streaming_input(
        self,
        state: _StreamingState,
        text_tokens: Optional[Tensor],
    ) -> tuple[Tensor, _ChannelStatus]:
        """Prepare the combined CFG input for a streaming rollout step.

        Args:
            state: Current streaming rollout state.
            text_tokens: Optional text token IDs at the current rollout position.

        Returns:
            A tuple containing the combined next-step input embeddings and the
            masks that indicate active channels for each batch item.
        """
        bs = state.batch_size
        device = state.device

        status = _create_status_from_state(state)
        next_input = torch.zeros(bs, 1, self.embedding_dim, device=device)

        next_input = self._prepare_input_context_phase(next_input, state, status)
        next_input, state = self._prepare_input_text_phase(next_input, state, status, text_tokens)
        next_input, state = self._prepare_input_phoneme_phase(next_input, state, status)
        next_input, audio_emb = self._prepare_input_audio_phase(next_input, state, status)
        next_input = self._prepare_input_cfg_phase(next_input, state, status, audio_emb)

        return next_input, status

    def _predict_phoneme_tokens(
        self,
        state: _StreamingState,
    ) -> _PhonemeTokensStep:
        """Predict phoneme tokens for the current streaming rollout step.

        Args:
            state: Current streaming rollout state.

        Returns:
            A container with the sampled phoneme tokens and their prediction logits.
        """
        bs = state.batch_size
        last_hidden = state.last_hidden
        temp = state.temperature
        topk = state.topk

        logits_t = self.model.phoneme_final_proj(last_hidden[:, -1, :])
        logits_t = logits_t[:bs]

        if state.phoneme_sampling_method == "argmax":
            tokens_t = self.model.sample_codes_from_logits_phoneme(logits_t, temperature=0.0)
        else:
            tokens_t = self.model.sample_codes_from_logits_phoneme(logits_t, temperature=temp, topk=topk)

        logits_t = logits_t.view(bs, self.phoneme_stacking_factor, self.phoneme_vocab_size)

        return _PhonemeTokensStep(tokens_t=tokens_t, logits_t=logits_t)

    def _predict_audio_codes(
        self,
        state: _StreamingState,
        use_lt: bool,
    ) -> _AudioCodesStep:
        """Predict backbone and optional local-transformer audio codes for one rollout step.

        Args:
            state: Current streaming rollout state.
            use_lt: Whether to generate local-transformer audio-code predictions.

        Returns:
            A container with sampled and greedy backbone codes, backbone logits and
            optional local-transformer predictions.
        """
        bs = state.batch_size
        last_hidden = state.last_hidden
        temp = state.temperature
        topk = state.topk
        cfg_scale = state.cfg_scale

        last_hidden_audio = self.model.audio_out_projection(last_hidden[:, -1, :])
        code_logits_t = self.model.final_proj(last_hidden_audio)

        cond_logits = code_logits_t[:bs]
        uncond_logits = code_logits_t[bs:]
        logits_t = cfg_scale * cond_logits + (1.0 - cfg_scale) * uncond_logits

        codes_t = self.model.sample_codes_from_logits(logits_t, temperature=temp, topk=topk)
        codes_t_argmax = codes_t if temp <= 0.0 else self.model.sample_codes_from_logits(logits_t, temperature=0.01)
        codes_t_lt, codes_t_argmax_lt, logits_t_lt = None, None, None

        if use_lt:
            codes_t_lt, logits_t_lt = _lt_sample_autoregressive(
                model=self.model,
                dec_output=last_hidden[:, -1, :],
                temperature=temp,
                topk=topk,
                cfg_scale=cfg_scale,
                use_kv_cache=True,
                sanitize_logits=True,
            )
            codes_t_lt = codes_t_lt.reshape(bs, -1)
            # As in the pre-training stage, we do not use greedy sampling for LT.
            codes_t_argmax_lt = codes_t_lt

        return _AudioCodesStep(
            codes_t=codes_t,
            codes_t_argmax=codes_t_argmax,
            logits_t=logits_t,
            codes_t_lt=codes_t_lt,
            codes_t_argmax_lt=codes_t_argmax_lt,
            logits_t_lt=logits_t_lt,
        )

    def _process_predictions(
        self,
        state: _StreamingState,
        status: _ChannelStatus,
    ) -> _StreamingState:
        """Update rollout state with predictions from the current streaming step.

        The method updates per-channel counters, generates active phoneme and audio
        predictions, records them in the state and tracks detected sequence endings.

        Args:
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.

        Returns:
            Updated streaming rollout state.
        """
        state.update_counters(status)

        if status.phoneme.any() and self.model.phoneme_tokenizer is not None:
            state.update_phoneme_start_idx(status.phoneme)
            phoneme_tokens = self._predict_phoneme_tokens(state)

            # Retain predicted phoneme tokens/logits for rollout debugging.
            # They are not used by the regular ground-truth phoneme training loss.
            state.update_phoneme_step(
                tokens=phoneme_tokens,
                fs_factor=self.phoneme_stacking_factor,
                vocab_size=self.phoneme_vocab_size,
            )
            state.update_phoneme_end_status(
                tokens=phoneme_tokens,
                needs_phoneme=status.phoneme,
                eos_id=self.phoneme_eos_id,
            )

        if status.audio.any():
            state.update_audio_start_idx(status.audio)
            audio_codes = self._predict_audio_codes(state, use_lt=state.use_lt)

            audio_codes.unstack(
                batch_size=state.batch_size,
                num_codebooks=self.num_audio_codebooks,
                fs_factor=self.audio_stacking_factor,
            )
            state.update_last_audio_codes(
                audio_codes=audio_codes,
                needs_audio=status.audio,
            )
            newly_ended_items = state.update_audio_end_status(
                audio_codes=audio_codes,
                needs_audio=status.audio,
                eos_id=self.audio_eos_id,
                fs_factor=self.audio_stacking_factor,
            )
            for idx in newly_ended_items:
                logging.info(f"End detected for item {idx} at streaming generation step {len(state.pred_codes) + 1}.")

            state.add_audio_codes(audio_codes)

        return state

    def _streaming_step(
        self,
        state: _StreamingState,
        text_tokens: Optional[Tensor] = None,
    ) -> _StreamingState:
        """Run one teacher-model streaming inference step.

        The method prepares the next CFG input, updates the transformer cache and
        hidden state and processes any phoneme or audio predictions.

        Args:
            state: Current streaming rollout state.
            text_tokens: Optional text token IDs at the current rollout position.

        Returns:
            Updated streaming rollout state.
        """
        if state.finished.all():
            return state

        with torch.no_grad():
            device = state.config.device

            next_input, status = self._prepare_streaming_input(state, text_tokens)
            cache_position = torch.tensor([state.cache_seq_len], device=device)

            transformer_out = self.model.forward(
                inputs_embeds=next_input,
                attention_mask=None,
                use_cache=True,
                past_key_values=state.past_key_values,
                cache_position=cache_position,
            )
            state.last_hidden = transformer_out.last_hidden_state
            state.past_key_values = transformer_out.past_key_values
            state.cache_seq_len += 1
            state = self._process_predictions(state, status)

        return state

    def _finalize_predicted_codes(
        self,
        state: _StreamingState,
    ) -> _TeacherPredictedCodes:
        """Finalize teacher-rollout predictions into padded batch tensors.

        The method extracts each item's predicted audio span, pads codes and logits to
        the batch maximum length and includes local-transformer predictions when enabled.

        Args:
            state: Streaming state containing buffered rollout predictions and their
                start and end indices.

        Returns:
            A container with finalized teacher audio codes, logits, sequence lengths
            and optional local-transformer predictions.

        Raises:
            RuntimeError: If the rollout contains no audio predictions or invalid prediction boundaries.
        """
        bs = state.batch_size
        device = state.device
        use_lt = state.use_lt
        fs_factor = self.audio_stacking_factor

        if len(state.pred_codes) == 0:
            raise RuntimeError("Teacher rollout produced no audio predictions.")

        pred_codes_lt, pred_logits_lt = None, None

        with torch.no_grad():
            codes = torch.cat(state.pred_codes, dim=-1)
            logits = torch.cat(state.pred_codes_logits, dim=1)

            if use_lt:
                codes_lt = torch.cat(state.pred_codes_lt, dim=-1)
                logits_lt = torch.cat(state.pred_codes_logits_lt, dim=1)

            raw_start_indices = state.audio_prediction_start_idx
            end_indices = torch.where(
                condition=state.audio_prediction_end_idx >= 0,
                input=state.audio_prediction_end_idx,
                other=torch.full_like(state.audio_prediction_end_idx, codes.size(-1)),
            )

            never_started = raw_start_indices < 0
            start_indices = torch.where(
                condition=never_started,
                input=end_indices,
                other=raw_start_indices,
            )
            pred_codes_lens = end_indices - start_indices

            _validate_predicted_codes(codes, pred_codes_lens, start_indices, end_indices, fs_factor)

            max_len = pred_codes_lens.max().item()
            max_stacked_len = max_len // self.audio_stacking_factor
            codes_size = (bs, self.num_audio_codebooks, max_len)
            logits_dim = self.num_audio_codebooks * self.num_audio_tokens_per_codebook * self.audio_stacking_factor
            logits_size = (bs, max_stacked_len, logits_dim)
            pred_codes = torch.zeros(*codes_size, dtype=codes.dtype, device=device)
            pred_logits = torch.zeros(*logits_size, dtype=logits.dtype, device=device)

            if use_lt:
                pred_codes_lt = torch.zeros(*codes_size, dtype=codes.dtype, device=device)
                pred_logits_lt = torch.zeros(*logits_size, dtype=logits.dtype, device=device)

            for i in range(bs):
                start, end = start_indices[i].item(), end_indices[i].item()
                start_, end_ = start // self.audio_stacking_factor, end // self.audio_stacking_factor
                length = end - start

                if length == 0:
                    continue

                pred_codes[i, :, :length] = codes[i, :, start:end]
                pred_logits[i, : end_ - start_, :] = logits[i, start_:end_, :]

                if use_lt:
                    pred_codes_lt[i, :, :length] = codes_lt[i, :, start:end]
                    pred_logits_lt[i, : end_ - start_, :] = logits_lt[i, start_:end_, :]

        return _TeacherPredictedCodes(
            codes=pred_codes,
            logits=pred_logits,
            codes_len=pred_codes_lens,
            codes_lt=pred_codes_lt,
            logits_lt=pred_logits_lt,
        )

    def _apply_rollout_upper_bound_rejection(
        self,
        state: _StreamingState,
        batch: dict[str, Tensor],
        upper_rejection_threshold: Optional[float],
        rejection_weight: Optional[float],
    ) -> _StreamingState:
        """Reject unfinished rollouts that reach their per-item upper length bound.

        Args:
            state: Current streaming rollout state.
            batch: Batch containing ground-truth audio lengths and optional metadata.
            upper_rejection_threshold: Maximum allowed generated-length fraction of the
                ground-truth audio length, or `None` to disable this rejection rule.
            rejection_weight: Optional loss weight assigned to rejected items.

        Returns:
            Streaming rollout state updated with rejected items, final audio end
            indices and optional sample weights.
        """
        gt_audio_lens = batch["audio_codes_lens"]

        if upper_rejection_threshold is None or gt_audio_lens is None:
            return state

        if len(state.pred_codes) == 0:
            return state

        # Indices in pred_codes use the global rollout-frame coordinate system.
        current_frame_count = state.current_frame_idx
        upper_thresholds = torch.round(upper_rejection_threshold * gt_audio_lens).long().to(state.device)

        # Compare the threshold with the number of frames generated for each item,
        # not with the global rollout position. Items may start audio generation at
        # different global positions because their context lengths differ.
        has_started = state.audio_prediction_start_idx >= 0
        start_indices = state.audio_prediction_start_idx.clamp_min(0)
        generated_lens = current_frame_count - start_indices

        should_reject = has_started & (~state.finished) & (generated_lens >= upper_thresholds)

        if not should_reject.any():
            return state

        state.finished = state.finished | should_reject

        if state.rejected is None:
            state.rejected = torch.zeros_like(state.finished)

        state.rejected = state.rejected | should_reject
        newly_rejected = should_reject & (state.audio_prediction_end_idx == -1)

        if newly_rejected.any():
            state.audio_prediction_end_idx = torch.where(
                condition=newly_rejected,
                input=torch.full_like(state.audio_prediction_end_idx, current_frame_count),
                other=state.audio_prediction_end_idx,
            )

        if rejection_weight is not None:
            if state.sample_weights is None:
                state.sample_weights = torch.ones(
                    state.batch_size,
                    dtype=torch.float,
                    device=state.device,
                )
            state.sample_weights = torch.where(
                condition=should_reject,
                input=torch.full_like(state.sample_weights, rejection_weight),
                other=state.sample_weights,
            )

        dataset_names = batch.get("dataset_names")
        languages = batch.get("languages")

        for item_idx in torch.nonzero(should_reject, as_tuple=False).flatten().tolist():
            dataset_name = dataset_names[item_idx] if dataset_names is not None else "Unknown"
            language = languages[item_idx] if languages is not None else "Unknown"
            logging.info(
                f"Item {item_idx} rejected by upper length bound at generation step "
                f"{len(state.pred_codes)} (current length: {current_frame_count}, "
                f"dataset: {dataset_name}, language: {language})."
            )
        return state

    def _apply_max_steps_rejection(
        self,
        state: _StreamingState,
        batch: dict[str, Tensor],
        rejection_weight: Optional[float],
    ) -> _StreamingState:
        """Reject rollouts that remain unfinished at the decoding-step limit.

        Args:
            state: Current streaming rollout state.
            batch: Batch containing optional dataset and language metadata.
            rejection_weight: Optional loss weight assigned to rejected items.

        Returns:
            Streaming rollout state updated with rejected unfinished items, final audio
            boundaries and optional sample weights.
        """
        should_reject = ~state.finished

        if not should_reject.any():
            return state

        current_frame_count = state.current_frame_idx
        has_started = state.audio_prediction_start_idx >= 0
        never_started = should_reject & (~has_started)

        # A never-started item must have zero finalized length. Using the current
        # position as both start and end prevents it from inheriting predictions
        # generated for other batch items.
        state.audio_prediction_start_idx = torch.where(
            condition=never_started,
            input=torch.full_like(state.audio_prediction_start_idx, current_frame_count),
            other=state.audio_prediction_start_idx,
        )
        state.audio_prediction_end_idx = torch.where(
            condition=should_reject & (state.audio_prediction_end_idx == -1),
            input=torch.full_like(state.audio_prediction_end_idx, current_frame_count),
            other=state.audio_prediction_end_idx,
        )
        state.finished = state.finished | should_reject

        if state.rejected is None:
            state.rejected = torch.zeros_like(state.finished)

        state.rejected = state.rejected | should_reject

        if rejection_weight is not None:
            if state.sample_weights is None:
                state.sample_weights = torch.ones(
                    state.batch_size,
                    dtype=torch.float,
                    device=state.device,
                )
            state.sample_weights = torch.where(
                condition=should_reject,
                input=torch.full_like(state.sample_weights, rejection_weight),
                other=state.sample_weights,
            )

        dataset_names = batch.get("dataset_names")
        languages = batch.get("languages")

        for item_idx in torch.nonzero(should_reject, as_tuple=False).flatten().tolist():
            dataset_name = dataset_names[item_idx] if dataset_names is not None else "Unknown"
            language = languages[item_idx] if languages is not None else "Unknown"
            logging.info(
                f"Item {item_idx} rejected after reaching max_decoder_steps, "
                f"(global frame position: {current_frame_count}, "
                f"audio generation started: {bool(has_started[item_idx].item())}, "
                f"dataset: {dataset_name}, language: {language})."
            )

        return state

    def _apply_final_lower_bound_rejection(
        self,
        teacher_output: _TeacherOutput,
        batch: dict[str, Tensor],
        lower_rejection_threshold: Optional[float],
        rejection_weight: Optional[float],
    ) -> _TeacherOutput:
        """Apply final lower-bound rejection to finalized teacher rollouts.

        Marks previously accepted rollouts as rejected when their finalized generated
        audio-code length is at or below the inclusive per-item lower bound. Optionally
        assigns the configured loss weight to newly rejected items.

        Args:
            teacher_output: Finalized teacher-rollout predictions and metadata.
            batch: Batch containing `"audio_codes_lens"` and optional sample metadata.
            lower_rejection_threshold: Fraction of each ground-truth audio-code length
                used to compute the inclusive lower rejection bound. A rollout at or
                below the rounded bound is rejected, or `None` to disable this rule.
            rejection_weight: Optional loss weight assigned to newly rejected items.

        Returns:
            Teacher output updated with lower-bound rejection flags and sample weights.

        Raises:
            KeyError: If `batch` does not contain `"audio_codes_lens"`.
        """
        dataset_names = batch.get("dataset_names")
        languages = batch.get("languages")
        gt_audio_lens = batch["audio_codes_lens"]

        if lower_rejection_threshold is None or gt_audio_lens is None:
            return teacher_output

        device = teacher_output.lens.device
        lower_thresholds = torch.round(lower_rejection_threshold * gt_audio_lens).long().to(device)
        should_reject = teacher_output.lens <= lower_thresholds

        # Do not apply lower-bound rejection to samples already rejected earlier.
        if teacher_output.rejected is not None:
            should_reject = should_reject & (~teacher_output.rejected)

        if not should_reject.any():
            return teacher_output

        if teacher_output.rejected is None:
            rejected = should_reject.clone()
        else:
            rejected = teacher_output.rejected | should_reject

        teacher_output.rejected = rejected

        if teacher_output.sample_weights is None:
            sample_weights = torch.ones_like(teacher_output.lens, dtype=torch.float, device=device)
        else:
            sample_weights = teacher_output.sample_weights.clone()

        if rejection_weight is not None:
            sample_weights = torch.where(
                condition=should_reject,
                input=torch.full_like(sample_weights, rejection_weight),
                other=sample_weights,
            )

        teacher_output.sample_weights = sample_weights

        for item_idx in torch.nonzero(should_reject, as_tuple=False).flatten().tolist():
            dataset_name = dataset_names[item_idx] if dataset_names is not None else "Unknown"
            language = languages[item_idx] if languages is not None else "Unknown"
            logging.info(
                f"Item {item_idx} rejected by lower length bound after finalization "
                f"(threshold: {lower_thresholds[item_idx].item()}, "
                f"predicted length: {teacher_output.lens[item_idx].item()}, "
                f"dataset: {dataset_name}, language: {language})."
            )
        return teacher_output

    def infer_batch(
        self,
        batch: dict[str, Tensor],
        use_lt: bool,
        max_decoder_steps: int = 500,
        temperature: float = 0.7,
        topk: int = 80,
        cfg_scale: float = 1.0,
        phoneme_input_type: str = "pred",
        phoneme_sampling_method: str = "argmax",
        lower_rejection_threshold: Optional[float] = None,
        upper_rejection_threshold: Optional[float] = None,
        rejection_weight: Optional[float] = None,
    ) -> _TeacherOutput:
        """Generate finalized teacher-rollout distillation targets for a batch.

        Performs streaming autoregressive generation from batch context and text
        inputs, applies optional rollout-length rejection, and finalizes generated
        audio-code targets and logits.

        Args:
            batch: Input batch.
            use_lt: Whether to generate local-transformer predictions.
            max_decoder_steps: Maximum number of streaming audio decoding steps.
            temperature: Audio sampling temperature.
            topk: Number of highest-logit audio tokens retained for sampling.
            cfg_scale: Classifier-free guidance scale.
            phoneme_input_type: Source used for phoneme inputs during rollout.
            phoneme_sampling_method: Method used to sample predicted phoneme tokens.
            lower_rejection_threshold: Fraction of ground-truth audio-code length used
                for final inclusive lower-bound rejection, or `None` to disable it.
            upper_rejection_threshold: Fraction of ground-truth audio-code length used
                for upper-bound rejection during rollout, or `None` to disable it.
            rejection_weight: Optional loss weight assigned to rejected rollouts.

        Returns:
            Teacher-rollout output containing finalized backbone codes, logits, and
            lengths; optional local-transformer predictions; and rejection metadata.

        Raises:
            ValueError: If local-transformer predictions are requested for a non-AR
                local transformer, or `batch` lacks `"context_audio_codes"`.
            KeyError: If `batch` lacks another required input field, or lacks
                `"audio_codes_lens"` while length-based rejection is enabled.
            RuntimeError: If finalized backbone and local-transformer outputs have
                inconsistent lengths.
        """
        if use_lt and self.model.local_transformer_type != LocalTransformerType.AR:
            raise ValueError(
                f"Only `LocalTransformerType.AR` is supported for local-transformer distillation, "
                f"but got `{self.model.local_transformer_type}`."
            )
        if "context_audio_codes" not in batch:
            raise ValueError("Missing required batch field `context_audio_codes` for teacher rollout inference.")

        text = batch["text"]
        bs = text.size(0)
        device = text.device
        text_lens = batch["text_lens"]
        use_rollout_rejection = lower_rejection_threshold is not None or upper_rejection_threshold is not None

        with torch.no_grad():
            state = self._streaming_init(
                context_audio_codes=batch["context_audio_codes"],
                context_audio_codes_lens=batch["context_audio_codes_lens"],
                context_text_tokens=batch["context_text_tokens"],
                context_text_tokens_lens=batch["context_text_tokens_lens"],
                cfg_scale=cfg_scale,
                use_lt=use_lt,
                temperature=temperature,
                topk=topk,
                phoneme_input_type=phoneme_input_type,
                phoneme_sampling_method=phoneme_sampling_method,
                gt_phoneme_tokens=batch.get("phoneme_tokens"),
                gt_phoneme_tokens_lens=batch.get("phoneme_tokens_lens"),
                use_rollout_rejection=use_rollout_rejection,
                rejection_weight=rejection_weight,
            )
            logging.info("Generation started")

            while not state.finished.all() and len(state.pred_codes) < max_decoder_steps:
                positions = state.text_tokens_seen.clamp(max=text.size(1) - 1)
                current_tokens = text[torch.arange(bs, device=device), positions]

                current_tokens = torch.where(
                    condition=state.text_tokens_seen >= text_lens,
                    input=torch.full_like(current_tokens, self.eos_id),
                    other=current_tokens,
                )
                state = self._streaming_step(state, text_tokens=current_tokens)

                if upper_rejection_threshold is not None:
                    state = self._apply_rollout_upper_bound_rejection(
                        state, batch, upper_rejection_threshold, rejection_weight
                    )

            if use_rollout_rejection and not state.finished.all():
                state = self._apply_max_steps_rejection(
                    state=state,
                    batch=batch,
                    rejection_weight=rejection_weight,
                )

            teacher_codes = self._finalize_predicted_codes(state)

            if teacher_codes.codes_lt is not None and teacher_codes.codes.size(-1) != teacher_codes.codes_lt.size(-1):
                raise RuntimeError(
                    "Mismatch between backbone and local-transformer generated code lengths: "
                    f"`teacher_codes.codes` has length {teacher_codes.codes.size(-1)}, "
                    f"while `teacher_codes.codes_lt` has length {teacher_codes.codes_lt.size(-1)}."
                )
            teacher_output = _TeacherOutput(
                codes=teacher_codes.codes,
                logits=teacher_codes.logits,
                lens=teacher_codes.codes_len,
                sample_weights=state.sample_weights,
                codes_lt=teacher_codes.codes_lt,
                logits_lt=teacher_codes.logits_lt,
                rejected=state.rejected.clone() if use_rollout_rejection else None,
            )
            if lower_rejection_threshold is not None:
                teacher_output = self._apply_final_lower_bound_rejection(
                    teacher_output, batch, lower_rejection_threshold, rejection_weight
                )

        if use_rollout_rejection:
            rejected_count = int(teacher_output.rejected.sum().item())
            logging.info(f"Total rejected samples in batch: {rejected_count}/{bs}.")

        return teacher_output


def _collect_validation_outputs(
    outputs: list[dict[str, Tensor]] | list[list[dict[str, Tensor]]],
    key: str,
) -> Optional[Tensor]:
    """Average a validation metric across flat or nested step outputs.

    Args:
        outputs: Validation-step outputs, optionally grouped by dataloader.
        key: Metric key to collect from each output dictionary.

    Returns:
        Mean value for the collected metric or `None` when no output contains
        the requested key.
    """
    values = []

    def _add(items: list[dict[str, Tensor]]) -> None:
        for item in items:
            val = item.get(key)
            if val is not None:
                values.append(val)

    if outputs and isinstance(outputs[0], list):
        for items in outputs:
            _add(items)
    else:
        _add(outputs)

    if not values:
        return None

    return torch.stack(values).mean()


def _aggregate_language_metrics(
    outputs: list[dict[str, Tensor]] | list[list[dict[str, Tensor]]],
    device: torch.device,
) -> Optional[dict[str, Tensor]]:
    """Aggregate CER and WER by language.

    Args:
        outputs: Validation-step outputs, optionally grouped by dataloader.
        device: Device for the returned metric tensors.

    Returns:
        A mapping of language-specific CER and WER metric names to their mean
        values, or `None` when no language metrics are available.
    """
    lang_wer, lang_cer, output = {}, {}, {}

    def _add(items: list[dict[str, Tensor]]):
        for item in items:
            languages = item.get(_LanguageMetricKey.languages)
            cer_list = item.get(_LanguageMetricKey.cer_list)
            wer_list = item.get(_LanguageMetricKey.wer_list)

            if languages is None or cer_list is None or wer_list is None:
                continue

            for lang, cer, wer in zip(languages, cer_list, wer_list):
                lang_cer.setdefault(lang, []).append(float(cer))
                lang_wer.setdefault(lang, []).append(float(wer))

    if outputs and isinstance(outputs[0], list):
        for items in outputs:
            _add(items)
    else:
        _add(outputs)

    for lang in lang_wer:
        key = f"{_MetricKey.cer}_lang_{lang}"
        output[key] = torch.tensor(np.mean(lang_cer[lang]), device=device)

        key = f"{_MetricKey.wer}_lang_{lang}"
        output[key] = torch.tensor(np.mean(lang_wer[lang]), device=device)

    if not output:
        return None

    return output


class _LossKey(str, Enum):
    """Enumeration of names used for logged loss values."""

    loss = "loss"
    kl_loss = "kl_loss"
    ce_loss = "ce_loss"
    nrmse_loss = "nrmse_loss"


class _LossMode(str, Enum):
    """Enumeration of model components associated with a loss."""

    main = ""
    local_transformer = "lt"
    phoneme = "phoneme"


def _get_loss_key(
    key: _LossKey,
    mode: _LossMode = _LossMode.main,
) -> str:
    """Build a logged loss key for a model component.

    Args:
        key: Base loss key.
        mode: Model component associated with the loss.

    Returns:
        Loss key.
    """
    if mode == _LossMode.main:
        return key.value
    return f"{key.value}_{mode.value}"


class _MetricKey(str, Enum):
    """Enumeration of names used for validation metrics."""

    cer = "cer"
    wer = "wer"
    ssim = "ssim"
    utmos = "utmos"
    rejection_rate = "rejection_rate"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


class _LanguageMetricKey(str, Enum):
    """Enumeration of keys used to collect per-language error metrics."""

    languages = "languages"
    cer_list = "cer_list"
    wer_list = "wer_list"


_MONITORED_KEYS_TRAIN: list[str] = [
    _get_loss_key(_LossKey.kl_loss),
    _get_loss_key(_LossKey.ce_loss),
    _get_loss_key(_LossKey.nrmse_loss),
    _get_loss_key(key=_LossKey.kl_loss, mode=_LossMode.local_transformer),
    _get_loss_key(key=_LossKey.ce_loss, mode=_LossMode.local_transformer),
    _get_loss_key(key=_LossKey.nrmse_loss, mode=_LossMode.local_transformer),
    _get_loss_key(key=_LossKey.loss, mode=_LossMode.phoneme),
    _get_loss_key(key=_LossKey.ce_loss, mode=_LossMode.phoneme),
    _MetricKey.rejection_rate.value,
]


_MONITORED_KEYS_VALID: list[str] = [
    *_MONITORED_KEYS_TRAIN,
    _MetricKey.cer.value,
    _MetricKey.wer.value,
    _MetricKey.ssim.value,
    _MetricKey.utmos.value,
]


@dataclass
class _TextAugParams:
    """Text and phoneme augmentation settings."""

    text_dropout: bool = False
    phoneme_corruption: bool = False
    phoneme_dropout: bool = False


@dataclass
class _BatchContext:
    """Prepared context embeddings, lengths and audio-code inputs."""

    embeds: Tensor
    lens: Tensor
    audio_codes: Tensor
    audio_codes_lens: Tensor


@dataclass
class _BatchText:
    """Prepared text embeddings and sequence lengths."""

    embeds: Tensor
    lens: Tensor


@dataclass
class _BatchPhonemes:
    """Prepared phoneme embeddings, tokens and sequence lengths."""

    embeds: Optional[Tensor]
    lens: Optional[Tensor]
    tokens_stacked: Optional[Tensor]
    tokens_lens_stacked: Optional[Tensor]


@dataclass
class _BatchAudio:
    """Prepared audio embeddings and ground-truth audio-code targets."""

    embeds: Tensor
    lens: Tensor
    codes_gt: Tensor
    codes_lens_gt: Tensor


@dataclass
class _PreparedStudentInput:
    """Combined student input embeddings and distillation targets."""

    embeds: Tensor
    lens: Tensor
    audio_codes_lens_gt: Tensor
    phoneme_tokens_lens_stacked: Optional[Tensor] = None
    audio_codes_gt_lt: Optional[Tensor] = None
    audio_codes_lens_gt_lt: Optional[Tensor] = None


@dataclass
class _BatchDelays:
    """Per-channel start delays for a batch."""

    text: Tensor
    phoneme: Tensor
    audio: Tensor


@dataclass
class _ContextAudio:
    """Context audio data and its sequence lengths."""

    audio: Tensor
    lens: Tensor


@dataclass
class _ValidationAudioPaths:
    """Generated and context audio file paths for validation."""

    generated: list[Path] = field(default_factory=list)
    context: list[Path] = field(default_factory=list)


@dataclass
class _ValidationSpeakerEmbeddings:
    """Generated and context speaker embeddings for validation."""

    generated: Optional[np.ndarray]
    context: Optional[np.ndarray]


def _compute_channel_delays(
    batch: dict[str, Tensor | list],
    batch_context: _BatchContext,
    training_mode: TrainingMode,
) -> _BatchDelays:
    """Compute the per-item start delay for each input channel.

    Args:
        batch: Batch containing text lengths and related input data.
        batch_context: Prepared context inputs and their sequence lengths.
        training_mode: Training-mode settings that define streaming delays.

    Returns:
        A container with text, phoneme and audio channel start delays.
    """
    text_lens = batch["text_lens"]

    text_delay = batch_context.lens.clone()
    phoneme_delay = batch_context.lens + training_mode.streaming_phonemes_delay

    if training_mode.text_input_mode == "full":
        audio_delay = batch_context.lens + text_lens + training_mode.streaming_speech_delay
    else:
        audio_delay = batch_context.lens + training_mode.streaming_speech_delay

    return _BatchDelays(text=text_delay, phoneme=phoneme_delay, audio=audio_delay)


def _pad_channel(
    embeds: Tensor,
    max_len: int,
) -> Tensor:
    """Right-pad a channel's embeddings with zeros to a target length.

    Args:
        embeds: Channel embeddings with shape `(batch, length, hidden_dim)`.
        max_len: Target sequence length.

    Returns:
        Original embeddings when already long enough, otherwise zero-padded
        embeddings with sequence length `max_len`.
    """
    bs, length, hid_dim = embeds.size()

    if length >= max_len:
        return embeds

    size = (bs, max_len - length, hid_dim)
    padding = torch.zeros(*size, device=embeds.device)
    embeds = torch.cat([embeds, padding], dim=1)

    return embeds


def _combine_channels(
    batch_context: _BatchContext,
    batch_text: _BatchText,
    batch_phonemes: _BatchPhonemes,
    batch_audio: _BatchAudio,
) -> _PreparedStudentInput:
    """Pad and combine context, text, phoneme and audio input channels.

    Args:
        batch_context: Prepared context channel.
        batch_text: Prepared text channel.
        batch_phonemes: Prepared optional phoneme channel.
        batch_audio: Prepared audio channel and ground-truth targets.

    Returns:
        A container with combined student input embeddings, sequence lengths
        and audio-code target lengths.
    """
    phonemes_len = batch_phonemes.embeds.size(1) if batch_phonemes.embeds is not None else 0
    max_len = max(batch_text.embeds.size(1), phonemes_len, batch_audio.embeds.size(1))

    context = _pad_channel(batch_context.embeds, max_len)
    text_channel = _pad_channel(batch_text.embeds, max_len)
    audio_channel = _pad_channel(batch_audio.embeds, max_len)
    combined_channel = text_channel + audio_channel

    if batch_phonemes.embeds is not None:
        phonemes_channel = _pad_channel(batch_phonemes.embeds, max_len)
        combined_channel = combined_channel + phonemes_channel

    combined_channel = context + combined_channel

    phonemes_lens = batch_phonemes.lens if batch_phonemes.embeds is not None else batch_audio.lens
    combined_lens = torch.stack([batch_text.lens, batch_audio.lens, phonemes_lens], dim=0).max(dim=0).values

    return _PreparedStudentInput(
        embeds=combined_channel,
        lens=combined_lens,
        audio_codes_lens_gt=batch_audio.codes_lens_gt,
    )


@dataclass
class _ValidationInferenceParams:
    """Configuration for validation-time inference and quality evaluation."""

    run_val_inference: bool
    use_multilingual_asr: bool
    use_utmos: bool
    whisper_model_name: str
    nemo_asr_model_name: str
    ssim_model_name: str

    @classmethod
    def create_from_config(
        cls,
        cfg: DictConfig,
    ) -> "_ValidationInferenceParams":
        """Create validation-inference parameters from a model configuration.

        Args:
            cfg: Model configuration containing validation-inference settings.

        Returns:
            Validation-inference parameter container.
        """
        obj = cls(
            run_val_inference=cfg.get("run_val_inference", False),
            use_multilingual_asr=cfg.get("use_multilingual_asr", False),
            use_utmos=cfg.get("use_utmos", False),
            whisper_model_name=cfg.get("whisper_model_name", "openai/whisper-large-v3"),
            nemo_asr_model_name=cfg.get("nemo_asr_model_name", "nvidia/parakeet-ctc-0.6b"),
            ssim_model_name=cfg.get("ssim_model_name", "titanet_large"),
        )
        return obj


class _ValidationInferenceEngine:
    """Validation-time inference and quality-evaluation engine."""

    def __init__(self, params: _ValidationInferenceParams) -> None:
        self.params = params
        self._is_active = False

        self.default_device: torch.device = torch.device("cpu")

        self.whisper_processor: Optional[WhisperProcessor] = None
        self.whisper_model: Optional[WhisperForConditionalGeneration] = None
        self.nemo_asr_model: Optional[nemo_asr.models.EncDecRNNTBPEModel] = None
        self.speaker_verification_model: Optional[nemo_asr.models.EncDecSpeakerLabelModel] = None
        self.utmos_calculator = None

        if self.params.use_utmos and not _HAVE_UTMOSV2:
            raise ImportError(
                "UTMOSv2 is required for UTMOS scoring but is not installed. "
                "Install it with: `pip install git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1`."
            )

    @property
    def is_active(self) -> bool:
        """Get whether validation inference is initialized and active."""
        return self._is_active

    @property
    def use_multilingual_asr(self) -> bool:
        """Get whether validation uses the multilingual ASR model."""
        return self.params.use_multilingual_asr

    def _setup_asr_model(self) -> None:
        """Initialize the ASR model used for validation transcription."""
        if self.use_multilingual_asr:
            self.whisper_processor = WhisperProcessor.from_pretrained(
                pretrained_model_name_or_path=self.params.whisper_model_name
            )
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.params.whisper_model_name
            )
            self.whisper_model.eval()
            self.whisper_model.to(self.default_device)
            self.nemo_asr_model = None

            for p in self.whisper_model.parameters():
                p.requires_grad = False

        else:
            self.nemo_asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name=self.params.nemo_asr_model_name
            )
            self.nemo_asr_model.freeze()
            self.nemo_asr_model.to(self.default_device)
            self.whisper_processor = None
            self.whisper_model = None

    def _setup_ssim_model(self) -> None:
        """Initialize the speaker-verification model used for similarity scoring."""
        self.speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=self.params.ssim_model_name
        )
        self.speaker_verification_model.freeze()
        self.speaker_verification_model.to(self.default_device)

    def _setup_utmos_model(self) -> None:
        """Initialize the UTMOS calculator used for audio-quality scoring."""
        self.utmos_calculator = UTMOSv2Calculator(device="cpu")

    def setup(self) -> None:
        """Initialize validation inference and quality-evaluation models when enabled."""
        if not self.params.run_val_inference:
            return

        logging.info("Loading eval models for validation inference...")
        self._setup_asr_model()
        logging.info("ASR model initialized.")
        self._setup_ssim_model()
        logging.info("Speaker verification model initialized.")

        if self.params.use_utmos:
            self._setup_utmos_model()
            logging.info("UTMOSv2 calculator initialized.")

        logging.info("Eval models loaded successfully.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_active = True

    def switch_device(self, device: torch.device | str) -> None:
        """Move loaded validation models to a target device.

        Args:
            device: Target device for the loaded ASR and speaker-verification models.
        """
        target = torch.device(device)

        if self.whisper_model is not None:
            self.whisper_model.to(target)

        if self.nemo_asr_model is not None:
            self.nemo_asr_model.to(target)

        if self.speaker_verification_model is not None:
            self.speaker_verification_model.to(target)

        if target.type == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _transcribe_using_whisper(
        self,
        paths: list[Path],
        languages: list[str],
    ) -> list[str | None]:
        """Transcribe audio files with the multilingual Whisper model.

        Args:
            paths: Audio file paths to transcribe.
            languages: Language codes corresponding to the audio files.

        Returns:
            Normalized transcripts, with `None` for files that could not be transcribed.
        """
        if self.whisper_processor is None:
            return [None] * len(paths)

        device = self.whisper_model.device

        try:
            transcripts = transcribe_with_whisper_from_filepaths(
                audio_filepaths=paths,
                language=languages,
                whisper_processor=self.whisper_processor,
                whisper_model=self.whisper_model,
                device=device,
                normalizer=None,
            )
            transcripts = [process_text_for_cer(x) for x in transcripts]

        except Exception as e:
            logging.warning(f"Batched transcription failed, falling back to per-file mode: {e}.")

            transcripts = []

            for i, p in enumerate(paths):
                lang = languages[i] if i < len(languages) else "en"

                try:
                    transcript = transcribe_with_whisper(
                        audio_filepath=p,
                        language=lang,
                        whisper_processor=self.whisper_processor,
                        whisper_model=self.whisper_model,
                        device=device,
                        normalizer=None,
                    )
                    transcripts.append(process_text_for_cer(transcript))

                except Exception as inner_e:
                    logging.warning(f"Transcription failed for {p}: {inner_e}.")
                    transcripts.append(None)

        return transcripts

    def _transcribe_using_nemo(
        self,
        paths: list[Path],
    ) -> list[str | None]:
        """Transcribe audio files with the configured NeMo ASR model.

        Args:
            paths: Audio file paths to transcribe.

        Returns:
            Normalized transcripts, or `None` values when the ASR model is not available.
        """
        if self.nemo_asr_model is None:
            return [None] * len(paths)

        config = TranscribeConfig(
            use_lhotse=False,
            batch_size=len(paths),
            num_workers=0,
        )
        transcripts = self.nemo_asr_model.transcribe(
            audio=paths,
            batch_size=len(paths),
            override_config=config,
        )
        transcripts = [process_text_for_cer(t.text) for t in transcripts]

        return transcripts

    @torch.no_grad()
    def transcribe_files(
        self,
        paths: list[Path],
        languages: Optional[list[str]],
    ) -> list[str | None]:
        """Transcribe audio files with the configured validation ASR backend.

        Args:
            paths: Audio file paths to transcribe.
            languages: Optional language codes corresponding to the audio files.
                English is used when no languages are supplied.

        Returns:
            Normalized transcripts, with `None` for files that could not be transcribed.
        """
        if self.use_multilingual_asr:
            if languages is None:
                languages = ["en"] * len(paths)
            transcripts = self._transcribe_using_whisper(paths, languages)
        else:
            transcripts = self._transcribe_using_nemo(paths)

        return transcripts

    @torch.no_grad()
    def verify_speakers(self, paths: _ValidationAudioPaths):
        """Extract speaker embeddings for generated and context audio.

        Args:
            paths: Generated and context audio file paths.

        Returns:
            A container with generated and context speaker embeddings or `None`
            values when speaker verification is unavailable or fails.
        """
        generated_embeds, context_embeds = None, None

        if self.speaker_verification_model is None:
            return _ValidationSpeakerEmbeddings(generated=generated_embeds, context=context_embeds)

        device = self.speaker_verification_model.device

        try:
            generated_embeds = get_speaker_embeddings_from_filepaths(
                filepaths=paths.generated,
                speaker_verification_model=self.speaker_verification_model,
                device=device,
            )
            generated_embeds = generated_embeds.cpu().float().numpy()

            context_embeds = get_speaker_embeddings_from_filepaths(
                filepaths=paths.context,
                speaker_verification_model=self.speaker_verification_model,
                device=device,
            )
            context_embeds = context_embeds.cpu().float().numpy()

        except Exception as e:
            logging.warning(f"Speaker verification failed: {e}.")

        return _ValidationSpeakerEmbeddings(generated=generated_embeds, context=context_embeds)

    @torch.no_grad()
    def get_utmos_scores(
        self,
        input_dir: Path,
        filenames: list[str],
    ) -> Optional[list[float]]:
        """Calculate UTMOS quality scores for generated audio files.

        Args:
            input_dir: Directory containing the audio files.
            filenames: Audio filenames to score.

        Returns:
            UTMOS scores in filename order or `None` when UTMOS scoring is not configured.

        Raises:
            RuntimeError: If UTMOS scoring fails.
        """
        scores = None

        if self.utmos_calculator is None:
            return scores

        try:
            res = self.utmos_calculator.process_directory(
                input_dir=input_dir.as_posix(),
                batch_size=len(filenames),
                num_workers=0,
                val_list=filenames,
            )
            scores = [float(item["predicted_mos"]) for item in res]

        except Exception as e:
            raise RuntimeError(f"UTMOSv2 scoring failed: {e}.") from e

        return scores


def _compute_ssim_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine similarity between two speaker-embedding vectors.

    Args:
        x: First speaker-embedding vector.
        y: Second speaker-embedding vector.

    Returns:
        Cosine similarity, or `0.0` when either embedding has zero norm.
    """
    norm = np.linalg.norm(x) * np.linalg.norm(y)

    if norm == 0.0:
        return 0.0

    score = float(np.dot(x, y) / norm)
    return score


def _compute_validation_metrics(
    batch: dict[str, Tensor | list],
    transcripts: list[str | None],
    speaker_embeds: _ValidationSpeakerEmbeddings,
    utmos_scores: Optional[list[float]],
    device: torch.device,
    use_multilingual_asr: bool,
) -> dict[str, Tensor | list[float] | list[str]]:
    """Compute validation metrics from generated audio and reference data.

    Args:
        batch: Validation batch containing reference transcripts and optional
            language metadata.
        transcripts: Generated-audio transcripts, with `None` for failed
            transcriptions.
        speaker_embeds: Generated and context speaker embeddings.
        utmos_scores: Optional UTMOS scores for generated audio.
        device: Device for aggregate metric tensors.
        use_multilingual_asr: Whether to include per-language CER and WER values.

    Returns:
        Mapping containing available aggregate CER, WER, speaker-similarity and
        UTMOS metrics, plus per-language error values when requested.
    """
    output = {}
    bs = len(transcripts)
    batch_cer, batch_wer, batch_ssim, batch_utmos = [], [], [], []

    for i in range(bs):
        if transcripts[i] is None:
            continue

        gt_transcript = process_text_for_cer(batch["raw_texts"][i])
        cer = min(word_error_rate([transcripts[i]], [gt_transcript], use_cer=True), 1.0)
        wer = min(word_error_rate([transcripts[i]], [gt_transcript], use_cer=False), 1.0)
        utmos = None if utmos_scores is None else float(utmos_scores[i])
        ssim = None

        if speaker_embeds.generated is not None and speaker_embeds.context is not None:
            ssim = _compute_ssim_score(speaker_embeds.generated[i], speaker_embeds.context[i])

        batch_cer.append(cer)
        batch_wer.append(wer)

        if ssim is not None:
            batch_ssim.append(ssim)

        if utmos is not None:
            batch_utmos.append(utmos)

    if batch_cer and batch_wer:
        output[_MetricKey.cer] = torch.tensor(np.mean(batch_cer), device=device)
        output[_MetricKey.wer] = torch.tensor(np.mean(batch_wer), device=device)

        if use_multilingual_asr:
            langs = batch.get("languages", ["en"] * bs)
            output[_LanguageMetricKey.languages] = [langs[i] for i in range(bs) if transcripts[i] is not None]
            output[_LanguageMetricKey.cer_list] = batch_cer
            output[_LanguageMetricKey.wer_list] = batch_wer

    if batch_ssim:
        output[_MetricKey.ssim] = torch.tensor(np.mean(batch_ssim), device=device)

    if batch_utmos:
        output[_MetricKey.utmos] = torch.tensor(np.mean(batch_utmos), device=device)

    return output


def _compute_rejection_rate(
    teacher_output: _TeacherOutput,
) -> Tensor:
    """Compute the fraction of teacher rollouts rejected in a batch.

    Args:
        teacher_output: Finalized teacher-rollout predictions and rejection flags.

    Returns:
        Rejected-rollout fraction as a scalar tensor.
    """
    rejected_count = 0
    sample_count = teacher_output.lens.size(0)
    device = teacher_output.codes.device

    if teacher_output.rejected is not None:
        rejected_count = int(teacher_output.rejected.sum().item())

    rate = torch.tensor(rejected_count / sample_count, device=device, dtype=torch.float)

    return rate


def _disable_default_val_inference(cfg: DictConfig) -> DictConfig:
    """Disable inherited validation-inference settings in a configuration.

    Args:
        cfg: Configuration to update.

    Returns:
        Configuration with default validation inference and UTMOS scoring disabled.
    """
    if cfg.get("run_val_inference", False):
        cfg["run_val_inference"] = False

    if cfg.get("use_utmos", False):
        cfg["use_utmos"] = False

    return cfg


def _validate_tts_tasks(batch: dict[str, Tensor | list]) -> None:
    """Validate that a batch contains only ordinary single-turn TTS cuts.

    CFG distillation supports `use_multiturn_dataset=True` for compatibility
    with multi-turn model checkpoints, but does not support multi-turn or dialog cuts.

    Args:
        batch: Collated batch containing a `task` entry with one task label per sample.

    Raises:
        ValueError: If any sample task is not `tts`.
    """
    if "task" not in batch:
        return

    unsupported_tasks = set()

    for task in batch["task"]:
        task = str(task).lower()
        if task != "tts":
            unsupported_tasks.add(task)

    if unsupported_tasks:
        raise ValueError(
            "CFG distillation does not support multi-turn/dialog cuts. "
            f"All batch tasks must be 'tts'; found: {', '.join(unsupported_tasks)}."
        )


class EasyMagpieCFGDistillation(EasyMagpieTTSModel):
    """Implements online classifier-free guidance (CFG) distillation for EasyMagpieTTS."""

    def __init__(
        self,
        cfg: DictConfig,
        trainer: "Trainer" = None,
    ) -> None:
        _validate_configuration(cfg)
        val_inference_params = _ValidationInferenceParams.create_from_config(cfg)
        cfg = _disable_default_val_inference(cfg)

        super().__init__(cfg, trainer)
        self._init_extra_attributes()
        self._init_losses()

        self.audio_dir: Optional[Path] = None
        self._teacher_model: Optional[EasyMagpieTTSModel] = None
        self._teacher_inference_engine: Optional[_TeacherInferenceEngine] = None
        self._val_inference_engine = _ValidationInferenceEngine(params=val_inference_params)

        self.rejection_debug_dir: Optional[Path] = None
        self.rejection_debug_meta_path: Optional[Path] = None
        self.rejection_debug_meta_fp: Optional[TextIOWrapper] = None
        self.debug_meta_flush_interval_steps: int = 1

    @property
    def use_rollout_rejection(self) -> bool:
        """Return whether either rollout rejection bound is enabled.

        Returns:
            True when a lower or upper rejection threshold is configured.
        """
        return self.lower_rejection_threshold is not None or self.upper_rejection_threshold is not None

    @rank_zero_only
    def maybe_init_from_pretrained_checkpoint(
        self,
        cfg: OmegaConf,
        map_location: str = "cpu",
    ) -> None:
        """See the ModelPT class docstring."""
        args = ["init_from_nemo_model", "init_from_ptl_ckpt"]
        arg_matches = [(1 if arg in cfg and cfg[arg] is not None else 0) for arg in args]

        if sum(arg_matches) == 0:
            return

        if sum(arg_matches) > 1:
            raise ValueError(
                f"Cannot pass more than one model initialization arguments to config!\n"
                f"Found : {[args[idx] for idx, arg_present in enumerate(arg_matches) if arg_present]}"
            )

        CallbackGroup.get_instance().on_load_checkpoint_start()

        if "init_from_nemo_model" in cfg and cfg.init_from_nemo_model is not None:
            model_path = cfg.init_from_nemo_model
            restore_cfg = copy.deepcopy(self.cfg)

            with open_dict(restore_cfg):
                restore_cfg.train_ds = None
                restore_cfg.validation_ds = None

            if isinstance(model_path, str):
                restored_model = EasyMagpieTTSModel.restore_from(
                    restore_path=model_path,
                    override_config_path=restore_cfg,
                    map_location=map_location,
                    strict=cfg.get("init_strict", True),
                )
                self.load_state_dict(restored_model.state_dict(), strict=False)
                logging.info(f'Model checkpoint restored from nemo file with path : `{model_path}`')
                del restored_model
            else:
                raise TypeError("Invalid type: init_from_nemo_model is not a string!")

        elif "init_from_ptl_ckpt" in cfg and cfg.init_from_ptl_ckpt is not None:
            with open_dict(cfg):
                if isinstance(cfg.init_from_ptl_ckpt, str):
                    ckpt_path = cfg.get("init_from_ptl_ckpt")
                    ckpt = torch.load(ckpt_path, map_location=map_location)
                    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                    self.load_state_dict(state_dict, strict=False)
                    logging.info(
                        f'Model checkpoint restored from pytorch lightning checkpoint with path : `{ckpt_path}`'
                    )
                    del ckpt
                else:
                    raise TypeError("Invalid type: init_from_ptl_ckpt is not a string!")

        CallbackGroup.get_instance().on_load_checkpoint_end()

    def _init_extra_attributes(self) -> None:
        """Initialize distillation attributes from configuration values and defaults."""
        defaults = vars(_DEFAULT_PARAMS)
        for k, v in defaults.items():
            setattr(self, k, self.cfg.get(k, v))

    @property
    def num_lt_codebooks(self) -> int:
        """Return the number of codebooks predicted by the local transformer.

        Returns:
            Number of audio codebooks after frame stacking.
        """
        return self.num_audio_codebooks * self.frame_stacking_factor

    def _init_losses(self) -> None:
        """Initialize enabled audio and phoneme distillation loss criteria."""
        if self.audio_kl_loss_weight > 0.0:
            self._codes_kl_criterion = KLDivergenceLoss(
                num_codebooks=self.num_audio_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=self.frame_stacking_factor,
                codebook_ordering="codebook-major",
            )
            self._lt_codes_kl_criterion = KLDivergenceLoss(
                num_codebooks=self.num_lt_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=1,
                codebook_ordering="codebook-major",
            )
        if self.audio_ce_loss_weight > 0.0:
            self._codes_ce_criterion = CodesCrossEntropyLoss(
                num_codebooks=self.num_audio_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=self.frame_stacking_factor,
                codebook_ordering="codebook-major",
            )
            self._lt_codes_ce_criterion = CodesCrossEntropyLoss(
                num_codebooks=self.num_lt_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=1,
                codebook_ordering="codebook-major",
            )
        if self.audio_nrmse_loss_weight > 0.0:
            self._codes_nrmse_criterion = NRMSELogitsLoss(
                num_codebooks=self.num_audio_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=self.frame_stacking_factor,
                codebook_ordering="codebook-major",
            )
            self._lt_codes_nrmse_criterion = NRMSELogitsLoss(
                num_codebooks=self.num_lt_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=1,
                codebook_ordering="codebook-major",
            )
        if self.phoneme_ce_loss_weight:
            self._phonemes_ce_criterion = CodesCrossEntropyLoss(
                num_codebooks=1,
                num_tokens_per_codebook=self.phoneme_vocab_size,
                frame_stacking_factor=self.phoneme_stacking_factor,
                codebook_ordering="codebook-major",
            )

    def _get_audio_log_dir(self) -> Path:
        """Create and return the validation-audio log directory.

        Returns:
            Path to the validation-audio log directory.
        """
        audio_dir = Path(self.trainer.log_dir) / "val_audios"
        audio_dir.mkdir(parents=True, exist_ok=True)
        return audio_dir

    def _get_rejection_debug_dir(self) -> Path:
        """Create and return the rollout-rejection debug directory.

        Returns:
            Path to the rollout-rejection debug directory.
        """
        debug_dir = Path(self.trainer.log_dir) / "rejection_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir

    def _load_teacher_model(self) -> None:
        """Load the frozen teacher model and initialize its inference engine."""
        if self._teacher_model is None:
            logging.info("Loading teacher model from checkpoint.")
            self._teacher_model = _get_teacher_model(self.cfg).to(self.device)
            self._teacher_inference_engine = _TeacherInferenceEngine(model=self._teacher_model)
            logging.info("Teacher model loaded and frozen.")

    def _configure_rejection_debug_attributes(self) -> None:
        """Initialize paths and metadata output for rollout-rejection debugging."""
        self.rejection_debug_dir = self._get_rejection_debug_dir()

        filename = f"rank_{self.global_rank}_metadata.jsonl"
        self.rejection_debug_meta_path = self.rejection_debug_dir / filename

        self.rejection_debug_meta_fp = open(
            file=self.rejection_debug_meta_path,
            mode="a",
            encoding="utf-8",
            buffering=self.debug_meta_flush_interval_steps,
        )

    def on_fit_start(self) -> None:
        """Initialize training-time resources and the frozen teacher model."""
        super().on_fit_start()
        self.audio_dir = self._get_audio_log_dir()

        if self.use_rejection_debug_mode:
            self._configure_rejection_debug_attributes()

        self._load_teacher_model()
        self._val_inference_engine.setup()

    def on_fit_end(self) -> None:
        """Close and flush rejection-debug metadata resources."""
        if self.rejection_debug_meta_fp is not None:
            self.rejection_debug_meta_fp.flush()
            self.rejection_debug_meta_fp.close()
            self.rejection_debug_meta_fp = None

    def on_validation_start(self) -> None:
        """Move active validation inference helpers to the model device."""
        super().on_validation_start()

        if self._val_inference_engine.is_active and torch.cuda.is_available():
            self._val_inference_engine.switch_device(device=self.device)

    def on_validation_end(self) -> None:
        """Return active validation inference helpers to CPU memory."""
        super().on_validation_end()

        if self._val_inference_engine.is_active:
            self._val_inference_engine.switch_device("cpu")

    def _get_state_dict_keys_to_exclude(self) -> list[str]:
        """Get state-dictionary keys excluded when saving this model.

        Returns:
            State-dictionary key names inherited from the base model plus
            distillation-only keys that must not be serialized.
        """
        return super()._get_state_dict_keys_to_exclude() + _STATE_DICT_EXCLUDE_NAMES

    def _process_audio_input(
        self,
        audio_codes: Tensor,
        audio_codes_lens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepare shifted stacked audio-code inputs and targets for the student.

        The method prepends an audio BOS token, stacks codec frames and separates the
        result into teacher-forced inputs and next-step ground-truth targets.

        Args:
            audio_codes: Teacher-generated audio codec tokens containing EOS tokens.
            audio_codes_lens: Lengths of the teacher-generated audio-code sequences.

        Returns:
            A tuple containing stacked audio-code inputs, ground-truth target codes
            and target sequence lengths.
        """
        # Add only BOS because teacher codes already contain EOS.
        audio_codes = torch.nn.functional.pad(input=audio_codes, pad=(1, 0), value=self.audio_bos_id)
        audio_codes_lens = audio_codes_lens + 1

        audio_codes, audio_codes_lens = self.stack_codes(
            codes=audio_codes,
            codes_lens=audio_codes_lens,
            bos_id=self.audio_bos_id,
            eos_id=self.audio_eos_id,
            stacking_factor=self.frame_stacking_factor,
            num_codebooks=self.num_audio_codebooks,
        )
        codes_lens_gt = audio_codes_lens - 1
        codes_gt = audio_codes[:, :, 1:]
        codes_input = audio_codes[:, :, :-1]

        return codes_input, codes_gt, codes_lens_gt

    def prepare_audio_channel_embeddings(
        self,
        audio_codes: Tensor,
        audio_codes_lens: Tensor,
        delay: Tensor,
        speech_eos_mask: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """See the EasyMagpieTTSModel class docstring."""
        batch_size = audio_codes.size(0)
        device = audio_codes.device
        max_delay = delay.max().item()

        embedded = self.embed_audio_tokens(audio_codes)
        size = (batch_size, max_delay, self.cfg.embedding_dim)
        zero_delay_tensor = torch.zeros(*size, device=device)

        embedding, lens = self.join_embeddings_temporally(
            embeddings=[zero_delay_tensor, embedded],
            lengths=[delay, audio_codes_lens],
        )
        return embedding, lens

    def _configure_training_mode(
        self,
        mode: str,
        training_mode: Optional[TrainingMode],
    ) -> TrainingMode:
        """Select the training mode for a training or validation step.

        Args:
            mode: Current execution mode, such as `"train"` or validation.
            training_mode: Explicit training mode, if one was provided.

        Returns:
            Provided training mode, a randomly selected training mode during training
            or the default mode during validation.
        """
        if training_mode is None:
            if mode == "train":
                training_mode = random.choice(self.training_modes)
            else:
                training_mode = self.training_modes[0]

        return training_mode

    def _configure_text_aug_params(
        self,
        mode: str,
    ) -> _TextAugParams:
        """Configure text and phoneme augmentation for the current step.

        Args:
            mode: Current execution mode, such as `"train"` or validation.

        Returns:
            Text and phoneme augmentation settings. Augmentations are
            disabled outside training mode.
        """
        params = _TextAugParams()

        if mode != "train":
            return params

        params.text_dropout = random.random() < self.dropout_text_input_prob

        if not params.text_dropout and self.phoneme_corruption_type == "repeat_skip_unk":
            params.phoneme_corruption = True

        if self.phoneme_corruption_type == "complete_channel":
            params.phoneme_dropout = torch.rand(1).item() < self.phoneme_corruption_batch_prob

        return params

    def _prepare_context(
        self,
        batch: dict[str, Tensor | list],
        training_mode: TrainingMode,
    ) -> _BatchContext:
        """Prepare context embeddings and audio-code inputs for a batch.

        Args:
            batch: Batch containing context text and audio-code tensors.
            training_mode: Settings controlling context preparation.

        Returns:
            Container with prepared context embeddings, lengths and audio codes.
        """
        embeds, lens, audio_codes, audio_codes_lens = self.prepare_context_tensors(
            context_text_tokens=batch["context_text_tokens"],
            context_text_tokens_lens=batch["context_text_tokens_lens"],
            context_audio_codes=batch["context_audio_codes"],
            context_audio_codes_lens=batch["context_audio_codes_lens"],
            training_mode=training_mode,
            dropout_conditional_input=False,
        )
        return _BatchContext(
            embeds=embeds,
            lens=lens,
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
        )

    def _prepare_text(
        self,
        batch: dict[str, Tensor | list],
        delay: Tensor,
        dropout_text_input: bool,
    ) -> _BatchText:
        """Prepare delayed text-channel embeddings for a batch.

        Args:
            batch: Batch containing text tokens and lengths.
            delay: Per-item text-channel start delays.
            dropout_text_input: Whether to drop text conditioning for this batch.

        Returns:
            Container with prepared text embeddings and sequence lengths.
        """
        embeds, lens = self.prepare_text_channel_embeddings(
            text=batch["text"],
            text_lens=batch["text_lens"],
            delay=delay,
            dropout_text_input=dropout_text_input,
            is_multiturn=self.cfg.get("use_multiturn_dataset", False),
            text_pad_id=self.pad_id,
        )
        return _BatchText(embeds=embeds, lens=lens)

    def _prepare_phoneme(
        self,
        batch: dict[str, Tensor | list],
        delay: Tensor,
        corrupt_phonemes: bool,
        dropout_channel: bool,
    ) -> _BatchPhonemes:
        """Prepare optional delayed phoneme-channel embeddings for a batch.

        Args:
            batch: Batch containing optional phoneme tokens and lengths.
            delay: Per-item phoneme-channel start delays.
            corrupt_phonemes: Whether to apply phoneme corruption.
            dropout_channel: Whether to drop the complete phoneme channel.

        Returns:
            Container with optional phoneme embeddings, stacked tokens and lengths.
        """
        phoneme_tokens = batch.get("phoneme_tokens")
        embeds = None
        lens = None
        tokens_stacked = None
        tokens_lens_stacked = None

        if self.phoneme_tokenizer is not None and phoneme_tokens is not None:
            embeds, lens, tokens_stacked, tokens_lens_stacked, _, _ = self.prepare_phoneme_channel_embeddings(
                phoneme_tokens=phoneme_tokens,
                phoneme_tokens_lens=batch["phoneme_tokens_lens"],
                delay=delay,
                apply_corruption=corrupt_phonemes,
                dropout_complete_phoneme_channel=dropout_channel,
            )
        return _BatchPhonemes(
            embeds=embeds,
            lens=lens,
            tokens_stacked=tokens_stacked,
            tokens_lens_stacked=tokens_lens_stacked,
        )

    def _prepare_audio(
        self,
        batch: dict[str, Tensor | list],
        delay: Tensor,
    ) -> _BatchAudio:
        """Prepare delayed audio embeddings and teacher-code targets for a batch.

        Args:
            batch: Batch containing teacher-generated audio codes and lengths.
            delay: Per-item audio-channel start delays.

        Returns:
            Container with prepared audio embeddings and ground-truth code targets.
        """
        codes_input, codes_gt, codes_lens_gt = self._process_audio_input(
            audio_codes=batch["audio_codes_teacher"],
            audio_codes_lens=batch["audio_codes_lens_teacher"],
        )
        embeds, lens = self.prepare_audio_channel_embeddings(
            audio_codes=codes_input,
            audio_codes_lens=codes_lens_gt,
            delay=delay,
        )
        return _BatchAudio(
            embeds=embeds,
            lens=lens,
            codes_gt=codes_gt,
            codes_lens_gt=codes_lens_gt,
        )

    def _prepare_student_input(
        self,
        batch: dict[str, Tensor | list],
        use_lt: bool,
        mode: str = "train",
        training_mode: Optional[TrainingMode] = None,
    ) -> tuple[_PreparedStudentInput, _BatchDelays]:
        """Prepare combined student inputs and distillation targets for a batch.

        Args:
            batch: Batch containing context, text, phoneme, and teacher-audio data.
            use_lt: Whether to prepare local-transformer audio-code targets.
            mode: Current execution mode.
            training_mode: Optional explicit streaming training mode.

        Returns:
            A tuple containing the combined student-input container and the
            per-channel start-delay container.
        """
        training_mode = self._configure_training_mode(mode, training_mode)
        aug_params = self._configure_text_aug_params(mode)

        batch_context = self._prepare_context(batch, training_mode)
        batch_delays = _compute_channel_delays(batch, batch_context, training_mode)

        batch_text = self._prepare_text(
            batch=batch,
            delay=batch_delays.text,
            dropout_text_input=aug_params.text_dropout,
        )
        batch_phonemes = self._prepare_phoneme(
            batch=batch,
            delay=batch_delays.phoneme,
            corrupt_phonemes=aug_params.phoneme_corruption,
            dropout_channel=aug_params.phoneme_dropout,
        )
        batch_audio = self._prepare_audio(batch, delay=batch_delays.audio)
        batch_combined = _combine_channels(batch_context, batch_text, batch_phonemes, batch_audio)

        if self.phoneme_tokenizer is not None and batch_phonemes.tokens_stacked is not None:
            batch_combined.phoneme_tokens_lens_stacked = batch_phonemes.tokens_lens_stacked

        if use_lt:
            _, audio_codes_gt_lt, audio_codes_lens_gt_lt = self._process_audio_input(
                audio_codes=batch["audio_codes_lt_teacher"],
                audio_codes_lens=batch["audio_codes_lens_teacher"],
            )
            batch_combined.audio_codes_gt_lt = audio_codes_gt_lt
            batch_combined.audio_codes_lens_gt_lt = audio_codes_lens_gt_lt

        return batch_combined, batch_delays

    def _process_batch(
        self,
        batch: dict[str, Tensor | list],
        use_lt: bool,
        mode: str = "train",
        training_mode: Optional[TrainingMode] = None,
    ) -> _StudentOutput:
        """Prepare a distillation batch and run the student forward pass.

        The method combines the context, text, phoneme and teacher-audio channels,
        executes the student backbone forward, and extracts logits aligned with the audio,
        optional local-transformer and phoneme targets.

        Args:
            batch: Batch containing context, text, phoneme, and teacher-audio data.
            use_lt: Whether to compute local-transformer logits.
            mode: Current execution mode.
            training_mode: Optional explicit streaming training mode.

        Returns:
            Student outputs containing backbone logits, optional local-transformer and
            phoneme logits and local-transformer target data.

        Raises:
            ValueError: If local-transformer distillation is requested for an
                unsupported local-transformer type.
        """
        if use_lt and self.local_transformer_type != LocalTransformerType.AR:
            raise ValueError(
                f"Only `LocalTransformerType.AR` is supported for local-transformer distillation, "
                f"but got `{self.local_transformer_type}`."
            )
        logits_lt = None
        logits_phonemes = None
        batch_combined, batch_delay = self._prepare_student_input(batch, use_lt, mode, training_mode)

        transformer_out = self.forward(
            inputs_embeds=batch_combined.embeds,
            attention_mask=get_mask_from_lengths(batch_combined.lens),
        )
        pred_embeddings = self.slice_sequence_embeddings(
            sequence_embeddings=transformer_out.last_hidden_state,
            context_lens=batch_delay.audio,
            target_lens=batch_combined.audio_codes_lens_gt,
        )
        pred_embeddings_audio = self.audio_out_projection(pred_embeddings)
        logits = self.final_proj(pred_embeddings_audio)

        if use_lt:
            logits_lt = self._lt_helper.compute_logits(
                dec_out=pred_embeddings.detach(),
                audio_codes_target=batch_combined.audio_codes_gt_lt,
                targets_offset_by_one=False,
            )
        if self.phoneme_tokenizer is not None and batch_combined.phoneme_tokens_lens_stacked is not None:
            pred_embeddings_phoneme = self.slice_sequence_embeddings(
                sequence_embeddings=transformer_out.last_hidden_state,
                context_lens=batch_delay.phoneme,
                target_lens=batch_combined.phoneme_tokens_lens_stacked - 1,
            )
            logits_phonemes = self.phoneme_final_proj(pred_embeddings_phoneme)

        return _StudentOutput(
            logits=logits,
            logits_lt=logits_lt,
            logits_phonemes=logits_phonemes,
            audio_codes_gt_lt=batch_combined.audio_codes_gt_lt,
            audio_codes_lens_gt_lt=batch_combined.audio_codes_lens_gt_lt,
        )

    def _update_batch(
        self,
        batch: dict[str, Tensor | list],
        teacher_output: _TeacherOutput,
        use_lt: bool,
    ) -> dict[str, Tensor | list]:
        """Add teacher-rollout audio targets to a batch in place.

        The method stores backbone teacher codes and lengths, and adds
        local-transformer teacher codes when local-transformer distillation is active.

        Args:
            batch: Batch to update with teacher-rollout targets.
            teacher_output: Finalized teacher-rollout predictions.
            use_lt: Whether local-transformer distillation is active.

        Returns:
            Updated batch containing teacher audio-code targets.

        Raises:
            ValueError: If local-transformer targets are required but unavailable.
        """
        batch["audio_codes_teacher"] = teacher_output.codes
        batch["audio_codes_lens_teacher"] = teacher_output.lens

        if use_lt:
            if teacher_output.codes_lt is None:
                raise ValueError(
                    "Local-transformer distillation is enabled for this step, but `teacher_output.codes_lt` is None."
                )
            batch["audio_codes_lt_teacher"] = teacher_output.codes_lt

        return batch

    def _get_local_transformer_status(self) -> bool:
        """Determine whether local-transformer distillation is active.

        Returns:
            Whether the model has a supported local transformer, local-transformer
            distillation is enabled and the configured start step has been reached.
        """
        if self.local_transformer_type == LocalTransformerType.NO_LT or not self.distill_local_transformer:
            return False

        return self.lt_loss_weight > 0.0 and self.global_step >= self.lt_distillation_start_step

    def _get_local_transformer_loss_weight(self) -> float:
        """Calculate the current local-transformer distillation loss weight.

        Returns:
            Zero before local-transformer distillation starts, a linearly ramped weight
            during its warmup period or the configured final loss weight afterward.
        """
        if self.global_step < self.lt_distillation_start_step:
            return 0.0

        elif self.global_step >= self.lt_distillation_start_step + self.lt_distillation_ramp_len:
            return self.lt_loss_weight

        weight = self.lt_loss_weight
        weight *= (self.global_step - self.lt_distillation_start_step) / self.lt_distillation_ramp_len

        return weight

    def _get_kl_criterion(self, mode: _LossMode) -> Callable:
        """Return the KL-divergence criterion for the requested model component.

        Args:
            mode: Defines model component whose loss criterion is requested.

        Returns:
            Configured KL-divergence loss criterion.
        """
        return self._lt_codes_kl_criterion if mode == _LossMode.local_transformer else self._codes_kl_criterion

    def _get_ce_criterion(self, mode: _LossMode) -> Callable:
        """Return the cross-entropy criterion for the requested model component.

        Args:
            mode: Defines model component whose loss criterion is requested.

        Returns:
            Configured cross-entropy loss criterion.
        """
        return self._lt_codes_ce_criterion if mode == _LossMode.local_transformer else self._codes_ce_criterion

    def _get_nrmse_criterion(self, mode: _LossMode) -> Callable:
        """Return the NRMSE criterion for the requested model component.

        Args:
            mode: Defines model component whose loss criterion is requested.

        Returns:
            Configured normalized root-mean-square-error loss criterion.
        """
        return self._lt_codes_nrmse_criterion if mode == _LossMode.local_transformer else self._codes_nrmse_criterion

    def _compute_codes_loss_helper(
        self,
        teacher_logits: Tensor,
        teacher_codes: Tensor,
        student_logits: Tensor,
        mask: Tensor,
        sample_weights: Optional[Tensor],
        mode: _LossMode,
    ) -> dict[str, Tensor]:
        """Compute audio-code distillation loss terms.

        Computes enabled KL-divergence, cross-entropy, and normalized root-mean-square
        error loss terms for the specified model component, then combines them using
        their configured weights. KL logits are temperature-scaled and the resulting
        loss is rescaled by the temperature squared.

        Args:
            teacher_logits: Teacher audio-code logits.
            teacher_codes: Discrete audio-code targets produced by the teacher.
            student_logits: Student audio-code logits.
            mask: Boolean mask identifying valid audio-code positions.
            sample_weights: Optional per-sample weights for the loss criteria.
            mode: Model component for which to compute the losses.

        Returns:
            Mapping containing every enabled individual loss term and
            its configured-weight total loss.
        """
        output: dict[str, Tensor] = {}

        if self.audio_kl_loss_weight > 0.0:
            temperature = self.distillation_temperature

            # Temperature is applied only to the soft KL distributions.
            student_logits_kl = student_logits / temperature
            teacher_logits_kl = teacher_logits / temperature

            kl_loss = self._get_kl_criterion(mode)(
                student_logits=student_logits_kl,
                teacher_logits=teacher_logits_kl,
                mask=mask,
                sample_weights=sample_weights,
            )
            kl_loss = kl_loss * (temperature**2)

            output[_get_loss_key(_LossKey.kl_loss, mode)] = kl_loss

        if self.audio_ce_loss_weight > 0.0:
            ce_loss = self._get_ce_criterion(mode)(
                predicted_logits=student_logits,
                target_codes=teacher_codes,
                mask=mask,
                sample_weights=sample_weights,
            )
            output[_get_loss_key(_LossKey.ce_loss, mode)] = ce_loss

        kl_term = output.get(_get_loss_key(_LossKey.kl_loss, mode), 0.0)
        ce_term = output.get(_get_loss_key(_LossKey.ce_loss, mode), 0.0)
        loss = self.audio_kl_loss_weight * kl_term + self.audio_ce_loss_weight * ce_term

        if self.audio_nrmse_loss_weight > 0.0:
            nrmse_loss = self._get_nrmse_criterion(mode)(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
                sample_weights=sample_weights,
            )
            output[_get_loss_key(_LossKey.nrmse_loss, mode)] = nrmse_loss
            loss = loss + self.audio_nrmse_loss_weight * nrmse_loss

        output[_get_loss_key(_LossKey.loss, mode)] = loss

        return output

    def _compute_phoneme_loss_guidance(
        self,
        batch: dict[str, Tensor | list],
        student_logits: Tensor,
        sample_weights: Optional[Tensor],
    ) -> dict[str, Tensor]:
        """Compute cross-entropy guidance loss for phoneme predictions.

        Uses phoneme targets from the batch when available. It removes BOS,
        aligns and pads targets to the time coverage of the stacked student logits
        and masks invalid positions. Returns a zero loss when phoneme targets are unavailable.

        Args:
            batch: Batch that may contain phoneme tokens and their lengths.
            student_logits: Stacked-time phoneme logits predicted by the student.
            sample_weights: Optional per-sample weights for the loss criterion.

        Returns:
            Mapping containing the mode-specific phoneme guidance loss.

        Raises:
            RuntimeError: If a phoneme target length exceeds the time coverage of the student logits.
        """
        output: dict[str, Tensor] = {}
        mode = _LossMode.phoneme

        target_tokens = batch.get("phoneme_tokens")
        tokens_lens = batch.get("phoneme_tokens_lens")

        if target_tokens is not None and tokens_lens is not None:
            # Remove BOS.
            target_tokens = target_tokens.long()[:, 1:].unsqueeze(1)
            tokens_lens = tokens_lens - 1

            # `student_logits` is stacked-time: (B, ceil(T / S), S * V).
            # The shared CE criterion expects an unstacked target/mask whose
            # width is exactly T_stacked * S.
            expected_unstacked_len = student_logits.size(1) * self.phoneme_stacking_factor

            if tokens_lens.max().item() > expected_unstacked_len:
                raise RuntimeError(
                    "Phoneme target length exceeds the time coverage of phoneme logits: "
                    f"max_target_len={tokens_lens.max().item()}, "
                    f"logit_coverage={expected_unstacked_len}."
                )

            target_tokens = target_tokens[:, :, :expected_unstacked_len]
            pad_len = expected_unstacked_len - target_tokens.size(-1)

            if pad_len > 0:
                target_tokens = torch.nn.functional.pad(
                    target_tokens,
                    pad=(0, pad_len),
                    value=self.phoneme_tokenizer.eos_token_id,
                )

            mask = get_mask_from_lengths(tokens_lens, x=target_tokens)

            if self.cfg.get("use_multiturn_dataset", False) or self.cfg.get("phoneme_loss_mask_padding", False):
                phoneme_pad_id = getattr(self.phoneme_tokenizer, "pad", -1)
                mask = mask & (target_tokens[:, 0, :] != phoneme_pad_id)

            loss = self._phonemes_ce_criterion(
                predicted_logits=student_logits,
                target_codes=target_tokens,
                mask=mask,
                sample_weights=sample_weights,
            )
        else:
            loss = torch.tensor(0.0, device=student_logits.device)

        output[_get_loss_key(_LossKey.loss, mode)] = loss

        return output

    def _compute_loss(
        self,
        batch: dict[str, Tensor | list],
        teacher_output: _TeacherOutput,
        student_output: _StudentOutput,
        use_lt: bool,
    ) -> dict[str, Tensor]:
        """Compute the total distillation loss for a batch.

        Computes audio-code losses for the main model, optionally blends them with
        local-transformer losses using the current local-transformer loss weight, and
        adds phoneme guidance loss when it is enabled and phoneme logits are available.

        Args:
            batch: Batch containing optional phoneme targets.
            teacher_output: Teacher rollout outputs used as distillation targets.
            student_output: Student model outputs used to compute losses.
            use_lt: Whether to include local-transformer distillation losses.

        Returns:
            Mapping containing individual loss terms and the combined distillation loss
            under the `"loss"` key.
        """
        codes_mask = get_mask_from_lengths(teacher_output.lens)
        output = self._compute_codes_loss_helper(
            teacher_logits=teacher_output.logits,
            teacher_codes=teacher_output.codes,
            student_logits=student_output.logits,
            mask=codes_mask,
            sample_weights=teacher_output.sample_weights,
            mode=_LossMode.main,
        )
        backbone_loss_key = _get_loss_key(key=_LossKey.loss, mode=_LossMode.main)
        output["loss"] = output[backbone_loss_key]

        if use_lt:
            lt_codes_mask = get_mask_from_lengths(student_output.audio_codes_lens_gt_lt)

            lt_output = self._compute_codes_loss_helper(
                teacher_logits=teacher_output.logits_lt,
                teacher_codes=student_output.audio_codes_gt_lt,
                student_logits=student_output.logits_lt,
                mask=lt_codes_mask,
                sample_weights=teacher_output.sample_weights,
                mode=_LossMode.local_transformer,
            )
            output.update(lt_output)
            lt_weight = self._get_local_transformer_loss_weight()
            lt_loss_key = _get_loss_key(key=_LossKey.loss, mode=_LossMode.local_transformer)
            output["loss"] = (1 - lt_weight) * output["loss"] + lt_weight * output[lt_loss_key]
            del output[lt_loss_key]

        if self.phoneme_ce_loss_weight and student_output.logits_phonemes is not None:
            phonemes_output = self._compute_phoneme_loss_guidance(
                batch=batch,
                student_logits=student_output.logits_phonemes,
                sample_weights=teacher_output.sample_weights,
            )
            output.update(phonemes_output)
            phoneme_loss_key = _get_loss_key(key=_LossKey.loss, mode=_LossMode.phoneme)
            output["loss"] = output["loss"] + self.phoneme_ce_loss_weight * output[phoneme_loss_key]

        return output

    def _add_audio_codes(
        self,
        batch: dict[str, Tensor | list],
    ) -> dict[str, Tensor | list]:
        """Add codec audio codes to a batch when they are not already present.

        Args:
            batch: Batch containing audio and audio lengths.

        Returns:
            Batch containing `"audio_codes"` and `"audio_codes_lens"`.
        """
        if "audio_codes" in batch:
            return batch

        audio_codes, audio_codes_lens = self._codec_helper.audio_to_codes(
            audio=batch["audio"],
            audio_len=batch["audio_lens"],
            sample_rate=batch.get("sample_rate"),
        )
        batch["audio_codes"] = audio_codes
        batch["audio_codes_lens"] = audio_codes_lens

        return batch

    def _add_context_audio_codes(
        self,
        batch: dict[str, Tensor | list],
    ) -> dict[str, Tensor | list]:
        """Add codec context-audio codes to a batch when they are not already present.

        Args:
            batch: Batch containing context audio and context-audio lengths.

        Returns:
            Batch containing `"context_audio_codes"` and `"context_audio_codes_lens"`.
        """
        if "context_audio_codes" in batch:
            return batch

        context_audio = batch["context_audio"]
        context_audio_lens = batch["context_audio_lens"]

        context_audio_codes, context_audio_codes_lens = self._codec_helper.audio_to_codes(
            audio=context_audio, audio_len=context_audio_lens
        )
        batch["context_audio_codes"] = context_audio_codes
        batch["context_audio_codes_lens"] = context_audio_codes_lens

        return batch

    def _log_rejections(
        self,
        batch: dict[str, Tensor | list],
        teacher_output: _TeacherOutput,
    ) -> None:
        """Save debugging artifacts for rejected teacher-rollout samples.

        For each rejected sample, decodes and writes the rejected rollout audio and
        its ground-truth audio to the rejection-debug directory. It also appends
        sample metadata, including optional language, dataset, and text fields, to the
        configured metadata file.

        Args:
            batch: Batch containing audio codes and optional sample metadata.
            teacher_output: Teacher rollout outputs, including rejection decisions and
                generated audio codes.

        Raises:
            ValueError: If the batch does not contain audio codes or their lengths.
            RuntimeError: If rejection-debug paths or the metadata file are not initialized.
        """
        if teacher_output.rejected is None:
            return

        audio_codes, audio_codes_lens = batch.get("audio_codes"), batch.get("audio_codes_lens")

        if audio_codes is None or audio_codes_lens is None:
            raise ValueError(
                "Cannot log rejected teacher rollouts: batch must contain 'audio_codes' and 'audio_codes_lens'."
            )

        rejected_indices = torch.nonzero(teacher_output.rejected, as_tuple=False).flatten().tolist()

        if not rejected_indices:
            return

        debug_dir, meta_fp = self.rejection_debug_dir, self.rejection_debug_meta_fp

        if debug_dir is None or meta_fp is None:
            raise RuntimeError("Rejection debug mode is enabled but debug paths/files are not initialized.")

        rejected_codes = teacher_output.codes[rejected_indices].clone()
        rejected_lens = teacher_output.lens[rejected_indices].clone()

        rejected_codes, rejected_lens = self._prepare_codes_for_decode(
            codes=rejected_codes,
            codes_len=rejected_lens,
        )
        rejected_audio, rejected_audio_lens, _ = self._codec_helper.codes_to_audio(
            codes=rejected_codes,
            codes_len=rejected_lens,
        )
        audio_codes_gt = audio_codes

        if self._codec_converter is not None:
            audio_codes_gt = self._codec_converter.convert_original_to_new(
                audio_tokens=audio_codes,
                audio_lens=audio_codes_lens,
            )

        audio_codes_gt = audio_codes_gt.long()

        audio_codes_gt, audio_codes_lens_gt = self._prepare_codes_for_decode(
            codes=audio_codes_gt,
            codes_len=audio_codes_lens,
        )
        audio_gt, audio_lens_gt, _ = self._codec_helper.codes_to_audio(
            codes=audio_codes_gt,
            codes_len=audio_codes_lens_gt,
        )
        languages = batch.get("languages")
        dataset_names = batch.get("dataset_names")
        texts = batch.get("raw_texts")

        for local_idx, batch_idx in enumerate(rejected_indices):
            audio_np = rejected_audio[local_idx].float().detach().cpu().numpy()
            audio_np = audio_np[: int(rejected_audio_lens[local_idx].item())]
            sample_id = f"step_{self.global_step}_rank_{self.global_rank}_batch_idx_{batch_idx}"
            audio_path = debug_dir / f"{sample_id}.wav"
            sf.write(audio_path, audio_np, self.output_sample_rate)

            audio_np_gt = audio_gt[batch_idx].float().detach().cpu().numpy()
            audio_np_gt = audio_np_gt[: int(audio_lens_gt[batch_idx].item())]
            sample_id_gt = f"{sample_id}_gt"
            audio_path_gt = debug_dir / f"{sample_id_gt}.wav"
            sf.write(audio_path_gt, audio_np_gt, self.output_sample_rate)

            item_meta = {
                "global_step": int(self.global_step),
                "global_rank": int(self.global_rank),
                "batch_index": batch_idx,
                "audio_path": audio_path.as_posix(),
                "audio_path_gt": audio_path_gt.as_posix(),
            }
            if languages is not None:
                item_meta["language"] = languages[batch_idx]

            if dataset_names is not None:
                item_meta["dataset_name"] = dataset_names[batch_idx]

            if texts is not None:
                item_meta["text"] = texts[batch_idx]

            meta_fp.write(json.dumps(item_meta, ensure_ascii=False) + "\n")

        logging.info(f"Saved {len(rejected_indices)} rejected samples for debugging to {debug_dir}.")

    def _process_batch_distillation(
        self,
        batch: dict[str, Tensor | list],
        mode: str = "train",
    ) -> dict[str, Tensor]:
        """Run teacher rollout, student forward pass and distillation-loss computation.

        Generates teacher targets for the batch, optionally records rejected rollouts,
        updates the batch with teacher outputs, and computes student distillation
        losses. During training with rollout rejection enabled, also reports the
        teacher-rollout rejection rate.

        Args:
            batch: Collated batch containing text, audio-code, and context inputs.
            mode: Execution mode, such as `"train"` or `"validation"`.

        Returns:
            Mapping containing the total loss, enabled component losses and the
            rejection-rate metric when applicable.
        """
        # This distillation path is intentionally restricted to ordinary single-turn TTS cuts.
        if self.cfg.get("use_multiturn_dataset", False):
            _validate_tts_tasks(batch)
            text = batch["text"]
            batch["text"] = text.masked_fill(text == self.interruption_token_id, self.pad_id)

        use_lt = self._get_local_transformer_status()

        teacher_output = self._teacher_inference_engine.infer_batch(
            batch=batch,
            use_lt=use_lt,
            max_decoder_steps=self.max_decoder_steps,
            temperature=self.rollout_temperature,
            topk=self.rollout_topk,
            cfg_scale=self.distillation_cfg_scale,
            phoneme_input_type="gt",
            phoneme_sampling_method="argmax",
            lower_rejection_threshold=self.lower_rejection_threshold if mode == "train" else None,
            upper_rejection_threshold=self.upper_rejection_threshold if mode == "train" else None,
            rejection_weight=self.rejection_weight if mode == "train" else None,
        )

        if self.use_rejection_debug_mode:
            self._log_rejections(batch, teacher_output)

        batch = self._update_batch(batch, teacher_output, use_lt)
        student_output = self._process_batch(batch, use_lt, mode)
        output = self._compute_loss(batch, teacher_output, student_output, use_lt)

        if mode == "train" and self.use_rollout_rejection:
            output[_MetricKey.rejection_rate.value] = _compute_rejection_rate(teacher_output)

        return output

    def training_step(
        self,
        batch: dict[str, Tensor | list],
        batch_idx: int,
    ) -> Tensor:
        """Run one CFG-distillation optimization step and logs training metrics.

        Args:
            batch: Collated training batch.
            batch_idx: Index of the batch within the current epoch.

        Returns:
            Scalar loss used for backpropagation.
        """
        batch = self._add_audio_codes(batch)
        batch = self._add_context_audio_codes(batch)
        outputs = self._process_batch_distillation(batch, mode="train")
        bs = batch["audio_codes"].size(0)
        loss = outputs["loss"]

        self.log(
            name="train/loss",
            value=loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
            on_step=True,
            on_epoch=True,
        )
        for key in _MONITORED_KEYS_TRAIN:
            if key in outputs:
                self.log(
                    name=f"train/{key}",
                    value=outputs[key],
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=bs,
                    on_step=True,
                    on_epoch=True,
                )

        if (
            self.use_rejection_debug_mode
            and batch_idx % self.debug_meta_flush_interval_steps == 0
            and self.rejection_debug_meta_fp is not None
        ):
            self.rejection_debug_meta_fp.flush()

        return loss

    @property
    def validation_step_outputs(self) -> list[list[dict[str, Tensor]]]:
        """Return lazily initialized per-dataloader validation outputs.

        Returns:
            Lists that accumulate validation outputs for each dataloader.
        """
        if self._validation_step_outputs is not None:
            return self._validation_step_outputs

        if isinstance(self._validation_dl, (list, tuple)):
            num_dl = len(self._validation_dl)
        else:
            num_dl = 1

        self._validation_step_outputs = [[] for _ in range(num_dl)]

        return self._validation_step_outputs

    def _prepare_validation_context_audio(
        self,
        batch: dict[str, Tensor | list],
    ) -> _ContextAudio:
        """Prepare decoded context audio for validation.

        Builds the validation context from the batch, removes special audio-code
        tokens, prepares the remaining codes for codec decoding, and decodes them to
        waveforms.

        Args:
            batch: Validation batch containing context inputs and audio codes.

        Returns:
            A container with decoded context audio and its lengths.
        """
        training_mode = self._configure_training_mode(mode="validation", training_mode=None)
        context = self._prepare_context(batch, training_mode)

        context_audio_codes, context_audio_codes_lens = remove_special_tokens(
            codes=context.audio_codes,
            codes_len=context.audio_codes_lens,
        )
        context_audio_codes, context_audio_codes_lens = self._prepare_codes_for_decode(
            codes=context_audio_codes,
            codes_len=context_audio_codes_lens,
        )
        context_audio, context_audio_lens, _ = self._codec_helper.codes_to_audio(
            codes=context_audio_codes,
            codes_len=context_audio_codes_lens,
        )
        return _ContextAudio(audio=context_audio, lens=context_audio_lens)

    def _log_generated_audio(
        self,
        audio_np: np.ndarray,
        idx: int,
    ) -> None:
        """Log a generated audio sample to supported experiment loggers.

        Args:
            audio_np: Generated audio waveform as a NumPy array.
            idx: Index used to identify the logged audio example.
        """
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                embedded_audio = wandb.Audio(audio_np, sample_rate=self.output_sample_rate, caption="generated")
                logger.experiment.log({f"Audio_Generated/Example_{idx}": embedded_audio})

            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_audio(
                    tag=f"Example_{idx}/generated",
                    snd_tensor=audio_np,
                    global_step=self.global_step,
                    sample_rate=self.output_sample_rate,
                )

    def _save_validation_audio(
        self,
        infer_output: InferBatchOutput,
        context_audio: _ContextAudio,
        batch_idx: int,
    ) -> _ValidationAudioPaths:
        """Save generated and context audio for a validation batch.

        Writes each generated waveform and its corresponding context waveform to the
        validation-audio directory. On global rank zero, logs the first three generated
        samples from the first batch to supported experiment loggers.

        Args:
            infer_output: Inference outputs containing generated audio and its lengths.
            context_audio: Decoded context audio and its lengths.
            batch_idx: Index of the validation batch.

        Returns:
            A container with paths to the saved generated and context audio files.
        """
        bs = infer_output.predicted_audio.size(0)
        paths = _ValidationAudioPaths()

        for idx in range(bs):
            audio_np = infer_output.predicted_audio[idx].float().detach().cpu().numpy()
            audio_np = audio_np[: infer_output.predicted_audio_lens[idx]]

            ctx_audio_np = context_audio.audio[idx].float().detach().cpu().numpy()
            ctx_audio_np = ctx_audio_np[: context_audio.lens[idx]]

            if batch_idx == 0 and self.global_rank == 0 and idx < 3:
                self._log_generated_audio(audio_np, idx)

            audio_path = self.audio_dir / f"rank{self.global_rank}_batch{batch_idx}_idx{idx}.wav"
            sf.write(audio_path, audio_np, self.output_sample_rate)
            paths.generated.append(audio_path)

            ctx_path = self.audio_dir / f"rank{self.global_rank}_batch{batch_idx}_idx{idx}_context.wav"
            sf.write(ctx_path, ctx_audio_np, self.output_sample_rate)
            paths.context.append(ctx_path)

        return paths

    def _validate_batch_using_verifiers(
        self,
        batch: dict[str, Tensor | list],
        batch_idx: int,
    ) -> dict[str, Tensor | list[float] | list[str]]:
        """Generate validation audio and compute quality-verifier metrics.

        Runs student inference without classifier-free guidance, decodes and
        saves generated and context audio, then obtains ASR transcripts,
        speaker-verification embeddings and UTMOS scores.
        These verifier outputs are converted into validation metrics.

        Args:
            batch: Collated validation batch.
            batch_idx: Index of the validation batch.

        Returns:
            Mapping containing validation quality metrics and any per-sample values
            needed for epoch-level aggregation.
        """
        use_lt = self.local_transformer_type == LocalTransformerType.AR

        infer_output = self.infer_batch(
            batch=batch,
            max_decoder_steps=self.max_decoder_steps,
            temperature=self.rollout_temperature,
            topk=self.rollout_topk,
            use_local_transformer_for_inference=use_lt,
            use_cfg=False,
        )

        context_audio = self._prepare_validation_context_audio(batch)
        paths = self._save_validation_audio(infer_output, context_audio, batch_idx)

        transcripts = self._val_inference_engine.transcribe_files(
            paths=paths.generated,
            languages=batch.get("languages"),
        )
        speaker_embeds = self._val_inference_engine.verify_speakers(paths)
        utmos_scores = self._val_inference_engine.get_utmos_scores(
            input_dir=self.audio_dir,
            filenames=[p.name for p in paths.generated],
        )
        output = _compute_validation_metrics(
            batch=batch,
            transcripts=transcripts,
            speaker_embeds=speaker_embeds,
            utmos_scores=utmos_scores,
            device=self.device,
            use_multilingual_asr=self._val_inference_engine.use_multilingual_asr,
        )
        return output

    def validation_step(
        self,
        batch: dict[str, Tensor | list],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        """Run one validation distillation step and optional quality verifiers.

        Args:
            batch: Collated validation batch.
            batch_idx: Index of the batch within the current epoch.
            dataloader_idx: Index of the validation dataloader.

        Returns:
            Validation losses and any configured quality metrics.
        """
        batch = self._add_audio_codes(batch)
        batch = self._add_context_audio_codes(batch)
        val_output = self._process_batch_distillation(batch, mode="validation")

        if self._val_inference_engine.is_active:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            val_metrics_output = self._validate_batch_using_verifiers(batch, batch_idx)
            val_output.update(val_metrics_output)

        self.validation_step_outputs[dataloader_idx].append(val_output)

        return val_output

    def _log_validation_epoch_output(
        self,
        val_outputs: list[dict[str, Tensor]] | list[list[dict[str, Tensor]]],
        prefix: str,
        aggregated: bool = False,
    ) -> None:
        """Aggregate and log validation loss and monitored metrics.

        Args:
            val_outputs: Per-batch validation outputs from one dataloader or from all
                validation dataloaders.
            prefix: Metric namespace to use when logging values.
            aggregated: Whether the outputs represent the aggregate across validation
                dataloaders. When `True`, also logs `"val_loss"` metric.
        """
        val_loss = _collect_validation_outputs(val_outputs, key="loss")

        if val_loss is not None:
            self.log(
                name=f"{prefix}/loss",
                value=val_loss,
                prog_bar=aggregated,
                sync_dist=True,
            )
            if aggregated:
                self.log(
                    name="val_loss",
                    value=val_loss,
                    prog_bar=False,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    logger=False,
                    enable_graph=False,
                )

        for key in _MONITORED_KEYS_VALID:
            value = _collect_validation_outputs(val_outputs, key)
            if value is not None:
                self.log(
                    name=f"{prefix}/{key}",
                    value=value,
                    prog_bar=aggregated,
                    sync_dist=True,
                )

    def _log_epoch_language_metrics(
        self,
        val_outputs: list[dict[str, Tensor]] | list[list[dict[str, Tensor]]],
        prefix: str,
    ) -> None:
        """Aggregate and log per-language validation metrics.

        Aggregates language-specific metrics from validation outputs and logs each
        available metric under the supplied namespace. Does nothing when no
        language-specific metrics are present.

        Args:
            val_outputs: Per-batch validation outputs from one dataloader or from all
                validation dataloaders.
            prefix: Metric namespace to use when logging values.
        """
        metrics = _aggregate_language_metrics(val_outputs, device=self.device)

        if metrics is None:
            return

        for key, value in metrics.items():
            self.log(
                name=f"{prefix}/{key}",
                value=value,
                prog_bar=True,
                sync_dist=True,
            )

    def on_validation_epoch_end(self) -> None:
        """Aggregate, log and clear validation outputs for the epoch."""
        self._log_validation_epoch_output(self.validation_step_outputs, prefix="val", aggregated=True)

        if self._val_inference_engine.use_multilingual_asr:
            self._log_epoch_language_metrics(self.validation_step_outputs, prefix="val")

        if len(self.validation_step_outputs) > 1:
            for dataloader_idx, val_outputs in enumerate(self.validation_step_outputs):
                if not val_outputs:
                    continue

                prefix = self.get_validation_dataloader_prefix(dataloader_idx)
                self._log_validation_epoch_output(val_outputs, prefix=f"val-{prefix}")

        for val_outputs in self.validation_step_outputs:
            val_outputs.clear()
