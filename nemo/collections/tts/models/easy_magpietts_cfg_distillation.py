# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Self, Sequence

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor

from nemo.collections.tts.losses.magpietts_cfg_distillation import (
    CodesCrossEntropyLoss,
    KLDivergenceLoss,
    NRMSELogitsLoss,
)
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_inference import (
    EasyMagpieTTSInferenceModel,
    StreamingConfig,
    TrainingMode,
)
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType, clear_forbidden_logits
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.lightning.callback_group import CallbackGroup
from nemo.utils import logging

__all__ = ["EasyMagpieCFGDistillation"]


_STATE_DICT_EXCLUDE_NAMES: list[str] = ["_teacher_model"]


@dataclass
class _DefaultParams:

    # Maximum number of decoding steps during audio rollout generation.
    max_decoder_steps: int = 430
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
    # Weight coefficient in the combined entropy-divergence distillation loss.
    alpha: float = 0.3
    # Weight coefficient for the NRMSE component in the distillation loss.
    beta: float = 2.0
    # Fraction of the ground-truth sequence length used as a cutoff for early rollout truncation.
    truncation_threshold: Optional[float] = 1.25
    # Weight assigned to truncated samples when computing the loss (used to down-weight truncated rollouts).
    truncation_weight: Optional[float] = 0.1
    # Whether to enable distillation of the local transformer head in addition to the main decoder logits.
    distill_local_transformer: bool = True
    # Target mixing weight for the local-transformer distillation loss in the final total loss.
    lt_loss_weight: float = 0.1
    # Global training step at which local-transformer distillation becomes active.
    lt_distillation_start_step: int = 0
    # Number of steps used to linearly ramp the local-transformer loss weight from 0 to `lt_loss_weight`.
    lt_distillation_ramp_len: int = 2000


_DEFAULT_PARAMS = _DefaultParams()


def _validate_configuration(cfg: DictConfig) -> None:
    if hasattr(cfg, "distillation_temperature") and cfg.get("distillation_temperature") <= 0:
        raise ValueError(
            "`distillation_temperature` must be greater than 0. "
            "Typical values for distillation are in the range [1.0, 4.0]."
        )

    if hasattr(cfg, "alpha") and not (0 <= cfg.get("alpha") <= 1):
        raise ValueError(
            "`alpha` must be in the range [0, 1]. "
            "It controls the weighting between KL-divergence and cross-entropy losses."
        )

    if hasattr(cfg, "beta") and cfg.get("beta") < 0:
        raise ValueError("`beta` must be non-negative. It scales the contribution of the NRMSE loss component.")

    if (
        hasattr(cfg, "truncation_threshold")
        and cfg.get("truncation_threshold") is not None
        and cfg.get("truncation_threshold") < 1.0
    ):
        raise ValueError(
            "`truncation_threshold` must be >= 1.0 or `None`. "
            "Values below 1.0 would truncate sequences shorter than the ground truth."
        )

    if (
        hasattr(cfg, "truncation_weight")
        and cfg.get("truncation_weight") is not None
        and cfg.get("truncation_weight") < 0
    ):
        raise ValueError(
            "`truncation_weight` must be non-negative or `None`. "
            "It defines the relative weighting for truncated samples in the loss."
        )

    if hasattr(cfg, "lt_loss_weight") and not (0.0 <= cfg.get("lt_loss_weight") <= 1.0):
        raise ValueError("`lt_loss_weight` must be in the range [0, 1].")

    if hasattr(cfg, "lt_distillation_start_step") and cfg.get("lt_distillation_start_step") < 0:
        raise ValueError("`lt_distillation_start_step` must be non-negative.")

    if hasattr(cfg, "lt_distillation_ramp_len") and cfg.get("lt_distillation_ramp_len") < 0:
        raise ValueError("`lt_distillation_ramp_len` must be non-negative.")


def _get_teacher_model(cfg: DictConfig) -> EasyMagpieTTSModel:
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

    predicted_codes = torch.cat(predicted_codes, dim=1)
    predicted_logits = torch.cat(predicted_logits, dim=1)
    dims = (-1, model.frame_stacking_factor, model.num_audio_codebooks)
    predicted_codes = predicted_codes.reshape(*dims).permute(0, 2, 1)

    predicted_codes = predicted_codes[:bs]
    predicted_logits = predicted_logits[:bs]

    return predicted_codes, predicted_logits


@dataclass
class _AudioCodesStep:

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
        self.codes_t = self.codes_t.view(batch_size, num_codebooks, fs_factor)
        self.codes_t_argmax = self.codes_t_argmax.view(batch_size, num_codebooks, fs_factor)

        if self.codes_t_lt is not None and self.codes_t_argmax_lt is not None and self.logits_t_lt is not None:
            self.codes_t_lt = self.codes_t_lt.view(batch_size, num_codebooks, fs_factor)
            self.codes_t_argmax_lt = self.codes_t_argmax_lt.view(batch_size, num_codebooks, fs_factor)


@dataclass
class _StreamingState:

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

    gt_phoneme_embeddings: Optional[Tensor] = None  # (B, T', E) pre-computed GT embeddings
    gt_phoneme_lens: Optional[Tensor] = None  # (B,) lengths after stacking

    truncated: Optional[Tensor] = None  # (B,) bool mask of samples truncated by rollout policy
    sample_weights: Optional[Tensor] = None  # (B,) optional per-sample weights

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
        truncation_weight: Optional[float],
    ) -> Self:
        bs = config.batch_size
        device = config.device
        sample_weights = None

        if truncation_weight is not None:
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
            truncated=torch.zeros(bs, dtype=torch.bool, device=device),
            sample_weights=sample_weights,
        )
        return obj

    def update_counters(
        self,
        needs_context: Tensor,
        needs_phoneme: Tensor,
        needs_audio: Tensor,
    ) -> None:
        self.context_position = self.context_position + needs_context.long()
        self.text_tokens_seen = self.text_tokens_seen + (~needs_context).long()
        self.phoneme_steps = self.phoneme_steps + needs_phoneme.long()
        self.audio_steps = self.audio_steps + needs_audio.long()

    def update_phoneme_start_idx(
        self,
        needs_phoneme: Tensor,
    ) -> None:
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
        pred_phoneme_tokens: Tensor,
        needs_phoneme: Tensor,
        eos_id: int,
    ) -> None:
        eos_detected = needs_phoneme & (pred_phoneme_tokens == eos_id).any(dim=1)
        self.phoneme_eos_detected = self.phoneme_eos_detected | eos_detected

        newly_ended = eos_detected & (self.phoneme_prediction_end_idx == -1)

        if newly_ended.any():
            self.phoneme_prediction_end_idx = torch.where(
                condition=newly_ended,
                input=torch.full_like(self.phoneme_prediction_end_idx, self.current_phoneme_step_idx),
                other=self.phoneme_prediction_end_idx,
            )

    def update_audio_start_idx(
        self,
        needs_audio: Tensor,
    ) -> None:
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
        num_codebooks: int,
        fs_factor: int,
    ) -> None:
        if self.last_audio_codes is None:
            self.last_audio_codes = audio_codes.codes_t
        else:
            update_mask = needs_audio.view(self.batch_size, 1).expand_as(audio_codes.codes_t)
            self.last_audio_codes = torch.where(update_mask, audio_codes.codes_t, self.last_audio_codes)

        if self.last_audio_codes_lt is None:
            self.last_audio_codes_lt = audio_codes.codes_t_lt
        else:
            update_mask = needs_audio.view(self.batch_size, 1).expand_as(audio_codes.codes_t_lt)
            self.last_audio_codes_lt = torch.where(update_mask, audio_codes.codes_t_lt, self.last_audio_codes_lt)

    def update_audio_end_status(
        self,
        audio_codes: _AudioCodesStep,
        needs_audio: Tensor,
        eos_id: int,
        fs_factor: int,
    ) -> None:
        if not self.use_lt:
            codes_t = audio_codes.codes_t
            codes_t_argmax = audio_codes.codes_t_argmax
        else:
            codes_t = audio_codes.codes_t_lt
            codes_t_argmax = audio_codes.codes_t_argmax_lt

        eos_in_sampled = codes_t == eos_id
        eos_in_argmax = codes_t_argmax == eos_id
        eos_any_codebook = eos_in_sampled.any(dim=1) | eos_in_argmax.any(dim=1)
        eos_detected = eos_any_codebook.any(dim=1) & needs_audio
        self.finished = self.finished | eos_detected

        newly_ended = eos_detected & (self.audio_prediction_end_idx == -1)

        if not newly_ended.any():
            return

        # Intentionally retain the full stacked frame containing EOS.
        current_frame_count = len(self.pred_codes) * fs_factor
        end_frame_idx = torch.full_like(self.audio_prediction_end_idx, current_frame_count + fs_factor)
        self.audio_prediction_end_idx = torch.where(newly_ended, end_frame_idx, self.audio_prediction_end_idx)

    def add_audio_codes(
        self,
        audio_codes: _AudioCodesStep,
    ) -> None:
        self.pred_codes.append(audio_codes.codes_t)
        self.pred_codes_logits.append(audio_codes.logits_t.unsqueeze(1))
        self.pred_codes_lt.append(audio_codes.codes_t_lt)
        self.pred_codes_logits_lt.append(audio_codes.logits_t_lt.unsqueeze(1))

    @property
    def current_phoneme_step_idx(self) -> int:
        return len(self.pred_phoneme_tokens)

    @property
    def current_frame_idx(self) -> int:
        return sum(p.size(-1) for p in self.pred_codes)

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @property
    def temperature(self) -> float:
        return self.config.temperature

    @property
    def topk(self) -> int:
        return self.config.topk

    @property
    def cfg_scale(self) -> float:
        return self.config.cfg_scale

    @property
    def device(self) -> torch.device:
        return self.config.device

    @property
    def use_lt(self) -> bool:
        return self.config.use_local_transformer


@dataclass
class _StreamingFinalizeOutput:
    codes: Tensor  # (B, num_codebooks, T) generated audio codes
    logits: Tensor
    codes_len: Tensor  # (B,) length of codes per batch item
    codes_lt: Optional[Tensor]
    logits_lt: Optional[Tensor]


@dataclass
class _TeacherOutput:
    """Outputs produced by the teacher rollout used as distillation targets."""

    codes: Tensor
    logits: Tensor
    lens: Tensor
    sample_weights: Optional[Tensor]

    codes_lt: Optional[Tensor]
    logits_lt: Optional[Tensor]

    tokens_phonemes: Optional[Tensor]
    lens_phonemes: Optional[Tensor]
    logits_phonemes: Optional[Tensor]


class _TeacherInferenceWrapper:

    def __init__(
        self,
        model: EasyMagpieTTSModel,
    ) -> None:
        self.model = model

    def _streaming_init(
        self,
        context_audio_codes: Tensor,
        context_audio_codes_lens: Tensor,
        context_text_tokens: Tensor,
        context_text_tokens_lens: Tensor,
        inference_mode: Optional[str] = None,
        cfg_scale: float = 1.0,
        use_lt: bool = False,
        temperature: float = 0.7,
        topk: int = 80,
        phoneme_input_type: str = "predicted",
        phoneme_sampling_method: str = "argmax",
        gt_phoneme_tokens: Optional[Tensor] = None,
        gt_phoneme_tokens_lens: Optional[Tensor] = None,
        truncation_weight: Optional[float] = None,
    ) -> _StreamingState:
        with torch.no_grad():
            batch_size = context_audio_codes.size(0)
            device = context_audio_codes.device

            mode_name = inference_mode if inference_mode is not None else self.model.default_inference_mode
            if mode_name not in self.model.mode_name_to_mode:
                available_modes = list(self.model.mode_name_to_mode.keys())
                raise ValueError(f"Unknown inference mode '{mode_name}'. Available modes: {available_modes}")

            selected_training_mode = self.model.mode_name_to_mode[mode_name]

            context_embedding, context_lens, context_audio_codes, context_audio_codes_lens = (
                self.model.prepare_context_tensors(
                    context_text_tokens=context_text_tokens,
                    context_text_tokens_lens=context_text_tokens_lens,
                    context_audio_codes=context_audio_codes,
                    context_audio_codes_lens=context_audio_codes_lens,
                    training_mode=selected_training_mode,
                    dropout_conditional_input=False,
                )
            )

            full_context_embedding = context_embedding.clone()  # (B, T_max, E)
            full_context_lens = context_lens.clone()  # (B,)

            min_context_len = context_lens.min().item()

            dummy_context_embedding_unconditional = None

            dummy_context_embedding_unconditional = self.model.embed_text_tokens(
                torch.full((1, 1), self.model.cfg_unk_token_id, device=device),
                text_lens=torch.ones(1, dtype=torch.long, device=device),
                disable_cas_embedding=self.model.disable_cas_for_context_text,
            )
            # Create unconditional context (same length as conditional)
            dummy_context_expanded = dummy_context_embedding_unconditional.expand(
                batch_size, context_embedding.size(1), -1
            )
            # Concatenate conditional and unconditional: (2*B, T, E)
            context_embedding = torch.cat([context_embedding, dummy_context_expanded], dim=0)

            cache_position = torch.arange(min_context_len, device=device)

            transformer_out = self.model.forward(
                inputs_embeds=context_embedding[:, :min_context_len, :],
                attention_mask=None,
                use_cache=True,
                past_key_values=None,
                cache_position=cache_position,
            )

            gt_phoneme_embeddings = None
            gt_phoneme_lens = None

            if gt_phoneme_tokens is not None and gt_phoneme_tokens_lens is not None:
                gt_phoneme_expanded = gt_phoneme_tokens.unsqueeze(1)  # (B, 1, L)
                gt_phoneme_stacked, gt_phoneme_lens = self.model.stack_codes(
                    gt_phoneme_expanded,
                    gt_phoneme_tokens_lens,
                    self.model.phoneme_tokenizer.bos_token_id,
                    self.model.phoneme_tokenizer.eos_token_id,
                    self.model.phoneme_stacking_factor,
                    1,
                )
                gt_phoneme_embeddings = self.model.embed_phoneme_tokens(gt_phoneme_stacked)  # (B, T', E)

            config = StreamingConfig(
                batch_size=batch_size,
                device=device,
                training_mode=selected_training_mode,
                use_cfg=True,
                cfg_scale=cfg_scale,
                use_local_transformer=use_lt,
                temperature=temperature,
                topk=topk,
                phoneme_input_type=phoneme_input_type,
                phoneme_sampling_method=phoneme_sampling_method,
                dummy_context_embedding_unconditional=dummy_context_embedding_unconditional,
            )
            state = _StreamingState.create(
                config=config,
                last_hidden=transformer_out.last_hidden_state,
                past_kv=transformer_out.past_key_values,
                min_context_len=min_context_len,
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                context_lens=context_lens,
                full_context_embedding=full_context_embedding,
                full_context_lens=full_context_lens,
                gt_phoneme_embeddings=gt_phoneme_embeddings,
                gt_phoneme_lens=gt_phoneme_lens,
                truncation_weight=truncation_weight,
            )
            return state

    def _prepare_streaming_input(
        self,
        state: _StreamingState,
        text_tokens: Optional[Tensor],
        force_dropout_text: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

        next_input = torch.zeros(batch_size, 1, self.model.cfg.embedding_dim, device=device)

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
            text_embedded = self.model.embed_text_tokens(
                text_tokens_2d,
                text_lens=torch.ones(batch_size, dtype=torch.long, device=device),
            )  # (B, 1, E)

            if force_dropout_text:
                text_embedded = text_embedded * 0

            is_eos_token = (text_tokens == self.model.eos_id) & needs_text  # (B,) bool
            text_add_mask = needs_text.view(batch_size, 1, 1).float()
            next_input = next_input + text_embedded * text_add_mask
            state.text_finished = state.text_finished | is_eos_token

        elif text_tokens is None:
            state.text_finished = state.text_finished | ~needs_context

        # --- Phoneme embedding for phoneme and audio phase items ---
        if self.model.phoneme_tokenizer is not None:
            if needs_phoneme.any():
                phoneme_emb = torch.zeros(batch_size, 1, self.model.cfg.embedding_dim, device=device)

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
                            (batch_size, self.model.phoneme_stacking_factor, 1),
                            self.model.phoneme_tokenizer.bos_token_id,
                            device=device,
                        ).long()
                        phoneme_bos_emb = self.model.embed_phoneme_tokens(phoneme_bos)  # (B, 1, E)
                        first_mask = first_phoneme_step.view(batch_size, 1, 1).float()
                        phoneme_emb = phoneme_emb + phoneme_bos_emb * first_mask

                    if has_last_phoneme.any() and state.last_phoneme_tokens is not None:
                        last_phoneme_emb = self.model.embed_phoneme_tokens(
                            state.last_phoneme_tokens.unsqueeze(2)
                        )  # (B, 1, E)
                        last_mask = has_last_phoneme.view(batch_size, 1, 1).float()
                        phoneme_emb = phoneme_emb + last_phoneme_emb * last_mask

                    state.phoneme_stream_ended = state.phoneme_stream_ended | state.phoneme_eos_detected

                next_input = next_input + phoneme_emb

        # --- Audio embedding for audio phase items ---
        audio_emb = None
        if needs_audio.any():
            audio_emb = torch.zeros(batch_size, 1, self.model.cfg.embedding_dim, device=device)
            first_audio_step = needs_audio & (state.audio_steps == 0)
            has_last_audio = needs_audio & ~first_audio_step & (state.last_audio_codes is not None)

            if first_audio_step.any():
                audio_bos = torch.full(
                    (batch_size, self.model.num_audio_codebooks * self.model.frame_stacking_factor, 1),
                    self.model.audio_bos_id,
                    device=device,
                ).long()
                audio_bos_emb = self.model.embed_audio_tokens(audio_bos)  # (B, 1, E)
                first_mask = first_audio_step.view(batch_size, 1, 1).float()
                audio_emb = audio_emb + audio_bos_emb * first_mask

            if has_last_audio.any() and state.last_audio_codes is not None:
                last_audio_emb = self.model.embed_audio_tokens(state.last_audio_codes.unsqueeze(2))  # (B, 1, E)
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

    def _predict_phoneme_tokens(
        self,
        state: _StreamingState,
    ) -> tuple[Tensor, Tensor]:
        actual_batch_size = state.config.batch_size
        last_hidden = state.last_hidden

        all_code_logits_t_phoneme = self.model.phoneme_final_proj(last_hidden[:, -1, :])
        all_code_logits_t_phoneme = all_code_logits_t_phoneme[:actual_batch_size]
        phoneme_logits = all_code_logits_t_phoneme.view(
            actual_batch_size, self.model.phoneme_stacking_factor, self.model.phoneme_vocab_size
        )
        max_probs = torch.softmax(phoneme_logits, dim=-1).max(dim=-1).values  # (B, phoneme_stacking_factor)

        # Sample phonemes
        if state.config.phoneme_sampling_method == 'argmax':
            pred_phoneme_tokens = self.model.sample_codes_from_logits_phoneme(
                all_code_logits_t_phoneme, temperature=0.0
            )
        else:
            pred_phoneme_tokens = self.model.sample_codes_from_logits_phoneme(
                all_code_logits_t_phoneme, temperature=state.config.temperature, topk=state.config.topk
            )

        # In prediction mode, low-confidence phoneme steps are replaced with UNK across
        # all stacked channels (except steps where EOS is predicted).
        # TODO: ptobably remove for distillation
        if (
            state.config.phoneme_input_type != 'gt'
            and hasattr(self.model.phoneme_tokenizer, 'unk_token_id')
            and self.model.phoneme_confidence_unk_threshold > 0.0
        ):
            underconfident_step = (max_probs < self.model.phoneme_confidence_unk_threshold).any(
                dim=1, keepdim=True
            )  # (B, 1)
            eos_predicted_step = (pred_phoneme_tokens == self.model.phoneme_tokenizer.eos_token_id).any(
                dim=1, keepdim=True
            )
            replace_with_unk = underconfident_step & (~eos_predicted_step)
            if replace_with_unk.any():
                unk_tokens = torch.full_like(pred_phoneme_tokens, self.model.phoneme_tokenizer.unk_token_id)
                pred_phoneme_tokens = torch.where(replace_with_unk, unk_tokens, pred_phoneme_tokens)
        # (B, phoneme_stacking_factor)
        return pred_phoneme_tokens, phoneme_logits

    def _predict_audio_codes(
        self,
        state: _StreamingState,
        use_lt: bool,
    ) -> _AudioCodesStep:
        last_hidden = state.last_hidden
        temp = state.temperature

        last_hidden_audio = self.model.audio_out_projection(last_hidden[:, -1, :])
        code_logits_t = self.model.final_proj(last_hidden_audio)

        cond_logits = code_logits_t[: state.batch_size]
        uncond_logits = code_logits_t[state.batch_size :]
        logits_t = state.cfg_scale * cond_logits + (1.0 - state.cfg_scale) * uncond_logits

        codes_t = self.model.sample_codes_from_logits(logits_t, temperature=temp, topk=state.topk)
        codes_t_argmax = codes_t if temp <= 0.0 else self.model.sample_codes_from_logits(logits_t, temperature=0.01)

        codes_t_lt, codes_t_argmax_lt, logits_t_lt = None, None, None

        if use_lt:
            codes_t_lt, logits_t_lt = _lt_sample_autoregressive(
                model=self.model,
                dec_output=last_hidden[:, -1, :],
                temperature=temp,
                topk=state.topk,
                cfg_scale=state.cfg_scale,
                use_kv_cache=True,
                sanitize_logits=True,
            )
            codes_t_lt = codes_t_lt.permute(0, 2, 1)
            codes_t_lt = codes_t_lt.reshape(state.batch_size, -1)
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
        needs_context: Tensor,
        needs_phoneme: Tensor,
        needs_audio: Tensor,
    ) -> _StreamingState:
        batch_size = state.config.batch_size
        device = state.config.device
        pred_phoneme_tokens, pred_phoneme_logits = None, None
        audio_codes_next = None

        state.update_counters(needs_context, needs_phoneme, needs_audio)

        if needs_phoneme.any() and self.model.phoneme_tokenizer is not None:
            state.update_phoneme_start_idx(needs_phoneme)

            pred_phoneme_tokens, pred_phoneme_logits = self._predict_phoneme_tokens(state)

            state.last_phoneme_tokens = pred_phoneme_tokens
            state.pred_phoneme_tokens.append(pred_phoneme_tokens)
            state.pred_phoneme_logits.append(pred_phoneme_logits)

            state.update_phoneme_end_status(
                pred_phoneme_tokens=pred_phoneme_tokens,
                needs_phoneme=needs_phoneme,
                eos_id=self.model.phoneme_tokenizer.eos_token_id,
            )

        if needs_audio.any():
            state.update_audio_start_idx(needs_audio)
            audio_codes = self._predict_audio_codes(state, use_lt=state.use_lt)

            state.update_last_audio_codes(
                audio_codes=audio_codes,
                needs_audio=needs_audio,
                num_codebooks=self.model.num_audio_codebooks,
                fs_factor=self.model.frame_stacking_factor,
            )
            audio_codes.unstack(
                batch_size=state.batch_size,
                num_codebooks=self.model.num_audio_codebooks,
                fs_factor=self.model.frame_stacking_factor,
            )
            state.update_audio_end_status(
                audio_codes=audio_codes,
                needs_audio=needs_audio,
                eos_id=self.model.audio_eos_id,
                fs_factor=self.model.frame_stacking_factor,
            )
            state.add_audio_codes(audio_codes)

        return state

    def _streaming_step(
        self,
        state: _StreamingState,
        text_tokens: Optional[Tensor] = None,
        force_dropout_text: bool = False,
    ) -> _StreamingState:
        if state.finished.all():
            return state

        with torch.no_grad():
            device = state.config.device

            next_input, needs_context, needs_phoneme, needs_audio = self._prepare_streaming_input(
                state, text_tokens, force_dropout_text
            )
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
            state = self._process_predictions(state, needs_context, needs_phoneme, needs_audio)

        return state

    def _streaming_finalize(
        self,
        state: _StreamingState,
    ) -> Optional[_StreamingFinalizeOutput]:
        batch_size = state.batch_size
        device = state.device
        use_lt = state.use_lt

        if len(state.pred_codes) == 0:
            return None

        with torch.no_grad():
            codes = torch.cat(state.pred_codes, dim=-1)  # (B, C, T_total_frames)
            logits = torch.cat(state.pred_codes_logits, dim=1)  # (B, T_stacked_frames, D)

            if use_lt:
                codes_lt = torch.cat(state.pred_codes_lt, dim=-1)
                logits_lt = torch.cat(state.pred_codes_logits_lt, dim=1)

            start_indices = torch.clamp(state.audio_prediction_start_idx, min=0)
            end_indices = torch.where(
                condition=state.audio_prediction_end_idx >= 0,
                input=state.audio_prediction_end_idx,
                other=torch.full_like(state.audio_prediction_end_idx, codes.size(-1)),
            )
            pred_codes_lens = end_indices - start_indices
            max_len = pred_codes_lens.max().item()

            if max_len == 0:
                return None

            max_stacked_len = max_len // self.model.frame_stacking_factor
            codes_size = (batch_size, self.model.num_audio_codebooks, max_len)
            stacked_logits_dim = (
                self.model.num_audio_codebooks
                * self.model.num_all_tokens_per_codebook
                * self.model.frame_stacking_factor
            )
            logits_size = (batch_size, max_stacked_len, stacked_logits_dim)
            pred_codes = torch.zeros(*codes_size, dtype=codes.dtype, device=device)
            pred_logits = torch.zeros(*logits_size, dtype=logits.dtype, device=device)

            if use_lt:
                pred_codes_lt = torch.zeros(*codes_size, dtype=codes.dtype, device=device)
                pred_logits_lt = torch.zeros(*logits_size, dtype=logits.dtype, device=device)

            for i in range(batch_size):
                start, end = start_indices[i].item(), end_indices[i].item()
                start_, end_ = start // self.model.frame_stacking_factor, end // self.model.frame_stacking_factor
                length = end - start

                if length == 0:
                    continue

                pred_codes[i, :, :length] = codes[i, :, start:end]
                pred_logits[i, : end_ - start_, :] = logits[i, start_:end_, :]

                if use_lt:
                    pred_codes_lt[i, :, :length] = codes_lt[i, :, start:end]
                    pred_logits_lt[i, : end_ - start_, :] = logits_lt[i, start_:end_, :]

            return _StreamingFinalizeOutput(
                codes=pred_codes,
                logits=pred_logits,
                codes_len=pred_codes_lens,
                codes_lt=pred_codes_lt,
                logits_lt=pred_logits_lt,
            )

    def _apply_rollout_truncation(
        self,
        state: _StreamingState,
        batch: dict[str, Tensor],
        truncation_threshold: float,
        truncation_weight: float,
    ) -> _StreamingState:
        gt_audio_lens = batch["audio_codes_lens"]

        if truncation_threshold is None or gt_audio_lens is None:
            return state

        if len(state.pred_codes) == 0:
            return state

        current_frame_count = len(state.pred_codes) * self.model.frame_stacking_factor
        truncation_thresholds = torch.round(truncation_threshold * gt_audio_lens).long().to(state.device)
        should_truncate = ~state.finished & (current_frame_count >= truncation_thresholds)

        if not should_truncate.any():
            return state

        state.finished = state.finished | should_truncate
        state.truncated = state.truncated | should_truncate
        newly_truncated = should_truncate & (state.audio_prediction_end_idx == -1)

        if newly_truncated.any():
            state.audio_prediction_end_idx = torch.where(
                condition=newly_truncated,
                input=torch.full_like(state.audio_prediction_end_idx, current_frame_count),
                other=state.audio_prediction_end_idx,
            )
        if truncation_weight is not None and state.sample_weights is not None:
            state.sample_weights = torch.where(
                condition=should_truncate,
                input=torch.full_like(state.sample_weights, truncation_weight),
                other=state.sample_weights,
            )
        for item_idx in torch.nonzero(should_truncate, as_tuple=False).flatten().tolist():
            logging.info(f"Item {item_idx} truncated at streaming generation step: {len(state.pred_codes)}")

        return state

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
        force_dropout_text: bool = False,
        use_teacher_forced: bool = False,
        truncation_threshold: Optional[float] = None,
        truncation_weight: Optional[float] = None,
    ) -> _TeacherOutput:
        if use_lt and self.model.local_transformer_type != LocalTransformerType.AR:
            raise ValueError(
                f"Only `LocalTransformerType.AR` is supported for local-transformer distillation, "
                f"but got `{self.model.local_transformer_type}`."
            )
        if "context_audio_codes" not in batch:
            raise ValueError()

        with torch.no_grad():
            text = batch["text"]
            text_lens = batch["text_lens"]
            batch_size = text.size(0)
            device = text.device

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
                truncation_weight=truncation_weight,
            )
            logging.info("Generation started")
            gen_step = 0

            while not state.finished.all() and len(state.pred_codes) < max_decoder_steps:
                gen_step += 1

                if gen_step % 10 == 0:
                    logging.info(f"Generation step {gen_step}")

                positions = state.text_tokens_seen.clamp(max=text.size(1) - 1)
                current_tokens = text[torch.arange(batch_size, device=device), positions]

                current_tokens = torch.where(
                    condition=state.text_tokens_seen >= text_lens,
                    input=torch.full_like(current_tokens, self.model.eos_id),
                    other=current_tokens,
                )
                state = self._streaming_step(
                    state=state,
                    text_tokens=current_tokens,
                    force_dropout_text=force_dropout_text,
                )
                if truncation_threshold is not None:
                    state = self._apply_rollout_truncation(state, batch, truncation_threshold, truncation_weight)

            finalize_output = self._streaming_finalize(state)

            predicted_phoneme_tokens, predicted_phoneme_logits, predicted_phoneme_tokens_lens = None, None, None

            if self.model.phoneme_tokenizer is not None and state.current_phoneme_step_idx > 0:
                predicted_phoneme_tokens = torch.stack(state.pred_phoneme_tokens, dim=-1)
                predicted_phoneme_logits = torch.stack(state.pred_phoneme_logits, dim=-1)

                phoneme_start = torch.clamp(state.phoneme_prediction_start_idx, min=0)

                phoneme_end = torch.where(
                    condition=state.phoneme_prediction_end_idx >= 0,
                    input=state.phoneme_prediction_end_idx,
                    other=torch.full_like(state.phoneme_prediction_end_idx, predicted_phoneme_tokens.size(-1)),
                )
                predicted_phoneme_tokens_lens = phoneme_end - phoneme_start

            return _TeacherOutput(
                codes=finalize_output.codes,
                logits=finalize_output.logits,
                lens=finalize_output.codes_len,
                sample_weights=state.sample_weights,
                codes_lt=finalize_output.codes_lt,
                logits_lt=finalize_output.logits_lt,
                tokens_phonemes=predicted_phoneme_tokens,
                logits_phonemes=predicted_phoneme_logits,
                lens_phonemes=predicted_phoneme_tokens_lens,
            )


def _get_loss_key(
    key: str,
    lt_mode: bool = False,
) -> str:
    if not lt_mode:
        return key
    return f"{key}_lt"


_MONITORED_LOSS_KEYS: list[str] = [
    _get_loss_key("kl_loss"),
    _get_loss_key("ce_loss"),
    _get_loss_key("nrmse_loss"),
    _get_loss_key("kl_loss", lt_mode=True),
    _get_loss_key("ce_loss", lt_mode=True),
    _get_loss_key("nrmse_loss", lt_mode=True),
]


class EasyMagpieCFGDistillation(EasyMagpieTTSModel):
    """Implements online classifier-free guidance (CFG) distillation for EasyMagpieTTS."""

    def __init__(
        self,
        cfg: DictConfig,
        trainer: "Trainer" = None,
    ) -> None:
        _validate_configuration(cfg)
        super().__init__(cfg, trainer)
        self._init_extra_attributes()

        if self.alpha != 1.0:
            self._kl_criterion = KLDivergenceLoss(
                num_codebooks=self.num_audio_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=self.frame_stacking_factor,
            )
        if self.alpha != 0.0:
            self._ce_criterion = CodesCrossEntropyLoss(
                num_codebooks=self.num_audio_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=self.frame_stacking_factor,
            )
        if self.beta != 0.0:
            self._nrmse_criterion = NRMSELogitsLoss(
                num_codebooks=self.num_audio_codebooks,
                num_tokens_per_codebook=self.num_all_tokens_per_codebook,
                frame_stacking_factor=self.frame_stacking_factor,
            )
        self._teacher_model: Optional[MagpieTTSModel] = None
        self._teacher_inference_wrapper: Optional[_TeacherInferenceWrapper] = None

    def _load_teacher_model(self) -> None:
        if self._teacher_model is None:
            print("Loading teacher model from checkpoint.")
            self._teacher_model = _get_teacher_model(self.cfg).to(self.device)
            self._teacher_inference_wrapper = _TeacherInferenceWrapper(model=self._teacher_model)
            print("Teacher model loaded and frozen.")

    def on_fit_start(self) -> None:
        """See the ModelPT class docstring."""
        super().on_fit_start()
        self._load_teacher_model()

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
        defaults = vars(_DEFAULT_PARAMS)
        for k, v in defaults.items():
            setattr(self, k, self.cfg.get(k, v))

    def _get_state_dict_keys_to_exclude(self) -> list[str]:
        return super()._get_state_dict_keys_to_exclude() + _STATE_DICT_EXCLUDE_NAMES

    def _process_audio_input(
        self,
        audio_codes: Tensor,
        audio_codes_lens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Add only BOS because teacher codes already contain EOS.
        audio_codes = torch.nn.functional.pad(input=audio_codes, pad=(1, 0), value=self.audio_bos_id)
        audio_codes_lens = audio_codes_lens + 1

        audio_codes, audio_codes_lens = self.stack_codes(
            audio_codes,
            audio_codes_lens,
            self.audio_bos_id,
            self.audio_eos_id,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )
        codes_lens_gt = audio_codes_lens - 1
        codes_gt = audio_codes[:, :, 1:]
        codes_input = audio_codes[:, :, :-1]

        return codes_input, codes_gt, codes_lens_gt

    def prepare_audio_channel_embeddings(
        self,
        audio_codes_input: Tensor,
        audio_codes_lens: Tensor,
        delay: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = audio_codes_input.size(0)
        device = audio_codes_input.device
        max_delay = delay.max().item()

        embedded = self.embed_audio_tokens(audio_codes_input)
        size = (batch_size, max_delay, self.cfg.embedding_dim)
        zero_delay_tensor = torch.zeros(*size, device=device)

        embedding, lens = self.join_embeddings_temporally(
            embeddings=[zero_delay_tensor, embedded],
            lengths=[delay, audio_codes_lens],
        )
        return embedding, lens

    def _process_batch(
        self,
        batch: dict[str, Tensor | list],
        use_lt: bool,
        mode: str = "train",
        training_mode: Optional[TrainingMode] = None,
    ) -> _StudentOutput:
        if use_lt and self.local_transformer_type != LocalTransformerType.AR:
            raise ValueError(
                f"Only `LocalTransformerType.AR` is supported for local-transformer distillation, "
                f"but got `{self.local_transformer_type}`."
            )
        text_lens = batch["text_lens"]
        phoneme_tokens = batch.get("phoneme_tokens")

        selected_training_mode = training_mode

        if selected_training_mode is None:
            if mode == "train":
                selected_training_mode = random.choice(self.training_modes)
            else:
                selected_training_mode = self.training_modes[0]

        current_text_input_mode = selected_training_mode.text_input_mode
        current_streaming_speech_delay = selected_training_mode.streaming_speech_delay
        current_streaming_phonemes_delay = selected_training_mode.streaming_phonemes_delay

        dropout_text_input = random.random() < self.dropout_text_input_prob if mode == "train" else False

        context_embedding, context_lens, context_audio_codes_processed, context_audio_codes_lens_processed = (
            self.prepare_context_tensors(
                context_text_tokens=batch["context_text_tokens"],
                context_text_tokens_lens=batch["context_text_tokens_lens"],
                context_audio_codes=batch["context_audio_codes"],
                context_audio_codes_lens=batch["context_audio_codes_lens"],
                training_mode=selected_training_mode,
                dropout_conditional_input=False,
            )
        )

        text_delay = context_lens.clone()
        phoneme_delay = context_lens + current_streaming_phonemes_delay

        if current_text_input_mode == "full":
            audio_delay = context_lens + text_lens + current_streaming_speech_delay
        else:
            audio_delay = context_lens + current_streaming_speech_delay

        text_channel_embedding, text_channel_lens = self.prepare_text_channel_embeddings(
            text=batch["text"],
            text_lens=text_lens,
            delay=text_delay,
            dropout_text_input=dropout_text_input,
        )

        phoneme_channel_embedding = None
        phoneme_tokens_stacked = None
        phoneme_tokens_lens_stacked = None
        phoneme_tokens_stacked_clean = None
        phoneme_corruption_mode = None
        dropout_complete_phoneme_channel = False

        if self.phoneme_tokenizer is not None and phoneme_tokens is not None:
            apply_phoneme_corruption = (
                mode == "train" and (not dropout_text_input) and self.phoneme_corruption_type == "repeat_skip_unk"
            )
            dropout_complete_phoneme_channel = mode == "train" and (
                self.phoneme_corruption_type == "complete_channel"
                and torch.rand(1).item() < self.phoneme_corruption_batch_prob
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
                phoneme_tokens_lens=batch["phoneme_tokens_lens"],
                delay=phoneme_delay,
                apply_corruption=apply_phoneme_corruption,
                dropout_complete_phoneme_channel=dropout_complete_phoneme_channel,
            )

        audio_codes_input, audio_codes_gt, audio_codes_lens_gt = self._process_audio_input(
            audio_codes=batch["audio_codes"],
            audio_codes_lens=batch["audio_codes_lens"],
        )
        audio_channel_embedding, audio_channel_lens = self.prepare_audio_channel_embeddings(
            audio_codes_input=audio_codes_input,
            audio_codes_lens=audio_codes_lens_gt,
            delay=audio_delay,
        )

        max_channel_len = max(
            text_channel_embedding.size(1),
            audio_channel_embedding.size(1),
            phoneme_channel_embedding.size(1) if phoneme_channel_embedding is not None else 0,
        )

        if text_channel_embedding.size(1) < max_channel_len:
            padding = torch.zeros(
                text_channel_embedding.size(0),
                max_channel_len - text_channel_embedding.size(1),
                text_channel_embedding.size(2),
                device=text_channel_embedding.device,
            )
            text_channel_embedding = torch.cat([text_channel_embedding, padding], dim=1)

        if audio_channel_embedding.size(1) < max_channel_len:
            padding = torch.zeros(
                audio_channel_embedding.size(0),
                max_channel_len - audio_channel_embedding.size(1),
                audio_channel_embedding.size(2),
                device=audio_channel_embedding.device,
            )
            audio_channel_embedding = torch.cat([audio_channel_embedding, padding], dim=1)

        combined_channel_embedding = text_channel_embedding + audio_channel_embedding

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

        context_padding = torch.zeros(
            context_embedding.size(0),
            combined_channel_embedding.size(1) - context_embedding.size(1),
            context_embedding.size(2),
            device=context_embedding.device,
        )
        context_embedding_padded = torch.cat([context_embedding, context_padding], dim=1)

        full_embedding = context_embedding_padded + combined_channel_embedding

        transformer_out = self.forward(
            inputs_embeds=full_embedding,
            attention_mask=get_mask_from_lengths(combined_channel_lens),
        )
        transformer_hidden_states = transformer_out.last_hidden_state

        pred_embeddings = self.slice_sequence_embeddings(
            transformer_hidden_states,
            context_lens=audio_delay,
            target_lens=audio_codes_lens_gt,
        )

        pred_embeddings_audio = self.audio_out_projection(pred_embeddings)
        logits = self.final_proj(pred_embeddings_audio)
        logits_lt = None
        logits_phonemes = None

        if use_lt:
            _, audio_codes_gt_lt, _ = self._process_audio_input(
                audio_codes=batch["audio_codes_lt"],
                audio_codes_lens=batch["audio_codes_lens"],
            )
            logits_lt = self._lt_helper.compute_logits(
                dec_out=pred_embeddings.detach(),
                audio_codes_target=audio_codes_gt_lt,
                targets_offset_by_one=False,
            )

        if self.phoneme_tokenizer is not None and phoneme_tokens_stacked is not None:
            pred_embeddings_phoneme = self.slice_sequence_embeddings(
                transformer_hidden_states,
                context_lens=phoneme_delay,
                target_lens=phoneme_tokens_lens_stacked - 1,
            )
            logits_phonemes = self.phoneme_final_proj(pred_embeddings_phoneme)

        return _StudentOutput(
            logits=logits,
            logits_lt=logits_lt,
            logits_phonemes=logits_phonemes,
        )

    def _update_batch(
        self,
        batch: dict[str, Tensor | list],
        teacher_output: _TeacherOutput,
        use_lt: bool,
    ) -> dict[str, Tensor | list]:
        batch["audio_codes"] = teacher_output.codes
        batch["audio_codes_lens"] = teacher_output.lens

        if use_lt:
            if teacher_output.codes_lt is None:
                raise ValueError(
                    "Local-transformer distillation is enabled for this step, but `teacher_output.codes_lt` is None."
                )
            batch["audio_codes_lt"] = teacher_output.codes_lt

        return batch

    def _get_local_transformer_status(self) -> bool:
        if self.local_transformer_type == LocalTransformerType.NO_LT or not self.distill_local_transformer:
            return False

        return self.global_step >= self.lt_distillation_start_step

    def _get_local_transformer_loss_weight(self) -> float:
        if self.global_step < self.lt_distillation_start_step:
            return 0.0

        elif self.global_step >= self.lt_distillation_start_step + self.lt_distillation_ramp_len:
            return self.lt_loss_weight

        weight = self.lt_loss_weight
        weight *= (self.global_step - self.lt_distillation_start_step) / self.lt_distillation_ramp_len

        return weight

    def _compute_loss_helper(
        self,
        teacher_logits: Tensor,
        teacher_codes: Tensor,
        student_logits: Tensor,
        mask: Tensor,
        sample_weights: Optional[Tensor],
        lt_mode: bool,
    ) -> dict[str, Tensor]:
        output: dict[str, Tensor] = {}

        if self.alpha != 1.0:
            kl_loss = self._kl_criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
                sample_weights=sample_weights,
            )
            if self.distillation_temperature != 1.0:
                kl_loss = kl_loss * (self.distillation_temperature**2)

            output[_get_loss_key("kl_loss", lt_mode)] = kl_loss

        if self.alpha != 0.0:
            ce_loss = self._ce_criterion(
                predicted_logits=student_logits,
                target_codes=teacher_codes,
                mask=mask,
                sample_weights=sample_weights,
            )
            output[_get_loss_key("ce_loss", lt_mode)] = ce_loss

        kl_term = output.get(_get_loss_key("kl_loss", lt_mode), 0.0)
        ce_term = output.get(_get_loss_key("ce_loss", lt_mode), 0.0)
        loss = (1 - self.alpha) * kl_term + self.alpha * ce_term

        if self.beta > 0.0:
            nrmse_loss = self._nrmse_criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
                sample_weights=sample_weights,
            )
            output[_get_loss_key("nrmse_loss", lt_mode)] = nrmse_loss
            loss = loss + self.beta * nrmse_loss

        output[_get_loss_key("loss", lt_mode)] = loss

        return output

    def _compute_loss(
        self,
        teacher_output: _TeacherOutput,
        student_output: _StudentOutput,
        mask: Tensor,
        use_lt: bool,
    ) -> dict[str, Tensor]:
        output = self._compute_loss_helper(
            teacher_logits=teacher_output.logits,
            teacher_codes=teacher_output.codes,
            student_logits=student_output.logits,
            mask=mask,
            sample_weights=teacher_output.sample_weights,
            lt_mode=False,
        )
        backbone_loss_key = _get_loss_key("loss")

        if backbone_loss_key != "loss":
            output["loss"] = output[backbone_loss_key]

        if use_lt:
            lt_output = self._compute_loss_helper(
                teacher_logits=teacher_output.logits_lt,
                teacher_codes=teacher_output.codes_lt,
                student_logits=student_output.logits_lt,
                mask=mask,
                sample_weights=teacher_output.sample_weights,
                lt_mode=True,
            )
            output.update(lt_output)
            lt_weight = self._get_local_transformer_loss_weight()
            lt_loss_key = _get_loss_key("loss", lt_mode=True)
            output["loss"] = (1 - lt_weight) * output["loss"] + lt_weight * output[lt_loss_key]
            del output[lt_loss_key]

        return output

    def _rescale_logits(
        self,
        teacher_output: _TeacherOutput,
        student_output: _StudentOutput,
        use_lt: bool,
    ) -> tuple[_TeacherOutput, _StudentOutput]:
        if self.distillation_temperature != 1.0:
            student_output.logits = student_output.logits / self.distillation_temperature
            teacher_output.logits = teacher_output.logits / self.distillation_temperature

            if use_lt:
                student_output.logits_lt = student_output.logits_lt / self.distillation_temperature
                teacher_output.logits_lt = teacher_output.logits_lt / self.distillation_temperature

        return teacher_output, student_output

    def _process_batch_distillation(
        self,
        batch: dict[str, Tensor | list],
        mode: str = "train",
    ) -> dict[str, Tensor]:
        use_lt = self._get_local_transformer_status()

        teacher_output = self._teacher_inference_wrapper.infer_batch(
            batch=batch,
            use_lt=use_lt,
            max_decoder_steps=self.max_decoder_steps,
            temperature=self.rollout_temperature,
            topk=self.rollout_topk,
            cfg_scale=self.distillation_cfg_scale,
            phoneme_input_type="gt",
            phoneme_sampling_method="argmax",
            force_dropout_text=False,
            use_teacher_forced=False,
            truncation_threshold=self.truncation_threshold if mode == "train" else None,
            truncation_weight=self.truncation_weight if mode == "train" else None,
        )
        batch = self._update_batch(batch, teacher_output, use_lt)
        student_output = self._process_batch(batch, use_lt, mode)
        mask = get_mask_from_lengths(teacher_output.lens)
        teacher_output, student_output = self._rescale_logits(teacher_output, student_output, use_lt)
        output = self._compute_loss(teacher_output, student_output, mask, use_lt)

        return output

    def training_step(
        self,
        batch: dict[str, Tensor | list],
        batch_idx: int,
    ) -> Tensor:
        """TBD."""
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
        for key in _MONITORED_LOSS_KEYS:
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
        return loss
