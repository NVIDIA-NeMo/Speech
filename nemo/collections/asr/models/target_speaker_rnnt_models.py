# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enrollment-conditioned RNN-T models.

The model has no diarization output. A Sortformer checkpoint supplies an
overlap-aware representation, which is converted into one binary posterior:
whether the enrolled speaker is active. That posterior drives the existing
target/background speaker kernels in a cache-aware RNN-T encoder. This initial
prototype uses the offline Sortformer forward path; exportable joint streaming
state is intentionally a later step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.collections.asr.parts.mixins.multitalker_asr_mixins import SpeakerKernelMixin
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo, Serialization


class EncDecTargetSpeakerRNNTBPEModel(EncDecRNNTBPEModel, SpeakerKernelMixin):
    """Cache-aware RNN-T conditioned on a caller-supplied speaker embedding.

    During training the Lhotse target-speaker dataset supplies an enrollment
    embedding and binary target-activity labels.  At inference callers pass the
    normalized enrollment embedding to :meth:`forward`; no speaker IDs,
    diarization labels, or background transcripts are accepted.
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        super().__init__(cfg=cfg, trainer=trainer)
        if cfg.get('spk_kernel_type', None) != 'ff':
            raise ValueError("Target-speaker RNN-T currently requires spk_kernel_type='ff'.")
        if not cfg.get('spk_kernel_layers', None):
            raise ValueError("Target-speaker RNN-T requires at least one spk_kernel_layers entry.")
        self._init_speaker_kernel_config(cfg)

        target_cfg = cfg.get('target_speaker', None)
        if target_cfg is None:
            raise ValueError("Missing required model.target_speaker configuration.")
        self.target_speaker_cfg = target_cfg
        self.target_sortformer = self._build_sortformer(target_cfg)
        self.enrollment_dim = int(target_cfg.get('enrollment_dim', 192))
        hidden_size = int(self.target_sortformer.sortformer_modules.hidden_size)
        # Sortformer's anonymous diarization heads are checkpoint initialization
        # only. The target head below is the sole runtime speaker output.
        self.target_sortformer.sortformer_modules.single_hidden_to_spks = nn.Identity()
        self.target_sortformer.sortformer_modules.hidden_to_spks = nn.Identity()
        self.target_sortformer.sortformer_modules.activity_head = None
        self.enrollment_film = nn.Sequential(
            nn.LayerNorm(self.enrollment_dim),
            nn.Linear(self.enrollment_dim, 2 * hidden_size),
        )
        self.target_activity_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.target_activity_loss_weight = float(target_cfg.get('activity_loss_weight', 0.2))
        if self.target_activity_loss_weight < 0:
            raise ValueError("target_speaker.activity_loss_weight must be non-negative.")
        self.freeze_backbones_until_step = int(target_cfg.get('freeze_backbones_until_step', 0))
        self._backbones_unfrozen = self.freeze_backbones_until_step == 0
        self._active_enrollment_embeddings: Optional[torch.Tensor] = None
        self._active_target_activity: Optional[torch.Tensor] = None
        self._active_target_activity_lengths: Optional[torch.Tensor] = None
        self._last_target_activity_logits: Optional[torch.Tensor] = None

        if int(self.target_sortformer.encoder.subsampling_factor) != int(self.encoder.subsampling_factor):
            raise ValueError(
                "Target Sortformer and ASR encoders must use the same subsampling_factor; got "
                f"{self.target_sortformer.encoder.subsampling_factor} and {self.encoder.subsampling_factor}."
            )
        self._validate_frontend_compatibility()
        if not self._backbones_unfrozen:
            self._set_backbone_trainability(False)

    @staticmethod
    def _build_sortformer(target_cfg: DictConfig) -> SortformerEncLabelModel:
        model_path = target_cfg.get('sortformer_model_path', None)
        inline_cfg = target_cfg.get('sortformer_model_cfg', None)
        if bool(model_path) == bool(inline_cfg):
            raise ValueError(
                "Specify exactly one of target_speaker.sortformer_model_path or target_speaker.sortformer_model_cfg."
            )
        if model_path:
            if str(model_path).endswith('.nemo'):
                return SortformerEncLabelModel.restore_from(model_path, map_location='cpu')
            if str(model_path).endswith('.ckpt'):
                return SortformerEncLabelModel.load_from_checkpoint(model_path, map_location='cpu')
            return SortformerEncLabelModel.from_pretrained(model_path, map_location='cpu')
        model = Serialization.from_config_dict(inline_cfg)
        if not isinstance(model, SortformerEncLabelModel):
            raise TypeError(
                "target_speaker.sortformer_model_cfg must construct SortformerEncLabelModel, "
                f"got {type(model).__name__}."
            )
        return model

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get('use_lhotse'):
            from nemo.collections.asr.data.audio_to_text_lhotse_target_speaker import LhotseTargetSpeakerBpeDataset

            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseTargetSpeakerBpeDataset(cfg=config, tokenizer=self.tokenizer),
            )
        return super()._setup_dataloader_from_config(config)

    def _set_backbone_trainability(self, trainable: bool) -> None:
        self.encoder.requires_grad_(trainable)
        self.target_sortformer.requires_grad_(trainable)
        if not trainable:
            self.encoder.eval()
            self.target_sortformer.eval()

    def train(self, mode: bool = True):
        """Keep frozen pretrained backbones deterministic during conditioning warmup."""
        super().train(mode)
        if not self._backbones_unfrozen:
            self.encoder.eval()
            self.target_sortformer.eval()
        return self

    def _validate_frontend_compatibility(self) -> None:
        """Reject branch frontends whose frame grids cannot be aligned safely."""
        asr_cfg = self.preprocessor._cfg
        sortformer_cfg = self.target_sortformer.preprocessor._cfg
        mismatches = []
        for field in ('sample_rate', 'features', 'window_stride'):
            asr_value = asr_cfg.get(field, None)
            sortformer_value = sortformer_cfg.get(field, None)
            if asr_value != sortformer_value:
                mismatches.append(f"{field}: ASR={asr_value!r}, Sortformer={sortformer_value!r}")
        if mismatches:
            raise ValueError(
                "Target Sortformer and ASR preprocessors must share a frame grid; " + "; ".join(mismatches)
            )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if not self._backbones_unfrozen and self.global_step >= self.freeze_backbones_until_step:
            self._set_backbone_trainability(True)
            self._backbones_unfrozen = True
            self.encoder.train()
            self.target_sortformer.train()
        return super().on_train_batch_start(batch, batch_idx)

    def _target_activity_logits(
        self, input_signal: torch.Tensor, input_signal_length: torch.Tensor, enrollment_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if enrollment_embeddings.ndim != 2 or enrollment_embeddings.shape != (
            input_signal.shape[0],
            self.enrollment_dim,
        ):
            raise ValueError(
                "enrollment_embeddings must have shape "
                f"({input_signal.shape[0]}, {self.enrollment_dim}), got {tuple(enrollment_embeddings.shape)}."
            )
        processed, processed_lengths = self.target_sortformer.process_signal(input_signal, input_signal_length)
        states, state_lengths = self.target_sortformer.frontend_encoder(processed, processed_lengths)
        mask = self.target_sortformer.sortformer_modules.length_to_mask(state_lengths, states.shape[1])
        if self.target_sortformer.transformer_encoder is not None:
            states = self.target_sortformer.transformer_encoder(encoder_states=states, encoder_mask=mask)
        states = self.target_sortformer.sortformer_modules.upsample_hidden(states)

        gamma, beta = self.enrollment_film(F.normalize(enrollment_embeddings, dim=-1)).chunk(2, dim=-1)
        states = states * (1.0 + torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        return self.target_activity_head(states).squeeze(-1)

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        enrollment_embeddings: Optional[torch.Tensor] = None,
    ):
        if enrollment_embeddings is None:
            enrollment_embeddings = self._active_enrollment_embeddings
        if enrollment_embeddings is None:
            raise ValueError("Target-speaker RNN-T requires enrollment_embeddings for every forward pass.")
        if processed_signal is not None:
            raise ValueError("Target-speaker RNN-T currently requires waveform inputs to run the Sortformer branch.")
        if input_signal is None or input_signal_length is None:
            raise ValueError("input_signal and input_signal_length are required.")

        target_logits = self._target_activity_logits(input_signal, input_signal_length, enrollment_embeddings)
        self._last_target_activity_logits = target_logits
        target_probabilities = torch.sigmoid(target_logits)
        self.set_speaker_targets(target_probabilities, 1.0 - target_probabilities)
        return super().forward(input_signal=input_signal, input_signal_length=input_signal_length)

    def _set_target_batch_state(self, batch) -> None:
        (
            _,
            _,
            _,
            _,
            self._active_enrollment_embeddings,
            self._active_target_activity,
            self._active_target_activity_lengths,
            _,
        ) = batch

    def _clear_target_batch_state(self) -> None:
        self._active_enrollment_embeddings = None
        self._active_target_activity = None
        self._active_target_activity_lengths = None
        self._last_target_activity_logits = None
        self.clear_speaker_targets()

    def _masked_target_activity_loss(self) -> torch.Tensor:
        if (
            self._last_target_activity_logits is None
            or self._active_target_activity is None
            or self._active_target_activity_lengths is None
        ):
            return torch.zeros((), device=self.device)
        targets = self._active_target_activity.to(self._last_target_activity_logits.device)
        logits = F.interpolate(
            self._last_target_activity_logits.unsqueeze(1), size=targets.shape[1], mode='linear', align_corners=False
        ).squeeze(1)
        activity_lengths = self._active_target_activity_lengths.to(targets.device).unsqueeze(1)
        mask = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0) < activity_lengths
        losses = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (losses * mask).sum() / mask.sum().clamp_min(1)

    def add_auxiliary_losses(self, loss: torch.Tensor, reset_registry: bool = True) -> torch.Tensor:
        loss = super().add_auxiliary_losses(loss, reset_registry=reset_registry)
        if self.training and self._active_target_activity is not None:
            activity_loss = self._masked_target_activity_loss()
            self.log('train_target_activity_loss', activity_loss, on_step=True, on_epoch=False, prog_bar=False)
            loss = loss + self.target_activity_loss_weight * activity_loss
        return loss

    def training_step(self, batch, batch_nb):
        self._set_target_batch_state(batch)
        try:
            return super().training_step(batch[:4], batch_nb)
        finally:
            self._clear_target_batch_state()

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        self._set_target_batch_state(batch)
        try:
            return super().validation_pass(batch[:4], batch_idx, dataloader_idx)
        finally:
            self._clear_target_batch_state()
