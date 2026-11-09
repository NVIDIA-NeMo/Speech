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
from dataclasses import dataclass
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

from nemo.collections.tts.losses.magpietts_cfg_distillation import (
    CodesCrossEntropyLoss,
    KLDivergenceLoss,
    NRMSELogitsLoss,
)
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_inference import InferBatchOutput, TrainingMode
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation import (
    MONITORED_KEYS_TRAIN,
    MONITORED_KEYS_VALID,
    ON_EPOCH_TRAIN_KEYS,
    ROLLOUT_TIMING_KEYS,
    LossKey,
    LossMode,
    MetricKey,
    StudentOutput,
    TeacherInferenceEngine,
    TeacherOutput,
    ValidationAudioPaths,
    ValidationInferenceEngine,
    ValidationInferenceParams,
    aggregate_language_metrics,
    collect_validation_outputs,
    compute_validation_metrics,
    get_loss_key,
)
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType, remove_special_tokens
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.lightning.callback_group import CallbackGroup
from nemo.utils import logging

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


def _compute_rejection_rate(
    teacher_output: TeacherOutput,
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
        val_inference_params = ValidationInferenceParams.create_from_config(cfg)
        cfg = _disable_default_val_inference(cfg)

        super().__init__(cfg, trainer)
        self._init_extra_attributes()
        self._init_losses()

        self.audio_dir: Optional[Path] = None
        self._teacher_model: Optional[EasyMagpieTTSModel] = None
        self._teacher_inference_engine: Optional[TeacherInferenceEngine] = None
        self._val_inference_engine = ValidationInferenceEngine(params=val_inference_params)

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
            self._teacher_inference_engine = TeacherInferenceEngine(model=self._teacher_model)
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
    ) -> StudentOutput:
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

        return StudentOutput(
            logits=logits,
            logits_lt=logits_lt,
            logits_phonemes=logits_phonemes,
            audio_codes_gt_lt=batch_combined.audio_codes_gt_lt,
            audio_codes_lens_gt_lt=batch_combined.audio_codes_lens_gt_lt,
        )

    def _update_batch(
        self,
        batch: dict[str, Tensor | list],
        teacher_output: TeacherOutput,
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

    def _get_kl_criterion(self, mode: LossMode) -> Callable:
        """Return the KL-divergence criterion for the requested model component.

        Args:
            mode: Defines model component whose loss criterion is requested.

        Returns:
            Configured KL-divergence loss criterion.
        """
        return self._lt_codes_kl_criterion if mode == LossMode.local_transformer else self._codes_kl_criterion

    def _get_ce_criterion(self, mode: LossMode) -> Callable:
        """Return the cross-entropy criterion for the requested model component.

        Args:
            mode: Defines model component whose loss criterion is requested.

        Returns:
            Configured cross-entropy loss criterion.
        """
        return self._lt_codes_ce_criterion if mode == LossMode.local_transformer else self._codes_ce_criterion

    def _get_nrmse_criterion(self, mode: LossMode) -> Callable:
        """Return the NRMSE criterion for the requested model component.

        Args:
            mode: Defines model component whose loss criterion is requested.

        Returns:
            Configured normalized root-mean-square-error loss criterion.
        """
        return self._lt_codes_nrmse_criterion if mode == LossMode.local_transformer else self._codes_nrmse_criterion

    def _compute_codes_loss_helper(
        self,
        teacher_logits: Tensor,
        teacher_codes: Tensor,
        student_logits: Tensor,
        mask: Tensor,
        sample_weights: Optional[Tensor],
        mode: LossMode,
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

            output[get_loss_key(LossKey.kl_loss, mode)] = kl_loss

        if self.audio_ce_loss_weight > 0.0:
            ce_loss = self._get_ce_criterion(mode)(
                predicted_logits=student_logits,
                target_codes=teacher_codes,
                mask=mask,
                sample_weights=sample_weights,
            )
            output[get_loss_key(LossKey.ce_loss, mode)] = ce_loss

        kl_term = output.get(get_loss_key(LossKey.kl_loss, mode), 0.0)
        ce_term = output.get(get_loss_key(LossKey.ce_loss, mode), 0.0)
        loss = self.audio_kl_loss_weight * kl_term + self.audio_ce_loss_weight * ce_term

        if self.audio_nrmse_loss_weight > 0.0:
            nrmse_loss = self._get_nrmse_criterion(mode)(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
                sample_weights=sample_weights,
            )
            output[get_loss_key(LossKey.nrmse_loss, mode)] = nrmse_loss
            loss = loss + self.audio_nrmse_loss_weight * nrmse_loss

        output[get_loss_key(LossKey.loss, mode)] = loss

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
        mode = LossMode.phoneme

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

        output[get_loss_key(LossKey.loss, mode)] = loss

        return output

    def _compute_loss(
        self,
        batch: dict[str, Tensor | list],
        teacher_output: TeacherOutput,
        student_output: StudentOutput,
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
            mode=LossMode.main,
        )
        backbone_loss_key = get_loss_key(key=LossKey.loss, mode=LossMode.main)
        output["loss"] = output[backbone_loss_key]

        if use_lt:
            lt_codes_mask = get_mask_from_lengths(student_output.audio_codes_lens_gt_lt)

            lt_output = self._compute_codes_loss_helper(
                teacher_logits=teacher_output.logits_lt,
                teacher_codes=student_output.audio_codes_gt_lt,
                student_logits=student_output.logits_lt,
                mask=lt_codes_mask,
                sample_weights=teacher_output.sample_weights,
                mode=LossMode.local_transformer,
            )
            output.update(lt_output)
            lt_weight = self._get_local_transformer_loss_weight()
            lt_loss_key = get_loss_key(key=LossKey.loss, mode=LossMode.local_transformer)
            output["loss"] = (1 - lt_weight) * output["loss"] + lt_weight * output[lt_loss_key]
            del output[lt_loss_key]

        if self.phoneme_ce_loss_weight and student_output.logits_phonemes is not None:
            phonemes_output = self._compute_phoneme_loss_guidance(
                batch=batch,
                student_logits=student_output.logits_phonemes,
                sample_weights=teacher_output.sample_weights,
            )
            output.update(phonemes_output)
            phoneme_loss_key = get_loss_key(key=LossKey.loss, mode=LossMode.phoneme)
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
        teacher_output: TeacherOutput,
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
        losses. Reports generation-loop and total teacher rollout wall times in
        seconds. During training with rollout rejection enabled, also reports the
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

        for key, value in teacher_output.timing_metrics.items():
            output[key] = torch.tensor(value, device=self.device)

        if mode == "train" and self.use_rollout_rejection:
            output[MetricKey.rejection_rate.value] = _compute_rejection_rate(teacher_output)

        return output

    def training_step(
        self,
        batch: dict[str, Tensor | list],
        batch_idx: int,
    ) -> Tensor:
        """Run one CFG-distillation optimization step and log training metrics.

        Logs worker-averaged ``train/rollout_generation_time`` and ``train/rollout_total_time``
        in seconds per batch, on each step and averaged over the epoch.

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
        for key in (*MONITORED_KEYS_TRAIN, *ROLLOUT_TIMING_KEYS):
            if key in outputs:
                self.log(
                    name=f"train/{key}",
                    value=outputs[key],
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=bs,
                    on_step=True,
                    on_epoch=key in ON_EPOCH_TRAIN_KEYS,
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
    ) -> ValidationAudioPaths:
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
        paths = ValidationAudioPaths()

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
        output = compute_validation_metrics(
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

        Rollout times are averaged over validation batches and workers, in seconds.

        Args:
            val_outputs: Per-batch validation outputs from one dataloader or from all
                validation dataloaders.
            prefix: Metric namespace to use when logging values.
            aggregated: Whether the outputs represent the aggregate across validation
                dataloaders. When `True`, also logs `"val_loss"` metric.
        """
        val_loss = collect_validation_outputs(val_outputs, key="loss")

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

        for key in (*MONITORED_KEYS_VALID, *ROLLOUT_TIMING_KEYS):
            value = collect_validation_outputs(val_outputs, key)
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
        metrics = aggregate_language_metrics(val_outputs, device=self.device)

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
        # Needed due change in #15764 that added require_backward_grad_sync = False to base
        # class (EasyMagpieTTSModel)'s on_validation_epoch_start()
        if torch.distributed.is_initialized():
            self.trainer.strategy.model.require_backward_grad_sync = True
