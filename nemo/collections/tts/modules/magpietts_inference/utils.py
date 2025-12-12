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
"""
Utility functions for MagpieTTS model loading and configuration.

This module provides helpers for:
- Loading models from checkpoints (.ckpt) or NeMo archives (.nemo)
- Updating legacy configurations for backward compatibility
- Checkpoint state dict transformations
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.tts.models import MagpieTTSModel
from nemo.utils import logging


@dataclass
class ModelLoadConfig:
    """Configuration for loading a MagpieTTS model.

    Attributes:
        hparams_file: Path to the hparams.yaml file (required with checkpoint_file).
        checkpoint_file: Path to the .ckpt file (required with hparams_file).
        nemo_file: Path to the .nemo archive (alternative to hparams + checkpoint).
        codecmodel_path: Path to the audio codec model.
        legacy_codebooks: Use legacy codebook indices for old checkpoints.
        legacy_text_conditioning: Use legacy text conditioning for old checkpoints.
        hparams_from_wandb: Whether hparams file is from wandb export.
    """

    hparams_file: Optional[str] = None
    checkpoint_file: Optional[str] = None
    nemo_file: Optional[str] = None
    codecmodel_path: Optional[str] = None
    legacy_codebooks: bool = False
    legacy_text_conditioning: bool = False
    hparams_from_wandb: bool = False

    def validate(self) -> None:
        """Validate that the configuration is complete and consistent."""
        has_ckpt_mode = self.hparams_file is not None and self.checkpoint_file is not None
        has_nemo_mode = self.nemo_file is not None

        if not (has_ckpt_mode or has_nemo_mode):
            raise ValueError(
                "Must provide either (hparams_file + checkpoint_file) or nemo_file. "
                f"Got: hparams_file={self.hparams_file}, checkpoint_file={self.checkpoint_file}, "
                f"nemo_file={self.nemo_file}"
            )

        if has_ckpt_mode and has_nemo_mode:
            logging.warning(
                "Both checkpoint mode and nemo_file provided. Using checkpoint mode (hparams_file + checkpoint_file)."
            )


def update_config_for_inference(
    model_cfg: DictConfig,
    codecmodel_path: Optional[str],
    legacy_codebooks: bool = False,
    legacy_text_conditioning: bool = False,
) -> Tuple[DictConfig, Optional[int]]:
    """Update model configuration for inference, handling backward compatibility.

    This function transforms legacy configuration options to their modern equivalents
    and disables training-specific settings. The function updates the model configuration in place and also returns.

    Args:
        model_cfg: The model configuration dictionary.
        codecmodel_path: Path to the codec model.
        legacy_codebooks: Whether to use legacy codebook token indices.
        legacy_text_conditioning: Whether to use legacy text conditioning.

    Returns:
        Tuple of (updated config, sample_rate from config if present).
    """
    model_cfg.codecmodel_path = codecmodel_path

    # Update text tokenizer paths for backward compatibility
    if hasattr(model_cfg, 'text_tokenizer'):
        model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
        model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
        model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0

    # Disable training datasets
    model_cfg.train_ds = None
    model_cfg.validation_ds = None
    model_cfg.legacy_text_conditioning = legacy_text_conditioning

    # Rename legacy t5 encoder/decoder to current names
    if "t5_encoder" in model_cfg:
        model_cfg.encoder = model_cfg.t5_encoder
        del model_cfg.t5_encoder
    if "t5_decoder" in model_cfg:
        model_cfg.decoder = model_cfg.t5_decoder
        del model_cfg.t5_decoder

    # Remove deprecated decoder args
    if hasattr(model_cfg, 'decoder') and hasattr(model_cfg.decoder, 'prior_eps'):
        del model_cfg.decoder.prior_eps

    # Handle legacy local transformer naming
    if hasattr(model_cfg, 'use_local_transformer') and model_cfg.use_local_transformer:
        model_cfg.local_transformer_type = "autoregressive"
        del model_cfg.use_local_transformer

    # Handle legacy downsample_factor -> frame_stacking_factor rename
    if hasattr(model_cfg, 'downsample_factor'):
        model_cfg.frame_stacking_factor = model_cfg.downsample_factor
        del model_cfg.downsample_factor

    # Handle legacy codebook indices
    if legacy_codebooks:
        logging.warning(
            "Using legacy codebook indices for backward compatibility. "
            "This should only be used with old checkpoints."
        )
        num_audio_tokens = model_cfg.num_audio_tokens_per_codebook
        model_cfg.forced_num_all_tokens_per_codebook = num_audio_tokens
        model_cfg.forced_audio_eos_id = num_audio_tokens - 1
        model_cfg.forced_audio_bos_id = num_audio_tokens - 2

        if model_cfg.model_type == 'decoder_context_tts':
            model_cfg.forced_context_audio_eos_id = num_audio_tokens - 3
            model_cfg.forced_context_audio_bos_id = num_audio_tokens - 4
            model_cfg.forced_mask_token_id = num_audio_tokens - 5
        else:
            model_cfg.forced_context_audio_eos_id = num_audio_tokens - 1
            model_cfg.forced_context_audio_bos_id = num_audio_tokens - 2

    # Extract and remove sample_rate (now in model class)
    sample_rate = None
    if hasattr(model_cfg, 'sample_rate'):
        sample_rate = model_cfg.sample_rate
        del model_cfg.sample_rate

    return model_cfg, sample_rate


def update_checkpoint_state_dict(state_dict: dict) -> dict:
    """Transform checkpoint state dict for backward compatibility.

    Renames legacy t5_encoder/t5_decoder keys to encoder/decoder.

    Args:
        state_dict: The original state dictionary from the checkpoint.

    Returns:
        Updated state dictionary with renamed keys.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if 't5_encoder' in key:
            new_key = key.replace('t5_encoder', 'encoder')
        elif 't5_decoder' in key:
            new_key = key.replace('t5_decoder', 'decoder')
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def load_magpie_model(config: ModelLoadConfig, device: str = "cuda") -> Tuple[MagpieTTSModel, str]:
    """Load a MagpieTTS model from checkpoint or NeMo archive.

    Supports two loading modes:
    1. Checkpoint mode: hparams.yaml + .ckpt file
    2. NeMo mode: .nemo archive file

    Args:
        config: Model loading configuration.
        device: Device to load the model onto ("cuda" or "cpu").

    Returns:
        Tuple of (loaded model, checkpoint name for output labeling).

    Raises:
        ValueError: If configuration is invalid or sample rates don't match.
    """
    config.validate()

    if config.hparams_file is not None and config.checkpoint_file is not None:
        # Mode 1: Load from hparams + checkpoint
        model_cfg = OmegaConf.load(config.hparams_file)

        # Handle different config structures
        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg
        if config.hparams_from_wandb:
            model_cfg = model_cfg.value

        with open_dict(model_cfg):
            model_cfg, cfg_sample_rate = update_config_for_inference(
                model_cfg,
                config.codecmodel_path,
                config.legacy_codebooks,
                config.legacy_text_conditioning,
            )

        model = MagpieTTSModel(cfg=model_cfg)
        model.use_kv_cache_for_inference = True

        # Load weights
        logging.info(f"Loading weights from checkpoint: {config.checkpoint_file}")
        ckpt = torch.load(config.checkpoint_file, weights_only=False)
        state_dict = update_checkpoint_state_dict(ckpt['state_dict'])
        model.load_state_dict(state_dict)

        checkpoint_name = os.path.basename(config.checkpoint_file).replace(".ckpt", "")

    else:
        # Mode 2: Load from .nemo archive
        logging.info(f"Loading model from NeMo archive: {config.nemo_file}")
        model_cfg = MagpieTTSModel.restore_from(config.nemo_file, return_config=True)

        with open_dict(model_cfg):
            model_cfg, cfg_sample_rate = update_config_for_inference(
                model_cfg,
                config.codecmodel_path,
                config.legacy_codebooks,
                config.legacy_text_conditioning,
            )

        model = MagpieTTSModel.restore_from(config.nemo_file, override_config_path=model_cfg)
        model.use_kv_cache_for_inference = True
        checkpoint_name = os.path.basename(config.nemo_file).replace(".nemo", "")

    # Validate sample rate
    if cfg_sample_rate is not None and cfg_sample_rate != model.sample_rate:
        raise ValueError(f"Sample rate mismatch: config has {cfg_sample_rate}, model has {model.sample_rate}")

    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    logging.info("Model loaded and ready for inference.")

    return model, checkpoint_name


def get_experiment_name_from_checkpoint_path(checkpoint_path: str) -> str:
    """Extract experiment name from checkpoint path.

    Assumes directory structure: `exp_name/checkpoints/checkpoint_name.ckpt`

    Args:
        checkpoint_path: Full path to the checkpoint file.

    Returns:
        The experiment name (parent directory of checkpoints folder).
    """
    return os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
