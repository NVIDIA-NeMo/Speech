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
Utility functions for MagpieTTS model loading, configuration and evaluation.

This module provides helpers for:
- Loading models from checkpoints (.ckpt) or NeMo archives (.nemo)
- Updating legacy configurations for backward compatibility
- Checkpoint state dict transformations
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.models import EasyMagpieTTSInferenceModel, MagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_inference import EasyModelInferenceParameters
from nemo.collections.tts.models.magpietts import ModelInferenceParameters
from nemo.collections.tts.modules.magpietts_inference.inference import (
    BaseInferenceRunner,
    EasyMagpieInferenceConfig,
    EasyMagpieInferenceRunner,
    EasyMagpieMultiturnUserAudioInferenceConfig,
    EasyMagpieMultiturnUserAudioInferenceRunner,
    MagpieInferenceConfig,
    MagpieInferenceRunner,
)
from nemo.collections.tts.modules.magpietts_modules import EOSDetectionMethod
from nemo.utils import logging


def compute_ffn_flops_per_token(
    d_model: int,
    d_ffn: int,
    is_moe: bool = False,
    num_experts: Optional[int] = None,
    top_k_experts: Optional[int] = None,
    has_gated_linear: bool = False,
) -> dict:
    """Compute inference FLOPs per token for the FFN layer (dense or MoE).

    Returns a structured breakdown (expert FLOPs, router FLOPs, etc.) for
    architecture logging. Per-token, inference-only (2x multiply-add), includes
    router overhead, and supports both dense and MoE.

    Note:
      - ``nemo.utils.flops_formulas.moe_mlp_flops_calculator`` computes similar
        MoE MLP math but targets whole-model training FLOPs (per-sequence,
        6x multiplier for forward + backward wgrad + backward dgrad, no router
        FLOPs, MoE-only).
      - ``nemo.lightning.pytorch.callbacks.FLOPsMeasurementCallback`` wraps those
        formulas as a runtime training callback.

    Args:
        d_model: Model dimension (hidden_size).
        d_ffn: FFN hidden dimension (per expert for MoE, total for dense).
        is_moe: Whether this is an MoE layer.
        num_experts: Number of experts (required if is_moe=True).
        top_k_experts: Number of experts activated per token (required if is_moe=True).
        has_gated_linear: Whether FFN uses gated linear units (SwiGLU).

    Returns:
        dict with FLOPs breakdown (inference FLOPs only, not training).
    """
    if is_moe:
        if num_experts is None or top_k_experts is None:
            raise ValueError("num_experts and top_k_experts are required when is_moe=True")
        if top_k_experts > num_experts:
            raise ValueError(f"top_k_experts ({top_k_experts}) must be <= num_experts ({num_experts})")

        # MoE FFN FLOPs (inference only)
        # Each expert: d_model -> d_ffn -> d_model
        gated_multiplier = 2 if has_gated_linear else 1
        flops_fc1 = top_k_experts * d_model * d_ffn * gated_multiplier  # Projection (possibly gated)
        flops_fc2 = top_k_experts * d_ffn * d_model  # Output projection
        expert_flops = 2 * (flops_fc1 + flops_fc2)  # 2x for multiply-add

        # Router FLOPs: d_model -> num_experts linear + top-k selection
        router_flops = 2 * d_model * num_experts + num_experts  # Linear + topk overhead

        total_flops = expert_flops + router_flops

        return {
            'ffn_type': 'MoE',
            'expert_flops_per_token': expert_flops,
            'router_flops_per_token': router_flops,
            'total_flops_per_token': total_flops,
            'num_experts': num_experts,
            'active_experts_per_token': top_k_experts,
            'has_gated_linear': has_gated_linear,
        }
    else:
        # Dense FFN FLOPs (inference only)
        # FFN: d_model -> d_ffn -> d_model
        gated_multiplier = 2 if has_gated_linear else 1
        flops_fc1 = d_model * d_ffn * gated_multiplier
        flops_fc2 = d_ffn * d_model
        total_flops = 2 * (flops_fc1 + flops_fc2)  # 2x for multiply-add

        return {
            'ffn_type': 'Dense',
            'total_flops_per_token': total_flops,
            'has_gated_linear': has_gated_linear,
        }


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
        phoneme_tokenizer_path: Override path to the phoneme tokenizer file (EasyMagpieTTS only).
        disable_cas_for_context_text: Skip CAS embeddings for context text in legacy EasyMagpieTTS models.
    """

    hparams_file: Optional[str] = None
    checkpoint_file: Optional[str] = None
    nemo_file: Optional[str] = None
    codecmodel_path: Optional[str] = None
    legacy_codebooks: bool = False
    legacy_text_conditioning: bool = False
    hparams_from_wandb: bool = False
    phoneme_tokenizer_path: Optional[str] = None
    disable_cas_for_context_text: bool = False

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

    # The versioned tokenizer fields (charset_version / punct_version / locale_specific_punct) need no
    # migration here: the model's __init__ resolves them from the config's own ``nemo_version`` stamp
    # and pins the result, and every path below constructs the model from this config afterwards.

    # Update text tokenizer paths for backward compatibility
    if hasattr(model_cfg, 'text_tokenizer'):
        model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv26.07.txt"
        model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
        model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
    if hasattr(model_cfg, 'text_tokenizers'):
        for tok_name in model_cfg.text_tokenizers:
            tok_cfg = model_cfg.text_tokenizers[tok_name]
            if (
                hasattr(tok_cfg, 'g2p')
                and tok_cfg.g2p.get('phoneme_dict') == "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
            ):
                tok_cfg.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv26.07.txt"

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
        ckpt = torch.load(config.checkpoint_file)
        state_dict = update_checkpoint_state_dict(ckpt['state_dict'])
        model.load_state_dict(state_dict)

        checkpoint_name = os.path.basename(config.checkpoint_file).replace(".ckpt", "")

    else:
        if config.nemo_file.startswith("nvidia/"):  # TODO @xueyang: why ignore `update_config_for_inference`?
            model = MagpieTTSModel.from_pretrained(config.nemo_file)
            model.use_kv_cache_for_inference = True
            checkpoint_name = config.nemo_file.split("/")[-1]
            cfg_sample_rate = None
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


def load_easy_magpie_model(config: ModelLoadConfig, device: str = "cuda") -> Tuple[EasyMagpieTTSInferenceModel, str]:
    """Load an EasyMagpieTTSInferenceModel (decoder-only) from checkpoint or NeMo archive.

    Uses the inference-only base class rather than the full training model,
    which avoids pulling in training-specific dependencies.

    Supports two loading modes:
    1. Checkpoint mode: hparams.yaml + .ckpt file
    2. NeMo mode: .nemo archive file

    Args:
        config: Model loading configuration.
        device: Device to load the model onto ("cuda" or "cpu").

    Returns:
        Tuple of (loaded model, checkpoint name for output labeling).

    Raises:
        ValueError: If configuration is invalid.
    """
    config.validate()

    if config.hparams_file is not None and config.checkpoint_file is not None:
        model_cfg = OmegaConf.load(config.hparams_file)

        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg
        if config.hparams_from_wandb:
            model_cfg = model_cfg.value

        with open_dict(model_cfg):
            model_cfg.codecmodel_path = config.codecmodel_path
            model_cfg.train_ds = None
            model_cfg.validation_ds = None
            model_cfg.run_val_inference = False
            model_cfg.use_utmos = False
            model_cfg.use_meta_init_for_decoder = True
            # Some legacy EasyMagpieTTS models trained context text without CAS embeddings.
            if config.disable_cas_for_context_text:
                model_cfg.disable_cas_for_context_text = True
            if config.phoneme_tokenizer_path and hasattr(model_cfg, 'phoneme_tokenizer'):
                model_cfg.phoneme_tokenizer.tokenizer_path = config.phoneme_tokenizer_path

        model = EasyMagpieTTSInferenceModel(cfg=model_cfg)

        logging.info(f"Loading weights from checkpoint: {config.checkpoint_file}")
        ckpt = torch.load(config.checkpoint_file)
        state_dict = ckpt['state_dict']
        model.load_state_dict(state_dict)

        checkpoint_name = os.path.basename(config.checkpoint_file).replace(".ckpt", "")
    else:
        if config.nemo_file.startswith("nvidia/"):
            model = EasyMagpieTTSInferenceModel.from_pretrained(config.nemo_file)
            checkpoint_name = config.nemo_file.split("/")[-1]
        else:
            logging.info(f"Loading model from NeMo archive: {config.nemo_file}")
            model_cfg = EasyMagpieTTSInferenceModel.restore_from(config.nemo_file, return_config=True)

            with open_dict(model_cfg):
                model_cfg.codecmodel_path = config.codecmodel_path
                model_cfg.train_ds = None
                model_cfg.validation_ds = None
                # Some legacy EasyMagpieTTS models trained context text without CAS embeddings.
                if config.disable_cas_for_context_text:
                    model_cfg.disable_cas_for_context_text = True
                if config.phoneme_tokenizer_path and hasattr(model_cfg, 'phoneme_tokenizer'):
                    model_cfg.phoneme_tokenizer.tokenizer_path = config.phoneme_tokenizer_path
                # Override target so restore_from instantiates the inference class,
                # not the training subclass stored in the .nemo config.
                model_cfg.target = 'nemo.collections.tts.models.easy_magpietts_inference.EasyMagpieTTSInferenceModel'

            model = EasyMagpieTTSInferenceModel.restore_from(config.nemo_file, override_config_path=model_cfg)
            checkpoint_name = os.path.basename(config.nemo_file).replace(".nemo", "")

    model.to(device)
    model.eval().float()
    logging.info("EasyMagpieTTS model loaded and ready for inference.")

    return model, checkpoint_name


def _log_transformer_component(name: str, cfg: DictConfig, use_moe: bool = False) -> dict:
    """Log architecture info for a single transformer component and return its FLOPs metrics.

    Args:
        name: Component name (e.g., "encoder", "decoder", "context_encoder").
        cfg: The transformer component's configuration.
        use_moe: Whether this component uses Mixture-of-Experts.

    Returns:
        FLOPs metrics dict from compute_ffn_flops_per_token.
    """
    d_model = cfg.d_model
    d_ffn = cfg.d_ffn

    if use_moe:
        num_experts = cfg.num_experts
        top_k_experts = cfg.top_k_experts
        routing_strategy = cfg.routing_strategy

        logging.info(f"{name.upper()}: MoE ENABLED")
        logging.info(f"  - Experts: {num_experts}")
        logging.info(f"  - Top-k: {top_k_experts}")
        logging.info(f"  - d_model: {d_model}")
        logging.info(f"  - d_ffn per expert: {d_ffn}")
        logging.info(f"  - Routing strategy: {routing_strategy}")
        if routing_strategy == 'sinkhorn':
            logging.info("    (Note: Sinkhorn only used in training; inference uses softmax for speed)")

        has_gated_linear = getattr(cfg, 'has_gated_linear', False)

        flops_info = compute_ffn_flops_per_token(
            d_model=d_model,
            d_ffn=d_ffn,
            is_moe=True,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            has_gated_linear=has_gated_linear,
        )

        # Expert params: proj (d_model -> d_ffn) + o_net (d_ffn -> d_model), plus gate if gated
        num_projections = 3 if has_gated_linear else 2
        params_per_expert = num_projections * d_model * d_ffn
        total_params_per_layer = num_experts * params_per_expert
        active_params_per_token = top_k_experts * params_per_expert

        logging.info(f"  - Params per expert: ~{params_per_expert:,} ({num_projections} projections)")
        logging.info(f"  - Params per layer (all experts): ~{total_params_per_layer:,}")
        logging.info(f"  - Active params per token (top-{top_k_experts}): ~{active_params_per_token:,}")
        logging.info(f"  - FLOPs per token (experts): ~{flops_info['expert_flops_per_token']:,}")
        logging.info(f"  - FLOPs per token (router): ~{flops_info['router_flops_per_token']:,}")
        logging.info(f"  - FLOPs per token (total): ~{flops_info['total_flops_per_token']:,}")

        # Compare to dense baseline using the standard transformer convention (d_ffn=4*d_model).
        # Note: this assumes the dense model it replaces uses d_ffn=4*d_model. If the actual dense
        # baseline uses a different d_ffn, adjust accordingly.
        dense_baseline_d_ffn = 4 * d_model
        dense_flops_info = compute_ffn_flops_per_token(d_model=d_model, d_ffn=dense_baseline_d_ffn, is_moe=False)
        flops_reduction = dense_flops_info['total_flops_per_token'] / flops_info['total_flops_per_token']
        logging.info(f"  - FLOPs reduction vs dense (d_ffn={dense_baseline_d_ffn}): ~{flops_reduction:.1f}x")

        return flops_info
    else:
        has_gated_linear = getattr(cfg, 'has_gated_linear', False)

        logging.info(f"{name.upper()}: Dense (no MoE)")
        logging.info(f"  - d_model: {d_model}")
        logging.info(f"  - d_ffn: {d_ffn}")

        flops_info = compute_ffn_flops_per_token(
            d_model=d_model, d_ffn=d_ffn, is_moe=False, has_gated_linear=has_gated_linear
        )
        logging.info(f"  - FLOPs per token: ~{flops_info['total_flops_per_token']:,}")

        return flops_info


def log_model_architecture_summary(model) -> Tuple[str, Dict[str, dict]]:
    """Log model architecture summary including MoE configuration.

    Detects and logs MoE configuration for each transformer component,
    computing FLOPs metrics and parameter counts. Gracefully handles
    decoder-only models (EasyMagpieTTSInferenceModel) that use HuggingFace/Nemotron
    decoders without the d_model/d_ffn config structure.

    Args:
        model: Loaded MagpieTTS or EasyMagpieTTS model.

    Returns:
        Tuple of:
            - moe_info: String for checkpoint naming (e.g., "MoE_8x2_d2048_softmax_"), empty for dense models
            - flops_per_component: Dict mapping component name (e.g., "decoder") to its FLOPs metrics dict
    """
    logging.info("=" * 60)
    logging.info("MODEL ARCHITECTURE SUMMARY")
    logging.info("=" * 60)

    flops_per_component: Dict[str, dict] = {}
    use_moe = getattr(model.cfg, 'use_moe', False)

    # Log optional encoder if present (encoder-decoder models)
    if hasattr(model.cfg, 'encoder') and hasattr(model.cfg.encoder, 'd_model'):
        flops_per_component['encoder'] = _log_transformer_component('encoder', model.cfg.encoder)

    # Log optional context_encoder if present
    if hasattr(model.cfg, 'context_encoder') and hasattr(model.cfg.context_encoder, 'd_model'):
        flops_per_component['context_encoder'] = _log_transformer_component(
            'context_encoder', model.cfg.context_encoder
        )

    # Decoder -- only log detailed FLOPs for encoder-decoder models whose
    # decoder config exposes d_model/d_ffn.  Decoder-only models (EasyMagpieTTS)
    # use HuggingFace or Nemotron decoders with a different config shape.
    decoder_cfg = getattr(model.cfg, 'decoder', None)
    if decoder_cfg is not None and hasattr(decoder_cfg, 'd_model'):
        flops_per_component['decoder'] = _log_transformer_component('decoder', decoder_cfg, use_moe=use_moe)
    else:
        logging.info("DECODER: detailed FLOPs logging not available for this model type")

    # Build MoE info string for checkpoint naming
    moe_info = ""
    if use_moe and decoder_cfg is not None and hasattr(decoder_cfg, 'num_experts'):
        moe_info = (
            f"decoder-MoE_{decoder_cfg.num_experts}x{decoder_cfg.top_k_experts}"
            f"_d{decoder_cfg.d_ffn}_{decoder_cfg.routing_strategy}_"
        )

    # Log MoE inference notes if any component uses MoE
    moe_components = {name: flops for name, flops in flops_per_component.items() if flops.get('ffn_type') == 'MoE'}
    if moe_components:
        logging.info("")
        logging.info("INFERENCE MODE NOTES:")
        for name, flops in moe_components.items():
            logging.info(
                f"  - {name}: only top-{flops['active_experts_per_token']} of "
                f"{flops['num_experts']} experts activated per token"
            )
        logging.info("  - Sinkhorn routing falls back to softmax (no iterative balancing)")

    logging.info("=" * 60)

    return moe_info, flops_per_component


def get_experiment_name_from_checkpoint_path(checkpoint_path: str) -> str:
    """Extract experiment name from checkpoint path.

    Assumes directory structure: `exp_name/checkpoints/checkpoint_name.ckpt`

    Args:
        checkpoint_path: Full path to the checkpoint file.

    Returns:
        The experiment name (parent directory of checkpoints folder).
    """
    return os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))


def parse_layer_list(layer_str: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated list of layer indices."""
    if layer_str is None:
        return None
    return [int(layer.strip()) for layer in layer_str.split(",")]


def write_csv_header_if_needed(csv_path: str, header: str) -> None:
    """Write CSV header if file doesn't exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(header + "\n")


def append_metrics_to_csv(csv_path: str, checkpoint_name: str, dataset: str, metrics: dict) -> None:
    """Append metrics to a CSV file."""
    values = [
        checkpoint_name,
        dataset,
        metrics.get('cer_filewise_avg', ''),
        metrics.get('wer_filewise_avg', ''),
        metrics.get('cer_cumulative', ''),
        metrics.get('wer_cumulative', ''),
        metrics.get('cer_pred_gt_audio_filewise_avg', ''),
        metrics.get('cer_pred_gt_audio_cumulative', ''),
        metrics.get('wer_pred_gt_audio_filewise_avg', ''),
        metrics.get('wer_pred_gt_audio_cumulative', ''),
        metrics.get('ssim_pred_gt_avg', ''),
        metrics.get('ssim_pred_context_avg', ''),
        metrics.get('ssim_gt_context_avg', ''),
        metrics.get('ssim_pred_gt_avg_alternate', ''),
        metrics.get('ssim_pred_context_avg_alternate', ''),
        metrics.get('ssim_gt_context_avg_alternate', ''),
        metrics.get('esim_pred_gt_avg', ''),
        metrics.get('ems_pred_gt_avg', ''),
        metrics.get('pitch_distance_avg', ''),
        metrics.get('intensity_distance_avg', ''),
        metrics.get('speech_rate_distance_avg', ''),
        metrics.get('cer_gt_audio_cumulative', ''),
        metrics.get('wer_gt_audio_cumulative', ''),
        metrics.get('utmosv2_avg', ''),
        metrics.get('total_gen_audio_seconds', ''),
        metrics.get('frechet_codec_distance', ''),
        metrics.get('eou_cutoff_rate', ''),
        metrics.get('eou_silence_rate', ''),
        metrics.get('eou_noise_rate', ''),
        metrics.get('eou_error_rate', ''),
        metrics.get('katakana_cer_filewise_avg', ''),
        metrics.get('katakana_cer_cumulative', ''),
    ]
    with open(csv_path, "a") as f:
        f.write(",".join(str(v) for v in values) + "\n")
    logging.info(f"Metrics appended to: {csv_path}")


def _mean_finite(values: list):
    vals = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            vals.append(value)
    return None if not vals else float(np.mean(vals))


def _enrich_filewise_metrics_with_manifest(filewise_metrics: list, manifest_path: str) -> list:
    """Attach multiturn manifest metadata back to evaluator filewise rows.

    evaluate_generated_audio_dir() returns one filewise row per generated turn,
    but the filtered row does not preserve source_sample_idx/turn_id. The
    generated multiturn manifest has the same order as predicted_audio_*.wav, so
    we merge by list index before grouping.
    """
    if manifest_path is None or not os.path.exists(manifest_path):
        logging.warning(f"Could not enrich multiturn filewise metrics; manifest missing: {manifest_path}")
        return filewise_metrics

    manifest_records = read_manifest(manifest_path)
    if len(manifest_records) != len(filewise_metrics):
        logging.warning(
            "Could not safely enrich multiturn filewise metrics; "
            f"manifest rows={len(manifest_records)} filewise rows={len(filewise_metrics)} "
            f"manifest_path={manifest_path}"
        )
        return filewise_metrics

    enriched = []
    for row, record in zip(filewise_metrics, manifest_records):
        new_row = dict(row)
        for key in [
            "source_sample_idx",
            "turn_id",
            "speaker",
            "rank",
            "rank_local_idx",
            "audio_filepath",
            "context_audio_filepath",
            "predicted_phoneme_text",
            "predicted_phoneme_tokens",
            "predicted_phoneme_token_labels",
        ]:
            if key in record and key not in new_row:
                new_row[key] = record[key]
        enriched.append(new_row)

    return enriched


def _group_multiturn_filewise_metrics_by_sample(filewise_metrics: list) -> list:
    """Group turn-level multiturn metrics into one old-style row per sample.

    Each grouped row keeps turn-by-turn CER/WER/SSIM/UTMOS/text/audio lists plus
    averaged sample-level values. Rows are sorted by averaged CER descending so
    the worst conversations/samples appear first.
    """
    grouped = {}

    for row_idx, row in enumerate(filewise_metrics):
        source_sample_idx = row.get("source_sample_idx", None)
        if source_sample_idx is None:
            source_sample_idx = row.get("speaker", None)
        if source_sample_idx is None:
            source_sample_idx = row_idx

        key = str(source_sample_idx)
        if key not in grouped:
            grouped[key] = {
                "source_sample_idx": source_sample_idx,
                "speaker": row.get("speaker", source_sample_idx),
                "rank": row.get("rank", None),
                "target_audio_path": row.get("gt_audio_filepath", row.get("audio_filepath", "")),
                "context_audio_path": row.get("context_audio_filepath", ""),
                "turn_rows": [],
            }
        grouped[key]["turn_rows"].append(row)

    grouped_rows = []
    for _, group in grouped.items():
        turns = group["turn_rows"]

        def turn_sort_key(r):
            try:
                return int(r.get("turn_id", 0))
            except (TypeError, ValueError):
                return 0

        turns = sorted(turns, key=turn_sort_key)

        cer_turns = [r.get("cer") for r in turns]
        cer_pred_gt_audio_turns = [r.get("cer_pred_gt_audio") for r in turns]
        wer_turns = [r.get("wer") for r in turns]
        wer_pred_gt_audio_turns = [r.get("wer_pred_gt_audio") for r in turns]
        pred_context_ssim_turns = [r.get("pred_context_ssim") for r in turns]
        pred_gt_ssim_turns = [r.get("pred_gt_ssim") for r in turns]
        gt_context_ssim_turns = [r.get("gt_context_ssim") for r in turns]
        pred_gt_esim_turns = [r.get("pred_gt_esim") for r in turns]
        pred_gt_ems_turns = [r.get("pred_gt_ems") for r in turns]
        pitch_distance_turns = [r.get("pitch_distance") for r in turns]
        intensity_distance_turns = [r.get("intensity_distance") for r in turns]
        speech_rate_distance_turns = [r.get("speech_rate_distance") for r in turns]
        utmosv2_turns = [r.get("utmosv2") for r in turns]
        eou_type_turns = [r.get("eou_type") for r in turns]
        eou_trailing_duration_turns = [r.get("eou_trailing_duration") for r in turns]
        eou_trail_rms_ratio_turns = [r.get("eou_trail_rms_ratio") for r in turns]
        predicted_phoneme_text_turns = [r.get("predicted_phoneme_text", "") for r in turns]
        predicted_phoneme_tokens_turns = [r.get("predicted_phoneme_tokens", []) for r in turns]
        predicted_phoneme_token_labels_turns = [r.get("predicted_phoneme_token_labels", []) for r in turns]

        grouped_rows.append(
            {
                "source_sample_idx": group["source_sample_idx"],
                "speaker": group["speaker"],
                "rank": group["rank"],
                "num_turns": len(turns),
                # Sample-level averages over all turns.
                "cer": _mean_finite(cer_turns),
                "cer_pred_gt_audio": _mean_finite(cer_pred_gt_audio_turns),
                "wer": _mean_finite(wer_turns),
                "wer_pred_gt_audio": _mean_finite(wer_pred_gt_audio_turns),
                "pred_context_ssim": _mean_finite(pred_context_ssim_turns),
                "pred_gt_ssim": _mean_finite(pred_gt_ssim_turns),
                "gt_context_ssim": _mean_finite(gt_context_ssim_turns),
                "pred_gt_esim": _mean_finite(pred_gt_esim_turns),
                "pred_gt_ems": _mean_finite(pred_gt_ems_turns),
                "pitch_distance": _mean_finite(pitch_distance_turns),
                "intensity_distance": _mean_finite(intensity_distance_turns),
                "speech_rate_distance": _mean_finite(speech_rate_distance_turns),
                "utmosv2": _mean_finite(utmosv2_turns),
                "eou_trailing_duration": _mean_finite(eou_trailing_duration_turns),
                "eou_trail_rms_ratio": _mean_finite(eou_trail_rms_ratio_turns),
                # Turn-by-turn values, old-script style.
                "turn_ids": [r.get("turn_id", i) for i, r in enumerate(turns)],
                "cer_turns": cer_turns,
                "cer_pred_gt_audio_turns": cer_pred_gt_audio_turns,
                "wer_turns": wer_turns,
                "wer_pred_gt_audio_turns": wer_pred_gt_audio_turns,
                "pred_context_ssim_turns": pred_context_ssim_turns,
                "pred_gt_ssim_turns": pred_gt_ssim_turns,
                "gt_context_ssim_turns": gt_context_ssim_turns,
                "pred_gt_esim_turns": pred_gt_esim_turns,
                "pred_gt_ems_turns": pred_gt_ems_turns,
                "pitch_distance_turns": pitch_distance_turns,
                "intensity_distance_turns": intensity_distance_turns,
                "speech_rate_distance_turns": speech_rate_distance_turns,
                "utmosv2_turns": utmosv2_turns,
                "eou_type_turns": eou_type_turns,
                "eou_trailing_duration_turns": eou_trailing_duration_turns,
                "eou_trail_rms_ratio_turns": eou_trail_rms_ratio_turns,
                "predicted_phoneme_text_turns": predicted_phoneme_text_turns,
                "predicted_phoneme_tokens_turns": predicted_phoneme_tokens_turns,
                "predicted_phoneme_token_labels_turns": predicted_phoneme_token_labels_turns,
                "reference_text": [r.get("gt_text", "") for r in turns],
                "asr_hyp": [r.get("pred_text", "") for r in turns],
                "pred_audio_paths": [r.get("pred_audio_filepath", "") for r in turns],
                "target_audio_path": group["target_audio_path"],
                "context_audio_path": group["context_audio_path"],
                "turn_metrics": turns,
            }
        )

    grouped_rows.sort(
        key=lambda r: (
            r.get("cer") is not None,
            float(r["cer"]) if r.get("cer") is not None else -1.0,
        ),
        reverse=True,
    )
    return grouped_rows


def _write_grouped_multiturn_filewise_metrics_csv(csv_path: str, grouped_rows: list) -> None:
    fieldnames = [
        "source_sample_idx",
        "speaker",
        "rank",
        "num_turns",
        "cer",
        "cer_pred_gt_audio",
        "wer",
        "wer_pred_gt_audio",
        "pred_context_ssim",
        "pred_gt_ssim",
        "gt_context_ssim",
        "pred_gt_esim",
        "pred_gt_ems",
        "pitch_distance",
        "intensity_distance",
        "speech_rate_distance",
        "utmosv2",
        "eou_trailing_duration",
        "eou_trail_rms_ratio",
        "turn_ids",
        "cer_turns",
        "cer_pred_gt_audio_turns",
        "wer_turns",
        "wer_pred_gt_audio_turns",
        "pred_context_ssim_turns",
        "pred_gt_ssim_turns",
        "gt_context_ssim_turns",
        "pred_gt_esim_turns",
        "pred_gt_ems_turns",
        "pitch_distance_turns",
        "intensity_distance_turns",
        "speech_rate_distance_turns",
        "utmosv2_turns",
        "eou_type_turns",
        "eou_trailing_duration_turns",
        "eou_trail_rms_ratio_turns",
        "target_audio_path",
        "context_audio_path",
        "pred_audio_paths",
        "reference_text",
        "asr_hyp",
        "predicted_phoneme_text_turns",
        "predicted_phoneme_tokens_turns",
        "predicted_phoneme_token_labels_turns",
    ]

    def csv_value(value):
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        if value is None:
            value = ""
        value = str(value).replace('"', '""')
        if "," in value or "\n" in value or "[" in value or "{" in value:
            value = f'"{value}"'
        return value

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in grouped_rows:
            f.write(",".join(csv_value(row.get(k, "")) for k in fieldnames) + "\n")


def _save_grouped_multiturn_filewise_metrics(
    eval_dir: str,
    dataset: str,
    repeat_idx: int,
    filewise_metrics: list,
    manifest_path: str,
) -> None:
    enriched_filewise = _enrich_filewise_metrics_with_manifest(filewise_metrics, manifest_path)
    grouped_rows = _group_multiturn_filewise_metrics_by_sample(enriched_filewise)

    json_path = os.path.join(eval_dir, f"{dataset}_grouped_filewise_metrics_{repeat_idx}.json")
    csv_path = os.path.join(eval_dir, f"{dataset}_grouped_filewise_metrics_{repeat_idx}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(grouped_rows, f, indent=4, ensure_ascii=False)

    _write_grouped_multiturn_filewise_metrics_csv(csv_path, grouped_rows)

    logging.info(f"Saved grouped multiturn filewise metrics JSON to: {json_path}")
    logging.info(f"Saved grouped multiturn filewise metrics CSV to: {csv_path}")


def create_formatted_metrics_mean_ci(metrics_mean_ci: dict) -> dict:
    """Create formatted metrics mean CI."""
    for k, v in metrics_mean_ci.items():
        if isinstance(v, list):
            mean, ci = float(v[0]), float(v[1])
            logging.info(f"Metric {k}: {mean:.4f} ± {ci:.4f}")
            metrics_mean_ci[k] = f"{mean:.4f} ± {ci:.4f}"
    return metrics_mean_ci


def filter_datasets(
    dataset_meta_info: dict,
    datasets: Optional[List[str]],
) -> List[str]:
    """Select datasets from the dataset meta info."""
    if datasets is None:
        # Dataset filtering not specified, return all datasets.
        return list(dataset_meta_info.keys())
    else:
        datasets = datasets.split(",")
        # Check if requested datasets are valid.
        for dataset in datasets:
            if dataset not in dataset_meta_info:
                raise ValueError(f"Dataset {dataset} not found in dataset meta info")
        # Return all requested datasets.
        return datasets


def _runner_eval_manifest_and_audio_dir(runner: BaseInferenceRunner, default_manifest: str, default_audio_dir: str):
    """Return evaluation manifest/audio dir produced by the runner, if any."""
    eval_manifest = getattr(runner, "evaluation_manifest_path", None) or default_manifest
    eval_audio_dir = getattr(runner, "evaluation_audio_dir", None) or default_audio_dir
    return eval_manifest, eval_audio_dir


def _get_torchrun_rank_info() -> Tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from torchrun/SLURM env vars.

    We intentionally do not initialize torch.distributed here. The inference
    script only needs env-based sharding, while NeMo evaluation models can run
    without distributed collectives.
    """
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    return rank, world_size, local_rank


def _configure_cuda_for_rank() -> Tuple[int, int, int]:
    rank, world_size, local_rank = _get_torchrun_rank_info()
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            torch.cuda.set_device(local_rank % device_count)
            logging.info(
                f"Using CUDA device {local_rank % device_count}; "
                f"rank={rank}, local_rank={local_rank}, world_size={world_size}"
            )
    return rank, world_size, local_rank


def _wait_for_multiturn_rank_manifests(repeat_audio_dir: str, world_size: int, timeout_sec: int = 7200) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        missing = []
        for rank in range(world_size):
            path = os.path.join(
                repeat_audio_dir,
                f"rank_{rank:04d}",
                f"multiturn_user_audio_turn_manifest_rank{rank:04d}.jsonl",
            )
            if not os.path.exists(path):
                missing.append(path)
        if not missing:
            return
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for multiturn rank manifests: {missing}")


def _copy_or_link(src: str, dst: str, required: bool = False) -> None:
    if src is None or not os.path.exists(src):
        if os.path.lexists(dst):
            os.remove(dst)
        if required:
            raise FileNotFoundError(f"Missing required merge source: {src} -> {dst}")
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.lexists(dst):
        os.remove(dst)

    # Prefer real files for evaluator inputs; broken symlinks confuse librosa/UTMOS.
    shutil.copyfile(src, dst)


def _merge_multiturn_rank_outputs(repeat_audio_dir: str, world_size: int, save_predicted_codes: bool) -> str:
    """Merge rank-local multiturn outputs into one EasyMagpie-compatible dir.

    Each rank writes local files named predicted_audio_0.wav, target_audio_0.wav,
    context_audio_0.wav, predicted_codes_0.pt, ... inside rank_XXXX/. This
    function remaps them to contiguous global indices in repeat_audio_dir/ and
    writes a merged turn-level manifest.
    """
    # clean previous merged files
    for pattern in [
        "predicted_audio_*.wav",
        "target_audio_*.wav",
        "context_audio_*.wav",
        "predicted_codes_*.pt",
    ]:
        for path in Path(repeat_audio_dir).glob(pattern):
            if path.is_symlink() or path.exists():
                path.unlink(missing_ok=True)

    merged_records = []
    global_idx = 0

    for rank in range(world_size):
        rank_dir = os.path.join(repeat_audio_dir, f"rank_{rank:04d}")
        rank_manifest = os.path.join(rank_dir, f"multiturn_user_audio_turn_manifest_rank{rank:04d}.jsonl")
        if not os.path.exists(rank_manifest):
            raise FileNotFoundError(f"Missing rank manifest: {rank_manifest}")

        with open(rank_manifest, "r", encoding="utf-8") as f:
            rank_records = [json.loads(line) for line in f if line.strip()]

        for local_idx, record in enumerate(rank_records):
            pred_src = os.path.join(rank_dir, f"predicted_audio_{local_idx}.wav")
            pred_dst = os.path.join(repeat_audio_dir, f"predicted_audio_{global_idx}.wav")
            _copy_or_link(pred_src, pred_dst, required=True)

            if save_predicted_codes:
                code_src = os.path.join(rank_dir, f"predicted_codes_{local_idx}.pt")
                code_dst = os.path.join(repeat_audio_dir, f"predicted_codes_{global_idx}.pt")
                _copy_or_link(code_src, code_dst, required=False)

            target_src = os.path.join(rank_dir, record.get("audio_filepath", f"target_audio_{local_idx}.wav"))
            target_dst = os.path.join(repeat_audio_dir, f"target_audio_{global_idx}.wav")
            _copy_or_link(target_src, target_dst, required=True)

            context_src = os.path.join(
                rank_dir,
                record.get("context_audio_filepath", f"context_audio_{local_idx}.wav"),
            )
            context_dst = os.path.join(repeat_audio_dir, f"context_audio_{global_idx}.wav")
            _copy_or_link(context_src, context_dst, required=True)

            merged = dict(record)
            merged["audio_filepath"] = f"target_audio_{global_idx}.wav"
            merged["context_audio_filepath"] = f"context_audio_{global_idx}.wav"
            merged["rank"] = rank
            merged["rank_local_idx"] = local_idx
            merged_records.append(merged)
            global_idx += 1

    merged_manifest = os.path.join(repeat_audio_dir, "multiturn_user_audio_turn_manifest.jsonl")
    with open(merged_manifest, "w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(f"Merged {len(merged_records)} multiturn turn records into {merged_manifest}")
    return merged_manifest


def _get_shared_inference_param_names() -> set:
    """Return the field names shared by ModelInferenceParameters and EasyModelInferenceParameters."""
    magpie_fields = {f.name for f in fields(ModelInferenceParameters)}
    easy_fields = {f.name for f in fields(EasyModelInferenceParameters)}
    return magpie_fields & easy_fields


def _add_inference_param_fields(
    group: argparse._ArgumentGroup,
    param_cls: type,
    skip_fields: Optional[set] = None,
    only_fields: Optional[set] = None,
) -> None:
    """Auto-generate argparse arguments from fields of a dataclass.

    Args:
        group: The argparse argument group to add arguments to.
        param_cls: The dataclass whose fields to add.
        skip_fields: Field names to skip (already added by another group).
        only_fields: If provided, only add fields whose names are in this set.
    """
    if skip_fields is None:
        skip_fields = set()
    for f in fields(param_cls):
        if f.name in skip_fields:
            continue
        if only_fields is not None and f.name not in only_fields:
            continue
        extra_args: dict = {"type": f.type}
        if f.type == bool:
            extra_args = {"action": "store_true"}
        if f.name in ("estimate_alignment_from_layers", "apply_prior_to_layers"):
            extra_args = {
                "help": "Must be a comma separate string. Not enclosed in brackets",
                "type": str,
            }
        elif f.name == "eos_detection_method":
            extra_args["choices"] = [m.value for m in EOSDetectionMethod]
        group.add_argument(f"--{f.name}", **extra_args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all model types."""

    parser.add_argument(
        '--model_type',
        type=str,
        default='magpie',
        choices=['magpie', 'easy_magpie'],
        help='Model type: "magpie" for encoder-decoder MagpieTTSModel, '
        '"easy_magpie" for decoder-only EasyMagpieTTSInferenceModel',
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Attempts to make results deterministic to the best that can be done. Used for testing',
    )

    # Model loading
    model_group = parser.add_argument_group('Model Loading')
    model_group.add_argument(
        '--hparams_files',
        type=str,
        default=None,
        help='Comma-separated paths to hparams.yaml files (use with --checkpoint_files)',
    )
    model_group.add_argument(
        '--checkpoint_files',
        type=str,
        default=None,
        help='Comma-separated paths to .ckpt files (use with --hparams_files)',
    )
    model_group.add_argument(
        '--nemo_files',
        type=str,
        default=None,
        help='Comma-separated paths to .nemo files (alternative to hparams + checkpoint)',
    )
    model_group.add_argument(
        '--codecmodel_path',
        type=str,
        required=True,
        help='Path to the audio codec model',
    )
    model_group.add_argument(
        '--hparams_file_from_wandb',
        action='store_true',
        help='Set if hparams file was exported from wandb',
    )
    model_group.add_argument(
        '--legacy_codebooks',
        action='store_true',
        help='Use legacy codebook indices (for old checkpoints)',
    )
    model_group.add_argument(
        '--legacy_text_conditioning',
        action='store_true',
        help='Use legacy text conditioning (for old checkpoints)',
    )

    # Dataset and output
    data_group = parser.add_argument_group('Dataset and Output')
    data_group.add_argument(
        '--datasets_json_path',
        type=str,
        required=True,
        default=None,
        help='Path to dataset configuration JSON file',
    )
    data_group.add_argument(
        '--datasets_base_path',
        type=Path,
        default=None,
        help='Optional base path that paths in the "datasets_json_path" file are relative to',
    )
    data_group.add_argument(
        '--datasets',
        type=str,
        default=None,
        help='Comma-separated list of dataset names to process',
    )
    data_group.add_argument(
        '--tokenizer_name',
        type=str,
        default="english_phoneme",
        help='Default tokenizer to use when a language or dataset specific tokenizer is not provided.',
    )
    data_group.add_argument('--out_dir', type=str, required=True, help='Output directory')
    data_group.add_argument('--log_exp_name', action='store_true')
    data_group.add_argument('--clean_up_disk', action='store_true')

    # Common inference parameters
    infer_group = parser.add_argument_group('Common Inference Parameters')
    infer_group.add_argument('--batch_size', type=int, default=32)
    infer_group.add_argument('--use_cfg', action='store_true', help='Enable classifier-free guidance')
    infer_group.add_argument('--use_local_transformer', action='store_true')

    # Model inference parameters shared by both MagpieTTS and EasyMagpieTTS
    shared_param_names = _get_shared_inference_param_names()
    _add_inference_param_fields(infer_group, ModelInferenceParameters, only_fields=shared_param_names)

    # Evaluation
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--run_evaluation', action='store_true', help='Run evaluation after inference')
    eval_group.add_argument('--sv_model', type=str, default="titanet", choices=["titanet", "wavlm"])
    eval_group.add_argument(
        '--asr_model_name',
        type=str,
        default='nvidia/parakeet-tdt-1.1b',
        help="ASR model to use for WER calculation, when not provided in dataset config",
    )
    eval_group.add_argument(
        '--asr_model_type',
        type=str,
        default='nemo',
        choices=['nemo', 'nemo_with_prompt', 'whisper'],
        help="Type of ASR model provided in 'asr_model_name'",
    )
    eval_group.add_argument(
        '--language', type=str, default="en", help='Language to use, when not provided in dataset config'
    )
    eval_group.add_argument(
        '--eou_model_name',
        type=str,
        default="facebook/wav2vec2-base-960h",
        help=(
            'Hugging Face model id or local path to the EoU wav2vec2 model directory. '
            'For offline use, download the model locally and pass the directory path here.'
        ),
    )
    eval_group.add_argument('--num_repeats', type=int, default=1)
    eval_group.add_argument('--confidence_level', type=float, default=0.95)
    eval_group.add_argument('--disable_utmosv2', action='store_true')
    eval_group.add_argument(
        '--with_prosody_metrics',
        action='store_true',
        help='Compute ESIM/EMS and pitch, intensity, and speech-rate distance metrics.',
    )
    eval_group.add_argument('--prosody_model_size', type=str, default="small", choices=["small", "large"])
    eval_group.add_argument(
        '--strip_text_annotations_for_metrics',
        action='store_true',
        help='Strip bracket/tag/control annotations from reference and ASR hypothesis text while computing text metrics.',
    )
    eval_group.add_argument(
        '--violin_plot_metrics',
        type=str,
        nargs='*',
        default=['cer', 'pred_context_ssim', 'utmosv2'],
    )
    eval_group.add_argument('--disable_fcd', action='store_true')
    eval_group.add_argument("--asr_batch_size", type=int, default=32)
    eval_group.add_argument("--eou_batch_size", type=int, default=32)

    # Quality targets
    target_group = parser.add_argument_group('Quality Targets')
    target_group.add_argument('--cer_target', type=float, default=None)
    target_group.add_argument('--ssim_target', type=float, default=None)


def seed_all(seed: int):
    """
    Attempts to make script deterministic
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _add_magpie_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to encoder-decoder MagpieTTSModel."""
    group = parser.add_argument_group('MagpieTTS-specific Parameters')

    # MagpieTTS-specific model inference parameters (attention prior, EOS, etc.)
    shared_param_names = _get_shared_inference_param_names()
    _add_inference_param_fields(group, ModelInferenceParameters, skip_fields=shared_param_names)

    group.add_argument('--maskgit_n_steps', type=int, default=3)
    group.add_argument('--maskgit_noise_scale', type=float, default=0.0)
    group.add_argument('--maskgit_fixed_schedule', type=int, nargs='+', default=None)
    group.add_argument(
        '--maskgit_sampling_type',
        default=None,
        choices=["default", "causal", "purity_causal", "purity_default"],
    )


def _add_easy_magpie_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to decoder-only EasyMagpieTTSInferenceModel."""
    group = parser.add_argument_group('EasyMagpieTTS-specific Parameters')
    group.add_argument(
        '--easy_magpie_inference_mode',
        type=str,
        default='single_turn',
        choices=['single_turn', 'multiturn_user_audio'],
    )
    group.add_argument('--max_eval_turns', type=int, default=6)
    group.add_argument('--no_save_debug_multiturn_audio', action='store_true')
    group.add_argument(
        '--phoneme_input_type',
        type=str,
        default='gt',
        choices=['gt', 'predicted'],
        help='Source of phoneme input for decoder-only model',
    )
    group.add_argument(
        '--phoneme_sampling_method',
        type=str,
        default='argmax',
        choices=['argmax', 'multinomial'],
        help='Sampling method for phoneme prediction',
    )
    group.add_argument('--dropout_text_input', action='store_true', help='Force dropout on text input')
    group.add_argument(
        '--phoneme_tokenizer_path',
        type=str,
        default=None,
        help='Override path to the phoneme tokenizer file (overrides the path stored in the checkpoint config)',
    )
    group.add_argument(
        '--disable_cas_for_context_text',
        action='store_true',
        help='Skip CAS embeddings for context text when loading legacy EasyMagpieTTS models',
    )


def _build_inference_params_from_args(param_cls: type, args):
    """Extract inference parameters from parsed CLI args for the given dataclass."""
    params = {}
    for f in fields(param_cls):
        arg_val = vars(args).get(f.name)
        if arg_val is not None:
            if f.name in ("estimate_alignment_from_layers", "apply_prior_to_layers"):
                params[f.name] = parse_layer_list(arg_val)
            else:
                params[f.name] = arg_val
    return param_cls.from_dict(params)


def _build_magpie_config(args) -> MagpieInferenceConfig:
    return MagpieInferenceConfig(
        model_inference_parameters=_build_inference_params_from_args(ModelInferenceParameters, args),
        batch_size=args.batch_size,
        use_cfg=args.use_cfg,
        apply_attention_prior=args.apply_attention_prior,
        use_local_transformer=args.use_local_transformer,
        maskgit_n_steps=args.maskgit_n_steps,
        maskgit_noise_scale=args.maskgit_noise_scale,
        maskgit_fixed_schedule=args.maskgit_fixed_schedule,
        maskgit_sampling_type=args.maskgit_sampling_type,
        default_tokenizer_name=args.tokenizer_name,
    )


def _build_easy_magpie_config(args) -> EasyMagpieInferenceConfig:
    cfg_cls = (
        EasyMagpieMultiturnUserAudioInferenceConfig
        if args.easy_magpie_inference_mode == 'multiturn_user_audio'
        else EasyMagpieInferenceConfig
    )
    kwargs = dict(
        model_inference_parameters=_build_inference_params_from_args(EasyModelInferenceParameters, args),
        batch_size=args.batch_size,
        use_cfg=args.use_cfg,
        use_local_transformer=args.use_local_transformer,
        phoneme_input_type=args.phoneme_input_type,
        phoneme_sampling_method=args.phoneme_sampling_method,
        dropout_text_input=args.dropout_text_input,
        default_tokenizer_name=args.tokenizer_name,
    )
    if cfg_cls is EasyMagpieMultiturnUserAudioInferenceConfig:
        kwargs.update(
            max_eval_turns=args.max_eval_turns,
            save_debug_multiturn_audio=not args.no_save_debug_multiturn_audio,
        )
    return cfg_cls(**kwargs)


def _select_runner_cls(args):
    if args.model_type == 'magpie':
        if args.easy_magpie_inference_mode != 'single_turn':
            raise ValueError('--easy_magpie_inference_mode is only supported with --model_type easy_magpie')
        return MagpieInferenceRunner
    if args.easy_magpie_inference_mode == 'multiturn_user_audio':
        return EasyMagpieMultiturnUserAudioInferenceRunner
    return EasyMagpieInferenceRunner
