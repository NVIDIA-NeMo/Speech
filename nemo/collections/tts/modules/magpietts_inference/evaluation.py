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
Evaluation wrapper for MagpieTTS generated audio.

This module provides a clean interface to the evaluation functionality,
wrapping the existing `examples.tts.magpietts.evaluate_generated_audio` module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.stats as stats

# Import the existing evaluation module
import nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio as evaluate_generated_audio
from nemo.utils import logging


@dataclass
class EvaluationConfig:
    """Configuration for audio quality evaluation.

    Attributes:
        sv_model: Speaker verification model type ("titanet" or "wavlm").
        asr_model_name: ASR model for transcription (e.g., "nvidia/parakeet-tdt-1.1b").
        language: Language code for transcription (e.g., "en").
        with_utmosv2: Whether to compute UTMOSv2 (Mean Opinion Score) metrics.
        with_fcd: Whether to compute Frechet Codec Distance metric.
        codec_model_path: Path to the audio codec model. If None, will skip computing Frechet Codec Distance metric.
        device: Device to use for running models used during evaluation.
    """

    sv_model: str = "titanet"
    asr_model_name: str = "nvidia/parakeet-tdt-1.1b"
    language: str = "en"
    with_utmosv2: bool = True
    with_fcd: bool = True
    codec_model_path: str = None
    device: str = "cuda"


def evaluate_generated_audio_dir(
    manifest_path: str,
    audio_dir: str,
    generated_audio_dir: str,
    config: EvaluationConfig,
) -> Tuple[Dict[str, float], List[dict]]:
    """Evaluate a batch of generated audio files against ground truth.

    This function computes:
    - ASR-based metrics: Character Error Rate (CER), Word Error Rate (WER)
    - Speaker similarity: Cosine similarity using speaker embeddings
    - Audio quality: UTMOSv2 scores (if enabled)
    - Freceht Codec Distance (FCD) metric (if enabled)

    Args:
        manifest_path: Path to the evaluation manifest (NDJSON format).
        audio_dir: Directory containing ground truth audio files.
        generated_audio_dir: Directory containing generated audio files.
        config: Evaluation configuration.

    Returns:
        Tuple of:
            - avg_metrics: Dictionary of averaged metrics across all files.
            - filewise_metrics: List of per-file metric dictionaries.
    """
    logging.info(f"Evaluating generated audio from {generated_audio_dir} " f"against manifest {manifest_path}")

    avg_metrics, filewise_metrics = evaluate_generated_audio.evaluate(
        manifest_path=manifest_path,
        audio_dir=audio_dir,
        generated_audio_dir=generated_audio_dir,
        language=config.language,
        sv_model_type=config.sv_model,
        asr_model_name=config.asr_model_name,
        with_utmosv2=config.with_utmosv2,
        with_fcd=config.with_fcd,
        codec_model_path=config.codec_model_path,
        device=config.device,
    )

    return avg_metrics, filewise_metrics


def compute_mean_with_confidence_interval(
    metrics_list: List[dict],
    confidence: float = 0.95,
) -> Dict[str, str]:
    """Compute mean and confidence interval for all metrics present in the dicts.

    Args:
        metrics_list: List of metric dictionaries (one per repeat/run).
        confidence: Confidence level (default: 0.95 for 95% CI).

    Returns:
        Dictionary mapping metric names to [mean, CI].
    """
    metric_keys = list(metrics_list[0].keys())

    if len(metrics_list) < 2:
        # Can't compute CI with fewer than 2 samples
        return {key: f"{metrics_list[0].get(key, 0):.4f} (single sample)" for key in metric_keys}

    results = {}
    for key in metric_keys:
        measurements = [m[key] for m in metrics_list if key in m]
        mean = np.mean(measurements)
        std_err = stats.sem(measurements)

        # t-distribution critical value for confidence interval
        t_critical = stats.t.ppf((1 + confidence) / 2, len(measurements) - 1)
        ci = std_err * t_critical

        results[key] = [mean, ci]

    return results


DEFAULT_VIOLIN_METRICS = ['cer', 'pred_context_ssim', 'utmosv2']
