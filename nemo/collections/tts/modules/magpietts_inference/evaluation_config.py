# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Evaluation configuration for MagpieTTS inference: the ``EvaluationConfig`` dataclass and the helpers that validate and
apply the per-dataset overrides of an evalset config entry.

This module has no heavy dependencies so that both ``evaluation`` and ``evaluate_generated_audio`` can import it
without an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

# Supported ASR backends: the values of --asr_model_type and of the "type" field of an evalset "asr_model" entry.
ASR_MODEL_TYPES = ("nemo", "nemo_with_prompt", "whisper")
# Keys of an evalset config entry that the inference/evaluation scripts read, plus "feature_dir", which shipped configs
# carry over from the training DatasetMeta schema and which is accepted but unused here. Other keys are ignored with a
# warning so that a misspelled override (e.g. "langauge") does not silently leave the CLI-level value in force.
EVALSET_ENTRY_KEYS = frozenset(
    {
        "manifest_path",
        "audio_dir",
        "feature_dir",  # accepted for compatibility with existing configs; not read by these scripts
        "tokenizer_names",
        "language",
        "asr_model",
        "strip_text_annotations_for_metrics",
    }
)


@dataclass
class EvaluationConfig:
    """Configuration for audio quality evaluation.

    Attributes:
        sv_model: Speaker verification model type ("titanet" or "wavlm").
        asr_model_name: ASR model for transcription (e.g., "nvidia/parakeet-tdt-1.1b").
        asr_model_type: ASR backend for ``asr_model_name``; one of ``ASR_MODEL_TYPES`` ("nemo", "nemo_with_prompt"
            or "whisper").
        eou_model_name: Hugging Face model id or local path to the EoU model.
        language: Language code for transcription (e.g., "en").
        with_utmosv2: Whether to compute UTMOSv2 (Mean Opinion Score) metrics.
        with_fcd: Whether to compute Frechet Codec Distance metric.
        codec_model_path: Path to the audio codec model. Required when ``with_fcd`` is True
            (``evaluate_generated_audio_dir`` raises otherwise); set ``with_fcd=False`` to skip the Frechet Codec
            Distance metric.
        with_prosody_metrics: Whether to compute ESIM/EMS plus pitch,
            intensity, and speech-rate distance metrics.
        prosody_model_size: Emotion encoder size ("small" or "large").
        strip_text_annotations_for_metrics: Whether to strip annotation/control markers (``<tag>``, ``{tag}``,
            ``[tag]``, ``--``, ``...``, ``*``) from reference and ASR hypothesis text before text metrics.
            Square-bracket spans are removed WITH their content, so datasets that mark emphasized spoken words
            as ``[word]`` must disable this per dataset via ``"strip_text_annotations_for_metrics": false`` in
            their evalset config entry (see ``resolve_evaluation_config_for_dataset``).
        device: Device to use for running models used during evaluation.
        asr_batch_size: Batch size for ASR transcription.
        eou_batch_size: Batch size for EoU classification.
    """

    sv_model: str = "titanet"
    asr_model_name: str = "nvidia/parakeet-tdt-1.1b"
    asr_model_type: str = "nemo"
    eou_model_name: str = "facebook/wav2vec2-base-960h"
    language: str = "en"
    with_utmosv2: bool = True
    with_fcd: bool = True
    codec_model_path: str = None
    with_prosody_metrics: bool = False
    prosody_model_size: str = "small"
    strip_text_annotations_for_metrics: bool = False
    device: str = "cuda"
    asr_batch_size: int = 32
    eou_batch_size: int = 32


def validate_evalset_entry(info: dict, dataset_name: Optional[str] = None) -> None:
    """Validate the optional per-dataset evaluation overrides of one evalset config entry.

    Recognized optional keys and their required types:

    - ``language``: non-empty string without surrounding whitespace (remove the key to use the CLI-level language).
    - ``asr_model``: ``{"name": <model name or .nemo path>, "type": <one of ASR_MODEL_TYPES>}``.
    - ``strip_text_annotations_for_metrics``: JSON boolean.

    Absent keys are fine. A JSON ``null`` is rejected like any other wrong type, so that a broken override cannot
    silently fall back to the CLI-level value. Used by ``load_evalset_config`` (``evaluate_generated_audio.py``) and
    by ``resolve_evaluation_config_for_dataset``.

    Args:
        info: One entry of the evalset config.
        dataset_name: Dataset name used to prefix error messages, if known.

    Raises:
        ValueError: If a recognized key has a malformed value.
    """
    prefix = f"Dataset {dataset_name}: " if dataset_name is not None else "Evalset entry: "
    if "strip_text_annotations_for_metrics" in info:
        value = info["strip_text_annotations_for_metrics"]
        if not isinstance(value, bool):
            raise ValueError(
                f"{prefix}'strip_text_annotations_for_metrics' must be a JSON boolean (true/false), got {value!r}."
            )
    if "language" in info:
        value = info["language"]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(
                f"{prefix}'language' must be a non-empty string without surrounding whitespace, such as \"en\" "
                f"(remove the key to use the CLI-level language), got {value!r}."
            )
    if "asr_model" in info:
        value = info["asr_model"]
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("name"), str)
            or not value["name"].strip()
            or value.get("type") not in ASR_MODEL_TYPES
        ):
            types = ", ".join(ASR_MODEL_TYPES)
            raise ValueError(
                f"{prefix}'asr_model' must be an object "
                f"{{\"name\": <model name or .nemo path>, \"type\": <{types}>}}, got {value!r}."
            )


def resolve_evaluation_config_for_dataset(eval_config: EvaluationConfig, dataset_meta: dict) -> EvaluationConfig:
    """Return a copy of ``eval_config`` with the per-dataset overrides of one evalset config entry applied.

    Recognized optional keys of an evalset config entry:

    - ``asr_model``: ``{"name": ..., "type": ...}`` overriding ``asr_model_name`` and ``asr_model_type``.
    - ``language``: overrides ``language``.
    - ``strip_text_annotations_for_metrics``: JSON boolean overriding the CLI-level flag. Set it to ``false`` for
      datasets whose square brackets mark emphasized *spoken* words (e.g. ``"[You] want to ski"``), so those words
      are not deleted from the CER/WER reference, and to ``true`` for datasets whose brackets are non-verbal tags
      such as ``[breath]``.

    Keys that are absent keep the value from ``eval_config``. ``eval_config`` itself is not mutated.

    Args:
        eval_config: CLI-level evaluation configuration.
        dataset_meta: One entry of the evalset config (see ``load_evalset_config``).

    Returns:
        A new ``EvaluationConfig`` with the overrides applied.

    Raises:
        ValueError: If a recognized key is malformed (see ``validate_evalset_entry``):
            ``language`` is not a non-empty string, ``asr_model`` lacks a valid
            ``name``/``type``, or ``strip_text_annotations_for_metrics`` is not a boolean. JSON ``null`` counts as
            malformed, so a broken override never silently falls back to the CLI-level value.
    """
    validate_evalset_entry(dataset_meta)
    overrides = {}
    if "asr_model" in dataset_meta:
        overrides["asr_model_name"] = dataset_meta["asr_model"]["name"]
        overrides["asr_model_type"] = dataset_meta["asr_model"]["type"]
    if "language" in dataset_meta:
        overrides["language"] = dataset_meta["language"]
    if "strip_text_annotations_for_metrics" in dataset_meta:
        overrides["strip_text_annotations_for_metrics"] = dataset_meta["strip_text_annotations_for_metrics"]
    return replace(eval_config, **overrides)
