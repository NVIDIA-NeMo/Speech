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

"""Validation metric calculation and aggregation for EasyMagpieTTS CFG distillation."""

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.logging_keys import LanguageMetricKey, MetricKey
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.validation_inference_engine import (
    ValidationSpeakerEmbeddings,
)
from nemo.collections.tts.parts.utils.helpers import process_text_for_cer


def collect_validation_outputs(
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


def aggregate_language_metrics(
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
            languages = item.get(LanguageMetricKey.languages)
            cer_list = item.get(LanguageMetricKey.cer_list)
            wer_list = item.get(LanguageMetricKey.wer_list)

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
        key = f"{MetricKey.cer}_lang_{lang}"
        output[key] = torch.tensor(np.mean(lang_cer[lang]), device=device)

        key = f"{MetricKey.wer}_lang_{lang}"
        output[key] = torch.tensor(np.mean(lang_wer[lang]), device=device)

    if not output:
        return None

    return output


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


def compute_validation_metrics(
    batch: dict[str, Tensor | list],
    transcripts: list[str | None],
    speaker_embeds: ValidationSpeakerEmbeddings,
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
        output[MetricKey.cer] = torch.tensor(np.mean(batch_cer), device=device)
        output[MetricKey.wer] = torch.tensor(np.mean(batch_wer), device=device)

        if use_multilingual_asr:
            langs = batch.get("languages", ["en"] * bs)
            output[LanguageMetricKey.languages] = [langs[i] for i in range(bs) if transcripts[i] is not None]
            output[LanguageMetricKey.cer_list] = batch_cer
            output[LanguageMetricKey.wer_list] = batch_wer

    if batch_ssim:
        output[MetricKey.ssim] = torch.tensor(np.mean(batch_ssim), device=device)

    if batch_utmos:
        output[MetricKey.utmos] = torch.tensor(np.mean(batch_utmos), device=device)

    return output
