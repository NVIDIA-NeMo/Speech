# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneLogger throughput policies for the ASR collection."""

from typing import Any

import torch

from nemo.lightning.speech_throughput import (
    SpeechThroughputPolicy,
    ThroughputMeasurements,
    add_audio_seconds,
    add_sum,
    audio_lengths,
    batch_value,
    first_positive,
    value_count,
)

__all__ = ["ASRThroughputPolicy", "DiarizationThroughputPolicy"]


class ASRThroughputPolicy(SpeechThroughputPolicy):
    """Measure input waveform duration and target text for ASR training."""

    name = "asr"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        add_audio_seconds(measurements, "input_audio_seconds", _audio_lengths(batch), _sample_rate(model))
        add_sum(measurements, "target_text_tokens", _text_lengths(batch))
        return measurements

    def num_examples(self, model: Any, batch: Any) -> int | None:
        del model
        lengths = _audio_lengths(batch)
        return value_count(lengths if lengths is not None else _text_lengths(batch))


class DiarizationThroughputPolicy(SpeechThroughputPolicy):
    """Measure input waveform duration for diarization training."""

    name = "diarization"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        add_audio_seconds(
            measurements,
            "input_audio_seconds",
            audio_lengths(batch),
            _sample_rate(model),
        )
        return measurements

    def num_examples(self, model: Any, batch: Any) -> int | None:
        del model
        return value_count(audio_lengths(batch))


def _audio_lengths(batch: Any) -> Any:
    if hasattr(batch, "has_processed_signal"):
        return None if batch.has_processed_signal else batch[1]
    signal = batch_value(batch, "audio", "audio_signal", "input_signal")
    lengths = batch_value(batch, "audio_lens", "audio_lengths", "audio_len", "input_length")
    if isinstance(batch, (tuple, list)) and len(batch) > 1:
        signal, lengths = batch[0], batch[1]
    if torch.is_tensor(signal) and signal.ndim <= 2:
        return lengths
    return None


def _text_lengths(batch: Any) -> Any:
    lengths = batch_value(
        batch,
        "text_token_lengths",
        "transcript_lens",
        "transcript_len",
        "prompted_transcript_lens",
        "token_lens",
    )
    if lengths is not None:
        return lengths
    if hasattr(batch, "has_processed_signal"):
        return batch[3]
    if isinstance(batch, (tuple, list)) and len(batch) > 3:
        return batch[3]
    return None


def _sample_rate(model: Any) -> float | None:
    return first_positive(
        model,
        "sampling_rate",
        "sample_rate",
        "preprocessor._sample_rate",
        "preprocessor.sample_rate",
        "preprocessor._cfg.sample_rate",
        "perception.preprocessor.featurizer.sample_rate",
        "perception.preprocessor._sample_rate",
        "perception.preprocessor._cfg.sample_rate",
        "cfg.sample_rate",
        "cfg.preprocessor.sample_rate",
    )
