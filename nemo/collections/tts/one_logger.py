# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneLogger throughput policies for the TTS collection."""

from typing import Any

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

__all__ = ["AudioCodecThroughputPolicy", "TTSThroughputPolicy"]


class TTSThroughputPolicy(SpeechThroughputPolicy):
    """Measure input text and target waveform duration for TTS training."""

    name = "tts"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        add_sum(measurements, "input_text_tokens", _text_lengths(batch))
        add_audio_seconds(measurements, "output_audio_seconds", audio_lengths(batch), _sample_rate(model))
        return measurements

    def num_examples(self, model: Any, batch: Any) -> int | None:
        del model
        lengths = _text_lengths(batch)
        return value_count(lengths if lengths is not None else audio_lengths(batch))


class AudioCodecThroughputPolicy(SpeechThroughputPolicy):
    """Measure input waveform duration for audio codec training."""

    name = "audio_codec"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        add_audio_seconds(measurements, "input_audio_seconds", audio_lengths(batch), _sample_rate(model))
        return measurements

    def num_examples(self, model: Any, batch: Any) -> int | None:
        del model
        return value_count(audio_lengths(batch))


def _text_lengths(batch: Any) -> Any:
    lengths = batch_value(batch, "text_lens", "text_len", "token_lens")
    if lengths is not None:
        return lengths
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
