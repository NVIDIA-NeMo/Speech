# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneLogger throughput policies for the SpeechLM2 collection."""

from numbers import Integral
from typing import Any

from nemo.lightning.speech_throughput import (
    SpeechThroughputPolicy,
    ThroughputMeasurements,
    add_audio_seconds,
    add_sum,
    add_value,
    batch_value,
    first_positive,
    value_count,
)

__all__ = [
    "DuplexSTTThroughputPolicy",
    "SALMThroughputPolicy",
    "SpeechToSpeechThroughputPolicy",
]


class SALMThroughputPolicy(SpeechThroughputPolicy):
    """Measure audio and exact post-insertion model tokens for every SALM variant."""

    name = "salm"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        add_audio_seconds(
            measurements,
            "input_audio_seconds",
            batch_value(batch, "audio_lens"),
            _sample_rate(model),
        )
        add_value(measurements, "multimodal_tokens", getattr(model, "_last_batch_num_tokens", None))
        return measurements

    def num_examples(self, model: Any, batch: Any) -> int | None:
        return _salm_num_examples(model, batch)


class DuplexSTTThroughputPolicy(SpeechThroughputPolicy):
    """Measure audio and text-only work in a mixed DuplexSTT batch."""

    name = "duplex_stt"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        audio_batch = batch_value(batch, "audio_data")
        text_batch = batch_value(batch, "text_data")
        if audio_batch is not None:
            add_audio_seconds(
                measurements,
                "input_audio_seconds",
                batch_value(audio_batch, "source_audio_lens"),
                _source_sample_rate(model),
            )
        if text_batch is not None:
            add_sum(measurements, "text_tokens", batch_value(text_batch, "text_token_lens"))
        return measurements

    def num_examples(self, model: Any, batch: Any) -> int | None:
        del model
        counts = []
        audio_batch = batch_value(batch, "audio_data")
        text_batch = batch_value(batch, "text_data")
        if audio_batch is not None:
            counts.append(value_count(batch_value(audio_batch, "source_audio_lens")))
        if text_batch is not None:
            counts.append(value_count(batch_value(text_batch, "text_token_lens")))
        if not counts or any(count is None for count in counts):
            return None
        return sum(counts)


class SpeechToSpeechThroughputPolicy(SpeechThroughputPolicy):
    """Measure each direction of a duplex speech batch independently."""

    name = "speech_to_speech"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        add_audio_seconds(
            measurements,
            "input_audio_seconds",
            batch_value(batch, "source_audio_lens"),
            _source_sample_rate(model),
        )
        add_audio_seconds(
            measurements,
            "output_audio_seconds",
            batch_value(batch, "target_audio_lens"),
            _target_sample_rate(model),
        )
        return measurements

    def num_examples(self, model: Any, batch: Any) -> int | None:
        del model
        lengths = batch_value(batch, "source_audio_lens")
        if lengths is None:
            lengths = batch_value(batch, "target_audio_lens")
        return value_count(lengths)


def _salm_num_examples(model: Any, batch: Any) -> int | None:
    exact = getattr(model, "_last_batch_num_examples", None)
    if isinstance(exact, Integral):
        return int(exact)

    offsets = batch_value(batch, "text_cu_seqlens")
    count = value_count(offsets)
    if count is not None:
        return max(count - 1, 0)

    input_ids = batch_value(batch, "input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) > 1:
        return int(shape[0])
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


def _source_sample_rate(model: Any) -> float | None:
    return first_positive(
        model,
        "source_sample_rate",
        "perception.preprocessor.featurizer.sample_rate",
        "perception.preprocessor._sample_rate",
        "perception.preprocessor._cfg.sample_rate",
    )


def _target_sample_rate(model: Any) -> float | None:
    return first_positive(
        model,
        "target_sample_rate",
        "audio_codec.sample_rate",
        "audio_codec.output_sample_rate",
    )
