# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneLogger throughput policies for the Audio collection."""

from typing import Any

from nemo.lightning.speech_throughput import (
    SpeechThroughputPolicy,
    ThroughputMeasurements,
    add_audio_seconds,
    audio_lengths,
    first_positive,
    value_count,
)

__all__ = ["AudioThroughputPolicy"]


class AudioThroughputPolicy(SpeechThroughputPolicy):
    """Measure input waveform duration for audio-to-audio training."""

    name = "audio"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        add_audio_seconds(
            measurements,
            "input_audio_seconds",
            audio_lengths(batch),
            first_positive(
                model,
                "sampling_rate",
                "sample_rate",
                "preprocessor._sample_rate",
                "preprocessor.sample_rate",
                "preprocessor._cfg.sample_rate",
                "cfg.sample_rate",
                "cfg.preprocessor.sample_rate",
            ),
        )
        return measurements

    def num_examples(self, model: Any, batch: Any) -> int | None:
        del model
        return value_count(audio_lengths(batch))
