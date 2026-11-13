# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative CPU silence compression with short fades at cut boundaries."""

from dataclasses import dataclass, fields
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SilenceTrimConfig:
    """Times are seconds; energy thresholds are frame RMS dB relative to full scale.

    The silence threshold is the minimum of the estimated background plus a margin,
    the speech reference minus a margin, and an absolute ceiling. Ambiguous, nearly
    stationary recordings and recordings without detected activity pass unchanged.
    Linear fades at deletions are limited to retained quiet samples; fade_duration
    is in seconds and zero disables fading without changing the deletion timeline.
    """

    max_internal_silence: float = 1.5
    max_edge_silence: float = 0.5
    frame_length: float = 0.02
    hop_length: float = 0.01
    noise_percentile: float = 20.0
    noise_margin_db: float = 6.0
    speech_margin_db: float = 30.0
    max_threshold_db: float = -50.0
    min_dynamic_range_db: float = 15.0
    speech_pad: float = 0.15
    min_removal: float = 0.1
    fade_duration: float = 0.005

    def __post_init__(self):
        if any(
            isinstance(getattr(self, f.name), bool) or not np.isfinite(getattr(self, f.name)) for f in fields(self)
        ):
            raise ValueError("Silence trimming parameters must be finite numbers")
        if not 0 <= self.noise_percentile < 95:
            raise ValueError("noise_percentile must be in [0, 95)")
        if not 0 < self.hop_length <= self.frame_length:
            raise ValueError("Require 0 < hop_length <= frame_length")
        if (
            min(
                self.noise_margin_db,
                self.speech_margin_db,
                self.min_dynamic_range_db,
                self.min_removal,
                self.fade_duration,
            )
            < 0
        ):
            raise ValueError("Energy margins, min_removal, and fade_duration must be nonnegative")
        if self.speech_pad < self.frame_length:
            raise ValueError("speech_pad must be at least frame_length")
        if self.max_edge_silence < self.speech_pad or self.max_internal_silence < 2 * self.speech_pad:
            raise ValueError("Silence limits must preserve speech_pad at each speech boundary")


@dataclass
class SilenceTrimResult:
    audio: np.ndarray
    # Half-open sample ranges deleted from the ORIGINAL waveform; fades do not alter the timeline.
    removed: list[tuple[int, int]]
    silence: list[tuple[int, int]]
    threshold_db: float | None
    reason: str

    def map_time(self, time: float, sample_rate: int) -> float:
        """Map a timestamp through the deletions, clamping points inside a deletion."""
        sample = time * sample_rate
        return (sample - sum(max(0, min(sample, end) - start) for start, end in self.removed)) / sample_rate


def silence_trim_config(value) -> SilenceTrimConfig | None:
    """Parse the per-source ``tags.silence_trim`` setting; absent/false disables it."""
    if value is None or value is False:
        return None
    if value is True:
        return SilenceTrimConfig()
    if isinstance(value, Mapping):
        return SilenceTrimConfig(**value)
    raise ValueError("silence_trim must be true, false, null, or a mapping of SilenceTrimConfig parameters")


def trim_long_silences(audio: np.ndarray, sample_rate: int, config: SilenceTrimConfig = None) -> SilenceTrimResult:
    """Remove the middle of long quiet runs and the outside of long edge runs.

    Only mono floating-point waveforms are accepted. Empty, nonfinite, all-quiet,
    or low-contrast signals pass unchanged: trimming must never empty a transcript's
    recording. Speech samples and dtype are preserved exactly. Only retained quiet
    samples next to deletions are faded; no overlap or extra deletion is introduced.
    """
    config = config or SilenceTrimConfig()
    if audio.ndim != 1 or not np.issubdtype(audio.dtype, np.floating) or sample_rate <= 0:
        raise ValueError("Expected a mono floating-point waveform and positive sample rate")
    unchanged = lambda reason, threshold=None: SilenceTrimResult(audio, [], [], threshold, reason)
    if not len(audio) or not np.isfinite(audio).all():
        return unchanged("empty_or_nonfinite")

    hop = max(1, round(config.hop_length * sample_rate))
    window = max(hop, round(config.frame_length * sample_rate))
    starts = np.arange(0, len(audio), hop)
    ends = np.minimum(starts + window, len(audio))
    # Float64 accumulation prevents loss of low-energy tails after loud speech.
    power = np.concatenate(([0.0], np.cumsum(np.square(audio, dtype=np.float64))))
    db = 10 * np.log10(np.maximum((power[ends] - power[starts]) / (ends - starts), 1e-12))
    noise, speech = np.percentile(db, [config.noise_percentile, 95])
    threshold = min(noise + config.noise_margin_db, speech - config.speech_margin_db, config.max_threshold_db)
    if speech - noise < config.min_dynamic_range_db:
        return unchanged("low_dynamic_range", float(threshold))
    quiet = db <= threshold
    if quiet.all() or not quiet.any():
        return unchanged("no_activity_or_silence", float(threshold))

    transitions = np.diff(np.concatenate(([False], quiet, [False])).astype(np.int8))
    silence = [
        (int(a * hop), min(int(b * hop), len(audio)))
        for a, b in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))
    ]
    internal = round(config.max_internal_silence * sample_rate)
    edge = round(config.max_edge_silence * sample_rate)
    minimum = max(1, round(config.min_removal * sample_rate))
    removed = []
    fades = []
    fade_samples = round(config.fade_duration * sample_rate)
    for start, end in silence:
        if start == 0:
            a, b = 0, end - edge
        elif end == len(audio):
            a, b = start + edge, end
        else:
            a, b = start + internal // 2, end - (internal - internal // 2)
        if b - a >= minimum:
            removed.append((a, b))
            fades.append((min(fade_samples, a - start), min(fade_samples, end - b)))
    if not removed:
        return SilenceTrimResult(audio, [], silence, float(threshold), "within_limits")
    kept = []
    previous = 0
    for start, end in removed:
        kept.append(audio[previous:start])
        previous = end
    kept.append(audio[previous:])
    trimmed = np.concatenate(kept)
    deleted = 0
    for (start, end), (fade_out, fade_in) in zip(removed, fades):
        boundary = start - deleted
        if fade_out:
            trimmed[boundary - fade_out : boundary] *= np.linspace(0, 1, fade_out, dtype=audio.dtype)[::-1]
        if fade_in:
            trimmed[boundary : boundary + fade_in] *= np.linspace(0, 1, fade_in, dtype=audio.dtype)
        deleted += end - start
    return SilenceTrimResult(trimmed, removed, silence, float(threshold), "trimmed")
