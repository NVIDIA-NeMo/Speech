# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Reference-based acoustic prosody distances for TTS evaluation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
from numba import njit


@dataclass(frozen=True)
class ProsodyDistanceResult:
    """Per-pair acoustic prosody distance metrics."""

    pitch_distance: float
    intensity_distance: float
    speech_rate_distance: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable dictionary."""
        return {
            "pitch_distance": self.pitch_distance,
            "intensity_distance": self.intensity_distance,
            "speech_rate_distance": self.speech_rate_distance,
        }


@njit
def _dtw_distance_1d_numba(x: np.ndarray, y: np.ndarray, nan_penalty: float, band_radius: int) -> float:
    n = x.shape[0]
    m = y.shape[0]
    if n == 0 or m == 0:
        return np.nan

    inf = 1.0e30
    prev = np.empty(m + 1, dtype=np.float64)
    curr = np.empty(m + 1, dtype=np.float64)
    for j in range(m + 1):
        prev[j] = inf
        curr[j] = inf
    prev[0] = 0.0

    for i in range(1, n + 1):
        for j in range(m + 1):
            curr[j] = inf

        j_start = max(1, i - band_radius)
        j_end = min(m, i + band_radius) + 1

        for j in range(j_start, j_end):
            xv = x[i - 1]
            yv = y[j - 1]
            x_nan = np.isnan(xv)
            y_nan = np.isnan(yv)
            if x_nan and y_nan:
                cost = 0.0
            elif x_nan or y_nan:
                cost = nan_penalty
            else:
                diff = xv - yv
                cost = diff if diff >= 0.0 else -diff

            best_prev = prev[j - 1]
            if prev[j] < best_prev:
                best_prev = prev[j]
            if curr[j - 1] < best_prev:
                best_prev = curr[j - 1]
            curr[j] = cost + best_prev

        tmp = prev
        prev = curr
        curr = tmp

    total = prev[m]
    if total >= inf / 2.0:
        return np.nan
    return total / float(n + m)


def compute_prosody_distances(
    gt_audio_path: str,
    pred_audio_path: str,
    text: Any,
    *,
    sample_rate: int = 16000,
    res_type: str = "soxr_hq",
    frame_shift_ms: float = 20.0,
    frame_length_ms: float = 64.0,
    fmin: float = 55.0,
    fmax: float = 450.0,
    pyin_n_thresholds: int = 24,
    pyin_beta_parameters: tuple[float, float] = (2.0, 18.0),
    pyin_boltzmann_parameter: float = 2.0,
    pyin_resolution: float = 0.25,
    pyin_max_transition_rate: float = 12.0,
    pyin_switch_prob: float = 0.01,
    pyin_no_trough_prob: float = 0.01,
    max_dtw_frames: int = 1000,
    dtw_band_ratio: float = 0.05,
    f0_nan_penalty: float = 6.0,
    min_voiced_frames: int = 5,
) -> ProsodyDistanceResult:
    """Compute acoustic prosody distances between reference and generated audio.

    Args:
        gt_audio_path: Ground-truth/reference audio path.
        pred_audio_path: Generated/predicted audio path.
        text: Reference text used for character-per-second speech rate.
        sample_rate: Audio sampling rate used before computing prosody features.
        res_type: Librosa resampling type.
        frame_shift_ms: Hop size in milliseconds.
        frame_length_ms: Analysis frame length in milliseconds.
        fmin: Minimum F0 in Hz for PYIN.
        fmax: Maximum F0 in Hz for PYIN.
        pyin_n_thresholds: Number of PYIN thresholds.
        pyin_beta_parameters: PYIN beta-distribution parameters.
        pyin_boltzmann_parameter: PYIN Boltzmann parameter.
        pyin_resolution: PYIN pitch-bin resolution in semitones.
        pyin_max_transition_rate: PYIN maximum pitch-transition rate.
        pyin_switch_prob: PYIN voiced/unvoiced switch probability.
        pyin_no_trough_prob: PYIN no-trough probability.
        max_dtw_frames: Maximum frame count before downsampling DTW inputs.
        dtw_band_ratio: Sakoe-Chiba band radius as a ratio of the longest sequence.
        f0_nan_penalty: DTW cost for a voiced/unvoiced mismatch.
        min_voiced_frames: Minimum voiced frames required to compute pitch distance.

    Returns:
        ProsodyDistanceResult with pitch, intensity, and speech-rate distances.
    """
    gt_audio, sr, gt_duration = _load_audio(gt_audio_path, sample_rate=sample_rate, res_type=res_type)
    pred_audio, _, pred_duration = _load_audio(pred_audio_path, sample_rate=sample_rate, res_type=res_type)

    hop_length, frame_length = _frame_params(sr, frame_shift_ms=frame_shift_ms, frame_length_ms=frame_length_ms)
    gt_log_energy = _compute_log_energy(gt_audio, frame_length=frame_length, hop_length=hop_length)
    pred_log_energy = _compute_log_energy(pred_audio, frame_length=frame_length, hop_length=hop_length)

    pyin_kwargs = {
        "fmin": fmin,
        "fmax": fmax,
        "n_thresholds": pyin_n_thresholds,
        "beta_parameters": pyin_beta_parameters,
        "boltzmann_parameter": pyin_boltzmann_parameter,
        "resolution": pyin_resolution,
        "max_transition_rate": pyin_max_transition_rate,
        "switch_prob": pyin_switch_prob,
        "no_trough_prob": pyin_no_trough_prob,
    }
    gt_f0 = _compute_f0(gt_audio, sr=sr, frame_length=frame_length, hop_length=hop_length, pyin_kwargs=pyin_kwargs)
    pred_f0 = _compute_f0(pred_audio, sr=sr, frame_length=frame_length, hop_length=hop_length, pyin_kwargs=pyin_kwargs)

    pitch_distance = float("nan")
    if np.isfinite(gt_f0).sum() >= min_voiced_frames and np.isfinite(pred_f0).sum() >= min_voiced_frames:
        gt_pitch, pred_pitch = _prepare_f0_for_metric(gt_f0, pred_f0)
        pitch_distance = _dtw_distance_1d(
            _maybe_reduce_for_dtw(gt_pitch, max_dtw_frames=max_dtw_frames),
            _maybe_reduce_for_dtw(pred_pitch, max_dtw_frames=max_dtw_frames),
            nan_penalty=f0_nan_penalty,
            band_ratio=dtw_band_ratio,
        )

    intensity_distance = _dtw_distance_1d(
        _maybe_reduce_for_dtw(_zscore(gt_log_energy), max_dtw_frames=max_dtw_frames),
        _maybe_reduce_for_dtw(_zscore(pred_log_energy), max_dtw_frames=max_dtw_frames),
        nan_penalty=0.0,
        band_ratio=dtw_band_ratio,
    )

    gt_char_count = _char_count(text)
    gt_speech_rate = gt_char_count / gt_duration if gt_duration > 0.0 else float("nan")
    pred_speech_rate = gt_char_count / pred_duration if pred_duration > 0.0 else float("nan")
    speech_rate_distance = abs(gt_speech_rate - pred_speech_rate)

    return ProsodyDistanceResult(
        pitch_distance=_safe_float(pitch_distance),
        intensity_distance=_safe_float(intensity_distance),
        speech_rate_distance=_safe_float(speech_rate_distance),
    )


def _load_audio(path: str, sample_rate: int, res_type: str) -> tuple[np.ndarray, int, float]:
    if not path:
        raise FileNotFoundError("empty audio filepath")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    audio, sr = librosa.load(path, sr=sample_rate, mono=True, res_type=res_type)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"empty audio after loading: {path}")
    return audio, int(sr), float(audio.shape[0] / sr)


def _frame_params(sr: int, frame_shift_ms: float, frame_length_ms: float) -> tuple[int, int]:
    hop_length = max(1, int(round(sr * frame_shift_ms / 1000.0)))
    frame_length = max(hop_length * 2, int(round(sr * frame_length_ms / 1000.0)))
    return hop_length, frame_length


def _compute_log_energy(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    return np.log(np.maximum(np.asarray(rms, dtype=np.float64), 1.0e-10))


def _compute_f0(
    audio: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    pyin_kwargs: dict[str, Any],
) -> np.ndarray:
    f0, voiced_flag, _ = librosa.pyin(
        y=np.asarray(audio, dtype=np.float64),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
        pad_mode="constant",
        fill_na=np.nan,
        **pyin_kwargs,
    )
    f0 = np.asarray(f0, dtype=np.float64)
    if voiced_flag is None:
        return f0

    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    min_len = min(len(f0), len(voiced_flag))
    f0 = f0[:min_len]
    f0[~voiced_flag[:min_len]] = np.nan
    return f0


def _prepare_f0_for_metric(gt_f0_hz: np.ndarray, pred_f0_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_median = float(np.nanmedian(gt_f0_hz)) if np.isfinite(gt_f0_hz).any() else float("nan")
    return _hz_to_semitones(gt_f0_hz, gt_median), _hz_to_semitones(pred_f0_hz, gt_median)


def _dtw_distance_1d(x: np.ndarray, y: np.ndarray, nan_penalty: float, band_ratio: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("DTW expects 1-D arrays")
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    band_radius = max(abs(len(x) - len(y)), int(math.ceil(band_ratio * max(len(x), len(y)))))
    return _safe_float(_dtw_distance_1d_numba(x, y, float(nan_penalty), int(band_radius)))


def _hz_to_semitones(f0_hz: np.ndarray, ref_hz: float) -> np.ndarray:
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    out = np.full_like(f0_hz, np.nan, dtype=np.float64)
    if not np.isfinite(ref_hz) or ref_hz <= 0.0:
        return out
    valid = np.isfinite(f0_hz) & (f0_hz > 0.0)
    out[valid] = 12.0 * np.log2(f0_hz[valid] / ref_hz)
    return out


def _zscore(values: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return values
    mean = float(np.mean(values[finite]))
    std = float(np.std(values[finite]))
    out = values.copy()
    if std < eps:
        out[finite] = out[finite] - mean
    else:
        out[finite] = (out[finite] - mean) / std
    return out


def _maybe_reduce_for_dtw(values: np.ndarray, max_dtw_frames: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= max_dtw_frames:
        return values
    return _resample_1d_preserve_nans(values, max_dtw_frames)


def _resample_1d_preserve_nans(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) == target_len:
        return values
    if len(values) == 0:
        return values
    if len(values) == 1:
        return np.full(target_len, values[0], dtype=np.float64)

    old_t = np.linspace(0.0, 1.0, len(values))
    new_t = np.linspace(0.0, 1.0, target_len)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return np.full(target_len, np.nan, dtype=np.float64)

    out = np.interp(new_t, old_t[valid], values[valid])
    valid_interp = np.interp(new_t, old_t, valid.astype(np.float64))
    out[valid_interp < 0.5] = np.nan
    return out.astype(np.float64)


def _char_count(text: Any) -> int:
    if text is None:
        return 0
    return sum(1 for ch in str(text) if not ch.isspace())


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(value):
        return float("nan")
    return value
