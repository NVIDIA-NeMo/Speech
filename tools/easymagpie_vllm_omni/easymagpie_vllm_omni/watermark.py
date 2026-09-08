# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-batched Perth watermarking for EasyMagpie codec waveforms."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DISABLED_VALUES = {"0", "false", "no", "off"}
_WATERMARKER: Any | None = None
_INITIALIZED_DEVICE: str | None = None
_SHORT_AUDIO_WARNING_EMITTED = False


def _normalized_device(device: str | torch.device) -> str:
    parsed = torch.device(device)
    if parsed.type == "cuda" and parsed.index is None and torch.cuda.is_available():
        parsed = torch.device("cuda", torch.cuda.current_device())
    return str(parsed)


def watermarking_enabled() -> bool:
    """Return whether production Perth watermarking is enabled."""
    value = os.environ.get("NEMOTRON_TTS_PERTH_WATERMARK", "1")
    return value.strip().lower() not in _DISABLED_VALUES


def initialize_watermarker(device: str | torch.device) -> Any | None:
    """Load Perth on ``device`` or fail startup when watermarking is enabled."""
    global _INITIALIZED_DEVICE, _WATERMARKER

    if not watermarking_enabled():
        logger.warning("Perth watermarking is explicitly disabled by NEMOTRON_TTS_PERTH_WATERMARK")
        return None

    requested_device = _normalized_device(device)
    if _WATERMARKER is not None:
        if requested_device != _INITIALIZED_DEVICE:
            raise RuntimeError(
                "Perth was initialized on "
                f"{_INITIALIZED_DEVICE}, not requested device {requested_device}"
            )
        return _WATERMARKER

    try:
        import perth

        watermarker_type = perth.PerthImplicitWatermarker
        if watermarker_type is None:
            raise ImportError("perth.PerthImplicitWatermarker is unavailable")
        _WATERMARKER = watermarker_type(device=requested_device)
        perth_net = _WATERMARKER.perth_net
        if torch.device(requested_device).type == "cuda":
            perth_net.encoder = torch.compile(
                perth_net.encoder, mode="default", dynamic=True
            )
            if torch.cuda.is_available():
                _warmup_compiled_encoder(_WATERMARKER)
    except Exception as error:
        _WATERMARKER = None
        raise RuntimeError(
            "Perth watermarking is enabled but its model could not be initialized"
        ) from error

    _INITIALIZED_DEVICE = requested_device
    logger.info(
        "Perth watermarking enabled on device=%s (model sample rate=%d)",
        requested_device,
        int(_WATERMARKER.perth_net.hp.sample_rate),
    )
    return _WATERMARKER


def _warmup_compiled_encoder(watermarker: Any) -> None:
    """Trace the compiled encoder on startup and 8-frame-scale spectrograms."""
    perth_net = watermarker.perth_net
    window = max(int(perth_net.hp.n_fft), 1)
    dummy_lengths = (window * 4, window * 8)
    with torch.inference_mode():
        for length in dummy_lengths:
            dummy = torch.zeros(length, device=perth_net.device)
            magnitudes, _ = perth_net.ap.signal_to_magphase(dummy.unsqueeze(0))
            perth_net.encoder(magnitudes)


def _restore_length(watermarked: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    target_samples = int(original.shape[-1])
    current_samples = int(watermarked.shape[-1])
    if current_samples > target_samples:
        return watermarked[..., :target_samples]
    if current_samples < target_samples:
        return F.pad(watermarked, (0, target_samples - current_samples))
    return watermarked


def _taper_watermark_delta(
    marked: torch.Tensor,
    original: torch.Tensor,
    *,
    edge_samples: int,
) -> torch.Tensor:
    """Fade only Perth's perturbation at independently processed chunk edges."""
    samples = int(original.shape[-1])
    edge = min(max(int(edge_samples), 0), samples // 2)
    if edge == 0:
        return marked

    ramp = torch.sin(
        torch.linspace(
            0,
            torch.pi / 2,
            edge,
            device=marked.device,
            dtype=marked.dtype,
        )
    ).square()
    envelope = torch.ones_like(marked)
    envelope[..., :edge] = ramp
    envelope[..., -edge:] = ramp.flip(0)
    return original + (marked - original) * envelope


def watermark_waveforms(
    waveforms: list[torch.Tensor],
    *,
    sample_rate: int,
    device: str | torch.device,
) -> list[torch.Tensor]:
    """Watermark non-empty waveforms, batching tensors with equal lengths."""
    global _SHORT_AUDIO_WARNING_EMITTED

    if not waveforms or not watermarking_enabled():
        return waveforms

    watermarker = initialize_watermarker(device)
    if watermarker is None:
        return waveforms

    perth_net = watermarker.perth_net
    perth_device = perth_net.device
    perth_rate = int(perth_net.hp.sample_rate)
    minimum_samples = int(perth_net.hp.n_fft // 2 + 1)
    # Perth reconstructs each chunk independently with an STFT. Tapering its
    # perturbation across one half-window preserves the codec waveform at both
    # edges and prevents audible seams when streaming chunks are concatenated.
    edge_samples = max(
        (int(perth_net.hp.n_fft) // 2 * sample_rate + perth_rate - 1) // perth_rate,
        1,
    )

    resample = None
    if sample_rate != perth_rate:
        from torchaudio.functional import resample

    length_groups: dict[int, list[int]] = {}
    for index, waveform in enumerate(waveforms):
        samples = int(waveform.numel())
        if samples:
            length_groups.setdefault(samples, []).append(index)

    with torch.inference_mode():
        for indices in length_groups.values():
            originals = torch.stack(
                [
                    waveforms[index].detach().float().reshape(-1).to(perth_device)
                    for index in indices
                ]
            )
            signals = originals
            if sample_rate != perth_rate:
                assert resample is not None
                signals = resample(signals, sample_rate, perth_rate)

            if signals.shape[-1] < minimum_samples:
                if not _SHORT_AUDIO_WARNING_EMITTED:
                    logger.warning(
                        "Skipping Perth for audio shorter than %d samples at %d Hz",
                        minimum_samples,
                        perth_rate,
                    )
                    _SHORT_AUDIO_WARNING_EMITTED = True
                continue

            magnitudes, phases = perth_net.ap.signal_to_magphase(signals)
            marked_magnitudes, _ = perth_net.encoder(magnitudes)
            marked = perth_net.ap.magphase_to_signal(marked_magnitudes, phases)
            if sample_rate != perth_rate:
                assert resample is not None
                marked = resample(marked, perth_rate, sample_rate)
            marked = _restore_length(marked, originals)
            marked = _taper_watermark_delta(
                marked,
                originals,
                edge_samples=edge_samples,
            ).clamp_(-1.0, 1.0)

            for batch_index, waveform_index in enumerate(indices):
                waveforms[waveform_index] = marked[batch_index].reshape(-1)

    return waveforms
