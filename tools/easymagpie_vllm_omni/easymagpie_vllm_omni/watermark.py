# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codec-owned watermarking for EasyMagpie waveforms.

The codec holds one ``AudioWatermarker``. Today's implementation watermarks each
chunk independently and tapers Perth's delta at the edges. ``AudioChunk`` carries
``request_id`` / ``is_final`` so a later overlap-save watermarker can keep left
context per request without changing the codec call site.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DISABLED_VALUES = {"0", "false", "no", "off"}


@dataclass
class AudioChunk:
    """One codec emission that a watermarker may process independently or with context."""

    request_id: str
    is_final: bool
    audio: torch.Tensor


class AudioWatermarker(Protocol):
    """Watermark a batch of codec chunks.

    Implementations may ignore ``request_id`` / ``is_final`` (tapered Perth) or
    accumulate left context keyed by ``request_id`` and flush on ``is_final``.
    """

    def apply(self, chunks: list[AudioChunk]) -> list[torch.Tensor]:
        """Return one waveform per input chunk, same sample count as ``chunk.audio``."""


class DisabledAudioWatermarker:
    """Pass-through used when ``NEMOTRON_TTS_PERTH_WATERMARK`` disables watermarking."""

    def apply(self, chunks: list[AudioChunk]) -> list[torch.Tensor]:
        return [chunk.audio for chunk in chunks]


def watermarking_enabled() -> bool:
    """Return whether production Perth watermarking is enabled."""
    value = os.environ.get("NEMOTRON_TTS_PERTH_WATERMARK", "1")
    return value.strip().lower() not in _DISABLED_VALUES


def normalized_device(device: str | torch.device) -> str:
    parsed = torch.device(device)
    if parsed.type == "cuda" and parsed.index is None and torch.cuda.is_available():
        parsed = torch.device("cuda", torch.cuda.current_device())
    return str(parsed)


def create_audio_watermarker(
    device: str | torch.device,
    *,
    sample_rate: int,
    models_dir: str | Path | None = None,
) -> AudioWatermarker:
    """Build a watermarker owned by the codec (or a no-op when disabled)."""
    if not watermarking_enabled():
        logger.warning("Perth watermarking is explicitly disabled by NEMOTRON_TTS_PERTH_WATERMARK")
        return DisabledAudioWatermarker()
    return TaperedPerthWatermarker(device, sample_rate=sample_rate, models_dir=models_dir)


def _perth_pretrained_dir() -> Path:
    import perth

    return Path(perth.__file__).resolve().parent / "perth_net" / "pretrained"


def _load_perth_net(device: str, models_dir: str | Path | None = None) -> Any:
    from perth.perth_net.perth_net_implicit.model.perth_net import PerthNet

    pretrained_dir = Path(models_dir) if models_dir is not None else _perth_pretrained_dir()
    checkpoint_dir = pretrained_dir / "implicit"
    if not (checkpoint_dir / "hparams.yaml").is_file():
        raise FileNotFoundError(
            f"Perth watermark checkpoint not found at {checkpoint_dir}. "
            "Re-run convert_to_vllm.py so codec_native/watermark/perth is bundled."
        )
    perth_net = PerthNet.load("implicit", models_dir=str(pretrained_dir))
    return perth_net.to(device).eval()


def _restore_length(watermarked: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    target_samples = int(original.shape[-1])
    current_samples = int(watermarked.shape[-1])
    if current_samples > target_samples:
        return watermarked[..., :target_samples]
    if current_samples < target_samples:
        return F.pad(watermarked, (0, target_samples - current_samples))
    return watermarked


def taper_watermark_delta(
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


class TaperedPerthWatermarker:
    """Independent-chunk Perth with an edge taper on the watermark delta.

    Does not yet use ``request_id`` / ``is_final``. Those fields are the hook
    for overlap-save (~150 ms history and delay) without changing ``apply``.
    """

    def __init__(
        self,
        device: str | torch.device,
        *,
        sample_rate: int,
        perth_net: Any | None = None,
        models_dir: str | Path | None = None,
    ) -> None:
        requested_device = normalized_device(device)
        self.sample_rate = int(sample_rate)
        try:
            self.perth_net = (
                perth_net if perth_net is not None else _load_perth_net(requested_device, models_dir=models_dir)
            )
            if torch.device(requested_device).type == "cuda":
                self.perth_net.encoder = torch.compile(self.perth_net.encoder, mode="default", dynamic=True)
                if torch.cuda.is_available() and perth_net is None:
                    self._warmup_compiled_encoder()
        except Exception as error:
            raise RuntimeError("Perth watermarking is enabled but its model could not be initialized") from error
        self._short_audio_warning_emitted = False
        logger.info(
            "Perth watermarking enabled on device=%s (model sample rate=%d)",
            requested_device,
            int(self.perth_net.hp.sample_rate),
        )

    def _warmup_compiled_encoder(self) -> None:
        perth_net = self.perth_net
        window = max(int(perth_net.hp.n_fft), 1)
        dummy_lengths = (window * 4, window * 8)
        with torch.inference_mode():
            for length in dummy_lengths:
                dummy = torch.zeros(length, device=perth_net.device)
                magnitudes, _ = perth_net.ap.signal_to_magphase(dummy.unsqueeze(0))
                perth_net.encoder(magnitudes)

    def apply(self, chunks: list[AudioChunk]) -> list[torch.Tensor]:
        if not chunks:
            return []

        perth_net = self.perth_net
        perth_device = perth_net.device
        perth_rate = int(perth_net.hp.sample_rate)
        sample_rate = self.sample_rate
        waveforms = [chunk.audio for chunk in chunks]
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
                    [waveforms[index].detach().float().reshape(-1).to(perth_device) for index in indices]
                )
                signals = originals
                if sample_rate != perth_rate:
                    assert resample is not None
                    signals = resample(signals, sample_rate, perth_rate)

                if signals.shape[-1] < minimum_samples:
                    if not self._short_audio_warning_emitted:
                        logger.warning(
                            "Skipping Perth for audio shorter than %d samples at %d Hz",
                            minimum_samples,
                            perth_rate,
                        )
                        self._short_audio_warning_emitted = True
                    continue

                magnitudes, phases = perth_net.ap.signal_to_magphase(signals)
                marked_magnitudes, _ = perth_net.encoder(magnitudes)
                marked = perth_net.ap.magphase_to_signal(marked_magnitudes, phases)
                if sample_rate != perth_rate:
                    assert resample is not None
                    marked = resample(marked, perth_rate, sample_rate)
                marked = _restore_length(marked, originals)
                marked = taper_watermark_delta(
                    marked,
                    originals,
                    edge_samples=edge_samples,
                ).clamp_(-1.0, 1.0)

                for batch_index, waveform_index in enumerate(indices):
                    waveforms[waveform_index] = marked[batch_index].reshape(-1)

        return waveforms
