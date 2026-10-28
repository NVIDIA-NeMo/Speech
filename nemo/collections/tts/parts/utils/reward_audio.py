# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waveform preparation shared by online preference-optimization rewards."""

from typing import List, Sequence

import torch
from nemo.collections.audio.parts.utils.transforms import resample


def split_audio_batch(audio: torch.Tensor, lengths: torch.Tensor) -> List[torch.Tensor]:
    """Trim padded mono audio to its true lengths and detach it from the training graph.

    Args:
        audio: Padded mono waveform tensor with shape ``[B, T]``.
        lengths: Tensor with shape ``[B]`` containing each waveform's positive
            integer length in samples, at most ``T``.

    Returns:
        A list of ``B`` detached float32 tensors with shapes ``[lengths[i]]``,
        in input order and on the input device. Float32 inputs may share storage
        with the returned views.

    Raises:
        ValueError: If tensor shapes or waveform lengths are invalid.
    """
    if audio.ndim != 2 or lengths.ndim != 1 or audio.shape[0] != lengths.numel():
        raise ValueError('Expected audio [B, T] and one length per waveform.')
    sizes = lengths.detach().cpu().tolist()
    if any(int(size) != size or size <= 0 or size > audio.shape[1] for size in sizes):
        raise ValueError('Audio lengths must be positive integers within the padded width.')
    audio = audio.detach().float()
    return [row[: int(size)] for row, size in zip(audio, sizes)]


def resample_audio(audio: Sequence[torch.Tensor], sample_rate: int, target_rate: int) -> List[torch.Tensor]:
    """Adapt variable-length waveforms to NeMo's existing tensor resampler.

    This helper detaches and validates a list of unpadded waveforms. The sample
    rate conversion itself uses ``audio.parts.utils.transforms.resample``.

    Args:
        audio: Sequence of nonempty mono tensors, each with shape ``[T_i]``.
        sample_rate: Positive input sample rate in Hz, shared by all waveforms.
        target_rate: Positive output sample rate in Hz.

    Returns:
        Detached float32 waveform tensors in input order, on their original
        devices, with sample counts rounded up after rate conversion. No batch
        padding is added. An empty sequence returns an empty list. If the rates
        match, float32 inputs may share storage with the returned tensors.

    Raises:
        ValueError: If either sample rate is nonpositive or a waveform is empty
            or not one-dimensional.
    """
    if sample_rate <= 0 or target_rate <= 0:
        raise ValueError('Sample rates must be positive.')
    result = []
    for waveform in audio:
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError('Expected nonempty mono waveforms.')
        waveform = waveform.detach().float()
        result.append(resample(waveform, sample_rate, target_rate) if sample_rate != target_rate else waveform)
    return result
