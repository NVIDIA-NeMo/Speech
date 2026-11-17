# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk planning and output selection for offline long-form transcription."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


@dataclass(frozen=True)
class ChunkingConfig:
    """Offline chunk sizes in seconds.

    ``chunk_size`` is the duration owned by one chunk. ``context_size`` is extra audio
    read on each side. Sizes are rounded up to complete encoder frames. Context is
    discarded before CTC/RNNT decoding, or selected by word time for aligned AED output.
    """

    chunk_size: float = 30.0
    context_size: float = 2.0


@dataclass(frozen=True)
class AudioChunk:
    """Read and owned intervals in samples, relative to the source utterance."""

    read_start: int
    read_end: int
    keep_start: int
    keep_end: int
    is_last: bool


def make_chunk_plan(
    num_samples: int, sample_rate: int, config: ChunkingConfig, frame_samples: int = 1
) -> list[AudioChunk]:
    """Partition an utterance into disjoint owned intervals with overlapping read context.

    Args:
        num_samples: Positive source length at the model sample rate.
        sample_rate: Model sample rate.
        config: Owned chunk and context durations.
        frame_samples: Samples per encoder frame; read starts align to this grid.

    Returns:
        Ordered chunks with half-open intervals. Every source sample has one owner.
    """
    if num_samples <= 0 or sample_rate <= 0 or frame_samples <= 0:
        raise ValueError('Audio length, sample rate, and frame size must be positive.')
    if not math.isfinite(config.chunk_size) or config.chunk_size <= 0:
        raise ValueError('chunk_size must be finite and greater than zero.')
    if not math.isfinite(config.context_size) or config.context_size < 0:
        raise ValueError('context_size must be finite and nonnegative.')
    size = max(1, math.ceil(config.chunk_size * sample_rate / frame_samples)) * frame_samples
    context = math.ceil(config.context_size * sample_rate / frame_samples) * frame_samples
    return [
        AudioChunk(
            read_start=max(0, start - context),
            read_end=min(num_samples, start + size + context),
            keep_start=start,
            keep_end=min(num_samples, start + size),
            is_last=start + size >= num_samples,
        )
        for start in range(0, num_samples, size)
    ]


def select_chunk_frames(
    frames: torch.Tensor, chunk: AudioChunk, frame_samples: int, upsampling_factor: int = 1
) -> torch.Tensor:
    """Select owned encoder frames along the first dimension, before decoding."""
    start = (chunk.keep_start - chunk.read_start) // frame_samples * upsampling_factor
    end = math.ceil((chunk.keep_end - chunk.read_start) / frame_samples) * upsampling_factor
    if end > frames.shape[0]:
        raise RuntimeError(f'Encoder returned {frames.shape[0]} frames, but chunk ownership requires {end}.')
    return frames[start:end]


def select_chunk_words(words: list[dict], chunk: AudioChunk, sample_rate: int) -> list[dict]:
    """Keep words whose midpoints belong to the chunk; shift times by its actual start.

    Repeated words remain separate entries. This does not compare word text, invent
    missing timestamps, or change valid word durations.
    """
    selected = []
    offset = chunk.read_start / sample_rate
    for word in words:
        start, end = word['start'], word['end']
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError('Aligned chunk words must have finite, nonnegative, ordered timestamps.')
        midpoint = (offset + (start + end) / 2) * sample_rate
        if chunk.keep_start <= midpoint < chunk.keep_end:
            entry = {k: v for k, v in word.items() if k not in ('start_offset', 'end_offset')}
            entry.update(start=offset + start, end=offset + end)
            selected.append(entry)
    return selected


def merge_rnnt_chunks(parts: list[Hypothesis], plan: list[AudioChunk], frame_samples: int) -> Hypothesis:
    """Combine RNNT/TDT output decoded from disjoint owned encoder regions.

    The decoder sees no context frames. Its token, alignment, and confidence entries
    therefore have the same ownership. Word confidence and text must be recomputed
    by the model's existing decoder after assembly.
    """
    merged = Hypothesis(
        score=sum(h.score for h in parts),
        y_sequence=torch.cat([torch.as_tensor(h.y_sequence, dtype=torch.long).cpu() for h in parts]),
        timestamp=[],
    )
    for hyp, chunk in zip(parts, plan):
        offset = chunk.keep_start // frame_samples
        times = hyp.timestamp.tolist() if isinstance(hyp.timestamp, torch.Tensor) else hyp.timestamp
        if len(times) != len(hyp.y_sequence):
            raise RuntimeError('RNNT chunk decoding must return one emission time per token.')
        merged.timestamp.extend(int(t) + offset for t in times)
    merged.length = math.ceil(plan[-1].keep_end / frame_samples)
    for name in ('alignments', 'frame_confidence', 'non_blank_step_confidence_precomputed'):
        values = [getattr(h, name, None) for h in parts]
        if all(v is not None for v in values):
            setattr(merged, name, [item for value in values for item in value])
    durations = [h.token_duration for h in parts]
    if all(d is not None and len(d) == len(h.y_sequence) for d, h in zip(durations, parts)):
        merged.token_duration = torch.cat([torch.as_tensor(d).cpu() for d in durations])
    return merged


class ChunkedAudioDataset(IterableDataset):
    """Read planned slices lazily from Lhotse cuts or mono tensors.

    Only interval metadata is expanded. Audio tensors are sliced on demand and files
    are read through Lhotse at the model sample rate. Duplicate file entries retain
    separate source indices.
    """

    def __init__(self, sources, sample_rate, config, frame_samples, channel_selector=None, batch_size=1):
        self.sources = sources
        self.sample_rate = sample_rate
        self.channel_selector = channel_selector
        self.config = config
        self.frame_samples = frame_samples
        self.batch_size = batch_size
        make_chunk_plan(1, sample_rate, config, frame_samples)

    def __iter__(self):
        worker = get_worker_info()
        ordinal = 0
        for source_idx, source in enumerate(self.sources):
            if isinstance(source, (np.ndarray, torch.Tensor)):
                if source.ndim != 1:
                    raise ValueError('Chunked tensor input must contain one mono waveform per entry.')
                length = len(source)
            else:
                length = round(source.duration * self.sample_rate)
            plan = make_chunk_plan(length, self.sample_rate, self.config, self.frame_samples)
            for chunk in plan:
                # Route whole batches to workers, so DataLoader's ordered task delivery
                # preserves source and chunk order, including a final partial batch.
                owner = ordinal // self.batch_size
                ordinal += 1
                if worker is None or owner % worker.num_workers == worker.id:
                    yield self._load_chunk(source, source_idx, chunk)

    def _load_chunk(self, source, source_idx, chunk):
        cut = None
        if isinstance(source, (np.ndarray, torch.Tensor)):
            audio = torch.as_tensor(source[chunk.read_start : chunk.read_end])
        else:
            from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

            cut = source.truncate(
                offset=chunk.read_start / self.sample_rate,
                duration=(chunk.read_end - chunk.read_start) / self.sample_rate,
                preserve_id=True,
            )
            samples = cut.resample(self.sample_rate).load_audio()
            samples = samples[0] if samples.shape[0] == 1 else samples.T
            samples = AudioSegment(samples, self.sample_rate, channel_selector=self.channel_selector).samples
            audio = torch.as_tensor(samples)
        if audio.ndim != 1:
            raise ValueError('Select one channel or average channels for chunked transcription.')
        return audio, source_idx, chunk, cut


def collate_audio_chunks(items):
    """Right-pad waveform slices and keep source and interval metadata on the CPU."""
    audio, indices, chunks, cuts = zip(*items)
    lengths = torch.tensor([len(a) for a in audio], dtype=torch.long)
    batch = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)
    return batch, lengths, indices, chunks, cuts
