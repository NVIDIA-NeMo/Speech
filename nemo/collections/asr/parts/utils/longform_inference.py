# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk planning and output selection for offline long-form transcription."""

import difflib
import math
import re
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
    discarded before CTC/RNNT decoding, or selected by word time for AED output.
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
    """Select owned encoder frames along the first dimension, before decoding.

    The feature extractor drops a final partial hop. A source that ends off the frame grid
    can therefore yield one encoder frame fewer than its last owned interval spans. The
    last chunk keeps every frame that the encoder returned. Any other shortfall raises.
    """
    start = (chunk.keep_start - chunk.read_start) // frame_samples * upsampling_factor
    end = math.ceil((chunk.keep_end - chunk.read_start) / frame_samples) * upsampling_factor
    if chunk.is_last and end - upsampling_factor <= frames.shape[0] < end:
        end = frames.shape[0]
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


def words_to_hypothesis(words, text_to_ids, frame_seconds, timestamps, segment_separators) -> Hypothesis:
    """Build one source's result from its selected words, each a dict with ``word`` and times.

    Independently decoded windows define no joint score, so the score is zero, and no
    per-token decoder state, confidence, or alignment is exposed.
    """
    from nemo.collections.asr.parts.utils.timestamp_utils import get_segment_offsets

    text = ' '.join(word['word'] for word in words)
    hyp = Hypothesis(score=0.0, y_sequence=torch.tensor(text_to_ids(text), dtype=torch.long), text=text)
    if timestamps:
        for word in words:
            word.update(
                start_offset=round(word['start'] / frame_seconds), end_offset=round(word['end'] / frame_seconds)
            )
        hyp.timestamp = {'word': words, 'segment': get_segment_offsets(words, segment_separators)}
    return hyp


def transfer_word_times(words: list[str], timed_words: list[dict], duration: float) -> list[dict]:
    """Time ``words`` by matching them to a timed transcript of the same audio.

    A matched word takes the times of its match. A run of unmatched words shares the span
    between its timed neighbours, or the audio edges, evenly. Matching ignores case and
    punctuation, so it cannot fail; times are in seconds from the audio start.
    """
    keys = [_word_key(w) for w in words], [_word_key(t['word']) for t in timed_words]
    times = [None] * len(words)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, *keys, autojunk=False).get_opcodes():
        if tag == 'equal' or (tag == 'replace' and i2 - i1 == j2 - j1):
            for i, j in zip(range(i1, i2), range(j1, j2)):
                times[i] = (timed_words[j]['start'], timed_words[j]['end'])
        elif tag == 'replace':
            _spread(times, i1, i2, timed_words[j1]['start'], timed_words[j2 - 1]['end'])
    i = 0
    while i < len(times):
        j = i
        while j < len(times) and times[j] is None:
            j += 1
        if j > i:
            start = times[i - 1][1] if i > 0 else 0.0
            _spread(times, i, j, start, max(start, times[j][0] if j < len(times) else duration))
        i = j + 1
    return [{'word': w, 'start': start, 'end': end} for w, (start, end) in zip(words, times)]


def stitch_chunk_words(parts: list[list[str]], plan: list[AudioChunk], sample_rate: int) -> list[str]:
    """Join the words of overlapping read windows when no word times exist.

    The longest run of at least two words shared by both sides of an overlap is the join, and
    each side keeps half of it. Windows without such a run, or without overlap, are concatenated.
    """
    out = list(parts[0])
    for prev_words, words, prev, chunk in zip(parts, parts[1:], plan, plan[1:]):
        overlap = (prev.read_end - chunk.read_start) / sample_rate
        # Search only the words each side spoke in the overlap (with margin), so that a common
        # phrase outside the overlap cannot become the join.
        tail = out[-_words_in(overlap, prev_words, prev, sample_rate) :]
        head = words[: _words_in(overlap, words, chunk, sample_rate)]
        match = difflib.SequenceMatcher(
            None, [_word_key(w) for w in tail], [_word_key(w) for w in head], autojunk=False
        ).find_longest_match(0, len(tail), 0, len(head))
        if overlap > 0 and match.size >= 2:
            half = match.size // 2
            out = out[: len(out) - len(tail) + match.a + half] + list(words[match.b + half :])
        else:
            out += words
    return out


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


def _words_in(seconds: float, words: list[str], chunk: AudioChunk, sample_rate: int) -> int:
    rate = len(words) * sample_rate / max(1, chunk.read_end - chunk.read_start)
    return math.ceil(seconds * rate * 1.5) + 2


def _word_key(word: str) -> str:
    return re.sub(r'\W', '', word.lower())


def _spread(times: list, i1: int, i2: int, start: float, end: float) -> None:
    step = (end - start) / (i2 - i1)
    for k in range(i2 - i1):
        times[i1 + k] = (start + k * step, start + (k + 1) * step)
