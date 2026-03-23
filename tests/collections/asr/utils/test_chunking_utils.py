# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import pytest
import torch

from nemo.collections.asr.parts.mixins.transcription import (
    TranscriptionTensorDataset,
    _pre_chunked_tensor_collate_fn,
    resolve_chunking,
)
from nemo.collections.asr.parts.utils.chunking_utils import (
    chunk_waveform,
    find_optimal_chunk_size,
    join_char_level_timestamps,
    merge_flat_chunk_hypotheses,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


def _make_hyp(text, id_, chunk_start=0.0):
    """Shared hypothesis factory for merge tests."""
    h = Hypothesis(score=0.0, y_sequence=torch.tensor([1]), timestamp={"word": [], "segment": []})
    h.text = text
    h.id = id_
    h.chunk_start = chunk_start
    h._encoded_len = 10
    h._timestamps_type = None
    return h


def _make_char(char, token_id, start_off, end_off, token=None):
    return {
        "char": char,
        "token": token if token is not None else char,
        "token_id": token_id,
        "start_offset": start_off,
        "end_offset": end_off,
    }


@pytest.mark.unit
def test_find_optimal_chunk_size_returns_total_for_short_audio():
    total_len = 95
    chunk_size = find_optimal_chunk_size(
        total_len=total_len,
        min_sec=3,
        max_sec=10,
        sample_rate=10,
        overlap_sec=1.0,
    )

    assert chunk_size == total_len


@pytest.mark.unit
def test_find_optimal_chunk_size_prefers_largest_last_chunk():
    total_len = 105
    chunk_size = find_optimal_chunk_size(
        total_len=total_len,
        min_sec=3,
        max_sec=5,
        sample_rate=10,
        overlap_sec=1.0,
    )

    assert chunk_size == 50  # 5 seconds * 10 Hz sample rate


@pytest.mark.unit
def test_chunk_waveform_produces_overlapping_padded_chunks():
    waveform = torch.arange(100, dtype=torch.float32)

    chunks, chunk_lens, chunk_starts = chunk_waveform(
        waveform=waveform,
        chunk_range=[3, 3],
        overlap_sec=1.0,
        sample_rate=10,
    )

    assert len(chunks) == 5
    assert chunk_lens == [30, 30, 30, 30, 20]
    assert chunk_starts == [0, 20, 40, 60, 80]
    assert all(chunk.shape[0] == 30 for chunk in chunks)
    assert torch.allclose(chunks[0], waveform[:30])

    padded_tail = chunks[-1][chunk_lens[-1] :]
    assert torch.allclose(padded_tail, torch.zeros_like(padded_tail))


@pytest.mark.unit
def test_chunk_waveform_requires_valid_range():
    waveform = torch.zeros(32)
    with pytest.raises((ValueError, TypeError)):
        chunk_waveform(waveform=waveform, chunk_range=42, sample_rate=10)
    with pytest.raises((ValueError, TypeError)):
        chunk_waveform(waveform=waveform, chunk_range=[1], sample_rate=10)


@pytest.mark.unit
def test_chunk_waveform_raises_when_overlap_not_smaller_than_chunk():
    waveform = torch.arange(25, dtype=torch.float32)
    with pytest.raises(ValueError):
        chunk_waveform(
            waveform=waveform,
            chunk_range=[1, 1],
            overlap_sec=1.5,
            sample_rate=10,
        )


@pytest.mark.unit
def test_join_char_level_timestamps_without_filter():
    # Merging char level timestamps within same audio segment.
    subsampling_factor = 8
    window_stride = 0.01
    chunk_offsets = [0, 32]

    h0 = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([]),
        timestamp={
            "char": [
                _make_char("a", 10, 0, 1),
                _make_char("b", 11, 2, 3),
            ]
        },
    )
    h1 = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([]),
        timestamp={
            "char": [
                _make_char("b", 12, 0, 1),
                _make_char("c", 13, 2, 3),
            ]
        },
    )

    out = join_char_level_timestamps(
        hypotheses=[h0, h1],
        chunk_offsets=chunk_offsets,
        subsampling_factor=subsampling_factor,
        window_stride=window_stride,
        merged_tokens=None,
    )

    assert len(out) == 4
    shift = chunk_offsets[1] // subsampling_factor

    assert out[0]["start_offset"] == 0 and out[0]["end_offset"] == 1
    assert out[1]["start_offset"] == 2 and out[1]["end_offset"] == 3

    assert out[2]["start_offset"] == 0 + shift and out[2]["end_offset"] == 1 + shift
    assert out[3]["start_offset"] == 2 + shift and out[3]["end_offset"] == 3 + shift

    sec_per_subsample = window_stride * subsampling_factor
    assert out[0]["start"] == pytest.approx(out[0]["start_offset"] * sec_per_subsample)
    assert out[3]["end"] == pytest.approx(out[3]["end_offset"] * sec_per_subsample)


@pytest.mark.unit
def test_join_char_level_timestamps_with_filter():
    # Merging char level timestamps within same audio segment.
    subsampling_factor = 8
    window_stride = 0.01
    chunk_offsets = [0, 200]

    # Chunk0: tokens 1..4
    h0 = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([]),
        timestamp={
            "char": [
                _make_char("a", 1, 0, 0),
                _make_char("b", 2, 1, 1),
                _make_char("c", 3, 2, 2),
                _make_char("d", 4, 3, 3),
            ]
        },
    )
    # Chunk1: overlaps and -1 offsets as provided
    h1 = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([]),
        timestamp={
            "char": [
                _make_char("a", 1, 0, 0),
                _make_char("c", 3, 1, 1),
                _make_char("d", 4, 2, 2),
                _make_char("e", 5, -1, 3),
                _make_char("f", 6, 4, 4),
                _make_char("g", 7, -1, -1),
            ]
        },
    )

    merged_tokens = [1, 2, 3, 4, 5, 6, 7]

    out = join_char_level_timestamps(
        hypotheses=[h0, h1],
        chunk_offsets=chunk_offsets,
        subsampling_factor=subsampling_factor,
        window_stride=window_stride,
        merged_tokens=merged_tokens,
    )

    # Token IDs in order
    assert [d["token_id"] for d in out] == merged_tokens
    # Expected global offsets (from your provided output)
    expected_start_offsets = [0, 1, 2, 3, -1, 29, -1]
    expected_end_offsets = [0, 1, 2, 3, 28, 29, -1]
    assert [d["start_offset"] for d in out] == expected_start_offsets
    assert [d["end_offset"] for d in out] == expected_end_offsets

    # Expected times
    expected_starts = [0.0, 0.08, 0.16, 0.24, -1, 2.32, -1]
    expected_ends = [0.0, 0.08, 0.16, 0.24, 2.24, 2.32, -1]

    assert [d["start"] for d in out] == pytest.approx(expected_starts)
    assert [d["end"] for d in out] == pytest.approx(expected_ends)


@pytest.mark.unit
def test_merge_flat_chunk_hypotheses_preserves_manifest_order():
    """merge_flat_chunk_hypotheses must output results in the order they first appear
    in the flat hypotheses list (which matches the presorted manifest order), NOT in
    alphabetical order of the string ID.  transcribe_speech presorts by duration and
    restore_transcription_order expects results back in that same presorted order."""

    # Simulates presorted order: "258" longest, then "113", then "11".
    # Alphabetical would be: "11", "113", "258" — the opposite of what we want.
    flat = [
        _make_hyp("a1", "258-0", 0.0),
        _make_hyp("a2", "258-0", 40.0),
        _make_hyp("b1", "113-0", 0.0),
        _make_hyp("b2", "113-0", 40.0),
        _make_hyp("c1", "11-0", 0.0),
        _make_hyp("c2", "11-0", 40.0),
    ]

    merged = merge_flat_chunk_hypotheses(
        hypotheses_list=flat,
        timestamps=False,
        tokenizer=None,
        subsampling_factor=8,
        window_stride=0.01,
        vocabulary=[chr(i) for i in range(128)],
        return_hypotheses=False,
    )

    # Output order must be 258, 113, 11 — not alphabetical 11, 113, 258.
    assert len(merged) == 3
    ids = [getattr(m, 'id', None) for m in merged]
    assert ids == ["258-0", "113-0", "11-0"], f"Expected presorted order, got {ids}"


@pytest.mark.unit
def test_resolve_chunking_enables():
    assert resolve_chunking(audio='single.wav', enable_chunking=True) is True
    assert resolve_chunking(audio='single.wav', enable_chunking=False) is False


@pytest.mark.unit
def test_resolve_chunking_disabled_for_dataloader():
    """External DataLoader bypasses chunking setup, so resolve_chunking must return False."""
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(1)))
    assert resolve_chunking(audio=dl, enable_chunking=True) is False


@pytest.mark.unit
def test_transcription_tensor_dataset_pre_chunks():
    """TranscriptionTensorDataset with enable_chunking pre-expands audio into chunks.

    Each item must carry (chunk, chunk_len, sample_idx, chunk_start_samples) so that
    _transcribe_output_processing can assign stable IDs and correct offsets for merging.
    """
    sample_rate = 100  # 100 Hz for easy arithmetic
    # Two audio tensors: 500 samples (5 s) and 250 samples (2.5 s).
    # chunk_range=[2, 2] → 2 s chunks (200 samples), overlap=1 s (100 samples), step=100 samples.
    # Audio 0 (500 samples): starts at 0, 100, 200, 300, 400 → 5 chunks
    # Audio 1 (250 samples): starts at 0, 100, 150 ... let's check: step=100,
    #   overlap=100, chunks while start + 100 < 250: start=0,100; start=200 → 200+100=300 > 250, stop → 3 chunks
    audio_tensors = [
        torch.ones(500, dtype=torch.float32),
        torch.ones(250, dtype=torch.float32) * 2,
    ]
    config = {
        'audio_tensors': audio_tensors,
        'channel_selector': None,
        'sample_rate': sample_rate,
        'enable_chunking': True,
        'chunk_range': [2, 2],  # 2-second chunks at 100 Hz = 200-sample chunks
    }
    ds = TranscriptionTensorDataset(config)

    # Dataset length = total number of chunks across all audio tensors
    assert ds.length > 0

    # All items for audio 0 must have sample_idx=0; all for audio 1 must have sample_idx=1
    sample_indices = [ds.get_item(i)[2].item() for i in range(ds.length)]
    assert 0 in sample_indices
    assert 1 in sample_indices

    # chunk_start_samples must be non-decreasing within each sample's group
    starts_by_sample = {}
    for i in range(ds.length):
        _, _, sidx, start = ds.get_item(i)
        sidx = sidx.item()
        starts_by_sample.setdefault(sidx, []).append(start.item())

    for sidx, starts in starts_by_sample.items():
        assert starts == sorted(starts), f"chunk starts not sorted for sample {sidx}: {starts}"
        # First chunk must start at sample 0
        assert starts[0] == 0

    # _pre_chunked_tensor_collate_fn must pad and stack correctly
    batch = [ds.get_item(i) for i in range(min(4, ds.length))]
    audio, lens, sample_ids, chunk_starts = _pre_chunked_tensor_collate_fn(batch)
    assert audio.ndim == 2
    assert lens.shape[0] == len(batch)
    assert sample_ids.shape[0] == len(batch)
    assert chunk_starts.shape[0] == len(batch)
    # All chunks padded to same length
    assert all(audio[i].shape[0] == audio[0].shape[0] for i in range(len(batch)))
