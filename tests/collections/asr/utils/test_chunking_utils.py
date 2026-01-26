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

from nemo.collections.asr.parts.utils.chunking_utils import (
    join_char_level_timestamps,
    merge_all_hypotheses,
    merge_hypotheses_of_same_audio,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


def _make_char(char, token_id, start_off, end_off, token=None):
    return {
        "char": char,
        "token": token if token is not None else char,
        "token_id": token_id,
        "start_offset": start_off,
        "end_offset": end_off,
    }


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
def test_merge_hypotheses_of_same_audio():
    # Different segments of the same audio file are correctly combined
    subsampling_factor = 8
    chunk_duration_seconds = 10
    frame_offset = int(chunk_duration_seconds * 1000 / subsampling_factor)

    h0 = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([1]),
        timestamp={
            "word": [{"word": "a", "start": 0.0, "end": 0.1, "start_offset": 0, "end_offset": 2}],
            "segment": [{"segment": "a", "start": 0.0, "end": 0.1, "start_offset": 0, "end_offset": 2}],
        },
    )
    h1 = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([2]),
        timestamp={
            "word": [{"word": "b", "start": 0.2, "end": 0.3, "start_offset": 0, "end_offset": 3}],
            "segment": [{"segment": "b", "start": 0.2, "end": 0.3, "start_offset": 0, "end_offset": 3}],
        },
    )
    h2 = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([3]),
        timestamp={
            "word": [],
            "segment": [],
        },
    )

    merged = merge_hypotheses_of_same_audio(
        hypotheses_list=[h0, h1, h2],
        timestamps=True,
        subsampling_factor=subsampling_factor,
        chunk_duration_seconds=chunk_duration_seconds,
    )

    words = merged.timestamp["word"]
    segs = merged.timestamp["segment"]

    assert [w["word"] for w in words] == ["a", "b"]
    assert words[0]["start"] == pytest.approx(0.0)
    assert words[0]["start_offset"] == 0
    assert words[1]["start"] == pytest.approx(0.2 + chunk_duration_seconds)
    assert words[1]["start_offset"] == frame_offset

    assert [s["segment"] for s in segs] == ["a", "b"]
    assert segs[1]["end"] == pytest.approx(0.3 + chunk_duration_seconds)
    assert segs[1]["end_offset"] == 3 + frame_offset


@pytest.mark.unit
def test_merge_all_hypotheses():
    # Testing if merging by id works
    def H(text, id_):
        h = Hypothesis(score=0.0, y_sequence=torch.tensor([1]), timestamp={"word": [], "segment": []})
        h.text = text
        h.id = id_
        return h

    hyps = [H("a", 1), H("b", 1), H("c", 2), H("d", 2)]

    merged_list = merge_all_hypotheses(
        hypotheses_list=hyps,
        timestamps=False,
        subsampling_factor=2,
        chunk_duration_seconds=3600,
    )

    assert len(merged_list) == 2
    texts = {m.text for m in merged_list}
    assert texts == {"a b", "c d"}


@pytest.mark.unit
def test_merge_all_hypotheses_with_cut_segmented_suffix():
    def H(text, id_):
        h = Hypothesis(score=0.0, y_sequence=torch.tensor([1]), timestamp={"word": [], "segment": []})
        h.text = text
        h.id = id_
        return h

    hyps = [
        H("root", "11-0"),
        H("cont1", "11-1_cut_segmented"),
        H("cont2", "11-2_cut_segmented"),
        H("other", "12-0"),
    ]

    merged_list = merge_all_hypotheses(
        hypotheses_list=hyps,
        timestamps=False,
        subsampling_factor=8,
        chunk_duration_seconds=3600,
    )

    assert len(merged_list) == 2
    texts = sorted(m.text for m in merged_list)
    assert texts == ["other", "root cont1 cont2"]
