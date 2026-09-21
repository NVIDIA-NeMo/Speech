# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nemo.collections.speechlm2.parts.ctc_timestamp_utils import MultiSpeakerSOTWordTimestampAligner


@pytest.mark.unit
def test_word_timestamp_alignment_defaults_to_ctc_only():
    assert MultiSpeakerSOTWordTimestampAligner().speaker_logprob_weight == 0.0


def _dense_reference(
    log_probs,
    labels,
    state_lengths,
    blank_id,
    state_speaker_columns,
    speaker_probs,
    speaker_logprob_weight,
    epsilon=1.0e-6,
):
    """Former dense implementation retained only as an exactness oracle."""
    num_streams, max_states = labels.shape
    num_frames = log_probs.shape[0]
    state_mask = torch.arange(max_states).unsqueeze(0) < state_lengths.unsqueeze(1)
    emissions = log_probs[:, labels].permute(1, 0, 2).contiguous()
    emissions.masked_fill_(~state_mask.unsqueeze(1), -float("inf"))
    token_states = state_mask & (state_speaker_columns >= 0)
    if speaker_probs is not None and speaker_logprob_weight > 0 and token_states.any():
        columns = state_speaker_columns.clamp_min(0)
        activity = speaker_probs[:, columns].permute(1, 0, 2)
        emissions += torch.where(
            token_states.unsqueeze(1),
            speaker_logprob_weight * torch.log(activity.clamp_min(epsilon)),
            torch.zeros_like(activity),
        )

    previous = torch.full((num_streams, max_states), -float("inf"))
    previous[:, 0] = emissions[:, 0, 0]
    previous[:, 1] = emissions[:, 0, 1]
    backpointers = torch.full((num_frames, num_streams, max_states), -1, dtype=torch.long)
    states = torch.arange(max_states).unsqueeze(0).expand(num_streams, -1)
    for frame in range(1, num_frames):
        best = previous
        previous_states = states
        advance = torch.full_like(previous, -float("inf"))
        advance[:, 1:] = previous[:, :-1]
        take = advance > best
        best = torch.where(take, advance, best)
        previous_states = torch.where(take, states - 1, previous_states)
        skip = torch.full_like(previous, -float("inf"))
        can_skip = state_mask[:, 2:] & (labels[:, 2:] != blank_id) & (labels[:, 2:] != labels[:, :-2])
        skip[:, 2:] = torch.where(can_skip, previous[:, :-2], skip[:, 2:])
        take = skip > best
        best = torch.where(take, skip, best)
        previous_states = torch.where(take, states - 2, previous_states)
        previous = best + emissions[:, frame]
        previous.masked_fill_(~state_mask, -float("inf"))
        backpointers[frame] = previous_states

    last_blank = state_lengths - 1
    last_token = state_lengths - 2
    blank_scores = previous.gather(1, last_blank[:, None]).squeeze(1)
    token_scores = previous.gather(1, last_token[:, None]).squeeze(1)
    final_states = torch.where(token_scores > blank_scores, last_token, last_blank)
    final_scores = torch.maximum(token_scores, blank_scores)
    paths = torch.empty((num_streams, num_frames), dtype=torch.long)
    current = final_states
    stream_indices = torch.arange(num_streams)
    for frame in range(num_frames - 1, -1, -1):
        paths[:, frame] = current
        if frame:
            current = backpointers[frame, stream_indices, current]
    return list(paths), [float(score) for score in final_scores]


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 7, 19])
@pytest.mark.parametrize("speaker_weight", [0.0, 0.25])
def test_compact_dp_matches_dense_reference_exactly(seed, speaker_weight):
    torch.manual_seed(seed)
    blank_id = 5
    labels = torch.tensor([[5, 0, 5, 1, 5, 1, 5], [5, 2, 5, 3, 5, 5, 5]])
    state_lengths = torch.tensor([7, 5])
    columns = torch.tensor([[-1, 0, -1, 0, -1, 0, -1], [-1, 1, -1, 1, -1, -1, -1]])
    log_probs = torch.log_softmax(torch.randn(15, blank_id + 1), dim=-1)
    speaker_probs = torch.sigmoid(torch.randn(15, 2))
    args = (
        log_probs,
        labels,
        state_lengths,
        blank_id,
        columns,
        speaker_probs,
        speaker_weight,
    )

    expected_paths, expected_scores = _dense_reference(*args)
    actual_paths, actual_scores = MultiSpeakerSOTWordTimestampAligner()._ctc_forced_align_batched(*args)

    assert all(torch.equal(actual, expected) for actual, expected in zip(actual_paths, expected_paths))
    assert actual_scores == pytest.approx(expected_scores, abs=1.0e-6)


@pytest.mark.unit
def test_compact_dp_storage_for_ten_minute_shape():
    storage = MultiSpeakerSOTWordTimestampAligner.estimate_dp_storage_bytes(
        num_frames=7500,
        num_streams=4,
        max_states=3001,
    )

    assert storage["dense_emissions"] == 360_120_000
    assert storage["dense_backpointers"] == 720_240_000
    assert storage["compact_backpointers"] == 22_526_996
    assert storage["compact_backpointers"] * 31 < storage["dense_backpointers"]


@pytest.mark.unit
def test_padded_record_batch_matches_independent_dp_and_backtraces():
    torch.manual_seed(23)
    blank_id = 4
    labels = torch.tensor([[4, 0, 4, 1, 4], [4, 2, 4, 3, 4]])
    state_lengths = torch.tensor([5, 5])
    columns = torch.tensor([[-1, 0, -1, 0, -1], [-1, 1, -1, 1, -1]])
    log_probs = torch.log_softmax(torch.randn(2, 9, blank_id + 1), dim=-1)
    speaker_probs = torch.sigmoid(torch.randn(2, 9, 2))
    frame_lengths = torch.tensor([9, 7])
    aligner = MultiSpeakerSOTWordTimestampAligner()

    batched_paths, batched_scores = aligner._ctc_forced_align_batched(
        log_probs,
        labels,
        state_lengths,
        blank_id,
        columns,
        speaker_probs,
        0.25,
        frame_lengths=frame_lengths,
    )

    for index, frame_length in enumerate(frame_lengths.tolist()):
        paths, scores = aligner._ctc_forced_align_batched(
            log_probs[index, :frame_length],
            labels[index : index + 1],
            state_lengths[index : index + 1],
            blank_id,
            columns[index : index + 1],
            speaker_probs[index, :frame_length],
            0.25,
        )
        assert torch.equal(batched_paths[index], paths[0])
        assert batched_scores[index] == pytest.approx(scores[0], abs=1.0e-6)


@pytest.mark.unit
def test_diarization_timestamps_are_contiguous_10ms_activity_segments():
    labels = torch.tensor(
        [
            [True, True, False, False, True, False],
            [False, True, True, True, False, False],
            [False, False, False, False, False, False],
        ]
    )

    segments = MultiSpeakerSOTWordTimestampAligner._diarization_segments(
        labels,
        frame_seconds=0.01,
        time_offset=1.0,
        audio_duration=0.055,
    )

    assert segments == [
        {"speaker": 0, "start": 1.0, "end": 1.02},
        {"speaker": 1, "start": 1.01, "end": 1.04},
        {"speaker": 0, "start": 1.04, "end": 1.05},
    ]


@pytest.mark.unit
def test_public_word_timestamps_only_expose_word_speaker_start_and_end():
    rows = [
        {
            "word": "hello",
            "speaker_tag": 2,
            "word_index": 7,
            "turn_index": 3,
            "start": 0.16,
            "end": 0.32,
            "start_frame": 2,
            "end_frame": 3,
            "ctc_confidence": 0.9,
            "sortformer_column": 1,
            "speaker_confidence": 0.8,
            "speaker_activity_start": 0.16,
            "speaker_activity_end": 0.32,
        }
    ]

    timestamps = MultiSpeakerSOTWordTimestampAligner._public_word_timestamps(rows)

    assert timestamps == {2: [{"word": "hello", "speaker": 2, "start": 0.16, "end": 0.32}]}
